from pathlib import Path
import os
import numpy as np
import polars as pl
from multiprocessing import Pool, cpu_count, get_context

from scipy.signal import lombscargle  # faster path for scoring (avoid per-cell grid creation)

from helpers import (
    lsp,                    # keep for high-res plotting only
    plot_lsp,
    plot_place_field_1d,
    plot_raw_signal,
)

# ------------------ CONFIG ------------------
track_length = 180.0
f_place = 1.0 / track_length  # fundamental place frequency (1/cm)

project = "StateSpaceOdyssey"
mouse_id = "26"

day1_file  = "2025-08-21-16-35-48-370149.feather"   # Session 1
day7_file  = "2025-08-27-17-18-55-361099.feather"   # Session 7
day14_file = "2025-09-03-17-02-46-836208.feather"   # Session 14

# Scoring grid (fast)
RANK_N_FREQS = 1000          # was 5000; usually plenty for SNR-at-target and much faster
RANK_F_MIN = 0.0
RANK_F_MAX = 0.05

# Plot grid (slow, but only for 6 cells x 3 sessions)
PLOT_N_FREQS = 50000
PLOT_F_MIN = 0.0
PLOT_F_MAX = 0.1

# Cap processes: memory bandwidth usually dominates; 32–64 is often faster than 250+
MAX_PROCS = 64

# Progress printing
PROGRESS_EVERY = 100
# -------------------------------------------


def load_run_session(feather_file: str):
    """Load one session, filter to running, return (X_cells, distance_cm)."""
    meso_data = pl.read_ipc(feather_file)
    run_data = meso_data.filter(pl.col("system_state") == "run")

    # X_run: (n_samples, n_cells)
    X_run = np.vstack(run_data["multi_day_spikes"]).astype(np.float32, copy=False)

    # Make cell-major contiguous array: (n_cells, n_samples)
    # This avoids costly column copies per worker.
    X_cells = np.ascontiguousarray(X_run.T)

    t = run_data["distance_cm"].to_numpy().astype(np.float64, copy=False)
    return X_cells, t


# -------- Globals inside worker processes (set in initializer) --------
_G_X_cells = None
_G_t = None
_G_omega = None
_G_idx_place = None


def _init_pool_for_session(X_cells, t, omega, idx_place):
    """Initializer: store session-specific arrays in worker-global vars."""
    global _G_X_cells, _G_t, _G_omega, _G_idx_place
    _G_X_cells = X_cells
    _G_t = t
    _G_omega = omega
    _G_idx_place = idx_place


def _snr_worker(cell_index: int):
    """
    Compute SNR for one cell using precomputed omega and target index.
    Uses scipy.signal.lombscargle directly for speed.
    """
    y = _G_X_cells[cell_index]  # contiguous 1D view

    # Handle NaNs (cheap guard). If you know there are none, you can remove this.
    mask = np.isfinite(_G_t) & np.isfinite(y)
    t = _G_t[mask]
    y = y[mask]

    if y.size < 10:
        return cell_index, np.nan

    y = y - y.mean(dtype=np.float64)

    power = lombscargle(t, y, _G_omega, normalize=True)

    # SNR at target frequency, relative to background (excluding DC)
    S_place = power[_G_idx_place]
    noise = power[1:]
    mu = noise.mean()
    sigma = noise.std()

    # Prevent divide-by-zero
    score = (S_place - mu) / (sigma + 1e-12)
    return cell_index, float(score)


def compute_scores_for_session(X_cells, t, n_procs, n_freqs=RANK_N_FREQS, f_min=RANK_F_MIN, f_max=RANK_F_MAX):
    """
    Compute SNR scores for all cells in one session.
    X_cells: (n_cells, n_samples) contiguous
    Returns:
      scores: (n_cells,)
      rankings: (n_cells,) cell ids sorted best->worst
    """
    n_cells = X_cells.shape[0]

    # Precompute frequency grid once per session (huge speedup vs per-cell)
    freqs = np.linspace(f_min, f_max, n_freqs, dtype=np.float64)
    omega = 2.0 * np.pi * freqs
    idx_place = int(np.argmin(np.abs(freqs - f_place)))

    # Choose a chunksize that reduces scheduling overhead
    # Rule of thumb: few * n_procs chunks total
    chunksize = max(5, n_cells // (n_procs * 8))
    scores = np.empty(n_cells, dtype=np.float64)

    # On Linux HPC, fork is typically available and avoids pickling huge arrays.
    # If fork isn't available, default context will still work but may be slower.
    try:
        ctx = get_context("fork")
    except ValueError:
        ctx = get_context()

    with ctx.Pool(
        processes=n_procs,
        initializer=_init_pool_for_session,
        initargs=(X_cells, t, omega, idx_place),
    ) as pool:
        done = 0
        for cell_index, score in pool.imap_unordered(_snr_worker, range(n_cells), chunksize=chunksize):
            scores[cell_index] = score
            done += 1
            if PROGRESS_EVERY and (done % PROGRESS_EVERY == 0):
                print(f"  scored {done}/{n_cells} cells", flush=True)

    rankings = np.argsort(scores)[::-1]
    return scores, rankings


def main():
    # ---------- LOAD ALL 3 SESSIONS ----------
    X1_cells, t1 = load_run_session(day1_file)
    X7_cells, t7 = load_run_session(day7_file)
    X14_cells, t14 = load_run_session(day14_file)

    # Use cells that exist in ALL sessions
    n_cells = min(X1_cells.shape[0], X7_cells.shape[0], X14_cells.shape[0])

    # Trim to intersection count (assumes cell ids align across sessions)
    X1_cells = X1_cells[:n_cells]
    X7_cells = X7_cells[:n_cells]
    X14_cells = X14_cells[:n_cells]

    # ---------- OUTPUT DIRS ----------
    fig_dir = Path("new_figures")
    fig_dir.mkdir(exist_ok=True)
    highest_dir = fig_dir / "highest"
    lowest_dir = fig_dir / "lowest"
    highest_dir.mkdir(exist_ok=True)
    lowest_dir.mkdir(exist_ok=True)

    results_txt = fig_dir / "results.txt"
    rankings_npz = fig_dir / "cell_rankings_by_session.npz"

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    n_procs = int(slurm_cpus) if slurm_cpus is not None else cpu_count()
    n_procs = min(n_procs, MAX_PROCS)

    print(f"Using {n_procs} processes (capped at {MAX_PROCS})")
    print(f"Computing SNR for {n_cells} cells per session")
    print(f"Scoring grid: n_freqs={RANK_N_FREQS}, f_min={RANK_F_MIN}, f_max={RANK_F_MAX}")

    # ---------- SCORE EACH SESSION ----------
    print("\nScoring day1...")
    scores_day1, rank_day1 = compute_scores_for_session(X1_cells, t1, n_procs=n_procs)
    print("Finished day1.\n", flush=True)

    print("Scoring day7...")
    scores_day7, rank_day7 = compute_scores_for_session(X7_cells, t7, n_procs=n_procs)
    print("Finished day7.\n", flush=True)

    print("Scoring day14...")
    scores_day14, rank_day14 = compute_scores_for_session(X14_cells, t14, n_procs=n_procs)
    print("Finished day14.\n", flush=True)

    # 3 x n_cells: each row is cell ids ordered best->worst
    rankings_by_session = np.vstack([rank_day1, rank_day7, rank_day14])

    np.savez_compressed(
        rankings_npz,
        rankings=rankings_by_session,
        scores_day1=scores_day1,
        scores_day7=scores_day7,
        scores_day14=scores_day14,
    )
    print(f"Saved rankings npz to: {rankings_npz}")

    # Choose top/bottom cells by day14 ranking (same spirit as earlier)
    highest_cells = rank_day14[:3]
    lowest_cells = rank_day14[-3:][::-1]

    print("Highest 3 cells (by day14):", highest_cells, scores_day14[highest_cells])
    print("Lowest 3 cells (by day14):", lowest_cells, scores_day14[lowest_cells])

    # ---------- RESULTS TEXT ----------
    def mean_std(x: np.ndarray):
        return float(np.nanmean(x)), float(np.nanstd(x))

    m1, s1 = mean_std(scores_day1)
    m7, s7 = mean_std(scores_day7)
    m14, s14 = mean_std(scores_day14)

    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("=== Lomb–Scargle Place SNR Summary ===\n")
        f.write(f"Mouse: {mouse_id}\n")
        f.write(f"Track length: {track_length} cm; f_place = {f_place:.8f} 1/cm\n")
        f.write(f"n_cells (intersection across all sessions): {n_cells}\n")
        f.write(f"Scoring grid: n_freqs={RANK_N_FREQS}, f_min={RANK_F_MIN}, f_max={RANK_F_MAX}\n\n")

        f.write("Per-session SNR distribution (mean ± std):\n")
        f.write(f"  day1 : mean = {m1:.6f}, std = {s1:.6f}\n")
        f.write(f"  day7 : mean = {m7:.6f}, std = {s7:.6f}\n")
        f.write(f"  day14: mean = {m14:.6f}, std = {s14:.6f}\n\n")

        f.write("Top 3 cells (ranked by day14 SNR):\n")
        for cell in highest_cells:
            f.write(
                f"  cell {cell:4d} | day1={scores_day1[cell]: .6f}  day7={scores_day7[cell]: .6f}  day14={scores_day14[cell]: .6f}\n"
            )
        f.write("\n")

        f.write("Bottom 3 cells (ranked by day14 SNR):\n")
        for cell in lowest_cells:
            f.write(
                f"  cell {cell:4d} | day1={scores_day1[cell]: .6f}  day7={scores_day7[cell]: .6f}  day14={scores_day14[cell]: .6f}\n"
            )
        f.write("\n")

    print(f"Wrote results text to: {results_txt}")

    # ---------- PLOTTING (still uses your helper lsp for high-res) ----------
    def plot_for_cell_in_session(cell: int, day_label: str, t, X_cells, base_dir: Path):
        # Convert to sample-major for your existing plotting assumptions (t vs y)
        y = X_cells[cell]

        cell_dir = base_dir / f"cell_{cell}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        freqs, power = lsp(
            t=t,
            y=y,
            n_freqs=PLOT_N_FREQS,
            f_min=PLOT_F_MIN,
            f_max=PLOT_F_MAX,
        )
        lsp_file = cell_dir / f"{day_label}_lsp.png"
        plot_lsp(freqs, power, plotfile=str(lsp_file), targets=[1 / 180])

        pf_file = cell_dir / f"{day_label}_placefield.html"
        plot_place_field_1d(
            distance_cm=t,
            spike_signal=y,
            track_length=track_length,
            save_path=str(pf_file),
        )

        raw_file = cell_dir / f"{day_label}_raw.png"
        plot_raw_signal(t, y, cell=cell, plotfile=str(raw_file))

    sessions = [
        ("day1", t1, X1_cells),
        ("day7", t7, X7_cells),
        ("day14", t14, X14_cells),
    ]

    for cell in highest_cells:
        for day_label, t, Xc in sessions:
            plot_for_cell_in_session(cell, day_label, t, Xc, highest_dir)

    for cell in lowest_cells:
        for day_label, t, Xc in sessions:
            plot_for_cell_in_session(cell, day_label, t, Xc, lowest_dir)

    print("Done.")


if __name__ == "__main__":
    main()
