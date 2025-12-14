from pathlib import Path
import os
import numpy as np
import polars as pl
from multiprocessing import Pool, cpu_count

from helpers import (
    lsp,
    plot_lsp,
    place_snr_from_periodogram,
    plot_place_field_1d,
    plot_raw_signal,
)

track_length = 180.0
f_place = 1.0 / track_length  # fundamental place frequency (1/cm)

project = "StateSpaceOdyssey"
mouse_id = "26"

# ---------- SESSION FILES ----------
session1_file = "2025-09-03-17-02-46-836208.feather"
session2_file = "2025-08-21-16-35-48-370149.feather"
# ----------------------------------


# ---------- LOAD SESSION 1 ----------
meso_data1 = pl.read_ipc(session1_file)
run_data1 = meso_data1.filter(pl.col("system_state") == "run")
X_run1 = np.vstack(run_data1["multi_day_spikes"])
t1 = run_data1["distance_cm"].to_numpy()

# ---------- LOAD SESSION 2 ----------
meso_data2 = pl.read_ipc(session2_file)
run_data2 = meso_data2.filter(pl.col("system_state") == "run")
X_run2 = np.vstack(run_data2["multi_day_spikes"])
t2 = run_data2["distance_cm"].to_numpy()

# Use cells that exist in BOTH sessions
n_cells = 100
n_cells = min(n_cells, X_run1.shape[1], X_run2.shape[1])

fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)
highest_dir = fig_dir / "highest"
lowest_dir = fig_dir / "lowest"
highest_dir.mkdir(exist_ok=True)
lowest_dir.mkdir(exist_ok=True)


def cell_snr_job(cell_index):
    """Compute place SNR for one cell in SESSION 1 only (for ranking)."""
    y = X_run1[:, cell_index]
    freqs, power = lsp(
        t=t1,
        y=y,
        n_freqs=5000,   # cheaper grid for ranking
        f_min=0.0,
        f_max=0.05,
    )
    score = place_snr_from_periodogram(freqs, power, f_place)
    return cell_index, score, freqs, power


def main():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        n_procs = int(slurm_cpus)
    else:
        n_procs = cpu_count()

    print(f"Using {n_procs} processes")

    # ---- Compute SNR scores in parallel for SESSION 1 ----
    with Pool(processes=n_procs) as pool:
        results = pool.map(cell_snr_job, range(n_cells))

    scores = np.zeros(n_cells)
    all_freqs = [None] * n_cells
    all_power = [None] * n_cells

    for cell_index, score, freqs, power in results:
        scores[cell_index] = score
        all_freqs[cell_index] = freqs
        all_power[cell_index] = power

    ranking = np.argsort(scores)
    lowest_cells = ranking[:5]
    highest_cells = ranking[-5:][::-1]

    print("Place SNR scores:", scores)
    print("Lowest 5 cells:", lowest_cells, scores[lowest_cells])
    print("Highest 5 cells:", highest_cells, scores[highest_cells])

    # ---------- PLOTTING FOR SESSION 1 & 2 ----------
    # For each of the 10 cells, we plot into:
    #   figures/highest/cell_i/...  or  figures/lowest/cell_i/...

    def plot_for_cell_in_session(cell, session_id, t, X_run, base_dir):
        """
        base_dir: highest_dir or lowest_dir
        """
        y = X_run[:, cell]

        # Make per-cell directory
        cell_dir = base_dir / f"cell_{cell}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        # High-res periodogram
        freqs, power = lsp(
            t=t,
            y=y,
            n_freqs=50000,
            f_min=0.0,
            f_max=0.1,
        )
        lsp_file = cell_dir / f"session{session_id}_lsp.png"
        plot_lsp(freqs, power, plotfile=str(lsp_file), targets=[1/180])

        # Place field (HTML)
        pf_file = cell_dir / f"session{session_id}_placefield.html"
        plot_place_field_1d(
            distance_cm=t,
            spike_signal=y,
            track_length=track_length,
            save_path=str(pf_file),
        )

        # Raw signal
        raw_file = cell_dir / f"session{session_id}_raw.png"
        plot_raw_signal(t, y, cell=cell, plotfile=str(raw_file))

    # Lowest 5 cells
    for cell in lowest_cells:
        # Session 1
        plot_for_cell_in_session(cell, session_id=1, t=t1, X_run=X_run1, base_dir=lowest_dir)
        # Session 2
        plot_for_cell_in_session(cell, session_id=2, t=t2, X_run=X_run2, base_dir=lowest_dir)

    # Highest 5 cells
    for cell in highest_cells:
        # Session 1
        plot_for_cell_in_session(cell, session_id=1, t=t1, X_run=X_run1, base_dir=highest_dir)
        # Session 2
        plot_for_cell_in_session(cell, session_id=2, t=t2, X_run=X_run2, base_dir=highest_dir)

    print("Done.")


if __name__ == "__main__":
    main()
