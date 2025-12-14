"""
plot_percentile_cells_day14.py

Plots n_cells around a chosen percentile of DAY 14 rankings and
prints the DAY 14 SNR for each plotted cell.

Outputs:
  results/<percentile>percentile/
    cell_<cell_id>/
      day1_*.*
      day7_*.*
      day14_*.*
    snr_summary.txt
"""

from pathlib import Path
import numpy as np
import polars as pl

from helpers import (
    lsp,
    plot_lsp,
    plot_place_field_1d,
    plot_raw_signal,
)

# ------------------ USER VARIABLES ------------------
n_cells = 3      # number of cells to plot
percentile = 20    # percentile in DAY 14 ranking (0=best, 100=worst)
# ----------------------------------------------------

track_length = 180.0

day1_file  = "2025-08-21-16-35-48-370149.feather"
day7_file  = "2025-08-27-17-18-55-361099.feather"
day14_file = "2025-09-03-17-02-46-836208.feather"

rankings_npz = Path("results") / "cell_rankings_by_session.npz"

PLOT_N_FREQS = 50000
PLOT_F_MIN = 0.0
PLOT_F_MAX = 0.1


def load_run_session(feather_file: str):
    meso_data = pl.read_ipc(feather_file)
    run_data = meso_data.filter(pl.col("system_state") == "run")

    X_run = np.vstack(run_data["multi_day_spikes"]).astype(np.float32, copy=False)
    X_cells = np.ascontiguousarray(X_run.T)
    t = run_data["distance_cm"].to_numpy().astype(np.float64, copy=False)
    return X_cells, t


def plot_for_cell_in_session(cell, day_label, t, X_cells, base_dir):
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
    plot_lsp(freqs, power, plotfile=str(cell_dir / f"{day_label}_lsp.png"),
             targets=[1 / track_length])

    plot_place_field_1d(
        distance_cm=t,
        spike_signal=y,
        track_length=track_length,
        save_path=str(cell_dir / f"{day_label}_placefield.html"),
    )

    plot_raw_signal(
        t, y, cell=cell,
        plotfile=str(cell_dir / f"{day_label}_raw.png")
    )


def main():
    if not rankings_npz.exists():
        raise FileNotFoundError("Rankings file not found. Run multisession script first.")

    data = np.load(rankings_npz)
    rankings = data["rankings"]          # (3, n_cells)
    scores_day14 = data["scores_day14"]  # (n_cells,)

    day14_rank = rankings[2]  # best → worst
    total_cells = day14_rank.size

    # Convert percentile → rank index
    center = int(round(((100.0 - percentile) / 100.0) * (total_cells - 1)))

    half = n_cells // 2
    start = max(0, center - half)
    end = min(total_cells, start + n_cells)
    start = max(0, end - n_cells)

    selected_cells = day14_rank[start:end]

    out_dir = Path("results") / f"{int(percentile)}percentile"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sessions
    X1, t1 = load_run_session(day1_file)
    X7, t7 = load_run_session(day7_file)
    X14, t14 = load_run_session(day14_file)

    n_common = min(X1.shape[0], X7.shape[0], X14.shape[0])
    selected_cells = [c for c in selected_cells if c < n_common]

    sessions = [
        ("day1", t1, X1),
        ("day7", t7, X7),
        ("day14", t14, X14),
    ]

    print("\n=== Cells plotted (DAY 14 ranking) ===")
    summary_lines = []
    for rank_idx, cell in enumerate(selected_cells, start=start):
        snr = scores_day14[cell]
        line = f"rank={rank_idx:4d}  cell={cell:4d}  SNR_day14={snr:.6f}"
        print(line)
        summary_lines.append(line)

        for day_label, t, X in sessions:
            plot_for_cell_in_session(cell, day_label, t, X, out_dir)

    # Save SNR summary to file
    with open(out_dir / "snr_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Percentile: {percentile}\n")
        f.write(f"Cells plotted: {len(selected_cells)}\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nSaved SNR summary to: {out_dir / 'snr_summary.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
