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
session = "2025-09-03-17-02-46-836208"

meso_data = pl.read_ipc("2025-09-03-17-02-46-836208.feather")
run_data = meso_data.filter(pl.col("system_state") == "run")
X_run = np.vstack(run_data["multi_day_spikes"])
t = run_data["distance_cm"].to_numpy()

n_cells = 2
n_cells = min(n_cells, X_run.shape[1])

fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)


def cell_snr_job(cell_index):
    y = X_run[:, cell_index]
    freqs, power = lsp(
        t=t,
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

    # Plot for lowest 5
    for cell in lowest_cells:
        y = X_run[:, cell]

        # High-res periodogram
        freqs, power = lsp(
            t=t,
            y=y,
            n_freqs=50000,
            f_min=0.0,
            f_max=0.1,
        )
        plotfile = fig_dir / f"cell_{cell}_lsp_lowest.png"
        plot_lsp(freqs, power, plotfile=str(plotfile), targets=[1/180, 1/60, 1/30])

        # Place field (HTML)
        pf_file = fig_dir / f"cell_{cell}_placefield_lowest.html"
        plot_place_field_1d(
            distance_cm=t,
            spike_signal=y,
            track_length=track_length,
            save_path=str(pf_file),
        )

        # Raw signal
        raw_file = fig_dir / f"cell_{cell}_raw_lowest.png"
        plot_raw_signal(t, y, cell=cell, plotfile=str(raw_file))

    # Plot for highest 5
    for cell in highest_cells:
        y = X_run[:, cell]

        # High-res periodogram
        freqs, power = lsp(
            t=t,
            y=y,
            n_freqs=50000,
            f_min=0.0,
            f_max=0.1,
        )
        plotfile = fig_dir / f"cell_{cell}_lsp_highest.png"
        plot_lsp(freqs, power, plotfile=str(plotfile), targets=[1/180, 1/60])

        # Place field (HTML)
        pf_file = fig_dir / f"cell_{cell}_placefield_highest.html"
        plot_place_field_1d(
            distance_cm=t,
            spike_signal=y,
            track_length=track_length,
            save_path=str(pf_file),
        )

        # Raw signal
        raw_file = fig_dir / f"cell_{cell}_raw_highest.png"
        plot_raw_signal(t, y, cell=cell, plotfile=str(raw_file))

    print("Done.")


if __name__ == "__main__":
    main()
