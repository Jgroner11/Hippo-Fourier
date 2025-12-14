from pathlib import Path

import numpy as np
import polars as pl

from helpers import lsp, plot_lsp, plot_raw_signal, place_snr_from_periodogram, df_to_scrollable_html, plot_place_field_1d

track_offset = 15
track_length = 180
cue_length = 30

f_place = 1.0 / track_length  # fundamental place frequency (1/cm)

project = "StateSpaceOdyssey"
mouse_id = "26"
session = "2025-09-03-17-02-46-836208"

meso_data=pl.read_ipc("2025-09-03-17-02-46-836208.feather")
run_data=meso_data.filter(pl.col("system_state")=="run")
X_run=np.vstack(run_data["multi_day_spikes"])
t=run_data["distance_cm"]

print(meso_data.columns)

n_samples, n_cells = X_run.shape

print(f"Number of samples: {n_samples}")
print(f"Number of cells: {n_cells}")
# n_cells = 1
# n_cells = min(n_cells, X_run.shape[1])

# scores = np.zeros(n_cells)
# all_freqs = [None] * n_cells
# all_power = [None] * n_cells

# for cell in range(n_cells):
#     print(cell)
#     y = X_run[:, cell]
#     freqs, power = lsp(
#         t=t,
#         y=y,
#         n_freqs=50000,
#         f_min=0.0,
#         f_max=0.1,
#     )
#     all_freqs[cell] = freqs
#     all_power[cell] = power
#     scores[cell] = place_snr_from_periodogram(freqs, power, f_place)

# ranking = np.argsort(scores)

# lowest_cells = ranking[:5]
# highest_cells = ranking[-5:][::-1]

# print("Place SNR scores (first 100 cells):", scores)
# print("Lowest 5 cells:", lowest_cells, scores[lowest_cells])
# print("Highest 5 cells:", highest_cells, scores[highest_cells])

# for cell in lowest_cells:
#     freqs = all_freqs[cell]
#     power = all_power[cell]
#     plotfile = f"cell_{cell}_lsp_lowest.png"
#     plot_lsp(freqs, power, plotfile=plotfile, targets=[1/180, 1/60])

# for cell in highest_cells:
#     freqs = all_freqs[cell]
#     power = all_power[cell]
#     plotfile = f"cell_{cell}_lsp_highest.png"
#     plot_lsp(freqs, power, plotfile=plotfile, targets=[1/180, 1/60])

# print("Done.")


# Extract time in microseconds as numpy array
time_us = meso_data["time_us"].to_numpy()

# Compute time differences (in seconds)
dt = np.diff(time_us) * 1e-6  # microseconds → seconds

# Robust estimate of sampling interval
dt_median = np.median(dt)

# Sampling frequency (Hz)
fs = 1.0 / dt_median

print(f"Estimated sampling interval: {dt_median:.6f} s")
print(f"Estimated sampling frequency: {fs:.2f} Hz")

print("Min dt (s):", dt.min())
print("Max dt (s):", dt.max())
print("Std dt (s):", dt.std())