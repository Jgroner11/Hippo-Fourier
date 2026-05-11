# Hippo-Fourier

Project detecting place cells via Lomb-Scargle Periodogram.

Project description: [`report/report.pdf`](report/report.pdf).

This repository does not include the data, which is proprietary. Instead, it stores periodograms and rate maps for many cells in [`results/`](results/).

---

## Results

The generated figures are in [`results/`](results/). Cells are ranked by Day 14 SNR, where higher SNR means stronger evidence that the cell is a place cell.

- [`highest/`](results/highest/) - the three cells with the highest Day 14 SNR, making them the strongest place-cell candidates.
- [`90percentile/`](results/90percentile/) - cells near the 90th percentile of Day 14 SNR, making them strong place-cell candidates.

  &vellip;

- [`10percentile/`](results/10percentile/) - cells near the 10th percentile of Day 14 SNR, making them weak place-cell candidates.
- [`lowest/`](results/lowest/) - the three cells with the lowest Day 14 SNR, making them the weakest place-cell candidates.

Each percentile folder contains three representative `cell_<id>/` folders and a `snr_summary.txt` file with the exact cell IDs, ranks, and Day 14 SNR values. Each cell folder contains Day 1, Day 7, and Day 14 versions of the following plots.

### Raw trace (`*_raw.png`)

Plots the cell's raw spike signal against cumulative distance traveled. This is the least-processed view of the cell's activity before it is summarized by the periodogram or place field.

### Lomb-Scargle periodogram (`*_lsp.png`)

Computes a Lomb-Scargle periodogram of the cell's activity with respect to position. The important spatial frequency is:

1 / 180 cm<sup>-1</sup> ≈ 0.0056 cm<sup>-1</sup>

A strong peak at this frequency means the cell tends to fire once per lap of the repeating 180 cm virtual track. SNR is computed from how strongly this target-frequency peak stands out from the rest of the periodogram.

### Place field (`*_placefield.html`)

Computes an interactive spatial rate map by wrapping cumulative distance onto the 180 cm track and averaging activity within position bins. This shows where along the repeated track segment the cell is most active.

The `.html` files need to be opened in a web browser to view the plot.

### Summary Files

- [`results/results.txt`](results/results.txt) gives the overall SNR summary, including mean SNR for Day 1, Day 7, and Day 14, plus the top and bottom ranked cells.
- `snr_summary.txt` inside each percentile folder lists the cells selected for that folder.
- [`results/cell_rankings_by_session.npz`](results/cell_rankings_by_session.npz) stores the numerical SNR scores and rankings used to generate the result folders.

---

## Code Files

- [`helpers.py`](helpers.py) - Lomb-Scargle, SNR, and plotting utilities.
- [`multisession_place_cell_ranking.py`](multisession_place_cell_ranking.py) - ranks cells across Day 1, Day 7, and Day 14.
- [`plot_specific_percentile.py`](plot_specific_percentile.py) - generates example plots for chosen Day 14 SNR percentiles.

---

## Citation

Jacob Groner,  
*Lomb-Scargle Periodogram for Place Cell Detection*,  
Cornell University (ASTRO 4523 Final Project).
