# Hippo-Fourier — Lomb–Scargle Place Cell Detection

This repository contains code and results for a class project on **detecting hippocampal place cells using Lomb–Scargle periodograms**.

The code is **not intended to be run**; it documents the analysis pipeline used to generate the figures and summary statistics.  
The primary outputs are in the `results/` directory.

---

## Idea (brief)

- The virtual track repeats every **180 cm**
- A place cell fires once per lap → strong power at  
  **f = 1 / 180 cm⁻¹**
- Mouse speed varies → non-uniform sampling in position
- A **Lomb–Scargle periodogram** is used instead of an FFT
- Cells are ranked by **signal-to-noise ratio (SNR)** at the target frequency

---

## Repository contents

- `helpers.py` — Lomb–Scargle, SNR, and plotting utilities  
- `multisession_place_cell_ranking.py` — ranks cells across Day 1 / 7 / 14  
- `plot_specific_percentile.py` — visualizes cells at chosen SNR percentiles  
- `results/` — all figures and summary statistics

---

## Results (`results/`)

This folder contains the scientific outputs of the project.

- **`cell_rankings_by_session.npz`**  
  SNR scores and per-session rankings for all tracked cells.

- **`results.txt`**  
  Mean and standard deviation of SNR per session, plus top and bottom ranked cells.

- **`highest/` and `lowest/`**  
  Example cells with the strongest and weakest Day 14 SNRs.

- **`<N>percentile/` (20–90)**  
  Example cells drawn from different Day 14 SNR percentiles.  
  Each folder includes:
  - Lomb–Scargle periodograms (`*_lsp.png`)
  - Raw activity traces (`*_raw.png`)
  - Interactive spatial rate maps (`*_placefield.html`)

Open the HTML files in a browser to explore place fields interactively.

---

## Citation

Jacob Groner,  
*Lomb–Scargle Periodogram for Place Cell Detection*,  
Cornell University (ASTRO 4523 Final Project).
