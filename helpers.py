from pathlib import Path
import re
from datetime import datetime
import numpy as np
from scipy.signal import lombscargle
import matplotlib.pyplot as plt
import polars as pl
import html


import numpy as np
from scipy.signal import lombscargle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

def plot_place_field_1d(
    distance_cm,
    spike_signal,
    track_length,
    save_path,
    n_bins=100,
):
    """
    Compute and plot a 1D place field (rate map) as a function of position
    on a circular track, and save as an interactive Plotly HTML file.

    Parameters
    ----------
    distance_cm : array-like
        Cumulative distance traveled (same axis you use for LSP), in cm.
    spike_signal : array-like
        Spike intensity / activity values for a single cell, same length as distance_cm.
    track_length : float
        Length of the track in cm (e.g. 180).
    save_path : str or Path
        Path to save the HTML file (e.g. "cell_0_placefield.html").
    n_bins : int, optional
        Number of position bins along the track (default: 100).
    """

    distance_cm = np.asarray(distance_cm)
    spike_signal = np.asarray(spike_signal)

    mask = np.isfinite(distance_cm) & np.isfinite(spike_signal)
    distance_cm = distance_cm[mask]
    spike_signal = spike_signal[mask]

    # Wrap distance onto [0, track_length): track position on circular track
    track_pos = np.mod(distance_cm, track_length)

    # Positional binning
    bin_edges = np.linspace(0.0, track_length, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(track_pos, bin_edges)  # 1..n_bins

    mean_rate = np.full(n_bins, np.nan)
    for b in range(1, n_bins + 1):
        in_bin = bin_idx == b
        if np.any(in_bin):
            mean_rate[b - 1] = spike_signal[in_bin].mean()

    # Build Plotly figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=mean_rate,
            mode="lines",
            line=dict(width=3),
            name="Mean activity",
        )
    )

    # Cue regions: 3 cues at 0–30, 60–90, 120–150 cm
    cue_length = 30.0
    cue_positions = [0.0, 60.0, 120.0]

    # Shaded rectangles for cues
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=pos,
                x1=pos + cue_length,
                y0=0,
                y1=1,
                fillcolor="lightsteelblue",
                opacity=0.4,
                layer="below",
                line_width=0,
            )
            for pos in cue_positions
        ]
    )

    # Cue labels
    annotations = [
        dict(
            text=f"Cue {i+1}",
            xref="x",
            yref="paper",
            x=pos + cue_length / 2,
            y=1,
            xanchor="center",
            yanchor="top",
            align="center",
            showarrow=False,
        )
        for i, pos in enumerate(cue_positions)
    ]

    fig.update_layout(
        title=dict(text="1D place field", x=0.5),
        plot_bgcolor="white",
        xaxis=dict(
            title="Track position (cm)",
            range=[0, track_length],
        ),
        yaxis=dict(
            title="Mean spike signal",
        ),
        annotations=annotations,
    )

    save_path = Path(save_path)
    fig.write_html(str(save_path))
    print(f"[PLACE FIELD] Saved interactive place field HTML to: {save_path}")

    return fig

def lsp(t, y, n_freqs=20000, f_min=None, f_max=None):
    t = np.asarray(t)
    y = np.asarray(y)

    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    y = y - y.mean()

    T = t.max() - t.min()
    dt_med = np.median(np.diff(t))

    if f_min is None:
        f_min = 1.0 / T
    if f_max is None:
        f_max = 0.5 / dt_med

    freqs = np.linspace(f_min, f_max, n_freqs)
    omega = 2 * np.pi * freqs

    power = lombscargle(t, y, omega, normalize=True)

    return freqs, power

def plot_lsp(freqs, power, plotfile="figures/lsp_spectrum.png", targets=None):
    if targets is None:
        targets = []

    plt.figure(figsize=(6, 4))
    plt.plot(freqs, power, linewidth=1.2, label="Power")

    ymax = power.max()
    # Draw vertical lines at target frequencies
    for f in targets:
        plt.vlines(f, -ymax*0.05, ymax * .25, color="red", linestyles="--", linewidth=1.5)

    plt.xlabel("Spatial frequency (1/cm)")
    plt.ylabel("Power")
    plt.title("Lomb-Scargle Periodogram")
    plt.tight_layout()
    plt.savefig(plotfile, dpi=150)
    plt.close()

    print(f"[LSP] Saved spectrum to: {plotfile}")

def place_snr_from_periodogram(freqs, power, f_place):
    idx = np.argmin(np.abs(freqs - f_place))
    S_place = power[idx]
    noise = power[1:]
    mu = noise.mean()
    sigma = noise.std()
    return (S_place - mu) / sigma

def plot_raw_signal(t, y, cell=0, plotfile="raw_signal.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(t, y, linewidth=0.8)
    plt.xlabel("Distance (cm)")
    plt.ylabel("Spike signal")
    plt.title(f"Raw spike signal for cell {cell}")
    plt.tight_layout()
    plt.savefig(plotfile, dpi=150)
    plt.close()
    print(f"[RAW] Saved raw signal plot to: {plotfile}")

def print_cell_scores_and_percentiles(cell_id):
    npz_path = "results/cell_rankings_by_session.npz"
    data = np.load(npz_path)

    scores = {
        "day1": data["scores_day1"],
        "day7": data["scores_day7"],
        "day14": data["scores_day14"],
    }

    print(f"Cell {cell_id} — Lomb–Scargle SNR summary\n")

    for day, s in scores.items():
        if cell_id < 0 or cell_id >= s.size:
            raise IndexError(f"cell_id {cell_id} out of bounds for {day}")

        cell_score = s[cell_id]

        valid = np.isfinite(s)
        percentile = 100.0 * np.sum(s[valid] < cell_score) / np.sum(valid)

        print(
            f"{day:>5} | "
            f"SNR = {cell_score: .6f} | "
            f"percentile = {percentile:6.2f}"
        )

def df_to_scrollable_html(df: pl.DataFrame, columns: list[str], filename: str = "table.html") -> None:
    """
    Create an HTML file containing a scrollable table with selected columns
    from a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame.
    columns : list of str
        List of column names to include in the HTML table.
    filename : str, optional
        Name of the HTML output file (default: 'table.html').
    """

    # Select only requested columns
    subdf = df.select(columns)

    # Convert to rows of dicts for easy iteration
    rows = subdf.to_dicts()

    # Build HTML
    html_parts = []

    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='utf-8'>")
    html_parts.append("<title>DataFrame Table</title>")
    html_parts.append("""
<style>
.container {
  width: 100%;
  max-height: 600px;
  overflow-y: scroll;
  border: 1px solid #ccc;
  font-family: sans-serif;
}
table {
  border-collapse: collapse;
  width: 100%;
}
th, td {
  border: 1px solid #ddd;
  padding: 4px 8px;
  font-size: 12px;
}
th {
  position: sticky;
  top: 0;
  background: #f2f2f2;
}
tr:nth-child(even) {
  background-color: #fafafa;
}
</style>
""")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("<div class='container'>")
    html_parts.append("<table>")

    # Header
    html_parts.append("<thead><tr>")
    for col in columns:
        html_parts.append(f"<th>{html.escape(str(col))}</th>")
    html_parts.append("</tr></thead>")

    # Body
    html_parts.append("<tbody>")
    for row in rows:
        html_parts.append("<tr>")
        for col in columns:
            val = row.get(col, "")
            html_parts.append(f"<td>{html.escape(str(val))}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody>")

    html_parts.append("</table>")
    html_parts.append("</div>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    html_string = "\n".join(html_parts)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_string)

    print(f"HTML table written to {filename}")



SESSION_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}$")

def print_session_info(session_name: str, root: str = ".") -> None:
    """
    Given a session folder name (e.g. '2025-09-10-18-46-18-994971'),
    find its project and mouse, count how many sessions that mouse has,
    and print the index (chronological order) of this session.
    """

    root = Path(root).resolve()
    matches = []

    # Find all session directories that match this session name
    for p in root.rglob(session_name):
        if p.is_dir():
            try:
                parts = p.relative_to(root).parts
            except ValueError:
                continue
            if len(parts) >= 3 and parts[-1] == session_name:
                project, mouse, session = parts[-3], parts[-2], parts[-1]
                matches.append((project, mouse, session, p))

    if not matches:
        print(f"No session named '{session_name}' found under: {root}")
        return

    # Process each match (in case same session name exists in multiple projects)
    for project, mouse, session, session_path in matches:
        # Find all other sessions for this mouse
        mouse_path = session_path.parent
        all_sessions = [
            s for s in mouse_path.iterdir() if s.is_dir() and SESSION_RE.match(s.name)
        ]

        # Sort sessions chronologically based on timestamp in the name
        def session_key(s):
            # Parse timestamp from name; fallback to 0 if parsing fails
            try:
                return datetime.strptime(s.name[:26], "%Y-%m-%d-%H-%M-%S-%f")
            except ValueError:
                return datetime.min

        all_sessions.sort(key=session_key)
        session_names = [s.name for s in all_sessions]

        # Find index (1-based)
        try:
            index = session_names.index(session) + 1
        except ValueError:
            index = None

        if index:
            print(f"Project: {project}")
            print(f"Mouse:   {mouse}")
            print(f"Session index: {index} of {len(session_names)}")
        else:
            print(f"Could not determine index for session '{session}' in mouse '{mouse}'.")
