# IEEE Paper Plot Generator — Handoff Document

## What to build

A small desktop/CLI tool that lets a researcher drop in data (CSV or manually typed) and get publication-ready IEEE-style figures out — PNG at 300 DPI and PDF with embedded fonts — without writing any matplotlib code.

The user should be able to:
- Pick a plot type (line, bode, histogram, scatter, bar)
- Load their data
- Set axis labels, units, legend entries
- Get a PNG + PDF out

No matplotlib knowledge required from the user's side.

---

## The core matplotlib recipe (copy this verbatim)

This is the entire "IEEE look" — one dict applied once before any plotting.

```python
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# IEEE column widths in inches
COL1 = 3.5    # single column
COL2 = 7.16   # double column (full page width)

IEEE_RC = {
    # Serif font. Liberation Serif ships on every Linux distro.
    # Falls back to DejaVu Serif if not found.
    # If Times New Roman is installed (Windows / macOS), it uses that.
    'font.family':         'serif',
    'font.serif':          ['Liberation Serif', 'DejaVu Serif', 'Times New Roman'],

    # Point sizes. IEEE body text is 9-10 pt; figure text should match.
    'font.size':           8,       # base — everything inherits from this
    'axes.labelsize':      8,       # x/y axis label text
    'axes.titlesize':      8,       # subplot title (rarely used in IEEE)
    'xtick.labelsize':     7,       # tick numbers
    'ytick.labelsize':     7,
    'legend.fontsize':     6.5,
    'legend.framealpha':   0.88,
    'legend.edgecolor':    '#BBBBBB',
    'legend.handlelength': 1.6,
    'legend.handletextpad':0.4,
    'legend.columnspacing':0.8,

    # Line widths — thin is IEEE standard
    'lines.linewidth':     1.2,
    'axes.linewidth':      0.7,
    'xtick.major.width':   0.7,
    'ytick.major.width':   0.7,
    'xtick.minor.width':   0.4,
    'ytick.minor.width':   0.4,
    'xtick.major.size':    3.5,
    'ytick.major.size':    3.5,
    'xtick.minor.size':    2.0,
    'ytick.minor.size':    2.0,

    # Inward ticks on all four sides — THE signature IEEE look
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,

    # Subtle grid
    'axes.grid':           True,
    'grid.linewidth':      0.35,
    'grid.alpha':          0.40,
    'grid.color':          '#AAAAAA',

    # Output
    'savefig.dpi':         300,          # 300 DPI minimum for IEEE raster
    'savefig.bbox':        'tight',      # trim whitespace automatically
    'savefig.pad_inches':  0.03,

    # CRITICAL: embed fonts as TrueType in PDF.
    # Without this some journal submission portals reject the PDF.
    'pdf.fonttype':        42,
    'ps.fonttype':         42,
}

matplotlib.rcParams.update(IEEE_RC)   # call once — all plots after this inherit it
```

---

## Saving every figure

Always save both formats. Call this after every `fig`:

```python
def save_figure(fig, name, outdir="."):
    import pathlib
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outdir / f"{name}.png"))   # 300 DPI raster
    fig.savefig(str(outdir / f"{name}.pdf"))   # vector, fonts embedded
    plt.close(fig)
```

In LaTeX:
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/my_plot.pdf}
  \caption{Your caption here.}
  \label{fig:myplot}
\end{figure}
```

---

## Plot type recipes

### 1. Line plot (general — AC sweep, IV curve, anything vs frequency/time)

```python
# Corner/series colors and line styles
SERIES_STYLES = [
    dict(color="#0055FF", ls="-",           lw=1.2),   # series 1
    dict(color="#FF2200", ls=":",           lw=1.3),   # series 2
    dict(color="#00BB00", ls="--",          lw=1.2),   # series 3
    dict(color="#FF8800", ls="-.",          lw=1.2),   # series 4
    dict(color="#CC00FF", ls=(0,(4,1,1,1)), lw=1.2),   # series 5
]

fig, ax = plt.subplots(figsize=(COL1, 2.5))

for i, (xdata, ydata, label) in enumerate(series):
    sty = SERIES_STYLES[i % len(SERIES_STYLES)]
    ax.plot(xdata, ydata, label=label, **sty)

ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Gain [dB]")
ax.legend(ncol=3, loc="best")
save_figure(fig, "gain_plot", outdir="figures")
```

### 2. Log-frequency plot (PSRR, Bode gain, any AC analysis)

```python
fig, ax = plt.subplots(figsize=(COL1, 2.5))

for i, (freq, gain, label) in enumerate(series):
    sty = SERIES_STYLES[i % len(SERIES_STYLES)]
    ax.semilogx(freq, gain, label=label, **sty)

ax.axhline(0, color="#888888", lw=0.8, ls="--")   # 0 dB reference
ax.axvline(1e6, color="#555555", lw=0.8, ls="--") # 1 MHz marker

# Minor ticks on log axis
ax.xaxis.set_minor_locator(ticker.LogLocator(subs=range(2, 10)))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Gain [dB]")
ax.legend(ncol=5, loc="lower left")
save_figure(fig, "psrr", outdir="figures")
```

### 3. Bode plot (two-panel, shared x-axis)

```python
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(COL1, 4.5),
    sharex=True,
    gridspec_kw=dict(hspace=0.08),   # tight gap between panels
)

for i, (freq, gain, phase, label) in enumerate(series):
    sty = SERIES_STYLES[i % len(SERIES_STYLES)]
    ax1.semilogx(freq, gain,  label=label, **sty)
    ax2.semilogx(freq, phase, **sty)   # no legend on lower panel

ax1.axhline(0,    color="#888", lw=0.8, ls="--")
ax2.axhline(-180, color="#888", lw=0.8, ls="--")

ax1.set_ylabel("Loop Gain [dB]")
ax2.set_ylabel("Phase [°]")
ax2.set_xlabel("Frequency [Hz]")

ax1.legend(ncol=5, loc="upper right")

# Minor log ticks on shared axis (apply to ax2 — it owns the x-axis label)
ax2.xaxis.set_minor_locator(ticker.LogLocator(subs=range(2, 10)))
ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
ax2.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

# Small inset table (UGF, PM per corner) — fits inside gain subplot
table_rows = ["C    UGF      PM"]
for label, ugf_hz, pm_deg in metrics:
    table_rows.append(f"{label}  {ugf_hz/1e6:.2f}M  {pm_deg:.0f}")
ax1.text(0.02, 0.02, "\n".join(table_rows),
         transform=ax1.transAxes, fontsize=5.5,
         va="bottom", ha="left", family="monospace",
         bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#BBBBBB", lw=0.6))

save_figure(fig, "bode", outdir="figures")
```

### 4. Two-panel shared x-axis (generic — DC transfer + offset, tran + offset)

```python
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(COL1, 4.0),
    sharex=True,
    gridspec_kw=dict(hspace=0.08),
)

# top panel
for i, (x, y, label) in enumerate(top_series):
    ax1.plot(x, y, label=label, **SERIES_STYLES[i % len(SERIES_STYLES)])

# bottom panel
for i, (x, y, label) in enumerate(bottom_series):
    ax2.plot(x, y, **SERIES_STYLES[i % len(SERIES_STYLES)])

ax1.set_ylabel("V(out) [V]")
ax2.set_ylabel("Offset [mV]")
ax2.set_xlabel("V(ICM) [V]")
ax1.legend(ncol=3, loc="upper left")
save_figure(fig, "dc_transfer", outdir="figures")
```

### 5. Histogram with sigma bands (Monte Carlo offset)

```python
import numpy as np

vals = data_in_mv   # already in mV
mu, sigma, n = vals.mean(), vals.std(), len(vals)

fig, ax = plt.subplots(figsize=(COL1, 2.8))

# Sigma bands — draw widest first (gets painted over by narrower ones)
for k, col, alpha in [(3,"#FF4444",0.10), (2,"#FFAA00",0.14), (1,"#22BB22",0.20)]:
    ax.axvspan(mu - k*sigma, mu + k*sigma, color=col, alpha=alpha, lw=0)

# Histogram bars
ax.hist(vals, bins=max(20, n//8),
        color="#00AAAA", edgecolor="white", linewidth=0.4, alpha=0.88)

# Reference lines
ax.axvline(mu,           color="black",  lw=1.1, ls="-",   label=f"$\\mu$={mu:.2f}")
ax.axvline(mu + sigma,   color="#22BB22", lw=0.9, ls="--", label=f"$\\pm\\sigma$={sigma:.2f}")
ax.axvline(mu - sigma,   color="#22BB22", lw=0.9, ls="--")
ax.axvline(mu + 2*sigma, color="#FF8800", lw=0.9, ls="-.", label="$\\pm 2\\sigma$")
ax.axvline(mu - 2*sigma, color="#FF8800", lw=0.9, ls="-.")
ax.axvline(mu + 3*sigma, color="#FF4444", lw=0.9, ls=":",  label="$\\pm 3\\sigma$")
ax.axvline(mu - 3*sigma, color="#FF4444", lw=0.9, ls=":")

# Compact stats box (top-right corner)
stats = (f"$N={n}$,  $\\mu={mu:.2f}$ mV\n"
         f"$\\sigma={sigma:.3f}$ mV,  $\\sigma^2={sigma**2:.3f}$ mV$^2$")
ax.text(0.97, 0.97, stats, transform=ax.transAxes,
        fontsize=6.0, va="top", ha="right",
        bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#BBBBBB", lw=0.6))

ax.legend(ncol=2, loc="upper left", handlelength=1.4, fontsize=6.0)
ax.set_xlabel("Input-Referred Offset [mV]")
ax.set_ylabel("Count")
save_figure(fig, "mc_histogram", outdir="figures")
```

---

## Why each setting exists

| Setting | Reason |
|---|---|
| `font.family = serif` | IEEE uses Times New Roman body text. Matching serif in figures looks intentional, not accidental |
| `figsize = (3.5, h)` | 3.5 in fills exactly one IEEE column. If you set it wider, LaTeX scaling distorts font sizes |
| `xtick.direction = in` | Inward ticks are standard in IEEE, Nature, Science. They don't obscure data at the edges |
| `xtick.top = True` | Ticks on all four sides creates a closed data frame — the classic journal look |
| `hspace = 0.08` | Closes the gap between shared-axis subplots so they read as one figure |
| `pdf.fonttype = 42` | Embeds fonts as TrueType in the PDF. Without this, some IEEE portals reject the file or fonts render as bitmaps |
| `savefig.dpi = 300` | IEEE minimum for raster. Use 600 if the figure has fine lines |
| `savefig.bbox = tight` | Trims whitespace so the figure fills its `\columnwidth` cleanly |

---

## Software requirements for the tool to build

### Inputs the user provides (via GUI or config file)
- Plot type: `line` / `bode` / `histogram` / `two_panel` / `scatter`
- CSV file path(s), or inline data paste
- X column name, Y column name(s)
- Series labels (one per column or file)
- Axis labels and units (e.g. `"Frequency [Hz]"`)
- X scale: `linear` or `log`
- Figure width: `single` (3.5 in) or `double` (7.16 in)
- Figure height (in inches, default suggestions per type)
- Output folder and filename stem

### What the tool does automatically
- Applies `IEEE_RC` before every plot
- Assigns colors and line styles from the `SERIES_STYLES` list (no user input needed)
- Adds minor log ticks when x-scale is log
- Saves `stem.png` and `stem.pdf` to the output folder
- For histograms: computes and annotates μ, σ, σ², draws sigma bands
- For Bode: auto-detects 0 dB crossing, computes phase margin, puts inset table

### Suggested tech stack
- **Backend**: Python, matplotlib, pandas, numpy
- **GUI options** (pick one):
  - `tkinter` — zero install, ships with Python, good enough for a utility tool
  - `PyQt6` — better looking, drag-and-drop support
  - `streamlit` — browser-based, fastest to build, easy to share
- **Config**: YAML or TOML file so users can save/reuse plot settings
- **Packaging**: `pyinstaller` to make a standalone `.exe` / `.app` / Linux binary

### Minimal viable feature list (v1)
1. Load CSV, pick x/y columns
2. Choose plot type
3. Type axis labels
4. Click "Generate" → PNG + PDF appear in output folder
5. "Last used settings" saved to `~/.ieeeplotter.toml` so next launch is one click

---

## Notes for the new chat

- The `IEEE_RC` dict at the top of this document is the complete, working, tested recipe. Start there.
- The font fallback chain (`Liberation Serif` → `DejaVu Serif` → `Times New Roman`) works on Linux, macOS, and Windows without installing anything extra.
- `pdf.fonttype = 42` is non-negotiable. Don't remove it.
- All plot sizes assume single-column IEEE (3.5 in). Double-column is 7.16 in — just change `figsize`.
- `gridspec_kw=dict(hspace=0.08)` for two-panel plots is the key to the tight-panel look.
- Do not use `plt.title()` in IEEE figures — titles belong in the caption, not the figure.
- Use `ax.text()` with `transform=ax.transAxes` for inset annotations (coordinates 0–1 relative to axes, not data units).
