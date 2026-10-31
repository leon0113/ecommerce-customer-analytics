"""Shared chart style and helpers so every figure in the project looks the same.

Colours follow a colour-blind-checked palette: one blue for single-series charts, a fixed
categorical order for multi-series charts, and a single-hue blue ramp for heatmaps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#c9c8c2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f0f6fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def apply_style() -> None:
    """Set project-wide matplotlib defaults: light surface, recessive grid, readable type."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 22,
        "axes.labelsize": 11,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def bdt(value: float, _position: int | None = None) -> str:
    """Compact taka label: 1.2M, 450K, 900."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def new_figure(width: float = 10, height: float = 5) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with one axes in the project style."""
    apply_style()
    return plt.subplots(figsize=(width, height))


def set_titles(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Bold left-aligned title with an optional one-line takeaway underneath."""
    ax.set_title(title)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=TEXT_SECONDARY, fontsize=10.5, va="bottom")


def format_axis(ax: plt.Axes, axis: str, kind: str) -> None:
    """Format an axis as 'bdt' (compact taka) or 'pct' (share 0-1 shown as %)."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(FuncFormatter(bdt) if kind == "bdt" else PercentFormatter(1.0, decimals=0))


def bar_chart(
    ax: plt.Axes,
    labels: list[str],
    values: list[float] | np.ndarray,
    value_format: str = "{:,.0f}",
    highlight: list[str] | None = None,
    horizontal: bool = True,
) -> None:
    """Single-series bar chart with value labels; `highlight` bars stay blue, the rest go muted."""
    colors = [SERIES[0] if highlight is None or label in highlight else MUTED for label in labels]
    if horizontal:
        bars = ax.barh(labels, values, color=colors, height=0.65)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.12)
    else:
        bars = ax.bar(labels, values, color=colors, width=0.65)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    ax.bar_label(bars, labels=[value_format.format(v) for v in values], padding=4, color=TEXT, fontsize=10)


def grouped_bar_chart(ax: plt.Axes, table: pd.DataFrame, value_format: str = "{:.0%}") -> None:
    """Side-by-side bars, one colour per column, with a legend and value labels."""
    n_series = len(table.columns)
    width = 0.8 / n_series
    positions = np.arange(len(table))
    for i, column in enumerate(table.columns):
        bars = ax.bar(positions + (i - (n_series - 1) / 2) * width, table[column], width * 0.92,
                      color=SERIES[i], label=column)
        ax.bar_label(bars, labels=[value_format.format(v) for v in table[column]], padding=3, fontsize=9)
    ax.set_xticks(positions, table.index)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.margins(y=0.12)


def heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    value_format: str = "{:.0%}",
    colorbar_label: str = "",
) -> None:
    """Single-hue heatmap with every cell labelled; empty cells stay blank."""
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    threshold = np.nanmin(values) + 0.6 * (np.nanmax(values) - np.nanmin(values))
    for (row, col), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(col, row, value_format.format(value), ha="center", va="center", fontsize=9,
                    color=SURFACE if value > threshold else TEXT)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(colorbar_label, color=TEXT_SECONDARY)
    colorbar.outline.set_visible(False)


def waterfall(ax: plt.Axes, start_label: str, start: float, steps: pd.Series, end_label: str) -> None:
    """Bridge from a starting total to an ending total through labelled positive/negative steps."""
    running = start + steps.cumsum()
    bottoms = np.concatenate([[0], np.minimum(running.shift(1, fill_value=start), running), [0]])
    heights = np.concatenate([[start], steps.abs(), [running.iloc[-1]]])
    labels = [start_label, *steps.index, end_label]
    colors = [SERIES[0], *[SERIES[2] if s >= 0 else SERIES[7] for s in steps], SERIES[0]]
    bars = ax.bar(labels, heights, bottom=bottoms, color=colors, width=0.65)
    deltas = [bdt(start), *[("+" if s >= 0 else "-") + bdt(abs(s)) for s in steps], bdt(running.iloc[-1])]
    ax.bar_label(bars, labels=deltas, padding=3, fontsize=9.5)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    format_axis(ax, "y", "bdt")


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a figure as PNG, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path
