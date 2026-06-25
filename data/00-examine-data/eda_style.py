"""Shared plotting style for the CXR-LT data-examination notebooks.

Import once at the top of any EDA notebook to get a consistent, publication
-quality look across every figure::

    from eda_style import apply_style, TIER_COLORS, tier_of, despine, save
    apply_style()

The goal is to replace the default ``sns.set_theme(style="whitegrid")`` look
(clashing categorical colours, heavy gridlines, no despining, tiny fonts) with
a calm, colour-blind-safe aesthetic suitable for a thesis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Colour system -----------------------------------------------------------
# Okabe-Ito derived, colour-blind safe.
INK = "#1A1A1A"          # near-black text
MUTED = "#5A5A5A"        # secondary text / reference lines
GRID = "#E2E2E2"         # light gridlines
ACCENT = "#0072B2"       # primary blue

# CXR-LT 2023 long-tail frequency tiers.
TIER_COLORS = {"Head": "#0072B2", "Medium": "#E69F00", "Tail": "#D55E00"}

# Sequential / diverging colormaps used consistently for heatmaps.
CMAP_SEQ = "rocket_r"     # counts / probabilities
CMAP_DIV = "vlag"         # correlations (centred at 0)

HEAD = [
    "Atelectasis", "Cardiomegaly", "Edema", "Lung Opacity", "No Finding",
    "Pleural Effusion", "Pneumonia", "Support Devices",
]
MEDIUM = [
    "Calcification of the Aorta", "Consolidation", "Emphysema",
    "Enlarged Cardiomediastinum", "Fracture", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pneumothorax",
]
TAIL = [
    "Fibrosis", "Lung Lesion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumoperitoneum", "Subcutaneous Emphysema",
    "Tortuous Aorta",
]
_TIER = {c: "Head" for c in HEAD}
_TIER.update({c: "Medium" for c in MEDIUM})
_TIER.update({c: "Tail" for c in TAIL})


def tier_of(label: str) -> str:
    """Return the CXR-LT frequency tier ('Head'/'Medium'/'Tail') for a label."""
    return _TIER.get(label, "Medium")


def apply_style() -> None:
    """Apply the shared rcParams. Safe to call repeatedly."""
    try:
        import seaborn as sns
        sns.set_theme(style="white", context="notebook")
    except Exception:
        pass
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.titlelocation": "left",
        "axes.labelsize": 11.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": "#9A9A9A",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 9.5,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def despine(ax, left: bool = False, bottom: bool = False) -> None:
    """Remove top/right (and optionally left/bottom) spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def save(fig, out: str | Path) -> Path:
    """Save a figure to PNG (300 dpi) and a sibling PDF; return the PNG path."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    return out
