#!/usr/bin/env python3
"""Thesis figure: long-tail performance of the prior-aware multimodal CXR model.

Builds a publication-style two-panel figure from a training run's
``logs/val_epochs.csv``:

  Panel A — per-class Average Precision (AP) at the best epoch, ranked and
            colour-coded by CXR-LT 2023 frequency tier (head / medium / tail).
  Panel B — head vs. medium vs. tail mean AP across training epochs, showing
            how the long tail lags the head throughout optimisation.

The figure tells the central long-tail story of CXR-LT multi-label disease
classification in a single, self-contained graphic.

Usage
-----
    python scripts/plot_thesis_longtail.py \
        --run output/prior_aware_v6nano/runs/20260624-111754-baseline \
        --out report/img/longtail_performance.png

All defaults point at the baseline run, so a bare invocation reproduces the
figure shipped with the report.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- CXR-LT 2023 long-tail frequency tiers (resolved by class name) ----------
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

TIER_OF = {c: "Head" for c in HEAD}
TIER_OF.update({c: "Medium" for c in MEDIUM})
TIER_OF.update({c: "Tail" for c in TAIL})

# Colour-blind-safe palette (Okabe-Ito derived).
COLORS = {"Head": "#0072B2", "Medium": "#E69F00", "Tail": "#D55E00"}

# Nicer short labels for crowded tick text.
SHORT = {
    "Calcification of the Aorta": "Calcification of Aorta",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomed.",
    "Subcutaneous Emphysema": "Subcut. Emphysema",
}


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def best_epoch(rows: list[dict]) -> dict:
    """Row with the highest overall validation AP."""
    return max(rows, key=lambda r: float(r["val/ap"]))


def per_class_ap(row: dict) -> dict[str, float]:
    out = {}
    prefix = "val/ap/"
    for k, v in row.items():
        if k.startswith(prefix) and k.split("/")[-1] in TIER_OF:
            out[k[len(prefix):]] = float(v)
    return out


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def build(run: Path, out: Path) -> None:
    rows = load_rows(run / "logs" / "val_epochs.csv")
    best = best_epoch(rows)
    aps = per_class_ap(best)

    ranked = sorted(aps.items(), key=lambda kv: kv[1])  # ascending -> barh bottom=worst
    names = [SHORT.get(n, n) for n, _ in ranked]
    vals = [v for _, v in ranked]
    tiers = [TIER_OF[n] for n, _ in ranked]
    bar_colors = [COLORS[t] for t in tiers]

    epochs = [int(r["epoch"]) for r in rows]
    head_c = [float(r["val/ap_head"]) for r in rows]
    med_c = [float(r["val/ap_medium"]) for r in rows]
    tail_c = [float(r["val/ap_tail"]) for r in rows]
    macro_c = [float(r["val/ap"]) for r in rows]

    overall_ap = float(best["val/ap"])
    overall_auroc = float(best["val/auroc"])
    be = int(best["epoch"])

    style()
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28,
                          left=0.20, right=0.965, top=0.86, bottom=0.10)

    # ---------------- Panel A: ranked per-class AP -----------------------
    axA = fig.add_subplot(gs[0, 0])
    y = range(len(names))
    axA.barh(y, vals, color=bar_colors, edgecolor="white", linewidth=0.6,
             height=0.74, zorder=3)
    for yi, v in zip(y, vals):
        axA.text(v + 0.008, yi, f"{v:.2f}", va="center", ha="left",
                 fontsize=8.3, color="#333333", zorder=4)
    axA.set_yticks(list(y))
    axA.set_yticklabels(names, fontsize=9.2)
    axA.set_xlim(0, max(vals) * 1.16)
    axA.set_xlabel("Average Precision (AP)")
    axA.set_title("A   Per-class AP by frequency tier", loc="left", pad=10)
    axA.axvline(overall_ap, color="#555555", ls="--", lw=1.1, zorder=2)
    axA.text(overall_ap, len(names) - 0.2, f"  macro AP = {overall_ap:.2f}",
             color="#555555", fontsize=8.6, va="top", ha="left")
    axA.grid(axis="x", color="#DDDDDD", lw=0.7, zorder=0)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    legend = [Patch(facecolor=COLORS[t], label=f"{t} ({n})")
              for t, n in (("Head", len(HEAD)), ("Medium", len(MEDIUM)),
                           ("Tail", len(TAIL)))]
    axA.legend(handles=legend, title="CXR-LT frequency tier", loc="lower right",
               frameon=True, fontsize=9, title_fontsize=9.5)

    # ---------------- Panel B: tier learning curves ----------------------
    axB = fig.add_subplot(gs[0, 1])
    for ys, key, lab in ((head_c, "Head", "Head"),
                         (med_c, "Medium", "Medium"),
                         (tail_c, "Tail", "Tail")):
        axB.plot(epochs, ys, color=COLORS[key], lw=2.4, marker="o", ms=5,
                 markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                 label=lab)
        axB.text(epochs[-1] + 0.12, ys[-1], f"{ys[-1]:.2f}", color=COLORS[key],
                 fontsize=9.5, va="center", fontweight="bold")
    # overall macro AP (across all 26 classes) as an aggregate reference
    axB.plot(epochs, macro_c, color="#444444", lw=1.8, ls="--", marker="s",
             ms=3.5, zorder=2, label="Macro (all)")
    axB.text(epochs[-1] + 0.12, macro_c[-1], f"{macro_c[-1]:.2f}",
             color="#444444", fontsize=9.5, va="center", fontweight="bold")
    # shade the head-tail gap
    axB.fill_between(epochs, tail_c, head_c, color="#999999", alpha=0.10,
                     zorder=1)
    axB.axvline(be, color="#555555", ls=":", lw=1.0, zorder=1)
    axB.text(be, axB.get_ylim()[0], " best", color="#555555", fontsize=8.3,
             va="bottom", ha="left")
    axB.set_xlabel("Training epoch")
    axB.set_ylabel("Mean AP")
    axB.set_title("B   Tier learning curves", loc="left", pad=10)
    axB.set_xlim(min(epochs) - 0.3, max(epochs) + 0.9)
    axB.set_ylim(0, max(head_c) * 1.12)
    axB.grid(color="#DDDDDD", lw=0.7, zorder=0)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.legend(loc="lower right", frameon=True, fontsize=9.5)

    gap = head_c[be] - tail_c[be]
    axB.annotate(f"head-tail gap\n{gap:.2f} AP", xy=(be, (head_c[be] + tail_c[be]) / 2),
                 xytext=(be - 4.2, max(head_c) * 0.55), fontsize=9,
                 color="#444444", ha="center",
                 arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.1))

    # ---------------- titles ---------------------------------------------
    fig.suptitle("Long-tail disease recognition: prior-aware multimodal CXR model",
                 x=0.20, ha="left", fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.20, 0.905,
             f"CaMCheX prior-aware (v6nano)  ·  CXR-LT 2023, 26 classes  ·  "
             f"best epoch {be}: macro AP {overall_ap:.2f}, macro AUROC {overall_auroc:.2f}",
             ha="left", fontsize=10.5, color="#555555")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  and  {out.with_suffix('.pdf')}")
    print(f"best epoch {be}: head {head_c[be]:.3f}  medium {med_c[be]:.3f}  "
          f"tail {tail_c[be]:.3f}  (gap {gap:.3f})")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path,
                   default=repo / "output/prior_aware_v6nano/runs/20260624-111754-baseline",
                   help="run directory containing logs/val_epochs.csv")
    p.add_argument("--out", type=Path,
                   default=repo / "report/img/longtail_performance.png",
                   help="output PNG path (a .pdf is written alongside)")
    a = p.parse_args()
    build(a.run, a.out)


if __name__ == "__main__":
    main()
