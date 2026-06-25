#!/usr/bin/env python3
"""Supplementary thesis EDA figures for the CaMCheX / CXR-LT dataset.

Companion to ``make_eda_figures.py``. These figures either redraw report
figures that currently use the default style, or add a figure for an analysis
that the report so far presents only as a table.

    fig4_cohort_modality.png  — cohort size and per-modality availability by
                                split (Table: cohort_summary -> figure).
    fig5_text.png             — clinical indication text: length distribution
                                + top terms split into clinical signal vs.
                                demographic boilerplate (Table -> figure).
    fig6_vitals_quality.png   — vital-sign missingness + observed-vs-plausible
                                ranges exposing data-entry outliers.
    fig7_longtail40.png       — full 40-class CXR-LT 2024 long-tail.

Figures are driven by the precomputed CSVs in ``report/img/figures/`` (the
canonical numbers behind the report) plus the prepared dataset for the text
length distribution. Shared look comes from ``eda_style.py``.

Usage
-----
    python data/00-examine-data/make_eda_figures_extra.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from eda_style import (MUTED, TIER_COLORS, apply_style, despine, save)

ACCENT = TIER_COLORS["Head"]      # blue
WARN = TIER_COLORS["Medium"]      # amber
ALERT = TIER_COLORS["Tail"]       # vermillion

# Plausible physiological ranges for the data-quality panel.
VITAL_PLAUSIBLE = {
    "Temperature": (95.0, 105.0),
    "Heart rate": (30.0, 200.0),
    "Respiratory rate": (5.0, 60.0),
    "O2 saturation": (70.0, 100.0),
    "Systolic BP": (60.0, 250.0),
    "Diastolic BP": (30.0, 150.0),
}

# Hand-curated split of the top indication terms (report common_text_terms).
DEMOGRAPHIC = {"old", "year", "man", "woman", "year-old", "history", "please",
               "status", "change", "interval", "placement", "assess",
               "evaluate", "eval"}


# ---------------------------------------------------------------------------
def fig_cohort(fig_csv_dir: Path, out: Path) -> None:
    c = pd.read_csv(fig_csv_dir / "cohort_summary.csv")
    c = c[c["Subset"] != "Total"].copy()
    splits = c["Subset"].tolist()
    x = np.arange(len(splits))
    w = 0.26

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.4, 1.0], wspace=0.25))
    fig.subplots_adjust(top=0.80)

    # NOTE: in cohort_summary.csv, "Images" is image-level while
    # "Text Available"/"Vitals Available" are study-level counts, so modality
    # availability is computed against Studies (not Images).
    bars = [("Studies", c["Studies"], ACCENT),
            ("with text", c["Text Available"], WARN),
            ("with vitals", c["Vitals Available"], ALERT)]
    for i, (lab, vals, col) in enumerate(bars):
        axL.bar(x + (i - 1) * w, vals, width=w, label=lab, color=col,
                edgecolor="white", linewidth=0.6, zorder=3)
    axL.set_xticks(x)
    axL.set_xticklabels(splits)
    axL.set_ylabel("Studies")
    axL.set_title("A   Study-level modality availability")
    axL.grid(axis="y")
    despine(axL)
    axL.legend(loc="upper right")
    for i, (_, vals, _) in enumerate(bars):
        for xi, v in zip(x, vals):
            axL.text(xi + (i - 1) * w, v * 1.01, f"{v/1000:.0f}k",
                     ha="center", va="bottom", fontsize=7.6, color="#444")

    # Panel B: vitals missingness rate per split (study-level)
    miss = (1 - c["Vitals Available"].values / c["Studies"].values) * 100
    txt_miss = (1 - c["Text Available"].values / c["Studies"].values) * 100
    axR.bar(x - 0.18, miss, width=0.34, color=ALERT, edgecolor="white",
            zorder=3, label="vitals")
    axR.bar(x + 0.18, txt_miss, width=0.34, color=WARN, edgecolor="white",
            zorder=3, label="text")
    axR.set_xticks(x)
    axR.set_xticklabels(splits)
    axR.set_ylabel("Studies missing modality (%)")
    axR.set_title("B   Missingness by split")
    axR.set_ylim(0, max(miss) * 1.4)
    for xi, v in zip(x, miss):
        axR.text(xi - 0.18, v + max(miss) * 0.03, f"{v:.0f}%", ha="center",
                 fontsize=9.5, fontweight="bold", color=ALERT)
    for xi, v in zip(x, txt_miss):
        axR.text(xi + 0.18, v + max(miss) * 0.03, f"{v:.0f}%", ha="center",
                 fontsize=9.5, fontweight="bold", color="#B07A00")
    axR.grid(axis="y")
    despine(axR)
    axR.legend(loc="upper right")

    tot = pd.read_csv(fig_csv_dir / "cohort_summary.csv").query("Subset=='Total'").iloc[0]
    fig.suptitle("Dataset cohort and modality availability",
                 x=0.07, ha="left", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.07, 0.92,
             f"{int(tot['Patients']):,} patients · {int(tot['Studies']):,} studies · "
             f"{int(tot['Images']):,} images · text for {tot['Text Available']/tot['Studies']*100:.0f}% "
             f"and vitals for {tot['Vitals Available']/tot['Studies']*100:.0f}% of studies",
             ha="left", fontsize=10.5, color=MUTED)
    save(fig, out)
    plt.close(fig)
    print(f"  fig4: {out.name}")


# ---------------------------------------------------------------------------
def fig_text(fig_csv_dir: Path, data_dir: Path, out: Path) -> None:
    # (A) text length distribution from prepared data (study-level)
    d = pd.read_csv(data_dir / "03_mimic_train.csv",
                    usecols=["study_id", "clinical_indication"], low_memory=False)
    txt = d.drop_duplicates("study_id")["clinical_indication"].dropna().astype(str)
    wl = txt.str.split().apply(len)
    wl_clip = wl.clip(upper=60)

    # (B) top terms, split clinical vs demographic
    terms = pd.read_csv(fig_csv_dir / "common_text_terms.csv").head(20)
    terms = terms.sort_values("frequency")
    cats = ["Demographic / template" if t in DEMOGRAPHIC else "Clinical signal"
            for t in terms["term"]]
    colors = [MUTED if c.startswith("Demo") else ACCENT for c in cats]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1.0, 1.05], wspace=0.22))
    fig.subplots_adjust(top=0.80)

    axA.hist(wl_clip, bins=range(0, 62, 2), color=ACCENT, edgecolor="white",
             linewidth=0.6, zorder=3)
    axA.axvline(wl.median(), color=ALERT, ls="--", lw=1.6, zorder=4,
                label=f"median = {wl.median():.0f} words")
    axA.axvline(wl.quantile(0.9), color=WARN, ls="--", lw=1.4, zorder=4,
                label=f"90th pct = {wl.quantile(0.9):.0f} words")
    axA.set_xlabel("Indication length (words, clipped at 60)")
    axA.set_ylabel("# studies")
    axA.set_title("A   Clinical indication length")
    axA.grid(axis="y")
    despine(axA)
    axA.legend(loc="upper right")

    y = np.arange(len(terms))
    axB.barh(y, terms["frequency"], color=colors, edgecolor="white",
             linewidth=0.6, height=0.78, zorder=3)
    axB.set_yticks(y)
    axB.set_yticklabels(terms["term"])
    axB.set_xlabel("Frequency in indication text")
    axB.set_title("B   Top-20 terms: clinical signal vs. boilerplate")
    axB.grid(axis="x")
    despine(axB)
    for yi, v in zip(y, terms["frequency"]):
        axB.text(v * 1.01, yi, f"{int(v/1000)}k", va="center", fontsize=7.8,
                 color="#444")
    legend = [Patch(facecolor=ACCENT, label="Clinical signal"),
              Patch(facecolor=MUTED, label="Demographic / template")]
    axB.legend(handles=legend, loc="lower right")

    fig.suptitle("Clinical indication text: short, skewed, and boilerplate-heavy",
                 x=0.07, ha="left", fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.07, 0.91,
             f"{len(txt):,} studies with indication text · "
             f"mean {wl.mean():.1f} words, max {wl.max()} · motivates a short "
             "max-token limit and clinical-domain embeddings",
             ha="left", fontsize=10.5, color=MUTED)
    save(fig, out)
    plt.close(fig)
    print(f"  fig5: {out.name}  (median {wl.median():.0f} words)")


# ---------------------------------------------------------------------------
def fig_vitals(fig_csv_dir: Path, out: Path) -> None:
    v = pd.read_csv(fig_csv_dir / "vitals_summary_train.csv")
    v = v.set_index("Variable").reindex(list(VITAL_PLAUSIBLE))
    order = v.index.tolist()
    y = np.arange(len(order))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[0.8, 1.25], wspace=0.32))
    fig.subplots_adjust(top=0.80)

    # Panel A: missingness per vital
    miss = v["Missing (%)"]
    axL.barh(y, miss.values, color=WARN, edgecolor="white", height=0.7, zorder=3)
    axL.set_yticks(y)
    axL.set_yticklabels(order)
    axL.invert_yaxis()
    axL.set_xlabel("Missing in train (%)")
    axL.set_title("A   Vital-sign missingness")
    axL.set_xlim(0, max(miss.values) * 1.25)
    for yi, m in zip(y, miss.values):
        axL.text(m + 0.4, yi, f"{m:.1f}%", va="center", fontsize=9)
    axL.grid(axis="x")
    despine(axL)
    axL.axvline(miss.mean(), color=ALERT, ls="--", lw=1.3)
    axL.text(miss.mean(), -0.6, f"mean {miss.mean():.1f}%", color=ALERT,
             fontsize=8.5, ha="center")

    # Panel B: observed range (min..max) vs plausible band, log scale
    axR.set_xscale("symlog")
    for yi, var in zip(y, order):
        lo, hi = float(v.loc[var, "Min"]), float(v.loc[var, "Max"])
        med = float(v.loc[var, "Median"])
        plo, phi = VITAL_PLAUSIBLE[var]
        # plausible band
        axR.plot([plo, phi], [yi, yi], color=ACCENT, lw=9, alpha=0.25,
                 solid_capstyle="round", zorder=2)
        # observed full range
        axR.plot([max(lo, 0.1), hi], [yi, yi], color="#888", lw=1.6, zorder=3)
        axR.scatter([max(lo, 0.1), hi], [yi, yi], color="#666", s=18, zorder=4)
        axR.scatter([med], [yi], color=ALERT, s=55, zorder=5,
                    edgecolor="white", linewidth=0.8)
        # annotate implausible max
        if hi > phi * 1.5:
            axR.text(hi, yi + 0.18, f"max {hi:,.0f}", fontsize=8, color=ALERT,
                     ha="right", va="bottom")
    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.invert_yaxis()
    axR.set_xlabel("Recorded value (symlog scale)")
    axR.set_title("B   Observed range vs. plausible range (data-entry outliers)")
    axR.grid(axis="x")
    despine(axR)
    legend = [Patch(facecolor=ACCENT, alpha=0.25, label="plausible range"),
              Line2D([0], [0], color="#888", lw=1.6, marker="o", mfc="#666",
                     mec="#666", label="observed min–max"),
              Line2D([0], [0], color="w", marker="o", mfc=ALERT, mec="white",
                     ms=8, label="median")]
    axR.legend(handles=legend, loc="lower right", fontsize=9)

    fig.suptitle("Vital signs: uniform missingness and heavy-tailed data-entry errors",
                 x=0.07, ha="left", fontsize=15.5, fontweight="bold", y=0.98)
    fig.text(0.07, 0.92,
             "Implausible extremes (e.g. diastolic BP up to 74,810 mmHg) motivate "
             "outlier clipping + train-set z-score normalization and a missingness mask",
             ha="left", fontsize=10.2, color=MUTED)
    save(fig, out)
    plt.close(fig)
    print(f"  fig6: {out.name}")


# ---------------------------------------------------------------------------
def fig_longtail40(fig_csv_dir: Path, out: Path) -> None:
    lc = pd.read_csv(fig_csv_dir / "label_positive_counts.csv")
    lc = lc.sort_values("positive_studies", ascending=True)
    n = len(lc)
    # tertile split by frequency rank -> head / medium / tail
    ranks = np.argsort(np.argsort(lc["positive_studies"].values))  # 0=rarest
    tier = np.where(ranks >= 2 * n / 3, "Head",
                    np.where(ranks >= n / 3, "Medium", "Tail"))
    colors = [TIER_COLORS[t] for t in tier]

    fig, ax = plt.subplots(figsize=(11, 12))
    y = np.arange(n)
    ax.barh(y, lc["positive_studies"], color=colors, edgecolor="white",
            linewidth=0.5, height=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(lc["label"], fontsize=8.6)
    ax.set_xlabel("Positive studies (log scale)")
    ax.set_xlim(left=80)
    ax.grid(axis="x")
    despine(ax)
    for yi, c in zip(y, lc["positive_studies"]):
        ax.text(c * 1.06, yi, f"{int(c):,}", va="center", fontsize=7.6,
                color="#444")

    ratio = lc["positive_studies"].max() / lc["positive_studies"].min()
    counts = {t: int((tier == t).sum()) for t in ("Head", "Medium", "Tail")}
    legend = [Patch(facecolor=TIER_COLORS[t], label=f"{t} ({counts[t]})")
              for t in ("Head", "Medium", "Tail")]
    ax.legend(handles=legend, title="frequency tertile", loc="lower right")

    fig.suptitle("Full long-tail: CXR-LT 2024 (40 findings)",
                 x=0.13, ha="left", fontsize=16, fontweight="bold", y=0.95)
    fig.text(0.13, 0.925,
             f"{n} classes · rarest {lc['label'].iloc[0]} "
             f"({int(lc['positive_studies'].iloc[0])}) to "
             f"{lc['label'].iloc[-1]} ({int(lc['positive_studies'].iloc[-1]):,}) "
             f"— {ratio:,.0f}× imbalance",
             ha="left", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out)
    plt.close(fig)
    print(f"  fig7: {out.name}  ({n} classes, {ratio:,.0f}x)")


# matplotlib Line2D import kept local to avoid top-level clutter
from matplotlib.lines import Line2D  # noqa: E402


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=repo / "data" / "data-camchex")
    ap.add_argument("--fig-csv-dir", type=Path,
                    default=repo / "report" / "img" / "figures")
    ap.add_argument("--out-dir", type=Path, default=repo / "report" / "img" / "eda")
    a = ap.parse_args()

    apply_style()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"csv source: {a.fig_csv_dir}")
    fig_cohort(a.fig_csv_dir, a.out_dir / "fig4_cohort_modality.png")
    fig_text(a.fig_csv_dir, a.data_dir, a.out_dir / "fig5_text.png")
    fig_vitals(a.fig_csv_dir, a.out_dir / "fig6_vitals_quality.png")
    fig_longtail40(a.fig_csv_dir, a.out_dir / "fig7_longtail40.png")
    print(f"done -> {a.out_dir}")


if __name__ == "__main__":
    main()
