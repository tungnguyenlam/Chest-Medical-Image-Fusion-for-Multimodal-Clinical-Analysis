#!/usr/bin/env python3
"""Polished thesis EDA figures for the CaMCheX / CXR-LT dataset.

Regenerates three publication-quality figures from the prepared dataset
(``data/data-camchex/03_mimic_{train,development,test}.csv``), replacing the
default-styled notebook plots:

    fig1_label_longtail.png   — ranked per-class prevalence + positive counts,
                                colour-coded by long-tail frequency tier.
    fig2_label_structure.png  — label correlation (Pearson) and conditional
                                co-occurrence P(B|A) heatmaps.
    fig3_prior_study.png      — prior-study coverage, follow-up time gaps, and
                                per-class prior-label informativeness.

All figures share the look defined in ``eda_style.py`` so notebooks can adopt
the same aesthetic with ``from eda_style import apply_style; apply_style()``.

Usage
-----
    python data/00-examine-data/make_eda_figures.py
    python data/00-examine-data/make_eda_figures.py --data-dir <dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from eda_style import (CMAP_DIV, CMAP_SEQ, MUTED, TIER_COLORS, apply_style,
                       despine, save, tier_of)

CLASSES = [
    "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation",
    "Edema", "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "No Finding",
    "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta",
]

SHORT = {
    "Calcification of the Aorta": "Calcification of Aorta",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomed.",
    "Subcutaneous Emphysema": "Subcut. Emphysema",
}


def load(data_dir: Path) -> pd.DataFrame:
    frames = []
    for split, name in (("train", "train"), ("development", "dev"), ("test", "test")):
        p = data_dir / f"03_mimic_{split}.csv"
        df = pd.read_csv(p, low_memory=False)
        df["split"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in CLASSES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Figure 1 — long-tail prevalence
# ---------------------------------------------------------------------------
def fig_longtail(df: pd.DataFrame, out: Path) -> None:
    train = df[df["split"] == "train"]
    n = len(train)
    pos = train[CLASSES].sum().sort_values(ascending=True)  # asc -> barh worst at bottom
    prev = pos / n
    names = [SHORT.get(c, c) for c in pos.index]
    colors = [TIER_COLORS[tier_of(c)] for c in pos.index]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15, 9), gridspec_kw=dict(width_ratios=[1.0, 0.62], wspace=0.06))
    y = np.arange(len(names))

    # Panel A: prevalence (%)
    axL.barh(y, prev.values * 100, color=colors, edgecolor="white",
             linewidth=0.6, height=0.78, zorder=3)
    for yi, (p, c) in enumerate(zip(prev.values, pos.values)):
        axL.text(p * 100 + max(prev.values) * 100 * 0.012, yi,
                 f"{p*100:.1f}%", va="center", ha="left", fontsize=8.6, color="#333")
    axL.set_yticks(y)
    axL.set_yticklabels(names, fontsize=9.4)
    axL.set_xlim(0, max(prev.values) * 100 * 1.16)
    axL.set_xlabel("Prevalence in train (%)")
    axL.set_title("A   Per-class prevalence")
    axL.grid(axis="x")
    despine(axL)

    # Panel B: positive counts on log scale (the long tail spans 3 orders of mag)
    axR.barh(y, pos.values, color=colors, edgecolor="white", linewidth=0.6,
             height=0.78, zorder=3)
    axR.set_xscale("log")
    axR.set_yticks(y)
    axR.set_yticklabels([])
    axR.set_xlabel("Positive images (log scale)")
    axR.set_title("B   Positive count")
    axR.grid(axis="x")
    despine(axR)
    for yi, c in zip(y, pos.values):
        axR.text(c * 1.15, yi, f"{int(c):,}", va="center", ha="left",
                 fontsize=8.0, color="#333")

    tiers = [("Head", len(["x" for c in CLASSES if tier_of(c) == "Head"])),
             ("Medium", sum(tier_of(c) == "Medium" for c in CLASSES)),
             ("Tail", sum(tier_of(c) == "Tail" for c in CLASSES))]
    legend = [Patch(facecolor=TIER_COLORS[t], label=f"{t} ({k})") for t, k in tiers]
    axL.legend(handles=legend, title="CXR-LT frequency tier", loc="lower right")

    ratio = pos.max() / max(pos.min(), 1)
    fig.suptitle("Long-tailed label distribution — CXR-LT 2023 (26 findings)",
                 x=0.075, ha="left", fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.075, 0.93,
             f"{n:,} train images  ·  most/least frequent ratio ≈ {ratio:,.0f}×  ·  "
             "tier colours reused across all figures",
             ha="left", fontsize=10.5, color=MUTED)
    save(fig, out)
    plt.close(fig)
    print(f"  fig1: {out.name}  (imbalance ratio {ratio:,.0f}x)")


# ---------------------------------------------------------------------------
# Figure 2 — correlation + conditional co-occurrence
# ---------------------------------------------------------------------------
def _tier_ticklabels(ax, axis="both"):
    """Colour tick labels by tier."""
    setter_x = ax.get_xticklabels() if axis in ("x", "both") else []
    setter_y = ax.get_yticklabels() if axis in ("y", "both") else []
    for t in list(setter_x) + list(setter_y):
        t.set_color(TIER_COLORS[tier_of(t.get_text())])


def fig_structure(df: pd.DataFrame, out: Path) -> None:
    import seaborn as sns

    order = df[CLASSES].sum().sort_values(ascending=False).index.tolist()
    X = df[order]

    corr = X.corr()
    co = X.T.dot(X)
    cond = co.div(np.diag(co), axis=0)  # P(col=1 | row=1)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(19, 9))

    sns.heatmap(corr, ax=axA, cmap=CMAP_DIV, center=0, vmin=-0.6, vmax=0.6,
                square=True, linewidths=0.3, linecolor="white",
                cbar_kws=dict(shrink=0.6, label="Pearson r", pad=0.02))
    axA.set_title("A   Label correlation (Pearson)")

    sns.heatmap(cond, ax=axB, cmap=CMAP_SEQ, vmin=0, vmax=1, square=True,
                linewidths=0.3, linecolor="white",
                cbar_kws=dict(shrink=0.6, label="P(col | row)", pad=0.02))
    axB.set_title("B   Conditional co-occurrence  P(B | A)")

    for ax in (axA, axB):
        ax.set_xticklabels(order, rotation=90, fontsize=7.3)
        ax.set_yticklabels(order, rotation=0, fontsize=7.3)
        _tier_ticklabels(ax)
        ax.tick_params(length=0)

    legend = [Patch(facecolor=TIER_COLORS[t], label=t) for t in ("Head", "Medium", "Tail")]
    axB.legend(handles=legend, title="tier (tick colour)", loc="upper left",
               bbox_to_anchor=(1.18, 1.0), fontsize=9)

    fig.suptitle("Label co-occurrence structure (rows/cols ordered by prevalence)",
                 x=0.09, ha="left", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out)
    plt.close(fig)
    print(f"  fig2: {out.name}")


# ---------------------------------------------------------------------------
# Figure 3 — prior-study viability
# ---------------------------------------------------------------------------
def fig_prior(df: pd.DataFrame, out: Path) -> None:
    d = df.copy()
    d["has_prior"] = d["PreviousStudy"].notna()
    d["StudyDateTime"] = pd.to_datetime(d["StudyDateTime"], errors="coerce")

    # (a) coverage by split (study-level)
    study = d.drop_duplicates("study_id")
    cov = study.groupby("split")["has_prior"].mean().reindex(["train", "dev", "test"])

    # (b) follow-up time gaps
    date_lookup = study.set_index("study_id")["StudyDateTime"].to_dict()

    def prev_date(pid):
        try:
            return date_lookup.get(int(pid))
        except (ValueError, TypeError):
            return None

    sp = study.dropna(subset=["PreviousStudy"]).copy()
    sp["prev_date"] = sp["PreviousStudy"].map(prev_date)
    sp["days"] = (sp["StudyDateTime"] - sp["prev_date"]).dt.total_seconds() / 86400
    gaps = sp["days"].dropna()
    gaps = gaps[gaps >= 0]
    buckets = pd.cut(gaps, bins=[-0.5, 1, 7, 30, 180, 365, 365 * 3, np.inf],
                     labels=["≤1d", "2-7d", "8-30d", "1-6mo", "6-12mo", "1-3y", ">3y"])
    bucket_counts = buckets.value_counts(sort=False)

    # (c) prior-label informativeness (study-level label vectors)
    lab = study.set_index("study_id")[CLASSES].astype(int)
    sp2 = study.dropna(subset=["PreviousStudy"]).copy()
    sp2["prev_id"] = sp2["PreviousStudy"].map(lambda x: int(x) if pd.notna(x) else -1)
    sp2 = sp2[sp2["prev_id"].isin(lab.index)]
    cur_b = lab.loc[sp2["study_id"].values].to_numpy()
    prv_b = lab.loc[sp2["prev_id"].values].to_numpy()
    eps = 1e-9
    p1 = (cur_b * prv_b).sum(0) / (prv_b.sum(0) + eps)
    p0 = (cur_b * (1 - prv_b)).sum(0) / ((1 - prv_b).sum(0) + eps)
    info = (pd.DataFrame({"class": CLASSES, "p1": p1, "p0": p0})
            .assign(lift=lambda t: t.p1 - t.p0)
            .sort_values("lift", ascending=True))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.45], height_ratios=[1, 1],
                          wspace=0.28, hspace=0.42, left=0.07, right=0.965,
                          top=0.86, bottom=0.09)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[:, 1])

    # (a)
    axa.bar(range(len(cov)), cov.values * 100, color=TIER_COLORS["Head"],
            width=0.62, zorder=3)
    axa.set_xticks(range(len(cov)))
    axa.set_xticklabels([s for s in cov.index])
    axa.set_ylim(0, 100)
    axa.set_ylabel("Studies with a prior (%)")
    axa.set_title("A   Prior-study coverage by split")
    for i, v in enumerate(cov.values):
        axa.text(i, v * 100 + 1.5, f"{v*100:.0f}%", ha="center", fontsize=9.5,
                 fontweight="bold")
    axa.grid(axis="y")
    despine(axa)

    # (b)
    axb.bar(range(len(bucket_counts)), bucket_counts.values,
            color=TIER_COLORS["Medium"], width=0.72, zorder=3)
    axb.set_xticks(range(len(bucket_counts)))
    axb.set_xticklabels(bucket_counts.index, rotation=0, fontsize=9)
    axb.set_ylabel("# follow-up studies")
    axb.set_title("B   Time gap to prior study")
    axb.grid(axis="y")
    despine(axb)
    peak = bucket_counts.idxmax()
    axb.text(0.98, 0.92, f"median gap {gaps.median():.0f} d",
             transform=axb.transAxes, ha="right", fontsize=9, color=MUTED)

    # (c) dumbbell: P(cur|prv=0) -> P(cur|prv=1)
    y = np.arange(len(info))
    axc.hlines(y, info["p0"], info["p1"], color="#C9C9C9", lw=2.4, zorder=1)
    axc.scatter(info["p0"], y, s=42, color="#BBBBBB", edgecolor="white",
                zorder=3, label="P(cur=1 | prior=0)  base")
    axc.scatter(info["p1"], y, s=58, color=[TIER_COLORS[tier_of(c)] for c in info["class"]],
                edgecolor="white", zorder=4, label="P(cur=1 | prior=1)")
    axc.set_yticks(y)
    axc.set_yticklabels([SHORT.get(c, c) for c in info["class"]], fontsize=8.6)
    for t in axc.get_yticklabels():
        t.set_color(TIER_COLORS[tier_of(t.get_text())])
    axc.set_xlim(-0.02, 1.02)
    axc.set_xlabel("P(finding present now)")
    axc.set_title("C   Prior-label informativeness  (gap = predictive lift)")
    axc.grid(axis="x")
    despine(axc)
    axc.legend(loc="lower right", fontsize=9)

    overall = study["has_prior"].mean()
    fig.suptitle("Prior-study viability for temporal / prior-aware modelling",
                 x=0.07, ha="left", fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.07, 0.905,
             f"{study['has_prior'].sum():,} of {len(study):,} studies have a prior "
             f"({overall*100:.0f}%)  ·  dumbbell length = how much the prior label "
             "shifts today's probability",
             ha="left", fontsize=10.5, color=MUTED)
    save(fig, out)
    plt.close(fig)
    print(f"  fig3: {out.name}  (prior coverage {overall*100:.0f}%)")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=repo / "data" / "data-camchex")
    ap.add_argument("--out-dir", type=Path, default=repo / "report" / "img" / "eda")
    a = ap.parse_args()

    apply_style()
    print(f"loading from {a.data_dir} ...")
    df = load(a.data_dir)
    print(f"  {len(df):,} image rows, {df['study_id'].nunique():,} studies")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig_longtail(df, a.out_dir / "fig1_label_longtail.png")
    fig_structure(df, a.out_dir / "fig2_label_structure.png")
    fig_prior(df, a.out_dir / "fig3_prior_study.png")
    print(f"done -> {a.out_dir}")


if __name__ == "__main__":
    main()
