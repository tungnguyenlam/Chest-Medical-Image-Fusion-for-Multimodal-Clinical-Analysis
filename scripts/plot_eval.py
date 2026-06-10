"""Plot evaluation diagnostics from an eval predictions/metrics dump.

Reads the `predictions.csv` written by `training/utils/evaluation.py`
(columns: optional `study_id`, `pred_<class>` = sigmoid probabilities,
`label_<class>` = ground-truth in {0.0, 0.5 uncertain, 1.0}) and the matching
`metrics.json` (authoritative per-class AP/AUROC), and writes a battery of
classification PNGs to `<predictions_dir>/eval_plots/`.

If the report-ablation variant (`predictions.no_report.csv`) sits next to the
main file it is picked up automatically for the leakage-delta plot.

Labels follow the same convention as `compute_metrics` (torchmetrics `.long()`):
the uncertain label 0.5 truncates to 0, so a class is positive iff its label is
1.0. Pass `--uncertain {neg,pos,ignore}` to override.

ROC/PR curves are computed here with numpy (no sklearn dependency); the per-class
AP/AUROC *bars* prefer the values in `metrics.json` so they match what eval
reported, falling back to the numpy computation when the JSON is absent.

Usage:
    python scripts/plot_eval.py output/camchex/predictions.csv
    python scripts/plot_eval.py output/camchex/predictions.csv --output-dir /tmp/plots
    python scripts/plot_eval.py output/camchex/predictions.csv --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CXR-LT long-tail split (head/medium/tail), mirrored from training/utils/constants.py.
CLASS_GROUPS = {
    "head": [0, 2, 4, 12, 14, 16, 20, 24],
    "medium": [1, 3, 5, 6, 8, 9, 10, 13, 15, 22],
    "tail": [7, 11, 17, 18, 19, 21, 23, 25],
}
GROUP_COLORS = {"head": "tab:green", "medium": "tab:orange", "tail": "tab:red"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, help="Path to predictions.csv from eval.")
    parser.add_argument("--metrics", type=Path, help="Path to metrics.json (default: sibling metrics.json).")
    parser.add_argument("--output-dir", type=Path, help="Write plots here (default: <predictions_dir>/eval_plots).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Operating threshold for confusion-style plots.")
    parser.add_argument(
        "--uncertain",
        choices=["neg", "pos", "ignore"],
        default="neg",
        help="How to treat the uncertain label 0.5 (default neg, matching compute_metrics).",
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


# --------------------------------------------------------------------------- I/O


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=fig.get_dpi(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def _load_predictions(path: Path, uncertain: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (classes, preds[N,C] float, labels[N,C] {0,1} or nan-for-ignored)."""
    df = pd.read_csv(path)
    classes = [c[len("pred_") :] for c in df.columns if c.startswith("pred_")]
    if not classes:
        raise SystemExit(f"no pred_<class> columns found in {path}")
    preds = df[[f"pred_{c}" for c in classes]].to_numpy(dtype=np.float64)
    label_cols = [f"label_{c}" for c in classes]
    if not all(c in df.columns for c in label_cols):
        raise SystemExit(f"{path} has no label_<class> columns; cannot compute metrics/curves")
    raw = df[label_cols].to_numpy(dtype=np.float64)
    labels = np.where(raw >= 1.0, 1.0, 0.0)  # 1.0 -> positive
    uncertain_mask = np.isclose(raw, 0.5)
    if uncertain == "pos":
        labels[uncertain_mask] = 1.0
    elif uncertain == "ignore":
        labels[uncertain_mask] = np.nan
    # "neg": leave as 0.0 (matches torchmetrics .long() truncation)
    return classes, preds, labels


def _group_of(idx: int) -> str:
    for g, idxs in CLASS_GROUPS.items():
        if idx in idxs:
            return g
    return "medium"


# ------------------------------------------------------------- curve primitives


def _clf_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Cumulative TP/FP at every distinct score threshold (sklearn-style)."""
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]
    distinct = np.where(np.diff(y_score))[0]
    idxs = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true)[idxs]
    fps = 1 + idxs - tps
    return fps, tps, y_score[idxs]


def roc_curve(y_true, y_score):
    fps, tps, thr = _clf_curve(y_true, y_score)
    fps = np.r_[0, fps]
    tps = np.r_[0, tps]
    if fps[-1] <= 0 or tps[-1] <= 0:
        return None
    return fps / fps[-1], tps / tps[-1]


def pr_curve(y_true, y_score):
    fps, tps, thr = _clf_curve(y_true, y_score)
    if tps[-1] <= 0:
        return None
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / tps[-1]
    # Prepend the (recall=0, precision=1) anchor for a clean left edge.
    return np.r_[1, precision[::-1]], np.r_[0, recall[::-1]]


def auc(x, y) -> float:
    return float(np.trapz(y, x))


def average_precision(y_true, y_score) -> float:
    fps, tps, _ = _clf_curve(y_true, y_score)
    if tps[-1] <= 0:
        return float("nan")
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / tps[-1]
    recall = np.r_[0, recall]
    precision = np.r_[1, precision]
    return float(np.sum(np.diff(recall) * precision[1:]))


def _valid(y_true, y_score):
    """Drop ignored (nan) labels for a single class column."""
    m = ~np.isnan(y_true)
    return y_true[m], y_score[m]


# ------------------------------------------------------------------ ROC / PR plots


def plot_curve_grid(classes, preds, labels, kind: str, out: Path, dpi: int) -> None:
    n = len(classes)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.6 * nrows), dpi=dpi)
    axes = np.array(axes).reshape(-1)
    for i, (ax, name) in enumerate(zip(axes, classes)):
        yt, ys = _valid(labels[:, i], preds[:, i])
        color = GROUP_COLORS[_group_of(i)]
        if kind == "roc":
            res = roc_curve(yt, ys)
            if res is not None:
                fpr, tpr = res
                ax.plot(fpr, tpr, color=color, linewidth=1.3)
                ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5)
                ax.set_title(f"{name}\nAUROC={auc(fpr, tpr):.3f}", fontsize=7)
        else:
            res = pr_curve(yt, ys)
            prev = np.nanmean(yt) if yt.size else 0.0
            if res is not None:
                prec, rec = res
                ax.plot(rec, prec, color=color, linewidth=1.3)
                ax.axhline(prev, color="k", linestyle="--", linewidth=0.6, alpha=0.5)
                ax.set_title(f"{name}\nAP={average_precision(yt, ys):.3f}", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.set_visible(False)
    label = "ROC curves (per class)" if kind == "roc" else "Precision-Recall curves (per class)"
    fig.suptitle(label, y=1.002, fontsize=12)
    _save(fig, out / (f"{kind}_per_class.png"))


def _micro_macro(classes, preds, labels, kind: str):
    """Pooled (micro) and averaged (macro) curve over all classes."""
    yt_all, ys_all, macro = [], [], []
    grid = np.linspace(0, 1, 200)
    for i in range(len(classes)):
        yt, ys = _valid(labels[:, i], preds[:, i])
        yt_all.append(yt)
        ys_all.append(ys)
        res = roc_curve(yt, ys) if kind == "roc" else pr_curve(yt, ys)
        if res is None:
            continue
        x, y = res
        if kind == "roc":
            macro.append(np.interp(grid, x, y))
        else:
            # PR comes back recall-descending; interp wants ascending x (recall).
            macro.append(np.interp(grid, y, x))
    yt_all = np.concatenate(yt_all)
    ys_all = np.concatenate(ys_all)
    micro = roc_curve(yt_all, ys_all) if kind == "roc" else pr_curve(yt_all, ys_all)
    macro_y = np.mean(macro, axis=0) if macro else None
    return micro, (grid, macro_y)


def plot_curve_grouped(classes, preds, labels, kind: str, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=dpi)
    for i, name in enumerate(classes):
        yt, ys = _valid(labels[:, i], preds[:, i])
        res = roc_curve(yt, ys) if kind == "roc" else pr_curve(yt, ys)
        if res is None:
            continue
        x, y = res
        if kind == "roc":
            ax.plot(x, y, color=GROUP_COLORS[_group_of(i)], linewidth=0.9, alpha=0.55)
        else:
            ax.plot(y, x, color=GROUP_COLORS[_group_of(i)], linewidth=0.9, alpha=0.55)
    micro, (grid, macro_y) = _micro_macro(classes, preds, labels, kind)
    if micro is not None:
        x, y = micro
        if kind == "roc":
            ax.plot(x, y, color="black", linewidth=2.2, label=f"micro (AUC={auc(x, y):.3f})")
        else:
            ax.plot(y, x, color="black", linewidth=2.2, label=f"micro (AUC={auc(y[::-1], x[::-1]):.3f})")
    if macro_y is not None:
        ax.plot(grid, macro_y, color="navy", linewidth=2.0, linestyle="--", label=f"macro (AUC={auc(grid, macro_y):.3f})")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC — all 26 classes (colored by head/medium/tail)")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall — all 26 classes (colored by head/medium/tail)")
    # Group legend handles.
    handles = [plt.Line2D([], [], color=c, label=g) for g, c in GROUP_COLORS.items()]
    leg1 = ax.legend(handles=handles, title="class group", loc="lower left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    _save(fig, out / f"{kind}_grouped.png")


# ---------------------------------------------------------------- bar / scatter


def _per_class_metric(classes, preds, labels, metrics, key_prefix: str, compute_fn):
    """Prefer metrics.json values; fall back to numpy computation."""
    vals = []
    for i, name in enumerate(classes):
        if metrics is not None and f"{key_prefix}{name}" in metrics:
            vals.append(float(metrics[f"{key_prefix}{name}"]))
        else:
            yt, ys = _valid(labels[:, i], preds[:, i])
            res = compute_fn(yt, ys)
            vals.append(res if res is not None else float("nan"))
    return np.array(vals)


def _auroc_np(yt, ys):
    res = roc_curve(yt, ys)
    return auc(*res) if res is not None else None


def plot_metric_bars(classes, vals, metric_name: str, out: Path, dpi: int) -> None:
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    colors = [GROUP_COLORS[_group_of(i)] for i in order]
    ax.barh(range(len(classes)), vals[order], color=colors)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([classes[i] for i in order], fontsize=8)
    mean = np.nanmean(vals)
    ax.axvline(mean, color="black", linestyle="--", linewidth=1, label=f"mean = {mean:.3f}")
    for y, v in enumerate(vals[order]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=6)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(metric_name)
    ax.set_title(f"Per-class {metric_name} (sorted; colored by head/medium/tail)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles + [plt.Line2D([], [], color="black", linestyle="--", label=f"mean={mean:.3f}")], fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, out / f"{metric_name.lower()}_per_class_bar.png")


def plot_group_summary(metrics, ap_vals, auroc_vals, out: Path, dpi: int) -> None:
    groups = ["head", "medium", "tail"]
    ap = [np.nanmean([ap_vals[i] for i in CLASS_GROUPS[g]]) for g in groups]
    au = [np.nanmean([auroc_vals[i] for i in CLASS_GROUPS[g]]) for g in groups]
    if metrics is not None:
        ap = [metrics.get(f"val/ap_{g}", ap[k]) for k, g in enumerate(groups)]
        au = [metrics.get(f"val/auroc_{g}", au[k]) for k, g in enumerate(groups)]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
    b1 = ax.bar(x - w / 2, ap, w, label="AP", color="tab:blue")
    b2 = ax.bar(x + w / 2, au, w, label="AUROC", color="tab:cyan")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n({len(CLASS_GROUPS[g])} classes)" for g in groups])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Head / medium / tail summary (long-tail breakdown)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out / "group_summary_bar.png")


def plot_metric_vs_prevalence(classes, labels, ap_vals, auroc_vals, out: Path, dpi: int) -> None:
    support = np.array([int(np.nansum(labels[:, i] == 1.0)) for i in range(len(classes))])
    prevalence = support / np.maximum(np.sum(~np.isnan(labels), axis=0), 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=dpi)
    for ax, vals, title in ((ax1, ap_vals, "AP"), (ax2, auroc_vals, "AUROC")):
        for i, name in enumerate(classes):
            ax.scatter(prevalence[i], vals[i], color=GROUP_COLORS[_group_of(i)], s=40, zorder=3)
            ax.annotate(name, (prevalence[i], vals[i]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("class prevalence (positive fraction, log scale)")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title} vs prevalence")
        ax.grid(alpha=0.3, which="both")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax1.legend(handles=handles, title="class group", fontsize=8)
    fig.suptitle("Does the model just track class frequency? (long-tail diagnostic)", y=1.01)
    _save(fig, out / "metric_vs_prevalence.png")


def plot_ap_vs_auroc(classes, ap_vals, auroc_vals, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=dpi)
    for i, name in enumerate(classes):
        ax.scatter(auroc_vals[i], ap_vals[i], color=GROUP_COLORS[_group_of(i)], s=45, zorder=3)
        ax.annotate(name, (auroc_vals[i], ap_vals[i]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("AUROC")
    ax.set_ylabel("AP")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("AP vs AUROC per class\n(AUROC stays high under imbalance; AP exposes it)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, title="class group", fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, out / "ap_vs_auroc_scatter.png")


# ------------------------------------------------------- threshold / confusion


def _threshold_sweep(yt, ys, grid):
    """F1 / precision / recall over a threshold grid for one class."""
    f1, prec, rec = [], [], []
    p = np.sum(yt == 1.0)
    for t in grid:
        pred = ys >= t
        tp = np.sum(pred & (yt == 1.0))
        fp = np.sum(pred & (yt == 0.0))
        pr = tp / max(tp + fp, 1)
        re = tp / max(p, 1)
        prec.append(pr)
        rec.append(re)
        f1.append(2 * pr * re / max(pr + re, 1e-9))
    return np.array(f1), np.array(prec), np.array(rec)


def plot_best_thresholds(classes, preds, labels, out: Path, dpi: int) -> None:
    grid = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = [], []
    for i in range(len(classes)):
        yt, ys = _valid(labels[:, i], preds[:, i])
        f1, _, _ = _threshold_sweep(yt, ys, grid)
        k = int(np.argmax(f1))
        best_t.append(grid[k])
        best_f1.append(f1[k])
    order = np.argsort(best_f1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8), dpi=dpi)
    colors = [GROUP_COLORS[_group_of(i)] for i in order]
    ax1.barh(range(len(classes)), [best_f1[i] for i in order], color=colors)
    ax1.set_yticks(range(len(classes)))
    ax1.set_yticklabels([classes[i] for i in order], fontsize=8)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("best F1")
    ax1.set_title("Best achievable F1 per class")
    ax1.grid(alpha=0.3, axis="x")
    ax2.barh(range(len(classes)), [best_t[i] for i in order], color=colors)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels([classes[i] for i in order], fontsize=8)
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1, label="default 0.5")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("F1-optimal threshold")
    ax2.set_title("Optimal threshold per class\n(far from 0.5 ⇒ probabilities need calibration)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="x")
    _save(fig, out / "best_thresholds.png")


def plot_confusion_summary(classes, preds, labels, threshold: float, out: Path, dpi: int) -> None:
    """Per-class recall / specificity / precision / F1 heatmap at a fixed threshold."""
    rows = []
    for i in range(len(classes)):
        yt, ys = _valid(labels[:, i], preds[:, i])
        pred = ys >= threshold
        tp = np.sum(pred & (yt == 1.0))
        fp = np.sum(pred & (yt == 0.0))
        fn = np.sum(~pred & (yt == 1.0))
        tn = np.sum(~pred & (yt == 0.0))
        recall = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-9)
        rows.append([recall, spec, prec, f1])
    data = np.array(rows)
    cols = ["Recall\n(sens.)", "Specificity", "Precision", "F1"]
    fig, ax = plt.subplots(figsize=(6, 9), dpi=dpi)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(f"Per-class operating point @ threshold={threshold:g}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out / "confusion_summary.png")


# ----------------------------------------------------- distributions / calibration


def plot_score_distributions(classes, preds, labels, out: Path, dpi: int) -> None:
    n = len(classes)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.4 * nrows), dpi=dpi)
    axes = np.array(axes).reshape(-1)
    bins = np.linspace(0, 1, 31)
    for i, (ax, name) in enumerate(zip(axes, classes)):
        yt, ys = _valid(labels[:, i], preds[:, i])
        ax.hist(ys[yt == 0.0], bins=bins, color="tab:blue", alpha=0.55, density=True, label="neg")
        ax.hist(ys[yt == 1.0], bins=bins, color="tab:red", alpha=0.55, density=True, label="pos")
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Predicted-probability distributions: positives vs negatives (per class)", y=1.002, fontsize=12)
    _save(fig, out / "score_distributions.png")


def plot_calibration(classes, preds, labels, out: Path, dpi: int) -> None:
    """Pooled and per-group reliability diagrams."""
    n_bins = 12
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    def reliability(idxs):
        yt = np.concatenate([labels[:, i] for i in idxs])
        ys = np.concatenate([preds[:, i] for i in idxs])
        m = ~np.isnan(yt)
        yt, ys = yt[m], ys[m]
        frac, conf = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (ys >= lo) & (ys < hi if hi < 1 else ys <= hi)
            if sel.sum() == 0:
                frac.append(np.nan)
                conf.append(np.nan)
            else:
                frac.append(np.mean(yt[sel] == 1.0))
                conf.append(np.mean(ys[sel]))
        return np.array(conf), np.array(frac)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    conf, frac = reliability(list(range(len(classes))))
    ax.plot(conf, frac, "o-", color="black", linewidth=2, label="all classes (pooled)")
    for g, idxs in CLASS_GROUPS.items():
        c, f = reliability(idxs)
        ax.plot(c, f, "s--", color=GROUP_COLORS[g], alpha=0.8, label=g)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram (calibration)\nbelow diagonal ⇒ over-confident")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out / "calibration.png")


def plot_prevalence(classes, labels, out: Path, dpi: int) -> None:
    support = np.array([int(np.nansum(labels[:, i] == 1.0)) for i in range(len(classes))])
    order = np.argsort(support)[::-1]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    colors = [GROUP_COLORS[_group_of(i)] for i in order]
    ax.barh(range(len(classes)), support[order], color=colors)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([classes[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    for y, v in enumerate(support[order]):
        ax.text(v, y, f" {v}", va="center", fontsize=7)
    ax.set_xlabel("# positive samples in eval split")
    ax.set_title("Class support / prevalence (the long tail)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, out / "class_prevalence.png")


# ------------------------------------------------------------ report-ablation delta


def plot_report_ablation(metrics, no_report_metrics, classes, out: Path, dpi: int) -> None:
    if metrics is None or no_report_metrics is None:
        return
    deltas, names = [], []
    for name in classes:
        k = f"val/auroc/{name}"
        if k in metrics and k in no_report_metrics:
            deltas.append(metrics[k] - no_report_metrics[k])
            names.append(name)
    if not deltas:
        return
    deltas = np.array(deltas)
    order = np.argsort(deltas)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    colors = ["tab:red" if d > 0 else "tab:gray" for d in deltas[order]]
    ax.barh(range(len(names)), deltas[order], color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AUROC(full) − AUROC(report dropped)")
    ax.set_title("Report-ablation delta per class\nΔ > 0 ⇒ clinical indication helped (possible leakage)")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, out / "report_ablation_delta.png")


# ----------------------------------------------------------------------- driver


def _maybe_load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    pred_path = args.predictions
    if not pred_path.exists():
        raise SystemExit(f"predictions not found: {pred_path}")
    out = args.output_dir if args.output_dir is not None else pred_path.parent / "eval_plots"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[plot-eval] {pred_path} -> {out}")

    classes, preds, labels = _load_predictions(pred_path, args.uncertain)
    metrics_path = args.metrics if args.metrics is not None else pred_path.with_name("metrics.json")
    metrics = _maybe_load_json(metrics_path)
    print(f"  {preds.shape[0]} samples x {len(classes)} classes; metrics.json: {'yes' if metrics else 'no'}")

    ap_vals = _per_class_metric(classes, preds, labels, metrics, "val/ap/", average_precision)
    auroc_vals = _per_class_metric(classes, preds, labels, metrics, "val/auroc/", _auroc_np)

    plot_curve_grid(classes, preds, labels, "roc", out, args.dpi)
    plot_curve_grid(classes, preds, labels, "pr", out, args.dpi)
    plot_curve_grouped(classes, preds, labels, "roc", out, args.dpi)
    plot_curve_grouped(classes, preds, labels, "pr", out, args.dpi)
    plot_metric_bars(classes, auroc_vals, "AUROC", out, args.dpi)
    plot_metric_bars(classes, ap_vals, "AP", out, args.dpi)
    plot_group_summary(metrics, ap_vals, auroc_vals, out, args.dpi)
    plot_metric_vs_prevalence(classes, labels, ap_vals, auroc_vals, out, args.dpi)
    plot_ap_vs_auroc(classes, ap_vals, auroc_vals, out, args.dpi)
    plot_best_thresholds(classes, preds, labels, out, args.dpi)
    plot_confusion_summary(classes, preds, labels, args.threshold, out, args.dpi)
    plot_score_distributions(classes, preds, labels, out, args.dpi)
    plot_calibration(classes, preds, labels, out, args.dpi)
    plot_prevalence(classes, labels, out, args.dpi)

    # Report-ablation (leakage) delta if the no_report dump sits alongside.
    nr_pred = pred_path.with_name(f"{pred_path.stem}.no_report{pred_path.suffix}")
    nr_metrics = _maybe_load_json(metrics_path.with_name(f"{metrics_path.stem}.no_report{metrics_path.suffix}"))
    if nr_metrics is not None:
        plot_report_ablation(metrics, nr_metrics, classes, out, args.dpi)
    elif nr_pred.exists():
        print(f"  note: {nr_pred.name} exists but its metrics.json is missing; skipping ablation delta")

    print(f"[plot-eval] done -> {out}")


if __name__ == "__main__":
    main()
