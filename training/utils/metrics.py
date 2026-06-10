"""Validation metrics and their console/CSV-friendly summaries.

``compute_metrics`` produces the per-class + head/medium/tail grouped AP/AUROC
dict consumed everywhere; the ``print_*`` helpers format it (including the
report-ablation delta table used by the eval entry points).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torchmetrics.functional import average_precision, auroc
from tqdm.auto import tqdm

from .constants import HEAD_IDX, MEDIUM_IDX, TAIL_IDX


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> dict[str, float]:
    per_ap: list[float] = []
    per_auc: list[float] = []
    metrics: dict[str, float] = {}
    labels_long = labels.long()
    for idx, name in enumerate(classes):
        ap = float(average_precision(preds[:, idx], labels_long[:, idx], task="binary"))
        au = float(auroc(preds[:, idx], labels_long[:, idx], task="binary"))
        per_ap.append(ap)
        per_auc.append(au)
        metrics[f"val/ap/{name}"] = ap
        metrics[f"val/auroc/{name}"] = au

    mean_ap = sum(per_ap) / len(per_ap)
    mean_au = sum(per_auc) / len(per_auc)
    metrics.update(
        {
            "val_ap": mean_ap,
            "val_auroc": mean_au,
            "val/ap": mean_ap,
            "val/auroc": mean_au,
            "val/ap_head": sum(per_ap[i] for i in HEAD_IDX) / len(HEAD_IDX),
            "val/ap_medium": sum(per_ap[i] for i in MEDIUM_IDX) / len(MEDIUM_IDX),
            "val/ap_tail": sum(per_ap[i] for i in TAIL_IDX) / len(TAIL_IDX),
            "val/auroc_head": sum(per_auc[i] for i in HEAD_IDX) / len(HEAD_IDX),
            "val/auroc_medium": sum(per_auc[i] for i in MEDIUM_IDX) / len(MEDIUM_IDX),
            "val/auroc_tail": sum(per_auc[i] for i in TAIL_IDX) / len(TAIL_IDX),
        }
    )
    return metrics


def _class_group_names(classes: list[str], idxs: list[int]) -> str:
    return ", ".join(classes[i] if i < len(classes) else f"#{i}" for i in idxs)


def print_validation_summary(metrics: dict[str, float], classes: list[str], header: str | None = None) -> None:
    if header:
        tqdm.write(f"\n=== {header} ===")
    name_w = max(len(c) for c in classes)
    tqdm.write(f"{'class':<{name_w}}  {'AP':>8}  {'AUROC':>8}")
    for name in classes:
        ap = metrics.get(f"val/ap/{name}", float("nan"))
        au = metrics.get(f"val/auroc/{name}", float("nan"))
        tqdm.write(f"{name:<{name_w}}  {ap:>8.4f}  {au:>8.4f}")
    tqdm.write("--- summary ---")
    tqdm.write(f"mean AP    : {metrics.get('val_ap', float('nan')):.4f}")
    tqdm.write(f"mean AUROC : {metrics.get('val_auroc', float('nan')):.4f}")
    tqdm.write(
        f"AP    head/medium/tail: "
        f"{metrics.get('val/ap_head', float('nan')):.4f} / "
        f"{metrics.get('val/ap_medium', float('nan')):.4f} / "
        f"{metrics.get('val/ap_tail', float('nan')):.4f}"
    )
    tqdm.write(
        f"AUROC head/medium/tail: "
        f"{metrics.get('val/auroc_head', float('nan')):.4f} / "
        f"{metrics.get('val/auroc_medium', float('nan')):.4f} / "
        f"{metrics.get('val/auroc_tail', float('nan')):.4f}"
    )
    tqdm.write(f"head classes  : {_class_group_names(classes, HEAD_IDX)}")
    tqdm.write(f"medium classes: {_class_group_names(classes, MEDIUM_IDX)}")
    tqdm.write(f"tail classes  : {_class_group_names(classes, TAIL_IDX)}")


def _no_report_path(path: str | Path) -> Path:
    """foo/metrics.json -> foo/metrics.no_report.json (the report-ablation variant)."""
    p = Path(path)
    return p.with_name(f"{p.stem}.no_report{p.suffix}")


def print_report_ablation_delta(full: dict[str, float], ablate: dict[str, float]) -> None:
    """Side-by-side of the full pass vs the report-dropped pass. A positive Δ
    (full − no_report) means the clinical indication was *helping* that metric;
    a large positive Δ is what you'd expect if the report leaks label information."""
    rows = [
        ("mean AP", "val_ap"), ("mean AUROC", "val_auroc"),
        ("AP head", "val/ap_head"), ("AP medium", "val/ap_medium"), ("AP tail", "val/ap_tail"),
        ("AUROC head", "val/auroc_head"), ("AUROC medium", "val/auroc_medium"), ("AUROC tail", "val/auroc_tail"),
    ]
    tqdm.write("\n=== report-ablation delta (full image+report[+vitals]  vs  report dropped) ===")
    tqdm.write(f"{'metric':<14}{'full':>10}{'no_report':>12}{'Δ (full-abl)':>14}")
    for label, key in rows:
        f = full.get(key, float("nan"))
        a = ablate.get(key, float("nan"))
        tqdm.write(f"{label:<14}{f:>10.4f}{a:>12.4f}{f - a:>+14.4f}")
    tqdm.write("Δ > 0 ⇒ the clinical indication helped (possible leakage); Δ ≈ 0 ⇒ report adds little.")
