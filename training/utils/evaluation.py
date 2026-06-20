"""Eval-time inference and the report-ablation (leakage probe) driver.

``predict_dataframe`` runs forward-only inference into a tidy predictions frame;
``evaluate_report_ablation`` runs the full pass and, unless skipped, a second
pass with the current study's clinical indication blanked, printing the delta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.dataloader.cxr_lt import CXRLT_2024_TASK2_LABELS

from .config import resolve_path
from .metrics import (
    _no_report_path,
    compute_metrics,
    print_report_ablation_delta,
    print_validation_summary,
)
from .model import move_to_device, precision_context


def _task2_gold_path(pred_df_path: str) -> str:
    """Map a CXR-LT 2024 task1 eval path to its task2 sibling. Covers both the
    prior_aware parquet (``prior_aware_cxrlt2024_task1_test.parquet``) and the
    per-image camchex CSV (``cxrlt2024_task1/test.csv``)."""
    return pred_df_path.replace("task1", "task2")


def maybe_evaluate_cxrlt2024_task2_gold(
    *,
    model,
    classes: list[str],
    device: torch.device,
    args: argparse.Namespace,
    make_loader,
    cfg: dict[str, Any],
    predictions_path: Path,
    metrics_path: Path,
    header: str,
) -> dict[str, float] | None:
    """Score a model on the CXR-LT 2024 task2 (gold) test set, when applicable.

    Fires only when the model's trained ``classes`` are a superset of the 26 task2
    labels (i.e. a 2024 task1 / all model) and the task2 gold file exists next to the
    configured eval file. The model's wide outputs are sliced to the 26 task2 columns
    and scored against the small expert-labeled gold set. Writes
    ``*_task2_gold.{csv,json}`` and returns the metrics (or None if skipped)."""
    if getattr(args, "skip_task2_gold", False):
        return None
    if not set(CXRLT_2024_TASK2_LABELS).issubset(classes):
        return None  # not a task2-superset label space (e.g. CXR-LT 2023)

    data_cfg = cfg.get("data", {}).get("datamodule_cfg", {}) or {}
    base_pred = args.test_df_path or data_cfg.get("pred_df_path")
    if not base_pred:
        return None
    task2_path = _task2_gold_path(str(base_pred))
    if task2_path == str(base_pred):
        return None  # not a task1-named path; nothing to map
    if not Path(resolve_path(task2_path)).exists():
        print(f"[eval] task2 gold set not found at {task2_path}; skipping task2-gold eval.")
        return None

    # Map each task2 label to its column in the wide (task1/all) output.
    output_indices = [classes.index(c) for c in CXRLT_2024_TASK2_LABELS]

    # Rebuild the eval loader pointed at the task2 gold file with the 26 task2 classes
    # (so labels come out 26-dim), then restore. make_loader closes over cfg/args.
    saved_test_df_path = args.test_df_path
    saved_model_classes = cfg["model"]["classes"]
    args.test_df_path = task2_path
    cfg["model"]["classes"] = list(CXRLT_2024_TASK2_LABELS)
    try:
        loader, labels_available = make_loader(False)
        out_df, preds, labels = predict_dataframe(
            model, loader, list(CXRLT_2024_TASK2_LABELS), device, output_indices=output_indices,
        )
    finally:
        args.test_df_path = saved_test_df_path
        cfg["model"]["classes"] = saved_model_classes

    def _gold(p: Path) -> Path:
        return p.with_name(f"{p.stem}_task2_gold{p.suffix}")

    pred_path = _gold(Path(predictions_path))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(pred_path, index=False)
    if not labels_available:
        print(f"[eval] wrote {pred_path} (no task2 label columns; metrics skipped)")
        return None
    metrics = compute_metrics(preds, labels, list(CXRLT_2024_TASK2_LABELS))
    met_path = _gold(Path(metrics_path))
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print_validation_summary(metrics, list(CXRLT_2024_TASK2_LABELS),
                             header=f"{header} | CXR-LT 2024 task2 gold")
    return metrics


def evaluate_report_ablation(
    *,
    model,
    classes: list[str],
    device: torch.device,
    args: argparse.Namespace,
    make_loader,
    predictions_path: str | Path,
    metrics_path: str | Path,
    header: str,
    cfg: dict[str, Any] | None = None,
) -> None:
    """Run the full eval pass, then (unless --skip-report-ablation) a second pass with
    the current study's clinical indication blanked, writing ``*.no_report.{csv,json}``
    and printing the metric delta.

    ``make_loader(drop_report: bool) -> (loader, labels_available)`` builds the eval
    loader for each pass (rebuilt per pass so DataLoader workers see the right frame).

    When ``cfg`` is supplied and the model was trained on a label set that is a
    superset of the CXR-LT 2024 task2 (gold) labels, an extra pass scores the model
    on the task2 gold test set (see :func:`maybe_evaluate_cxrlt2024_task2_gold`).
    """
    def _run(drop_report: bool, pred_path: Path, met_path: Path, tag: str) -> dict[str, float] | None:
        loader, labels_available = make_loader(drop_report)
        out_df, preds, labels = predict_dataframe(model, loader, classes, device)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(pred_path, index=False)
        if not labels_available:
            print(f"[eval] wrote {pred_path} (no label columns; metrics skipped)")
            return None
        metrics = compute_metrics(preds, labels, classes)
        met_path.parent.mkdir(parents=True, exist_ok=True)
        with open(met_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print_validation_summary(metrics, classes, header=f"{header} | {tag}")
        return metrics

    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)
    full = _run(False, predictions_path, metrics_path, "image+report[+vitals]")

    if getattr(args, "skip_report_ablation", False):
        return

    ablate = _run(True, _no_report_path(predictions_path), _no_report_path(metrics_path),
                  "report dropped (image[+vitals])")
    if full is not None and ablate is not None:
        print_report_ablation_delta(full, ablate)

    if cfg is not None:
        maybe_evaluate_cxrlt2024_task2_gold(
            model=model, classes=classes, device=device, args=args,
            make_loader=make_loader, cfg=cfg,
            predictions_path=predictions_path, metrics_path=metrics_path, header=header,
        )


@torch.inference_mode()
def predict_dataframe(model, loader, classes: list[str], device: torch.device, ids: list[Any] | None = None, precision: str | None = None, output_indices: list[int] | None = None) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    model.to(device)
    model.eval()
    # Inference needs no autograd graph and no fp32: building the backward graph
    # for 4 image backbones + CXR-BERT + the transformer is the dominant eval cost,
    # and the forward is Tensor-Core-friendly. Default to bf16 autocast on CUDA
    # (resolve_precision downgrades to fp32 where bf16 has no hardware path).
    if precision is None:
        precision = "bf16-mixed" if device.type == "cuda" else "32-true"
    preds = []
    labels = []
    batch_ids = []
    with torch.inference_mode(), precision_context(device, precision):
        for batch in tqdm(loader, desc="predict", dynamic_ncols=True):
            data, label = batch
            if isinstance(data, dict) and "study_id" in data:
                sid = data["study_id"]
                batch_ids.extend(sid.tolist() if torch.is_tensor(sid) else list(sid))
            elif isinstance(data, (tuple, list)) and data and not torch.is_tensor(data[0]):
                batch_ids.extend(list(data[0]))
            data = move_to_device(data, device)
            out = model(data)
            if isinstance(out, tuple):  # (logits, aux_loss) -- keep only logits for inference
                out = out[0]
            pred = torch.sigmoid(out.float()).cpu()
            preds.append(pred)
            labels.append(label.cpu().float())

    pred_tensor = torch.cat(preds)
    label_tensor = torch.cat(labels)
    # Restrict the model's outputs to a subset of its trained classes (e.g. a CXR-LT
    # 2024 task1 model scored on the 26 task2 labels). ``classes``/``label_tensor``
    # are already in the subset's order; only the wide prediction vector is sliced.
    if output_indices is not None:
        pred_tensor = pred_tensor[:, output_indices]
    out = pd.DataFrame(pred_tensor.numpy(), columns=[f"pred_{c}" for c in classes])
    if batch_ids:
        out.insert(0, "study_id", batch_ids)
    elif ids is not None:
        out.insert(0, "sample_id", ids[: len(out)])
    for idx, name in enumerate(classes):
        out[f"label_{name}"] = label_tensor[:, idx].numpy()
    return out, pred_tensor, label_tensor
