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

from .metrics import (
    _no_report_path,
    compute_metrics,
    print_report_ablation_delta,
    print_validation_summary,
)
from .model import move_to_device, precision_context


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
) -> None:
    """Run the full eval pass, then (unless --skip-report-ablation) a second pass with
    the current study's clinical indication blanked, writing ``*.no_report.{csv,json}``
    and printing the metric delta.

    ``make_loader(drop_report: bool) -> (loader, labels_available)`` builds the eval
    loader for each pass (rebuilt per pass so DataLoader workers see the right frame).
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


@torch.inference_mode()
def predict_dataframe(model, loader, classes: list[str], device: torch.device, ids: list[Any] | None = None, precision: str | None = None) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
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
    out = pd.DataFrame(pred_tensor.numpy(), columns=[f"pred_{c}" for c in classes])
    if batch_ids:
        out.insert(0, "study_id", batch_ids)
    elif ids is not None:
        out.insert(0, "sample_id", ids[: len(out)])
    for idx, name in enumerate(classes):
        out[f"label_{name}"] = label_tensor[:, idx].numpy()
    return out, pred_tensor, label_tensor
