"""Training entry for CaMCheX.

Run from project root:
    python -m src.modules.models.camchex.train_camchex \
        --data-root data --subset-name subset \
        --backbone-name convnext_small.fb_in22k_ft_in1k_384 \
        --batch-size 4 --max-epochs 30 --lr 1e-4

Outputs land in ``output/camchex/runs/<run_id>-<run_name>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.modules.callbacks.ema import EMACallback
from src.modules.callbacks.prediction_writer import PredictionWriter
from src.modules.callbacks.run_logger import RunLoggerCallback
from src.modules.dataloaders.mimic_multiview_datamodule import MimicMultiViewDataModule
from src.modules.models.camchex.lightning_module import CaMCheX


MODEL_NAME = "camchex"

CXR_LT_CLASSES: List[str] = [
    "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation",
    "Edema", "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "No Finding",
    "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--data-root", default="data")
    g.add_argument("--subset-name", default="subset")
    g.add_argument("--labels-dirname", default="labels")
    g.add_argument("--image-size", type=int, default=384)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--max-views", type=int, default=4)
    g.add_argument("--clinical-max-length", type=int, default=384)
    g.add_argument("--obs-max-length", type=int, default=128)

    g = p.add_argument_group("model")
    g.add_argument("--backbone-name", default="convnext_small.fb_in22k_ft_in1k_384")
    g.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--frontal-pretrained-path", default=None)
    g.add_argument("--lateral-pretrained-path", default=None)
    g.add_argument("--text-model", default="dmis-lab/biobert-v1.1")
    g.add_argument("--feature-dim", type=int, default=768)
    g.add_argument("--fusion-num-layers", type=int, default=2)
    g.add_argument("--fusion-nhead", type=int, default=8)
    g.add_argument("--fusion-dropout", type=float, default=0.1)
    g.add_argument("--decoder-embedding", type=int, default=768)
    g.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False,
                   help="Trade compute for memory: re-run forward in backward on backbone & text encoder.")

    g = p.add_argument_group("loss")
    g.add_argument("--gamma-neg", type=float, default=4.0)
    g.add_argument("--gamma-pos", type=float, default=1.0)
    g.add_argument("--clip", type=float, default=0.05)

    g = p.add_argument_group("optim")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--accumulate-grad-batches", type=int, default=16)

    g = p.add_argument_group("trainer")
    g.add_argument("--max-epochs", type=int, default=30)
    g.add_argument("--devices", default="auto")
    g.add_argument("--accelerator", default="auto")
    g.add_argument("--precision", default="16-mixed")
    g.add_argument("--val-check-interval", type=float, default=0.25)
    g.add_argument("--gradient-clip-val", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("run")
    g.add_argument("--run-name", default="baseline")
    g.add_argument("--run-id", default=None,
                   help="Override auto timestamp (e.g. for resuming).")
    g.add_argument("--output-root", default="output")
    g.add_argument("--log-every-n-steps", type=int, default=50)

    g = p.add_argument_group("callbacks")
    g.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--prediction-writer", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--run-logger", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--save-top-k", type=int, default=3)

    return p.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def make_run_dir(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(args.output_root) / MODEL_NAME / "runs" / f"{run_id}-{_slugify(args.run_name)}"
    path = base
    i = 2
    while path.exists():
        path = Path(f"{base}-{i}")
        i += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def compute_class_counts(train_csv: Path, classes: List[str]) -> tuple[List[float], int]:
    """Sum each class column in the training CSV for ASL re-weighting."""
    df = pd.read_csv(train_csv, usecols=lambda c: c in classes, low_memory=False)
    missing = [c for c in classes if c not in df.columns]
    if missing:
        sys.exit(f"Class columns missing from {train_csv}: {missing}")
    counts = df[classes].fillna(0).sum().tolist()
    total = int(df[classes].fillna(0).any(axis=1).sum())
    return [float(c) for c in counts], max(1, total)


def build_datamodule(args: argparse.Namespace, labels_dir: Path) -> MimicMultiViewDataModule:
    return MimicMultiViewDataModule(
        train_csv=str(labels_dir / "train.csv"),
        val_csv=str(labels_dir / "val.csv"),
        test_csv=str(labels_dir / "test.csv"),
        image_root=str(Path(args.data_root) / args.subset_name / "MIMIC-CXR-JPG" / "files"),
        classes=CXR_LT_CLASSES,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_model=args.text_model,
        max_views=args.max_views,
        clinical_max_length=args.clinical_max_length,
        obs_max_length=args.obs_max_length,
    )


def build_model(args: argparse.Namespace, class_counts: List[float], total_instances: int) -> CaMCheX:
    timm_init_args = {
        "model_name": args.backbone_name,
        "pretrained": args.pretrained,
        "num_classes": 0,
        "features_only": False,
    }
    loss_init_args = {
        "class_instance_nums": class_counts,
        "total_instance_num": total_instances,
        "gamma_neg": args.gamma_neg,
        "gamma_pos": args.gamma_pos,
        "clip": args.clip,
    }
    return CaMCheX(
        lr=args.lr,
        classes=CXR_LT_CLASSES,
        loss_init_args=loss_init_args,
        timm_init_args=timm_init_args,
        text_model=args.text_model,
        frontal_pretrained_path=args.frontal_pretrained_path,
        lateral_pretrained_path=args.lateral_pretrained_path,
        feature_dim=args.feature_dim,
        num_views=args.max_views,
        fusion_num_layers=args.fusion_num_layers,
        fusion_nhead=args.fusion_nhead,
        fusion_dropout=args.fusion_dropout,
        decoder_embedding=args.decoder_embedding,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def build_callbacks(args: argparse.Namespace, run_dir: Path, test_csv: Path) -> list:
    callbacks: list = [
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="epoch={epoch}-val_ap={val/ap:.4f}",
            monitor="val/ap",
            mode="max",
            save_top_k=args.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    if args.ema:
        callbacks.append(EMACallback(decay=0.9999))
    if args.prediction_writer:
        callbacks.append(PredictionWriter(
            output_path=str(run_dir / "predictions" / "predictions.csv"),
            pred_df_path=str(test_csv),
        ))
    if args.run_logger:
        callbacks.append(RunLoggerCallback(
            run_dir=str(run_dir),
            grad_norm_every_n_steps=args.log_every_n_steps,
            log_module_grad_norms=True,
            save_git_diff=True,
            save_pip_freeze=True,
        ))
    return callbacks


def build_trainer(args: argparse.Namespace, run_dir: Path, callbacks: list) -> pl.Trainer:
    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    return pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(run_dir), name="logs", version=""),
    )


def main() -> int:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    if not Path("data").is_dir() or not Path("src").is_dir():
        sys.exit("Run from project root.")

    labels_dir = Path(args.data_root) / args.subset_name / args.labels_dirname
    train_csv = labels_dir / "train.csv"
    test_csv = labels_dir / "test.csv"
    if not train_csv.exists():
        sys.exit(f"Missing {train_csv}. Run scripts/prepare_subset_labels.py first.")

    run_dir = make_run_dir(args)
    print(f"Run directory: {run_dir}")

    config_path = run_dir / "config.resolved.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    class_counts, total_instances = compute_class_counts(train_csv, CXR_LT_CLASSES)
    print(f"Class instance totals: {total_instances} studies-with-positive across {len(CXR_LT_CLASSES)} classes")

    dm = build_datamodule(args, labels_dir)
    model = build_model(args, class_counts, total_instances)
    callbacks = build_callbacks(args, run_dir, test_csv)
    trainer = build_trainer(args, run_dir, callbacks)

    trainer.fit(model, datamodule=dm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
