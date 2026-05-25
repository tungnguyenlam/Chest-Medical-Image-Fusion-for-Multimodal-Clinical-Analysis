from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics import AveragePrecision, AUROC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.CaMCheXDataset import CaMCheXDataset
from src.dataloader.SingleViewDataset import SingleViewDataset
from src.dataloader.utils import get_transforms
from src.loss.AsymetricLoss import AsymetricLoss
from src.optimizer import build_adamw_optimizer
from src.scheduler import build_warmup_cosine_scheduler


VIEW_ALIASES = {
    "frontal": {"AP", "PA", "FRONTAL"},
    "lateral": {"LATERAL", "LL"},
}


def add_common_args(parser: argparse.ArgumentParser, model_name: str, default_config: str | None = None) -> None:
    parser.add_argument("--config", default=default_config or f"training/{model_name}/config.yaml")
    parser.add_argument("--train-df-path")
    parser.add_argument("--val-df-path")
    parser.add_argument("--test-df-path")
    parser.add_argument("--output-dir", default=f"output/{model_name}/runs")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--checkpoint-path",
        help="Checkpoint path. In eval: weights to load. In train: weights-only init (fresh optimizer/scheduler/epoch).",
    )
    parser.add_argument(
        "--resume-from",
        help="Train only: full resume from checkpoint (model + optimizer + scheduler + epoch + global_step + best_val_ap). Run dir is inferred from the checkpoint path.",
    )
    parser.add_argument("--seed", type=int, help="Optional seed for python/numpy/torch RNGs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument("--val-check-interval", type=float)
    parser.add_argument("--backbone-name")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    if str(path).startswith("../"):
        return ROOT / str(path)[3:]
    return ROOT / p


def make_run_dir(base_dir: str | Path, run_name: str, run_id: str | None) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve_path(base_dir) or Path(base_dir)
    run_dir = run_dir / f"{run_id}-{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resume_run_dir(checkpoint_path: str | Path) -> Path:
    resolved = resolve_path(checkpoint_path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"--resume-from checkpoint not found: {checkpoint_path}")
    # Expects layout: <run_dir>/checkpoints/<file>.pt
    return resolved.resolve().parents[1]


def prepare_run_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "resume_from", None):
        return resume_run_dir(args.resume_from)
    return make_run_dir(args.output_dir, args.run_name, args.run_id)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_resolved_config(run_dir: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    payload = {"args": vars(args), "config": cfg}
    with open(run_dir / "config.resolved.json", "w") as f:
        json.dump(payload, f, indent=2)


def timm_args_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    timm_args = dict(cfg["model"]["timm_init_args"])
    if args.backbone_name:
        timm_args["model_name"] = args.backbone_name
    if args.no_pretrained:
        timm_args["pretrained"] = False
    return timm_args


def classes_from_config(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["model"]["classes"])


def loss_args_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["model"]["loss_init_args"])


def lr_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> float:
    return args.lr if args.lr is not None else float(cfg["model"]["lr"])


def data_cfg_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data_cfg = dict(cfg["data"]["datamodule_cfg"])
    data_cfg["classes"] = classes_from_config(cfg)
    if args.image_size is not None:
        data_cfg["size"] = args.image_size
    if args.train_df_path:
        data_cfg["train_df_path"] = args.train_df_path
    if args.val_df_path:
        data_cfg["devel_df_path"] = args.val_df_path
    if args.test_df_path:
        data_cfg["pred_df_path"] = args.test_df_path
    return data_cfg


def dataloader_args_from_config(cfg: dict[str, Any], args: argparse.Namespace, shuffle: bool) -> dict[str, Any]:
    dl_args = dict(cfg["data"]["dataloader_init_args"])
    if args.batch_size is not None:
        dl_args["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dl_args["num_workers"] = args.num_workers
    if dl_args.get("num_workers", 0) == 0:
        dl_args["persistent_workers"] = False
    dl_args["shuffle"] = shuffle
    return dl_args


def read_dataframe(path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(path)
    if resolved is None:
        raise FileNotFoundError(path)
    return pd.read_csv(resolved)


def filter_single_view(df: pd.DataFrame, view_position: str) -> pd.DataFrame:
    if view_position == "all" or "ViewPosition" not in df.columns:
        return df.reset_index(drop=True)
    valid = VIEW_ALIASES[view_position]
    mask = df["ViewPosition"].fillna("").astype(str).str.upper().isin(valid)
    return df.loc[mask].reset_index(drop=True)


def make_single_view_loaders(cfg: dict[str, Any], args: argparse.Namespace, view_position: str):
    data_cfg = data_cfg_from_config(cfg, args)
    transforms_train, transforms_val = get_transforms(data_cfg["size"])
    train_df = filter_single_view(read_dataframe(data_cfg["train_df_path"]), view_position)
    val_df = filter_single_view(read_dataframe(data_cfg["devel_df_path"]), view_position)
    train_ds = SingleViewDataset(data_cfg, train_df, transforms_train)
    val_ds = SingleViewDataset(data_cfg, val_df, transforms_val)
    train_loader = DataLoader(train_ds, **dataloader_args_from_config(cfg, args, shuffle=True))
    val_loader = DataLoader(val_ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    return train_loader, val_loader


def make_camchex_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    transforms_train, transforms_val = get_transforms(data_cfg["size"])
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.get("tokenizer") or "dmis-lab/biobert-v1.1")
    train_ds = CaMCheXDataset(data_cfg, read_dataframe(data_cfg["train_df_path"]), transforms_train, tokenizer)
    val_ds = CaMCheXDataset(data_cfg, read_dataframe(data_cfg["devel_df_path"]), transforms_val, tokenizer)
    train_loader = DataLoader(train_ds, **dataloader_args_from_config(cfg, args, shuffle=True))
    val_loader = DataLoader(val_ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    return train_loader, val_loader


def make_single_view_eval_loader(cfg: dict[str, Any], args: argparse.Namespace, view_position: str):
    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"])
    df = filter_single_view(read_dataframe(data_cfg["pred_df_path"]), view_position)
    ds = SingleViewDataset(data_cfg, df, transforms_val)
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    ids = df["path"].tolist() if "path" in df.columns else list(range(len(df)))
    labels_available = all(c in df.columns for c in data_cfg["classes"])
    return loader, ids, labels_available


def make_camchex_eval_loader(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"])
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.get("tokenizer") or "dmis-lab/biobert-v1.1")
    df = read_dataframe(data_cfg["pred_df_path"])
    ds = CaMCheXDataset(data_cfg, df, transforms_val, tokenizer)
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    labels_available = all(c in df.columns for c in data_cfg["classes"])
    return loader, labels_available


def precision_context(device: torch.device, precision: str | None):
    precision = (precision or "32-true").lower()
    enabled = precision not in {"32", "32-true"} and device.type == "cuda"
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if "bf16" in precision else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def train_step(model, criterion, batch, device: torch.device, precision: str | None):
    data, label = batch
    data = move_to_device(data, device)
    label = label.to(device).float()
    with precision_context(device, precision):
        pred = model(data)
        loss = criterion(pred, label)
    return loss, pred, label


@torch.inference_mode()
def validate_model(model, criterion, loader, classes: list[str], device: torch.device, precision: str | None, max_batches: int | None = None):
    model.eval()
    losses = []
    preds = []
    labels = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        loss, pred, label = train_step(model, criterion, batch, device, precision)
        losses.append(loss.detach().cpu())
        preds.append(torch.sigmoid(pred.detach()).cpu())
        labels.append(label.detach().cpu())

    if not losses:
        raise RuntimeError("Validation loader produced no batches")

    pred_tensor = torch.cat(preds)
    label_tensor = torch.cat(labels)
    metrics = compute_metrics(pred_tensor, label_tensor, classes)
    val_loss = torch.stack(losses).mean()
    metrics["val/loss"] = val_loss
    metrics["val_loss"] = val_loss
    return metrics


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, global_step: int, best_val_ap: float, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_ap": best_val_ap,
            "classes": classes,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def append_metric_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in row.items():
        if torch.is_tensor(value):
            serializable[key] = float(value.detach().cpu())
        else:
            serializable[key] = value
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(serializable.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(serializable)


def load_training_checkpoint(model, optimizer, scheduler, checkpoint_path: str | Path) -> tuple[int, int, float]:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    load_model_state(model, checkpoint)
    if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if isinstance(checkpoint, dict) and scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1 if isinstance(checkpoint, dict) else 0
    global_step = int(checkpoint.get("global_step", 0)) if isinstance(checkpoint, dict) else 0
    best_val_ap = float(checkpoint.get("best_val_ap", float("-inf"))) if isinstance(checkpoint, dict) else float("-inf")
    return start_epoch, global_step, best_val_ap


def train_model(model, train_loader, val_loader, args: argparse.Namespace, run_dir: Path, lr: float, classes: list[str], loss_init_args: dict[str, Any]):
    set_seed(getattr(args, "seed", None))
    device = select_device(args.accelerator)
    model.to(device)
    criterion = AsymetricLoss(**loss_init_args).to(device)
    optimizer = build_adamw_optimizer(model.parameters(), lr=lr)
    max_epochs = args.max_epochs if args.max_epochs is not None else 1000
    accumulate_grad_batches = args.accumulate_grad_batches or 1
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulate_grad_batches))
    total_steps = max(1, steps_per_epoch * max_epochs)
    warmup_steps = max(1, min(int(0.05 * steps_per_epoch), total_steps))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        lr=lr,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
    )

    start_epoch = 0
    global_step = 0
    best_val_ap = float("-inf")
    if args.resume_from:
        start_epoch, global_step, best_val_ap = load_training_checkpoint(model, optimizer, scheduler, args.resume_from)
        print(f"resumed from {args.resume_from} at epoch={start_epoch} global_step={global_step} best_val_ap={best_val_ap:.6f}")
    elif args.checkpoint_path:
        load_weights(model, args.checkpoint_path)
        print(f"initialized weights from {args.checkpoint_path} (fresh optimizer/scheduler)")

    use_scaler = device.type == "cuda" and (args.precision or "16-mixed").startswith("16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "logs" / "metrics.csv"
    val_interval = args.val_check_interval
    val_every_batches = None
    if val_interval is not None:
        if val_interval <= 0:
            raise ValueError("--val-check-interval must be positive")
        val_every_batches = max(1, int(len(train_loader) * val_interval)) if val_interval <= 1 else int(val_interval)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.fast_dev_run and batch_idx > 0:
                break
            loss, _, _ = train_step(model, criterion, batch, device, args.precision or "16-mixed")
            scaled_loss = loss / accumulate_grad_batches
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(train_loader)
            if should_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += float(loss.detach().cpu())
            train_batches += 1

            if global_step > 0 and global_step % 50 == 0 and should_step:
                print(f"epoch={epoch} step={global_step} loss={running_loss / max(1, train_batches):.6f}")

            if val_every_batches is not None and (batch_idx + 1) % val_every_batches == 0:
                metrics = validate_model(
                    model,
                    criterion,
                    val_loader,
                    classes,
                    device,
                    args.precision or "16-mixed",
                    max_batches=1 if args.fast_dev_run else None,
                )
                metrics.update({"epoch": epoch, "global_step": global_step, "train/loss": running_loss / max(1, train_batches)})
                append_metric_row(metrics_path, metrics)
                val_ap = float(metrics["val_ap"].detach().cpu())
                print(f"epoch={epoch} step={global_step} val_ap={val_ap:.6f} val_loss={float(metrics['val_loss']):.6f}")
                model.train()

        metrics = validate_model(
            model,
            criterion,
            val_loader,
            classes,
            device,
            args.precision or "16-mixed",
            max_batches=1 if args.fast_dev_run else None,
        )
        train_loss = running_loss / max(1, train_batches)
        metrics.update({"epoch": epoch, "global_step": global_step, "train/loss": train_loss})
        append_metric_row(metrics_path, metrics)
        val_ap = float(metrics["val_ap"].detach().cpu())
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_ap={val_ap:.6f} val_loss={float(metrics['val_loss']):.6f}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, global_step, best_val_ap, classes)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, global_step, best_val_ap, classes)

        if args.fast_dev_run:
            break

    return model


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> dict[str, torch.Tensor]:
    val_ap = []
    val_auc = []
    metrics = {}
    for idx, name in enumerate(classes):
        ap = AveragePrecision(task="binary").to(preds.device)(preds[:, idx], labels[:, idx].long())
        auc = AUROC(task="binary").to(preds.device)(preds[:, idx], labels[:, idx].long())
        val_ap.append(ap)
        val_auc.append(auc)
        metrics[f"val/ap/{name}"] = ap
        metrics[f"val/auroc/{name}"] = auc

    head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
    medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
    tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]
    metrics.update(
        {
            "val_ap": sum(val_ap) / len(val_ap),
            "val_auroc": sum(val_auc) / len(val_auc),
            "val/ap": sum(val_ap) / len(val_ap),
            "val/auroc": sum(val_auc) / len(val_auc),
            "val/ap_head": sum(val_ap[i] for i in head_idx) / len(head_idx),
            "val/ap_medium": sum(val_ap[i] for i in medium_idx) / len(medium_idx),
            "val/ap_tail": sum(val_ap[i] for i in tail_idx) / len(tail_idx),
        }
    )
    return metrics


def load_model_state(model: torch.nn.Module, checkpoint: Any) -> None:
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    candidates = [
        state_dict,
        {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")},
        {k.removeprefix("model.model."): v for k, v in state_dict.items() if k.startswith("model.model.")},
    ]
    errors = []
    model_keys = set(model.state_dict().keys())
    for candidate in candidates:
        if not model_keys.intersection(candidate.keys()):
            continue
        try:
            model.load_state_dict(candidate, strict=False)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "no checkpoint keys matched the model"
    raise RuntimeError(f"Could not load checkpoint: {detail}")


def load_weights(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    try:
        load_model_state(model, checkpoint)
    except RuntimeError as exc:
        raise RuntimeError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc


def save_single_view_encoder(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = model.model
    if hasattr(encoder, "model"):
        encoder = encoder.model
    torch.save(encoder.state_dict(), path)


def select_device(requested: str | None = None) -> torch.device:
    if requested and requested not in {"auto", "gpu"}:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


@torch.inference_mode()
def predict_dataframe(model, loader, classes: list[str], device: torch.device, ids: list[Any] | None = None) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    model.to(device)
    model.eval()
    preds = []
    labels = []
    batch_ids = []
    for batch in loader:
        data, label = batch
        if isinstance(data, (tuple, list)) and data and not torch.is_tensor(data[0]):
            batch_ids.extend(list(data[0]))
        data = move_to_device(data, device)
        pred = torch.sigmoid(model(data)).cpu()
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
