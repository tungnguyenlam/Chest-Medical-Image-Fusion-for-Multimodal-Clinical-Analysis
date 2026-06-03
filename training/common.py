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
from torchmetrics.functional import average_precision, auroc
from tqdm.auto import tqdm

HEAD_IDX = [0, 2, 4, 12, 14, 16, 20, 24]
MEDIUM_IDX = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
TAIL_IDX = [7, 11, 17, 18, 19, 21, 23, 25]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.CaMCheXDataset import CaMCheXDataset
from src.dataloader.PriorAwareDataset import PriorAwareDataset
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
    parser.add_argument(
        "--compile-model",
        action="store_true",
        default=None,
        help="Opt in to torch.compile(model) before training. Default comes from trainer.compile_model, or false if unset.",
    )
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument("--grad-clip", type=float, help="Max grad norm. Set to 0 or negative to disable. Default 1.0 if neither CLI nor config set.")
    parser.add_argument("--val-check-interval", type=float)
    parser.add_argument("--log-every-n-steps", type=int, help="How often to write a row to train_steps.csv. 1 = every optimizer step.")
    parser.add_argument(
        "--quick-val-every-steps",
        type=int,
        help="If set, run a partial validation every N optimizer steps (= N effective batches, where effective = batch_size * accumulate_grad_batches). Logged to val_quick.csv. Does not affect best-checkpoint tracking.",
    )
    parser.add_argument(
        "--quick-val-frac",
        type=float,
        help="Fraction of val loader batches to use for quick validation (default 0.1).",
    )
    parser.add_argument("--prefetch-factor", type=int, help="DataLoader prefetch_factor (requires num_workers > 0).")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-ratio", type=float, help="Warmup steps as a fraction of steps_per_epoch (default 0.05).")
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
    if getattr(args, "prefetch_factor", None) is not None:
        dl_args["prefetch_factor"] = args.prefetch_factor
    if dl_args.get("num_workers", 0) == 0:
        dl_args["persistent_workers"] = False
        dl_args.pop("prefetch_factor", None)
    dl_args["shuffle"] = shuffle
    return dl_args


def trainer_cfg_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("trainer", {}) or {})


def resolve_trainer_arg(args: argparse.Namespace, cfg: dict[str, Any] | None, key: str, default: Any) -> Any:
    cli_val = getattr(args, key, None)
    if cli_val is not None:
        return cli_val
    if cfg is not None:
        tcfg = trainer_cfg_from_config(cfg)
        if key in tcfg and tcfg[key] is not None:
            return tcfg[key]
    return default


def optimizer_args_from_config(cfg: dict[str, Any] | None, args: argparse.Namespace | None = None) -> dict[str, Any]:
    opt_args: dict[str, Any] = {}
    if cfg is not None:
        opt_args = dict((cfg.get("model", {}) or {}).get("optimizer_init_args", {}) or {})
    if args is not None and getattr(args, "weight_decay", None) is not None:
        opt_args["weight_decay"] = args.weight_decay
    return opt_args


def scheduler_args_from_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    return dict((cfg.get("model", {}) or {}).get("scheduler_init_args", {}) or {})


def read_dataframe(path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(path)
    if resolved is None:
        raise FileNotFoundError(path)
    return pd.read_csv(resolved, low_memory=False)


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
    train_dl_args = dataloader_args_from_config(cfg, args, shuffle=True)
    val_dl_args = dataloader_args_from_config(cfg, args, shuffle=False)
    print(f"[dataloader] train: {train_dl_args}")
    print(f"[dataloader] val:   {val_dl_args}")
    train_loader = DataLoader(train_ds, **train_dl_args)
    val_loader = DataLoader(val_ds, **val_dl_args)
    return train_loader, val_loader


def make_camchex_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    transforms_train, transforms_val = get_transforms(data_cfg["size"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_cfg.get("tokenizer") or "dmis-lab/biobert-v1.1",
        trust_remote_code=True,
    )
    train_ds = CaMCheXDataset(data_cfg, read_dataframe(data_cfg["train_df_path"]), transforms_train, tokenizer)
    val_ds = CaMCheXDataset(data_cfg, read_dataframe(data_cfg["devel_df_path"]), transforms_val, tokenizer)
    train_dl_args = dataloader_args_from_config(cfg, args, shuffle=True)
    val_dl_args = dataloader_args_from_config(cfg, args, shuffle=False)
    print(f"[dataloader] train: {train_dl_args}")
    print(f"[dataloader] val:   {val_dl_args}")
    train_loader = DataLoader(train_ds, **train_dl_args)
    val_loader = DataLoader(val_ds, **val_dl_args)
    return train_loader, val_loader


def make_prior_aware_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    """Build train/val loaders backed by the pre-generated prior-aware parquet."""
    data_cfg = data_cfg_from_config(cfg, args)
    transforms_train, transforms_val = get_transforms(data_cfg["size"])
    label_dropout_p = float(data_cfg.get("label_dropout_p", 0.3))

    train_ds = PriorAwareDataset(
        parquet_path=str(resolve_path(data_cfg["train_df_path"])),
        image_size=data_cfg["size"],
        transform=transforms_train,
        label_dropout_p=label_dropout_p,
    )
    val_ds = PriorAwareDataset(
        parquet_path=str(resolve_path(data_cfg["devel_df_path"])),
        image_size=data_cfg["size"],
        transform=transforms_val,
        label_dropout_p=0.0,
    )
    train_dl_args = dataloader_args_from_config(cfg, args, shuffle=True)
    val_dl_args = dataloader_args_from_config(cfg, args, shuffle=False)
    print(f"[dataloader] train: {train_dl_args} (label_dropout_p={label_dropout_p})")
    print(f"[dataloader] val:   {val_dl_args}")
    train_loader = DataLoader(train_ds, **train_dl_args)
    val_loader = DataLoader(val_ds, **val_dl_args)
    return train_loader, val_loader


def make_prior_aware_eval_loader(cfg: dict[str, Any], args: argparse.Namespace):
    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"])
    ds = PriorAwareDataset(
        parquet_path=str(resolve_path(data_cfg["pred_df_path"])),
        image_size=data_cfg["size"],
        transform=transforms_val,
        label_dropout_p=0.0,
    )
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    labels_available = True  # label column is always present in the pregenerated parquet
    return loader, labels_available


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
    tokenizer = AutoTokenizer.from_pretrained(
        data_cfg.get("tokenizer") or "dmis-lab/biobert-v1.1",
        trust_remote_code=True,
    )
    df = read_dataframe(data_cfg["pred_df_path"])
    ds = CaMCheXDataset(data_cfg, df, transforms_val, tokenizer)
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    labels_available = all(c in df.columns for c in data_cfg["classes"])
    return loader, labels_available


_PRECISION_FALLBACK_WARNED: set[tuple[str, str]] = set()


def resolve_precision(device: torch.device, precision: str | None) -> str:
    """Return the effective precision string, downgrading to '32-true' when the device can't support the request.

    bf16 needs torch.cuda.is_bf16_supported() (Ampere+). fp16 autocast only buys
    speed on Volta+ (sm_70+) Tensor Cores; older GPUs (e.g. Kepler K80, sm_37)
    silently upcast and gain nothing, so we fall back to fp32 there too.
    """
    precision = (precision or "32-true").lower()
    if precision in {"32", "32-true"} or device.type != "cuda":
        return precision
    reason: str | None = None
    if "bf16" in precision:
        if not torch.cuda.is_bf16_supported():
            reason = "bf16 not supported on this CUDA device"
    else:
        major, _ = torch.cuda.get_device_capability(device)
        if major < 7:
            reason = f"fp16 has no Tensor Core path on sm_{major}x (pre-Volta)"
    if reason is None:
        return precision
    key = (precision, reason)
    if key not in _PRECISION_FALLBACK_WARNED:
        _PRECISION_FALLBACK_WARNED.add(key)
        print(f"[precision] requested {precision!r} → falling back to '32-true' ({reason})")
    return "32-true"


def precision_context(device: torch.device, precision: str | None):
    precision = resolve_precision(device, precision)
    enabled = precision not in {"32", "32-true"} and device.type == "cuda"
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if "bf16" in precision else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def maybe_compile_model(model: torch.nn.Module, args: argparse.Namespace, cfg: dict[str, Any] | None) -> torch.nn.Module:
    compile_model = bool(resolve_trainer_arg(args, cfg, "compile_model", False))
    if not compile_model:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("--compile-model requires torch.compile, which is unavailable in this PyTorch build")
    print("[train] compiling model with torch.compile()")
    return torch.compile(model)


def train_step(model, criterion, batch, device: torch.device, precision: str | None):
    data, label = batch
    data = move_to_device(data, device)
    label = label.to(device).float()
    with precision_context(device, precision):
        pred = model(data)
        loss = criterion(pred, label)
    return loss, pred, label


@torch.inference_mode()
def validate_model(model, criterion, loader, classes: list[str], device: torch.device, precision: str | None, max_batches: int | None = None, desc: str = "val"):
    was_training = model.training
    model.eval()
    try:
        losses: list[float] = []
        preds: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        total = max_batches if max_batches is not None else len(loader)
        pbar = tqdm(loader, total=total, desc=desc, leave=False, dynamic_ncols=True)
        try:
            for batch_idx, batch in enumerate(pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                loss, pred, label = train_step(model, criterion, batch, device, precision)
                losses.append(float(loss.detach().cpu()))
                preds.append(torch.sigmoid(pred.detach()).cpu())
                labels.append(label.detach().cpu())
        finally:
            pbar.close()

        if not losses:
            raise RuntimeError("Validation loader produced no batches")

        pred_tensor = torch.cat(preds)
        label_tensor = torch.cat(labels)
        metrics = compute_metrics(pred_tensor, label_tensor, classes)
        val_loss = sum(losses) / len(losses)
        metrics["val/loss"] = val_loss
        metrics["val_loss"] = val_loss
        del pred_tensor, label_tensor, preds, labels, losses
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return metrics
    finally:
        if was_training:
            model.train()


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, global_step: int, best_val_ap: float, classes: list[str], scaler=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_compiled_model(model)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_ap": best_val_ap,
            "classes": classes,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        },
        path,
    )


def append_metric_row(csv_path: Path, row: dict[str, Any], fieldnames: list[str] | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in row.items():
        if torch.is_tensor(value):
            serializable[key] = float(value.detach().cpu())
        else:
            serializable[key] = value
    fields = fieldnames if fieldnames is not None else list(serializable.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: serializable.get(k, "") for k in fields})


def load_training_checkpoint(model, optimizer, scheduler, checkpoint_path: str | Path, scaler=None) -> tuple[int, int, float]:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    load_model_state(model, checkpoint)
    if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if isinstance(checkpoint, dict) and scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if isinstance(checkpoint, dict) and scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1 if isinstance(checkpoint, dict) else 0
    global_step = int(checkpoint.get("global_step", 0)) if isinstance(checkpoint, dict) else 0
    best_val_ap = float(checkpoint.get("best_val_ap", float("-inf"))) if isinstance(checkpoint, dict) else float("-inf")
    return start_epoch, global_step, best_val_ap


def train_model(model, train_loader, val_loader, args: argparse.Namespace, run_dir: Path, lr: float, classes: list[str], loss_init_args: dict[str, Any], cfg: dict[str, Any] | None = None):
    seed = resolve_trainer_arg(args, cfg, "seed", None)
    set_seed(seed)
    accelerator = resolve_trainer_arg(args, cfg, "accelerator", None)
    device = select_device(accelerator)
    model.to(device)
    criterion = AsymetricLoss(**loss_init_args).to(device)
    optimizer = build_adamw_optimizer(model, lr=lr, **optimizer_args_from_config(cfg, args))
    max_epochs = resolve_trainer_arg(args, cfg, "max_epochs", 1000)
    accumulate_grad_batches = resolve_trainer_arg(args, cfg, "accumulate_grad_batches", None) or 1
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulate_grad_batches))
    total_steps = max(1, steps_per_epoch * max_epochs)
    sched_kwargs = scheduler_args_from_config(cfg)
    warmup_ratio = getattr(args, "warmup_ratio", None)
    if warmup_ratio is None:
        warmup_ratio = sched_kwargs.pop("warmup_ratio", 0.05)
    else:
        sched_kwargs.pop("warmup_ratio", None)
    warmup_steps = max(1, min(int(warmup_ratio * steps_per_epoch), total_steps))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        lr=lr,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        **sched_kwargs,
    )

    precision = resolve_trainer_arg(args, cfg, "precision", "16-mixed")
    precision = resolve_precision(device, precision)
    use_scaler = device.type == "cuda" and precision.startswith("16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 0
    global_step = 0
    best_val_ap = float("-inf")
    if args.resume_from:
        start_epoch, global_step, best_val_ap = load_training_checkpoint(model, optimizer, scheduler, args.resume_from, scaler=scaler)
        print(f"resumed from {args.resume_from} at epoch={start_epoch} global_step={global_step} best_val_ap={best_val_ap:.6f}")
    elif args.checkpoint_path:
        load_weights(model, args.checkpoint_path)
        print(f"initialized weights from {args.checkpoint_path} (fresh optimizer/scheduler)")

    model = maybe_compile_model(model, args, cfg)

    checkpoint_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    train_csv = logs_dir / "train_steps.csv"
    val_csv = logs_dir / "val_epochs.csv"
    quick_val_csv = logs_dir / "val_quick.csv"

    train_fields = [
        "epoch",
        "global_step",
        "batch_idx",
        "train/loss_step",
        "train/loss_running",
        "train/lr",
        "train/grad_norm",
        "train/scaler_scale",
    ]
    val_fields = (
        [
            "epoch",
            "global_step",
            "trigger",
            "train/loss_epoch",
            "train/grad_norm_epoch",
            "val/loss",
            "val/ap",
            "val/auroc",
            "val/ap_head",
            "val/ap_medium",
            "val/ap_tail",
            "val/auroc_head",
            "val/auroc_medium",
            "val/auroc_tail",
        ]
        + [f"val/ap/{c}" for c in classes]
        + [f"val/auroc/{c}" for c in classes]
    )

    grad_clip = resolve_trainer_arg(args, cfg, "grad_clip", 1.0)
    val_interval = resolve_trainer_arg(args, cfg, "val_check_interval", None)
    val_every_batches = None
    if val_interval is not None:
        if val_interval <= 0:
            raise ValueError("val_check_interval must be positive")
        val_every_batches = max(1, int(len(train_loader) * val_interval)) if val_interval <= 1 else int(val_interval)

    log_every_n_steps = max(1, int(resolve_trainer_arg(args, cfg, "log_every_n_steps", 1) or 1))

    quick_val_every_steps = resolve_trainer_arg(args, cfg, "quick_val_every_steps", None)
    quick_val_frac = float(resolve_trainer_arg(args, cfg, "quick_val_frac", 0.1) or 0.1)
    quick_val_max_batches = None
    if quick_val_every_steps:
        if quick_val_every_steps <= 0:
            raise ValueError("quick_val_every_steps must be positive")
        if not (0.0 < quick_val_frac <= 1.0):
            raise ValueError("quick_val_frac must be in (0, 1]")
        quick_val_max_batches = max(1, int(len(val_loader) * quick_val_frac))

    print(
        f"[train] device={device} precision={precision} scaler={use_scaler} "
        f"lr={lr:.2e} max_epochs={max_epochs} accumulate={accumulate_grad_batches} "
        f"grad_clip={grad_clip} val_every={val_every_batches} log_every={log_every_n_steps} "
        f"quick_val_every={quick_val_every_steps} quick_val_batches={quick_val_max_batches} "
        f"run_dir={run_dir}"
    )

    def _run_validation(epoch: int, gstep: int, trigger: str, train_loss_running: float, avg_grad_norm: float) -> float:
        metrics = validate_model(
            model,
            criterion,
            val_loader,
            classes,
            device,
            precision,
            max_batches=1 if args.fast_dev_run else None,
            desc=f"val @ {trigger} step {gstep}",
        )
        row = {
            "epoch": epoch,
            "global_step": gstep,
            "trigger": trigger,
            "train/loss_epoch": train_loss_running,
            "train/grad_norm_epoch": avg_grad_norm,
            **metrics,
        }
        append_metric_row(val_csv, row, fieldnames=val_fields)
        print_validation_summary(metrics, classes, header=f"epoch {epoch} | step {gstep} | {trigger}")
        return float(metrics["val_ap"])

    quick_val_fields = [
        "epoch",
        "global_step",
        "batch_idx",
        "val_batches",
        "train/loss_running",
        "val/loss",
        "val/ap",
        "val/auroc",
        "val/ap_head",
        "val/ap_medium",
        "val/ap_tail",
        "val/auroc_head",
        "val/auroc_medium",
        "val/auroc_tail",
    ]

    def _run_quick_validation(epoch: int, gstep: int, batch_idx: int, train_loss_running: float) -> None:
        metrics = validate_model(
            model,
            criterion,
            val_loader,
            classes,
            device,
            precision,
            max_batches=quick_val_max_batches,
            desc=f"quick-val @ step {gstep}",
        )
        row = {
            "epoch": epoch,
            "global_step": gstep,
            "batch_idx": batch_idx,
            "val_batches": quick_val_max_batches,
            "train/loss_running": train_loss_running,
            **metrics,
        }
        append_metric_row(quick_val_csv, row, fieldnames=quick_val_fields)
        tqdm.write(
            f"[quick-val] epoch={epoch} step={gstep} "
            f"val_loss={metrics.get('val_loss', float('nan')):.4f} "
            f"val_ap={metrics.get('val_ap', float('nan')):.4f} "
            f"AP h/m/t="
            f"{metrics.get('val/ap_head', float('nan')):.4f}/"
            f"{metrics.get('val/ap_medium', float('nan')):.4f}/"
            f"{metrics.get('val/ap_tail', float('nan')):.4f}"
        )

    running_loss_total = 0.0
    running_batches_total = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_batches = 0
        running_grad_norm = 0.0
        grad_norm_count = 0
        last_grad_norm = float("nan")

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{max_epochs - 1}",
            leave=True,
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx > 0:
                break
            loss, _, _ = train_step(model, criterion, batch, device, precision)
            scaled_loss = loss / accumulate_grad_batches
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(train_loader)
            stepped = False
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_norm = grad_clip if (grad_clip and grad_clip > 0) else float("inf")
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                last_grad_norm = float(total_norm)
                if math.isfinite(last_grad_norm):
                    running_grad_norm += last_grad_norm
                    grad_norm_count += 1
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                stepped = True

            loss_value = float(loss.detach().cpu())
            epoch_loss += loss_value
            epoch_batches += 1
            running_loss_total += loss_value
            running_batches_total += 1
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_loss_avg = epoch_loss / max(1, epoch_batches)
            running_loss_avg = running_loss_total / max(1, running_batches_total)

            pbar.set_postfix(
                loss=f"{epoch_loss_avg:.4f}",
                run_loss=f"{running_loss_avg:.4f}",
                lr=f"{current_lr:.2e}",
                grad_norm=f"{last_grad_norm:.3f}",
                step=global_step,
            )

            if stepped and (global_step % log_every_n_steps == 0):
                append_metric_row(
                    train_csv,
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_idx": batch_idx,
                        "train/loss_step": loss_value,
                        "train/loss_running": running_loss_avg,
                        "train/lr": current_lr,
                        "train/grad_norm": last_grad_norm,
                        "train/scaler_scale": float(scaler.get_scale()) if scaler.is_enabled() else float("nan"),
                    },
                    fieldnames=train_fields,
                )

            if val_every_batches is not None and (batch_idx + 1) % val_every_batches == 0:
                avg_grad_norm = running_grad_norm / grad_norm_count if grad_norm_count else float("nan")
                _run_validation(epoch, global_step, "interval", running_loss_avg, avg_grad_norm)

            if (
                stepped
                and quick_val_every_steps is not None
                and global_step > 0
                and global_step % quick_val_every_steps == 0
            ):
                _run_quick_validation(epoch, global_step, batch_idx, running_loss_avg)
        pbar.close()

        train_loss = epoch_loss / max(1, epoch_batches)
        avg_grad_norm = running_grad_norm / grad_norm_count if grad_norm_count else float("nan")
        val_ap = _run_validation(epoch, global_step, "epoch", train_loss, avg_grad_norm)
        tqdm.write(
            f"epoch={epoch} train_loss={train_loss:.6f} val_ap={val_ap:.6f} grad_norm={avg_grad_norm:.4f}"
        )

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, global_step, best_val_ap, classes, scaler=scaler)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, global_step, best_val_ap, classes, scaler=scaler)

        if args.fast_dev_run:
            break

    return model


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
    model = unwrap_compiled_model(model)
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
    for batch in tqdm(loader, desc="predict", dynamic_ncols=True):
        data, label = batch
        if isinstance(data, dict) and "study_id" in data:
            sid = data["study_id"]
            batch_ids.extend(sid.tolist() if torch.is_tensor(sid) else list(sid))
        elif isinstance(data, (tuple, list)) and data and not torch.is_tensor(data[0]):
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
