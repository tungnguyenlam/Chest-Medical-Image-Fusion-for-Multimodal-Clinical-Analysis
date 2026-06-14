"""The training loop and its checkpoint/validation plumbing.

``train_model`` is the single entry point the *_train.py scripts call. The rest
(train_step, validate_model, save/load checkpoint, append_metric_row) are the
pieces it composes, kept here so the loop reads top-to-bottom.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .constants import ROOT
from .config import (
    build_criterion,
    optimizer_args_from_config,
    resolve_train_batch_size,
    resolve_trainer_arg,
    scheduler_args_from_config,
    set_seed,
)
from .metrics import compute_metrics, print_validation_summary
from .summary import print_model_summary
from .model import (
    ModelEMA,
    gradcam_runner_module,
    load_model_state,
    maybe_channels_last,
    maybe_compile_model,
    move_to_device,
    precision_context,
    resolve_precision,
    select_device,
    unwrap_compiled_model,
)
from .system import host_rss_mb, log_rss

from src.optimizer import build_adamw_optimizer
from src.scheduler import build_warmup_cosine_scheduler


def train_step(model, criterion, batch, device: torch.device, precision: str | None):
    data, label = batch
    data = move_to_device(data, device)
    label = label.to(device).float()
    with precision_context(device, precision):
        pred = model(data)
        loss = criterion(pred, label)
    return loss, pred, label


def batch_study_ids(data: Any) -> list[str] | None:
    if isinstance(data, dict) and "study_id" in data:
        study_ids = data["study_id"]
    elif isinstance(data, (tuple, list)) and data:
        study_ids = data[0]
    else:
        return None
    if torch.is_tensor(study_ids):
        study_ids = study_ids.tolist()
    try:
        return [str(sid) for sid in study_ids]
    except TypeError:
        return [str(study_ids)]


@torch.inference_mode()
def validate_model(model, criterion, loader, classes: list[str], device: torch.device, precision: str | None, max_batches: int | None = None, desc: str = "val", selection_out: dict[str, dict] | None = None):
    """Run validation. If ``selection_out`` is provided, fill it in-place per class with
    {study_id (highest-confidence true positive), prob, first_study_id (first true positive
    in loader order — stable across epochs for progression views)}, reusing the predictions
    already computed here (no extra forward pass)."""
    was_training = model.training
    model.eval()
    n_classes = len(classes)
    sel_prob = [-1.0] * n_classes if selection_out is not None else None
    sel_sid: list[str | None] = [None] * n_classes if selection_out is not None else []
    sel_first: list[str | None] = [None] * n_classes if selection_out is not None else []
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
                prob = torch.sigmoid(pred.detach()).cpu()
                preds.append(prob)
                labels.append(label.detach().cpu())
                if selection_out is not None:
                    study_ids = batch_study_ids(batch[0])
                    if study_ids is None:
                        continue
                    lab = label.detach().cpu()
                    for bi in range(prob.shape[0]):
                        for c in range(n_classes):
                            if lab[bi, c] != 1:
                                continue
                            if sel_first[c] is None:
                                sel_first[c] = study_ids[bi]
                            if prob[bi, c] > sel_prob[c]:
                                sel_prob[c] = float(prob[bi, c])
                                sel_sid[c] = study_ids[bi]
        finally:
            pbar.close()

        if selection_out is not None:
            for c in range(n_classes):
                if sel_sid[c] is not None:
                    selection_out[classes[c]] = {
                        "study_id": sel_sid[c],
                        "prob": sel_prob[c],
                        "first_study_id": sel_first[c],
                    }

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


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    classes: list[str],
    scaler=None,
    checkpoint_kind: str = "epoch",
    best_monitor_value: float | None = None,
    best_monitor_epoch: int | None = None,
    early_stop_bad_epochs: int = 0,
    early_stop_monitor: str = "val_ap",
    early_stop_mode: str = "max",
    schedule: str = "warm_restarts",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_compiled_model(model)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "classes": classes,
        "checkpoint_kind": checkpoint_kind,
        "completed_epoch": checkpoint_kind == "epoch",
        "schedule": schedule,
        "best_monitor_value": best_monitor_value,
        "best_monitor_epoch": best_monitor_epoch,
        "early_stop_bad_epochs": early_stop_bad_epochs,
        "early_stop_monitor": early_stop_monitor,
        "early_stop_mode": early_stop_mode,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
    }
    if early_stop_monitor == "val_ap" and best_monitor_value is not None:
        payload["best_val_ap"] = best_monitor_value
    torch.save(
        payload,
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


def load_training_checkpoint(model, optimizer, scheduler, checkpoint_path: str | Path, scaler=None, expected_schedule: str | None = None) -> tuple[int, int, float | None, int, int | None]:
    from .config import resolve_path

    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    load_model_state(model, checkpoint)
    # A full resume restores the optimizer + scheduler *state*; that state is only
    # valid for the schedule shape that produced it. Switching schedules (e.g.
    # warm_restarts -> single_cosine) mid-run would silently load a mismatched
    # state, so refuse it and point at weights-only init instead.
    if isinstance(checkpoint, dict) and expected_schedule is not None:
        ckpt_schedule = checkpoint.get("schedule", "warm_restarts")
        if ckpt_schedule != expected_schedule:
            raise RuntimeError(
                f"--resume-from checkpoint was trained with schedule={ckpt_schedule!r} but the "
                f"current config uses schedule={expected_schedule!r}. The scheduler state cannot "
                f"be transplanted across schedules. To start a fresh single-cosine run from these "
                f"weights, use --checkpoint-path (weights-only init, fresh optimizer/scheduler) "
                f"instead of --resume-from."
            )
    if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if isinstance(checkpoint, dict) and scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if isinstance(checkpoint, dict) and scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if not isinstance(checkpoint, dict):
        return 0, 0, None, 0, None
    epoch = int(checkpoint.get("epoch", -1))
    # Checkpoints are only ever written at epoch boundaries, so resume always continues
    # at the next epoch.
    start_epoch = epoch + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_monitor_value = checkpoint.get("best_monitor_value", checkpoint.get("best_val_ap"))
    if best_monitor_value is not None:
        best_monitor_value = float(best_monitor_value)
    early_stop_bad_epochs = int(checkpoint.get("early_stop_bad_epochs", 0))
    best_monitor_epoch = checkpoint.get("best_monitor_epoch")
    if best_monitor_epoch is not None:
        best_monitor_epoch = int(best_monitor_epoch)
    return start_epoch, global_step, best_monitor_value, early_stop_bad_epochs, best_monitor_epoch


def train_model(model, train_loader, val_loader, args: argparse.Namespace, run_dir: Path, lr: float, classes: list[str], loss_init_args: dict[str, Any], cfg: dict[str, Any] | None = None):
    seed = resolve_trainer_arg(args, cfg, "seed", None)
    set_seed(seed)
    accelerator = resolve_trainer_arg(args, cfg, "accelerator", None)
    device = select_device(accelerator)
    model.to(device)
    log_rss("model -> device (GPU-resident embedding table host copy released)")
    model = maybe_channels_last(model, args, cfg)
    # Parameter-count summary before the loop starts (printed once; counted before
    # torch.compile wraps the module so names stay readable). Same renderer as
    # scripts/model_summary.py, so it works for every model.
    print_model_summary(model, fmt="plain", depth=2)
    criterion = build_criterion(args, cfg, loss_init_args).to(device)
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
    # Schedule shape is recorded in the checkpoint so a resume into a config with a
    # different schedule fails loudly instead of silently loading a mismatched state.
    schedule = sched_kwargs.get("schedule", "warm_restarts")
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        lr=lr,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        **sched_kwargs,
    )

    precision = resolve_trainer_arg(args, cfg, "precision", "16-mixed")
    precision = resolve_precision(device, precision)
    use_scaler = device.type == "cuda" and precision.startswith("16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 0
    global_step = 0
    best_monitor_value: float | None = None
    best_monitor_epoch: int | None = None
    early_stop_bad_epochs = 0
    if args.resume_from:
        (
            start_epoch,
            global_step,
            best_monitor_value,
            early_stop_bad_epochs,
            best_monitor_epoch,
        ) = load_training_checkpoint(model, optimizer, scheduler, args.resume_from, scaler=scaler, expected_schedule=schedule)
        best_label = "nan" if best_monitor_value is None else f"{best_monitor_value:.6f}"
        print(
            f"resumed from {args.resume_from} at epoch={start_epoch} global_step={global_step} "
            f"best_monitor={best_label} bad_epochs={early_stop_bad_epochs}"
        )
    elif args.checkpoint_path:
        # Weights-only init tolerates shape mismatches (drops them, keeps fresh init) so a
        # related checkpoint can warm-start a different architecture -- e.g. camchex_v3nano
        # -> prior_aware_v3nano (segment_embedding 6->14, prior-only heads init fresh).
        from .model import load_weights

        load_weights(model, args.checkpoint_path, allow_shape_mismatch=True)
        print(f"initialized weights from {args.checkpoint_path} (fresh optimizer/scheduler)")

    model = maybe_compile_model(model, args, cfg)
    log_rss("after compile setup (Inductor graphs build lazily on first batch)")

    # EMA is initialised after weights are loaded (resume/checkpoint_path) and after
    # compile, so the shadow tracks the same parameters the optimizer updates.
    ema_enabled = bool(resolve_trainer_arg(args, cfg, "ema", False))
    ema: ModelEMA | None = None
    if ema_enabled:
        ema_decay = float(resolve_trainer_arg(args, cfg, "ema_decay", 0.999))
        ema = ModelEMA(model, ema_decay)
        print(f"[ema] enabled decay={ema_decay} (EMA weights are evaluated and saved)")

    checkpoint_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    train_csv = logs_dir / "train_steps.csv"
    val_csv = logs_dir / "val_epochs.csv"
    quick_val_csv = logs_dir / "val_quick.csv"

    # Per-component LR columns (discriminative LR) and per-loss-component columns
    # (composite loss). Both collapse to nothing in the common single-LR / single-loss
    # case, so the CSV schema is unchanged unless those features are in use.
    lr_group_names: list[str] = []
    for g in optimizer.param_groups:
        nm = g.get("name", "base")
        if nm not in lr_group_names:
            lr_group_names.append(nm)
    per_group_lr = len(lr_group_names) > 1
    loss_term_names: list[str] = list(getattr(criterion, "names", []) or [])

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
    if per_group_lr:
        train_fields += [f"train/lr/{nm}" for nm in lr_group_names]
    train_fields += [f"train/loss/{nm}" for nm in loss_term_names]
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

    early_stop_monitor = str(resolve_trainer_arg(args, cfg, "early_stop_monitor", "val_ap"))
    early_stop_mode = str(resolve_trainer_arg(args, cfg, "early_stop_mode", "max")).lower()
    if early_stop_mode not in {"min", "max"}:
        raise ValueError("early_stop_mode must be 'min' or 'max'")
    early_stop_patience = int(resolve_trainer_arg(args, cfg, "early_stop_patience", 3) or 0)
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be >= 0")
    early_stop_min_delta = float(resolve_trainer_arg(args, cfg, "early_stop_min_delta", 0.0) or 0.0)
    if early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be >= 0")

    quick_val_every_steps = resolve_trainer_arg(args, cfg, "quick_val_every_steps", None)
    quick_val_frac = float(resolve_trainer_arg(args, cfg, "quick_val_frac", 0.1) or 0.1)
    quick_val_max_batches = None
    if quick_val_every_steps:
        if quick_val_every_steps <= 0:
            raise ValueError("quick_val_every_steps must be positive")
        if not (0.0 < quick_val_frac <= 1.0):
            raise ValueError("quick_val_frac must be in (0, 1]")
        quick_val_max_batches = max(1, int(len(val_loader) * quick_val_frac))

    # Fraction-of-epoch validation scheduling (batch-size independent). End-of-epoch full
    # validation always runs; these schedule extra full/quick validations within an epoch.
    n_train_batches = len(train_loader)

    def _epoch_fracs_to_batches(fracs, name: str) -> set[int]:
        targets: set[int] = set()
        if n_train_batches < 2:
            return targets
        for f in fracs:
            f = float(f)
            if not (0.0 < f < 1.0):
                raise ValueError(f"{name} entries must each be in (0, 1); got {f}")
            targets.add(max(1, min(n_train_batches - 1, round(f * n_train_batches))))
        return targets

    full_val_fracs = resolve_trainer_arg(args, cfg, "full_val_fracs", []) or []
    quick_val_fracs = resolve_trainer_arg(args, cfg, "quick_val_fracs", [0.25, 0.5, 0.75]) or []
    full_val_batches = _epoch_fracs_to_batches(full_val_fracs, "full_val_fracs")
    # Honor the legacy fixed-interval knob too (null by default): treat its mid-epoch points
    # as additional full-validation batches, excluding the final batch (covered by epoch val).
    if val_every_batches is not None:
        full_val_batches |= set(range(val_every_batches, n_train_batches, val_every_batches))
    # Quick validations never collide with full ones; full takes precedence on shared batches.
    quick_val_batches = _epoch_fracs_to_batches(quick_val_fracs, "quick_val_fracs") - full_val_batches

    print(
        f"[train] device={device} precision={precision} scaler={use_scaler} "
        f"lr={lr:.2e} max_epochs={max_epochs} accumulate={accumulate_grad_batches} "
        f"grad_clip={grad_clip} val_every={val_every_batches} log_every={log_every_n_steps} "
        f"early_stop={early_stop_monitor}/{early_stop_mode}/patience={early_stop_patience or None}/min_delta={early_stop_min_delta:g} "
        f"quick_val_every={quick_val_every_steps} quick_val_batches={quick_val_max_batches} "
        f"full_val@batches={sorted(full_val_batches) or None} quick_val@batches={sorted(quick_val_batches) or None} "
        f"run_dir={run_dir}"
    )

    # Validation runs a larger (forward-only) batch than training; if that doesn't fit alongside
    # the live training state (weights + grads + optimizer moments), downgrade once to the train
    # batch size and keep going, rather than crashing the run.
    val_oom_downgraded = False

    def _rebuild_val_loader(loader, batch_size: int):
        dl_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory,
            "collate_fn": loader.collate_fn,
            "drop_last": loader.drop_last,
        }
        if loader.num_workers > 0:
            dl_kwargs["persistent_workers"] = loader.persistent_workers
            if loader.prefetch_factor is not None:
                dl_kwargs["prefetch_factor"] = loader.prefetch_factor
        return DataLoader(loader.dataset, **dl_kwargs)

    def _validate(**kwargs):
        nonlocal val_loader, val_oom_downgraded
        # When EMA is on, evaluate the smoothed weights (then restore the raw ones).
        if ema is not None:
            ema.apply_to(model)
        try:
            try:
                return validate_model(model, criterion, val_loader, classes, device, precision, **kwargs)
            except torch.cuda.OutOfMemoryError:
                train_bs = resolve_train_batch_size(cfg, args) if cfg is not None else None
                if val_oom_downgraded or device.type != "cuda" or train_bs is None or val_loader.batch_size <= train_bs:
                    raise
                torch.cuda.empty_cache()
                tqdm.write(
                    f"[val] OOM at val batch_size={val_loader.batch_size}; rebuilding at {train_bs} "
                    f"and retrying (one-time downgrade for the rest of the run)."
                )
                val_loader = _rebuild_val_loader(val_loader, train_bs)
                val_oom_downgraded = True
                return validate_model(model, criterion, val_loader, classes, device, precision, **kwargs)
        finally:
            if ema is not None:
                ema.restore(model)

    def _run_validation(epoch: int, gstep: int, trigger: str, train_loss_running: float, avg_grad_norm: float, selection_out: dict | None = None) -> dict[str, float]:
        metrics = _validate(
            max_batches=1 if args.fast_dev_run else None,
            desc=f"val @ {trigger} step {gstep}",
            selection_out=selection_out,
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
        return metrics

    def _is_improvement(current: float, best: float | None) -> bool:
        if best is None:
            return True
        if early_stop_mode == "max":
            return current > best + early_stop_min_delta
        return current < best - early_stop_min_delta

    def _save_training_checkpoint(path: Path, epoch: int, kind: str) -> None:
        # Save the EMA weights as the model state (eval-ready), then restore raw weights.
        if ema is not None:
            ema.apply_to(model)
        try:
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                classes,
                scaler=scaler,
                checkpoint_kind=kind,
                best_monitor_value=best_monitor_value,
                best_monitor_epoch=best_monitor_epoch,
                early_stop_bad_epochs=early_stop_bad_epochs,
                early_stop_monitor=early_stop_monitor,
                early_stop_mode=early_stop_mode,
                schedule=schedule,
            )
        finally:
            if ema is not None:
                ema.restore(model)

    def _should_gradcam(epoch: int) -> bool:
        # Default: dump every epoch (it's cheap). Disable with 'none'/'off'/''; or pass
        # a comma list of 0-indexed epochs (e.g. '0,4,9') to restrict.
        default_gradcam = "all" if gradcam_runner_module(model) else "none"
        raw = str(resolve_trainer_arg(args, cfg, "gradcam_epochs", default_gradcam)).strip().lower()
        if raw in {"none", "off", "false", "-1", ""}:
            return False
        if raw in {"all", "every", "true"}:
            return True
        return epoch in {int(tok) for tok in raw.replace(" ", "").split(",") if tok != ""}

    def _maybe_dump_gradcam(epoch: int, epoch_path: Path, selection: dict[str, dict] | None) -> None:
        if not _should_gradcam(epoch):
            return
        runner_module = gradcam_runner_module(model)
        if not runner_module:
            model_name = type(unwrap_compiled_model(model)).__name__
            tqdm.write(f"[gradcam] skipping: {model_name} does not define gradcam_runner_module")
            return
        if not selection:
            tqdm.write("[gradcam] no per-class true-positive selection captured during validation; skipping.")
            return
        # Reuse the studies chosen from this epoch's validation logits — no extra scan.
        #   best  = highest-confidence true positive per class (varies across epochs)
        #   first = first true positive per class in val order (fixed -> progression view)
        gradcam_dir = run_dir / "gradcam" / f"epoch_{epoch}"
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        sel_path = gradcam_dir / "selection.json"
        sets = {
            "best": {cls: info["study_id"] for cls, info in selection.items()},
            "first": {cls: info["first_study_id"] for cls, info in selection.items() if info.get("first_study_id")},
        }
        with open(sel_path, "w") as f:
            json.dump(sets, f, indent=2)
        default_gradcam_device = str(device)
        gradcam_device = str(
            resolve_trainer_arg(args, cfg, "gradcam_device", default_gradcam_device)
            or default_gradcam_device
        )
        cmd = [
            sys.executable, "-m", runner_module,
            "--config", str(args.config),
            "--checkpoint-path", str(epoch_path),
            "--split", "val",
            "--gradcam-epoch", str(epoch),
            "--studies-json", str(sel_path),
            "--device", gradcam_device,
        ]
        if getattr(args, "val_df_path", None):
            cmd += ["--val-df-path", str(args.val_df_path)]
        if getattr(args, "image_size", None):
            cmd += ["--image-size", str(args.image_size)]
        tqdm.write(f"[gradcam] dumping {len(selection)} per-class panels after epoch {epoch} (device={gradcam_device})...")
        # The subprocess is blocking, so training is paused — but the parent still holds its
        # model + optimizer + reserved caching-allocator VRAM. Hand the reserved-but-free blocks
        # back to the driver so a cuda Grad-CAM child has headroom and doesn't OOM against us.
        if "cuda" in gradcam_device and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            subprocess.run(cmd, cwd=str(ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            tqdm.write(f"[gradcam] subprocess failed (rc={exc.returncode}); continuing training.")

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
        metrics = _validate(
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
        # TEMP vram probe: reset per-epoch peak so the postfix shows this epoch's true high-water.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
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
                if ema is not None:
                    ema.update(model)

            loss_value = float(loss.detach().cpu())
            epoch_loss += loss_value
            epoch_batches += 1
            running_loss_total += loss_value
            running_batches_total += 1
            group_lrs = {g.get("name", "base"): float(g["lr"]) for g in optimizer.param_groups}
            current_lr = group_lrs.get("base", max(group_lrs.values()))
            epoch_loss_avg = epoch_loss / max(1, epoch_batches)
            running_loss_avg = running_loss_total / max(1, running_batches_total)

            # TEMP vram probe: alloc=live tensors, peak=epoch max alloc, resv=what nvidia-smi sees.
            if device.type == "cuda":
                vram = (
                    f"{torch.cuda.memory_allocated(device)/1e9:.1f}/"
                    f"{torch.cuda.max_memory_allocated(device)/1e9:.1f}/"
                    f"{torch.cuda.memory_reserved(device)/1e9:.1f}"
                )
            else:
                vram = "n/a"

            # Host RSS / process peak (GB): catches main-process RAM growth across the
            # epoch and the torch.compile spike (peak) that num_workers tuning can't touch.
            rss_mb, hwm_mb = host_rss_mb()
            ram = "n/a" if rss_mb != rss_mb else f"{rss_mb/1024:.1f}/{hwm_mb/1024:.1f}"

            pbar.set_postfix(
                loss=f"{epoch_loss_avg:.4f}",
                run_loss=f"{running_loss_avg:.4f}",
                lr=f"{current_lr:.2e}",
                grad_norm=f"{last_grad_norm:.3f}",
                step=global_step,
                vram=vram,
                ram=ram,
            )

            if stepped and (global_step % log_every_n_steps == 0):
                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_idx": batch_idx,
                    "train/loss_step": loss_value,
                    "train/loss_running": running_loss_avg,
                    "train/lr": current_lr,
                    "train/grad_norm": last_grad_norm,
                    "train/scaler_scale": float(scaler.get_scale()) if scaler.is_enabled() else float("nan"),
                }
                if per_group_lr:
                    row.update({f"train/lr/{nm}": group_lrs.get(nm, float("nan")) for nm in lr_group_names})
                # Weighted per-loss-component values from the composite loss's last forward.
                for nm, val in getattr(criterion, "last_terms", {}).items():
                    row[f"train/loss/{nm}"] = val
                append_metric_row(train_csv, row, fieldnames=train_fields)

            do_full_val = (batch_idx + 1) in full_val_batches
            do_quick_val = (batch_idx + 1) in quick_val_batches or (
                stepped
                and quick_val_every_steps is not None
                and global_step > 0
                and global_step % quick_val_every_steps == 0
            )
            if do_full_val:
                avg_grad_norm = running_grad_norm / grad_norm_count if grad_norm_count else float("nan")
                _run_validation(epoch, global_step, "interval", running_loss_avg, avg_grad_norm)
            elif do_quick_val:
                _run_quick_validation(epoch, global_step, batch_idx, running_loss_avg)
        pbar.close()

        train_loss = epoch_loss / max(1, epoch_batches)
        avg_grad_norm = running_grad_norm / grad_norm_count if grad_norm_count else float("nan")
        gradcam_selection: dict[str, dict] | None = {} if _should_gradcam(epoch) else None
        metrics = _run_validation(epoch, global_step, "epoch", train_loss, avg_grad_norm, selection_out=gradcam_selection)
        val_ap = float(metrics["val_ap"])
        tqdm.write(
            f"epoch={epoch} train_loss={train_loss:.6f} val_ap={val_ap:.6f} grad_norm={avg_grad_norm:.4f}"
        )

        if early_stop_monitor not in metrics:
            raise KeyError(f"early_stop_monitor={early_stop_monitor!r} was not found in validation metrics")
        monitor_value = float(metrics[early_stop_monitor])
        improved = _is_improvement(monitor_value, best_monitor_value)
        if improved:
            best_monitor_value = monitor_value
            best_monitor_epoch = epoch
            early_stop_bad_epochs = 0
        else:
            early_stop_bad_epochs += 1
        best_label = "nan" if best_monitor_value is None else f"{best_monitor_value:.6f}"
        tqdm.write(
            f"[early-stop] monitor={early_stop_monitor} current={monitor_value:.6f} "
            f"best={best_label} best_epoch={best_monitor_epoch} "
            f"bad_epochs={early_stop_bad_epochs}/{early_stop_patience or 'disabled'}"
        )

        epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        _save_training_checkpoint(epoch_path, epoch, "epoch")
        tqdm.write(f"[checkpoint] saved epoch checkpoint: {epoch_path}")
        _maybe_dump_gradcam(epoch, epoch_path, gradcam_selection)

        if early_stop_patience and early_stop_bad_epochs >= early_stop_patience:
            tqdm.write(
                f"[early-stop] stopping at epoch={epoch}: {early_stop_monitor} did not improve "
                f"for {early_stop_bad_epochs} epoch(s)"
            )
            break

        if args.fast_dev_run:
            break

    return model
