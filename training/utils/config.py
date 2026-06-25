"""Config loading, path/run-dir resolution, and the cfg/CLI -> kwargs helpers.

This is the layer the entry-point scripts lean on most: it reads the YAML config,
resolves repo-relative paths, prepares (or resumes / quick-continues) the run
directory, and turns the merged config+CLI into the dicts the datasets, optimizer,
scheduler, criterion and trainer want. No torch model code lives here.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from .constants import ROOT, THIRD_CHANNEL_TO_MODE, _IMAGENET_MEAN, _IMAGENET_STD

from src.loss import LOSS_REGISTRY, CompositeLoss


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


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    stem = path.stem
    epoch = -1
    if stem.startswith("epoch_"):
        try:
            epoch = int(stem.removeprefix("epoch_"))
        except ValueError:
            epoch = -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return epoch, mtime, str(path)


def _run_sort_key(path: Path) -> tuple[float, str]:
    # Order by mtime first so sorting reverse=True puts the most recently
    # created run dir first, regardless of how the dir is named. Name is a
    # stable tiebreaker for runs sharing an mtime.
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return mtime, path.name


def find_quick_continue_checkpoint(args: argparse.Namespace) -> Path:
    base_dir = resolve_path(args.output_dir) or Path(args.output_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"--quick-continue could not find output dir: {base_dir}")

    if getattr(args, "run_id", None):
        exact = base_dir / str(args.run_id)
        run_dirs = [exact] if exact.is_dir() else sorted(
            [p for p in base_dir.glob(f"{args.run_id}-*") if p.is_dir()],
            key=_run_sort_key,
            reverse=True,
        )
    else:
        run_dirs = sorted(
            [p for p in base_dir.iterdir() if p.is_dir()],
            key=_run_sort_key,
            reverse=True,
        )

    saw_checkpoint_dir = False
    # run_dirs are sorted most-recent-first, so the first run with a checkpoint
    # is the most recently created run.
    for run_dir in run_dirs:
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        saw_checkpoint_dir = True
        # last.pt is the rolling full checkpoint -- the correct seamless-resume target.
        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists():
            return last_ckpt.resolve()
        # Legacy / --keep-epoch-checkpoints runs: fall back to the highest epoch archive,
        # then any other checkpoint (but never best.pt, which is weights-only and can't
        # drive a full resume).
        run_candidates = list(ckpt_dir.glob("epoch_*.pt"))
        if not run_candidates:
            run_candidates = [p for p in ckpt_dir.glob("*.pt") if p.name != "best.pt"]
        if run_candidates:
            return max(run_candidates, key=_checkpoint_sort_key).resolve()

    if not saw_checkpoint_dir:
        scope = f"run_id={args.run_id}" if getattr(args, "run_id", None) else "all runs"
        raise FileNotFoundError(f"--quick-continue found no checkpoint dirs under {base_dir} ({scope})")
    scope = f"run_id={args.run_id}" if getattr(args, "run_id", None) else "all runs"
    raise FileNotFoundError(f"--quick-continue found no checkpoints under {base_dir} ({scope})")


def _explicit_cli_dests(argv: list[str] | None = None) -> set[str]:
    argv = list(sys.argv[1:] if argv is None else argv)
    dests: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dests.add(option.removeprefix("--").replace("-", "_"))
    return dests


def _config_from_run_dir(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Return the config saved alongside a checkpoint's run, or None if not found.

    Runs write ``config.resolved.json`` (or ``config.resume.json``) into the run dir,
    capturing the exact ``{"args", "config"}`` they were launched with. The on-disk
    layout is ``<run_dir>/checkpoints/<file>.pt``, so the run dir is the checkpoint's
    grandparent; we also look in the checkpoint's own directory as a fallback for
    flatter layouts. Returns the ``config`` block so eval can rebuild the matching
    architecture/data pipeline from the checkpoint alone."""
    resolved = resolve_path(checkpoint_path)
    if resolved is None or not resolved.exists():
        return None
    resolved = resolved.resolve()
    parents = resolved.parents
    candidates = [resolved.parent]
    if len(parents) >= 2:
        candidates.append(parents[1])
    for run_dir in candidates:
        for name in ("config.resolved.json", "config.resume.json"):
            path = run_dir / name
            if not path.exists():
                continue
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[eval] found {path} but could not read it ({exc}); falling back to --config", flush=True)
                continue
            cfg = payload.get("config") if isinstance(payload, dict) else None
            if isinstance(cfg, dict):
                print(f"[eval] config resolved from checkpoint's run: {path} (pass --config to override)", flush=True)
                return cfg
    return None


def resolve_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    """Pick the eval config the sane way: hand over a checkpoint and the matching
    config is found automatically, while flags still override.

    Resolution order:

    1. An explicit ``--config`` on the CLI always wins.
    2. Otherwise, if ``--checkpoint-path`` is given, use the config saved next to that
       checkpoint's run (``config.resolved.json``) — the exact config it was trained with.
    3. Otherwise, fall back to ``--config``'s default (the model dir's live ``config.yaml``).

    Either way, individual CLI flags continue to override single config values downstream
    (this only chooses which config dict those overrides apply to)."""
    if "config" not in _explicit_cli_dests():
        ckpt = getattr(args, "checkpoint_path", None)
        if ckpt:
            cfg = _config_from_run_dir(ckpt)
            if cfg is not None:
                return cfg
    return load_config(args.config)


def _restore_quick_continue_args(args: argparse.Namespace, run_dir: Path) -> None:
    config_path = run_dir / "config.resolved.json"
    if not config_path.exists():
        print(f"[quick-continue] no {config_path}; using current CLI/config defaults", flush=True)
        return

    try:
        with open(config_path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[quick-continue] could not read {config_path}: {exc}; using current CLI/config defaults", flush=True)
        return
    saved_args = payload.get("args") if isinstance(payload, dict) else None
    if not isinstance(saved_args, dict):
        print(f"[quick-continue] {config_path} has no saved args; using current CLI/config defaults", flush=True)
        return
    if saved_args.get("quick_continue"):
        print(
            f"[quick-continue] {config_path} looks like it was written by a previous resume attempt; "
            "not restoring those args",
            flush=True,
        )
        return

    explicit = _explicit_cli_dests()
    protected = {"quick_continue", "resume_from", "checkpoint_path"}
    restored: list[str] = []
    for key, value in saved_args.items():
        if key in protected or key in explicit:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
            restored.append(key)

    if restored:
        preview = ", ".join(sorted(restored)[:12])
        suffix = "" if len(restored) <= 12 else f", ... ({len(restored)} total)"
        print(f"[quick-continue] restored saved args: {preview}{suffix}", flush=True)


def _infer_quick_continue_model_args(args: argparse.Namespace, checkpoint_path: Path) -> None:
    if not hasattr(args, "use_precomputed_text_embeddings"):
        return
    if getattr(args, "use_precomputed_text_embeddings", False):
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"[quick-continue] could not inspect checkpoint for model args: {exc}", flush=True)
        return

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or {}
    if not isinstance(state_dict, dict):
        return

    keys = list(state_dict.keys())
    has_model_keys = any(k.startswith(("image_encoder.", "head.", "transformer_encoder.")) for k in keys)
    has_text_encoder = any("text_encoder." in k for k in keys)
    if has_model_keys and not has_text_encoder:
        args.use_precomputed_text_embeddings = True
        if hasattr(args, "freeze_text_encoder"):
            args.freeze_text_encoder = True
        print(
            "[quick-continue] inferred --use-precomputed-text-embeddings from checkpoint "
            "(no text_encoder weights present)",
            flush=True,
        )


def prepare_run_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "quick_continue", False):
        if getattr(args, "resume_from", None):
            raise ValueError("--quick-continue cannot be combined with --resume-from")
        if getattr(args, "checkpoint_path", None):
            raise ValueError("--quick-continue cannot be combined with --checkpoint-path")
        checkpoint = find_quick_continue_checkpoint(args)
        run_dir = resume_run_dir(checkpoint)
        _restore_quick_continue_args(args, run_dir)
        _infer_quick_continue_model_args(args, checkpoint)
        args.resume_from = str(checkpoint)
        args.checkpoint_path = str(checkpoint)
        print(f"[quick-continue] selected checkpoint: {checkpoint}", flush=True)
        return run_dir
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
    path = run_dir / ("config.resume.json" if getattr(args, "resume_from", None) else "config.resolved.json")
    with open(path, "w") as f:
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


def model_init_args_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    model_init_args = dict(cfg.get("model", {}).get("model_init_args", {}) or {})
    model_init_args.setdefault("n_classes", len(classes_from_config(cfg)))
    return model_init_args


def loss_args_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["model"]["loss_init_args"])


def resolve_loss_names(args: argparse.Namespace, cfg: dict[str, Any] | None) -> list[str]:
    """Loss names to use: CLI --loss, else model.loss, else the default ['ASL']."""
    raw = getattr(args, "loss", None)
    if not raw and cfg is not None:
        raw = (cfg.get("model", {}) or {}).get("loss")
    if not raw:
        return ["ASL"]
    if isinstance(raw, str):
        raw = [raw]
    return [str(n).upper() for n in raw]


def build_criterion(args: argparse.Namespace, cfg: dict[str, Any] | None, asl_init_args: dict[str, Any]):
    """Build the training criterion from the loss registry.

    One name -> that loss; several -> a CompositeLoss (weighted sum). Per-loss kwargs come
    from ``model.loss_kwargs.<NAME>``; ASL additionally inherits the flat
    ``model.loss_init_args`` (``asl_init_args``) for backward compatibility. Weights come
    from ``--loss-weights`` (positional) or ``model.loss_weights`` (default 1.0 each).
    """
    model_cfg = (cfg.get("model", {}) or {}) if cfg else {}
    loss_kwargs = model_cfg.get("loss_kwargs", {}) or {}
    cfg_weights = model_cfg.get("loss_weights", {}) or {}
    names = resolve_loss_names(args, cfg)

    cli_weights = getattr(args, "loss_weights", None)
    if cli_weights is not None and len(cli_weights) != len(names):
        raise ValueError(f"--loss-weights has {len(cli_weights)} values but --loss has {len(names)} losses")

    losses, weights = [], []
    for i, name in enumerate(names):
        if name not in LOSS_REGISTRY:
            raise ValueError(f"unknown loss {name!r}; available: {sorted(LOSS_REGISTRY)}")
        kwargs = dict(loss_kwargs.get(name, {}) or {})
        if name == "ASL":
            kwargs = {**asl_init_args, **kwargs}  # flat loss_init_args is ASL's args
            cli_smoothing = getattr(args, "label_smoothing", None)
            if cli_smoothing is not None:  # --label-smoothing overrides config pos_smoothing
                kwargs["pos_smoothing"] = cli_smoothing
        losses.append(LOSS_REGISTRY[name](**kwargs))
        weights.append(float(cli_weights[i]) if cli_weights is not None else float(cfg_weights.get(name, 1.0)))

    print(f"[loss] {names}" + (f" weights={weights}" if len(names) > 1 else ""))
    return losses[0] if len(losses) == 1 else CompositeLoss(losses, weights, names=names)


def lr_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> float:
    return args.lr if args.lr is not None else float(cfg["model"]["lr"])


def data_cfg_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data_cfg = dict(cfg["data"]["datamodule_cfg"])
    data_cfg["classes"] = classes_from_config(cfg)
    if args.image_size is not None:
        data_cfg["size"] = args.image_size
    # --third-channel-mode names only ch2 (short name); map it to the full internal
    # mode ("histeq" -> "raw_clahe_histeq"). data_cfg["channel_mode"] stays the full
    # name everywhere downstream (cache keys, stats, datasets, attribution).
    if getattr(args, "third_channel_mode", None) is not None:
        if args.third_channel_mode == "none":
            data_cfg["channel_mode"] = None
        else:
            data_cfg["channel_mode"] = THIRD_CHANNEL_TO_MODE[args.third_channel_mode]
    if args.train_df_path:
        data_cfg["train_df_path"] = args.train_df_path
    if args.val_df_path:
        data_cfg["devel_df_path"] = args.val_df_path
    if args.test_df_path:
        data_cfg["pred_df_path"] = args.test_df_path
    return data_cfg


def resolve_train_batch_size(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    if args.batch_size is not None:
        return int(args.batch_size)
    return int(cfg["data"]["dataloader_init_args"]["batch_size"])


def optional_bool_arg(value: str | bool | None) -> bool | None:
    """argparse type for ``--flag [true|false]``: omitted -> None, bare flag -> True."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    v = str(value).lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def resolve_pin_memory(args: argparse.Namespace, cfg: dict[str, Any] | None) -> bool:
    cli_val = getattr(args, "pin_memory", None)
    if cli_val is not None:
        return bool(cli_val)
    if cfg is not None:
        dl = (cfg.get("data", {}) or {}).get("dataloader_init_args", {}) or {}
        if "pin_memory" in dl:
            return bool(dl["pin_memory"])
    return True


def resolve_val_batch_size(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Eval is forward-only, so the val/eval loader can run a larger batch than training.
    Explicit --val-batch-size wins; otherwise default to 2x the train batch size."""
    if getattr(args, "val_batch_size", None) is not None:
        return int(args.val_batch_size)
    return 2 * resolve_train_batch_size(cfg, args)


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
    if args is not None and getattr(args, "backbone_lr_mult", None) is not None:
        opt_args["backbone_lr_mult"] = args.backbone_lr_mult
    if args is not None and getattr(args, "text_lr_mult", None) is not None:
        opt_args["text_lr_mult"] = args.text_lr_mult
    return opt_args


def scheduler_args_from_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    return dict((cfg.get("model", {}) or {}).get("scheduler_init_args", {}) or {})


def image_norm_stats(data_cfg: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Per-channel (mean, std) on the [0,1] scale for the configured channel_mode,
    matching what A.Normalize applied on the CPU. Used to set up the model's
    on-device normalization for --uint8-image-pipeline. ImageNet for mode=None."""
    mode = data_cfg.get("channel_mode")
    if mode:
        from src.dataloader.image_channel_preprocessing import CHANNEL_STATS
        stats = CHANNEL_STATS[mode]
        return list(stats["mean"]), list(stats["std"])
    return list(_IMAGENET_MEAN), list(_IMAGENET_STD)


def resolve_uint8_image_pipeline(args: argparse.Namespace, data_cfg: dict[str, Any]) -> bool:
    """Whether the loaders should emit uint8 images for on-device normalization.
    Opt-in via --uint8-image-pipeline (scripts that support it define the flag).
    Requires a channel_mode so every decode path yields uint8 [0,255]."""
    enabled = bool(getattr(args, "uint8_image_pipeline", False))
    if enabled and not data_cfg.get("channel_mode"):
        raise SystemExit(
            "--uint8-image-pipeline requires a channel mode (e.g. --third-channel-mode histeq); "
            "without one the legacy decode path is not guaranteed to be uint8."
        )
    return enabled
