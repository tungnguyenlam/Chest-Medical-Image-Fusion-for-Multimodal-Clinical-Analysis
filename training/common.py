from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import multiprocessing
import subprocess
import sys
from multiprocessing import cpu_count
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
from src.dataloader.CaMCheXVitalsDataset import CaMCheXVitalsDataset
from src.dataloader.PriorAwareDataset import PriorAwareDataset
from src.dataloader.SingleViewDataset import SingleViewDataset
from src.dataloader.image_channel_preprocessing import CHANNEL_MODES
from src.dataloader.utils import (
    channel_cache_path,
    get_transforms,
    load_or_build_channels,
    make_preprocess_config,
    resolve_preferred_image_path,
)
from src.loss.AsymetricLoss import AsymetricLoss
from src.optimizer import build_adamw_optimizer
from src.scheduler import build_warmup_cosine_scheduler
from src.utils.text_embedding_cache import TextEmbeddingCache


VIEW_ALIASES = {
    "frontal": {"AP", "PA", "FRONTAL"},
    "lateral": {"LATERAL", "LL"},
}

# 3-channel CXR modes exposed on the CLI. Every mode in CHANNEL_MODES is
# raw + CLAHE + a third channel -- only the third varies (hist_eq / sobel /
# laplacian / log / lbp). We only enable hist-eq for now; to allow another,
# append its mode name here (e.g. "raw_clahe_sobel") -- no other change needed.
ENABLED_CHANNEL_MODES = ["raw_clahe_histeq"]
assert all(m in CHANNEL_MODES for m in ENABLED_CHANNEL_MODES), (
    "ENABLED_CHANNEL_MODES has names not in CHANNEL_MODES: "
    f"{[m for m in ENABLED_CHANNEL_MODES if m not in CHANNEL_MODES]}"
)

# Fraction of CPU cores used to precompute the 3-channel image cache before
# training. Matches src/prepare/01_make_dataset.py: half the cores by default so
# the build leaves headroom and does not get OOM/CPU-killed on a shared server.
CPU_FRACTION = 0.5


def _build_channel_cache_entry(image_path, mode, preprocess_cfg, cache_dir):
    """Pool worker: build+cache one image's channels. Returns a small bool.

    We deliberately return a bool (not the array) so the worker keeps the
    512x512x3 buffer to itself instead of pickling it back to the parent.
    """
    arr = load_or_build_channels(image_path, mode, preprocess_cfg, cache_dir)
    return arr is not None


def precompute_channel_cache(data_cfg: dict[str, Any], dfs: list[pd.DataFrame], desc: str = "channels") -> None:
    """Build the 3-channel cache for every image up front (default before training).

    Runs in parallel over CPU_FRACTION of cores so the first epoch reads a warm
    cache instead of paying decode+filter on every miss. No-op unless both a
    ``channel_mode`` and an ``image_channel_cache_dir`` are configured (legacy RGB
    or cache-disabled runs have nothing to prebuild). Already-cached images are
    skipped, so re-runs are cheap.
    """
    mode = data_cfg.get("channel_mode")
    cache_dir = data_cfg.get("image_channel_cache_dir")
    if not mode:
        print(f"[precompute] {desc}: skipped -- channel_mode not set (legacy ImageNet RGB)", flush=True)
        return
    if not cache_dir:
        print(
            f"[precompute] {desc}: skipped -- image_channel_cache_dir not set in config; "
            "channels will be rebuilt on the fly every epoch (no cache). Set it to enable "
            "the shared cache + this prebuild.",
            flush=True,
        )
        return

    preprocess_cfg = make_preprocess_config(data_cfg)
    seen: set[str] = set()
    paths: list[str] = []
    for df in dfs:
        if "path" not in df.columns:
            continue
        for raw in df["path"].tolist():
            resolved = resolve_preferred_image_path(raw)
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)

    todo = [p for p in paths if not channel_cache_path(cache_dir, p, mode, preprocess_cfg).exists()]
    if not todo:
        print(f"[precompute] {desc}: all {len(paths)} images already cached in {cache_dir}")
        return

    n_workers = max(1, int(cpu_count() * CPU_FRACTION))
    print(
        f"[precompute] {desc}: building {len(todo)}/{len(paths)} channel images "
        f"(mode={mode}) with {n_workers} workers -> {cache_dir}",
        flush=True,
    )
    worker = functools.partial(
        _build_channel_cache_entry, mode=mode, preprocess_cfg=preprocess_cfg, cache_dir=cache_dir
    )
    failures = 0
    mp_ctx = multiprocessing.get_context("fork")
    with mp_ctx.Pool(n_workers) as pool:
        for ok in tqdm(pool.imap_unordered(worker, todo, chunksize=32), total=len(todo), desc=desc):
            if not ok:
                failures += 1
    if failures:
        print(f"[precompute] {desc}: {failures}/{len(todo)} images were unreadable and skipped")


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
        help="Train only: full resume from checkpoint (model + optimizer + scheduler + epoch + global_step + early-stop state). Run dir is inferred from the checkpoint path.",
    )
    parser.add_argument("--seed", type=int, help="Optional seed for python/numpy/torch RNGs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument(
        "--channel-mode",
        choices=ENABLED_CHANNEL_MODES + ["none"],
        help=(
            "3-channel CXR representation. Each mode is raw + CLAHE + a third "
            "channel; only the third varies. Enabled: "
            f"{ENABLED_CHANNEL_MODES} (raw_clahe_histeq = raw + CLAHE + "
            "histogram equalization). 'none' = legacy ImageNet RGB. Overrides "
            "data.datamodule_cfg.channel_mode; the on-disk cache "
            "(image_channel_cache_dir) stays shared across models via config."
        ),
    )
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
        "--checkpoint-every-steps",
        type=int,
        help="Save one temporary mid-epoch recovery checkpoint every N optimizer steps. Default 1000; set 0 to disable.",
    )
    parser.add_argument("--early-stop-monitor", help="Epoch validation metric to monitor for early stopping. Default val_ap.")
    parser.add_argument("--early-stop-mode", choices=["min", "max"], help="Whether early-stop monitor should decrease or increase. Default max.")
    parser.add_argument("--early-stop-patience", type=int, help="Stop after this many non-improving epochs. Default 3; set 0 to disable.")
    parser.add_argument("--early-stop-min-delta", type=float, help="Minimum metric change required to count as an improvement. Default 0.0.")
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
    parser.add_argument(
        "--gradcam-epochs",
        help="When to dump per-class Grad-CAM panels: 'all' (default, every epoch — it's cheap), "
             "'none' to disable, or a comma list of 0-indexed epochs (e.g. '0,4,9'). Vitals model only.",
    )
    parser.add_argument("--gradcam-device", help="Device for the Grad-CAM subprocess (cpu/cuda/mps). Default: cpu.")


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
    if getattr(args, "channel_mode", None) is not None:
        data_cfg["channel_mode"] = None if args.channel_mode == "none" else args.channel_mode
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
    precompute_channel_cache(data_cfg, [train_df, val_df], desc="singleview channels")
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
    train_df = read_dataframe(data_cfg["train_df_path"])
    val_df = read_dataframe(data_cfg["devel_df_path"])
    precompute_channel_cache(data_cfg, [train_df, val_df], desc="camchex channels")
    train_ds = CaMCheXDataset(data_cfg, train_df, transforms_train, tokenizer)
    val_ds = CaMCheXDataset(data_cfg, val_df, transforms_val, tokenizer)
    train_dl_args = dataloader_args_from_config(cfg, args, shuffle=True)
    val_dl_args = dataloader_args_from_config(cfg, args, shuffle=False)
    print(f"[dataloader] train: {train_dl_args}")
    print(f"[dataloader] val:   {val_dl_args}")
    train_loader = DataLoader(train_ds, **train_dl_args)
    val_loader = DataLoader(val_ds, **val_dl_args)
    return train_loader, val_loader


def _clinical_text(row: pd.Series) -> str:
    text = row.get("clinical_indication", "")
    if pd.isna(text) or str(text).strip() == "":
        return "No clinical history available."
    return str(text)


def maybe_add_camchex_vitals_text_embeddings(
    cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    dfs: list[pd.DataFrame],
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    model_cfg = cfg.get("model", {}) or {}
    model_init_args = dict(model_cfg.get("model_init_args", {}) or {})
    use_cache = bool(
        data_cfg.get("use_text_embedding_cache", False)
        or model_init_args.get("use_precomputed_text_embeddings", False)
        or getattr(args, "use_precomputed_text_embeddings", False)
    )
    if not use_cache:
        return data_cfg

    rows = []
    seen = set()
    for df in dfs:
        for _, row in df.groupby("study_id", sort=False).head(1).iterrows():
            study_id = str(row["study_id"])
            if study_id in seen:
                continue
            seen.add(study_id)
            rows.append((study_id, _clinical_text(row)))

    if not rows:
        return data_cfg

    text_model = (
        getattr(args, "text_model", None)
        or model_cfg.get("text_model")
        or data_cfg.get("tokenizer")
        or "microsoft/BiomedVLP-CXR-BERT-specialized"
    )
    cache = TextEmbeddingCache(
        text_model=text_model,
        cache_root=getattr(args, "text_embedding_cache_dir", None) or data_cfg.get("text_embedding_cache_dir", "data/text_embeddings"),
        batch_size=int(data_cfg.get("text_embedding_batch_size", 32) or 32),
        device=data_cfg.get("text_embedding_device", "auto"),
    )
    cache.ensure_texts([text for _, text in rows], max_length=384, desc=f"{text_model} clinical embeddings")
    cache.unload_model()
    data_cfg = dict(data_cfg)
    data_cfg["clinical_embedding_cache"] = cache
    return data_cfg


def make_camchex_vitals_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    transforms_train, transforms_val = get_transforms(data_cfg["size"])
    train_df = read_dataframe(data_cfg["train_df_path"])
    val_df = read_dataframe(data_cfg["devel_df_path"])
    precompute_channel_cache(data_cfg, [train_df, val_df], desc="camchex_vitals channels")
    data_cfg = maybe_add_camchex_vitals_text_embeddings(cfg, data_cfg, [train_df, val_df], args=args)
    tokenizer = None
    if "clinical_embedding_cache" not in data_cfg and "clinical_embeddings" not in data_cfg:
        tokenizer = AutoTokenizer.from_pretrained(
            data_cfg.get("tokenizer") or "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )
    train_ds = CaMCheXVitalsDataset(data_cfg, train_df, transforms_train, tokenizer)
    val_ds = CaMCheXVitalsDataset(data_cfg, val_df, transforms_val, tokenizer)
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


def make_camchex_vitals_eval_loader(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"])
    df = read_dataframe(data_cfg["pred_df_path"])
    data_cfg = maybe_add_camchex_vitals_text_embeddings(cfg, data_cfg, [df], args=args)
    tokenizer = None
    if "clinical_embedding_cache" not in data_cfg and "clinical_embeddings" not in data_cfg:
        tokenizer = AutoTokenizer.from_pretrained(
            data_cfg.get("tokenizer") or "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )
    ds = CaMCheXVitalsDataset(data_cfg, df, transforms_val, tokenizer)
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
                    study_ids = batch[0][0]  # data[0] = collated study_id list (len B)
                    lab = label.detach().cpu()
                    for bi in range(prob.shape[0]):
                        for c in range(n_classes):
                            if lab[bi, c] != 1:
                                continue
                            if sel_first[c] is None:
                                sel_first[c] = str(study_ids[bi])
                            if prob[bi, c] > sel_prob[c]:
                                sel_prob[c] = float(prob[bi, c])
                                sel_sid[c] = str(study_ids[bi])
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_compiled_model(model)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "classes": classes,
        "checkpoint_kind": checkpoint_kind,
        "completed_epoch": checkpoint_kind == "epoch",
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


def load_training_checkpoint(model, optimizer, scheduler, checkpoint_path: str | Path, scaler=None) -> tuple[int, int, float | None, int, int | None]:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    load_model_state(model, checkpoint)
    if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if isinstance(checkpoint, dict) and scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if isinstance(checkpoint, dict) and scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if not isinstance(checkpoint, dict):
        return 0, 0, None, 0, None
    epoch = int(checkpoint.get("epoch", -1))
    checkpoint_kind = str(checkpoint.get("checkpoint_kind", "epoch"))
    completed_epoch = bool(checkpoint.get("completed_epoch", checkpoint_kind != "mid_epoch"))
    start_epoch = epoch + 1 if completed_epoch else max(0, epoch)
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
        ) = load_training_checkpoint(model, optimizer, scheduler, args.resume_from, scaler=scaler)
        best_label = "nan" if best_monitor_value is None else f"{best_monitor_value:.6f}"
        print(
            f"resumed from {args.resume_from} at epoch={start_epoch} global_step={global_step} "
            f"best_monitor={best_label} bad_epochs={early_stop_bad_epochs}"
        )
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
    checkpoint_every_steps = int(resolve_trainer_arg(args, cfg, "checkpoint_every_steps", 1000) or 0)
    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be >= 0")

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

    print(
        f"[train] device={device} precision={precision} scaler={use_scaler} "
        f"lr={lr:.2e} max_epochs={max_epochs} accumulate={accumulate_grad_batches} "
        f"grad_clip={grad_clip} val_every={val_every_batches} log_every={log_every_n_steps} "
        f"checkpoint_every={checkpoint_every_steps or None} "
        f"early_stop={early_stop_monitor}/{early_stop_mode}/patience={early_stop_patience or None}/min_delta={early_stop_min_delta:g} "
        f"quick_val_every={quick_val_every_steps} quick_val_batches={quick_val_max_batches} "
        f"run_dir={run_dir}"
    )

    def _run_validation(epoch: int, gstep: int, trigger: str, train_loss_running: float, avg_grad_norm: float, selection_out: dict | None = None) -> dict[str, float]:
        metrics = validate_model(
            model,
            criterion,
            val_loader,
            classes,
            device,
            precision,
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

    def _mid_checkpoint_path(epoch: int, gstep: int) -> Path:
        return checkpoint_dir / f"epoch_{epoch:03d}_step_{gstep:08d}_mid.pt"

    def _delete_mid_checkpoints(epoch: int | None = None) -> None:
        pattern = "*_mid.pt" if epoch is None else f"epoch_{epoch:03d}_step_*_mid.pt"
        for path in checkpoint_dir.glob(pattern):
            path.unlink(missing_ok=True)

    def _save_training_checkpoint(path: Path, epoch: int, kind: str) -> None:
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
        )

    def _should_gradcam(epoch: int) -> bool:
        # Default: dump every epoch (it's cheap). Disable with 'none'/'off'/''; or pass
        # a comma list of 0-indexed epochs (e.g. '0,4,9') to restrict.
        raw = str(resolve_trainer_arg(args, cfg, "gradcam_epochs", "all")).strip().lower()
        if raw in {"none", "off", "false", "-1", ""}:
            return False
        if raw in {"all", "every", "true"}:
            return True
        return epoch in {int(tok) for tok in raw.replace(" ", "").split(",") if tok != ""}

    def _maybe_dump_gradcam(epoch: int, epoch_path: Path, selection: dict[str, dict] | None) -> None:
        if not _should_gradcam(epoch):
            return
        model_name = type(unwrap_compiled_model(model)).__name__
        if model_name != "CaMCheXV2NanoVitalsModel":
            tqdm.write(f"[gradcam] skipping: --gradcam-epochs only supports CaMCheXV2NanoVitalsModel, got {model_name}")
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
        gradcam_device = str(resolve_trainer_arg(args, cfg, "gradcam_device", "cpu") or "cpu")
        cmd = [
            sys.executable, "-m", "src.interpret.run_gradcam",
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
                if checkpoint_every_steps and global_step % checkpoint_every_steps == 0:
                    _delete_mid_checkpoints()
                    mid_path = _mid_checkpoint_path(epoch, global_step)
                    _save_training_checkpoint(mid_path, epoch, "mid_epoch")
                    tqdm.write(f"[checkpoint] saved temporary mid-epoch checkpoint: {mid_path}")

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
        _delete_mid_checkpoints(epoch)
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
