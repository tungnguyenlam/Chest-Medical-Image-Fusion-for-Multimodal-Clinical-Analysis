"""Dataset / DataLoader construction, the up-front image-channel cache prebuild,
and the training-time text-embedding cache.

Every ``make_*_loaders`` / ``make_*_eval_loader`` entry point lives here, one per
model family (single-view, camchex, camchex+vitals, prior-aware). They share the
channel-precompute and text-embedding helpers below.
"""
from __future__ import annotations

import argparse
import functools
import multiprocessing
import os
from collections.abc import Iterable
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .constants import CPU_FRACTION, VIEW_ALIASES
from .config import (
    data_cfg_from_config,
    resolve_path,
    resolve_trainer_arg,
    resolve_uint8_image_pipeline,
    resolve_val_batch_size,
)
from .system import (
    cap_malloc_arenas,
    configure_dataloader_sharing,
    log_rss,
    resolve_cpu_fraction,
    resolve_malloc_arena_max,
)

from src.dataloader.CaMCheXDataset import CaMCheXDataset
from src.dataloader.CaMCheXVitalsDataset import CaMCheXVitalsDataset
from src.dataloader.PriorAwareDataset import PriorAwareDataset
from src.dataloader.SingleViewDataset import SingleViewDataset
from src.dataloader.image_channel_preprocessing import describe_mode
from src.dataloader.utils import (
    channel_cache_path,
    get_transforms,
    load_or_build_channels,
    make_preprocess_config,
)
from src.utils.text_embedding_cache import TextEmbeddingCache


def _build_channel_cache_entry(image_path, mode, preprocess_cfg, cache_dir):
    """Pool worker: build+cache one image's channels. Returns a small bool.

    We deliberately return a bool (not the array) so the worker keeps the
    512x512x3 buffer to itself instead of pickling it back to the parent.
    """
    arr = load_or_build_channels(image_path, mode, preprocess_cfg, cache_dir)
    return arr is not None


def precompute_channels_for_paths(
    data_cfg: dict[str, Any], raw_paths: Iterable[str], desc: str = "channels",
    cpu_fraction: float | None = None, skip: bool = False,
) -> None:
    """Build the 3-channel cache for the given image paths up front.

    Both phases are parallel: the cache-miss scan runs on a thread pool (the
    per-path work is filesystem ``stat``, which releases the GIL, so threads help
    on slow WSL drvfs), and the build runs on a process pool. Both pools are sized
    from ``cpu_fraction`` of the cores (default CPU_FRACTION). No-op unless both a
    ``channel_mode`` and an ``image_channel_cache_dir`` are configured. Already-cached
    images are skipped, so re-runs are cheap.

    With ``skip=True`` (``--skip-precompute``) the upfront scan/build is bypassed
    entirely; the cache config is untouched, so channels are built lazily on first
    access during training (and cached as usual). Use it to shave startup latency
    when the cache is already warm.
    """
    # Log the resolved channel composition up front (every train start, regardless of
    # --skip-precompute) so a wrong --third-channel-mode or stale stats is caught here
    # rather than after the first epoch.
    _mode = data_cfg.get("channel_mode")
    if _mode:
        composition = describe_mode(_mode, make_preprocess_config(data_cfg))
        print(f"[channels] {desc}:\n  " + composition.replace("\n", "\n  "), flush=True)
    else:
        print(f"[channels] {desc}: legacy ImageNet RGB (channel_mode unset)", flush=True)

    if skip:
        print(f"[precompute] {desc}: skipped -- --skip-precompute set (channels build lazily on first access)", flush=True)
        return
    frac = cpu_fraction if cpu_fraction is not None else CPU_FRACTION
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
    # Dedup raw paths (cheap, no I/O).
    unique_raw = list(dict.fromkeys(str(p) for p in raw_paths))

    # Existing cache files: one directory listing of the mode shard. The cache key
    # is derived from the path string alone (see channel_cache_path), so the whole
    # miss scan is pure CPU -- no per-image stat on the (slow) source filesystem.
    existing_digests: set[str] = set()
    try:
        with os.scandir(Path(cache_dir) / mode) as it:
            for entry in it:
                if entry.name.endswith(".npy"):
                    existing_digests.add(entry.name[:-4])
    except FileNotFoundError:
        pass  # shard not created yet -> everything is a miss

    todo = [
        raw for raw in unique_raw
        if channel_cache_path(cache_dir, raw, mode, preprocess_cfg).stem not in existing_digests
    ]

    if not todo:
        print(f"[precompute] {desc}: all {len(unique_raw)} images already cached in {cache_dir}", flush=True)
        return

    n_workers = max(1, int(cpu_count() * frac))
    print(
        f"[precompute] {desc}: building {len(todo)}/{len(unique_raw)} channel images "
        f"(mode={mode}) with {n_workers} workers -> {cache_dir} "
        f"(resolve+decode happens in the workers, on misses only)",
        flush=True,
    )
    worker = functools.partial(
        _build_channel_cache_entry, mode=mode, preprocess_cfg=preprocess_cfg, cache_dir=cache_dir
    )
    failures = 0
    mp_ctx = multiprocessing.get_context("fork")
    with mp_ctx.Pool(n_workers) as pool:
        # chunksize=1: each build is ~1s of I/O on a slow mount, so per-item
        # dispatch costs nothing and the bar moves per image (chunking would
        # batch results and make it look frozen for ~chunksize seconds) and load
        # balances better when some images are slower to read than others.
        for ok in tqdm(pool.imap_unordered(worker, todo, chunksize=1), total=len(todo), desc=desc):
            if not ok:
                failures += 1
    if failures:
        print(f"[precompute] {desc}: {failures}/{len(todo)} images were unreadable and skipped")


def precompute_channel_cache(
    data_cfg: dict[str, Any], dfs: list[pd.DataFrame], desc: str = "channels",
    cpu_fraction: float | None = None, skip: bool = False,
) -> None:
    """Prebuild channels for every image in a ``path``-column dataframe (camchex/singleview)."""
    raw_paths: list[str] = []
    for df in dfs:
        if "path" in df.columns:
            raw_paths.extend(df["path"].tolist())
    precompute_channels_for_paths(data_cfg, raw_paths, desc, cpu_fraction=cpu_fraction, skip=skip)


def _prior_aware_image_paths(dfs: list[pd.DataFrame]) -> list[str]:
    """Flatten every current/prior image path out of prior-aware parquet frames."""
    raw_paths: list[str] = []
    for df in dfs:
        for col in ("img_paths", "prior_img_paths"):
            if col not in df.columns:
                continue
            for lst in df[col].tolist():
                if lst is None:
                    continue
                raw_paths.extend(str(p) for p in lst)
    return raw_paths


def dataloader_args_from_config(cfg: dict[str, Any], args: argparse.Namespace, shuffle: bool, for_eval: bool = False) -> dict[str, Any]:
    # Cap glibc arenas before any DataLoader (and its workers) is built. Idempotent,
    # so calling it on every loader build (train/val/eval) is cheap.
    cap_malloc_arenas(resolve_malloc_arena_max(args))
    dl_args = dict(cfg["data"]["dataloader_init_args"])
    if for_eval:
        dl_args["batch_size"] = resolve_val_batch_size(cfg, args)
    elif args.batch_size is not None:
        dl_args["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dl_args["num_workers"] = args.num_workers
    # Validation/eval uses its own worker count, defaulting to 0 (in-process) so it doesn't fork a
    # second pool on top of the persistent train workers (the mid-epoch RAM spike that OOM-kills).
    # CLI --val-num-workers or trainer.val_num_workers override; raise it to use val workers.
    if for_eval:
        dl_args["num_workers"] = int(resolve_trainer_arg(args, cfg, "val_num_workers", 0))
    if getattr(args, "prefetch_factor", None) is not None:
        dl_args["prefetch_factor"] = args.prefetch_factor
    if dl_args.get("num_workers", 0) == 0:
        dl_args["persistent_workers"] = False
        dl_args.pop("prefetch_factor", None)
    # Pick a shm-safe worker IPC strategy before this loader's workers fork.
    configure_dataloader_sharing(dl_args.get("num_workers", 0))
    dl_args["shuffle"] = shuffle
    return dl_args


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
    transforms_train, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    train_df = filter_single_view(read_dataframe(data_cfg["train_df_path"]), view_position)
    val_df = filter_single_view(read_dataframe(data_cfg["devel_df_path"]), view_position)
    precompute_channel_cache(data_cfg, [train_df, val_df], desc="singleview channels", cpu_fraction=resolve_cpu_fraction(args), skip=getattr(args, "skip_precompute", False))
    train_ds = SingleViewDataset(data_cfg, train_df, transforms_train)
    val_ds = SingleViewDataset(data_cfg, val_df, transforms_val)
    train_dl_args = dataloader_args_from_config(cfg, args, shuffle=True)
    val_dl_args = dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True)
    print(f"[dataloader] train: {train_dl_args}")
    print(f"[dataloader] val:   {val_dl_args}")
    train_loader = DataLoader(train_ds, **train_dl_args)
    val_loader = DataLoader(val_ds, **val_dl_args)
    return train_loader, val_loader


def make_camchex_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    transforms_train, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    tokenizer = AutoTokenizer.from_pretrained(
        data_cfg.get("tokenizer") or "dmis-lab/biobert-v1.1",
        trust_remote_code=True,
    )
    train_df = read_dataframe(data_cfg["train_df_path"])
    val_df = read_dataframe(data_cfg["devel_df_path"])
    precompute_channel_cache(data_cfg, [train_df, val_df], desc="camchex channels", cpu_fraction=resolve_cpu_fraction(args), skip=getattr(args, "skip_precompute", False))
    train_ds = CaMCheXDataset(data_cfg, train_df, transforms_train, tokenizer)
    val_ds = CaMCheXDataset(data_cfg, val_df, transforms_val, tokenizer)
    train_dl_args = dataloader_args_from_config(cfg, args, shuffle=True)
    val_dl_args = dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True)
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


def _blank_current_indication(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the CURRENT study's clinical indication emptied.

    Used for the report-ablation eval pass: the datasets turn an empty
    ``clinical_indication`` into the in-distribution "No clinical history
    available." placeholder (the same string they emit for genuinely-missing
    indications), so the report/indication carries no information while staying
    on the training distribution. Vitals and prior-study text are untouched.
    """
    out = df.copy()
    if "clinical_indication" in out.columns:
        out["clinical_indication"] = ""
    return out


def _blank_prior_aware_current_indication(ds) -> None:
    """In-place blank the CURRENT study's clinical indication on a PriorAwareDataset
    for the report-ablation pass. The dataset tokenizes / caches from ``clin_text`` at
    load time, so emptying that one column makes it emit the in-distribution
    "No clinical history available." placeholder. Prior-study streams (indication,
    vitals, report) and the current vitals are untouched.
    """
    if "clin_text" in ds.df.columns:
        ds.df["clin_text"] = ""


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
        cache_root=getattr(args, "text_embedding_cache_dir", None) or data_cfg.get("text_embedding_cache_dir", "../cache/text_embeddings"),
        batch_size=int(data_cfg.get("text_embedding_batch_size", 32) or 32),
        device=data_cfg.get("text_embedding_device", "auto"),
    )
    cache.ensure_texts([text for _, text in rows], max_length=384, desc=f"{text_model} clinical embeddings")
    cache.unload_model()
    data_cfg = dict(data_cfg)
    data_cfg["clinical_embedding_cache"] = cache
    return data_cfg


def maybe_add_prior_aware_text_embeddings(
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

    required = {"clin_text", "obs_text", "prior_clin_text", "prior_obs_text", "prior_report_text"}
    missing = sorted({col for df in dfs for col in required if col not in df.columns})
    if missing:
        raise KeyError(
            "Prior-aware training-time text embedding cache requires raw text columns "
            f"{sorted(required)}; missing {missing}. Rebuild parquet with "
            "src/prepare/04_build_prior_aware_dataset.py."
        )

    streams = set(data_cfg.get("text_embedding_streams") or required)
    unknown_streams = streams - required
    if unknown_streams:
        raise ValueError(f"Unknown prior-aware text_embedding_streams entries: {sorted(unknown_streams)}")

    clinical_texts: list[str] = []
    obs_texts: list[str] = []
    for df in dfs:
        if "clin_text" in streams:
            clinical_texts.extend(df["clin_text"].fillna("No clinical history available.").astype(str).tolist())
        if "prior_clin_text" in streams:
            clinical_texts.extend(df["prior_clin_text"].fillna("No clinical history available.").astype(str).tolist())
        if "obs_text" in streams:
            obs_texts.extend(df["obs_text"].fillna("").astype(str).tolist())
        if "prior_obs_text" in streams:
            obs_texts.extend(df["prior_obs_text"].fillna("").astype(str).tolist())
        if "prior_report_text" in streams:
            # Prior radiology report shares the clinical 384-token budget (CLS-pooled).
            clinical_texts.extend(
                df["prior_report_text"].fillna("No prior report available.").astype(str).tolist()
            )

    text_model = (
        getattr(args, "text_model", None)
        or model_cfg.get("text_model")
        or data_cfg.get("tokenizer")
        or "microsoft/BiomedVLP-CXR-BERT-specialized"
    )
    cache = TextEmbeddingCache(
        text_model=text_model,
        cache_root=getattr(args, "text_embedding_cache_dir", None) or data_cfg.get("text_embedding_cache_dir", "../cache/text_embeddings"),
        batch_size=int(data_cfg.get("text_embedding_batch_size", 32) or 32),
        device=data_cfg.get("text_embedding_device", "auto"),
    )
    if clinical_texts:
        cache.ensure_texts(list(dict.fromkeys(clinical_texts)), max_length=384, desc=f"{text_model} prior clinical embeddings")
    if obs_texts:
        cache.ensure_texts(list(dict.fromkeys(obs_texts)), max_length=128, desc=f"{text_model} prior observation embeddings")
    cache.unload_model()
    log_rss("text embeddings loaded to RAM (per-row dict cache before build_index_table)")
    data_cfg = dict(data_cfg)
    data_cfg["text_embedding_cache"] = cache
    return data_cfg


def _prior_aware_tokenizer(cfg: dict[str, Any], data_cfg: dict[str, Any], args: argparse.Namespace | None = None):
    """Tokenizer for PriorAwareDataset's load-time text encoding. The parquet stores
    raw text only, so the tokenizer comes from the training config (model.text_model /
    data.tokenizer), which is what lets one parquet serve any text model."""
    from transformers import AutoTokenizer

    text_model = (
        getattr(args, "text_model", None)
        or (cfg.get("model", {}) or {}).get("text_model")
        or data_cfg.get("tokenizer")
        or "dmis-lab/biobert-v1.1"
    )
    return AutoTokenizer.from_pretrained(text_model, trust_remote_code=True)


def make_camchex_vitals_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    uint8_pipeline = resolve_uint8_image_pipeline(args, data_cfg)
    data_cfg["uint8_image_pipeline"] = uint8_pipeline
    transforms_train, transforms_val = get_transforms(
        data_cfg["size"], data_cfg.get("channel_mode"), normalize_on_gpu=uint8_pipeline
    )
    train_df = read_dataframe(data_cfg["train_df_path"])
    val_df = read_dataframe(data_cfg["devel_df_path"])
    precompute_channel_cache(data_cfg, [train_df, val_df], desc="camchex_vitals channels", cpu_fraction=resolve_cpu_fraction(args), skip=getattr(args, "skip_precompute", False))
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
    val_dl_args = dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True)
    print(f"[dataloader] train: {train_dl_args}")
    print(f"[dataloader] val:   {val_dl_args}")
    train_loader = DataLoader(train_ds, **train_dl_args)
    val_loader = DataLoader(val_ds, **val_dl_args)
    return train_loader, val_loader


def make_prior_aware_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    """Build train/val loaders backed by the pre-generated prior-aware parquet."""
    data_cfg = data_cfg_from_config(cfg, args)
    uint8_pipeline = resolve_uint8_image_pipeline(args, data_cfg)
    data_cfg["uint8_image_pipeline"] = uint8_pipeline
    transforms_train, transforms_val = get_transforms(
        data_cfg["size"], data_cfg.get("channel_mode"), normalize_on_gpu=uint8_pipeline
    )
    label_dropout_p = float(data_cfg.get("label_dropout_p", 0.3))
    tokenizer = _prior_aware_tokenizer(cfg, data_cfg, args)

    train_ds = PriorAwareDataset(
        parquet_path=str(resolve_path(data_cfg["train_df_path"])),
        image_size=data_cfg["size"],
        transform=transforms_train,
        label_dropout_p=label_dropout_p,
        cfg=data_cfg,
        tokenizer=tokenizer,
    )
    val_ds = PriorAwareDataset(
        parquet_path=str(resolve_path(data_cfg["devel_df_path"])),
        image_size=data_cfg["size"],
        transform=transforms_val,
        label_dropout_p=0.0,
        cfg=data_cfg,
        tokenizer=tokenizer,
    )
    # Build the image channel cache BEFORE the text-embedding cache. The channel
    # prebuild is the more time-costly step, and it forks a multiprocessing pool --
    # running it first means the pool forks before the ~0.7GB text-embedding RAM dict
    # exists, so copy-on-write can't duplicate that dict across the pool workers.
    # (image paths don't depend on the text cache; the text columns are dropped after.)
    precompute_channels_for_paths(
        data_cfg, _prior_aware_image_paths([train_ds.df, val_ds.df]), desc="prior_aware channels",
        cpu_fraction=resolve_cpu_fraction(args),
    )
    data_cfg = maybe_add_prior_aware_text_embeddings(cfg, data_cfg, [train_ds.df, val_ds.df], args=args)
    train_ds.text_embedding_cache = data_cfg.get("text_embedding_cache")
    val_ds.text_embedding_cache = data_cfg.get("text_embedding_cache")
    for ds in (train_ds, val_ds):
        dropped = ds.drop_unused_text_columns()
    if dropped:
        print(f"[dataloader] dropped {dropped} unused raw-text column(s) from the parquet to save host RAM")
    train_dl_args = dataloader_args_from_config(cfg, args, shuffle=True)
    val_dl_args = dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True)
    print(f"[dataloader] train: {train_dl_args} (label_dropout_p={label_dropout_p})")
    print(f"[dataloader] val:   {val_dl_args}")
    train_loader = DataLoader(train_ds, **train_dl_args)
    val_loader = DataLoader(val_ds, **val_dl_args)
    log_rss("loaders built (parquet dfs + text-embedding RAM cache resident)")
    return train_loader, val_loader


def make_prior_aware_eval_loader(cfg: dict[str, Any], args: argparse.Namespace, drop_report: bool = False):
    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    ds = PriorAwareDataset(
        parquet_path=str(resolve_path(data_cfg["pred_df_path"])),
        image_size=data_cfg["size"],
        transform=transforms_val,
        label_dropout_p=0.0,
        cfg=data_cfg,
        tokenizer=_prior_aware_tokenizer(cfg, data_cfg, args),
    )
    if drop_report:
        _blank_prior_aware_current_indication(ds)
    data_cfg = maybe_add_prior_aware_text_embeddings(cfg, data_cfg, [ds.df], args=args)
    ds.text_embedding_cache = data_cfg.get("text_embedding_cache")
    ds.drop_unused_text_columns()
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True))
    labels_available = True  # label column is always present in the pregenerated parquet
    return loader, labels_available


def make_single_view_eval_loader(cfg: dict[str, Any], args: argparse.Namespace, view_position: str):
    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    df = filter_single_view(read_dataframe(data_cfg["pred_df_path"]), view_position)
    ds = SingleViewDataset(data_cfg, df, transforms_val)
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True))
    ids = df["path"].tolist() if "path" in df.columns else list(range(len(df)))
    labels_available = all(c in df.columns for c in data_cfg["classes"])
    return loader, ids, labels_available


def make_camchex_eval_loader(cfg: dict[str, Any], args: argparse.Namespace, drop_report: bool = False):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    tokenizer = AutoTokenizer.from_pretrained(
        data_cfg.get("tokenizer") or "dmis-lab/biobert-v1.1",
        trust_remote_code=True,
    )
    df = read_dataframe(data_cfg["pred_df_path"])
    if drop_report:
        df = _blank_current_indication(df)
    ds = CaMCheXDataset(data_cfg, df, transforms_val, tokenizer)
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True))
    labels_available = all(c in df.columns for c in data_cfg["classes"])
    return loader, labels_available


def make_camchex_vitals_eval_loader(cfg: dict[str, Any], args: argparse.Namespace, drop_report: bool = False):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    df = read_dataframe(data_cfg["pred_df_path"])
    if drop_report:
        df = _blank_current_indication(df)
    data_cfg = maybe_add_camchex_vitals_text_embeddings(cfg, data_cfg, [df], args=args)
    tokenizer = None
    if "clinical_embedding_cache" not in data_cfg and "clinical_embeddings" not in data_cfg:
        tokenizer = AutoTokenizer.from_pretrained(
            data_cfg.get("tokenizer") or "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )
    ds = CaMCheXVitalsDataset(data_cfg, df, transforms_val, tokenizer)
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True))
    labels_available = all(c in df.columns for c in data_cfg["classes"])
    return loader, labels_available
