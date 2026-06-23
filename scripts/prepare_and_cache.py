#!/usr/bin/env python3
"""One-shot CPU-bound prepare + channel-cache builder.

Runs the full ``src/prepare/0{1,2,3,4}_*.py`` pipeline (defaulting to the
``full`` MIMIC source) AND warms the deterministic 3-channel image cache that
training reads, in a single process. The win is *decode-once*: stage 03's
image verification and the training-time channel precompute each decode every
JPEG independently. Here, stage 03 runs existence-only (cheap ``stat``) and the
single expensive JPEG decode is moved into the channel-build pass, which both
caches the channels and detects corrupt/unreadable files in one go.

Pipeline (run from the project root):

    01_make_dataset --subset <subset>          (report parse, ED/vitals merge)
    02_split_dataset                            (re-derive train/dev/test)
    03_filter_existing_images --no-verify-images (existence-only; drops MISSING)
    [FUSED]  resolve -> decode -> build+cache raw_clahe_histeq -> record CORRUPT
    [PRUNE]  drop corrupt rows from 03_mimic_*.csv (+ *_corrupt.txt)
    04_build_prior_aware_dataset (build_all_variants)  -> prior_aware parquets

The channel cache is keyed on ``resolve_preferred_image_path(path)`` -- the
exact string stage 04 stores in the parquet and that training passes to
``load_or_build_channels`` -- so the files written here are cache *hits* at
train time (no rebuild). ``--cache-dir``/``--size``/``--channel-mode`` must
therefore match the training config (defaults already do).

Examples::

    python scripts/prepare_and_cache.py                 # full pipeline + cache
    python scripts/prepare_and_cache.py --skip-prepare  # only warm the cache
    python scripts/prepare_and_cache.py --skip-precompute  # only run 01-04
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.image_channel_preprocessing import CHANNEL_MODES, describe_mode
from src.dataloader.utils import (
    channel_cache_path,
    load_or_build_channels,
    make_preprocess_config,
    resolve_preferred_image_path,
)

DATA_CAMCHEX_ROOT = ROOT / "data" / "data-camchex"
# Stage 03 names its outputs 03_<source>_<split>.csv. For --subset full/kaggle the
# source tag is "mimic"; for --subset subset it is also "mimic" (see 03's
# _SOURCE_BY_SUBSET). kaggle is the only one that emits a "kaggle" tag.
_SOURCE_TAG = {"full": "mimic", "subset": "mimic", "kaggle": "kaggle"}
SPLITS = ["train", "development", "test"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--subset", default="full", choices=["full", "subset", "kaggle"],
        help="MIMIC source passed to stages 01/03. Default: full.",
    )
    p.add_argument(
        "--cpu-fraction", type=float, default=0.5,
        help="Fraction of CPU cores to use for every parallel stage. Default: 0.5 "
             "(half the cores, matching src/prepare/01_make_dataset.py).",
    )
    p.add_argument(
        "--channel-mode", default="raw_clahe_histeq", choices=sorted(CHANNEL_MODES),
        help="3-channel mode to precompute. Must match the training config "
             "(default raw_clahe_histeq: ch0=raw, ch1=mild CLAHE, ch2=histeq).",
    )
    p.add_argument(
        "--size", type=int, default=512,
        help="Output square size baked into the cache key. Must match data.size. Default: 512.",
    )
    p.add_argument(
        "--cache-dir", default="../cache/channels",
        help="Channel cache root (per-mode shards created under it). Resolved "
             "relative to the CWD, so run from the repo root exactly like training. "
             "Default ../cache/channels matches the training configs.",
    )
    p.add_argument("--skip-prepare", action="store_true", help="Skip stages 01-04; only warm the channel cache.")
    p.add_argument("--skip-precompute", action="store_true", help="Skip the channel-cache build; only run stages 01-04.")
    p.add_argument("--overwrite", action="store_true", help="Rebuild every channel cache entry even if present.")
    args = p.parse_args()
    args.workers = max(1, int(cpu_count() * args.cpu_fraction))
    return args


# ---- stages 01-03 (subprocess) ---------------------------------------------

def run_prepare_01_to_03(subset: str, workers: int) -> None:
    """Run stages 01, 02, 03 as subprocesses from the repo root. Stage 03 is
    existence-only (--no-verify-images): decoding is deferred to the fused pass."""
    stages = [
        [sys.executable, "src/prepare/01_make_dataset.py", "--subset", subset],
        [sys.executable, "src/prepare/02_split_dataset.py"],
        [sys.executable, "src/prepare/03_filter_existing_images.py",
         "--subset", subset, "--no-verify-images", "--workers", str(workers)],
    ]
    for cmd in stages:
        print(f"\n[prepare] $ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            raise SystemExit(f"[prepare] stage failed (exit {result.returncode}): {' '.join(cmd)}")


# ---- fused decode-once + channel build -------------------------------------

def _csv_paths(source_tag: str) -> list[Path]:
    return [DATA_CAMCHEX_ROOT / f"03_{source_tag}_{split}.csv" for split in SPLITS]


def _collect_unique_raw_paths(csv_paths: list[Path]) -> list[str]:
    """Every distinct ``path`` string across the stage-03 split CSVs (no I/O)."""
    seen: dict[str, None] = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"[precompute] skipping missing CSV: {csv_path}", flush=True)
            continue
        df = pd.read_csv(csv_path, usecols=["path"], low_memory=False)
        for raw in df["path"].dropna().astype(str):
            seen.setdefault(raw, None)
    return list(seen)


def _parallel_resolve(raw_paths: list[str], workers: int) -> dict[str, str]:
    """Resolve every raw path to the preferred on-disk string, in parallel.

    ``resolve_preferred_image_path`` is filesystem ``stat`` (GIL-releasing), so
    threads parallelize it on slow mounts, and it is lru_cached -- which warms the
    cache that stage 04 reuses for its own path resolution later in this process.
    """
    if not raw_paths:
        return {}
    if workers <= 1 or len(raw_paths) <= 1:
        return {p: resolve_preferred_image_path(p)
                for p in tqdm(raw_paths, desc="resolve paths", dynamic_ncols=True)}
    chunksize = max(1, min(256, len(raw_paths) // (workers * 4) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        resolved = list(tqdm(
            ex.map(resolve_preferred_image_path, raw_paths, chunksize=chunksize),
            total=len(raw_paths), desc="resolve paths", dynamic_ncols=True,
        ))
    return dict(zip(raw_paths, resolved))


def _build_one(resolved_path: str, mode: str, cfg, cache_dir: str) -> tuple[str, bool]:
    """Pool worker: build+cache one image's channels (decode-once).

    Returns ``(resolved_path, ok)``; ``ok`` is False when the JPEG is missing or
    corrupt (decode failed). We return a bool, never the array, so the worker keeps
    the 512x512x3 buffer to itself instead of pickling it back to the parent.
    """
    arr = load_or_build_channels(resolved_path, mode, cfg, cache_dir)
    return resolved_path, arr is not None


def precompute_channels(
    args: argparse.Namespace, source_tag: str
) -> tuple[set[str], dict[str, int]]:
    """Decode-once channel cache build. Returns (corrupt_raw_paths, stats).

    ``corrupt_raw_paths`` are stage-03 CSV ``path`` strings whose image failed to
    decode, so the caller can prune them before stage 04.
    """
    cfg = make_preprocess_config({"size": args.size})
    mode = args.channel_mode
    cache_dir = args.cache_dir
    print(f"[channels] {describe_mode(mode, cfg)}", flush=True)

    csv_paths = _csv_paths(source_tag)
    unique_raw = _collect_unique_raw_paths(csv_paths)
    print(f"[precompute] {len(unique_raw)} unique image paths across {len(csv_paths)} split CSVs", flush=True)

    resolved_map = _parallel_resolve(unique_raw, args.workers)
    # Reverse map resolved -> raw(s): several raw strings can resolve to the same
    # on-disk file, so a corrupt file must mark every raw that points at it.
    resolved_to_raw: dict[str, list[str]] = {}
    for raw, resolved in resolved_map.items():
        resolved_to_raw.setdefault(resolved, []).append(raw)
    unique_resolved = list(resolved_to_raw)

    # Existing cache files: one directory listing of the mode shard. The cache key
    # is derived from the (resolved) path string alone, so the miss scan is pure CPU.
    existing_digests: set[str] = set()
    if not args.overwrite:
        try:
            with os.scandir(Path(cache_dir) / mode) as it:
                for entry in it:
                    if entry.name.endswith(".npy"):
                        existing_digests.add(entry.name[:-4])
        except FileNotFoundError:
            pass

    todo = [
        r for r in unique_resolved
        if channel_cache_path(cache_dir, r, mode, cfg).stem not in existing_digests
    ]
    skipped = len(unique_resolved) - len(todo)
    corrupt_raw: set[str] = set()

    if not todo:
        print(f"[precompute] all {len(unique_resolved)} images already cached in {cache_dir}/{mode}", flush=True)
        return corrupt_raw, {"built": 0, "skipped": skipped, "corrupt": 0}

    print(
        f"[precompute] building {len(todo)}/{len(unique_resolved)} channel images "
        f"(mode={mode}, size={args.size}) with {args.workers} workers -> {cache_dir} "
        f"({skipped} already cached)",
        flush=True,
    )
    worker = functools.partial(_build_one, mode=mode, cfg=cfg, cache_dir=cache_dir)
    try:
        pool_cls = multiprocessing.get_context("fork").Pool
    except ValueError:
        from multiprocessing.pool import ThreadPool as pool_cls
        print("[precompute] 'fork' start method unavailable; using ThreadPool fallback.", flush=True)

    built = 0
    with pool_cls(args.workers) as pool:
        for resolved, ok in tqdm(
            pool.imap_unordered(worker, todo, chunksize=1),
            total=len(todo), desc="channels", dynamic_ncols=True,
        ):
            if ok:
                built += 1
            else:
                corrupt_raw.update(resolved_to_raw.get(resolved, []))

    if corrupt_raw:
        print(f"[precompute] {len(corrupt_raw)} raw path(s) were unreadable/corrupt and will be pruned", flush=True)
    return corrupt_raw, {"built": built, "skipped": skipped, "corrupt": len(corrupt_raw)}


# ---- prune corrupt rows from the stage-03 CSVs -----------------------------

def prune_corrupt_rows(source_tag: str, corrupt_raw: set[str]) -> None:
    """Drop rows whose ``path`` is corrupt from each 03 CSV, and record the
    corrupt paths to 03_<tag>_<split>_corrupt.txt (mirrors stage 03)."""
    if not corrupt_raw:
        return
    for split in SPLITS:
        csv_path = DATA_CAMCHEX_ROOT / f"03_{source_tag}_{split}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        mask = df["path"].astype(str).isin(corrupt_raw)
        n_corrupt = int(mask.sum())
        if not n_corrupt:
            continue
        df[~mask].to_csv(csv_path, index=False)
        corrupt_txt = DATA_CAMCHEX_ROOT / f"03_{source_tag}_{split}_corrupt.txt"
        corrupt_txt.write_text("\n".join(df.loc[mask, "path"].astype(str)) + "\n")
        print(f"[prune] {split}: dropped {n_corrupt} corrupt row(s) -> {csv_path.name} (+{corrupt_txt.name})", flush=True)


# ---- stage 04 (in-process import) ------------------------------------------

def run_stage_04(workers: int) -> None:
    """Import 04_build_prior_aware_dataset and run build_all_variants in-process,
    so the resolve lru_cache warmed by the fused step turns 04's path resolution
    into dict hits. The module name starts with a digit, so load it by file path."""
    mod_path = ROOT / "src" / "prepare" / "04_build_prior_aware_dataset.py"
    spec = importlib.util.spec_from_file_location("build_prior_aware_dataset", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stage_args = argparse.Namespace(
        in_dir="data/data-camchex",
        out_dir="data/data-camchex",
        data_root="data",
        splits=list(SPLITS),
        in_prefix="03_mimic_",
        out_prefix="prior_aware_",
        cxr_lt_version=None,
        cxr_lt_label_set="auto",
        workers=workers,
        store_raw_paths=False,
    )
    print("\n[stage04] building prior-aware parquets (all CXR-LT variants)...", flush=True)
    mod.build_all_variants(DATA_CAMCHEX_ROOT, DATA_CAMCHEX_ROOT, stage_args)


# ---- key-parity sanity check -----------------------------------------------

def verify_cache_parity(args: argparse.Namespace) -> None:
    """Confirm a channel file exists under cache-dir/<mode> for the first image
    path stored in a freshly-built parquet -- i.e. training will get a cache hit."""
    parquet = DATA_CAMCHEX_ROOT / "prior_aware_train.parquet"
    if not parquet.exists():
        print(f"[verify] no parquet at {parquet}; skipping parity check", flush=True)
        return
    df = pd.read_parquet(parquet, columns=["img_paths"])
    sample = None
    for lst in df["img_paths"].tolist():
        if lst is not None and len(lst) > 0:
            sample = str(lst[0])
            break
    if sample is None:
        print("[verify] no image paths in parquet; skipping parity check", flush=True)
        return
    cfg = make_preprocess_config({"size": args.size})
    cpath = channel_cache_path(args.cache_dir, sample, args.channel_mode, cfg)
    status = "HIT" if cpath.exists() else "MISS"
    print(f"[verify] parquet path -> cache {status}: {cpath}", flush=True)
    if status == "MISS":
        print("[verify] WARNING: cache key mismatch -- training would rebuild. "
              "Check --cache-dir/--size/--channel-mode match the training config.", flush=True)


def main() -> int:
    args = parse_args()
    if not (ROOT / "data").is_dir() or not (ROOT / "src").is_dir():
        sys.exit("Run from the project root (data/ and src/ must exist).")
    print(f"[prepare_and_cache] subset={args.subset} workers={args.workers} "
          f"(cpu_fraction={args.cpu_fraction}) mode={args.channel_mode} size={args.size} "
          f"cache_dir={args.cache_dir}", flush=True)

    source_tag = _SOURCE_TAG[args.subset]

    if not args.skip_prepare:
        run_prepare_01_to_03(args.subset, args.workers)

    stats = {"built": 0, "skipped": 0, "corrupt": 0}
    if not args.skip_precompute:
        corrupt_raw, stats = precompute_channels(args, source_tag)
        if not args.skip_prepare:
            prune_corrupt_rows(source_tag, corrupt_raw)
        elif corrupt_raw:
            print(f"[prune] --skip-prepare set; not rewriting CSVs ({len(corrupt_raw)} corrupt paths found)", flush=True)

    if not args.skip_prepare:
        run_stage_04(args.workers)
        verify_cache_parity(args)

    print(
        f"\n[done] channels built={stats['built']} skipped(cached)={stats['skipped']} "
        f"corrupt={stats['corrupt']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
