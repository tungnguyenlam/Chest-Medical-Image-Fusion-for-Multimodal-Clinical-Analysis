from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.utils import _safe_decode_jpeg, image_cache_path, resolve_preferred_image_path


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predecode CXR JPEGs to RGB .npy arrays for optional dataloader caching.")
    parser.add_argument("--input-csv", action="append", required=True, help="CSV to read. Repeat for train/val/test.")
    parser.add_argument("--cache-dir", required=True, help="Directory where hashed .npy RGB arrays will be written.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    seen = set()
    for csv_path in args.input_csv:
        df = pd.read_csv(resolve_path(csv_path), usecols=["path"], low_memory=False)
        for raw_path in df["path"].dropna().astype(str):
            image_path = resolve_preferred_image_path(raw_path)
            if image_path in seen:
                continue
            seen.add(image_path)
            paths.append(image_path)

    written = 0
    skipped = 0
    failed = 0
    for image_path in tqdm(paths, desc="image cache"):
        out_path = image_cache_path(cache_dir, image_path)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        img = _safe_decode_jpeg(image_path)
        if img is None:
            failed += 1
            continue
        np.save(out_path, img.astype(np.uint8, copy=False))
        written += 1

    print(f"cache_dir={cache_dir} written={written} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
