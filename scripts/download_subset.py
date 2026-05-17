#!/usr/bin/env python3
"""
Download the MIMIC subset bundle from a private HuggingFace dataset repo and
extract it into data/. Mirror of scripts/build_mimic_subset.py.

Run from the project root:
    python scripts/download_subset.py

Reads HF_TOKEN and DATA_PASSWORD from .env (see .env.example).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-repo", default="tung-thesis",
                   help="HuggingFace dataset repo (default: tung-thesis)")
    p.add_argument("--archive-name", default="bundle-a3f9.7z",
                   help="Archive file name in the repo (default: bundle-a3f9.7z)")
    p.add_argument("--data-dir", default="data", help="Extraction target (default: data)")
    p.add_argument("--keep-archive", action="store_true",
                   help="Keep the downloaded .7z after extraction")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    if not Path("camchex").is_dir():
        sys.exit("Run from project root.")

    token = os.environ.get("HF_TOKEN")
    password = os.environ.get("DATA_PASSWORD")
    if not token:
        sys.exit("HF_TOKEN not set in .env")
    if not password:
        sys.exit("DATA_PASSWORD not set in .env")

    from huggingface_hub import hf_hub_download

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.archive_name} from {args.hf_repo} ...")
    archive_path = hf_hub_download(
        repo_id=args.hf_repo,
        filename=args.archive_name,
        repo_type="dataset",
        token=token,
        local_dir=str(data_dir / "_bundles"),
    )
    archive_path = Path(archive_path).resolve()
    print(f"  -> {archive_path} ({archive_path.stat().st_size / 1e9:.2f} GB)")

    print(f"Extracting into {data_dir} ...")
    subprocess.run(
        ["7z", "x", "-y", f"-p{password}", str(archive_path)],
        cwd=str(data_dir),
        check=True,
    )

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)
        print(f"Removed {archive_path.name} (use --keep-archive to retain).")

    print("Done. Subset extracted at data/subset/, companion datasets at data/CXR-LT/ etc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
