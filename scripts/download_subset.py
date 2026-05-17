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
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-repo", default="tungnguyenlam/tung-thesis",
                   help="HuggingFace dataset repo (default: tungnguyenlam/tung-thesis)")
    p.add_argument("--archive-name", default="bundle-a3f9.7z",
                   help="Archive base name in the repo (default: bundle-a3f9.7z)")
    p.add_argument("--data-dir", default="data", help="Extraction target (default: data)")
    p.add_argument("--keep-archive", action="store_true",
                   help="Keep the downloaded .7z file or .7z.* parts after extraction")
    return p.parse_args()


def repo_archive_names(repo_files: list[str], archive_name: str) -> list[str]:
    volume_pattern = re.compile(rf"{re.escape(archive_name)}\.\d{{3}}")
    volume_names = sorted(name for name in repo_files if volume_pattern.fullmatch(name))
    if volume_names:
        return volume_names
    if archive_name in repo_files:
        return [archive_name]
    return []


def resolve_hf_repo_id(api, repo_id: str, token: str) -> str:
    if "/" in repo_id:
        return repo_id
    username = api.whoami(token=token)["name"]
    resolved = f"{username}/{repo_id}"
    print(f"Resolved HuggingFace repo: {repo_id} -> {resolved}")
    return resolved


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

    from huggingface_hub import HfApi, hf_hub_download

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    args.hf_repo = resolve_hf_repo_id(api, args.hf_repo, token)
    repo_files = api.list_repo_files(repo_id=args.hf_repo, repo_type="dataset", token=token)
    archive_names = repo_archive_names(repo_files, args.archive_name)
    if not archive_names:
        sys.exit(f"No archive named {args.archive_name} or split volumes {args.archive_name}.001, ... found")

    print(f"Downloading {len(archive_names)} archive file(s) from {args.hf_repo} ...")
    archive_paths = []
    for archive_name in archive_names:
        archive_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=archive_name,
            repo_type="dataset",
            token=token,
            local_dir=str(data_dir / "_bundles"),
        )
        archive_path = Path(archive_path).resolve()
        archive_paths.append(archive_path)
        print(f"  -> {archive_path} ({archive_path.stat().st_size / 1e9:.2f} GB)")

    print(f"Extracting into {data_dir} ...")
    subprocess.run(
        ["7z", "x", "-y", f"-p{password}", str(archive_paths[0])],
        cwd=str(data_dir),
        check=True,
    )

    if not args.keep_archive:
        for archive_path in archive_paths:
            archive_path.unlink(missing_ok=True)
        print("Removed downloaded archive file(s) (use --keep-archive to retain).")

    print("Done. Subset extracted at data/subset/, companion datasets at data/CXR-LT/ etc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
