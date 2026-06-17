#!/usr/bin/env python3
"""
Download the MIMIC subset bundle from a private HuggingFace dataset repo and
extract it into data/. Mirror of scripts/build_mimic_subset.py.

Run from the project root:
    python scripts/download_subset.py

Reads HF_TOKEN, HF_USERNAME, and DATA_PASSWORD from .env (see .env.example).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

from dotenv import load_dotenv


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"{name} not set in .env")
    return value


def resolve_7z_executable() -> str:
    """Prefer a Conda base p7zip wrapper when env-local 7zip lacks 7z.so."""
    override = os.environ.get("MIMIC_BUNDLE_7Z", "").strip()
    if override:
        return override

    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        prefix = Path(conda_prefix)
        if prefix.parent.name == "envs":
            candidates.append(prefix.parent.parent / "bin" / "7z")
        candidates.append(prefix / "bin" / "7z")

    which_7z = shutil.which("7z")
    if which_7z:
        candidates.append(Path(which_7z))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        return str(candidate)

    return "7z"


def _workers_from_cpu_fraction(fraction: float) -> int:
    """Visible CPU cores multiplied by fraction, floored at 1."""
    return max(1, int(cpu_count() * fraction))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-repo", default="tung-thesis",
                   help="HuggingFace dataset repo. Bare names use HF_USERNAME from .env "
                        "(default: tung-thesis)")
    p.add_argument("--archive-name", default="bundle-a3f9.7z",
                   help="Archive base name in the repo (default: bundle-a3f9.7z)")
    p.add_argument("--data-dir", default="data", help="Extraction target (default: data)")
    p.add_argument("--keep-archive", action="store_true",
                   help="Keep the downloaded .7z file or .7z.* parts after extraction")
    p.add_argument("--cpu-fraction", type=float, default=0.5,
                   help="Fraction of visible CPU cores used for default worker/thread counts "
                        "(default: 0.5)")
    p.add_argument("--download-workers", type=int, default=None,
                   help="Number of archive files to download in parallel "
                        "(default: cpu_count() * --cpu-fraction)")
    p.add_argument("--extract-threads", type=int, default=None,
                   help="Number of CPU threads for 7z extraction "
                        "(default: cpu_count() * --cpu-fraction)")
    p.add_argument("--extract-workers", type=int, default=None,
                   help="Number of independent shard archives to extract in parallel "
                        "(default: cpu_count() * --cpu-fraction)")
    args = p.parse_args()
    if not 0 < args.cpu_fraction <= 1:
        p.error("--cpu-fraction must be > 0 and <= 1")
    default_workers = _workers_from_cpu_fraction(args.cpu_fraction)
    if args.download_workers is None:
        args.download_workers = default_workers
    if args.extract_threads is None:
        args.extract_threads = default_workers
    if args.extract_workers is None:
        args.extract_workers = default_workers
    if args.download_workers < 1:
        p.error("--download-workers must be >= 1")
    if args.extract_threads < 1:
        p.error("--extract-threads must be >= 1")
    if args.extract_workers < 1:
        p.error("--extract-workers must be >= 1")
    return args


def repo_archive_names(repo_files: list[str], archive_name: str) -> list[str]:
    volume_pattern = re.compile(rf"{re.escape(archive_name)}\.\d{{3}}")
    volume_names = sorted(name for name in repo_files if volume_pattern.fullmatch(name))
    if volume_names:
        return volume_names
    if archive_name in repo_files:
        return [archive_name]
    return []


def repo_opaque_shard_groups(repo_files: list[str], archive_name: str) -> list[list[str]]:
    prefix = Path(archive_name).stem
    single_pattern = re.compile(rf"{re.escape(prefix)}\.(\d{{4}})\.7z")
    volume_pattern = re.compile(rf"{re.escape(prefix)}\.(\d{{4}})\.7z\.(\d{{3}})")
    singles: dict[str, str] = {}
    volumes: dict[str, list[str]] = {}

    for name in repo_files:
        volume_match = volume_pattern.fullmatch(name)
        if volume_match:
            volumes.setdefault(volume_match.group(1), []).append(name)
            continue
        single_match = single_pattern.fullmatch(name)
        if single_match:
            singles[single_match.group(1)] = name

    groups: list[list[str]] = []
    for shard_id in sorted(set(singles) | set(volumes)):
        if shard_id in volumes:
            groups.append(sorted(volumes[shard_id]))
        else:
            groups.append([singles[shard_id]])
    return groups


def resolve_hf_repo_id(repo_id: str, username: str) -> str:
    if "/" in repo_id:
        return repo_id
    if not username:
        sys.exit("HF_USERNAME not set in .env; set it or pass --hf-repo owner/name")
    resolved = f"{username}/{repo_id}"
    print(f"Resolved HuggingFace repo: {repo_id} -> {resolved}")
    return resolved


def download_archive(
    archive_name: str,
    *,
    repo_id: str,
    token: str,
    local_dir: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"  starting download: {archive_name}", flush=True)
    archive_path = hf_hub_download(
        repo_id=repo_id,
        filename=archive_name,
        repo_type="dataset",
        token=token,
        local_dir=str(local_dir),
    )
    return Path(archive_path).resolve()


def download_archives(
    archive_names: list[str],
    *,
    repo_id: str,
    token: str,
    local_dir: Path,
    download_workers: int,
) -> list[Path]:
    if download_workers < 1:
        sys.exit("--download-workers must be >= 1")

    archive_paths_by_name: dict[str, Path] = {}
    worker_count = min(download_workers, len(archive_names))
    if worker_count == 1:
        for archive_name in archive_names:
            archive_path = download_archive(
                archive_name,
                repo_id=repo_id,
                token=token,
                local_dir=local_dir,
            )
            archive_paths_by_name[archive_name] = archive_path
            print(f"  -> {archive_path} ({archive_path.stat().st_size / 1e9:.2f} GB)")
        return [archive_paths_by_name[name] for name in archive_names]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                download_archive,
                archive_name,
                repo_id=repo_id,
                token=token,
                local_dir=local_dir,
            ): archive_name
            for archive_name in archive_names
        }
        for future in as_completed(futures):
            archive_name = futures[future]
            archive_path = future.result()
            archive_paths_by_name[archive_name] = archive_path
            print(f"  -> {archive_path} ({archive_path.stat().st_size / 1e9:.2f} GB)")

    return [archive_paths_by_name[name] for name in archive_names]


def extract_archive(archive_path: Path, *, data_dir: Path, password: str, extract_threads: int) -> None:
    if extract_threads < 1:
        sys.exit("--extract-threads must be >= 1")

    print(f"Extracting into {data_dir} with {extract_threads} thread(s) ...")
    seven_zip = resolve_7z_executable()
    print(f"  $ {seven_zip} x -y -p<DATA_PASSWORD> -mmt={extract_threads} {archive_path}", flush=True)
    subprocess.run(
        [seven_zip, "x", "-y", f"-p{password}", f"-mmt={extract_threads}", str(archive_path)],
        cwd=str(data_dir),
        check=True,
    )


def extract_independent_archives(
    archive_paths: list[Path],
    *,
    data_dir: Path,
    password: str,
    extract_threads: int,
    extract_workers: int,
) -> None:
    if extract_workers < 1:
        sys.exit("--extract-workers must be >= 1")

    worker_count = min(extract_workers, len(archive_paths))
    print(f"Extracting {len(archive_paths)} independent archive(s) with {worker_count} worker(s) ...")
    if worker_count == 1:
        for archive_path in archive_paths:
            extract_archive(
                archive_path,
                data_dir=data_dir,
                password=password,
                extract_threads=extract_threads,
            )
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                extract_archive,
                archive_path,
                data_dir=data_dir,
                password=password,
                extract_threads=extract_threads,
            )
            for archive_path in archive_paths
        ]
        for future in as_completed(futures):
            future.result()


def main() -> int:
    args = parse_args()
    load_dotenv()

    if not Path("camchex").is_dir():
        sys.exit("Run from project root.")

    token = require_env("HF_TOKEN")
    username = os.environ.get("HF_USERNAME", "").strip()
    password = require_env("DATA_PASSWORD")

    from huggingface_hub import HfApi

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    args.hf_repo = resolve_hf_repo_id(args.hf_repo, username)
    repo_files = api.list_repo_files(repo_id=args.hf_repo, repo_type="dataset", token=token)
    opaque_shard_groups = repo_opaque_shard_groups(repo_files, args.archive_name)
    if opaque_shard_groups:
        archive_names = [name for group in opaque_shard_groups for name in group]
        layout = "opaque-shards"
    else:
        archive_names = repo_archive_names(repo_files, args.archive_name)
        layout = "single"
    if not archive_names:
        prefix = Path(args.archive_name).stem
        sys.exit(
            f"No archive named {args.archive_name}, split volumes {args.archive_name}.001, "
            f"or opaque shards {prefix}.0001.7z found"
        )

    worker_count = min(args.download_workers, len(archive_names))
    print(
        f"Downloading {len(archive_names)} archive file(s) ({layout}) from {args.hf_repo} "
        f"with {worker_count} worker(s) ..."
    )
    archive_paths = download_archives(
        archive_names,
        repo_id=args.hf_repo,
        token=token,
        local_dir=data_dir / "_bundles",
        download_workers=args.download_workers,
    )

    if layout == "opaque-shards":
        archive_paths_by_name = dict(zip(archive_names, archive_paths, strict=True))
        shard_entry_paths = [archive_paths_by_name[group[0]] for group in opaque_shard_groups]
        extract_independent_archives(
            shard_entry_paths,
            data_dir=data_dir,
            password=password,
            extract_threads=args.extract_threads,
            extract_workers=args.extract_workers,
        )
    else:
        extract_archive(
            archive_paths[0],
            data_dir=data_dir,
            password=password,
            extract_threads=args.extract_threads,
        )

    if not args.keep_archive:
        for archive_path in archive_paths:
            archive_path.unlink(missing_ok=True)
        print("Removed downloaded archive file(s) (use --keep-archive to retain).")

    print("Done. Subset extracted at data/subset/, companion datasets at data/CXR-LT/ etc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
