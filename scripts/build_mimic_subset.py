#!/usr/bin/env python3
"""
Build a deterministic patient-level subset of MIMIC-CXR-JPG + MIMIC-CXR reports,
bundle it with CXR-LT and MIMIC-IV-ED-2-2 into a password-protected 7z archive,
and optionally upload to a private HuggingFace dataset repo.

Run from the project root:
    python scripts/build_mimic_subset.py
    python scripts/build_mimic_subset.py --fraction 0.05 --skip-upload
    python scripts/build_mimic_subset.py --skip-copy --compression-level 0 --archive-threads 8
    python scripts/build_mimic_subset.py --cpu-fraction 0.7
    python scripts/build_mimic_subset.py --skip-copy --volume-size 10g
    python scripts/build_mimic_subset.py --upload-existing

Reads HF_TOKEN, HF_USERNAME, and DATA_PASSWORD from .env (see .env.example).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path


def _workers_from_cpu_fraction(fraction: float) -> int:
    """Visible CPU cores multiplied by fraction, floored at 1."""
    return max(1, int(cpu_count() * fraction))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="Project data/ directory (default: data)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for patient sampling (default: 42)")
    p.add_argument("--fraction", type=float, default=0.1, help="Patient fraction to sample (default: 0.1)")
    p.add_argument("--subset-name", default=None,
                   help="Subset folder name under data/ (default: derived as 'subset' for the canonical 10%% bundle, "
                        "otherwise 'subset_seed{seed}_{pct}pct')")
    p.add_argument("--archive-name", default="bundle-a3f9.7z",
                   help="Output archive filename (default: bundle-a3f9.7z)")
    p.add_argument("--archive-dir", default="data/_bundles",
                   help="Directory to write the archive to (default: data/_bundles)")
    p.add_argument("--hf-repo", default="tung-thesis",
                   help="HuggingFace dataset repo. Bare names use HF_USERNAME from .env "
                        "(default: tung-thesis)")
    p.add_argument("--public", action="store_true",
                   help="Create the HF repo as public. DEFAULT is private. "
                        "Public re-hosting of MIMIC violates the PhysioNet DUA — "
                        "only pass this for sanitized or non-credentialed bundles.")
    p.add_argument("--cpu-fraction", type=float, default=0.5,
                   help="Fraction of visible CPU cores used for default worker/thread counts "
                        "(default: 0.5)")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel copy workers (default: cpu_count() * --cpu-fraction)")
    p.add_argument("--archive-threads", type=int, default=None,
                   help="7z compression threads (default: cpu_count() * --cpu-fraction)")
    p.add_argument("--compression-level", type=int, choices=range(10), default=0,
                   metavar="{0..9}",
                   help="7z compression level: 0=store only, 1=fastest, 9=ultra (default: 0)")
    p.add_argument("--volume-size", default="10g",
                   help="Split 7z archive into volumes of this size, e.g. 5g, 10g, 500m (default: 10g)")
    p.add_argument("--single-archive", action="store_true",
                   help="Disable 7z volume splitting and write one archive file")
    p.add_argument("--preserve-hf-history", action="store_true",
                   help="Do not delete/recreate the HF dataset repo before upload")
    p.add_argument("--hf-use-xet", action="store_true",
                   help="Allow HuggingFace's Xet transfer backend. Default disables it because "
                        "large WSL/server uploads have been observed getting killed mid-transfer.")
    p.add_argument("--upload-existing", action="store_true",
                   help="Skip copy/archive and upload existing archive file(s) from --archive-dir")
    p.add_argument("--skip-copy", action="store_true", help="Reuse existing data/<subset>/ tree")
    p.add_argument("--skip-archive", action="store_true", help="Skip 7z step")
    p.add_argument("--skip-upload", action="store_true", help="Skip HuggingFace upload")
    p.add_argument("--dry-run", action="store_true", help="Sample + print stats, then exit")
    args = p.parse_args()
    if not 0 < args.cpu_fraction <= 1:
        p.error("--cpu-fraction must be > 0 and <= 1")
    default_workers = _workers_from_cpu_fraction(args.cpu_fraction)
    if args.workers is None:
        args.workers = default_workers
    if args.archive_threads is None:
        args.archive_threads = default_workers
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.archive_threads < 1:
        p.error("--archive-threads must be >= 1")
    return args


def resolve_subset_name(args: argparse.Namespace) -> str:
    if args.subset_name:
        return args.subset_name
    if args.seed == 42 and abs(args.fraction - 0.1) < 1e-9:
        return "subset"
    return f"subset_seed{args.seed}_{int(round(args.fraction * 100))}pct"


def sample_patients(split_csv: Path, seed: int, fraction: float) -> tuple[set[int], pd.DataFrame]:
    df = pd.read_csv(split_csv)
    patients = np.sort(df["subject_id"].unique())
    rng = np.random.default_rng(seed)
    n = int(round(len(patients) * fraction))
    chosen = set(rng.choice(patients, size=n, replace=False).tolist())
    sub_df = df[df["subject_id"].isin(chosen)].reset_index(drop=True)
    return chosen, sub_df


def jpg_src_path(jpg_root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    s = str(subject_id)
    return jpg_root / f"p{s[:2]}" / f"p{s}" / f"s{study_id}" / f"{dicom_id}.jpg"


def jpg_dst_path(jpg_dst_root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    s = str(subject_id)
    return jpg_dst_root / f"p{s[:2]}" / f"p{s}" / f"s{study_id}" / f"{dicom_id}.jpg"


def txt_src_path(txt_root: Path, subject_id: int, study_id: int) -> Path:
    s = str(subject_id)
    return txt_root / f"p{s[:2]}" / f"p{s}" / f"s{study_id}.txt"


def txt_dst_path(txt_dst_root: Path, subject_id: int, study_id: int) -> Path:
    s = str(subject_id)
    return txt_dst_root / f"p{s[:2]}" / f"p{s}" / f"s{study_id}.txt"


def copy_file(src: Path, dst: Path) -> tuple[bool, int]:
    if not src.exists():
        return False, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst, follow_symlinks=True)
    return True, dst.stat().st_size


def parallel_copy(pairs: list[tuple[Path, Path]], workers: int, desc: str) -> tuple[int, int, int]:
    ok = missing = total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(copy_file, s, d) for s, d in pairs]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            success, nbytes = f.result()
            if success:
                ok += 1
                total_bytes += nbytes
            else:
                missing += 1
    return ok, missing, total_bytes


def patient_hash(patients: set[int]) -> str:
    h = hashlib.sha256()
    for sid in sorted(patients):
        h.update(str(sid).encode())
        h.update(b",")
    return h.hexdigest()


def run_7z(
    archive: Path,
    sources: list[Path],
    password: str,
    workdir: Path,
    compression_level: int,
    archive_threads: int,
    volume_size: str | None,
) -> None:
    workdir = workdir.absolute()
    rels = []
    for source in sources:
        source = source if source.is_absolute() else Path.cwd() / source
        rels.append(str(source.absolute().relative_to(workdir)))
    volume_args = [f"-v{volume_size}"] if volume_size else []
    cmd = [
        "7z",
        "a",
        "-t7z",
        "-mhe=on",
        f"-p{password}",
        f"-mx={compression_level}",
        f"-mmt={archive_threads}",
        *volume_args,
        str(archive),
        *rels,
    ]
    volume_display = f" -v{volume_size}" if volume_size else ""
    print(
        "  $ 7z a -t7z -mhe=on -p<DATA_PASSWORD> "
        f"-mx={compression_level} -mmt={archive_threads}{volume_display} "
        f"{archive.name} {' '.join(rels)}"
    )
    subprocess.run(cmd, cwd=str(workdir), check=True)


def archive_outputs(archive: Path, volume_size: str | None) -> list[Path]:
    if volume_size:
        return sorted(archive.parent.glob(f"{archive.name}.[0-9][0-9][0-9]"))
    if archive.exists():
        return [archive]
    return []


def discover_existing_archives(archive: Path) -> list[Path]:
    parts = sorted(archive.parent.glob(f"{archive.name}.[0-9][0-9][0-9]"))
    if parts:
        return parts
    if archive.exists():
        return [archive]
    return []


def print_archive_summary(archives: list[Path]) -> None:
    total_size = sum(path.stat().st_size for path in archives)
    if len(archives) == 1:
        print(f"  archive size: {total_size / 1e9:.2f} GB")
    else:
        print(f"  archive parts: {len(archives)} / total size: {total_size / 1e9:.2f} GB")


def remove_archive_outputs(archive: Path) -> None:
    archive.unlink(missing_ok=True)
    for part in archive.parent.glob(f"{archive.name}.[0-9][0-9][0-9]"):
        part.unlink()


def resolve_hf_repo_id(repo_id: str) -> str:
    if "/" in repo_id:
        return repo_id
    username = os.environ.get("HF_USERNAME", "").strip()
    if not username:
        sys.exit("HF_USERNAME not set in .env; set it or pass --hf-repo owner/name")
    resolved = f"{username}/{repo_id}"
    print(f"Resolved HuggingFace repo: {repo_id} -> {resolved}")
    return resolved


def upload_info_with_progress(path: Path):
    from huggingface_hub._commit_api import UploadInfo

    size = path.stat().st_size
    with path.open("rb") as file:
        sample = file.peek(512)[:512]
        file.seek(0)
        sha = hashlib.sha256()
        with tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Prepare {path.name}",
        ) as bar:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
                bar.update(len(chunk))
    return UploadInfo(size=size, sha256=sha.digest(), sample=sample)


def upload_archive_file(api, archive: Path, repo_id: str, token: str) -> None:
    from huggingface_hub._commit_api import CommitOperationAdd

    upload_info = upload_info_with_progress(archive)
    operation = CommitOperationAdd(path_or_fileobj=b"", path_in_repo=archive.name)
    operation.path_or_fileobj = str(archive)
    operation.upload_info = upload_info
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[operation],
        commit_message=f"Upload {archive.name}",
        token=token,
    )


def upload_to_hf(
    archives: list[Path],
    repo_id: str,
    token: str,
    private: bool = True,
    preserve_history: bool = False,
    use_xet: bool = False,
) -> None:
    if use_xet:
        print("HuggingFace Xet transfer backend is allowed for this upload.")
    else:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        print("HuggingFace Xet transfer backend disabled for this upload.")

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)
    repo_id = resolve_hf_repo_id(repo_id)
    if preserve_history:
        print(f"Preserving existing HuggingFace dataset repo history for {repo_id}.")
    else:
        print(f"Deleting HuggingFace dataset repo {repo_id} before upload to avoid retained history.")
        api.delete_repo(repo_id=repo_id, repo_type="dataset", token=token, missing_ok=True)
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    for archive in archives:
        print(f"Uploading {archive.name} ({archive.stat().st_size / 1e9:.2f} GB) ...")
        upload_archive_file(api, archive, repo_id, token)
        print(f"Uploaded {archive.name} to dataset repo {repo_id}.")


def build_archives(args: argparse.Namespace, data_dir: Path, archive: Path, volume_size: str | None) -> list[Path]:
    jpg_root = data_dir / "MIMIC-CXR-JPG"
    txt_root = data_dir / "MIMIC-CXR"
    split_csv = jpg_root / "mimic-cxr-2.0.0-split.csv"
    if not split_csv.exists():
        sys.exit(f"Missing {split_csv}")

    subset_name = resolve_subset_name(args)
    subset_root = data_dir / subset_name
    jpg_dst_root = subset_root / "MIMIC-CXR-JPG" / "files"
    txt_dst_root = subset_root / "MIMIC-CXR" / "files"

    print(f"Sampling: seed={args.seed} fraction={args.fraction} -> {subset_root}")
    chosen, sub_split = sample_patients(split_csv, args.seed, args.fraction)
    print(f"  patients: {len(chosen)} / studies: {sub_split['study_id'].nunique()} / images: {len(sub_split)}")

    if args.dry_run:
        return []

    if not args.skip_copy:
        subset_root.mkdir(parents=True, exist_ok=True)

        jpg_pairs = [
            (jpg_src_path(jpg_root / "files", r.subject_id, r.study_id, r.dicom_id),
             jpg_dst_path(jpg_dst_root, r.subject_id, r.study_id, r.dicom_id))
            for r in sub_split.itertuples(index=False)
        ]
        ok_j, miss_j, bytes_j = parallel_copy(jpg_pairs, args.workers, "JPG")
        print(f"  jpg copied: {ok_j} (missing {miss_j}) {bytes_j / 1e9:.2f} GB")

        studies = sub_split[["subject_id", "study_id"]].drop_duplicates()
        txt_pairs = [
            (txt_src_path(txt_root / "files", r.subject_id, r.study_id),
             txt_dst_path(txt_dst_root, r.subject_id, r.study_id))
            for r in studies.itertuples(index=False)
        ]
        ok_t, miss_t, bytes_t = parallel_copy(txt_pairs, args.workers, "TXT")
        print(f"  txt copied: {ok_t} (missing {miss_t}) {bytes_t / 1e6:.1f} MB")

        # Mirror small JPG-side metadata CSVs (full copies; they're small and harmless).
        for fname in [
            "mimic-cxr-2.0.0-metadata.csv",
            "mimic-cxr-2.0.0-metadata.csv.gz",
            "mimic-cxr-2.0.0-split.csv",
            "mimic-cxr-2.0.0-split.csv.gz",
            "mimic-cxr-2.0.0-chexpert.csv",
            "mimic-cxr-2.0.0-chexpert.csv.gz",
            "mimic-cxr-2.0.0-negbio.csv",
            "mimic-cxr-2.0.0-negbio.csv.gz",
            "mimic-cxr-2.1.0-test-set-labeled.csv",
            "IMAGE_FILENAMES",
        ]:
            src = jpg_root / fname
            if src.exists():
                dst = subset_root / "MIMIC-CXR-JPG" / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst, follow_symlinks=True)

        for fname in ["cxr-record-list.csv.gz", "cxr-study-list.csv.gz", "cxr-provider-list.csv.gz"]:
            src = txt_root / fname
            if src.exists():
                dst = subset_root / "MIMIC-CXR" / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst, follow_symlinks=True)

        manifest = {
            "seed": args.seed,
            "fraction": args.fraction,
            "unit": "subject_id",
            "n_patients": len(chosen),
            "n_studies": int(sub_split["study_id"].nunique()),
            "n_images": int(len(sub_split)),
            "patient_sha256": patient_hash(chosen),
            "subset_name": subset_name,
        }
        (subset_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  manifest -> {subset_root / 'manifest.json'}")

    if args.skip_archive:
        return []

    if args.archive_threads < 1:
        sys.exit("--archive-threads must be >= 1")

    password = os.environ.get("DATA_PASSWORD")
    if not password:
        sys.exit("DATA_PASSWORD not set in .env")

    remove_archive_outputs(archive)

    # 7z sources, all relative to data_dir so the archive extracts straight into data/.
    sources = [subset_root]
    for companion in ["CXR-LT", "MIMIC-IV-ED-2-2"]:
        path = data_dir / companion
        if path.exists():
            sources.append(path)
        else:
            print(f"  warning: {path} missing; not bundled")

    print(f"Archiving -> {archive}")
    run_7z(
        archive,
        sources,
        password,
        data_dir.resolve(),
        args.compression_level,
        args.archive_threads,
        volume_size,
    )
    archives = archive_outputs(archive, volume_size)
    if not archives:
        sys.exit(f"No archive outputs found for {archive}")
    print_archive_summary(archives)
    return archives


def main() -> int:
    args = parse_args()
    load_dotenv()

    if not Path("camchex").is_dir() or not Path("data").is_dir():
        sys.exit("Run from project root.")

    data_dir = Path(args.data_dir)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = (archive_dir / args.archive_name).resolve()
    volume_size = None if args.single_archive else args.volume_size

    if args.upload_existing:
        archives = discover_existing_archives(archive)
        if not archives:
            sys.exit(f"No existing archive outputs found for {archive}")
        print(f"Using existing archive output(s) from {archive.parent}")
        print_archive_summary(archives)
    else:
        archives = build_archives(args, data_dir, archive, volume_size)

    if args.skip_upload or args.dry_run:
        return 0

    if not archives:
        sys.exit("No archive outputs available to upload. Use --upload-existing to upload existing files.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set in .env")

    visibility = "PUBLIC" if args.public else "private"
    print(f"Uploading to HuggingFace dataset repo: {args.hf_repo} ({visibility})")
    upload_to_hf(
        archives,
        args.hf_repo,
        token,
        private=not args.public,
        preserve_history=args.preserve_hf_history,
        use_xet=args.hf_use_xet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
