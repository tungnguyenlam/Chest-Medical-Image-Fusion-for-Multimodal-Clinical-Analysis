#!/usr/bin/env python3
"""
Build a deterministic patient-level subset of MIMIC-CXR-JPG + MIMIC-CXR reports,
bundle it with CXR-LT and MIMIC-IV-ED-2-2 into a password-protected 7z archive,
and optionally upload to a private HuggingFace dataset repo.

Run from the project root:
    python scripts/build_mimic_subset.py
    python scripts/build_mimic_subset.py --fraction 0.05 --no-upload

Reads HF_TOKEN and DATA_PASSWORD from .env (see .env.example).
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


def _default_workers() -> int:
    """Half the visible CPU cores, floored at 1."""
    return max(1, cpu_count() // 2)

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
                   help="HuggingFace dataset repo (default: tung-thesis under your user)")
    p.add_argument("--public", action="store_true",
                   help="Create the HF repo as public. DEFAULT is private. "
                        "Public re-hosting of MIMIC violates the PhysioNet DUA — "
                        "only pass this for sanitized or non-credentialed bundles.")
    p.add_argument("--workers", type=int, default=_default_workers(),
                   help=f"Parallel copy workers (default: half of cpu_count() = {_default_workers()})")
    p.add_argument("--skip-copy", action="store_true", help="Reuse existing data/<subset>/ tree")
    p.add_argument("--skip-archive", action="store_true", help="Skip 7z step")
    p.add_argument("--skip-upload", action="store_true", help="Skip HuggingFace upload")
    p.add_argument("--dry-run", action="store_true", help="Sample + print stats, then exit")
    return p.parse_args()


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


def run_7z(archive: Path, sources: list[Path], password: str, workdir: Path) -> None:
    rels = [str(s.relative_to(workdir)) for s in sources]
    cmd = ["7z", "a", "-t7z", "-mhe=on", f"-p{password}", "-mx=5", str(archive), *rels]
    print(f"  $ 7z a -t7z -mhe=on -p<DATA_PASSWORD> -mx=5 {archive.name} {' '.join(rels)}")
    subprocess.run(cmd, cwd=str(workdir), check=True)


def upload_to_hf(archive: Path, repo_id: str, token: str, private: bool = True) -> None:
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=str(archive),
        path_in_repo=archive.name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    print(f"Uploaded {archive.name} to dataset repo {repo_id}.")


def main() -> int:
    args = parse_args()
    load_dotenv()

    if not Path("camchex").is_dir() or not Path("data").is_dir():
        sys.exit("Run from project root.")

    data_dir = Path(args.data_dir)
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
        return 0

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
        return 0

    password = os.environ.get("DATA_PASSWORD")
    if not password:
        sys.exit("DATA_PASSWORD not set in .env")

    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = (archive_dir / args.archive_name).resolve()
    if archive.exists():
        archive.unlink()

    # 7z sources, all relative to data_dir so the archive extracts straight into data/.
    sources = [subset_root]
    for companion in ["CXR-LT", "MIMIC-IV-ED-2-2"]:
        path = data_dir / companion
        if path.exists():
            sources.append(path)
        else:
            print(f"  warning: {path} missing; not bundled")

    print(f"Archiving -> {archive}")
    run_7z(archive, sources, password, data_dir.resolve())
    print(f"  archive size: {archive.stat().st_size / 1e9:.2f} GB")

    if args.skip_upload:
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set in .env")

    visibility = "PUBLIC" if args.public else "private"
    print(f"Uploading to HuggingFace dataset repo: {args.hf_repo} ({visibility})")
    upload_to_hf(archive, args.hf_repo, token, private=not args.public)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
