#!/usr/bin/env python3
"""Build train/val/test label CSVs for the MIMIC subset.

Ports the camchex/data/{01,02,03}_*.py pipeline targeted at a single subset
directory (default: data/subset/), producing:

    data/<subset>/labels/train.csv
    data/<subset>/labels/val.csv
    data/<subset>/labels/test.csv

Each CSV has one row per (study_id, dicom_id), columns:
  study_id, subject_id, dicom_id, ViewPosition, path,
  clinical_indication, temperature, heartrate, resprate,
  o2sat, sbp, dbp, gender, + 26 CXR-LT class columns.

``path`` is stored relative to data/<subset>/MIMIC-CXR-JPG/files/ so the
training-time MimicMultiViewDataModule can prepend that directory as image_root.

Run from project root:
    python scripts/prepare_subset_labels.py
    python scripts/prepare_subset_labels.py --subset-name subset_seed7_5pct
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


LABEL_COLS = [
    "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation",
    "Edema", "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "No Finding",
    "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta",
]
VITAL_SIGNS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--subset-name", default="subset")
    p.add_argument("--cxr-lt-version", default="cxr-lt-2023",
                   help="Subdirectory under CXR-LT/.../2.0.0/ to read split labels from")
    p.add_argument("--cpu-fraction", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-dirname", default="labels",
                   help="Output dir under data/<subset>/")
    p.add_argument("--keep-checkpoint", action="store_true",
                   help="Write an intermediate merged.csv next to the labels/ dir")
    args = p.parse_args()
    if args.workers is None:
        args.workers = max(1, int(cpu_count() * args.cpu_fraction))
    return args


# ---- report parsing (uses mimic-cxr/txt/section_parser.py) ------------------

def _clean_report(report: str) -> str:
    report = report.replace("\\n", " ").replace("\n", " ")
    report = re.sub(r"(?<![a-zA-Z])/{1,}|/{1,}(?![a-zA-Z])", "", report)
    report = re.sub(r"\s+", " ", report).strip()

    def report_cleaner(t: str) -> str:
        return (t.replace("__", "_").replace("..", ".").replace("  ", " ")
                .replace("1. ", "").replace("2. ", "").replace("3. ", "")
                .replace("4. ", "").replace("5. ", "").strip().lower())

    def sent_cleaner(t: str) -> str:
        return re.sub(r"[.,?;*!%^&_+():\-\[\]{}]", "",
                      t.replace('"', "").replace("'", "").strip().lower())

    cleaned = [sent_cleaner(s) for s in report_cleaner(report).split(". ") if s]
    return re.sub(r"\s+", " ", " . ".join(cleaned).strip() + " .").strip()


_REPORTS_BASE: Optional[str] = None
_CUSTOM_SECTION_NAMES: dict = {}
_CUSTOM_INDICES: dict = {}


def _worker_init(reports_base: str, custom_section_names: dict, custom_indices: dict):
    global _REPORTS_BASE, _CUSTOM_SECTION_NAMES, _CUSTOM_INDICES
    _REPORTS_BASE = reports_base
    _CUSTOM_SECTION_NAMES = custom_section_names
    _CUSTOM_INDICES = custom_indices
    sys.path.insert(0, "mimic-cxr/txt")


def _parse_single_report(args_tuple):
    import section_parser as sp
    idx, subject_id, study_id = args_tuple
    sid_str = str(subject_id)
    study_str = f"s{study_id}"
    report_path = os.path.join(
        _REPORTS_BASE, f"p{sid_str[:2]}", f"p{sid_str}", f"{study_str}.txt",
    )
    if not os.path.exists(report_path):
        return None

    with open(report_path, "r") as f:
        text = f.read()

    if study_str in _CUSTOM_INDICES:
        ci = _CUSTOM_INDICES[study_str]
        text = text[ci[0]:ci[1]]

    sections, section_names, _ = sp.section_text(text)
    section_dict = dict(zip(section_names, sections))

    result = {"_idx": idx}
    if study_str in _CUSTOM_SECTION_NAMES:
        custom_sec = _CUSTOM_SECTION_NAMES[study_str]
        if custom_sec in section_dict:
            result[custom_sec] = _clean_report(section_dict[custom_sec])
        return result

    for section in ("impression", "findings", "last_paragraph", "comparison", "indication", "history"):
        if section in section_dict:
            result[section] = _clean_report(section_dict[section])
    return result


# ---- pipeline ---------------------------------------------------------------

def load_metadata(data_root: Path, subset_dir: Path) -> pd.DataFrame:
    metadata_fp = subset_dir / "MIMIC-CXR-JPG" / "mimic-cxr-2.0.0-metadata.csv"
    if not metadata_fp.exists():
        sys.exit(f"Missing {metadata_fp}; was build_mimic_subset.py run?")

    metadata_df = pd.read_csv(metadata_fp)
    study_df = (
        metadata_df
        .drop(columns=["dicom_id", "ViewPosition", "Rows", "Columns"])
        .drop_duplicates(subset="study_id")
        .sort_values(["subject_id", "StudyDate"])
        .reset_index(drop=True)
    )
    study_df["PreviousStudy"] = (
        study_df.groupby("subject_id")["study_id"].shift(1).astype("Int64")
    )
    study_df["StudyDateTime"] = pd.to_datetime(
        (study_df["StudyDate"] * 1000000) + study_df["StudyTime"].astype(int),
        format="%Y%m%d%H%M%S",
    )
    study_df = study_df.drop(columns=["StudyDate", "StudyTime"])
    return study_df, metadata_df


def merge_ed_vitals(data_root: Path, study_df: pd.DataFrame) -> pd.DataFrame:
    ed_root = data_root / "MIMIC-IV-ED-2-2" / "mimic-iv-ed-2.2" / "ed"
    triage = pd.read_csv(ed_root / "triage.csv.gz")
    edstays = pd.read_csv(ed_root / "edstays.csv.gz")
    ed = pd.merge(triage, edstays, on=["subject_id", "stay_id"])
    ed["intime"] = pd.to_datetime(ed["intime"])
    ed["outtime"] = pd.to_datetime(ed["outtime"])

    mimic = pd.merge(study_df, ed, on="subject_id", how="left")
    mimic["time_to_intime"] = (mimic["StudyDateTime"] - mimic["intime"]).abs().dt.total_seconds()

    has_stay = mimic.dropna(subset=["time_to_intime"])
    closest_idx = has_stay.groupby("study_id")["time_to_intime"].idxmin()
    mimic = pd.concat([
        mimic.loc[closest_idx],
        mimic[~mimic["study_id"].isin(mimic.loc[closest_idx, "study_id"])],
    ]).drop(columns=["time_to_intime"]).reset_index(drop=True)

    vitals = pd.read_csv(ed_root / "vitalsign.csv.gz")
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    merged = pd.merge(mimic, vitals, on=["subject_id", "stay_id"], suffixes=("", "_chart"), how="left")
    merged["time_diff"] = (merged["StudyDateTime"] - merged["charttime"]).abs()

    closest_charts = (
        merged[merged[VITAL_SIGNS].isna().any(axis=1)]
        .sort_values(["study_id", "time_diff"])
        .drop_duplicates(subset=["study_id"], keep="first")
    )
    for vital in VITAL_SIGNS:
        mimic[vital] = mimic[vital].combine_first(closest_charts[f"{vital}_chart"])

    return mimic.reset_index(drop=True)


def parse_all_reports(mimic_df: pd.DataFrame, reports_base: Path, workers: int) -> pd.DataFrame:
    sys.path.insert(0, "mimic-cxr/txt")
    import section_parser as sp
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    for col in ("impression", "findings", "last_paragraph", "comparison", "indication", "history"):
        mimic_df[col] = None

    args = list(zip(mimic_df.index, mimic_df["subject_id"], mimic_df["study_id"]))
    ctx = get_context("fork")
    with ctx.Pool(
        workers, initializer=_worker_init,
        initargs=(str(reports_base), custom_section_names, custom_indices),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_parse_single_report, args, chunksize=256),
            total=len(args), desc="reports",
        ):
            if result is None:
                continue
            idx = result.pop("_idx")
            for col, val in result.items():
                mimic_df.at[idx, col] = val

    def merge_indication(row):
        ind = str(row["indication"]) if pd.notna(row["indication"]) else ""
        hist = str(row["history"]) if pd.notna(row["history"]) else ""
        joined = (ind.strip() + (" " if ind.strip() and hist.strip() else "") + hist.strip()).strip()
        return joined if joined else np.nan

    mimic_df["clinical_indication"] = mimic_df.apply(merge_indication, axis=1)

    mimic_df = mimic_df.drop(
        columns=["impression", "findings", "last_paragraph", "comparison",
                 "indication", "history", "examination", "technique",
                 "recommendations", "note"],
        errors="ignore",
    )
    return mimic_df


def attach_labels(mimic_df: pd.DataFrame, data_root: Path, cxr_lt_version: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    cxr_lt_root = data_root / "CXR-LT" / "cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0" / cxr_lt_version
    train_df = pd.read_csv(cxr_lt_root / "train.csv")
    val_df = pd.read_csv(cxr_lt_root / "development.csv")
    test_df = pd.read_csv(cxr_lt_root / "test.csv")
    train_df["split"] = "train"
    val_df["split"] = "validate"
    test_df["split"] = "test"
    labels_dicom = pd.concat([train_df, val_df, test_df], ignore_index=True)
    labels_dicom = labels_dicom.drop(
        columns=["subject_id", "study_id", "ViewPosition", "ViewCodeSequence_CodeMeaning", "path"],
        errors="ignore",
    )

    meta_slim = metadata_df[["dicom_id", "study_id", "ViewPosition"]]
    mimic_df = mimic_df.drop(
        columns=["PerformedProcedureStepDescription", "ProcedureCodeSequence_CodeMeaning",
                 "ViewCodeSequence_CodeMeaning", "PatientOrientationCodeSequence_CodeMeaning"],
        errors="ignore",
    )
    merged = mimic_df.merge(meta_slim, on="study_id", how="left")
    merged = merged.merge(labels_dicom, on="dicom_id", how="left")
    merged = merged.dropna(subset=["clinical_indication", "split"])
    return merged


def build_paths_and_filter(merged: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    def rel_path(row):
        sid = str(row["subject_id"])
        return f"p{sid[:2]}/p{sid}/s{row['study_id']}/{row['dicom_id']}.jpg"

    merged["path"] = merged.apply(rel_path, axis=1)
    exists = merged["path"].map(lambda p: (image_root / p).exists())
    kept = merged[exists].copy()
    dropped = len(merged) - len(kept)
    print(f"  path-existence filter: kept {len(kept)} / {len(merged)} ({dropped} missing dropped)")
    return kept


def main() -> int:
    args = parse_args()

    if not Path("data").is_dir():
        sys.exit("Run from project root (data/ not found).")

    data_root = Path(args.data_root)
    subset_dir = data_root / args.subset_name
    reports_base = subset_dir / "MIMIC-CXR" / "files"
    image_root = subset_dir / "MIMIC-CXR-JPG" / "files"
    out_dir = subset_dir / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    if not reports_base.is_dir():
        sys.exit(f"Missing reports tree: {reports_base}")
    if not image_root.is_dir():
        sys.exit(f"Missing image tree: {image_root}")

    print(f"[1/4] Loading metadata from {subset_dir}")
    study_df, metadata_df = load_metadata(data_root, subset_dir)

    print(f"[2/4] Merging ED vitals from {data_root / 'MIMIC-IV-ED-2-2'}")
    mimic_df = merge_ed_vitals(data_root, study_df)
    # gender comes from ED triage; align column to match dataset's expectations
    if "gender" not in mimic_df.columns:
        mimic_df["gender"] = np.nan

    print(f"[3/4] Parsing reports with {args.workers} workers")
    mimic_df = parse_all_reports(mimic_df, reports_base, args.workers)

    print("[4/4] Attaching CXR-LT labels and filtering to existing images")
    merged = attach_labels(mimic_df, data_root, args.cxr_lt_version, metadata_df)
    merged = build_paths_and_filter(merged, image_root)

    if args.keep_checkpoint:
        ckpt = subset_dir / "merged.csv"
        merged.to_csv(ckpt, index=False)
        print(f"  checkpoint -> {ckpt}")

    keep_cols = [
        "study_id", "subject_id", "dicom_id", "ViewPosition", "path",
        "clinical_indication", *VITAL_SIGNS, "gender", *LABEL_COLS, "split",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged[keep_cols]

    splits = {
        "train": merged[merged["split"] == "train"].drop(columns=["split"]),
        "val":   merged[merged["split"] == "validate"].drop(columns=["split"]),
        "test":  merged[merged["split"] == "test"].drop(columns=["split"]),
    }
    for name, df in splits.items():
        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  {name}: {len(df):>6} rows -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
