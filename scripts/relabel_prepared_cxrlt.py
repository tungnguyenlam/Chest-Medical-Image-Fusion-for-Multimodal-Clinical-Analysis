#!/usr/bin/env python3
"""Relabel already-prepared CaMCheX CSVs with a selected CXR-LT release.

This is useful when `data/data-camchex/03_mimic_{train,development,test}.csv`
already has reports, vitals, and image paths, but we want a different CXR-LT
label release/split without reparsing reports. The input CSVs are combined
first, then split using the selected CXR-LT labels, so CXR-LT 2024 split
reassignment is respected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.dataloader.cxr_lt import (
    CXRLT_2023_LABELS,
    CXRLT_2024_ALL_LABELS,
    load_cxr_lt_labels,
)


KNOWN_LABEL_COLS = sorted(set(CXRLT_2023_LABELS) | set(CXRLT_2024_ALL_LABELS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--input-csv",
        action="append",
        default=None,
        help="Prepared CSV to relabel. Repeat for train/development/test. Defaults to data/data-camchex/03_mimic_*.csv.",
    )
    parser.add_argument("--cxr-lt-version", default="cxr-lt-2024")
    parser.add_argument(
        "--cxr-lt-label-set",
        default="task1",
        choices=["auto", "all", "task1", "task2", "task3"],
    )
    parser.add_argument("--out-dir", default="data/data-camchex/cxrlt2024_task1")
    parser.add_argument("--keep-source-split", action="store_true",
                        help="Keep the old input split column as source_split for audits.")
    return parser.parse_args()


def default_inputs() -> list[str]:
    return [
        "data/data-camchex/03_mimic_train.csv",
        "data/data-camchex/03_mimic_development.csv",
        "data/data-camchex/03_mimic_test.csv",
    ]


def read_inputs(paths: list[str], keep_source_split: bool) -> pd.DataFrame:
    frames = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"missing input CSV: {path}")
        df = pd.read_csv(path)
        if "split" in df.columns and keep_source_split:
            df = df.rename(columns={"split": "source_split"})
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["dicom_id"], keep="first")
    drop_cols = [c for c in [*KNOWN_LABEL_COLS, "split"] if c in merged.columns]
    return merged.drop(columns=drop_cols)


def attach_labels(prepared: pd.DataFrame, data_root: str, version: str, label_set: str) -> tuple[pd.DataFrame, list[str], str]:
    labels, label_cols, resolved_label_set = load_cxr_lt_labels(data_root, version=version, label_set=label_set)
    labels = labels.drop(
        columns=["subject_id", "study_id", "ViewPosition", "ViewCodeSequence_CodeMeaning", "path", "fpath"],
        errors="ignore",
    )
    out = prepared.merge(labels, on="dicom_id", how="inner", validate="one_to_one")
    return out, label_cols, resolved_label_set


def main() -> int:
    args = parse_args()
    input_paths = args.input_csv or default_inputs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = read_inputs(input_paths, args.keep_source_split)
    relabeled, label_cols, resolved_label_set = attach_labels(
        prepared,
        args.data_root,
        args.cxr_lt_version,
        args.cxr_lt_label_set,
    )

    split_to_file = {"train": "train.csv", "validate": "val.csv", "test": "test.csv"}
    split_counts: dict[str, int] = {}
    for split, filename in split_to_file.items():
        df = relabeled[relabeled["split"] == split].drop(columns=["split"])
        path = out_dir / filename
        df.to_csv(path, index=False)
        split_counts[split] = int(len(df))
        print(f"{split:>8}: {len(df):>7} rows -> {path}")

    metadata = {
        "input_csvs": input_paths,
        "cxr_lt_version": args.cxr_lt_version,
        "cxr_lt_label_set": resolved_label_set,
        "num_classes": len(label_cols),
        "classes": label_cols,
        "input_rows_after_dedup": int(len(prepared)),
        "matched_rows": int(len(relabeled)),
        "splits": split_counts,
    }
    meta_path = out_dir / "label_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"metadata -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
