"""CXR-LT release/schema helpers shared by subset prep and training.

The 2023 release is a single 26-label task split across train/development/test
CSV files. The 2024 release ships a combined 45-label ``labels.csv`` plus
task-specific labeled files: task1 (40 labels), task2 (the old 26-label set
with ``Normal`` replacing ``No Finding``), and task3 (5 additional labels).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


CXR_LT_DATASET_DIR = "cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0"
METADATA_COLS = {
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
    "ViewCodeSequence_CodeMeaning",
    "path",
    "fpath",
    "split",
}

CXRLT_2023_LABELS = [
    "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation",
    "Edema", "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "No Finding",
    "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta",
]

CXRLT_2024_TASK1_LABELS = [
    "Adenopathy", "Atelectasis", "Azygos Lobe", "Calcification of the Aorta",
    "Cardiomegaly", "Clavicle Fracture", "Consolidation", "Edema", "Emphysema",
    "Enlarged Cardiomediastinum", "Fibrosis", "Fissure", "Fracture", "Granuloma",
    "Hernia", "Hydropneumothorax", "Infarction", "Infiltration", "Kyphosis",
    "Lobar Atelectasis", "Lung Lesion", "Lung Opacity", "Mass", "Nodule", "Normal",
    "Pleural Effusion", "Pleural Other", "Pleural Thickening", "Pneumomediastinum",
    "Pneumonia", "Pneumoperitoneum", "Pneumothorax", "Pulmonary Embolism",
    "Pulmonary Hypertension", "Rib Fracture", "Round(ed) Atelectasis",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta", "Tuberculosis",
]

CXRLT_2024_TASK2_LABELS = [
    "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation",
    "Edema", "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "Normal",
    "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta",
]

CXRLT_2024_TASK3_LABELS = ["Bulla", "Cardiomyopathy", "Hilum", "Osteopenia", "Scoliosis"]

CXRLT_2024_ALL_LABELS = [
    "Adenopathy", "Atelectasis", "Azygos Lobe", "Bulla", "Calcification of the Aorta",
    "Cardiomegaly", "Cardiomyopathy", "Clavicle Fracture", "Consolidation", "Edema",
    "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fissure", "Fracture",
    "Granuloma", "Hernia", "Hilum", "Hydropneumothorax", "Infarction", "Infiltration",
    "Kyphosis", "Lobar Atelectasis", "Lung Lesion", "Lung Opacity", "Mass", "Nodule",
    "Normal", "Osteopenia", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
    "Pulmonary Embolism", "Pulmonary Hypertension", "Rib Fracture",
    "Round(ed) Atelectasis", "Scoliosis", "Subcutaneous Emphysema", "Support Devices",
    "Tortuous Aorta", "Tuberculosis",
]

CXRLT_2024_LABELS_BY_SET = {
    "all": CXRLT_2024_ALL_LABELS,
    "task1": CXRLT_2024_TASK1_LABELS,
    "task2": CXRLT_2024_TASK2_LABELS,
    "task3": CXRLT_2024_TASK3_LABELS,
}

LabelSet = Literal["auto", "all", "task1", "task2", "task3"]


def cxr_lt_root(data_root: str | Path, version: str) -> Path:
    return Path(data_root) / "CXR-LT" / CXR_LT_DATASET_DIR / version


def infer_label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLS]


def normalize_cxrlt_split(value: object) -> str:
    split = str(value).strip().lower()
    if split in {"development", "val", "validate", "validation", "dev"}:
        return "validate"
    if split in {"train", "test"}:
        return split
    raise ValueError(f"unknown CXR-LT split value: {value!r}")


def resolve_label_set(version: str, label_set: LabelSet = "auto") -> str:
    if label_set != "auto":
        return label_set
    if version == "cxr-lt-2024":
        return "task1"
    return "standard"


def _read_csv(path: Path, required_cols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CXR-LT label file: {path}")
    df = pd.read_csv(path)
    if "fpath" in df.columns and "path" not in df.columns:
        df = df.rename(columns={"fpath": "path"})
    if required_cols is not None:
        cols = [
            c
            for c in [
                "dicom_id", "subject_id", "study_id", "ViewPosition",
                "ViewCodeSequence_CodeMeaning", "path", "split",
            ]
            if c in df.columns
        ]
        cols.extend(required_cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        df = df[cols]
    return df


def _add_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["split"] = split
    return df


def _load_2023(root: Path) -> tuple[pd.DataFrame, list[str], str]:
    frames = [
        _add_split(_read_csv(root / "train.csv", CXRLT_2023_LABELS), "train"),
        _add_split(_read_csv(root / "development.csv", CXRLT_2023_LABELS), "validate"),
        _add_split(_read_csv(root / "test.csv", CXRLT_2023_LABELS), "test"),
    ]
    return pd.concat(frames, ignore_index=True), list(CXRLT_2023_LABELS), "standard"


def _load_2024(root: Path, label_set: str) -> tuple[pd.DataFrame, list[str], str]:
    if label_set not in CXRLT_2024_LABELS_BY_SET:
        valid = ", ".join(sorted(CXRLT_2024_LABELS_BY_SET))
        raise ValueError(f"unsupported CXR-LT 2024 label set {label_set!r}; choose one of: {valid}")
    labels = list(CXRLT_2024_LABELS_BY_SET[label_set])
    if label_set == "all":
        df = _read_csv(root / "labels.csv", labels)
        df["split"] = df["split"].map(normalize_cxrlt_split)
        return df, labels, label_set
    if label_set == "task1":
        frames = [
            _add_split(_read_csv(root / "train_labeled.csv", labels), "train"),
            _add_split(_read_csv(root / "development_labeled_task1.csv", labels), "validate"),
            _add_split(_read_csv(root / "test_labeled_task1.csv", labels), "test"),
        ]
        return pd.concat(frames, ignore_index=True), labels, label_set
    if label_set == "task2":
        frames = [
            _add_split(_read_csv(root / "train_labeled.csv", labels), "train"),
            _add_split(_read_csv(root / "development_labeled_task2.csv", labels), "validate"),
            _add_split(_read_csv(root / "test_labeled_task2.csv", labels), "test"),
        ]
        return pd.concat(frames, ignore_index=True), labels, label_set

    train = _read_csv(root / "labels.csv", labels)
    train = train[train["split"].map(normalize_cxrlt_split) == "train"]
    frames = [
        _add_split(train, "train"),
        _add_split(_read_csv(root / "development_labeled_task3.csv", labels), "validate"),
        _add_split(_read_csv(root / "test_labeled_task3.csv", labels), "test"),
    ]
    return pd.concat(frames, ignore_index=True), labels, label_set


def load_cxr_lt_labels(
    data_root: str | Path,
    version: str = "cxr-lt-2023",
    label_set: LabelSet = "auto",
) -> tuple[pd.DataFrame, list[str], str]:
    """Load a normalized CXR-LT label frame.

    Returns ``(df, label_cols, resolved_label_set)``. ``df`` always contains a
    ``split`` column with values ``train``, ``validate``, or ``test``.
    """
    root = cxr_lt_root(data_root, version)
    resolved = resolve_label_set(version, label_set)
    if version == "cxr-lt-2023":
        if resolved not in {"standard", "task2"}:
            raise ValueError("CXR-LT 2023 only supports the standard 26-label set")
        return _load_2023(root)
    if version == "cxr-lt-2024":
        return _load_2024(root, resolved)
    raise ValueError(f"unsupported CXR-LT version: {version!r}")
