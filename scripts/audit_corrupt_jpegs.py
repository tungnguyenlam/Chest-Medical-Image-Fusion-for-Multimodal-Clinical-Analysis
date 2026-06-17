#!/usr/bin/env python3
"""Audit unreadable/corrupt JPEGs and their CXR-LT label impact.

Examples:
    python scripts/audit_corrupt_jpegs.py --label-dir data/subset/labels
    python scripts/audit_corrupt_jpegs.py --label-dir data/subset/labels_cxrlt2024_task1
    python scripts/audit_corrupt_jpegs.py --csv data/data-camchex/03_mimic_train.csv --image-root .
    python scripts/audit_corrupt_jpegs.py --config training/camchex_v2nano_vitals_cxrlt2024/config.yaml --pipeline channels

The default ``--pipeline auto`` uses channel preprocessing when a training
config or ``--channel-mode`` is provided; otherwise it falls back to raw JPEG
decode checks. ``--pipeline channels`` is the path that matters for the active
training and Grad-CAM loaders when ``channel_mode`` is enabled: it calls the
same ``load_or_build_channels`` helper and counts decode failures or channel
preprocessing exceptions as lost rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.cxr_lt import CXRLT_2023_LABELS, CXRLT_2024_ALL_LABELS


DEFAULT_SPLIT_FILES = ("train.csv", "val.csv", "development.csv", "test.csv")
CONFIG_SPLIT_KEYS = {
    "train": "train_df_path",
    "val": "devel_df_path",
    "test": "pred_df_path",
}
BAD_STATUSES = {"missing", "unreadable", "pipeline_error"}
KNOWN_LABEL_COLS = list(dict.fromkeys([*CXRLT_2023_LABELS, *CXRLT_2024_ALL_LABELS]))
NON_LABEL_COLS = {
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
    "ViewCodeSequence_CodeMeaning",
    "path",
    "fpath",
    "split",
    "report",
    "clinical_indication",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "gender",
}


@dataclass(frozen=True)
class ImageCheck:
    split: str
    row_id: int
    study_id: object
    dicom_id: object
    path: str
    resolved_path: str
    status: str
    decoder_warning: str
    error: str
    positive_labels: str

    @property
    def is_bad(self) -> bool:
        return self.status in BAD_STATUSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--label-dir",
        default="data/subset/labels",
        help="Directory containing train/val/test label CSVs (default: data/subset/labels)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training config YAML. Uses data.datamodule_cfg split paths/channel settings unless overridden.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="When --config is used, choose which configured split(s) to audit (default: all).",
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        default=None,
        help="Explicit CSV file(s) to audit. Overrides --label-dir.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root to prepend to relative path values. Defaults to inferred data/<subset>/MIMIC-CXR-JPG/files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: output/corrupt_jpeg_audit/<input-name>",
    )
    parser.add_argument(
        "--decoder",
        choices=["legacy", "active", "jpeg4py"],
        default="legacy",
        help=(
            "legacy: jpeg4py then cv2 fallback, matching current camchex/; "
            "active: cv2 only, matching root src/ dataloaders; "
            "jpeg4py: original imported camchex behavior with no fallback "
            "(default: legacy)"
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=["auto", "decode", "channels"],
        default="auto",
        help=(
            "decode: only test JPEG decoding; channels: run the active training image-channel "
            "preprocessing path; auto: channels when --config/--channel-mode supplies a channel_mode, "
            "else decode (default: auto)"
        ),
    )
    parser.add_argument(
        "--channel-mode",
        default=None,
        help="Training channel_mode to audit, e.g. raw_clahe_histeq. Use 'none' for raw decode.",
    )
    parser.add_argument(
        "--image-channel-cache-dir",
        default=None,
        help="Channel cache directory. With --config, defaults to data.datamodule_cfg.image_channel_cache_dir.",
    )
    parser.add_argument(
        "--disable-channel-cache",
        action="store_true",
        help="Force --pipeline channels to rebuild from JPEGs without reading or writing the channel cache.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Preprocessing output size for --pipeline channels. With --config, defaults to data.datamodule_cfg.size.",
    )
    parser.add_argument(
        "--uint8-image-pipeline",
        action="store_true",
        help="Match training runs that pass dequantize=False to load_or_build_channels.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=1.0,
        help="Label value counted as positive/associated (default: 1.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Parallel decode workers (default: half of visible CPUs)",
    )
    parser.add_argument(
        "--write-all",
        action="store_true",
        help="Also write per_image_status.csv for every checked row, not only failures/warnings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit only the first N rows after loading CSVs. Intended for smoke tests.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def load_config_datamodule_cfg(config_path: str | None) -> dict:
    if not config_path:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("Reading --config requires PyYAML to be installed") from exc

    with Path(config_path).open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return ((cfg.get("data") or {}).get("datamodule_cfg") or {}).copy()


def normalize_none(value: object) -> object:
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return value


def apply_config_defaults(args: argparse.Namespace) -> dict:
    datamodule_cfg = load_config_datamodule_cfg(args.config)

    args.channel_mode = normalize_none(args.channel_mode)
    if args.channel_mode is None and "channel_mode" in datamodule_cfg:
        args.channel_mode = normalize_none(datamodule_cfg.get("channel_mode"))

    if args.image_channel_cache_dir is None:
        args.image_channel_cache_dir = normalize_none(datamodule_cfg.get("image_channel_cache_dir"))
    if args.size is None:
        args.size = int(datamodule_cfg.get("size", 512))

    if args.pipeline == "auto":
        args.pipeline = "channels" if args.channel_mode else "decode"
    if args.pipeline == "channels" and not args.channel_mode:
        raise ValueError("--pipeline channels requires --channel-mode or a config with data.datamodule_cfg.channel_mode")
    if args.disable_channel_cache:
        args.image_channel_cache_dir = None

    return datamodule_cfg


def split_name_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if stem == "development":
        return "val"
    return stem


def discover_csvs(args: argparse.Namespace, datamodule_cfg: dict) -> list[Path]:
    if args.csv:
        csvs = [Path(p) for p in args.csv]
    elif args.config:
        split_names = CONFIG_SPLIT_KEYS if args.split == "all" else {args.split: CONFIG_SPLIT_KEYS[args.split]}
        csvs = []
        unsupported = []
        for split_name, key in split_names.items():
            value = datamodule_cfg.get(key)
            if not value:
                continue
            path = Path(value)
            if path.suffix.lower() != ".csv":
                unsupported.append(f"{split_name}={path}")
                continue
            csvs.append(path)
        if unsupported:
            raise ValueError(
                "This audit currently supports CSV split files only; unsupported configured split(s): "
                + ", ".join(unsupported)
            )
    else:
        label_dir = Path(args.label_dir)
        csvs = [label_dir / name for name in DEFAULT_SPLIT_FILES if (label_dir / name).exists()]
    missing = [p for p in csvs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV file(s): " + ", ".join(str(p) for p in missing))
    if not csvs:
        raise FileNotFoundError(f"No split CSVs found under {args.label_dir}")
    return csvs


def infer_image_root(args: argparse.Namespace, csvs: list[Path]) -> Path | None:
    if args.image_root:
        return Path(args.image_root)
    if args.csv:
        return None
    label_dir = Path(args.label_dir)
    subset_root = label_dir.parent
    image_root = subset_root / "MIMIC-CXR-JPG" / "files"
    if image_root.exists():
        return image_root
    common_parent = Path(os.path.commonpath([str(p.parent) for p in csvs]))
    image_root = common_parent.parent / "MIMIC-CXR-JPG" / "files"
    return image_root if image_root.exists() else None


def default_out_dir(args: argparse.Namespace, csvs: list[Path]) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    if args.csv:
        name = "custom"
        if len(csvs) == 1:
            name = csvs[0].stem
    elif args.config:
        config_stem = Path(args.config).parent.name or Path(args.config).stem
        suffix = args.pipeline
        if args.pipeline == "channels" and args.disable_channel_cache:
            suffix = "channels_no_cache"
        name = f"{config_stem}_{args.split}_{suffix}"
    else:
        label_dir = Path(args.label_dir)
        name = f"{label_dir.parent.name}_{label_dir.name}"
    return Path("output") / "corrupt_jpeg_audit" / name


def label_columns(df: pd.DataFrame) -> list[str]:
    known = [c for c in KNOWN_LABEL_COLS if c in df.columns]
    if known:
        return known
    numeric_cols = []
    for col in df.columns:
        if col in NON_LABEL_COLS:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            numeric_cols.append(col)
    return numeric_cols


def normalize_path_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def candidate_paths(path_value: str, image_root: Path | None) -> Iterable[Path]:
    raw = Path(path_value)
    candidates: list[Path] = [raw]
    if image_root is not None and not raw.is_absolute():
        candidates.append(image_root / raw)
    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, ROOT / raw, ROOT / "camchex" / raw])

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def resolve_existing_path(path_value: str, image_root: Path | None) -> Path | None:
    for candidate in candidate_paths(path_value, image_root):
        if candidate.exists():
            return candidate
    return None


def cv2_decode(path: Path) -> tuple[str, str, str]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception as exc:  # pragma: no cover - cv2 usually returns None.
        return "unreadable", "", f"cv2 exception: {exc}"
    if img is None:
        return "unreadable", "", "cv2.imread returned None"
    return "ok", "", ""


def jpeg4py_decode(path: Path) -> tuple[str, str, str]:
    try:
        import jpeg4py as jpeg
    except Exception as exc:
        return "unreadable", "", f"jpeg4py import failed: {exc}"
    try:
        jpeg.JPEG(str(path)).decode()
    except Exception as exc:
        return "unreadable", "", f"jpeg4py exception: {exc}"
    return "ok", "", ""


def decode_path(path: Path, decoder: str) -> tuple[str, str, str]:
    if decoder == "active":
        return cv2_decode(path)
    if decoder == "jpeg4py":
        return jpeg4py_decode(path)

    status, _, error = jpeg4py_decode(path)
    if status == "ok":
        return status, "", ""
    fallback_status, _, fallback_error = cv2_decode(path)
    if fallback_status == "ok":
        return "ok_cv2_fallback", error, ""
    return "unreadable", error, fallback_error


def check_channel_path(
    raw_path: str,
    resolved: Path,
    args: argparse.Namespace,
    preprocess_cfg,
) -> tuple[str, str, str, Path]:
    from src.dataloader.utils import load_or_build_channels

    # Prefer the raw dataframe path when it is resolvable by the training helper;
    # fall back to the audit-resolved path for subset CSVs that require --image-root.
    training_resolved = resolve_existing_path(raw_path, None)
    pipeline_input = raw_path if training_resolved is not None else str(resolved)
    display_path = resolved
    if display_path.name.endswith(".jpg") and "_resized_1024" not in display_path.name:
        resized = display_path.with_name(display_path.name.replace(".jpg", "_resized_1024.jpg"))
        if resized.exists():
            display_path = resized

    try:
        img = load_or_build_channels(
            pipeline_input,
            args.channel_mode,
            preprocess_cfg,
            args.image_channel_cache_dir,
            dequantize=not args.uint8_image_pipeline,
        )
    except Exception as exc:
        return "pipeline_error", "", f"{type(exc).__name__}: {exc}", display_path

    if img is None:
        return "unreadable", "", "load_or_build_channels returned None", display_path
    return "ok", "", "", display_path


def positive_labels(row: pd.Series, labels: list[str], threshold: float) -> list[str]:
    values = pd.to_numeric(row[labels], errors="coerce").fillna(0)
    return [label for label, value in values.items() if float(value) >= threshold]


def check_one(
    row: pd.Series,
    labels: list[str],
    image_root: Path | None,
    args: argparse.Namespace,
    preprocess_cfg,
) -> ImageCheck:
    raw_path = normalize_path_value(row.get("path", row.get("fpath", "")))
    positives = positive_labels(row, labels, args.positive_threshold)
    resolved = resolve_existing_path(raw_path, image_root)
    if resolved is None:
        return ImageCheck(
            split=str(row["_split"]),
            row_id=int(row["_row_id"]),
            study_id=row.get("study_id", ""),
            dicom_id=row.get("dicom_id", ""),
            path=raw_path,
            resolved_path="",
            status="missing",
            decoder_warning="",
            error="no candidate path exists",
            positive_labels="|".join(positives),
        )

    if args.pipeline == "channels":
        status, warning, error, preferred = check_channel_path(raw_path, resolved, args, preprocess_cfg)
    else:
        # Match the training loader preference for pre-resized files when present.
        preferred = resolved
        if preferred.name.endswith(".jpg") and "_resized_1024" not in preferred.name:
            resized = preferred.with_name(preferred.name.replace(".jpg", "_resized_1024.jpg"))
            if resized.exists():
                preferred = resized

        status, warning, error = decode_path(preferred, args.decoder)
        if status == "unreadable" and "_resized_1024.jpg" in str(preferred):
            original = Path(str(preferred).replace("_resized_1024.jpg", ".jpg"))
            if original.exists():
                retry_status, retry_warning, retry_error = decode_path(original, args.decoder)
                if retry_status != "unreadable":
                    status = retry_status
                    warning = "resized failed; original decoded"
                    error = retry_warning or retry_error
                    preferred = original

    return ImageCheck(
        split=str(row["_split"]),
        row_id=int(row["_row_id"]),
        study_id=row.get("study_id", ""),
        dicom_id=row.get("dicom_id", ""),
        path=raw_path,
        resolved_path=str(preferred),
        status=status,
        decoder_warning=warning,
        error=error,
        positive_labels="|".join(positives),
    )


def load_inputs(csvs: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    all_labels: list[str] = []
    for path in csvs:
        df = pd.read_csv(path)
        labels = label_columns(df)
        for label in labels:
            if label not in all_labels:
                all_labels.append(label)
        df = df.copy()
        df["_split"] = split_name_from_path(path)
        df["_source_csv"] = str(path)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged["_row_id"] = np.arange(len(merged))
    return merged, all_labels


def build_label_impact(df: pd.DataFrame, checks_df: pd.DataFrame, labels: list[str], threshold: float) -> pd.DataFrame:
    working = df.merge(checks_df[["row_id", "status"]], left_on="_row_id", right_on="row_id", how="left")
    working["_bad"] = working["status"].isin(BAD_STATUSES)
    records = []
    for label in labels:
        values = pd.to_numeric(working[label], errors="coerce").fillna(0)
        pos = values >= threshold
        positive_rows = int(pos.sum())
        bad_positive_rows = int((pos & working["_bad"]).sum())
        affected_positive_rows = int((pos & (working["status"] != "ok")).sum())

        if "study_id" in working.columns:
            pos_studies = set(working.loc[pos, "study_id"].dropna().tolist())
            bad_by_study = working.groupby("study_id")["_bad"].agg(["any", "all"])
            affected_studies = set(bad_by_study.index[bad_by_study["any"]])
            lost_studies = set(bad_by_study.index[bad_by_study["all"]])
            affected_positive_studies = len(pos_studies & affected_studies)
            lost_positive_studies = len(pos_studies & lost_studies)
            positive_studies = len(pos_studies)
        else:
            positive_studies = affected_positive_studies = lost_positive_studies = 0

        records.append(
            {
                "label": label,
                "positive_image_rows": positive_rows,
                "bad_positive_image_rows": bad_positive_rows,
                "bad_positive_image_row_pct": bad_positive_rows / positive_rows if positive_rows else 0.0,
                "affected_positive_image_rows": affected_positive_rows,
                "positive_studies": positive_studies,
                "affected_positive_studies_any_bad_view": affected_positive_studies,
                "lost_positive_studies_all_views_bad": lost_positive_studies,
                "lost_positive_study_pct": lost_positive_studies / positive_studies if positive_studies else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["lost_positive_studies_all_views_bad", "bad_positive_image_rows", "positive_image_rows"],
        ascending=[False, False, True],
    )


def build_split_summary(checks_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in checks_df.groupby("split", dropna=False):
        bad = group["status"].isin(BAD_STATUSES)
        warning = group["status"].ne("ok") & ~bad
        rows.append(
            {
                "split": split,
                "image_rows": len(group),
                "ok_rows": int((group["status"] == "ok").sum()),
                "decoder_warning_rows": int(warning.sum()),
                "missing_rows": int((group["status"] == "missing").sum()),
                "unreadable_rows": int((group["status"] == "unreadable").sum()),
                "pipeline_error_rows": int((group["status"] == "pipeline_error").sum()),
                "lost_rows": int(bad.sum()),
                "lost_row_pct": float(bad.mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("split")


def build_study_summary(checks_df: pd.DataFrame) -> pd.DataFrame:
    if "study_id" not in checks_df.columns:
        return pd.DataFrame()
    tmp = checks_df.copy()
    tmp["_bad"] = tmp["status"].isin(BAD_STATUSES)
    rows = []
    for (split, study_id), group in tmp.groupby(["split", "study_id"], dropna=False):
        if not group["_bad"].any() and (group["status"] == "ok").all():
            continue
        rows.append(
            {
                "split": split,
                "study_id": study_id,
                "image_rows": len(group),
                "bad_rows": int(group["_bad"].sum()),
                "all_views_bad": bool(group["_bad"].all()),
                "statuses": "|".join(sorted(group["status"].unique())),
                "paths": "|".join(group.loc[group["_bad"], "path"].astype(str).tolist()),
                "positive_labels": "|".join(
                    sorted(
                        {
                            label
                            for value in group["positive_labels"].dropna().astype(str)
                            for label in value.split("|")
                            if label
                        }
                    )
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "split",
                "study_id",
                "image_rows",
                "bad_rows",
                "all_views_bad",
                "statuses",
                "paths",
                "positive_labels",
            ]
        )
    return pd.DataFrame(rows).sort_values(["all_views_bad", "bad_rows"], ascending=[False, False])


def write_outputs(
    out_dir: Path,
    df: pd.DataFrame,
    checks: list[ImageCheck],
    labels: list[str],
    args: argparse.Namespace,
    csvs: list[Path],
    image_root: Path | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks_df = pd.DataFrame([asdict(check) for check in checks]).sort_values("row_id")

    bad_or_warning = checks_df[
        checks_df["status"].isin(BAD_STATUSES) | checks_df["decoder_warning"].astype(str).ne("")
    ].copy()
    bad_or_warning.to_csv(out_dir / "corrupt_or_warning_images.csv", index=False)

    if args.write_all:
        checks_df.to_csv(out_dir / "per_image_status.csv", index=False)

    split_summary = build_split_summary(checks_df)
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)

    study_summary = build_study_summary(checks_df)
    if not study_summary.empty:
        study_summary.to_csv(out_dir / "affected_studies.csv", index=False)

    label_impact = build_label_impact(df, checks_df, labels, args.positive_threshold)
    label_impact.to_csv(out_dir / "label_impact.csv", index=False)

    bad = checks_df["status"].isin(BAD_STATUSES)
    manifest = {
        "csvs": [str(p) for p in csvs],
        "image_root": str(image_root) if image_root is not None else None,
        "pipeline": args.pipeline,
        "decoder": args.decoder,
        "channel_mode": args.channel_mode,
        "image_channel_cache_dir": args.image_channel_cache_dir,
        "disable_channel_cache": args.disable_channel_cache,
        "size": args.size,
        "uint8_image_pipeline": args.uint8_image_pipeline,
        "positive_threshold": args.positive_threshold,
        "workers": args.workers,
        "n_rows": int(len(checks_df)),
        "n_lost_rows": int(bad.sum()),
        "n_decoder_warning_rows": int((checks_df["decoder_warning"].astype(str) != "").sum()),
        "n_labels": len(labels),
        "outputs": {
            "corrupt_or_warning_images": str(out_dir / "corrupt_or_warning_images.csv"),
            "split_summary": str(out_dir / "split_summary.csv"),
            "label_impact": str(out_dir / "label_impact.csv"),
            "affected_studies": str(out_dir / "affected_studies.csv") if not study_summary.empty else None,
            "per_image_status": str(out_dir / "per_image_status.csv") if args.write_all else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nWrote audit outputs:")
    for key, value in manifest["outputs"].items():
        if value:
            print(f"  {key}: {value}")
    print("\nSplit summary:")
    print(split_summary.to_string(index=False))
    print("\nMost impacted labels:")
    cols = [
        "label",
        "positive_image_rows",
        "bad_positive_image_rows",
        "positive_studies",
        "lost_positive_studies_all_views_bad",
    ]
    print(label_impact[cols].head(20).to_string(index=False))


def main() -> int:
    args = parse_args()
    try:
        datamodule_cfg = apply_config_defaults(args)
        csvs = discover_csvs(args, datamodule_cfg)
        image_root = infer_image_root(args, csvs)
        out_dir = default_out_dir(args, csvs)
        df, labels = load_inputs(csvs)
        if args.limit is not None:
            df = df.head(args.limit).copy()
        if "path" not in df.columns and "fpath" not in df.columns:
            raise ValueError("Input CSVs must contain a path or fpath column")
        if not labels:
            raise ValueError("Could not identify label columns in the input CSVs")
        if args.pipeline == "channels":
            from src.dataloader.utils import make_preprocess_config

            preprocess_cfg = make_preprocess_config({"size": args.size})
        else:
            preprocess_cfg = None

        print(f"CSV files: {', '.join(str(p) for p in csvs)}")
        print(f"Image root: {image_root if image_root is not None else '(path values only)'}")
        print(f"Pipeline: {args.pipeline}")
        if args.pipeline == "channels":
            cache = args.image_channel_cache_dir if args.image_channel_cache_dir else "(disabled)"
            print(f"Channel mode: {args.channel_mode} / size: {args.size} / cache: {cache}")
        else:
            print(f"Decoder: {args.decoder}")
        print(f"Rows: {len(df)} / labels: {len(labels)} / workers: {args.workers}")

        checks: list[ImageCheck] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(check_one, row, labels, image_root, args, preprocess_cfg)
                for _, row in df.iterrows()
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=args.pipeline):
                checks.append(future.result())

        write_outputs(out_dir, df, checks, labels, args, csvs, image_root)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
