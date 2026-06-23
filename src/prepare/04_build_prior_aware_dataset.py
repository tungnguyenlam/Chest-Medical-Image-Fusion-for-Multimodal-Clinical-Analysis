"""Stage 4: pre-generate per-study parquet files with current + prior fields.

Reads the existence-filtered `data/data-camchex/03_<source>_{train,development,test}.csv`
files produced by `src/prepare/03_filter_existing_images.py` (row-per-image, with a
`PreviousStudy` column pointing to a prior study_id and `path` rewritten to point
at the chosen image source) and emits:

  data/data-camchex/prior_aware_{train,development,test}.parquet

One row per study. Groupby/path-resolution happens here; the parquet stores the
RAW text columns (no baked token ids). The runtime Dataset decodes JPEGs, applies
transforms, and tokenizes the raw text at load time with the training config's
tokenizer (so one parquet serves any text model -- BioBERT, CXR-BERT, etc. -- with
no rebuild), or reads frozen CLS embeddings from the shared cache when the text
encoder is frozen.

Defaults to the `mimic` source (03_mimic_*.csv) to match training/camchex/config.yaml.
Use --in-prefix 03_kaggle_ to build from the kaggle-hosted images instead.

Label space is selectable, and by default it builds *every* CXR-LT release/task
variant in one pass. Invoked with no ``--cxr-lt-version`` flag, it reads the
label-agnostic existence-filtered base CSVs once, then for each variant below it
re-attaches that release's labels, re-splits train/val/test the way that release
defines them (CXR-LT 2024 reassigns splits relative to 2023), and writes a parquet
trio plus a ``*_label_metadata.json`` sidecar:

  cxr-lt-2023 / standard (26) -> prior_aware_{train,development,test}.parquet
  cxr-lt-2024 / all      (45) -> prior_aware_cxrlt2024_all_{train,val,test}.parquet
  cxr-lt-2024 / task1    (40) -> prior_aware_cxrlt2024_task1_{train,val,test}.parquet
  cxr-lt-2024 / task2    (26) -> prior_aware_cxrlt2024_task2_{train,val,test}.parquet
  cxr-lt-2024 / task3     (5) -> prior_aware_cxrlt2024_task3_{train,val,test}.parquet

Variants whose label files are absent under ``data/CXR-LT/.../<version>`` are
skipped with a warning, so a checkout with only the 2023 release still works. The
labeled/re-split CSVs are also written under ``<out-dir>/<variant>/`` (these are the
same artifacts ``scripts/relabel_prepared_cxrlt.py`` used to produce by hand).

To rebuild just one variant, pass ``--cxr-lt-version cxr-lt-2024 --cxr-lt-label-set
task1`` (single-variant mode); it then reads the pre-split CSVs named by
``--in-prefix``/``--splits`` instead of re-splitting the base. The 2024 images are
identical to 2023, so the expensive image-channel and text-embedding caches are
keyed by path/content and reused verbatim -- only these lightweight parquets are
rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.cxr_lt import (
    CXRLT_2023_LABELS,
    CXRLT_2024_ALL_LABELS,
    cxr_lt_classes,
    load_cxr_lt_labels,
    resolve_label_set,
)
from src.dataloader.utils import resolve_preferred_image_path

# Union of every label column any CXR-LT release/task can carry. Used to strip
# stale labels off the base CSVs before re-attaching a variant's own labels.
KNOWN_LABEL_COLS = sorted(set(CXRLT_2023_LABELS) | set(CXRLT_2024_ALL_LABELS))

# Every variant the no-flag default build emits. Each entry is
# (version, label_set, out_prefix, labeled_subdir, {split_value: output_name}).
# CXR-LT 2024 names its dev split "val"; 2023 keeps the historic "development".
ALL_VARIANTS = [
    ("cxr-lt-2023", "standard", "prior_aware_", "cxrlt2023",
     {"train": "train", "validate": "development", "test": "test"}),
    ("cxr-lt-2024", "all", "prior_aware_cxrlt2024_all_", "cxrlt2024_all",
     {"train": "train", "validate": "val", "test": "test"}),
    ("cxr-lt-2024", "task1", "prior_aware_cxrlt2024_task1_", "cxrlt2024_task1",
     {"train": "train", "validate": "val", "test": "test"}),
    ("cxr-lt-2024", "task2", "prior_aware_cxrlt2024_task2_", "cxrlt2024_task2",
     {"train": "train", "validate": "val", "test": "test"}),
    ("cxr-lt-2024", "task3", "prior_aware_cxrlt2024_task3_", "cxrlt2024_task3",
     {"train": "train", "validate": "val", "test": "test"}),
]

OBS_FIELDS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "gender"]
MAX_VIEWS = 4
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
# The prior study's full radiology report (findings + impression) is fed only for
# the PRIOR study: it was authored before the current exam, so it is legitimate
# prior information rather than label leakage (unlike the current study's report,
# which directly states the labels and is never used). Token budgets (clinical 384,
# vitals 128, report 384) now live on PriorAwareDataset, which tokenizes at load.
NO_PRIOR_REPORT = "No prior report available."


def _view_code(vp: object) -> int:
    if not isinstance(vp, str):
        return 0
    v = vp.upper()
    if v in ("AP", "PA", "FRONTAL"):
        return 1
    if v in ("LATERAL", "LL"):
        return 2
    return 0


def _label_vector(row: pd.Series, classes: list[str]) -> np.ndarray:
    """Per-class label vector over ``classes``. CheXpert uncertain (-1) → 0.5, NaN → 0."""
    out = np.zeros(len(classes), dtype=np.float32)
    for i, c in enumerate(classes):
        v = row.get(c)
        if pd.isna(v):
            continue
        v = float(v)
        if v == -1.0:
            out[i] = 0.5
        elif v > 0:
            out[i] = 1.0
        else:
            out[i] = 0.0
    return out


def _obs_text(row: pd.Series) -> str:
    vals = {f: (str(row.get(f)) if not pd.isna(row.get(f)) else "NA") for f in OBS_FIELDS}
    return " | ".join([
        f"Temperature: {vals['temperature']}",
        f"Heart rate: {vals['heartrate']}",
        f"Respiratory rate: {vals['resprate']}",
        f"O2 Saturation: {vals['o2sat']}",
        f"Systolic BP: {vals['sbp']}",
        f"Diastolic BP: {vals['dbp']}",
        f"Gender: {vals['gender']}",
    ])


def _vital_vector(row: pd.Series) -> np.ndarray:
    out = np.zeros(len(OBS_FIELDS), dtype=np.float32)
    out.fill(np.nan)
    for i, field in enumerate(OBS_FIELDS):
        raw_value = row.get(field)
        if field == "gender" and not pd.isna(raw_value):
            raw = str(raw_value).strip().upper()
            if raw in {"M", "MALE"}:
                out[i] = 1.0
            elif raw in {"F", "FEMALE"}:
                out[i] = 0.0
            continue
        value = pd.to_numeric(raw_value, errors="coerce")
        if not pd.isna(value):
            out[i] = float(value)
    return out


def _clin_text(row: pd.Series) -> str:
    text = row.get("clinical_indication", "")
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return "No clinical history available."
    return text


def _report_text(row: pd.Series) -> str:
    """The radiology report (findings + impression), parsed in stage 01 into the
    ``report`` column. Used only for the PRIOR study (see REPORT_MAX_LEN note)."""
    text = row.get("report", "")
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return NO_PRIOR_REPORT
    return text


def collapse_to_study(df: pd.DataFrame) -> pd.DataFrame:
    """One row per study_id: pick the first row for scalar fields and aggregate paths/views."""
    df = df.copy()
    df["view_code"] = df["ViewPosition"].apply(_view_code).astype(np.int8)

    groups = df.groupby("study_id", sort=False)
    head = groups.head(1).set_index("study_id")

    paths = groups["path"].apply(list)
    views = groups["view_code"].apply(list)

    out = head.copy()
    out["img_paths_all"] = paths
    out["view_codes_all"] = views
    out = out.reset_index(drop=False)
    return out


def _collect_unique_image_paths(studies: pd.DataFrame, cap: int = MAX_VIEWS) -> list[str]:
    """Every distinct image path that the study loop will resolve, deduped, no I/O.

    Prior views are a subset of these same studies' paths (the prior lookup indexes
    into ``studies``), so collecting from ``img_paths_all`` covers current + prior."""
    seen: dict[str, None] = {}
    for paths in studies["img_paths_all"]:
        for p in paths[:cap]:
            seen.setdefault(str(p), None)
    return list(seen)


def _bulk_resolve_preferred(paths: list[str], workers: int) -> dict[str, str]:
    """Resolve every unique path once, in parallel. Returns {raw_path: resolved_path}.

    The per-path work is filesystem ``stat`` (Path.exists), which releases the GIL, so
    threads parallelize it well -- mirroring classify_images in 03_filter_existing_images.
    resolve_preferred_image_path is itself lru_cached, so a repeat (e.g. across CXR-LT
    variants) is a dict hit, not a re-stat."""
    if not paths:
        return {}
    if workers <= 1 or len(paths) <= 1:
        return {p: resolve_preferred_image_path(p) for p in tqdm(paths, desc="resolve paths", dynamic_ncols=True)}
    chunksize = max(1, min(256, len(paths) // (workers * 4) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        resolved = list(tqdm(
            ex.map(resolve_preferred_image_path, paths, chunksize=chunksize),
            total=len(paths), desc="resolve paths", dynamic_ncols=True,
        ))
    return dict(zip(paths, resolved))


def _resolve_and_trim(
    paths: list, views: list, resolved_map: dict[str, str] | None = None, cap: int = MAX_VIEWS,
) -> tuple[list, list]:
    """Trim to <=cap and map each path to its resolved form, keeping aligned order.

    With ``resolved_map`` (the bulk-resolve path), lookups are O(1) and never touch the
    filesystem. With ``None`` (the --store-raw-paths fast path) the raw path strings are
    passed through unchanged -- training resolves them lazily on cache miss."""
    if len(paths) > cap:
        paths = paths[:cap]
        views = views[:cap]
    if resolved_map is None:
        resolved = [str(p) for p in paths]
    else:
        resolved = [resolved_map[str(p)] for p in paths]
    return resolved, list(map(int, views))


def build_split(
    csv_path: Path,
    out_path: Path,
    classes: list[str],
    workers: int = DEFAULT_WORKERS,
    store_raw_paths: bool = False,
) -> dict:
    """Build one split parquet from a CSV on disk. Thin wrapper over
    :func:`build_split_df` for the single-variant (pre-split CSV) path."""
    print(f"[build] reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in classes if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing {len(missing)} label column(s) for the selected "
            f"label set: {missing}. Did you point --in-dir/--cxr-lt-label-set at the "
            f"matching relabeled CSVs (see scripts/relabel_prepared_cxrlt.py)?"
        )
    return build_split_df(df, out_path, classes, workers=workers, store_raw_paths=store_raw_paths)


def build_split_df(
    df: pd.DataFrame,
    out_path: Path,
    classes: list[str],
    workers: int = DEFAULT_WORKERS,
    store_raw_paths: bool = False,
) -> dict:
    """Build one split parquet from an in-memory frame. Returns a small stats dict
    (study count + per-class positive counts) so the caller can record loss-weight
    metadata. ``df`` must already carry every column in ``classes``."""
    studies = collapse_to_study(df)
    print(f"[build] {len(df)} image rows -> {len(studies)} studies")

    # Build a lookup by study_id (int) for fast prior joins. Cast to int (PreviousStudy is float).
    studies["study_id"] = studies["study_id"].astype(np.int64)
    lookup = studies.set_index("study_id")
    has_prev = studies["PreviousStudy"].notna()
    studies["_prior_id"] = studies["PreviousStudy"].where(has_prev, other=-1).astype(np.int64)

    print("[build] preparing current clinical_indication...")
    clin_texts = studies.apply(_clin_text, axis=1).tolist()

    print("[build] preparing current vitals...")
    obs_texts = studies.apply(_obs_text, axis=1).tolist()

    # Prior text: pull from lookup when available, else use the same placeholder text the
    # current path uses for empties (kept consistent so empty != no-prior at the token level).
    print("[build] gathering prior text...")
    prior_clin_texts, prior_obs_texts, prior_report_texts, prior_has_text = [], [], [], []
    for pid in tqdm(studies["_prior_id"].values, dynamic_ncols=True):
        if pid >= 0 and pid in lookup.index:
            prow = lookup.loc[pid]
            if isinstance(prow, pd.DataFrame):  # duplicate study_id; take first
                prow = prow.iloc[0]
            prior_clin_texts.append(_clin_text(prow))
            prior_obs_texts.append(_obs_text(prow))
            prior_report_texts.append(_report_text(prow))
            prior_has_text.append(True)
        else:
            prior_clin_texts.append("No clinical history available.")
            prior_obs_texts.append(" | ".join([
                "Temperature: NA", "Heart rate: NA", "Respiratory rate: NA",
                "O2 Saturation: NA", "Systolic BP: NA", "Diastolic BP: NA", "Gender: NA",
            ]))
            prior_report_texts.append(NO_PRIOR_REPORT)
            prior_has_text.append(False)

    # Per-row resolved paths + views (current + prior), label vectors, time delta.
    if store_raw_paths:
        resolved_map: dict[str, str] | None = None
        print("[build] storing raw image paths (--store-raw-paths; resolution deferred to training)...")
    else:
        unique_paths = _collect_unique_image_paths(studies)
        print(f"[build] resolving {len(unique_paths)} unique image paths with {workers} workers...")
        resolved_map = _bulk_resolve_preferred(unique_paths, workers)
    print("[build] assembling rows...")
    studies["StudyDateTime"] = pd.to_datetime(studies["StudyDateTime"], errors="coerce")

    cur_paths, cur_views = [], []
    prv_paths, prv_views = [], []
    cur_labels, prv_labels = [], []
    cur_vitals, prv_vitals = [], []
    has_prior, days_since = [], []
    prior_has_image = []

    for i in tqdm(range(len(studies)), dynamic_ncols=True):
        row = studies.iloc[i]
        p, v = _resolve_and_trim(row["img_paths_all"], row["view_codes_all"], resolved_map)
        cur_paths.append(p)
        cur_views.append(v)
        cur_labels.append(_label_vector(row, classes))
        cur_vitals.append(_vital_vector(row))

        pid = int(row["_prior_id"])
        if pid >= 0 and pid in lookup.index:
            prow = lookup.loc[pid]
            if isinstance(prow, pd.DataFrame):
                prow = prow.iloc[0]
            pp, pv = _resolve_and_trim(prow["img_paths_all"], prow["view_codes_all"], resolved_map)
            prv_paths.append(pp)
            prv_views.append(pv)
            prv_labels.append(_label_vector(prow, classes))
            prv_vitals.append(_vital_vector(prow))
            prior_has_image.append(len(pp) > 0)
            has_prior.append(True)
            cur_dt = row["StudyDateTime"]
            prv_dt = prow.get("StudyDateTime")
            prv_dt = pd.to_datetime(prv_dt, errors="coerce") if not isinstance(prv_dt, pd.Timestamp) else prv_dt
            if pd.notna(cur_dt) and pd.notna(prv_dt):
                days_since.append(float((cur_dt - prv_dt).total_seconds()) / 86400.0)
            else:
                days_since.append(float("nan"))
        else:
            prv_paths.append([])
            prv_views.append([])
            prv_labels.append(np.zeros(len(classes), dtype=np.float32))
            prv_vitals.append(np.full(len(OBS_FIELDS), np.nan, dtype=np.float32))
            prior_has_image.append(False)
            has_prior.append(False)
            days_since.append(float("nan"))

    payload = {
        "study_id": studies["study_id"].astype(np.int64).values,
        "subject_id": studies["subject_id"].astype(np.int64).values,
        "img_paths": cur_paths,
        "view_positions": cur_views,
        "label": cur_labels,
        "vital_values_raw": cur_vitals,
        "has_prior": np.array(has_prior, dtype=bool),
        "prior_has_image": np.array(prior_has_image, dtype=bool),
        "days_since_prior": np.array(days_since, dtype=np.float32),
        "prior_img_paths": prv_paths,
        "prior_view_positions": prv_views,
        "prior_label": prv_labels,
        "prior_vital_values_raw": prv_vitals,
        "clin_text": clin_texts,
        "obs_text": obs_texts,
        "prior_clin_text": prior_clin_texts,
        "prior_obs_text": prior_obs_texts,
        "prior_report_text": prior_report_texts,
    }
    out_df = pd.DataFrame(payload)

    print(f"[build] writing {out_path} ({len(out_df)} studies)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out_df, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")

    coverage = float(out_df["has_prior"].mean())
    print(f"[build] done. prior coverage = {coverage:.2%}")

    # Per-class positive counts (label == 1.0) over this split's current studies.
    # The loss's class_instance_nums wants the *train* split's positives; emitting
    # them here makes a 2024 config trivial to fill for any model.
    label_mat = np.stack(cur_labels) if cur_labels else np.zeros((0, len(classes)), np.float32)
    pos_counts = (label_mat == 1.0).sum(axis=0).astype(int).tolist()
    return {"num_studies": int(len(out_df)), "positive_counts": pos_counts}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-generate prior-aware parquet datasets.")
    p.add_argument("--in-dir", default="data/data-camchex")
    p.add_argument("--out-dir", default="data/data-camchex")
    p.add_argument("--data-root", default="data",
                   help="Root holding data/CXR-LT/... (used to find each release's label CSVs in build-all mode).")
    p.add_argument("--splits", nargs="+", default=["train", "development", "test"])
    p.add_argument(
        "--in-prefix",
        default="03_mimic_",
        help="Input CSV filename prefix. Default 03_mimic_ matches the output of src/prepare/03_filter_existing_images.py with --subset mimic. For relabeled 2024 CSVs (train.csv/val.csv/test.csv) pass --in-prefix '' --splits train val test.",
    )
    p.add_argument("--out-prefix", default="prior_aware_")
    p.add_argument(
        "--cxr-lt-version",
        default=None,
        choices=["cxr-lt-2023", "cxr-lt-2024"],
        help="CXR-LT release to bake into the parquet. Omit (the default) to build "
             "EVERY release/task variant in one pass from the label-agnostic base "
             "CSVs (--in-prefix). Pass a version for single-variant mode, which "
             "reads the pre-split CSVs named by --in-prefix/--splits.",
    )
    p.add_argument(
        "--cxr-lt-label-set",
        default="auto",
        choices=["auto", "all", "task1", "task2", "task3", "standard"],
        help="Label set within the release (single-variant mode only). 'auto' = standard for 2023, task1 for 2024.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers for bulk image-path resolution (Path.exists stats release "
             f"the GIL, so threads help on slow mounts). Default: min(16, cpu_count) = {DEFAULT_WORKERS}.",
    )
    p.add_argument(
        "--store-raw-paths",
        action="store_true",
        help="Skip image-path resolution entirely and write the stage-03 CSV path strings "
             "into the parquet as-is. Training resolves + decodes lazily on channel-cache "
             "miss, so this is safe and turns the path step into pure pandas work. Trade-off: "
             "no baked _resized_1024.jpg preference (usually absent on full-res MIMIC anyway).",
    )
    return p.parse_args()


def read_base_csvs(in_dir: Path, prefix: str, splits: list[str]) -> pd.DataFrame:
    """Combine the label-agnostic existence-filtered base CSVs into one frame, drop
    any stale label/split columns, and de-dup by dicom_id. Each variant re-attaches
    its own labels + split onto this base."""
    frames = []
    for split in splits:
        path = in_dir / f"{prefix}{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing base CSV: {path}")
        frames.append(pd.read_csv(path, low_memory=False))
    base = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["dicom_id"], keep="first")
    drop_cols = [c for c in [*KNOWN_LABEL_COLS, "split"] if c in base.columns]
    return base.drop(columns=drop_cols)


def write_variant_metadata(meta_path: Path, version: str, label_set: str,
                           classes: list[str], split_stats: dict) -> None:
    metadata = {
        "cxr_lt_version": version,
        "cxr_lt_label_set": label_set,
        "num_classes": len(classes),
        "classes": classes,
        "splits": {
            split: {
                "num_studies": stats["num_studies"],
                "class_instance_nums": stats["positive_counts"],
                "total_instance_num": int(sum(stats["positive_counts"])),
            }
            for split, stats in split_stats.items()
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[build] metadata -> {meta_path}")


def build_all_variants(in_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    """No-flag default: build every CXR-LT release/task variant from the base CSVs."""
    base = read_base_csvs(in_dir, args.in_prefix, args.splits)
    print(f"[build-all] base: {len(base)} unique images from {args.in_prefix}{{{','.join(args.splits)}}}.csv")

    data_root = (ROOT / args.data_root).resolve()
    for version, label_set, out_prefix, labeled_subdir, split_files in ALL_VARIANTS:
        try:
            labels, classes, resolved = load_cxr_lt_labels(
                data_root, version=version, label_set=label_set
            )
        except FileNotFoundError as exc:
            print(f"[build-all] skipping {version}/{label_set}: {exc}")
            continue

        print(f"[build-all] === {version}/{resolved} ({len(classes)} classes) ===")
        # Re-attach this release's labels + split onto the base by dicom_id. Inner
        # join drops base images this release does not label.
        labels = labels.drop(
            columns=["subject_id", "study_id", "ViewPosition",
                     "ViewCodeSequence_CodeMeaning", "path", "fpath"],
            errors="ignore",
        )
        relabeled = base.merge(labels, on="dicom_id", how="inner", validate="one_to_one")

        labeled_dir = out_dir / labeled_subdir
        labeled_dir.mkdir(parents=True, exist_ok=True)
        split_stats: dict[str, dict] = {}
        for split_value, out_name in split_files.items():
            sub = relabeled[relabeled["split"] == split_value].drop(columns=["split"])
            sub.to_csv(labeled_dir / f"{out_name}.csv", index=False)
            parquet_path = out_dir / f"{out_prefix}{out_name}.parquet"
            split_stats[out_name] = build_split_df(
                sub, parquet_path, classes,
                workers=args.workers, store_raw_paths=args.store_raw_paths,
            )

        write_variant_metadata(
            out_dir / f"{out_prefix}label_metadata.json",
            version, resolved, classes, split_stats,
        )


def main() -> None:
    args = parse_args()
    in_dir = (ROOT / args.in_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()

    if args.cxr_lt_version is None:
        build_all_variants(in_dir, out_dir, args)
        return

    resolved_label_set = resolve_label_set(args.cxr_lt_version, args.cxr_lt_label_set)
    classes = cxr_lt_classes(args.cxr_lt_version, args.cxr_lt_label_set)
    print(
        f"[build] label space: {args.cxr_lt_version} / {resolved_label_set} "
        f"({len(classes)} classes)"
    )

    split_stats: dict[str, dict] = {}
    for split in args.splits:
        csv_path = in_dir / f"{args.in_prefix}{split}.csv"
        out_path = out_dir / f"{args.out_prefix}{split}.parquet"
        split_stats[split] = build_split(
            csv_path, out_path, classes,
            workers=args.workers, store_raw_paths=args.store_raw_paths,
        )

    # Sidecar metadata: the resolved class order plus per-split positive counts, so
    # the loss's class_instance_nums (train positives) is copy-pasteable into any
    # model config for this label space.
    write_variant_metadata(
        out_dir / f"{args.out_prefix}label_metadata.json",
        args.cxr_lt_version, resolved_label_set, classes, split_stats,
    )


if __name__ == "__main__":
    main()
