"""Stage 4: pre-generate per-study parquet files with current + prior fields.

Reads the existence-filtered `data/data-camchex/03_<source>_{train,development,test}.csv`
files produced by `src/prepare/03_filter_existing_images.py` (row-per-image, with a
`PreviousStudy` column pointing to a prior study_id and `path` rewritten to point
at the chosen image source) and emits:

  data/data-camchex/prior_aware_{train,development,test}.parquet

One row per study. All groupby/fillna/path-resolution/tokenizer work happens
here so the runtime Dataset only has to decode JPEGs and apply transforms.

Defaults to the `mimic` source (03_mimic_*.csv) to match training/camchex/config.yaml.
Use --in-prefix 03_kaggle_ to build from the kaggle-hosted images instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.utils import resolve_preferred_image_path
from src.utils.text_embedding_cache import TextEmbeddingCache

CLASSES = [
    "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation",
    "Edema", "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "No Finding",
    "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta",
]
OBS_FIELDS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "gender"]
MAX_VIEWS = 4
CLIN_MAX_LEN = 384
OBS_MAX_LEN = 128


def _view_code(vp: object) -> int:
    if not isinstance(vp, str):
        return 0
    v = vp.upper()
    if v in ("AP", "PA", "FRONTAL"):
        return 1
    if v in ("LATERAL", "LL"):
        return 2
    return 0


def _label_vector(row: pd.Series) -> np.ndarray:
    """26-d label vector. CheXpert uncertain (-1) → 0.5, NaN → 0."""
    out = np.zeros(len(CLASSES), dtype=np.float32)
    for i, c in enumerate(CLASSES):
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


def _clin_text(row: pd.Series) -> str:
    text = row.get("clinical_indication", "")
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return "No clinical history available."
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


def _resolve_and_trim(paths: list, views: list, cap: int = MAX_VIEWS) -> tuple[list, list]:
    """Resolve preferred image paths and trim to <=cap, keeping aligned order."""
    if len(paths) > cap:
        paths = paths[:cap]
        views = views[:cap]
    resolved = [resolve_preferred_image_path(p) for p in paths]
    return resolved, list(map(int, views))


def _tokenize_batch(tokenizer, texts: list[str], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int8)


def build_split(
    csv_path: Path,
    out_path: Path,
    tokenizer,
    batch_size: int = 256,
    text_cache: TextEmbeddingCache | None = None,
) -> None:
    print(f"[build] reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    studies = collapse_to_study(df)
    print(f"[build] {len(df)} image rows -> {len(studies)} studies")

    # Build a lookup by study_id (int) for fast prior joins. Cast to int (PreviousStudy is float).
    studies["study_id"] = studies["study_id"].astype(np.int64)
    lookup = studies.set_index("study_id")
    has_prev = studies["PreviousStudy"].notna()
    studies["_prior_id"] = studies["PreviousStudy"].where(has_prev, other=-1).astype(np.int64)

    precompute_embeddings = text_cache is not None

    print("[build] preparing current clinical_indication...")
    clin_texts = studies.apply(_clin_text, axis=1).tolist()

    print("[build] preparing current vitals...")
    obs_texts = studies.apply(_obs_text, axis=1).tolist()

    # Prior text: pull from lookup when available, else use the same placeholder text the
    # current path uses for empties (kept consistent so empty != no-prior at the token level).
    print("[build] gathering prior text...")
    prior_clin_texts, prior_obs_texts, prior_has_text = [], [], []
    for pid in tqdm(studies["_prior_id"].values, dynamic_ncols=True):
        if pid >= 0 and pid in lookup.index:
            prow = lookup.loc[pid]
            if isinstance(prow, pd.DataFrame):  # duplicate study_id; take first
                prow = prow.iloc[0]
            prior_clin_texts.append(_clin_text(prow))
            prior_obs_texts.append(_obs_text(prow))
            prior_has_text.append(True)
        else:
            prior_clin_texts.append("No clinical history available.")
            prior_obs_texts.append(" | ".join([
                "Temperature: NA", "Heart rate: NA", "Respiratory rate: NA",
                "O2 Saturation: NA", "Systolic BP: NA", "Diastolic BP: NA", "Gender: NA",
            ]))
            prior_has_text.append(False)

    if precompute_embeddings:
        print("[build] embedding current clinical_indication...")
        clin_emb = text_cache.embed_texts(clin_texts, CLIN_MAX_LEN, desc="current clinical embeddings")
        print("[build] embedding current vitals...")
        obs_emb = text_cache.embed_texts(obs_texts, OBS_MAX_LEN, desc="current vitals embeddings")
        print("[build] embedding prior clinical_indication...")
        prior_clin_emb = text_cache.embed_texts(prior_clin_texts, CLIN_MAX_LEN, desc="prior clinical embeddings")
        print("[build] embedding prior vitals...")
        prior_obs_emb = text_cache.embed_texts(prior_obs_texts, OBS_MAX_LEN, desc="prior vitals embeddings")
    else:
        print("[build] tokenizing current clinical_indication...")
        clin_ids, clin_mask = _tokenize_batch(tokenizer, clin_texts, CLIN_MAX_LEN)
        print("[build] tokenizing current vitals...")
        obs_ids, obs_mask = _tokenize_batch(tokenizer, obs_texts, OBS_MAX_LEN)
        print("[build] tokenizing prior clinical_indication...")
        prior_clin_ids, prior_clin_mask = _tokenize_batch(tokenizer, prior_clin_texts, CLIN_MAX_LEN)
        print("[build] tokenizing prior vitals...")
        prior_obs_ids, prior_obs_mask = _tokenize_batch(tokenizer, prior_obs_texts, OBS_MAX_LEN)

    # Per-row resolved paths + views (current + prior), label vectors, time delta.
    print("[build] resolving paths and assembling rows...")
    studies["StudyDateTime"] = pd.to_datetime(studies["StudyDateTime"], errors="coerce")

    cur_paths, cur_views = [], []
    prv_paths, prv_views = [], []
    cur_labels, prv_labels = [], []
    has_prior, days_since = [], []
    prior_has_image = []

    for i in tqdm(range(len(studies)), dynamic_ncols=True):
        row = studies.iloc[i]
        p, v = _resolve_and_trim(row["img_paths_all"], row["view_codes_all"])
        cur_paths.append(p)
        cur_views.append(v)
        cur_labels.append(_label_vector(row))

        pid = int(row["_prior_id"])
        if pid >= 0 and pid in lookup.index:
            prow = lookup.loc[pid]
            if isinstance(prow, pd.DataFrame):
                prow = prow.iloc[0]
            pp, pv = _resolve_and_trim(prow["img_paths_all"], prow["view_codes_all"])
            prv_paths.append(pp)
            prv_views.append(pv)
            prv_labels.append(_label_vector(prow))
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
            prv_labels.append(np.zeros(len(CLASSES), dtype=np.float32))
            prior_has_image.append(False)
            has_prior.append(False)
            days_since.append(float("nan"))

    payload = {
        "study_id": studies["study_id"].astype(np.int64).values,
        "subject_id": studies["subject_id"].astype(np.int64).values,
        "img_paths": cur_paths,
        "view_positions": cur_views,
        "label": cur_labels,
        "has_prior": np.array(has_prior, dtype=bool),
        "prior_has_image": np.array(prior_has_image, dtype=bool),
        "days_since_prior": np.array(days_since, dtype=np.float32),
        "prior_img_paths": prv_paths,
        "prior_view_positions": prv_views,
        "prior_label": prv_labels,
    }
    if precompute_embeddings:
        payload.update({
            "clin_embedding": list(clin_emb),
            "obs_embedding": list(obs_emb),
            "prior_clin_embedding": list(prior_clin_emb),
            "prior_obs_embedding": list(prior_obs_emb),
        })
    else:
        payload.update({
            "clin_input_ids": list(clin_ids),
            "clin_attn_mask": list(clin_mask),
            "obs_input_ids": list(obs_ids),
            "obs_attn_mask": list(obs_mask),
            "prior_clin_input_ids": list(prior_clin_ids),
            "prior_clin_attn_mask": list(prior_clin_mask),
            "prior_obs_input_ids": list(prior_obs_ids),
            "prior_obs_attn_mask": list(prior_obs_mask),
        })
    out_df = pd.DataFrame(payload)

    print(f"[build] writing {out_path} ({len(out_df)} studies)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out_df, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")

    coverage = float(out_df["has_prior"].mean())
    print(f"[build] done. prior coverage = {coverage:.2%}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-generate prior-aware parquet datasets.")
    p.add_argument("--in-dir", default="data/data-camchex")
    p.add_argument("--out-dir", default="data/data-camchex")
    p.add_argument("--tokenizer", default="dmis-lab/biobert-v1.1")
    p.add_argument("--splits", nargs="+", default=["train", "development", "test"])
    p.add_argument(
        "--in-prefix",
        default="03_mimic_",
        help="Input CSV filename prefix. Default 03_mimic_ matches the output of src/prepare/03_filter_existing_images.py with --subset mimic.",
    )
    p.add_argument("--out-prefix", default="prior_aware_")
    p.add_argument(
        "--precompute-text-embeddings",
        action="store_true",
        help="Store frozen text CLS embeddings instead of token ids/attention masks.",
    )
    p.add_argument("--embedding-batch-size", type=int, default=32)
    p.add_argument(
        "--text-embedding-cache-dir",
        default="data/text_embeddings",
        help="Shared cache root for frozen text embeddings, grouped by embedding model.",
    )
    p.add_argument("--device", default="auto", help="Device for --precompute-text-embeddings: auto, cpu, cuda, etc.")
    return p.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    in_dir = (ROOT / args.in_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    tokenizer = None
    text_cache = None
    if args.precompute_text_embeddings:
        text_cache = TextEmbeddingCache(
            text_model=args.tokenizer,
            cache_root=args.text_embedding_cache_dir,
            batch_size=args.embedding_batch_size,
            device=args.device,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for split in args.splits:
        csv_path = in_dir / f"{args.in_prefix}{split}.csv"
        out_path = out_dir / f"{args.out_prefix}{split}.parquet"
        build_split(
            csv_path,
            out_path,
            tokenizer,
            text_cache=text_cache,
        )


if __name__ == "__main__":
    main()
