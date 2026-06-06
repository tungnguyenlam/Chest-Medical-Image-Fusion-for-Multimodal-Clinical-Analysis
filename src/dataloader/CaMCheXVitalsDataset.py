import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _log_dataset_warning(msg):
    """Emit a dataset warning without letting the tqdm bar clobber it.

    Best-effort only: any failure here (e.g. closed stream in a worker) is
    swallowed so it can never interrupt or crash the training loop.
    """
    try:
        from tqdm import tqdm

        tqdm.write(msg)
    except Exception:
        try:
            logger.warning(msg)
        except Exception:
            pass

from src.dataloader.utils import (
    _safe_decode_jpeg,
    load_cached_rgb,
    load_or_build_channels,
    make_preprocess_config,
    resolve_preferred_image_path,
)


ROOT = Path(__file__).resolve().parents[2]
VITAL_FIELDS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "gender"]
DEFAULT_VITAL_STATS = {
    "temperature": {"mean": 98.6, "std": 3.0},
    "heartrate": {"mean": 85.0, "std": 25.0},
    "resprate": {"mean": 18.0, "std": 6.0},
    "o2sat": {"mean": 96.0, "std": 5.0},
    "sbp": {"mean": 120.0, "std": 25.0},
    "dbp": {"mean": 70.0, "std": 15.0},
    "gender": {"mean": 0.5, "std": 0.5},
}


class CaMCheXVitalsDataset(Dataset):
    def __init__(self, cfg, df, transform=None, tokenizer=None):
        self.cfg = cfg
        self.transform = transform
        self.tokenizer = tokenizer
        self.df = df.groupby("study_id")
        self.study_ids = list(self.df.groups.keys())
        self.vital_stats = {**DEFAULT_VITAL_STATS, **dict(cfg.get("vital_stats", {}) or {})}
        self.clinical_embeddings = self._load_clinical_embeddings(cfg.get("clinical_embeddings"))
        self.clinical_embedding_cache = cfg.get("clinical_embedding_cache")
        self.image_cache_dir = cfg.get("image_cache_dir")
        self.channel_mode = cfg.get("channel_mode")
        self.channel_cache_dir = cfg.get("image_channel_cache_dir")
        self.channel_cfg = make_preprocess_config(cfg) if self.channel_mode else None

    def _resolve_path(self, path):
        p = Path(path)
        if p.is_absolute() or p.exists():
            return p
        return ROOT / p

    def _load_clinical_embeddings(self, embeddings):
        if not embeddings:
            return None
        return {str(k): np.asarray(v, dtype=np.float32) for k, v in embeddings.items()}

    def __len__(self):
        return len(self.study_ids)

    def _encode_gender(self, value):
        if pd.isna(value):
            return None
        value = str(value).strip().upper()
        if value in {"M", "MALE"}:
            return 1.0
        if value in {"F", "FEMALE"}:
            return 0.0
        return None

    def _encode_vitals(self, row):
        values = []
        missing = []
        for field in VITAL_FIELDS:
            raw_value = self._encode_gender(row.get(field)) if field == "gender" else row.get(field)
            value = pd.to_numeric(raw_value, errors="coerce")
            if pd.isna(value):
                values.append(0.0)
                missing.append(True)
                continue
            stats = self.vital_stats[field]
            std = float(stats.get("std", 1.0)) or 1.0
            values.append((float(value) - float(stats.get("mean", 0.0))) / std)
            missing.append(False)
        return np.array(values, dtype=np.float32), np.array(missing, dtype=np.bool_)

    def _clinical_text(self, row):
        text = row.get("clinical_indication", "")
        if pd.isna(text) or str(text).strip() == "":
            return "No clinical history available."
        return str(text)

    def _encode_clinical_text(self, study_id, row):
        if self.clinical_embeddings is not None:
            embedding = self.clinical_embeddings.get(str(study_id))
            if embedding is not None:
                return embedding.astype(np.float32), np.zeros(1, dtype=np.int64)
        clinical_text = self._clinical_text(row)
        if self.clinical_embedding_cache is not None:
            embedding = self.clinical_embedding_cache.get_embedding(clinical_text, max_length=384)
            return embedding.astype(np.float32, copy=False), np.zeros(1, dtype=np.int64)
        if self.tokenizer is None:
            raise RuntimeError(
                f"Missing precomputed clinical embedding for study {study_id}; "
                "disable use_precomputed_text_embeddings or rebuild the shared text embedding cache."
            )

        clinical_tokens = self.tokenizer(
            clinical_text,
            padding="max_length",
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        return clinical_tokens["input_ids"].squeeze(0), clinical_tokens["attention_mask"].squeeze(0)

    def __getitem__(self, index):
        df = self.df.get_group(self.study_ids[index])
        study_id = self.study_ids[index]
        if len(df) > 4:
            df = df.sample(4)

        if all([c in df.columns for c in self.cfg["classes"]]):
            label = df[self.cfg["classes"]].iloc[0].to_numpy().astype(np.float32)
        else:
            label = np.zeros(len(self.cfg["classes"]))

        imgs = []
        view_positions = []
        for i in range(len(df)):
            path = df.iloc[i]["path"]

            if self.channel_mode:
                # Pass the raw path: a cache hit is keyed on the string alone and
                # never stats the (slow) source FS; resolution happens on a miss.
                img = load_or_build_channels(path, self.channel_mode, self.channel_cfg, self.channel_cache_dir)
            else:
                resolved = resolve_preferred_image_path(path)
                img = load_cached_rgb(self.image_cache_dir, resolved)
                if img is None:
                    img = _safe_decode_jpeg(resolved)
            if img is None:
                _log_dataset_warning(f"Skipping unreadable image {path} in study {study_id}")
                continue

            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]
                img = np.moveaxis(img, -1, 0)

            imgs.append(img)

            vp = df.iloc[i].get("ViewPosition", "")
            vp = "" if not isinstance(vp, str) else vp.upper()
            if vp in ["AP", "PA", "FRONTAL"]:
                view_positions.append(1)
            elif vp in ["LATERAL", "LL"]:
                view_positions.append(2)
            else:
                view_positions.append(0)

        if len(imgs) == 0:
            _log_dataset_warning(f"All images unreadable for study {study_id}; using neighbor study")
            return self.__getitem__((index + 1) % len(self.study_ids))

        n = len(imgs)
        img = np.stack(imgs, axis=0)
        img = np.concatenate([img, np.zeros((4 - n, 3, self.cfg["size"], self.cfg["size"]))], axis=0).astype(np.float32)
        view_positions = np.array(view_positions + [0] * (4 - n), dtype=np.int64)

        row = df.iloc[0]
        clinical_input, clinical_attention_mask = self._encode_clinical_text(study_id, row)
        vital_values, vital_missing_mask = self._encode_vitals(row)

        return (
            study_id,
            img,
            view_positions,
            clinical_input,
            clinical_attention_mask,
            vital_values,
            vital_missing_mask,
        ), label
