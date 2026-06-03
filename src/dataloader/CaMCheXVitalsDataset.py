import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataloader.utils import _safe_decode_jpeg, load_cached_rgb, resolve_preferred_image_path


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
        assert tokenizer is not None, "Tokenizer must be provided for this dataset."
        self.tokenizer = tokenizer
        self.df = df.groupby("study_id")
        self.study_ids = list(self.df.groups.keys())
        self.vital_stats = {**DEFAULT_VITAL_STATS, **dict(cfg.get("vital_stats", {}) or {})}
        self.clinical_embeddings = self._load_clinical_embeddings(cfg.get("clinical_embedding_path"))
        self.image_cache_dir = cfg.get("image_cache_dir")

    def _resolve_path(self, path):
        p = Path(path)
        if p.is_absolute() or p.exists():
            return p
        return ROOT / p

    def _load_clinical_embeddings(self, path):
        if not path:
            return None
        payload = torch.load(self._resolve_path(path), map_location="cpu")
        embeddings = payload.get("embeddings", payload) if isinstance(payload, dict) else payload
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

    def _encode_clinical_text(self, study_id, row):
        if self.clinical_embeddings is not None:
            embedding = self.clinical_embeddings.get(str(study_id))
            if embedding is not None:
                return embedding.astype(np.float32), np.zeros(1, dtype=np.int64)

        clinical_text = row.get("clinical_indication", "")
        if pd.isna(clinical_text) or str(clinical_text).strip() == "":
            clinical_text = "No clinical history available."
        else:
            clinical_text = str(clinical_text)
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
            path = resolve_preferred_image_path(path)

            img = load_cached_rgb(self.image_cache_dir, path)
            if img is None:
                img = _safe_decode_jpeg(path)
            if img is None:
                warnings.warn(f"Skipping unreadable image {path} in study {study_id}")
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
            warnings.warn(f"All images unreadable for study {study_id}; using neighbor study")
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
