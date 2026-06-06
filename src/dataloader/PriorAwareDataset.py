"""Prior-aware dataset.

Loads a pre-generated parquet (see src/prepare/04_build_prior_aware_dataset.py) and
only does JPEG decode + transforms in __getitem__. No groupby, no fillna,
no tokenizer call, no path resolution at runtime.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataloader.CaMCheXVitalsDataset import DEFAULT_VITAL_STATS, VITAL_FIELDS
from src.dataloader.utils import (
    _safe_decode_jpeg,
    load_or_build_channels,
    make_preprocess_config,
)

MAX_VIEWS = 4
N_CLASSES = 26
CLIN_MAX_LEN = 384
OBS_MAX_LEN = 128


def _to_array(x, dtype):
    """parquet list/array → np.ndarray with the right dtype."""
    if x is None:
        return np.zeros(0, dtype=dtype)
    return np.asarray(list(x), dtype=dtype)


def _has_value(row, key: str) -> bool:
    return key in row.index and row[key] is not None


def _text_array(row, embedding_key: str, token_key: str, dtype):
    if _has_value(row, embedding_key):
        return _to_array(row[embedding_key], np.float32)
    return _to_array(row[token_key], dtype)


def _text_value(row, text_key: str, fallback: str) -> str:
    if text_key not in row.index:
        raise KeyError(
            f"Prior-aware parquet is missing {text_key!r}; rebuild it with "
            "src/prepare/04_build_prior_aware_dataset.py before using the training-time text embedding cache."
        )
    text = row[text_key]
    if pd.isna(text) or str(text).strip() == "":
        return fallback
    return str(text)


def _zero_image_block(size: int) -> np.ndarray:
    return np.zeros((MAX_VIEWS, 3, size, size), dtype=np.float32)


def _nan_vitals() -> np.ndarray:
    out = np.zeros(len(VITAL_FIELDS), dtype=np.float32)
    out.fill(np.nan)
    return out


class PriorAwareDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        image_size: int,
        transform=None,
        label_dropout_p: float = 0.0,
        cfg: dict | None = None,
    ):
        """
        Args:
            parquet_path: path to a prior_aware_*.parquet built by the script.
            image_size: HxW the transform pipeline resizes to (matches data_cfg["size"]).
            transform: Albumentations Compose for image augmentation. Applied per-image.
            label_dropout_p: training-only probability to drop the entire prior block.
            cfg: datamodule cfg. ``channel_mode`` selects the 3-channel CXR build
                (raw+CLAHE+third channel) shared with the other datasets; left
                unset (or ``--channel-mode none``) it keeps the legacy direct JPEG
                decode -- i.e. the plain grayscale-duplicated-to-3-channels image.
        """
        super().__init__()
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.size = int(image_size)
        self.transform = transform
        self.label_dropout_p = float(label_dropout_p)
        cfg = cfg or {}
        self.channel_mode = cfg.get("channel_mode")
        self.channel_cache_dir = cfg.get("image_channel_cache_dir")
        self.channel_cfg = make_preprocess_config(cfg) if self.channel_mode else None
        self.text_embedding_cache = cfg.get("text_embedding_cache")
        self.text_embedding_streams = set(
            cfg.get("text_embedding_streams")
            or ["clin_text", "obs_text", "prior_clin_text", "prior_obs_text"]
        )
        self.vital_stats = {**DEFAULT_VITAL_STATS, **dict(cfg.get("vital_stats", {}) or {})}

    def _decode(self, path: str):
        """Decode one image: built 3-channel array when channel_mode is set, else
        the legacy direct RGB decode (plain 3-channel duplicate)."""
        if self.channel_mode:
            # Raw path -> cache hit is string-keyed, no source-FS stat; the file
            # is resolved + decoded only on a miss inside load_or_build_channels.
            return load_or_build_channels(
                path, self.channel_mode, self.channel_cfg, self.channel_cache_dir
            )
        return _safe_decode_jpeg(path)

    def __len__(self) -> int:
        return len(self.df)

    # ---- image loading ----------------------------------------------------
    def _load_image_block(self, paths: list[str], views: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Load up to MAX_VIEWS images, pad to fixed shape. Returns (img, view_codes)."""
        imgs = []
        view_codes = []
        for path, vp in zip(paths, views):
            img = self._decode(path)
            if img is None:
                warnings.warn(f"Skipping unreadable image {path}")
                continue
            if self.transform is not None:
                img = self.transform(image=img)["image"]
                img = np.moveaxis(img, -1, 0)
            imgs.append(img)
            view_codes.append(int(vp))

        if len(imgs) == 0:
            return _zero_image_block(self.size), np.zeros(MAX_VIEWS, dtype=np.int64)

        n = len(imgs)
        stacked = np.stack(imgs, axis=0).astype(np.float32)
        if n < MAX_VIEWS:
            pad = np.zeros((MAX_VIEWS - n, 3, self.size, self.size), dtype=np.float32)
            stacked = np.concatenate([stacked, pad], axis=0)
        vc = np.array(view_codes + [0] * (MAX_VIEWS - n), dtype=np.int64)
        return stacked, vc

    def _text_input(
        self,
        row,
        embedding_key: str,
        token_key: str,
        text_key: str,
        max_length: int,
        fallback: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.text_embedding_cache is not None and text_key in self.text_embedding_streams:
            text = _text_value(row, text_key, fallback)
            embedding = self.text_embedding_cache.get_embedding(text, max_length=max_length)
            return embedding.astype(np.float32, copy=False), np.zeros(1, dtype=np.int64)
        return _text_array(row, embedding_key, token_key, np.int64), (
            _to_array(row[token_key.replace("input_ids", "attn_mask")], np.int64)
            if token_key.replace("input_ids", "attn_mask") in row.index
            else np.zeros(1, dtype=np.int64)
        )

    def _normalize_vitals(self, raw_values) -> tuple[np.ndarray, np.ndarray]:
        raw = _to_array(raw_values, np.float32) if raw_values is not None else _nan_vitals()
        if raw.shape[0] != len(VITAL_FIELDS):
            padded = _nan_vitals()
            padded[: min(len(padded), raw.shape[0])] = raw[: min(len(padded), raw.shape[0])]
            raw = padded
        values = []
        missing = []
        for field, value in zip(VITAL_FIELDS, raw):
            if pd.isna(value):
                values.append(0.0)
                missing.append(True)
                continue
            stats = self.vital_stats[field]
            std = float(stats.get("std", 1.0)) or 1.0
            values.append((float(value) - float(stats.get("mean", 0.0))) / std)
            missing.append(False)
        return np.array(values, dtype=np.float32), np.array(missing, dtype=np.bool_)

    # ---- main entrypoint --------------------------------------------------
    def __getitem__(self, index: int) -> tuple[dict[str, Any], np.ndarray]:
        row = self.df.iloc[index]

        cur_img, cur_views = self._load_image_block(list(row["img_paths"]), list(row["view_positions"]))
        if cur_img.sum() == 0:
            # Defensive: every current image unreadable. Fall through to neighbor like the legacy
            # dataset did, so we never return an all-zero current block to the model.
            warnings.warn(f"All current images unreadable for study {int(row['study_id'])}")
            return self.__getitem__((index + 1) % len(self))

        has_prior = bool(row["has_prior"])
        # Label dropout: zero the prior block stochastically during training.
        if has_prior and self.label_dropout_p > 0.0 and np.random.rand() < self.label_dropout_p:
            has_prior = False

        if has_prior and bool(row.get("prior_has_image", True)):
            prv_img, prv_views = self._load_image_block(
                list(row["prior_img_paths"]), list(row["prior_view_positions"])
            )
        else:
            prv_img = _zero_image_block(self.size)
            prv_views = np.zeros(MAX_VIEWS, dtype=np.int64)

        clin_input, clin_mask = self._text_input(
            row, "clin_embedding", "clin_input_ids", "clin_text", CLIN_MAX_LEN, "No clinical history available."
        )
        obs_input, obs_mask = self._text_input(
            row, "obs_embedding", "obs_input_ids", "obs_text", OBS_MAX_LEN,
            "Temperature: NA | Heart rate: NA | Respiratory rate: NA | O2 Saturation: NA | Systolic BP: NA | Diastolic BP: NA | Gender: NA",
        )
        prior_clin_input, prior_clin_mask = self._text_input(
            row, "prior_clin_embedding", "prior_clin_input_ids", "prior_clin_text", CLIN_MAX_LEN, "No clinical history available."
        )
        prior_obs_input, prior_obs_mask = self._text_input(
            row, "prior_obs_embedding", "prior_obs_input_ids", "prior_obs_text", OBS_MAX_LEN,
            "Temperature: NA | Heart rate: NA | Respiratory rate: NA | O2 Saturation: NA | Systolic BP: NA | Diastolic BP: NA | Gender: NA",
        )

        data = {
            "study_id": int(row["study_id"]),
            "img": cur_img,
            "view_positions": cur_views,
            "clin_input_ids": clin_input,
            "clin_attn_mask": clin_mask,
            "obs_input_ids": obs_input,
            "obs_attn_mask": obs_mask,

            "has_prior": bool(has_prior),
            "prior_img": prv_img,
            "prior_view_positions": prv_views,
            "prior_clin_input_ids": prior_clin_input,
            "prior_clin_attn_mask": prior_clin_mask,
            "prior_obs_input_ids": prior_obs_input,
            "prior_obs_attn_mask": prior_obs_mask,
            "prior_label": _to_array(row["prior_label"], np.float32) if has_prior else np.zeros(N_CLASSES, dtype=np.float32),
            "days_since_prior": float(row["days_since_prior"]) if has_prior and not pd.isna(row["days_since_prior"]) else 0.0,
        }
        vital_values, vital_missing = self._normalize_vitals(row["vital_values_raw"] if _has_value(row, "vital_values_raw") else None)
        prior_vital_values, prior_vital_missing = self._normalize_vitals(
            row["prior_vital_values_raw"] if has_prior and _has_value(row, "prior_vital_values_raw") else None
        )
        data["vital_values"] = vital_values
        data["vital_missing_mask"] = vital_missing
        data["prior_vital_values"] = prior_vital_values if has_prior else np.zeros(len(VITAL_FIELDS), dtype=np.float32)
        data["prior_vital_missing_mask"] = prior_vital_missing if has_prior else np.ones(len(VITAL_FIELDS), dtype=np.bool_)
        label = _to_array(row["label"], np.float32)
        return data, label


# --- time-delta bucketing (used by the model side) -------------------------
# Bucket index 0 is reserved for "no prior / unknown".
# 1: <=1d, 2: 2-7d, 3: 8-30d, 4: 1-6mo, 5: 6-12mo, 6: 1-3y, 7: >3y
N_DELTA_BUCKETS = 8


def bucket_days(days: torch.Tensor, has_prior: torch.Tensor) -> torch.Tensor:
    """Return an int64 bucket index in [0, N_DELTA_BUCKETS) per sample."""
    out = torch.zeros_like(days, dtype=torch.long)
    out = torch.where(days <= 1, torch.full_like(out, 1), out)
    out = torch.where((days > 1) & (days <= 7), torch.full_like(out, 2), out)
    out = torch.where((days > 7) & (days <= 30), torch.full_like(out, 3), out)
    out = torch.where((days > 30) & (days <= 180), torch.full_like(out, 4), out)
    out = torch.where((days > 180) & (days <= 365), torch.full_like(out, 5), out)
    out = torch.where((days > 365) & (days <= 365 * 3), torch.full_like(out, 6), out)
    out = torch.where(days > 365 * 3, torch.full_like(out, 7), out)
    # Force bucket 0 (unknown) wherever has_prior is False.
    out = torch.where(has_prior, out, torch.zeros_like(out))
    return out
