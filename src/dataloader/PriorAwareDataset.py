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

from src.dataloader.utils import _safe_decode_jpeg

MAX_VIEWS = 4
N_CLASSES = 26
CLIN_MAX_LEN = 384
OBS_MAX_LEN = 128


def _to_array(x, dtype):
    """parquet list/array → np.ndarray with the right dtype."""
    if x is None:
        return np.zeros(0, dtype=dtype)
    return np.asarray(list(x), dtype=dtype)


def _zero_image_block(size: int) -> np.ndarray:
    return np.zeros((MAX_VIEWS, 3, size, size), dtype=np.float32)


class PriorAwareDataset(Dataset):
    def __init__(self, parquet_path: str, image_size: int, transform=None, label_dropout_p: float = 0.0):
        """
        Args:
            parquet_path: path to a prior_aware_*.parquet built by the script.
            image_size: HxW the transform pipeline resizes to (matches data_cfg["size"]).
            transform: Albumentations Compose for image augmentation. Applied per-image.
            label_dropout_p: training-only probability to drop the entire prior block.
        """
        super().__init__()
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.size = int(image_size)
        self.transform = transform
        self.label_dropout_p = float(label_dropout_p)

    def __len__(self) -> int:
        return len(self.df)

    # ---- image loading ----------------------------------------------------
    def _load_image_block(self, paths: list[str], views: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Load up to MAX_VIEWS images, pad to fixed shape. Returns (img, view_codes)."""
        imgs = []
        view_codes = []
        for path, vp in zip(paths, views):
            img = _safe_decode_jpeg(path)
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

        data = {
            "study_id": int(row["study_id"]),
            "img": cur_img,
            "view_positions": cur_views,
            "clin_input_ids": _to_array(row["clin_input_ids"], np.int64),
            "clin_attn_mask": _to_array(row["clin_attn_mask"], np.int64),
            "obs_input_ids": _to_array(row["obs_input_ids"], np.int64),
            "obs_attn_mask": _to_array(row["obs_attn_mask"], np.int64),

            "has_prior": bool(has_prior),
            "prior_img": prv_img,
            "prior_view_positions": prv_views,
            "prior_clin_input_ids": _to_array(row["prior_clin_input_ids"], np.int64),
            "prior_clin_attn_mask": _to_array(row["prior_clin_attn_mask"], np.int64),
            "prior_obs_input_ids": _to_array(row["prior_obs_input_ids"], np.int64),
            "prior_obs_attn_mask": _to_array(row["prior_obs_attn_mask"], np.int64),
            "prior_label": _to_array(row["prior_label"], np.float32) if has_prior else np.zeros(N_CLASSES, dtype=np.float32),
            "days_since_prior": float(row["days_since_prior"]) if has_prior and not pd.isna(row["days_since_prior"]) else 0.0,
        }
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
