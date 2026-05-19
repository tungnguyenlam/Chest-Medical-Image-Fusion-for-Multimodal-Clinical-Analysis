"""Multi-view MIMIC dataset for image + clinical-text + ED-vitals fusion models.

Produces per-study samples in the tuple shape CaMCheXModel expects:

    (
        (study_id, image_tensor, view_positions,
         clinical_input_ids, clinical_attention_mask,
         obs_input_ids, obs_attention_mask),
        label,
    )

The CSV passed in must have one row per (study_id, image) pair, with at
minimum: study_id, path, ViewPosition, clinical_indication, the seven
ED-vitals columns, and the configured class columns.

``path`` values are resolved against ``image_root`` if provided.
"""
import os
import warnings
from pathlib import Path
from typing import Optional, Sequence

import cv2
import jpeg4py as jpeg
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


OBS_FIELDS: tuple[str, ...] = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "gender")


def _safe_decode_jpeg(path: str):
    """Decode a JPEG, falling back to cv2 if jpeg4py fails. Returns HWC uint8 RGB or None."""
    candidates = [path]
    if "_resized_1024.jpg" in path:
        candidates.append(path.replace("_resized_1024.jpg", ".jpg"))
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            return jpeg.JPEG(p).decode()
        except Exception as e:
            warnings.warn(f"jpeg4py failed on {p}: {e}; falling back to cv2")
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        warnings.warn(f"cv2 also failed to decode {p}")
    return None


def _format_obs_text(row: pd.Series) -> str:
    def v(field: str) -> str:
        val = row.get(field)
        return "NA" if pd.isna(val) else str(val)
    return " | ".join([
        f"Temperature: {v('temperature')}",
        f"Heart rate: {v('heartrate')}",
        f"Respiratory rate: {v('resprate')}",
        f"O2 Saturation: {v('o2sat')}",
        f"Systolic BP: {v('sbp')}",
        f"Diastolic BP: {v('dbp')}",
        f"Gender: {v('gender')}",
    ])


class MimicMultiViewDataset(Dataset):
    """Per-study multi-view dataset with clinical indication + ED vitals text streams.

    Args:
        df: DataFrame with required columns described in the module docstring.
        classes: list of class column names (multi-label targets).
        image_size: square side length for transforms (used for the zero-padding shape).
        transform: an albumentations-style transform with ``__call__(image=hwc_uint8)``.
        tokenizer: HF tokenizer used for both text streams.
        image_root: prepended to each row's ``path`` if given. If None, ``path`` is
            taken verbatim (already absolute or cwd-relative).
        max_views: max images per study (default 4 matches CaMCheX).
        clinical_max_length: tokenizer max_length for the clinical-indication stream.
        obs_max_length: tokenizer max_length for the vitals stream.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        classes: Sequence[str],
        image_size: int,
        transform=None,
        tokenizer=None,
        image_root: Optional[str] = None,
        max_views: int = 4,
        clinical_max_length: int = 384,
        obs_max_length: int = 128,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        self.classes = list(classes)
        self.image_size = int(image_size)
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else None
        self.max_views = int(max_views)
        self.clinical_max_length = int(clinical_max_length)
        self.obs_max_length = int(obs_max_length)

        self._groups = df.groupby("study_id")
        self.study_ids = list(self._groups.groups.keys())

    def __len__(self) -> int:
        return len(self.study_ids)

    def _resolve_path(self, path: str) -> str:
        if self.image_root is None:
            return path
        return str(self.image_root / path)

    def _load_image(self, raw_path: str) -> Optional[np.ndarray]:
        path = self._resolve_path(raw_path)
        resized = path.replace(".jpg", "_resized_1024.jpg")
        path = resized if os.path.exists(resized) else path
        return _safe_decode_jpeg(path)

    @staticmethod
    def _view_code(view_position) -> int:
        vp = "" if not isinstance(view_position, str) else view_position.upper()
        if vp in ("AP", "PA", "FRONTAL"):
            return 1
        if vp in ("LATERAL", "LL"):
            return 2
        return 0

    def __getitem__(self, index: int):
        df = self._groups.get_group(self.study_ids[index])
        study_id = self.study_ids[index]
        if len(df) > self.max_views:
            df = df.sample(self.max_views)

        if all(c in df.columns for c in self.classes):
            label = df[self.classes].iloc[0].to_numpy().astype(np.float32)
        else:
            label = np.zeros(len(self.classes), dtype=np.float32)

        imgs, view_positions = [], []
        for i in range(len(df)):
            img = self._load_image(df.iloc[i]["path"])
            if img is None:
                warnings.warn(f"Skipping unreadable image in study {study_id}")
                continue

            if self.transform is not None:
                img = self.transform(image=img)["image"]
                img = np.moveaxis(img, -1, 0)
            imgs.append(img)
            view_positions.append(self._view_code(df.iloc[i].get("ViewPosition", "")))

        if not imgs:
            warnings.warn(f"All images unreadable for study {study_id}; using neighbor")
            return self.__getitem__((index + 1) % len(self.study_ids))

        n = len(imgs)
        img_stack = np.stack(imgs, axis=0)
        pad = np.zeros((self.max_views - n, 3, self.image_size, self.image_size), dtype=np.float32)
        img_stack = np.concatenate([img_stack, pad], axis=0).astype(np.float32)
        view_positions = np.array(view_positions + [0] * (self.max_views - n), dtype=np.int64)

        clinical_text = df.iloc[0].get("clinical_indication", "")
        if pd.isna(clinical_text) or (isinstance(clinical_text, str) and clinical_text.strip() == ""):
            clinical_text = "No clinical history available."

        clinical_tokens = self.tokenizer(
            clinical_text,
            padding="max_length", truncation=True,
            max_length=self.clinical_max_length, return_tensors="pt",
        )
        obs_tokens = self.tokenizer(
            _format_obs_text(df.iloc[0]),
            padding="max_length", truncation=True,
            max_length=self.obs_max_length, return_tensors="pt",
        )

        return (
            (
                study_id,
                img_stack,
                view_positions,
                clinical_tokens["input_ids"].squeeze(0),
                clinical_tokens["attention_mask"].squeeze(0),
                obs_tokens["input_ids"].squeeze(0),
                obs_tokens["attention_mask"].squeeze(0),
            ),
            label,
        )
