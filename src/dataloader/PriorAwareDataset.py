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
REPORT_MAX_LEN = 384  # prior study's findings + impression (legitimate prior info, not leakage)

_VITALS_NA = "Temperature: NA | Heart rate: NA | Respiratory rate: NA | O2 Saturation: NA | Systolic BP: NA | Diastolic BP: NA | Gender: NA"

# (raw-text column, max token length, fallback when blank) for every text stream.
_TEXT_STREAM_SPECS = (
    ("clin_text", CLIN_MAX_LEN, "No clinical history available."),
    ("obs_text", OBS_MAX_LEN, _VITALS_NA),
    ("prior_clin_text", CLIN_MAX_LEN, "No clinical history available."),
    ("prior_obs_text", OBS_MAX_LEN, _VITALS_NA),
    ("prior_report_text", REPORT_MAX_LEN, "No prior report available."),
)


def _to_array(x, dtype):
    """parquet list/array → np.ndarray with the right dtype."""
    if x is None:
        return np.zeros(0, dtype=dtype)
    return np.asarray(list(x), dtype=dtype)


def _has_value(row, key: str) -> bool:
    return key in row.index and row[key] is not None


def _text_value(row, text_key: str, fallback: str) -> str:
    if text_key not in row.index:
        raise KeyError(
            f"Prior-aware parquet is missing {text_key!r}; rebuild it with "
            "src/prepare/04_build_prior_aware_dataset.py."
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
        tokenizer=None,
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
            tokenizer: HF tokenizer used to encode the raw text columns at load time.
                The parquet stores only raw text (no baked token ids), so one parquet
                serves any text model -- the tokenizer is chosen by the training config.
                May be None only when every text stream is served by the embedding cache.
        """
        super().__init__()
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.size = int(image_size)
        self.transform = transform
        self.label_dropout_p = float(label_dropout_p)
        self.tokenizer = tokenizer
        cfg = cfg or {}
        self.channel_mode = cfg.get("channel_mode")
        self.channel_cache_dir = cfg.get("image_channel_cache_dir")
        self.channel_cfg = make_preprocess_config(cfg) if self.channel_mode else None
        self.text_embedding_cache = cfg.get("text_embedding_cache")
        self.text_embedding_streams = set(
            cfg.get("text_embedding_streams")
            or ["clin_text", "obs_text", "prior_clin_text", "prior_obs_text", "prior_report_text"]
        )
        self.vital_stats = {**DEFAULT_VITAL_STATS, **dict(cfg.get("vital_stats", {}) or {})}

    def precompute_text_indices(self) -> None:
        """Index-mode only: resolve every (row, text stream) to its embedding-table
        row once, in the parent process, then DROP the raw-text columns.

        Without this, every ``__getitem__`` reads the raw clinical/report strings to
        hash them into an index. Touching those Python ``str`` objects bumps their
        refcounts, which forces the OS to copy the shared parquet pages into each
        dataloader worker -- so worker RAM climbs across an epoch under copy-on-write
        ``fork`` until all text has been touched. Precomputing into a contiguous
        int64 array (and dropping the now-unused text columns) means workers touch
        only fork-stable numeric data, so per-worker RAM stays flat. No-op unless the
        cache is in index mode (see TextEmbeddingCache.build_index_table)."""
        cache = self.text_embedding_cache
        if cache is None or not getattr(cache, "index_mode", False):
            return
        self._text_idx: dict[str, np.ndarray] = {}
        drop: list[str] = []
        for key, max_len, fallback in _TEXT_STREAM_SPECS:
            if key not in self.text_embedding_streams:
                continue
            if key not in self.df.columns:
                raise KeyError(
                    f"Prior-aware parquet is missing {key!r}; rebuild it with "
                    "src/prepare/04_build_prior_aware_dataset.py."
                )
            col = self.df[key].tolist()
            idxs = np.empty(len(col), dtype=np.int64)
            for i, value in enumerate(col):
                text = fallback if (pd.isna(value) or str(value).strip() == "") else str(value)
                idxs[i] = cache.get_index(text, max_length=max_len)
            self._text_idx[key] = idxs
            drop.append(key)
        if drop:
            self.df = self.df.drop(columns=drop)

    def drop_unused_text_columns(self) -> int:
        """Free raw-text columns the model never consumes, shrinking steady-state
        host RAM (and the copy-on-write duplication into each fork worker the moment
        a row is touched). ``_emit_streams`` is the set of streams ``__getitem__``
        actually produces; any other text column is dead weight. For the Nano
        variants that feed vitals numerically, the ``obs`` streams are never emitted,
        so their (often long) strings sit unused in the parquet otherwise.

        Safe in every mode: in tokenizer mode every stream is emitted so this is a
        no-op; in cache mode the emitted streams' text is still needed for the lookup
        and is kept. Call AFTER the text-embedding cache is attached (so _emit_streams
        is final) and after the cache is built (the build still reads emitted streams).
        Disjoint from precompute_text_indices, which drops the *emitted* columns once
        they are resolved to indices. Returns the number of columns dropped."""
        emit = self._emit_streams
        drop = [
            key for key, _, _ in _TEXT_STREAM_SPECS
            if key not in emit and key in self.df.columns
        ]
        if drop:
            self.df = self.df.drop(columns=drop)
        return len(drop)

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

    @property
    def _emit_streams(self) -> set[str]:
        """Text streams ``__getitem__`` actually produces. With a precomputed-embedding
        cache the model consumes exactly ``text_embedding_streams`` -- the rest (the
        vitals-as-text ``obs`` streams, dead weight for the Nano variants that feed
        vitals numerically) are skipped, so their raw-text columns are never touched
        and never copy-on-write duplicated into workers. In tokenizer mode (cache off,
        e.g. attribution or the base model's token runs) every stream is emitted."""
        if self.text_embedding_cache is not None:
            return self.text_embedding_streams
        return {key for key, _, _ in _TEXT_STREAM_SPECS}

    def _text_input(
        self,
        row,
        text_key: str,
        max_length: int,
        fallback: str,
        index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode one raw-text stream. Frozen-encoder runs read a cached CLS vector;
        otherwise the text is tokenized at load time with ``self.tokenizer`` (the
        parquet stores raw text only, so the tokenizer is the training config's)."""
        # Precomputed index mode (precompute_text_indices): the text column is gone;
        # read the row's table index straight from the contiguous int array.
        precomputed = getattr(self, "_text_idx", None)
        if precomputed is not None and index is not None and text_key in precomputed:
            return np.int64(precomputed[text_key][index]), np.zeros(1, dtype=np.int64)
        text = _text_value(row, text_key, fallback)
        if self.text_embedding_cache is not None and text_key in self.text_embedding_streams:
            if getattr(self.text_embedding_cache, "index_mode", False):
                # GPU-resident table: emit a row index; the model gathers on-device.
                idx = self.text_embedding_cache.get_index(text, max_length=max_length)
                return np.int64(idx), np.zeros(1, dtype=np.int64)
            embedding = self.text_embedding_cache.get_embedding(text, max_length=max_length)
            return embedding.astype(np.float32, copy=False), np.zeros(1, dtype=np.int64)
        if self.tokenizer is None:
            raise RuntimeError(
                f"PriorAwareDataset has no tokenizer to encode {text_key!r}; pass a tokenizer "
                "or enable the text embedding cache / --use-precomputed-text-embeddings."
            )
        enc = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=max_length, return_tensors="np"
        )
        return enc["input_ids"][0].astype(np.int64), enc["attention_mask"][0].astype(np.int64)

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

        data = {
            "study_id": int(row["study_id"]),
            "img": cur_img,
            "view_positions": cur_views,
            "has_prior": bool(has_prior),
            "prior_img": prv_img,
            "prior_view_positions": prv_views,
            "prior_label": _to_array(row["prior_label"], np.float32) if has_prior else np.zeros(N_CLASSES, dtype=np.float32),
            "days_since_prior": float(row["days_since_prior"]) if has_prior and not pd.isna(row["days_since_prior"]) else 0.0,
        }
        # Emit only the text streams the model consumes (see _emit_streams). The
        # dict key prefix is the column name without the trailing "_text".
        emit = self._emit_streams
        for text_key, max_len, fallback in _TEXT_STREAM_SPECS:
            if text_key not in emit:
                continue
            ids, mask = self._text_input(row, text_key, max_len, fallback, index)
            prefix = text_key[: -len("_text")]
            data[f"{prefix}_input_ids"] = ids
            data[f"{prefix}_attn_mask"] = mask
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
