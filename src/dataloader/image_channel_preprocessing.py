"""Deterministic 3-channel CXR representations.

Given a grayscale chest X-ray (ideally the preferred 1024x1024 image), build a
3-channel float32 array in ``[0, 1]`` for a chosen ``mode``.

Pipeline (per the agreed preprocessing order)::

    grayscale uint8 (>= target size, e.g. 1024)
      -> construct each single channel at the INPUT resolution
      -> resize each processed channel separately to (out_size, out_size)
      -> stack into H x W x 3
      -> values already scaled to [0, 1]

Filters are applied at full input resolution *before* downsizing so that the
512x512 result preserves high-frequency structure. There are NO random
augmentations here -- everything is deterministic so the output can be cached
and so per-channel normalization statistics are stable.
"""

from dataclasses import dataclass
from typing import Dict, List
import hashlib
import warnings

import cv2
import numpy as np

try:  # Optional: preferred uniform-LBP implementation.
    from skimage.feature import local_binary_pattern as _sk_lbp

    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover - exercised only when skimage absent.
    _sk_lbp = None
    _HAS_SKIMAGE = False


# Single-channel transforms supported by the modes below.
SINGLE_CHANNEL_TRANSFORMS = ("raw", "clahe", "hist_eq", "laplacian", "log", "sobel", "lbp")

# Each mode is an ordered list of single-channel transform names.
CHANNEL_MODES: Dict[str, List[str]] = {
    "gray3": ["raw", "raw", "raw"],
    "raw_clahe_laplacian": ["raw", "clahe", "laplacian"],
    "raw_clahe_log": ["raw", "clahe", "log"],
    "raw_clahe_sobel": ["raw", "clahe", "sobel"],
    "raw_clahe_lbp": ["raw", "clahe", "lbp"],
    # raw + the two contrast-enhancement methods side by side: CLAHE (adaptive,
    # contrast-limited) vs global histogram equalization (flattens the whole
    # intensity histogram toward uniform).
    "raw_clahe_histeq": ["raw", "clahe", "hist_eq"],
}


# Per-channel dataset normalization statistics, precomputed on the training split
# with the default PreprocessConfig (see data/00-examine-data/
# compute_channel_statistics.ipynb). Values are in [0, 1] -- feed them to
# A.Normalize with max_pixel_value=1.0 since build_channels already emits floats.
#
# Note the edge channels (sobel/log/laplacian) have near-degenerate std (~0.02):
# normalization would amplify their noise hugely. raw_clahe_histeq keeps all three
# channels dense with balanced std (~0.30/0.28/0.30), so normalization is stable
# and photometric augmentations stay meaningful on every channel.
CHANNEL_STATS: Dict[str, Dict[str, List[float]]] = {
    "gray3": {
        "mean": [0.472141, 0.472141, 0.472141],
        "std": [0.303715, 0.303715, 0.303715],
    },
    "raw_clahe_sobel": {
        "mean": [0.472141, 0.482434, 0.018656],
        "std": [0.303715, 0.278741, 0.022390],
    },
    "raw_clahe_log": {
        "mean": [0.472141, 0.482434, 0.015399],
        "std": [0.303715, 0.278741, 0.021523],
    },
    "raw_clahe_lbp": {
        "mean": [0.472141, 0.482434, 0.622842],
        "std": [0.303715, 0.278741, 0.107859],
    },
    "raw_clahe_laplacian": {
        "mean": [0.472141, 0.482434, 0.015018],
        "std": [0.303715, 0.278741, 0.019673],
    },
    "raw_clahe_histeq": {
        "mean": [0.472141, 0.482434, 0.470616],
        "std": [0.303715, 0.278741, 0.303455],
    },
}


@dataclass
class PreprocessConfig:
    """Configurable parameters for the deterministic channel transforms."""

    out_size: int = 512
    # cv2 interpolation used when downsizing each channel to out_size.
    # INTER_AREA is the most faithful choice for large -> small.
    resize_interpolation: int = cv2.INTER_AREA

    # CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)

    # Laplacian of Gaussian (LoG): Gaussian blur first, then Laplacian.
    log_gaussian_ksize: int = 5  # odd kernel size; <=0 derives from sigma
    log_gaussian_sigma: float = 1.0
    laplacian_ksize: int = 3  # kernel size for cv2.Laplacian (raw + LoG edge)

    # Sobel
    sobel_ksize: int = 3

    # Local Binary Pattern
    lbp_radius: int = 1
    lbp_points: int = 8
    lbp_method: str = "uniform"

    def fingerprint(self) -> str:
        """Stable short hash of every parameter that affects the output array.

        Folded into the on-disk cache key so a change to out_size or any filter
        parameter transparently invalidates stale cached channels.
        """
        parts = (
            self.out_size,
            self.resize_interpolation,
            self.clahe_clip_limit,
            tuple(self.clahe_tile_grid_size),
            self.log_gaussian_ksize,
            self.log_gaussian_sigma,
            self.laplacian_ksize,
            self.sobel_ksize,
            self.lbp_radius,
            self.lbp_points,
            self.lbp_method,
        )
        return hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()[:12]


def _to_unit_float(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an arbitrary-range float map into [0, 1] safely."""
    arr = arr.astype(np.float32)
    a_min = float(arr.min())
    a_max = float(arr.max())
    span = a_max - a_min
    if span <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - a_min) / span


# --- single-channel transforms: input uint8 grayscale -> float32 [0,1] -------


def _ch_raw(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    return gray.astype(np.float32) / 255.0


def _ch_clahe(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=tuple(cfg.clahe_tile_grid_size),
    )
    return clahe.apply(gray).astype(np.float32) / 255.0


def _ch_histeq(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Global histogram equalization (cv2.equalizeHist).

    Unlike CLAHE this is neither tile-local nor contrast-limited: it remaps
    intensities so the *whole-image* histogram is (approximately) uniform. Output
    is dense in [0, 1], so unlike the edge channels its dataset std stays large.
    """
    return cv2.equalizeHist(gray).astype(np.float32) / 255.0


def _ch_laplacian(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=cfg.laplacian_ksize)
    return _to_unit_float(np.abs(lap))


def _ch_log(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        gray,
        ksize=(cfg.log_gaussian_ksize, cfg.log_gaussian_ksize),
        sigmaX=cfg.log_gaussian_sigma,
    )
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=cfg.laplacian_ksize)
    return _to_unit_float(np.abs(lap))


def _ch_sobel(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cfg.sobel_ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=cfg.sobel_ksize)
    return _to_unit_float(np.sqrt(gx * gx + gy * gy))


def _lbp_fallback(gray: np.ndarray) -> np.ndarray:
    """Plain 8-neighbour, radius-1 LBP code image (0..255), no skimage.

    Ignores radius/points; only the classic 3x3 neighbourhood is supported.
    """
    g = gray.astype(np.int16)
    padded = np.pad(g, 1, mode="edge")
    center = g
    # 8 neighbours in a fixed order; each contributes one bit.
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros_like(g, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neigh = padded[1 + dy : 1 + dy + g.shape[0], 1 + dx : 1 + dx + g.shape[1]]
        code |= ((neigh >= center).astype(np.uint8) << bit)
    return code


def _ch_lbp(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if _HAS_SKIMAGE:
        lbp = _sk_lbp(gray, P=cfg.lbp_points, R=cfg.lbp_radius, method=cfg.lbp_method)
        if cfg.lbp_method == "uniform":
            # 'uniform' labels span 0..P+1 (P+2 distinct codes).
            denom = float(cfg.lbp_points + 1)
            return np.clip(lbp.astype(np.float32) / denom, 0.0, 1.0)
        return _to_unit_float(lbp)

    warnings.warn(
        "scikit-image not available: falling back to plain 8-neighbour radius-1 "
        "LBP (lbp_radius/lbp_points/lbp_method are ignored). Install scikit-image "
        "for uniform LBP.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _lbp_fallback(gray).astype(np.float32) / 255.0


_TRANSFORM_FUNCS = {
    "raw": _ch_raw,
    "clahe": _ch_clahe,
    "hist_eq": _ch_histeq,
    "laplacian": _ch_laplacian,
    "log": _ch_log,
    "sobel": _ch_sobel,
    "lbp": _ch_lbp,
}


def _resize_channel(channel: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    resized = cv2.resize(
        channel,
        (cfg.out_size, cfg.out_size),
        interpolation=cfg.resize_interpolation,
    )
    # Interpolation can nudge values slightly outside [0, 1]; clamp back.
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def build_channels(
    gray: np.ndarray,
    mode: str,
    cfg: PreprocessConfig = None,
) -> np.ndarray:
    """Build a deterministic 3-channel representation for ``mode``.

    Parameters
    ----------
    gray:
        Grayscale ``uint8`` image (HxW), ideally 1024x1024. Filters run at this
        input resolution before downsizing.
    mode:
        One of :data:`CHANNEL_MODES`.
    cfg:
        Optional :class:`PreprocessConfig`. Defaults are used when ``None``.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(out_size, out_size, 3)`` in ``[0, 1]``.
    """
    if cfg is None:
        cfg = PreprocessConfig()
    if mode not in CHANNEL_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of {sorted(CHANNEL_MODES)}"
        )

    gray = _as_gray_uint8(gray)

    # Cache per-transform full-res results so 'raw,raw,raw' computes raw once.
    cache: Dict[str, np.ndarray] = {}
    channels = []
    for name in CHANNEL_MODES[mode]:
        if name not in cache:
            full = _TRANSFORM_FUNCS[name](gray, cfg)
            cache[name] = _resize_channel(full, cfg)
        channels.append(cache[name])

    out = np.stack(channels, axis=-1).astype(np.float32)
    _validate_output(out, cfg)
    return out


def _as_gray_uint8(gray: np.ndarray) -> np.ndarray:
    """Coerce an input image to a 2-D uint8 grayscale array."""
    if gray is None:
        raise ValueError("Input image is None")
    arr = np.asarray(gray)
    if arr.ndim == 3:
        # Accept an accidental 3-channel image by converting to gray.
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _validate_output(out: np.ndarray, cfg: PreprocessConfig) -> None:
    expected = (cfg.out_size, cfg.out_size, 3)
    if out.shape != expected:
        raise ValueError(f"Output shape {out.shape} != expected {expected}")
    if out.dtype != np.float32:
        raise ValueError(f"Output dtype {out.dtype} != float32")
    if not np.isfinite(out).all():
        raise ValueError("Output contains NaN or Inf")
    lo, hi = float(out.min()), float(out.max())
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(f"Output values out of [0, 1] range: min={lo}, max={hi}")
