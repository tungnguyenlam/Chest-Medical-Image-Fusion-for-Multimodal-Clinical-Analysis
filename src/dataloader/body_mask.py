"""Confident *outside-patient* background mask for chest X-rays (margin/letterbox).

This is the "separate the background" step behind background-attention
regularization. We learned the hard way that you cannot cleanly separate the
patient body from the outside on a CXR by image content: lungs are radiolucent, so
they sit at almost the same intensity as outside-patient air *and* connect to it
laterally and at the apices. Both a brightness threshold and a border-connected
flood fill end up flagging the lungs as background -- catastrophic, since the lungs
are the most diagnostic region.

So we deliberately give up on a full body silhouette and target only what is
*unambiguously* not the patient:

  * **collimation letterbox** -- near-black bars the scanner pads the frame with,
  * **direct-exposure margins** -- saturated near-white regions outside the body,
  * the **border-connected corners** these form.

Two guards keep this from ever touching anatomy:

  1. **Extreme-value gate.** A pixel is background only if it's near-black or
     near-white (collimation / direct exposure). Lung tissue is *moderately* dark,
     never extreme, so it fails this gate.
  2. **Outer-band restriction.** Background is only allowed within a thin band of
     the image edge. Interior lung (well inside the frame) can never be flagged,
     no matter what.

The result is a small, conservative mask -- exactly matching the observation that
the real outside-patient region in these films is only a thin frame. It catches
the "model is staring at the black bars / corners" shortcut; it does **not** try to
police body-adjacent attention (we can't separate that safely, and off-lung
findings legitimately live on the chest wall).

The downstream penalty consumes the soft ``background_weight`` (feathered for a
smooth gradient). A learned segmenter could later replace :func:`confident_background`
behind the same interface, but this needs no model, GPU, or extra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BodyMaskConfig:
    """Parameters for the conservative outside-patient background detector.

    Spatial sizes are fractions of the shorter side so the same config behaves
    identically at 512, 1024, etc. Intensity thresholds are fractions of 255.
    """

    blur_frac: float = 0.008        # Gaussian sigma (kill speckle before thresholding)
    dark_frac: float = 0.12         # <= this * 255 counts as collimation black (~31)
    bright_frac: float = 0.97       # >= this * 255 counts as direct-exposure white (~247)
    band_frac: float = 0.22         # background allowed only within this margin of the edge
    open_frac: float = 0.01         # open the background to drop isolated speckle
    feather_frac: float = 0.015     # blur the bg-weight edge (0 disables)
    min_bg_area_frac: float = 0.002  # below this, treat as "no background" (weight ~0)

    def _px(self, frac: float, short_side: int) -> int:
        return max(1, int(round(frac * short_side)))


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _edge_band(h: int, w: int, band_px: int) -> np.ndarray:
    """Boolean mask, True only within ``band_px`` of any image edge."""
    band = np.ones((h, w), dtype=bool)
    if 2 * band_px < h and 2 * band_px < w:
        band[band_px : h - band_px, band_px : w - band_px] = False
    return band


def confident_background(gray: np.ndarray, cfg: BodyMaskConfig | None = None) -> np.ndarray:
    """Soft confident-background weight in [0, 1] (1 = punish attention here).

    ``gray`` is a 2-D uint8 (or coercible) chest X-ray at any resolution. Returns a
    float32 map the same size. Only border-connected, extreme-valued pixels within
    the outer band are flagged; everything else (all anatomy) is 0.
    """
    if cfg is None:
        cfg = BodyMaskConfig()

    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)

    h, w = g.shape
    short = min(h, w)
    zeros = np.zeros((h, w), dtype=np.float32)

    sigma = cfg.blur_frac * short
    blurred = cv2.GaussianBlur(g, (0, 0), sigmaX=max(sigma, 0.1))

    # Guard 1 -- extreme value: collimation black or direct-exposure white only.
    dark_thr = cfg.dark_frac * 255.0
    bright_thr = cfg.bright_frac * 255.0
    extreme = (blurred <= dark_thr) | (blurred >= bright_thr)
    if not extreme.any():
        return zeros

    # Keep only extreme regions connected to the image border (true outside frame).
    n, labels = cv2.connectedComponents(extreme.astype(np.uint8), connectivity=8)
    if n <= 1:
        return zeros
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_labels = [int(v) for v in np.unique(border) if v != 0]
    if not border_labels:
        return zeros
    bg = np.isin(labels, border_labels)

    # Guard 2 -- outer band only: interior anatomy can never be flagged.
    bg &= _edge_band(h, w, cfg._px(cfg.band_frac, short))

    # Drop isolated speckle (e.g. bright text glyphs in a corner).
    o = cfg._px(cfg.open_frac, short)
    okern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(o), _odd(o)))
    bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_OPEN, okern) > 0

    if bg.mean() < cfg.min_bg_area_frac:
        return zeros

    weight = bg.astype(np.float32)
    if cfg.feather_frac > 0:
        fsigma = cfg.feather_frac * short
        weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=max(fsigma, 0.1))
        # Re-clamp to the band so feathering can't bleed the penalty inward.
        weight *= _edge_band(h, w, cfg._px(cfg.band_frac, short)).astype(np.float32)
        weight = np.clip(weight, 0.0, 1.0)

    return weight.astype(np.float32)


def body_and_background(gray: np.ndarray, cfg: BodyMaskConfig | None = None):
    """Convenience: ``(kept_weight, background_weight)`` for one image.

    ``kept_weight = 1 - background_weight`` -- everything not flagged as confident
    outside-patient background (i.e. all anatomy plus the ambiguous body edge).
    """
    if cfg is None:
        cfg = BodyMaskConfig()
    bg = confident_background(gray, cfg)
    return (1.0 - bg).astype(np.float32), bg
