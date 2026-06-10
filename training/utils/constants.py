"""Shared constants for the training/eval helpers.

Imported first by every other ``training.utils`` module: it pins the repo ROOT
onto ``sys.path`` so the ``from src...`` imports in the sibling modules resolve
regardless of the entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

# CXR-LT long-tail split indices (head/medium/tail) used for the grouped metrics.
HEAD_IDX = [0, 2, 4, 12, 14, 16, 20, 24]
MEDIUM_IDX = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
TAIL_IDX = [7, 11, 17, 18, 19, 21, 23, 25]

# Repo root (training/utils/constants.py -> parents[2]). Inserted on sys.path so the
# `from src...` imports in this package resolve regardless of how a script is launched.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.image_channel_preprocessing import CHANNEL_MODES, THIRD_CHANNEL_TO_MODE


VIEW_ALIASES = {
    "frontal": {"AP", "PA", "FRONTAL"},
    "lateral": {"LATERAL", "LL"},
}

# Third channel exposed on the CLI as --third-channel-mode. ch0=raw and ch1=mild
# CLAHE are pinned for every mode; only the third channel is a degree of freedom,
# so the flag names just that channel (e.g. "histeq", not "raw_clahe_histeq").
# Restricted to the *dense* channels that survive normalization well: the edge
# channels (sobel/log/laplacian) collapse to std ~0.02 and are intentionally not
# offered here (they still exist in CHANNEL_MODES for the stats notebook).
#   clahe  = ch2 strong CLAHE (clip 4.0 / 16x16) -- same signal, higher contrast
#   histeq = ch2 global histogram equalization (dense, balanced std)
#   lbp    = ch2 uniform Local Binary Pattern (dense local micro-texture, std ~0.11)
# To offer another, add its short name here (must be a key of THIRD_CHANNEL_TO_MODE,
# and its full mode must have CHANNEL_STATS) -- no other change needed.
ENABLED_THIRD_CHANNELS = ["clahe", "histeq", "lbp"]
assert all(t in THIRD_CHANNEL_TO_MODE for t in ENABLED_THIRD_CHANNELS), (
    "ENABLED_THIRD_CHANNELS has names not in THIRD_CHANNEL_TO_MODE: "
    f"{[t for t in ENABLED_THIRD_CHANNELS if t not in THIRD_CHANNEL_TO_MODE]}"
)
assert all(THIRD_CHANNEL_TO_MODE[t] in CHANNEL_MODES for t in ENABLED_THIRD_CHANNELS), (
    "ENABLED_THIRD_CHANNELS maps to modes missing from CHANNEL_MODES"
)

# Default fraction of CPU cores used to precompute the 3-channel image cache
# before training. Matches src/prepare/01_make_dataset.py: half the cores by
# default so the build leaves headroom and is not OOM/CPU-killed on a shared
# server. Override per-run with --cpu-fraction (see resolve_cpu_fraction).
CPU_FRACTION = 0.5

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
