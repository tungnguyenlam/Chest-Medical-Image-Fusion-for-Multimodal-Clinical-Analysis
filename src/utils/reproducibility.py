"""Reproducibility helpers.

Call :func:`seed_everything` at entry points (notebooks, training scripts),
never inside low-level dataset/dataloader logic.
"""

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if available) PyTorch for reproducible runs.

    Parameters
    ----------
    seed:
        The base seed applied to every RNG.
    deterministic:
        When ``True`` and CUDA is available, request deterministic cuDNN
        kernels (``cudnn.deterministic=True``, ``cudnn.benchmark=False``).
        Set to ``False`` if you want cuDNN autotuning for throughput and do
        not need bitwise reproducibility.

    Returns
    -------
    int
        The seed that was applied (convenient for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn may be absent on CPU-only builds.
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = bool(deterministic)
        cudnn.benchmark = not bool(deterministic)

    return seed
