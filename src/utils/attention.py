"""Pick the best available attention backend for HuggingFace text encoders.

On CUDA we want Flash Attention. Flash-Attention-2 needs the ``flash_attn``
package and an Ampere-or-newer GPU (compute capability >= 8.0); when both are
present we request it directly. Otherwise we fall back to PyTorch's SDPA, which
itself dispatches to the fused flash / memory-efficient CUDA kernels when the
inputs qualify -- so we still get flash attention on compatible hardware without
the extra dependency. On CPU/MPS we leave transformers on its default ("eager").
"""
from __future__ import annotations

import importlib.util

import torch


def _flash_attn_2_available() -> bool:
    """True only on CUDA + Ampere(or newer) + an installed ``flash_attn``."""
    if not torch.cuda.is_available():
        return False
    if importlib.util.find_spec("flash_attn") is None:
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8  # Ampere (A100 / 30xx) and newer


def attn_candidates() -> list[str | None]:
    """Ordered ``attn_implementation`` values to try, best first.

    ``None`` means "no kwarg -> let transformers pick its default (eager)" and is
    always last so loading still succeeds if the faster backends are rejected by
    this transformers/model/hardware combination.
    """
    if _flash_attn_2_available():
        return ["flash_attention_2", "sdpa", None]
    if torch.cuda.is_available():
        return ["sdpa", None]
    return [None]


def from_pretrained_best_attention(model_cls, model_name, **kwargs):
    """``model_cls.from_pretrained`` using the fastest attention backend that loads.

    Tries flash-attention-2 / SDPA / eager in order (see :func:`attn_candidates`),
    falling back when a backend is unsupported. Logs the backend actually used.
    """
    # Defensive: recent transformers can initialise a model on the meta device
    # and load the checkpoint by assignment (the accelerate "low_cpu_mem_usage"
    # path). Any param/buffer not present in the state dict then stays on meta,
    # and a later ``.to(device)`` raises "Cannot copy out of meta tensor". Pin
    # the eager path so every tensor is materialised on CPU first. Caller may
    # override via kwargs.
    kwargs.setdefault("low_cpu_mem_usage", False)
    last_err: Exception | None = None
    for impl in attn_candidates():
        extra = {} if impl is None else {"attn_implementation": impl}
        try:
            model = model_cls.from_pretrained(model_name, **extra, **kwargs)
            if impl is not None:
                print(f"[attn] {model_name}: using attn_implementation={impl!r}", flush=True)
            return model
        except (ValueError, ImportError) as err:
            # Backend not supported here (e.g. fp32 + flash-attn-2, or an older
            # transformers without sdpa for this arch) -> try the next one.
            last_err = err
            continue
    raise last_err  # only reached if even the default (eager) load raised
