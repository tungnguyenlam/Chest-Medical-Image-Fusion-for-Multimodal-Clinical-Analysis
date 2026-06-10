"""Model-side runtime helpers: precision, torch.compile, channels_last, EMA,
device selection, batch movement and checkpoint (de)serialization.

These wrap a ``torch.nn.Module`` for training/eval but are model-agnostic; the
trainer in ``training.utils.train`` orchestrates them.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from .config import resolve_path, resolve_trainer_arg
from .constants import ROOT  # noqa: F401  (kept importable from this module historically)


_PRECISION_FALLBACK_WARNED: set[tuple[str, str]] = set()


def resolve_precision(device: torch.device, precision: str | None) -> str:
    """Return the effective precision string, downgrading to '32-true' when the device can't support the request.

    bf16 needs torch.cuda.is_bf16_supported() (Ampere+). fp16 autocast only buys
    speed on Volta+ (sm_70+) Tensor Cores; older GPUs (e.g. Kepler K80, sm_37)
    silently upcast and gain nothing, so we fall back to fp32 there too.
    """
    precision = (precision or "32-true").lower()
    if precision in {"32", "32-true"} or device.type != "cuda":
        return precision
    reason: str | None = None
    if "bf16" in precision:
        if not torch.cuda.is_bf16_supported():
            reason = "bf16 not supported on this CUDA device"
    else:
        major, _ = torch.cuda.get_device_capability(device)
        if major < 7:
            reason = f"fp16 has no Tensor Core path on sm_{major}x (pre-Volta)"
    if reason is None:
        return precision
    key = (precision, reason)
    if key not in _PRECISION_FALLBACK_WARNED:
        _PRECISION_FALLBACK_WARNED.add(key)
        print(f"[precision] requested {precision!r} → falling back to '32-true' ({reason})")
    return "32-true"


def precision_context(device: torch.device, precision: str | None):
    precision = resolve_precision(device, precision)
    enabled = precision not in {"32", "32-true"} and device.type == "cuda"
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if "bf16" in precision else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


class ModelEMA:
    """Exponential moving average of the model's state (params + buffers).

    Opt-in via ``trainer.ema`` / ``--ema``. The EMA weights are used as the
    *evaluated and saved* weights: validation (hence early-stopping/best
    tracking) and the saved ``model_state_dict`` reflect the smoothed model,
    while the raw weights keep training. This is the standard "max result"
    trick and it specifically needs a *stable* (non-restarting) LR tail --
    averaging across warm-restart sawtooth snapshots would mix unrelated basins.

    Note: EMA state is not itself checkpointed, and the saved model weights are
    the EMA snapshot -- these checkpoints are eval-ready but not intended for
    ``--resume-from`` (the optimizer state belongs to the raw weights).
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        base = unwrap_compiled_model(model)
        self.shadow = {k: v.detach().clone() for k, v in base.state_dict().items()}
        self._backup: dict | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        base = unwrap_compiled_model(model)
        d = self.decay
        for k, v in base.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                # ints/bool buffers (e.g. num_batches_tracked): track the latest.
                s.copy_(v.detach())

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """Swap EMA weights into the live model, stashing the raw weights."""
        base = unwrap_compiled_model(model)
        self._backup = {k: v.detach().clone() for k, v in base.state_dict().items()}
        base.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Undo :meth:`apply_to`, putting the raw training weights back."""
        if self._backup is None:
            return
        unwrap_compiled_model(model).load_state_dict(self._backup, strict=True)
        self._backup = None


def gradcam_runner_module(model: torch.nn.Module) -> str | None:
    model = unwrap_compiled_model(model)
    runner = getattr(model, "gradcam_runner_module", None)
    if runner:
        return str(runner)
    return None


def maybe_compile_model(model: torch.nn.Module, args: argparse.Namespace, cfg: dict[str, Any] | None) -> torch.nn.Module:
    # We compile the compile-safe submodules in place rather than the whole model:
    # the fusion forward has data-dependent boolean scatter / .any() routing that
    # graph-breaks under torch.compile, while the heavy, static-shape islands
    # (image backbones, BioBERT, transformer encoder, head) compile cleanly. Using
    # the in-place nn.Module.compile() keeps state_dict keys unchanged (no _orig_mod
    # prefix), so checkpoints stay compatible with eager runs and existing weights.
    compile_model = bool(resolve_trainer_arg(args, cfg, "compile_model", False))
    if not compile_model:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("--compile-model requires torch.compile, which is unavailable in this PyTorch build")
    # Make any Inductor compile failure fall back to eager instead of crashing the
    # run. Submodule compile happens lazily on the first forward/backward, so the
    # error surfaces deep inside backward() (e.g. the CantSplit floor-div bug in
    # codegen_mix_order_reduction) where a try/except around .compile() can't reach
    # it; this dynamo flag is the only place to degrade gracefully.
    import torch._dynamo as _dynamo  # aliased: `import torch._dynamo` would bind `torch` as a function-local and break the torch.* refs above
    _dynamo.config.suppress_errors = True
    # Cap Inductor's parallel compile workers to 1 by default. Inductor otherwise forks
    # min(32, ncpu) compile-worker subprocesses; each is a fork of the already-large
    # training process, and their copy-on-write pages dirty during compilation -- a host-RAM
    # spike exactly when the model, parquet caches and (warm) dataloader are all resident,
    # which is what tips a 15.5GB box into the OOM killer. Serial compilation is slower
    # (first epoch only, while graphs are built) but flattens the spike. Set this BEFORE the
    # first forward triggers lazy compilation so the worker pool is sized to 1. Export
    # TORCHINDUCTOR_COMPILE_THREADS to override (e.g. =4 on a roomy box for faster compile).
    import torch._inductor.config as _inductor_config
    env_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if env_threads is None:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        _inductor_config.compile_threads = 1
        print("[train] capped Inductor compile workers to 1 (TORCHINDUCTOR_COMPILE_THREADS=1) "
              "to avoid the compile-time host-RAM spike; export the var to override", flush=True)
    else:
        print(f"[train] Inductor compile workers from env: TORCHINDUCTOR_COMPILE_THREADS={env_threads}", flush=True)
    # Attribute chains to the submodules whose forwards are static-shape and safe to
    # compile. Missing ones are skipped (e.g. text_encoder is None with precomputed
    # embeddings; some model variants lack frontal/lateral splits).
    candidates = [
        ("image_encoder", "frontal_encoder"),
        ("image_encoder", "lateral_encoder"),
        ("text_encoder", "biobert_encoder"),
        ("transformer_encoder",),
        ("head",),
    ]
    compiled = []
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is None:
            continue
        # dynamic=None (automatic dynamic), NOT dynamic=True. dynamic=True marks every
        # input dim symbolic, including the image H/W that never change; the symbolic
        # spatial side then feeds (s//4)**2 feature-map flattening and trips an Inductor
        # backward-codegen bug (CantSplit: ((s//4)**2)//(s//4) not provably divisible).
        # Automatic dynamic keeps the static spatial dims specialized and only promotes
        # genuinely-varying dims (batch, transformer token count) to dynamic on recompile.
        obj.compile(dynamic=None)  # in-place: patches forward, leaves state_dict keys intact
        compiled.append(".".join(path))
    if compiled:
        print(f"[train] compiled submodules with torch.compile(dynamic=None, automatic): {', '.join(compiled)}")
    else:
        print("[train] --compile-model set but found no compile-safe submodules to compile; running eager")
    return model


def maybe_channels_last(model: torch.nn.Module, args: argparse.Namespace, cfg: dict[str, Any] | None) -> torch.nn.Module:
    """Convert the model to channels_last (NHWC) memory format when opted in via
    --channels-last / trainer.channels_last. This is a layout-only change (numerics
    are identical) that lets cuDNN dispatch the native NHWC Tensor-Core conv kernels
    and skip the per-layer NCHW<->NHWC transposes it would otherwise insert around
    each conv. Only 4D tensors (conv weights) change layout; 1D/2D params (norms,
    linears, the transformer, text encoder) are untouched.

    model.to(memory_format=...) flips the conv *weights*; the matching conv *inputs*
    must also be channels_last for the NHWC path to actually fire end to end, so we
    tell the image encoder to feed its backbone inputs in channels_last too. Call
    this BEFORE the optimizer is built and before torch.compile, so the compiled
    graphs capture the NHWC layout and the optimizer tracks the same Parameters."""
    enabled = bool(resolve_trainer_arg(args, cfg, "channels_last", False))
    if not enabled:
        return model
    model.to(memory_format=torch.channels_last)
    encoder = getattr(model, "image_encoder", None)
    if encoder is not None and hasattr(encoder, "enable_channels_last"):
        encoder.enable_channels_last()
        print("[train] channels_last (NHWC) enabled: conv weights + image inputs")
    else:
        # Weights are converted but inputs aren't routed -- cuDNN would re-transpose
        # at the first conv, negating the win. Surface it rather than silently no-op.
        print("[train] channels_last requested but image_encoder has no enable_channels_last(); "
              "conv weights converted but inputs stay NCHW (no speedup expected)")
    return model


def select_device(requested: str | None = None) -> torch.device:
    if requested and requested not in {"auto", "gpu"}:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        # Async H2D copy so the transfer overlaps the current batch's compute.
        # This is only safe-and-useful when the source is pinned CPU memory and
        # the target is CUDA (our loaders set pin_memory: true); on CPU/MPS or a
        # non-pinned tensor non_blocking is a silent no-op, so gate on CUDA.
        non_blocking = device.type == "cuda"
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


def load_model_state(model: torch.nn.Module, checkpoint: Any, allow_shape_mismatch: bool = False) -> None:
    """Load weights into ``model`` (strict=False on names: missing/extra keys are tolerated).

    ``allow_shape_mismatch=True`` additionally drops any checkpoint key whose tensor shape
    does not match the model's, leaving those params at their fresh init. This is for
    *weights-only warm-start* across related-but-not-identical architectures (e.g.
    camchex_v3nano -> prior_aware_v3nano, where ``segment_embedding`` grows 6->14 and the
    prior-only heads don't exist in the source). Keep it False for eval and full resume,
    where a shape mismatch is a real error you want surfaced.
    """
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    candidates = [
        state_dict,
        {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")},
        {k.removeprefix("model.model."): v for k, v in state_dict.items() if k.startswith("model.model.")},
    ]
    errors = []
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    for candidate in candidates:
        if not model_keys.intersection(candidate.keys()):
            continue
        if allow_shape_mismatch:
            dropped = [
                k for k, v in candidate.items()
                if k in model_state and hasattr(v, "shape") and tuple(v.shape) != tuple(model_state[k].shape)
            ]
            if dropped:
                candidate = {k: v for k, v in candidate.items() if k not in dropped}
                preview = ", ".join(sorted(dropped)[:6])
                suffix = "" if len(dropped) <= 6 else f", ... ({len(dropped)} total)"
                print(f"[warm-start] shape mismatch -> kept fresh init for: {preview}{suffix}", flush=True)
        try:
            transferred = len(model_keys.intersection(candidate.keys()))
            model.load_state_dict(candidate, strict=False)
            if allow_shape_mismatch:
                print(f"[warm-start] transferred {transferred}/{len(model_keys)} model tensors from checkpoint", flush=True)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "no checkpoint keys matched the model"
    raise RuntimeError(f"Could not load checkpoint: {detail}")


def load_weights(model: torch.nn.Module, checkpoint_path: str | Path, allow_shape_mismatch: bool = False) -> None:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    try:
        load_model_state(model, checkpoint, allow_shape_mismatch=allow_shape_mismatch)
    except RuntimeError as exc:
        raise RuntimeError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc


def save_single_view_encoder(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = unwrap_compiled_model(model)
    encoder = model.model
    if hasattr(encoder, "model"):
        encoder = encoder.model
    torch.save(encoder.state_dict(), path)
