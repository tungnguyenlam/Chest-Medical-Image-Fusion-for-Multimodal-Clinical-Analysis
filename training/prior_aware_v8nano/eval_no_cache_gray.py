"""Eval v8nano as a 1-channel (grayscale) ablation, channel cache disabled.

Channel-input ablation counterpart to eval_no_cache.py. The model was trained on
the 3-channel raw_clahe_histeq build (ch0=raw, ch1=mild CLAHE, ch2=global hist-eq);
this wrapper instead feeds it ``gray3`` -- the raw grayscale replicated into all three
channels -- so the engineered ch1/ch2 are removed and only single-channel content
reaches the backbone. The delta vs eval_no_cache.py measures how much the model
actually leans on the 3-channel build at inference time.

NOTE: this is an *eval-time* diagnostic, not a justification of the 3-channel design.
A model trained on raw_clahe_histeq may degrade under gray3 partly from train/eval
input mismatch, not only from lost channel signal. The confirmatory ablation is a
train-time channel_mode sweep (retrain per mode); see
docs/prior_aware_v8nano_graph_head_architecture.md discussion.

Like eval_no_cache.py it:

1. Nulls data.datamodule_cfg.image_channel_cache_dir so channels are built fresh from
   JPEG in-process (no .npy cache reads/writes) -- and additionally overrides
   channel_mode to ``gray3`` so the cached 3-channel arrays are never touched.

2. Forces low_cpu_mem_usage=False when loading CXR-BERT (tied MLM head otherwise stays
   on the meta device and model.to(device) blows up with "Cannot copy out of meta").

3. Materializes any parameter/buffer still left on meta before evaluation (the unused
   LM head is fine as zeros).

All CLI flags are forwarded to prior_aware_eval (run with -h to see them).
"""
from __future__ import annotations

import importlib

import torch

import training.prior_aware_v8nano.prior_aware_eval as ev

# The src.encoder package re-exports the BioBertEncoder *class* under the same name,
# shadowing the submodule, so import the module object explicitly.
_bb = importlib.import_module("src.encoder.BioBertEncoder")

# --- 1. disable the channel cache + force 1-channel (gray3) build ----------------
_orig_resolve = ev.resolve_eval_config


def _resolve_gray_no_cache(args):
    cfg = _orig_resolve(args)
    dm = cfg.setdefault("data", {}).setdefault("datamodule_cfg", {})
    dm["image_channel_cache_dir"] = None  # build channels fresh from JPEG, no cache
    dm["channel_mode"] = "gray3"  # raw grayscale replicated x3 -> 1-channel ablation
    print(
        "[eval_no_cache_gray] image_channel_cache_dir disabled -> straight JPEG decode; "
        "channel_mode forced to gray3 (1-channel grayscale ablation)",
        flush=True,
    )
    return cfg


ev.resolve_eval_config = _resolve_gray_no_cache

# --- 2. avoid meta tensors when loading CXR-BERT ---------------------------------
_orig_fp = _bb.from_pretrained_best_attention


def _fp_no_meta(model_cls, model_name, **kwargs):
    kwargs.setdefault("low_cpu_mem_usage", False)
    return _orig_fp(model_cls, model_name, **kwargs)


_bb.from_pretrained_best_attention = _fp_no_meta

# --- 3. materialize any leftover meta params before .to(device) -------------------
_orig_eval = ev.evaluate_report_ablation


def _materialize_meta(model):
    n = 0
    for mod in model.modules():
        for name, p in list(mod._parameters.items()):
            if p is not None and p.is_meta:
                mod._parameters[name] = torch.nn.Parameter(
                    torch.zeros(p.shape, dtype=p.dtype), requires_grad=False
                )
                n += 1
        for name, b in list(mod._buffers.items()):
            if b is not None and b.is_meta:
                mod._buffers[name] = torch.zeros(b.shape, dtype=b.dtype)
                n += 1
    if n:
        print(f"[eval_no_cache_gray] materialized {n} leftover meta tensor(s) (unused LM head)", flush=True)


def _eval_with_materialize(*args, **kwargs):
    _materialize_meta(kwargs["model"])
    return _orig_eval(*args, **kwargs)


ev.evaluate_report_ablation = _eval_with_materialize

if __name__ == "__main__":
    ev.main()
