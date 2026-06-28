"""Eval v8nano with the on-disk channel cache disabled (straight JPEG decode).

Thin wrapper around prior_aware_eval that:

1. Nulls data.datamodule_cfg.image_channel_cache_dir so load_or_build_channels()
   decodes each JPEG and rebuilds the raw_clahe_histeq channels fresh in-process
   (no .npy cache reads/writes). channel_mode is left intact, so the model still
   sees the exact 3-channel input it was trained on.

2. Forces low_cpu_mem_usage=False when loading CXR-BERT. The model's tied MLM head
   (cls.predictions.decoder) otherwise stays on the meta device, and model.to('mps')
   blows up with "Cannot copy out of meta tensor". The repo's 05_build_label_graph.py
   loads this same encoder with low_cpu_mem_usage=False for the same reason.

3. As a belt-and-suspenders safety net, materializes any parameter/buffer still left
   on meta before evaluation (the LM head is unused by the classifier, so zeros are
   fine) -- guards against meta tensors from any other tied-weight path.

All CLI flags are forwarded to prior_aware_eval (run with -h to see them).
"""
from __future__ import annotations

import importlib

import torch

import training.prior_aware_v8nano.prior_aware_eval as ev

# The src.encoder package re-exports the BioBertEncoder *class* under the same name,
# shadowing the submodule, so import the module object explicitly.
_bb = importlib.import_module("src.encoder.BioBertEncoder")

# --- 1. disable the channel cache ------------------------------------------------
_orig_resolve = ev.resolve_eval_config


def _resolve_no_cache(args):
    cfg = _orig_resolve(args)
    dm = cfg.setdefault("data", {}).setdefault("datamodule_cfg", {})
    dm["image_channel_cache_dir"] = None  # build channels fresh from JPEG, no cache
    print("[eval_no_cache] image_channel_cache_dir disabled -> straight JPEG decode", flush=True)
    return cfg


ev.resolve_eval_config = _resolve_no_cache

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
        print(f"[eval_no_cache] materialized {n} leftover meta tensor(s) (unused LM head)", flush=True)


def _eval_with_materialize(*args, **kwargs):
    _materialize_meta(kwargs["model"])
    return _orig_eval(*args, **kwargs)


ev.evaluate_report_ablation = _eval_with_materialize

if __name__ == "__main__":
    ev.main()
