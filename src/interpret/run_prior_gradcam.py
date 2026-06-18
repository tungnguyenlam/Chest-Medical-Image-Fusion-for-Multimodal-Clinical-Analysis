"""Dump per-class Grad-CAM / attribution panels for the prior-aware CaMCheX models.

Prior-aware analogue of ``src.interpret.run_gradcam``. The prior-aware models are a
superset of the single-study Nano models, so this driver mirrors that one but builds a
``PriorAwareDataset`` and a :class:`PriorAwareAttributor`, and writes the extra
prior-branch panels (prior image, prior clinical, prior report, prior label, time-delta)
next to the unchanged current-branch panels.

Per class (default ``--layout split``) it writes ``<Class>/`` containing:

    image.png  prior_image.png            current / prior CXR Grad-CAM
    cur_clin.png  prv_clin.png  prv_report.png   current indication / prior indication / prior report
                                           (no current-report panel: it would leak the labels)
    vitals.png prior_vitals.png            (Nano variants; base model writes obs streams)
    prior_label.png                        per-class grad×value on the prior label vector
    time_delta.png                         the time-gap bucket token's contribution
    modality.png                           current-vs-prior contribution breakdown

Study selection is identical to the single-study runner: ``--studies-json`` reuses the
trainer's per-class true-positive picks (no scan), else it scans the split for the
highest-confidence true positive per class.

Standalone example:
    python -m src.interpret.run_prior_gradcam \
        --config training/prior_aware_v3nano/config.yaml \
        --checkpoint-path output/prior_aware_v3nano/runs/<run>/checkpoints/best.pt \
        --split val --gradcam-epoch 1 --scan-limit 800

This runs automatically every epoch during training because the prior models declare
``gradcam_runner_module = "src.interpret.run_prior_gradcam"``. It forces the live
CXR-BERT path (cache off, grads on) so per-word/-token attribution is possible even when
the checkpoint was trained with cached embeddings — the CLS vectors are identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.CaMCheXVitalsDataset import DEFAULT_VITAL_STATS, VITAL_FIELDS
from src.dataloader.PriorAwareDataset import PriorAwareDataset
from src.dataloader.utils import get_transforms
from src.interpret.prior_attribution import PriorAwareAttributor
from src.interpret.visualize import render_prior_attribution_split
from src.model.PriorAwareV4NanoModel import PriorAwareV4NanoModel
from src.model.PriorAwareV5NanoModel import PriorAwareV5NanoModel
from src.model.PriorAwareCaMCheXModel import PriorAwareCaMCheXModel
from src.model.PriorAwareV2NanoModel import PriorAwareV2NanoModel
from src.model.PriorAwareV3NanoModel import PriorAwareV3NanoModel
from training.common import (
    add_common_args,
    classes_from_config,
    data_cfg_from_config,
    load_config,
    load_weights,
    resolve_path,
    select_device,
    set_seed,
    timm_args_from_config,
)

SPLIT_KEY = {"val": "devel_df_path", "test": "pred_df_path", "train": "train_df_path"}

_MODEL_CLASSES = {
    "prior_aware": PriorAwareCaMCheXModel,
    "prior_aware_v2nano": PriorAwareV2NanoModel,
    "prior_aware_v3nano": PriorAwareV3NanoModel,
    "prior_aware_v4nano": PriorAwareV4NanoModel,
    "prior_aware_v5nano": PriorAwareV5NanoModel,
    # Same model class as v5; only the background-penalty knob differs (and that is
    # stripped for attribution in build_model, so the forward stays logits-only).
    "prior_aware_v5nano_bgpenalty": PriorAwareV5NanoModel,
}
_DEFAULT_TEXT_MODEL = {
    "prior_aware": "dmis-lab/biobert-v1.1",
    "prior_aware_v2nano": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "prior_aware_v3nano": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "prior_aware_v4nano": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "prior_aware_v5nano": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "prior_aware_v5nano_bgpenalty": "microsoft/BiomedVLP-CXR-BERT-specialized",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class Grad-CAM / attribution panels for prior-aware CaMCheX models.")
    add_common_args(parser, model_name="prior_aware")
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--gradcam-out", help="Output dir. Default: <run_dir>/gradcam/epoch_<n> next to the checkpoint.")
    parser.add_argument("--gradcam-epoch", type=int, default=1, help="Label used in the default output folder name.")
    parser.add_argument("--scan-limit", type=int, default=0, help="Cap studies scanned during selection (0 = all). Ignored if --studies-json is given.")
    parser.add_argument("--studies-json", help="JSON map {class_name: study_id} of preselected studies. Skips the selection scan.")
    parser.add_argument("--tokens-per-row", type=int, default=10)
    parser.add_argument("--device", default=None, help="Override device (cuda/mps/cpu). Default: auto.")
    return parser.parse_args()


def _arch_from_config(cfg) -> str:
    arch = str(cfg.get("model", {}).get("arch", "prior_aware")).lower()
    if arch not in _MODEL_CLASSES:
        raise SystemExit(f"unknown prior-aware arch {arch!r}; expected one of {sorted(_MODEL_CLASSES)}")
    return arch


def build_dataset(cfg, args) -> PriorAwareDataset:
    """Token-path dataset (text embedding cache forced OFF) so per-token attribution works."""
    from transformers import AutoTokenizer

    arch = _arch_from_config(cfg)
    data_cfg = data_cfg_from_config(cfg, args)
    # Strip cache hints so PriorAwareDataset takes the live tokenizer branch.
    for key in ("text_embedding_cache", "use_text_embedding_cache"):
        data_cfg.pop(key, None)
    data_cfg["text_embedding_streams"] = []
    data_cfg["compute_bg_mask"] = False  # attribution doesn't need the bg mask
    _, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    text_model = cfg["model"].get("text_model") or data_cfg.get("tokenizer") or _DEFAULT_TEXT_MODEL[arch]
    tokenizer = AutoTokenizer.from_pretrained(text_model, trust_remote_code=True)
    ds = PriorAwareDataset(
        parquet_path=str(resolve_path(data_cfg[SPLIT_KEY[args.split]])),
        image_size=data_cfg["size"],
        transform=transforms_val,
        label_dropout_p=0.0,
        cfg=data_cfg,
        tokenizer=tokenizer,
    )
    ds.text_embedding_cache = None  # belt-and-suspenders: force the tokenizer branch
    return ds, tokenizer


def build_model(cfg, args, device):
    arch = _arch_from_config(cfg)
    timm_args = timm_args_from_config(cfg, args)
    timm_args["pretrained"] = False  # checkpoint supplies the weights
    init_args = dict(cfg.get("model", {}).get("model_init_args", {}) or {})
    init_args["use_precomputed_text_embeddings"] = False  # force live BERT
    init_args["freeze_text_encoder"] = False              # let gradients flow into embeddings
    init_args.pop("background_penalty_lambda", None)       # logits-only forward for attribution
    model = _MODEL_CLASSES[arch](
        timm_init_args=timm_args,
        text_model=cfg["model"].get("text_model", _DEFAULT_TEXT_MODEL[arch]),
        **init_args,
    )
    load_weights(model, args.checkpoint_path)
    return model.to(device).eval()


def select_studies(attributor, dataset, classes, scan_limit: int):
    """Highest-confidence true-positive study index per class."""
    n = len(dataset)
    if scan_limit > 0:
        n = min(n, scan_limit)
    best_idx = [-1] * len(classes)
    best_prob = [-1.0] * len(classes)
    for i in range(n):
        sample = dataset[i]
        _, label = sample
        label = np.asarray(label)
        probs = attributor.predict_probs(sample)
        for c in range(len(classes)):
            if c < len(label) and label[c] == 1 and probs[c] > best_prob[c]:
                best_prob[c] = float(probs[c])
                best_idx[c] = i
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{n} studies", flush=True)
    return best_idx, best_prob


def indices_from_mapping(mapping, dataset, classes):
    """Map a {class_name: study_id} dict to per-class dataset indices (no scan)."""
    sid_to_idx = {str(sid): i for i, sid in enumerate(dataset.df["study_id"].tolist())}
    idx = [-1] * len(classes)
    for c, name in enumerate(classes):
        sid = mapping.get(name)
        if sid is not None and str(sid) in sid_to_idx:
            idx[c] = sid_to_idx[str(sid)]
    return idx


def dump_set(attributor, dataset, classes, best_idx, out_dir, tokens_per_row, set_label=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved, missing = 0, []
    for c, name in enumerate(classes):
        if best_idx[c] < 0:
            missing.append(name)
            continue
        set_seed(best_idx[c])  # stable >4-image sampling -> identical image across epochs
        sample = dataset[best_idx[c]]
        result = attributor.attribute(sample, c)
        stem = name.replace(" ", "_").replace("/", "-")
        render_prior_attribution_split(result, out_dir / stem, tokens_per_row=tokens_per_row)
        saved += 1
        tag = f"{set_label}/" if set_label else ""
        flag = "" if result.has_prior else "  (no prior)"
        print(f"  [{saved:2d}] {tag}{name:28s} p={result.prob:.3f}{flag}", flush=True)
    return saved, missing


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path:
        raise SystemExit("--checkpoint-path is required")
    set_seed(args.seed if args.seed is not None else 0)  # deterministic >4-image sampling
    cfg = load_config(args.config)
    classes = classes_from_config(cfg)
    device = select_device(args.device)

    out_dir = (
        Path(args.gradcam_out)
        if args.gradcam_out
        else resolve_path(args.checkpoint_path).resolve().parents[1] / "gradcam" / f"epoch_{args.gradcam_epoch}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gradcam] device={device}  split={args.split}  arch={_arch_from_config(cfg)}  out={out_dir}")
    dataset, tokenizer = build_dataset(cfg, args)
    model = build_model(cfg, args, device)

    vital_stats = {**DEFAULT_VITAL_STATS, **dict(cfg.get("data", {}).get("datamodule_cfg", {}).get("vital_stats", {}) or {})}
    channel_mode = data_cfg_from_config(cfg, args).get("channel_mode")
    with PriorAwareAttributor(model, tokenizer, classes, device, VITAL_FIELDS, vital_stats, channel_mode) as attributor:
        if args.studies_json:
            raw = json.loads(Path(resolve_path(args.studies_json)).read_text())
            sets = raw if raw and all(isinstance(v, dict) for v in raw.values()) else {"": raw}
            print(f"[gradcam] using preselected studies from {args.studies_json} "
                  f"(sets: {', '.join(k or 'flat' for k in sets)})")
        else:
            print(f"[gradcam] selecting 1 study/class over {len(dataset)} studies "
                  f"({'all' if args.scan_limit <= 0 else args.scan_limit})...")
            best_idx, _ = select_studies(attributor, dataset, classes, args.scan_limit)
            sets = None

        if sets is None:
            saved, missing = dump_set(attributor, dataset, classes, best_idx, out_dir, args.tokens_per_row)
            print(f"[gradcam] saved {saved} panels to {out_dir}")
            if missing:
                print(f"[gradcam] no positive study found for: {', '.join(missing)}")
        else:
            for set_name, mapping in sets.items():
                idx = indices_from_mapping(mapping, dataset, classes)
                set_dir = out_dir / set_name if set_name else out_dir
                saved, missing = dump_set(attributor, dataset, classes, idx, set_dir, args.tokens_per_row, set_label=set_name)
                print(f"[gradcam] saved {saved} '{set_name or 'flat'}' panels to {set_dir}")
                if missing:
                    print(f"[gradcam]   no study mapped for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
