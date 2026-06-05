"""Dump per-class Grad-CAM / attribution panels from a checkpoint.

For each class it runs gradient attribution (image Grad-CAM + text-token grad x embedding
+ per-vital grad x value) on one study and writes, by default, three inspect-by-hand PNGs —
``<Class>/{image,text,vitals}.png`` (``--layout combined`` gives the old single panel).

Two ways to pick the study per class:

* default (standalone): scan the split and take the *highest-confidence true positive*
  for each class. ``--scan-limit N`` caps how many studies are scanned.
* ``--studies-json``: skip the scan and attribute preselected studies. Accepts either a
  flat ``{class: study_id}`` map (-> panels in the output dir) or a nested
  ``{set_name: {class: study_id}}`` map (-> one subfolder per set). The trainer uses the
  nested form to emit two sets reusing its validation logits (no scan):
  ``best`` (highest-confidence TP, varies per epoch) and ``first`` (first TP in val order,
  fixed across epochs so you can watch a single study's heatmap evolve).

Multilabel: attribution is class-conditional (one logit backprops per panel), so a study
with several findings gets a separate panel under each class; the header lists co-positives.

Standalone example:
    python -m src.interpret.run_gradcam \
        --config training/camchex_v2nano_vitals/config.yaml \
        --checkpoint-path output/camchex_v2nano_vitals/runs/<run>/checkpoints/best.pt \
        --split val --gradcam-epoch 1 --scan-limit 800

During training this runs automatically every epoch (see ``--gradcam-epochs`` in the
trainer). It forces the *live* CXR-BERT text path (cache off, grads on) so per-word
attribution is possible even if the checkpoint was trained with cached embeddings — the
CLS vector, and therefore the predictions, are identical either way.
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

from src.dataloader.CaMCheXVitalsDataset import DEFAULT_VITAL_STATS, VITAL_FIELDS, CaMCheXVitalsDataset
from src.dataloader.utils import get_transforms
from src.interpret.attribution import CaMCheXAttributor
from src.interpret.visualize import render_attribution, render_attribution_split
from src.model.CaMCheXV2NanoVitalsModel import CaMCheXV2NanoVitalsModel
from training.common import (
    add_common_args,
    classes_from_config,
    data_cfg_from_config,
    load_config,
    load_weights,
    read_dataframe,
    resolve_path,
    select_device,
    set_seed,
    timm_args_from_config,
)

SPLIT_KEY = {"val": "devel_df_path", "test": "pred_df_path", "train": "train_df_path"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class Grad-CAM / attribution panels for CaMCheX vitals model.")
    add_common_args(parser, model_name="camchex_v2nano_vitals")
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--gradcam-out", help="Output dir. Default: <run_dir>/gradcam/epoch_<n> next to the checkpoint.")
    parser.add_argument("--gradcam-epoch", type=int, default=1, help="Label used in the default output folder name.")
    parser.add_argument("--scan-limit", type=int, default=0, help="Cap studies scanned during selection (0 = all). Ignored if --studies-json is given.")
    parser.add_argument("--studies-json", help="JSON map {class_name: study_id} of preselected studies (e.g. from validation). Skips the selection scan.")
    parser.add_argument("--tokens-per-row", type=int, default=10)
    parser.add_argument("--layout", choices=["split", "combined"], default="split",
                        help="split = one image/text/vitals PNG per class (default); combined = single panel.")
    parser.add_argument("--device", default=None, help="Override device (cuda/mps/cpu). Default: auto.")
    return parser.parse_args()


def build_dataset(cfg, args) -> CaMCheXVitalsDataset:
    """Token-path dataset (text cache forced OFF) so per-word attribution works."""
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    # Strip any cache hints so CaMCheXVitalsDataset takes the tokenizer branch.
    for key in ("clinical_embeddings", "clinical_embedding_cache"):
        data_cfg.pop(key, None)
    _, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    df = read_dataframe(data_cfg[SPLIT_KEY[args.split]])
    tokenizer = AutoTokenizer.from_pretrained(
        data_cfg.get("tokenizer") or cfg["model"].get("text_model"),
        trust_remote_code=True,
    )
    ds = CaMCheXVitalsDataset(data_cfg, df, transforms_val, tokenizer)
    return ds, tokenizer


def build_model(cfg, args, device) -> CaMCheXV2NanoVitalsModel:
    timm_args = timm_args_from_config(cfg, args)
    timm_args["pretrained"] = False  # checkpoint supplies the weights
    init_args = dict(cfg.get("model", {}).get("model_init_args", {}) or {})
    init_args["use_precomputed_text_embeddings"] = False  # force live BERT
    init_args["freeze_text_encoder"] = False              # let gradients flow into embeddings
    model = CaMCheXV2NanoVitalsModel(
        timm_init_args=timm_args,
        text_model=cfg["model"].get("text_model", "microsoft/BiomedVLP-CXR-BERT-specialized"),
        **init_args,
    )
    load_weights(model, args.checkpoint_path)
    return model.to(device).eval()


def select_studies(attributor: CaMCheXAttributor, dataset, classes, scan_limit: int):
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
            if label[c] == 1 and probs[c] > best_prob[c]:
                best_prob[c] = float(probs[c])
                best_idx[c] = i
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{n} studies", flush=True)
    return best_idx, best_prob


def indices_from_mapping(mapping, dataset, classes):
    """Map a {class_name: study_id} dict to per-class dataset indices (no scan)."""
    sid_to_idx = {str(sid): i for i, sid in enumerate(dataset.study_ids)}
    idx = [-1] * len(classes)
    for c, name in enumerate(classes):
        sid = mapping.get(name)
        if sid is not None and str(sid) in sid_to_idx:
            idx[c] = sid_to_idx[str(sid)]
    return idx


def dump_set(attributor, dataset, classes, best_idx, out_dir, layout, tokens_per_row, set_label=""):
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
        if layout == "split":
            render_attribution_split(result, out_dir / stem, tokens_per_row=tokens_per_row)
        else:
            render_attribution(result, out_dir / f"{stem}.png", tokens_per_row=tokens_per_row)
        saved += 1
        tag = f"{set_label}/" if set_label else ""
        print(f"  [{saved:2d}] {tag}{name:28s} p={result.prob:.3f}", flush=True)
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

    print(f"[gradcam] device={device}  split={args.split}  out={out_dir}")
    dataset, tokenizer = build_dataset(cfg, args)
    model = build_model(cfg, args, device)

    vital_stats = {**DEFAULT_VITAL_STATS, **dict(cfg.get("data", {}).get("datamodule_cfg", {}).get("vital_stats", {}) or {})}
    channel_mode = cfg.get("data", {}).get("datamodule_cfg", {}).get("channel_mode")
    with CaMCheXAttributor(model, tokenizer, classes, device, VITAL_FIELDS, vital_stats, channel_mode) as attributor:
        if args.studies_json:
            raw = json.loads(Path(resolve_path(args.studies_json)).read_text())
            # nested {set_name: {class: sid}} -> subfolders; flat {class: sid} -> out_dir.
            sets = raw if raw and all(isinstance(v, dict) for v in raw.values()) else {"": raw}
            print(f"[gradcam] using preselected studies from {args.studies_json} "
                  f"(sets: {', '.join(k or 'flat' for k in sets)})")
        else:
            print(f"[gradcam] selecting 1 study/class over {len(dataset)} studies "
                  f"({'all' if args.scan_limit <= 0 else args.scan_limit})...")
            best_idx, _ = select_studies(attributor, dataset, classes, args.scan_limit)
            sets = None

        if sets is None:
            saved, missing = dump_set(attributor, dataset, classes, best_idx, out_dir, args.layout, args.tokens_per_row)
            print(f"[gradcam] saved {saved} panels to {out_dir}")
            if missing:
                print(f"[gradcam] no positive study found for: {', '.join(missing)}")
        else:
            for set_name, mapping in sets.items():
                idx = indices_from_mapping(mapping, dataset, classes)
                set_dir = out_dir / set_name if set_name else out_dir
                saved, missing = dump_set(attributor, dataset, classes, idx, set_dir, args.layout, args.tokens_per_row, set_label=set_name)
                print(f"[gradcam] saved {saved} '{set_name or 'flat'}' panels to {set_dir}")
                if missing:
                    print(f"[gradcam]   no study mapped for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
