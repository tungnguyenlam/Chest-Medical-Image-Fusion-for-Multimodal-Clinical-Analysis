from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.CaMCheXV2NanoVitalsModel import CaMCheXV2NanoVitalsModel
from training.common import (
    add_common_args,
    classes_from_config,
    data_cfg_from_config,
    image_norm_stats,
    load_config,
    loss_args_from_config,
    lr_from_config,
    make_camchex_vitals_loaders,
    model_init_args_from_config,
    prepare_run_dir,
    resolve_path,
    timm_args_from_config,
    train_model,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable-tail variant of CaMCheX V2Nano vitals (single-cosine LR + EMA). "
        "Same model as camchex_v2nano_vitals; differs only in LR schedule and weight averaging. "
        "Typically warm-started from the baseline's best checkpoint via --checkpoint-path."
    )
    add_common_args(parser, model_name="camchex_v2nano_vitals_stable")
    parser.add_argument("--frontal-pretrained-path", help="Optional frontal timm backbone state_dict.")
    parser.add_argument("--lateral-pretrained-path", help="Optional lateral timm backbone state_dict.")
    parser.add_argument("--text-model", help="Override model.text_model from config.")
    parser.add_argument("--freeze-text-encoder", action="store_true", help="Freeze the CXR-BERT text encoder.")
    parser.add_argument(
        "--use-precomputed-text-embeddings",
        action="store_true",
        help="Use the shared frozen text embedding cache and skip loading CXR-BERT in the model.",
    )
    parser.add_argument("--text-embedding-cache-dir", help="Override the shared text embedding cache root.")
    parser.add_argument(
        "--uint8-image-pipeline",
        action="store_true",
        help="(opt-in) Ship images through the DataLoader as uint8 [0,255] and dequantize + "
        "normalize on-device in the model, instead of CPU float32 normalization. ~4x smaller "
        "per-batch host buffer, pinned-memory staging, and H2D copy. Requires a channel mode. "
        "Note: train-time augmentations then run on uint8, which shifts value-scale aug numerics "
        "(noise/brightness) -- validate with a short ablation before adopting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = prepare_run_dir(args)
    cfg = load_config(args.config)
    write_resolved_config(run_dir, args, cfg)

    train_loader, val_loader = make_camchex_vitals_loaders(cfg, args)
    frontal_pretrained_path = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral_pretrained_path = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None
    text_model = args.text_model or cfg.get("model", {}).get("text_model") or "microsoft/BiomedVLP-CXR-BERT-specialized"
    model_init_args = model_init_args_from_config(cfg)
    data_cfg = cfg.get("data", {}).get("datamodule_cfg", {}) or {}
    if args.freeze_text_encoder:
        model_init_args["freeze_text_encoder"] = True
    if args.use_precomputed_text_embeddings or data_cfg.get("use_text_embedding_cache", False):
        model_init_args["use_precomputed_text_embeddings"] = True
        model_init_args["freeze_text_encoder"] = True
    model = CaMCheXV2NanoVitalsModel(
        timm_init_args=timm_args_from_config(cfg, args),
        frontal_pretrained_path=frontal_pretrained_path,
        lateral_pretrained_path=lateral_pretrained_path,
        text_model=text_model,
        **model_init_args,
    )
    if args.uint8_image_pipeline:
        mean, std = image_norm_stats(data_cfg_from_config(cfg, args))
        model.enable_input_normalization(mean, std)
        print(f"[train] uint8 image pipeline: model normalizes on-device (mean={mean}, std={std})")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        run_dir=run_dir,
        lr=lr_from_config(cfg, args),
        classes=classes_from_config(cfg),
        loss_init_args=loss_args_from_config(cfg),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
