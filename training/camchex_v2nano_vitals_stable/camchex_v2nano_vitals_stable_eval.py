from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.CaMCheXV2NanoVitalsModel import CaMCheXV2NanoVitalsModel
from training.common import (
    add_common_args,
    classes_from_config,
    compute_metrics,
    load_config,
    resolve_eval_config,
    load_weights,
    evaluate_report_ablation,
    make_camchex_vitals_eval_loader,
    predict_dataframe,
    print_validation_summary,
    resolve_path,
    select_device,
    timm_args_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the stable-tail CaMCheX V2Nano vitals variant.")
    add_common_args(parser, model_name="camchex_v2nano_vitals_stable")
    parser.add_argument("--frontal-pretrained-path", help="Optional raw frontal timm backbone state_dict.")
    parser.add_argument("--lateral-pretrained-path", help="Optional raw lateral timm backbone state_dict.")
    parser.add_argument("--text-model", help="Override model.text_model from config.")
    parser.add_argument("--freeze-text-encoder", action="store_true", help="Freeze the CXR-BERT text encoder.")
    parser.add_argument(
        "--use-precomputed-text-embeddings",
        action="store_true",
        help="Use the shared frozen text embedding cache and skip loading CXR-BERT in the model.",
    )
    parser.add_argument("--text-embedding-cache-dir", help="Override the shared text embedding cache root.")
    parser.add_argument("--predictions-path", default="output/camchex_v2nano_vitals_stable/predictions.csv")
    parser.add_argument("--metrics-path", default="output/camchex_v2nano_vitals_stable/metrics.json")
    args = parser.parse_args()
    if not args.checkpoint_path:
        parser.error("--checkpoint-path is required for evaluation")
    return args


def main() -> None:
    args = parse_args()
    cfg = resolve_eval_config(args)
    classes = classes_from_config(cfg)
    frontal_pretrained_path = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral_pretrained_path = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None

    text_model = args.text_model or cfg.get("model", {}).get("text_model") or "microsoft/BiomedVLP-CXR-BERT-specialized"
    model_init_args = dict(cfg.get("model", {}).get("model_init_args", {}) or {})
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
    load_weights(model, args.checkpoint_path)
    evaluate_report_ablation(
        model=model,
        classes=classes,
        device=select_device(args.accelerator),
        args=args,
        make_loader=lambda drop_report: make_camchex_vitals_eval_loader(cfg, args, drop_report=drop_report),
        predictions_path=args.predictions_path,
        metrics_path=args.metrics_path,
        header=f"eval | {args.checkpoint_path}",
    )


if __name__ == "__main__":
    main()
