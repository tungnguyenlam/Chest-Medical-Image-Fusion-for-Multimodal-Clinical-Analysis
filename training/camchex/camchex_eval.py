from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.CaMCheXModel import CaMCheXModel
from training.common import (
    add_common_args,
    classes_from_config,
    compute_metrics,
    load_config,
    load_weights,
    make_camchex_eval_loader,
    predict_dataframe,
    print_validation_summary,
    resolve_path,
    select_device,
    timm_args_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or predict with a multimodal CaMCheX checkpoint.")
    add_common_args(parser, model_name="camchex")
    parser.add_argument("--frontal-pretrained-path", help="Optional raw frontal timm backbone state_dict.")
    parser.add_argument("--lateral-pretrained-path", help="Optional raw lateral timm backbone state_dict.")
    parser.add_argument("--text-model", help="Override model.text_model from config.")
    parser.add_argument("--predictions-path", default="output/camchex/predictions.csv")
    parser.add_argument("--metrics-path", default="output/camchex/metrics.json")
    args = parser.parse_args()
    if not args.checkpoint_path:
        parser.error("--checkpoint-path is required for evaluation")
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    classes = classes_from_config(cfg)
    loader, labels_available = make_camchex_eval_loader(cfg, args)
    frontal_pretrained_path = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral_pretrained_path = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None

    text_model = args.text_model or cfg.get("model", {}).get("text_model") or "dmis-lab/biobert-v1.1"
    model = CaMCheXModel(
        timm_init_args=timm_args_from_config(cfg, args),
        frontal_pretrained_path=frontal_pretrained_path,
        lateral_pretrained_path=lateral_pretrained_path,
        text_model=text_model,
    )
    load_weights(model, args.checkpoint_path)
    out_df, preds, labels = predict_dataframe(model, loader, classes, select_device(args.accelerator))

    predictions_path = Path(args.predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(predictions_path, index=False)

    if labels_available:
        metrics = compute_metrics(preds, labels, classes)
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print_validation_summary(metrics, classes, header=f"eval | {args.checkpoint_path}")


if __name__ == "__main__":
    main()
