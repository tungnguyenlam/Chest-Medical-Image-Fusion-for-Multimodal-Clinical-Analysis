from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.SingleViewModel import SingleViewModel
from training.common import (
    add_common_args,
    classes_from_config,
    compute_metrics,
    load_config,
    load_weights,
    make_single_view_eval_loader,
    predict_dataframe,
    print_validation_summary,
    select_device,
    timm_args_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or predict with a single-view model checkpoint.")
    add_common_args(parser, model_name="singleview")
    parser.add_argument("--view-position", choices=("all", "frontal", "lateral"), default="all")
    parser.add_argument("--predictions-path", default="output/singleview/predictions.csv")
    parser.add_argument("--metrics-path", default="output/singleview/metrics.json")
    args = parser.parse_args()
    if not args.checkpoint_path:
        parser.error("--checkpoint-path is required for evaluation")
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    classes = classes_from_config(cfg)
    loader, ids, labels_available = make_single_view_eval_loader(cfg, args, args.view_position)

    model = SingleViewModel(timm_args_from_config(cfg, args))
    load_weights(model, args.checkpoint_path)
    out_df, preds, labels = predict_dataframe(model, loader, classes, select_device(args.accelerator), ids=ids)

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
