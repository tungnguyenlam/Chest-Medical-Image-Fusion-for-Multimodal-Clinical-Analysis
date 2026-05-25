from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.PriorAwareCaMCheXModel import PriorAwareCaMCheXModel
from training.common import (
    add_common_args,
    classes_from_config,
    load_config,
    loss_args_from_config,
    lr_from_config,
    make_prior_aware_loaders,
    prepare_run_dir,
    resolve_path,
    timm_args_from_config,
    train_model,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the prior-aware CaMCheX model.")
    add_common_args(parser, model_name="prior_aware")
    parser.add_argument("--frontal-pretrained-path", help="Stage-1 frontal timm backbone state_dict.")
    parser.add_argument("--lateral-pretrained-path", help="Stage-1 lateral timm backbone state_dict.")
    parser.add_argument("--text-model", default="dmis-lab/biobert-v1.1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = prepare_run_dir(args)
    write_resolved_config(run_dir, args, cfg)

    train_loader, val_loader = make_prior_aware_loaders(cfg, args)
    frontal_pretrained_path = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral_pretrained_path = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None
    model = PriorAwareCaMCheXModel(
        timm_init_args=timm_args_from_config(cfg, args),
        frontal_pretrained_path=frontal_pretrained_path,
        lateral_pretrained_path=lateral_pretrained_path,
        text_model=args.text_model,
    )
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        run_dir=run_dir,
        lr=lr_from_config(cfg, args),
        classes=classes_from_config(cfg),
        loss_init_args=loss_args_from_config(cfg),
    )


if __name__ == "__main__":
    main()
