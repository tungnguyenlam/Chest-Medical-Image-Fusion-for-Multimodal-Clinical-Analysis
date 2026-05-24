from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.CaMCheXModel import CaMCheXModel
from training.common import (
    MultiLabelModule,
    add_common_args,
    classes_from_config,
    load_config,
    loss_args_from_config,
    lr_from_config,
    make_camchex_loaders,
    make_run_dir,
    resolve_path,
    timm_args_from_config,
    trainer_from_args,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full multimodal CaMCheX model.")
    add_common_args(parser, model_name="camchex")
    parser.add_argument("--frontal-pretrained-path", help="Stage-1 frontal timm backbone state_dict.")
    parser.add_argument("--lateral-pretrained-path", help="Stage-1 lateral timm backbone state_dict.")
    parser.add_argument("--text-model", default="dmis-lab/biobert-v1.1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir(args.output_dir, args.run_name, args.run_id)
    write_resolved_config(run_dir, args, cfg)

    train_loader, val_loader = make_camchex_loaders(cfg, args)
    frontal_pretrained_path = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral_pretrained_path = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None
    model = CaMCheXModel(
        timm_init_args=timm_args_from_config(cfg, args),
        frontal_pretrained_path=frontal_pretrained_path,
        lateral_pretrained_path=lateral_pretrained_path,
        text_model=args.text_model,
    )
    module = MultiLabelModule(
        model=model,
        lr=lr_from_config(cfg, args),
        classes=classes_from_config(cfg),
        loss_init_args=loss_args_from_config(cfg),
    )
    trainer = trainer_from_args(args, run_dir)
    ckpt_path = str(resolve_path(args.checkpoint_path)) if args.checkpoint_path else None
    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
