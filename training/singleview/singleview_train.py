from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.SingleViewModel import SingleViewModel
from training.common import (
    MultiLabelModule,
    add_common_args,
    classes_from_config,
    load_config,
    loss_args_from_config,
    lr_from_config,
    make_run_dir,
    make_single_view_loaders,
    save_single_view_encoder,
    resolve_path,
    timm_args_from_config,
    trainer_from_args,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stage-1 single-view image model.")
    add_common_args(parser, model_name="singleview")
    parser.add_argument(
        "--view-position",
        choices=("all", "frontal", "lateral"),
        default="all",
        help="Optional row filter for training separate frontal/lateral image encoders.",
    )
    parser.add_argument(
        "--encoder-output-path",
        help="Optional path for saving the trained timm backbone state_dict for CaMCheX stage-2 loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir(args.output_dir, args.run_name, args.run_id)
    write_resolved_config(run_dir, args, cfg)

    train_loader, val_loader = make_single_view_loaders(cfg, args, args.view_position)
    model = SingleViewModel(timm_args_from_config(cfg, args))
    module = MultiLabelModule(
        model=model,
        lr=lr_from_config(cfg, args),
        classes=classes_from_config(cfg),
        loss_init_args=loss_args_from_config(cfg),
    )
    trainer = trainer_from_args(args, run_dir)
    ckpt_path = str(resolve_path(args.checkpoint_path)) if args.checkpoint_path else None
    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)

    encoder_output_path = args.encoder_output_path
    if encoder_output_path is None and args.view_position in {"frontal", "lateral"}:
        encoder_output_path = run_dir / f"{args.view_position}_encoder.pt"
    if encoder_output_path is not None:
        save_single_view_encoder(module, encoder_output_path)


if __name__ == "__main__":
    main()
