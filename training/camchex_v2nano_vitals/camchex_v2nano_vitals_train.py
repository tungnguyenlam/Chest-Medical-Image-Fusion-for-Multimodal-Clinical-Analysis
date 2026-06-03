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
    load_config,
    loss_args_from_config,
    lr_from_config,
    make_camchex_vitals_loaders,
    prepare_run_dir,
    resolve_path,
    timm_args_from_config,
    train_model,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CaMCheX with ConvNeXtV2 Nano, frozen CXR-BERT, and numeric vitals.")
    add_common_args(parser, model_name="camchex_v2nano_vitals")
    parser.add_argument("--frontal-pretrained-path", help="Optional frontal timm backbone state_dict.")
    parser.add_argument("--lateral-pretrained-path", help="Optional lateral timm backbone state_dict.")
    parser.add_argument("--text-model", help="Override model.text_model from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = prepare_run_dir(args)
    write_resolved_config(run_dir, args, cfg)

    train_loader, val_loader = make_camchex_vitals_loaders(cfg, args)
    frontal_pretrained_path = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral_pretrained_path = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None
    text_model = args.text_model or cfg.get("model", {}).get("text_model") or "microsoft/BiomedVLP-CXR-BERT-specialized"
    model_init_args = dict(cfg.get("model", {}).get("model_init_args", {}) or {})
    model = CaMCheXV2NanoVitalsModel(
        timm_init_args=timm_args_from_config(cfg, args),
        frontal_pretrained_path=frontal_pretrained_path,
        lateral_pretrained_path=lateral_pretrained_path,
        text_model=text_model,
        **model_init_args,
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
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
