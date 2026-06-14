#!/usr/bin/env python3
"""Print model structure and parameter counts for the active src/ models.

Examples:
    python scripts/model_summary.py --model camchex
    python scripts/model_summary.py --model camchex_v2nano_vitals
    python scripts/model_summary.py --model singleview
    python scripts/model_summary.py --model prior_aware --format markdown
    python scripts/model_summary.py --model prior_aware_v2nano --use-precomputed-text-embeddings
    python scripts/model_summary.py --config training/camchex_cxrbert/config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.*")
warnings.filterwarnings("ignore", message="Error fetching version info.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.utils.summary import print_model_summary


DEFAULT_CONFIGS = {
    "camchex": "training/camchex/config.yaml",
    "camchex_cxrbert": "training/camchex_cxrbert/config.yaml",
    "camchex_v2nano_vitals": "training/camchex_v2nano_vitals/config.yaml",
    "singleview": "training/singleview/config.yaml",
    "prior_aware": "training/prior_aware/config.yaml",
    "prior_aware_cxrbert": "training/prior_aware_cxrbert/config.yaml",
    "prior_aware_v2nano": "training/prior_aware_v2nano/config.yaml",
    "prior_aware_v3nano": "training/prior_aware_v3nano/config.yaml",
    "prior_aware_v4nano": "training/prior_aware_v4nano/config.yaml",
    "prior_aware_v5nano": "training/prior_aware_v5nano/config.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        choices=sorted(DEFAULT_CONFIGS),
        help="Model family to build. Defaults to camchex unless --config has a known training/<model>/ parent.",
    )
    p.add_argument("--config", help="Training config to read. Defaults from --model.")
    p.add_argument("--backbone-name", help="Override cfg.model.timm_init_args.model_name.")
    p.add_argument("--text-model", help="Override cfg.model.text_model.")
    p.add_argument("--format", choices=["plain", "markdown"], default="plain")
    p.add_argument("--depth", type=int, default=2, help="Recursive module table depth. Default: 2.")
    p.add_argument(
        "--freeze-text-encoder",
        action="store_true",
        help="Mark the text encoder frozen for models that support it.",
    )
    p.add_argument(
        "--use-precomputed-text-embeddings",
        action="store_true",
        help="Build models in cached-text mode when supported, so no text encoder is instantiated.",
    )
    p.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Keep timm pretrained=True from config and load text weights. Parameter counts do not require this.",
    )
    p.add_argument(
        "--print-repr",
        action="store_true",
        help="Also print the full torch module repr. This can be long.",
    )
    return p.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def install_text_config_loader() -> None:
    """Avoid loading BioBERT weights just to count architecture parameters."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoConfig, AutoModel, BertConfig

    original_from_pretrained = AutoModel.from_pretrained

    def from_config_only(model_name_or_path, *args, **kwargs):
        trust_remote_code = kwargs.get("trust_remote_code", False)
        config = kwargs.get("config")
        if config is None:
            try:
                config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
            except Exception:
                if str(model_name_or_path) != "dmis-lab/biobert-v1.1":
                    raise
                # BioBERT v1.1 uses the BERT-base shape. This fallback keeps
                # offline parameter counting usable when the HF config is not cached.
                config = BertConfig(vocab_size=28996)
        return AutoModel.from_config(config, trust_remote_code=trust_remote_code)

    AutoModel.from_pretrained = from_config_only
    AutoModel._summary_original_from_pretrained = original_from_pretrained


def build_model(args: argparse.Namespace, cfg: dict[str, Any]) -> torch.nn.Module:
    timm_args = dict(cfg["model"]["timm_init_args"])
    if args.backbone_name:
        timm_args["model_name"] = args.backbone_name
    if not args.use_pretrained:
        timm_args["pretrained"] = False

    model_key = args.model or "camchex"
    if args.config:
        config_name = Path(args.config).parent.name
        if args.model is None and config_name in DEFAULT_CONFIGS:
            model_key = config_name
        elif args.model is None and config_name != ".":
            known = ", ".join(sorted(DEFAULT_CONFIGS))
            raise ValueError(
                f"Cannot infer model type from config directory {config_name!r}. "
                f"Use --model with one of: {known}"
            )

    if model_key == "singleview":
        from src.model.SingleViewModel import SingleViewModel

        return SingleViewModel(timm_init_args=timm_args)

    text_model = args.text_model or cfg["model"].get("text_model", "dmis-lab/biobert-v1.1")
    if not args.use_pretrained:
        install_text_config_loader()

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {}).get("datamodule_cfg", {}) or {}
    init_args = dict(model_cfg.get("model_init_args", {}) or {})
    if args.freeze_text_encoder:
        init_args["freeze_text_encoder"] = True
    if args.use_precomputed_text_embeddings or data_cfg.get("use_text_embedding_cache", False):
        init_args["use_precomputed_text_embeddings"] = True
        init_args["freeze_text_encoder"] = True

    if model_key == "camchex_v2nano_vitals":
        from src.model.CaMCheXV2NanoVitalsModel import CaMCheXV2NanoVitalsModel

        return CaMCheXV2NanoVitalsModel(
            timm_init_args=timm_args,
            text_model=text_model,
            **init_args,
        )

    if model_key == "prior_aware_v2nano":
        from src.model.PriorAwareV2NanoModel import PriorAwareV2NanoModel

        return PriorAwareV2NanoModel(
            timm_init_args=timm_args,
            text_model=text_model,
            **init_args,
        )

    if model_key == "prior_aware_v3nano":
        from src.model.PriorAwareV3NanoModel import PriorAwareV3NanoModel

        return PriorAwareV3NanoModel(
            timm_init_args=timm_args,
            text_model=text_model,
            **init_args,
        )

    if model_key == "prior_aware_v4nano":
        from src.model.PriorAwareV4NanoModel import PriorAwareV4NanoModel

        return PriorAwareV4NanoModel(
            timm_init_args=timm_args,
            text_model=text_model,
            **init_args,
        )

    if model_key == "prior_aware_v5nano":
        from src.model.PriorAwareV5NanoModel import PriorAwareV5NanoModel

        return PriorAwareV5NanoModel(
            timm_init_args=timm_args,
            text_model=text_model,
            **init_args,
        )

    if model_key.startswith("prior_aware"):
        from src.model.PriorAwareCaMCheXModel import PriorAwareCaMCheXModel

        return PriorAwareCaMCheXModel(
            timm_init_args=timm_args,
            text_model=text_model,
            **init_args,
        )

    from src.model.CaMCheXModel import CaMCheXModel

    return CaMCheXModel(timm_init_args=timm_args, text_model=text_model)


def main() -> int:
    args = parse_args()
    default_model = args.model or "camchex"
    cfg_path = Path(args.config or DEFAULT_CONFIGS[default_model])
    cfg = load_config(cfg_path)
    model = build_model(args, cfg)
    print_model_summary(
        model,
        cfg_path=cfg_path,
        fmt=args.format,
        depth=args.depth,
        print_repr=args.print_repr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
