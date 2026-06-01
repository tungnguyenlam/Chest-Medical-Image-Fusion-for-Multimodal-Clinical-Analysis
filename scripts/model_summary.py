#!/usr/bin/env python3
"""Print model structure and parameter counts for the active src/ models.

Examples:
    python scripts/model_summary.py --model camchex
    python scripts/model_summary.py --model singleview
    python scripts/model_summary.py --model prior_aware --format markdown
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


DEFAULT_CONFIGS = {
    "camchex": "training/camchex/config.yaml",
    "camchex_cxrbert": "training/camchex_cxrbert/config.yaml",
    "singleview": "training/singleview/config.yaml",
    "prior_aware": "training/prior_aware/config.yaml",
    "prior_aware_cxrbert": "training/prior_aware_cxrbert/config.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(DEFAULT_CONFIGS), default="camchex")
    p.add_argument("--config", help="Training config to read. Defaults from --model.")
    p.add_argument("--backbone-name", help="Override cfg.model.timm_init_args.model_name.")
    p.add_argument("--text-model", help="Override cfg.model.text_model.")
    p.add_argument("--format", choices=["plain", "markdown"], default="plain")
    p.add_argument("--depth", type=int, default=2, help="Recursive module table depth. Default: 2.")
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

    model_key = args.model
    if args.config:
        config_name = Path(args.config).parent.name
        if config_name in DEFAULT_CONFIGS:
            model_key = config_name

    if model_key == "singleview":
        from src.model.SingleViewModel import SingleViewModel

        return SingleViewModel(timm_init_args=timm_args)

    text_model = args.text_model or cfg["model"].get("text_model", "dmis-lab/biobert-v1.1")
    if not args.use_pretrained:
        install_text_config_loader()

    if model_key.startswith("prior_aware"):
        from src.model.PriorAwareCaMCheXModel import PriorAwareCaMCheXModel

        init_args = dict(cfg["model"].get("model_init_args", {}))
        return PriorAwareCaMCheXModel(
            timm_init_args=timm_args,
            text_model=text_model,
            **init_args,
        )

    from src.model.CaMCheXModel import CaMCheXModel

    return CaMCheXModel(timm_init_args=timm_args, text_model=text_model)


def count_params(module: torch.nn.Module, recurse: bool = True) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters(recurse=recurse):
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def fmt_count(n: int) -> str:
    return f"{n:,}"


def fmt_million(n: int) -> str:
    return f"{n / 1_000_000:.2f}M"


def child_rows(model: torch.nn.Module, max_depth: int) -> list[tuple[str, str, int, int]]:
    rows: list[tuple[str, str, int, int]] = []

    def visit(prefix: str, module: torch.nn.Module, depth: int) -> None:
        if depth > max_depth:
            return
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            total, trainable = count_params(child)
            rows.append((full_name, child.__class__.__name__, total, trainable))
            visit(full_name, child, depth + 1)

    visit("", model, 1)
    return rows


def print_plain(model: torch.nn.Module, rows: list[tuple[str, str, int, int]], cfg_path: Path) -> None:
    total, trainable = count_params(model)
    frozen = total - trainable
    print(f"Config: {cfg_path}")
    print(f"Model:  {model.__class__.__name__}")
    print(f"Total parameters:     {fmt_count(total)} ({fmt_million(total)})")
    print(f"Trainable parameters: {fmt_count(trainable)} ({fmt_million(trainable)})")
    print(f"Frozen parameters:    {fmt_count(frozen)} ({fmt_million(frozen)})")
    print()

    header = ("module", "type", "params", "trainable", "share")
    widths = [48, 30, 16, 16, 10]
    print("".join(text.ljust(width) for text, width in zip(header, widths)))
    print("".join("-" * width for width in widths))
    for name, module_type, params, trainable_params in rows:
        share = f"{(params / total * 100) if total else 0:.1f}%"
        values = (name, module_type, fmt_count(params), fmt_count(trainable_params), share)
        print("".join(text.ljust(width) for text, width in zip(values, widths)))


def print_markdown(model: torch.nn.Module, rows: list[tuple[str, str, int, int]], cfg_path: Path) -> None:
    total, trainable = count_params(model)
    frozen = total - trainable
    print(f"**Config:** `{cfg_path}`")
    print(f"**Model:** `{model.__class__.__name__}`")
    print()
    print("| scope | parameters |")
    print("|---|---:|")
    print(f"| total | {fmt_count(total)} ({fmt_million(total)}) |")
    print(f"| trainable | {fmt_count(trainable)} ({fmt_million(trainable)}) |")
    print(f"| frozen | {fmt_count(frozen)} ({fmt_million(frozen)}) |")
    print()
    print("| module | type | params | trainable | share |")
    print("|---|---|---:|---:|---:|")
    for name, module_type, params, trainable_params in rows:
        share = f"{(params / total * 100) if total else 0:.1f}%"
        print(f"| `{name}` | `{module_type}` | {fmt_count(params)} | {fmt_count(trainable_params)} | {share} |")


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config or DEFAULT_CONFIGS[args.model])
    cfg = load_config(cfg_path)
    model = build_model(args, cfg)
    rows = child_rows(model, max_depth=args.depth)

    if args.format == "markdown":
        print_markdown(model, rows, cfg_path)
    else:
        print_plain(model, rows, cfg_path)

    if args.print_repr:
        print()
        print(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
