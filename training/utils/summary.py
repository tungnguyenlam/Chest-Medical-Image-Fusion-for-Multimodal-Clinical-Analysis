"""Model parameter-count summary rendering, shared by the standalone
``scripts/model_summary.py`` tool and the training loop (printed once before
training starts). Keep the rendering here so both paths stay in sync."""
from __future__ import annotations

from pathlib import Path

import torch


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


def print_plain(model: torch.nn.Module, rows: list[tuple[str, str, int, int]], cfg_path: Path | None) -> None:
    total, trainable = count_params(model)
    frozen = total - trainable
    if cfg_path is not None:
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


def print_markdown(model: torch.nn.Module, rows: list[tuple[str, str, int, int]], cfg_path: Path | None) -> None:
    total, trainable = count_params(model)
    frozen = total - trainable
    if cfg_path is not None:
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


def print_model_summary(
    model: torch.nn.Module,
    cfg_path: Path | str | None = None,
    fmt: str = "plain",
    depth: int = 2,
    print_repr: bool = False,
) -> None:
    """Print a parameter-count summary for any model (works for every src/ model;
    parameter counting is architecture-agnostic). Safe to call right before training."""
    cfg_path = Path(cfg_path) if cfg_path is not None else None
    rows = child_rows(model, max_depth=depth)
    if fmt == "markdown":
        print_markdown(model, rows, cfg_path)
    else:
        print_plain(model, rows, cfg_path)
    if print_repr:
        print()
        print(model)
