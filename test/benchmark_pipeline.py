#!/usr/bin/env python3
"""Time the main training pipeline stages (startup + per-batch work).

Run from the repo root, e.g.:

    python test/benchmark_pipeline.py
    python test/benchmark_pipeline.py --config training/prior_aware/config.yaml --skip-precompute
    python test/benchmark_pipeline.py --config training/camchex/config.yaml --data-only

Reports wall-clock for config load, dataloader construction (channel prebuild and
text-embedding cache when enabled), batch fetch, model init, forward, and
forward+backward. Also prints host RSS after the heavy startup steps.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common import (  # noqa: E402
    build_criterion,
    load_config,
    loss_args_from_config,
    make_camchex_loaders,
    make_camchex_vitals_loaders,
    make_prior_aware_loaders,
    make_single_view_loaders,
    model_init_args_from_config,
    resolve_path,
    timm_args_from_config,
)
from training.utils.model import (  # noqa: E402
    move_to_device,
    precision_context,
    resolve_precision,
    select_device,
)
from training.utils.system import host_rss_mb, log_rss  # noqa: E402
from training.utils.train import train_step  # noqa: E402


@dataclass
class BenchRow:
    label: str
    seconds: float
    detail: str = ""


@dataclass
class BenchReport:
    rows: list[BenchRow] = field(default_factory=list)

    def add(self, label: str, seconds: float, detail: str = "") -> None:
        self.rows.append(BenchRow(label, seconds, detail))

    def print_table(self) -> None:
        if not self.rows:
            print("No timings recorded.")
            return
        width = max(len(r.label) for r in self.rows)
        print(f"\n{'stage':<{width}}  {'seconds':>10}  detail")
        print("-" * (width + 14 + 20))
        for row in self.rows:
            detail = f"  {row.detail}" if row.detail else ""
            print(f"{row.label:<{width}}  {row.seconds:10.3f}{detail}")


@contextmanager
def timed(report: BenchReport, label: str, detail: str = ""):
    t0 = time.perf_counter()
    yield
    report.add(label, time.perf_counter() - t0, detail)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def default_args(**overrides: Any) -> argparse.Namespace:
    base = {
        "skip_precompute": False,
        "cpu_fraction": None,
        "third_channel_mode": None,
        "uint8_image_pipeline": False,
        "use_precomputed_text_embeddings": False,
        "text_embedding_cache_dir": None,
        "text_model": None,
        "freeze_text_encoder": False,
        "prefetch_factor": None,
        "val_batch_size": None,
        "train_df_path": None,
        "val_df_path": None,
        "test_df_path": None,
        "view_position": "PA",
        "frontal_pretrained_path": None,
        "lateral_pretrained_path": None,
        "image_size": None,
        "batch_size": None,
        "num_workers": None,
        "backbone_name": None,
        "no_pretrained": False,
        "lr": None,
        "malloc_arena_max": None,
        "label_smoothing": None,
        "loss": None,
        "loss_weights": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_prior_aware_model(cfg: dict[str, Any], args: argparse.Namespace, arch: str):
    model_init_args = dict(model_init_args_from_config(cfg))
    if args.freeze_text_encoder:
        model_init_args["freeze_text_encoder"] = True
    data_cfg = cfg.get("data", {}).get("datamodule_cfg", {}) or {}
    if args.use_precomputed_text_embeddings or data_cfg.get("use_text_embedding_cache", False):
        model_init_args["use_precomputed_text_embeddings"] = True
        model_init_args["freeze_text_encoder"] = True

    frontal = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None
    text_model = args.text_model or cfg.get("model", {}).get("text_model") or "dmis-lab/biobert-v1.1"
    timm_kwargs = {
        "timm_init_args": timm_args_from_config(cfg, args),
        "frontal_pretrained_path": frontal,
        "lateral_pretrained_path": lateral,
        "text_model": text_model,
        **model_init_args,
    }

    if arch in {"prior_aware", "prior_aware_cxrbert"}:
        from src.model.PriorAwareCaMCheXModel import PriorAwareCaMCheXModel

        return PriorAwareCaMCheXModel(**timm_kwargs)
    if arch == "prior_aware_v2nano":
        from src.model.PriorAwareV2NanoModel import PriorAwareV2NanoModel

        return PriorAwareV2NanoModel(**timm_kwargs)
    if arch == "prior_aware_v3nano":
        from src.model.PriorAwareV3NanoModel import PriorAwareV3NanoModel

        return PriorAwareV3NanoModel(**timm_kwargs)
    if arch == "prior_aware_v4nano":
        from src.model.PriorAwareV4NanoModel import PriorAwareV4NanoModel

        return PriorAwareV4NanoModel(**timm_kwargs)
    if arch in {"prior_aware_v5nano", "prior_aware_v5nano_explore_exploit", "prior_aware_v5nano_explore_exploit_lsmooth"}:
        from src.model.PriorAwareV5NanoModel import PriorAwareV5NanoModel

        return PriorAwareV5NanoModel(**timm_kwargs)
    if arch == "prior_aware_v5nano_bgpenalty":
        from src.model.PriorAwareV5NanoModel import PriorAwareV5NanoModel

        return PriorAwareV5NanoModel(**timm_kwargs)
    if arch == "prior_aware_v6nano":
        from src.model.PriorAwareV6NanoModel import PriorAwareV6NanoModel

        return PriorAwareV6NanoModel(**timm_kwargs)
    raise ValueError(f"unsupported prior-aware arch: {arch}")


def _build_camchex_model(cfg: dict[str, Any], args: argparse.Namespace, arch: str):
    model_init_args = dict(model_init_args_from_config(cfg))
    data_cfg = cfg.get("data", {}).get("datamodule_cfg", {}) or {}
    if args.use_precomputed_text_embeddings or data_cfg.get("use_text_embedding_cache", False):
        model_init_args["use_precomputed_text_embeddings"] = True
        model_init_args["freeze_text_encoder"] = True
    if args.freeze_text_encoder:
        model_init_args["freeze_text_encoder"] = True

    frontal = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None
    text_model = args.text_model or cfg.get("model", {}).get("text_model") or "dmis-lab/biobert-v1.1"
    timm_kwargs = {
        "timm_init_args": timm_args_from_config(cfg, args),
        "frontal_pretrained_path": frontal,
        "lateral_pretrained_path": lateral,
        "text_model": text_model,
        **model_init_args,
    }

    if arch in {"camchex", "camchex_cxrbert"}:
        from src.model.CaMCheXModel import CaMCheXModel

        return CaMCheXModel(**timm_kwargs)
    if arch in {"camchex_v2nano_vitals", "camchex_v2nano_vitals_stable"}:
        from src.model.CaMCheXV2NanoVitalsModel import CaMCheXV2NanoVitalsModel

        return CaMCheXV2NanoVitalsModel(**timm_kwargs)
    if arch == "camchex_v3nano":
        from src.model.CaMCheXV3NanoModel import CaMCheXV3NanoModel

        return CaMCheXV3NanoModel(**timm_kwargs)
    raise ValueError(f"unsupported camchex arch: {arch}")


def resolve_pipeline(arch: str) -> tuple[Callable, Callable]:
    prior_arches = {
        "prior_aware",
        "prior_aware_cxrbert",
        "prior_aware_v2nano",
        "prior_aware_v3nano",
        "prior_aware_v4nano",
        "prior_aware_v5nano",
        "prior_aware_v5nano_bgpenalty",
        "prior_aware_v5nano_explore_exploit",
        "prior_aware_v5nano_explore_exploit_lsmooth",
        "prior_aware_v6nano",
    }
    camchex_arches = {"camchex", "camchex_cxrbert"}
    vitals_arches = {"camchex_v2nano_vitals", "camchex_v2nano_vitals_stable", "camchex_v3nano"}

    if arch in prior_arches:
        return make_prior_aware_loaders, lambda cfg, args, a=arch: _build_prior_aware_model(cfg, args, a)
    if arch in camchex_arches:
        return make_camchex_loaders, lambda cfg, args, a=arch: _build_camchex_model(cfg, args, a)
    if arch in vitals_arches:
        return make_camchex_vitals_loaders, lambda cfg, args, a=arch: _build_camchex_model(cfg, args, a)
    if arch == "singleview":
        from src.model.SingleViewModel import SingleViewModel

        def _build_singleview(cfg, args):
            return SingleViewModel(timm_args_from_config(cfg, args), **model_init_args_from_config(cfg))

        return make_single_view_loaders, _build_singleview
    raise ValueError(
        f"unknown arch {arch!r}; supported: "
        + ", ".join(sorted(prior_arches | camchex_arches | vitals_arches | {"singleview"}))
    )


def benchmark_dataloader(
    loader,
    *,
    warmup_batches: int,
    timed_batches: int,
    report: BenchReport,
) -> Any:
    it = iter(loader)
    last_batch = None
    for i in range(warmup_batches):
        with timed(report, f"dataloader warmup batch {i + 1}"):
            last_batch = next(it)

    batch_times: list[float] = []
    for i in range(timed_batches):
        t0 = time.perf_counter()
        last_batch = next(it)
        batch_times.append(time.perf_counter() - t0)

    mean_s = statistics.mean(batch_times)
    detail = (
        f"mean={mean_s:.3f}s min={min(batch_times):.3f}s max={max(batch_times):.3f}s "
        f"batch_size={loader.batch_size} workers={loader.num_workers}"
    )
    report.add("dataloader (timed batches)", sum(batch_times), detail)
    return last_batch


def benchmark_model(
    model,
    criterion,
    batch,
    *,
    device: torch.device,
    precision: str,
    report: BenchReport,
    warmup_steps: int,
    timed_steps: int,
) -> None:
    model = model.to(device)
    log_rss("model on device")

    for i in range(warmup_steps):
        with timed(report, f"forward warmup {i + 1}"):
            with torch.inference_mode():
                data, _ = batch
                data = move_to_device(data, device)
                with precision_context(device, precision):
                    _ = model(data)
                _sync(device)

    forward_times: list[float] = []
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        with torch.inference_mode():
            data, _ = batch
            data = move_to_device(data, device)
            with precision_context(device, precision):
                _ = model(data)
            _sync(device)
        forward_times.append(time.perf_counter() - t0)
    report.add(
        "forward (timed)",
        sum(forward_times),
        f"mean={statistics.mean(forward_times):.3f}s device={device.type}",
    )

    backward_times: list[float] = []
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        loss, _, _ = train_step(model, criterion, batch, device, precision)
        loss.backward()
        model.zero_grad(set_to_none=True)
        _sync(device)
        backward_times.append(time.perf_counter() - t0)
    report.add(
        "forward+backward (timed)",
        sum(backward_times),
        f"mean={statistics.mean(backward_times):.3f}s device={device.type}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="training/prior_aware/config.yaml", help="Training YAML config.")
    parser.add_argument("--arch", help="Override model.arch from the config.")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip upfront channel prebuild.")
    parser.add_argument(
        "--use-precomputed-text-embeddings",
        action="store_true",
        help="Use the frozen text-embedding cache (and skip loading BioBERT).",
    )
    parser.add_argument("--data-only", action="store_true", help="Only benchmark config + dataloader startup.")
    parser.add_argument("--warmup-batches", type=int, default=2, help="Dataloader batches before timing.")
    parser.add_argument("--timed-batches", type=int, default=5, help="Dataloader batches to average.")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Model forward warmups before timing.")
    parser.add_argument("--timed-steps", type=int, default=3, help="Model steps to average.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    report = BenchReport()

    with timed(report, "load config"):
        cfg = load_config(cli.config)
    arch = cli.arch or (cfg.get("model", {}) or {}).get("arch") or "prior_aware"
    make_loaders, build_model = resolve_pipeline(arch)

    train_args = default_args(
        skip_precompute=cli.skip_precompute,
        use_precomputed_text_embeddings=cli.use_precomputed_text_embeddings,
    )

    with timed(report, "build dataloaders", f"arch={arch}"):
        train_loader, _val_loader = make_loaders(cfg, train_args)
    log_rss("after dataloader build")

    batch = benchmark_dataloader(
        train_loader,
        warmup_batches=cli.warmup_batches,
        timed_batches=cli.timed_batches,
        report=report,
    )

    if cli.data_only:
        rss, peak = host_rss_mb()
        if rss == rss:
            report.add("host RSS", rss, f"peak={peak:.0f} MB")
        report.print_table()
        return

    device = select_device(cli.device)
    precision = resolve_precision(device, (cfg.get("trainer", {}) or {}).get("precision"))

    with timed(report, "build model", f"device={device.type}"):
        model = build_model(cfg, train_args)
    criterion = build_criterion(train_args, cfg, loss_args_from_config(cfg)).to(device)

    benchmark_model(
        model,
        criterion,
        batch,
        device=device,
        precision=precision,
        report=report,
        warmup_steps=cli.warmup_steps,
        timed_steps=cli.timed_steps,
    )

    rss, peak = host_rss_mb()
    if rss == rss:
        report.add("host RSS", rss, f"peak={peak:.0f} MB")
    report.print_table()


if __name__ == "__main__":
    main()
