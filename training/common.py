"""Shared training/eval entry-point glue.

The implementation now lives in the ``training.utils`` package, split by concern:

* ``training.utils.constants``   — class-group indices, view aliases, channel config
* ``training.utils.system``      — host-RAM / malloc / DataLoader-IPC tuning
* ``training.utils.config``      — config loading, path/run-dir resolution, cfg->kwargs
* ``training.utils.model``       — precision, torch.compile, channels_last, EMA, ckpt I/O
* ``training.utils.data``        — datasets, loaders, channel precompute, text embeddings
* ``training.utils.metrics``     — compute_metrics + console summaries
* ``training.utils.train``       — the training loop (``train_model``) and its plumbing
* ``training.utils.evaluation``  — ``predict_dataframe`` + report-ablation driver

Everything is re-exported here, so ``from training.common import X`` keeps working
unchanged for every training/eval script. ``add_common_args`` (the shared CLI flag
definitions) is kept *in this file* on purpose, so the full flag set is in one place
to look up — see ``training/FLAGS.md`` for which flag applies to which script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

# Re-export the full helper API so existing `from training.common import ...` lines
# keep resolving. (Star imports from modules without __all__ pull only public names.)
from training.utils.constants import *  # noqa: F401,F403
from training.utils.system import *  # noqa: F401,F403
from training.utils.config import *  # noqa: F401,F403
from training.utils.model import *  # noqa: F401,F403
from training.utils.data import *  # noqa: F401,F403
from training.utils.metrics import *  # noqa: F401,F403
from training.utils.train import *  # noqa: F401,F403
from training.utils.evaluation import *  # noqa: F401,F403

# Names add_common_args references directly (kept explicit so they survive the
# star-import shuffle and read clearly here).
from training.utils.constants import CPU_FRACTION, ENABLED_THIRD_CHANNELS
from src.loss import LOSS_REGISTRY


class _DefaultsHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Preserve layout and append config/runtime defaults beside each flag."""

    def __init__(self, *args, config_defaults: dict[str, Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_defaults = config_defaults or {}

    def _get_help_string(self, action):
        help_text = action.help or ""
        if action.dest in self._config_defaults and "%(default)" not in help_text:
            default_text = f"(default: {_format_default_value(self._config_defaults[action.dest])})"
            return f"{help_text} {default_text}" if help_text else default_text
        return help_text


def _load_default_config(config_path: str | Path) -> tuple[Path, dict[str, Any] | None, str | None]:
    path = resolve_path(config_path) or Path(config_path)
    try:
        with open(path, "r") as f:
            return path, yaml.safe_load(f) or {}, None
    except OSError as exc:
        return path, None, str(exc)


def _format_default_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return "[" + ", ".join(_format_default_value(v) for v in value) + "]"
    if isinstance(value, dict):
        compact = yaml.safe_dump(value, sort_keys=False, default_flow_style=True, width=120).strip()
        return compact.removesuffix("\n")
    return str(value)


def _third_channel_default(channel_mode: Any) -> str | None:
    if channel_mode is None:
        return "none"
    for short_name, full_name in THIRD_CHANNEL_TO_MODE.items():
        if channel_mode == full_name:
            return short_name
    return str(channel_mode)


def _value_at(root: dict[str, Any], path: tuple[str, ...], fallback: Any = None) -> Any:
    cur: Any = root
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return fallback
        cur = cur[key]
    return cur


def _config_default_by_dest(
    cfg: dict[str, Any],
    *,
    config_default: str,
    model_name: str,
    train_only: bool,
) -> dict[str, Any]:
    model = dict(cfg.get("model", {}) or {})
    data = dict(cfg.get("data", {}) or {})
    datamodule = dict(data.get("datamodule_cfg", {}) or {})
    dataloader = dict(data.get("dataloader_init_args", {}) or {})
    model_init = dict(model.get("model_init_args", {}) or {})
    optimizer = dict(model.get("optimizer_init_args", {}) or {})
    scheduler = dict(model.get("scheduler_init_args", {}) or {})
    trainer = dict(cfg.get("trainer", {}) or {})

    defaults: dict[str, Any] = {
        "config": config_default,
        "print_config_defaults": False,
        "train_df_path": datamodule.get("train_df_path"),
        "val_df_path": datamodule.get("devel_df_path"),
        "test_df_path": datamodule.get("pred_df_path"),
        "output_dir": f"output/{model_name}/runs",
        "run_name": "baseline",
        "batch_size": dataloader.get("batch_size"),
        "val_batch_size": "2 x train batch_size",
        "num_workers": dataloader.get("num_workers"),
        "val_num_workers": trainer.get("val_num_workers", 0),
        "prefetch_factor": dataloader.get("prefetch_factor"),
        "malloc_arena_max": 2,
        "image_size": datamodule.get("size"),
        "third_channel_mode": _third_channel_default(datamodule.get("channel_mode")),
        "cpu_fraction": CPU_FRACTION,
        "skip_precompute": False,
        "seed": trainer.get("seed"),
        "accelerator": trainer.get("accelerator"),
        "precision": trainer.get("precision", "16-mixed"),
        "backbone_name": _value_at(model, ("timm_init_args", "model_name")),
        "no_pretrained": not bool(_value_at(model, ("timm_init_args", "pretrained"), True)),
        "text_model": model.get("text_model") or datamodule.get("tokenizer"),
        "freeze_text_encoder": model_init.get("freeze_text_encoder"),
        "use_precomputed_text_embeddings": (
            datamodule.get("use_text_embedding_cache")
            or model_init.get("use_precomputed_text_embeddings")
        ),
        "text_embedding_cache_dir": datamodule.get("text_embedding_cache_dir"),
        "text_embeddings_gpu_resident": False,
        "uint8_image_pipeline": datamodule.get("uint8_image_pipeline", False),
        "skip_report_ablation": False,
    }

    if train_only:
        defaults.update(
            {
                "quick_continue": False,
                "loss": model.get("loss", "ASL"),
                "loss_weights": model.get("loss_weights"),
                "lr": model.get("lr"),
                "weight_decay": optimizer.get("weight_decay", 0.01),
                "warmup_ratio": scheduler.get("warmup_ratio", 0.05),
                "max_epochs": trainer.get("max_epochs", 1000),
                "accumulate_grad_batches": trainer.get("accumulate_grad_batches", 1),
                "grad_clip": trainer.get("grad_clip", 1.0),
                "ema": trainer.get("ema", False),
                "ema_decay": trainer.get("ema_decay", 0.999),
                "val_check_interval": trainer.get("val_check_interval"),
                "log_every_n_steps": trainer.get("log_every_n_steps", 1),
                "early_stop_monitor": trainer.get("early_stop_monitor", "val_ap"),
                "early_stop_mode": trainer.get("early_stop_mode", "max"),
                "early_stop_patience": trainer.get("early_stop_patience", 3),
                "early_stop_min_delta": trainer.get("early_stop_min_delta", 0.0),
                "quick_val_every_steps": trainer.get("quick_val_every_steps"),
                "quick_val_frac": trainer.get("quick_val_frac", 0.1),
                "full_val_fracs": trainer.get("full_val_fracs", []),
                "quick_val_fracs": trainer.get("quick_val_fracs", [0.33, 0.66]),
                "compile_model": trainer.get("compile_model", False),
                "channels_last": trainer.get("channels_last", False),
                "gradcam_epochs": trainer.get("gradcam_epochs", "all"),
                "gradcam_device": trainer.get("gradcam_device", "training device"),
                "fast_dev_run": False,
            }
        )

    return {key: value for key, value in defaults.items() if value is not None}


def _print_full_config_defaults(config_path: str | Path) -> None:
    path, cfg, error = _load_default_config(config_path)
    if error:
        raise SystemExit(f"--print-config-defaults: could not read {path}: {error}")
    assert cfg is not None
    print(f"# Default config params from {path}")
    print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100).rstrip())


def _install_print_config_defaults(parser: argparse.ArgumentParser) -> None:
    if getattr(parser, "_prints_config_defaults", False):
        return
    original_parse_args = parser.parse_args

    def parse_args_with_config_defaults(args=None, namespace=None):
        parsed = original_parse_args(args=args, namespace=namespace)
        if getattr(parsed, "print_config_defaults", False):
            _print_full_config_defaults(parsed.config)
            raise SystemExit(0)
        return parsed

    parser.parse_args = parse_args_with_config_defaults
    setattr(parser, "_prints_config_defaults", True)


def add_common_args(parser: argparse.ArgumentParser, model_name: str, default_config: str | None = None, mode: str = "train") -> None:
    """Flags shared by training/eval entry points, grouped by concern so the set is easy
    to scan here and in ``--help``. Model-specific flags (image pretrained paths,
    text-model / embedding-cache options, single-view position, eval output paths) are
    added by each script's own ``parse_args``. See ``training/FLAGS.md`` for the
    full map of which flag applies where.

    ``mode`` selects which flags are registered. ``mode="eval"`` (passed by the
    ``*_eval.py`` scripts) drops the train-only groups the eval path never reads --
    optimization, EMA, validation/early-stop, ``--compile-model`` / ``--channels-last``,
    Grad-CAM, ``--resume-from`` / ``--quick-continue``, ``--fast-dev-run`` -- so
    ``eval --help`` shows only what actually does something, and adds the eval-only
    ``--skip-report-ablation``. The full flag set is still defined here in one place;
    the gating only controls visibility per entry point."""
    train_only = mode != "eval"
    config_default = default_config or f"training/{model_name}/config.yaml"
    config_path, cfg, config_error = _load_default_config(config_default)
    config_defaults = (
        _config_default_by_dest(
            cfg,
            config_default=str(config_path),
            model_name=model_name,
            train_only=train_only,
        )
        if cfg is not None
        else {"config": str(config_path)}
    )
    parser.formatter_class = lambda prog: _DefaultsHelpFormatter(prog, config_defaults=config_defaults)
    defaults_note = (
        f"Defaults shown above are read from {config_path}; CLI flags override them. "
        "Use --print-config-defaults for the full YAML."
    )
    if config_error:
        defaults_note = f"Could not read default config {config_path}: {config_error}"
    parser.epilog = f"{parser.epilog}\n\n{defaults_note}" if parser.epilog else defaults_note
    _install_print_config_defaults(parser)

    # --- Run identity & I/O -------------------------------------------------
    g = parser.add_argument_group("run identity & I/O")
    g.add_argument("--config", default=config_default, help="YAML config to use.")
    g.add_argument(
        "--print-config-defaults",
        action="store_true",
        help="Print the full YAML defaults for this model (or --config) and exit.",
    )
    g.add_argument("--train-df-path", help="Override data.datamodule_cfg.train_df_path.")
    g.add_argument("--val-df-path", help="Override data.datamodule_cfg.devel_df_path.")
    g.add_argument("--test-df-path", help="Override data.datamodule_cfg.pred_df_path.")
    g.add_argument("--output-dir", default=f"output/{model_name}/runs", help="Directory for run outputs.")
    g.add_argument("--run-name", default="baseline", help="Human-readable suffix for a new run directory.")
    g.add_argument("--run-id", help="Run id prefix/name to create or select.")

    # --- Checkpointing & resume --------------------------------------------
    g = parser.add_argument_group("checkpointing & resume")
    g.add_argument(
        "--checkpoint-path",
        help="Checkpoint path. In eval: weights to load (and, unless --config is passed, the config is auto-resolved from this checkpoint's run dir). In train: weights-only init (fresh optimizer/scheduler/epoch).",
    )
    if train_only:
        g.add_argument(
            "--resume-from",
            help="Train only: full resume from checkpoint (model + optimizer + scheduler + epoch + global_step + early-stop state). Run dir is inferred from the checkpoint path.",
        )
        g.add_argument(
            "--quick-continue",
            action="store_true",
            help=(
                "Train only: resume the most recently created run under --output-dir (its last.pt). "
                "Use --run-id to force a specific run instead."
            ),
        )
        g.add_argument(
            "--keep-epoch-checkpoints",
            action="store_true",
            help=(
                "Train only: also archive a full epoch_NNN.pt every epoch. Off by default -- "
                "training keeps only last.pt (full, for resume) and best.pt (weights-only)."
            ),
        )
    g.add_argument("--seed", type=int, help="Optional seed for python/numpy/torch RNGs.")

    # --- Data & batching ----------------------------------------------------
    g = parser.add_argument_group("data & batching")
    g.add_argument("--batch-size", type=int, help="Training DataLoader batch size.")
    g.add_argument(
        "--val-batch-size",
        type=int,
        help="Batch size for validation/eval loaders (forward-only, so it can exceed the train "
        "batch size). Defaults to 2x the train batch size. A one-time OOM fallback downgrades it "
        "to the train batch size if the larger batch doesn't fit alongside the live training state.",
    )
    g.add_argument("--num-workers", type=int, help="Training DataLoader worker processes.")
    g.add_argument(
        "--val-num-workers",
        type=int,
        help="DataLoader num_workers for validation/eval loaders. Defaults to 0 (from trainer.val_num_workers) "
        "so validation runs in the main process and does NOT fork a second worker pool on top of the "
        "persistent train workers — avoids the host-RAM spike (and OOM kill) when mid-epoch validation "
        "fires. Quick-val is only a few batches, so in-process cost is small. Raise it to use val workers.",
    )
    g.add_argument("--prefetch-factor", type=int, help="DataLoader prefetch_factor (requires num_workers > 0).")
    g.add_argument(
        "--malloc-arena-max",
        type=int,
        help="Cap glibc malloc arenas to control host-RAM/RSS under num_workers>0 (each fork "
        "worker otherwise grows its own arena set, up to ~8*ncpu, fragmenting RSS). Default 2; "
        "0 leaves the glibc default (no cap). Applied via mallopt + MALLOC_ARENA_MAX before "
        "workers fork. No effect off glibc (musl/non-Linux).",
    )
    g.add_argument("--image-size", type=int, help="Input image size.")
    g.add_argument(
        "--third-channel-mode",
        choices=ENABLED_THIRD_CHANNELS + ["none"],
        help=(
            "Third channel of the 3-channel CXR build. ch0=raw and ch1=mild CLAHE "
            "(clip 2.0 / 8x8) are always pinned; this flag picks only ch2. Enabled: "
            f"{ENABLED_THIRD_CHANNELS} (clahe = strong CLAHE clip 4.0/16x16; histeq = "
            "global histogram equalization; lbp = uniform Local Binary Pattern). "
            "'none' = legacy ImageNet RGB (no channel build). Overrides "
            "data.datamodule_cfg.channel_mode; the on-disk cache "
            "(image_channel_cache_dir) stays shared across models via config."
        ),
    )
    g.add_argument(
        "--cpu-fraction",
        type=float,
        help=(
            "Fraction of CPU cores used to precompute the image channel cache "
            "(threads for the scan, processes for the build). "
            f"Default {CPU_FRACTION}. e.g. 0.25 to be gentle on a shared box, "
            "1.0 to use all cores."
        ),
    )
    g.add_argument(
        "--skip-precompute",
        action="store_true",
        help=(
            "Skip the upfront image channel-cache scan/build. The cache config is "
            "left intact, so channels build lazily on first access (and still cache). "
            "Use when the cache is already warm to shave startup latency."
        ),
    )

    if train_only:
        # --- Optimization ---------------------------------------------------
        g = parser.add_argument_group("optimization")
        g.add_argument(
            "--loss",
            nargs="+",
            metavar="NAME",
            help="Loss function(s), overriding model.loss / the default ASL. One name -> that loss "
            f"(available: {', '.join(sorted(LOSS_REGISTRY))}); several -> their weighted sum, e.g. "
            "'--loss FC ASL'. Per-loss kwargs come from model.loss_kwargs.<NAME>; ASL also inherits "
            "model.loss_init_args. Weights: --loss-weights or model.loss_weights (default 1.0 each).",
        )
        g.add_argument(
            "--loss-weights",
            nargs="+",
            type=float,
            metavar="W",
            help="Weights for a multi-loss --loss, positionally matched (e.g. '--loss FC ASL "
            "--loss-weights 1.0 0.5'). Must match the number of losses.",
        )
        g.add_argument(
            "--label-smoothing",
            type=float,
            metavar="EPS",
            help="Positive-only (asymmetric) label smoothing for ASL: soften positive targets "
            "1 -> 1-EPS (negatives stay 0). Overrides model.loss_init_args.pos_smoothing. "
            "Default 0.0 (off). A noise/calibration knob, not an imbalance fix; try ~0.05.",
        )
        g.add_argument("--lr", type=float, help="Base learning rate.")
        g.add_argument("--weight-decay", type=float, help="AdamW weight decay.")
        g.add_argument(
            "--backbone-lr-mult",
            type=float,
            help="Discriminative LR: train the pretrained image backbone (params under "
            "'image_encoder.') at this multiple of the base --lr. Default 0.3; set 1.0 to "
            "disable (uniform LR). Ignored if the config sets optimizer_init_args.param_group_lrs.",
        )
        g.add_argument("--warmup-ratio", type=float, help="Warmup steps as a fraction of steps_per_epoch (default 0.05).")
        g.add_argument("--max-epochs", type=int, help="Maximum training epochs.")
        g.add_argument("--accumulate-grad-batches", type=int, help="Gradient accumulation steps.")
        g.add_argument("--grad-clip", type=float, help="Max grad norm. Set to 0 or negative to disable. Default 1.0 if neither CLI nor config set.")

        # --- EMA (weight averaging) ----------------------------------------
        g = parser.add_argument_group("EMA (weight averaging)")
        g.add_argument(
            "--ema",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Maintain an exponential moving average of the weights and use it as the evaluated/saved "
            "model. Off by default (config trainer.ema overrides). Best paired with schedule=single_cosine; "
            "do NOT --resume-from an EMA checkpoint (saved weights are the EMA snapshot, not the raw state).",
        )
        g.add_argument(
            "--ema-decay",
            type=float,
            help="EMA decay factor (default 0.999, or trainer.ema_decay). Higher = smoother/slower.",
        )

        # --- Validation, early stopping & logging --------------------------
        g = parser.add_argument_group("validation, early stopping & logging")
        g.add_argument("--val-check-interval", type=float)
        g.add_argument("--log-every-n-steps", type=int, help="How often to write a row to train_steps.csv. 1 = every optimizer step.")
        g.add_argument("--early-stop-monitor", help="Epoch validation metric to monitor for early stopping. Default val_ap.")
        g.add_argument("--early-stop-mode", choices=["min", "max"], help="Whether early-stop monitor should decrease or increase. Default max.")
        g.add_argument("--early-stop-patience", type=int, help="Stop after this many non-improving epochs. Default 3; set 0 to disable.")
        g.add_argument("--early-stop-min-delta", type=float, help="Minimum metric change required to count as an improvement. Default 0.0.")
        g.add_argument(
            "--quick-val-every-steps",
            type=int,
            help="If set, run a partial validation every N optimizer steps (= N effective batches, where effective = batch_size * accumulate_grad_batches). Logged to val_quick.csv. Does not affect best-checkpoint tracking.",
        )
        g.add_argument(
            "--quick-val-frac",
            type=float,
            help="Fraction of val loader batches to use for quick validation (default 0.1).",
        )
        g.add_argument(
            "--full-val-fracs",
            type=float,
            nargs="*",
            help="Epoch fractions (each in (0,1)) at which to run a full mid-epoch validation. "
            "Independent of batch size. End-of-epoch full validation always runs. Default: none "
            "(no mid-epoch full validation). Pass fractions to enable.",
        )
        g.add_argument(
            "--quick-val-fracs",
            type=float,
            nargs="*",
            help="Epoch fractions (each in (0,1)) at which to run a partial quick validation "
            "(logged to val_quick.csv, does not affect best-checkpoint tracking). Independent of "
            "batch size. Default 0.33 0.66. Pass with no values to disable.",
        )

    # --- Hardware, precision & compile -------------------------------------
    g = parser.add_argument_group("hardware, precision & compile")
    g.add_argument("--accelerator", default=None, help="Device selector: auto/gpu/cpu/cuda/mps.")
    g.add_argument("--devices", default=None, help="Reserved compatibility flag; single-device trainer currently ignores it.")
    g.add_argument("--precision", default=None, help="Training/eval precision, e.g. 16-mixed, bf16-mixed, 32-true.")
    if train_only:
        g.add_argument(
            "--compile-model",
            action="store_true",
            default=None,
            help="Opt in to torch.compile (automatic dynamic) on the compile-safe submodules (image backbones, text encoder, transformer encoder, head) before training; the data-dependent fusion stays eager. Compiled in place so checkpoint keys are unchanged. Compile failures fall back to eager. Default comes from trainer.compile_model, or false if unset.",
        )
        g.add_argument(
            "--channels-last",
            action="store_true",
            default=None,
            help="(opt-in) Run the conv image backbone(s) in torch.channels_last (NHWC) memory format. "
            "Layout-only (numerics identical), but it lets cuDNN use the native NHWC fp16/bf16 "
            "Tensor-Core conv kernels and skip the per-layer NCHW<->NHWC transposes, typically a "
            "10-30%% throughput win on conv-heavy nets (ConvNeXtV2). Converts the model's 4D conv "
            "weights and feeds conv inputs as channels_last. Default comes from trainer.channels_last, "
            "or false if unset.",
        )

    # --- Backbone -----------------------------------------------------------
    g = parser.add_argument_group("backbone")
    g.add_argument("--backbone-name", help="Override model.timm_init_args.model_name.")
    g.add_argument("--no-pretrained", action="store_true", help="Disable timm pretrained backbone weights.")

    if train_only:
        # --- Debug ----------------------------------------------------------
        g = parser.add_argument_group("debug")
        g.add_argument("--fast-dev-run", action="store_true", help="Run a tiny one-epoch smoke test.")

        # --- Grad-CAM panels ------------------------------------------------
        g = parser.add_argument_group("Grad-CAM panels")
        g.add_argument(
            "--gradcam-epochs",
            help="When to dump per-class Grad-CAM panels: 'all' (default, every epoch — it's cheap), "
                 "'none' to disable, or a comma list of 0-indexed epochs (e.g. '0,4,9'). "
                 "Only models that define gradcam_runner_module emit panels.",
        )
        g.add_argument(
            "--gradcam-device",
            help="Device for the Grad-CAM subprocess (cpu/cuda/mps). Default: the training device.",
        )
    else:
        # --- Eval-only ------------------------------------------------------
        g = parser.add_argument_group("eval-only")
        g.add_argument(
            "--skip-report-ablation",
            action="store_true",
            help=(
                "Eval only: by default every text model is evaluated twice -- once with the full "
                "inputs (image + clinical indication [+ vitals]) and once with the CURRENT study's "
                "clinical indication blanked to the in-distribution 'No clinical history available.' "
                "placeholder, written to *.no_report.{csv,json}, to measure how much the report text "
                "is carrying (a leakage probe). The prior study's report/text is kept either way. "
                "Pass this flag to run only the full pass."
            ),
        )
        g.add_argument(
            "--skip-task2-gold",
            action="store_true",
            help=(
                "Eval only: by default a model trained on a CXR-LT 2024 label set that is a "
                "superset of the 26 task2 labels (i.e. task1 / all) is ALSO scored on the task2 "
                "(gold) test set -- its wide outputs are sliced to the 26 task2 columns and written "
                "to *_task2_gold.{csv,json}. The extra pass needs the task2 gold file to exist next "
                "to the eval file (task1 -> task2 in the name). Pass this flag to skip it."
            ),
        )
