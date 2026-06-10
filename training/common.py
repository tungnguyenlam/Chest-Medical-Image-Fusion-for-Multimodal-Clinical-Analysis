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

    # --- Run identity & I/O -------------------------------------------------
    g = parser.add_argument_group("run identity & I/O")
    g.add_argument("--config", default=default_config or f"training/{model_name}/config.yaml")
    g.add_argument("--train-df-path")
    g.add_argument("--val-df-path")
    g.add_argument("--test-df-path")
    g.add_argument("--output-dir", default=f"output/{model_name}/runs")
    g.add_argument("--run-name", default="baseline")
    g.add_argument("--run-id")

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
                "Train only: resume the most recently created run under --output-dir (its latest checkpoint). "
                "Use --run-id to force a specific run instead."
            ),
        )
    g.add_argument("--seed", type=int, help="Optional seed for python/numpy/torch RNGs.")

    # --- Data & batching ----------------------------------------------------
    g = parser.add_argument_group("data & batching")
    g.add_argument("--batch-size", type=int)
    g.add_argument(
        "--val-batch-size",
        type=int,
        help="Batch size for validation/eval loaders (forward-only, so it can exceed the train "
        "batch size). Defaults to 2x the train batch size. A one-time OOM fallback downgrades it "
        "to the train batch size if the larger batch doesn't fit alongside the live training state.",
    )
    g.add_argument("--num-workers", type=int)
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
    g.add_argument("--image-size", type=int)
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
        g.add_argument("--lr", type=float)
        g.add_argument("--weight-decay", type=float)
        g.add_argument("--warmup-ratio", type=float, help="Warmup steps as a fraction of steps_per_epoch (default 0.05).")
        g.add_argument("--max-epochs", type=int)
        g.add_argument("--accumulate-grad-batches", type=int)
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
            "batch size. Default 0.25 0.5 0.75. Pass with no values to disable.",
        )

    # --- Hardware, precision & compile -------------------------------------
    g = parser.add_argument_group("hardware, precision & compile")
    g.add_argument("--accelerator", default=None)
    g.add_argument("--devices", default=None)
    g.add_argument("--precision", default=None)
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
    g.add_argument("--backbone-name")
    g.add_argument("--no-pretrained", action="store_true")

    if train_only:
        # --- Debug ----------------------------------------------------------
        g = parser.add_argument_group("debug")
        g.add_argument("--fast-dev-run", action="store_true")

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
