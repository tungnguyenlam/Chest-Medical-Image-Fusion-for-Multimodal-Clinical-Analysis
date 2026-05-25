"""Plot training/validation curves from a run directory.

Reads `<run_dir>/logs/train_steps.csv` and `<run_dir>/logs/val_epochs.csv`
(produced by training/common.py) and writes PNGs to `<run_dir>/plots/`.

Usage:
    python scripts/plot_run.py <run_dir> [<run_dir> ...]
    python scripts/plot_run.py <run_dir> --output-dir <other_dir>
    python scripts/plot_run.py <run_dir> --smooth 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Write plots here instead of <run_dir>/plots.")
    parser.add_argument("--smooth", type=int, default=0, help="Rolling-mean window (in steps) for the train loss curve. 0 disables.")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=fig.get_dpi())
    plt.close(fig)
    print(f"  wrote {path.relative_to(path.parents[1]) if path.parents[1].exists() else path}")


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  skip: {path} not found")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"  skip: {path} is empty")
        return None
    return df


def plot_loss(train_df: pd.DataFrame | None, val_df: pd.DataFrame | None, smooth: int, out: Path, dpi: int) -> None:
    if train_df is None and val_df is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    if train_df is not None and "train/loss_step" in train_df.columns:
        ax.plot(train_df["global_step"], train_df["train/loss_step"], color="tab:blue", alpha=0.25, linewidth=0.6, label="train loss (step)")
        if smooth and smooth > 1 and len(train_df) >= smooth:
            smoothed = train_df["train/loss_step"].rolling(smooth, min_periods=1).mean()
            ax.plot(train_df["global_step"], smoothed, color="tab:blue", linewidth=1.4, label=f"train loss (rolling {smooth})")
        elif "train/loss_running" in train_df.columns:
            ax.plot(train_df["global_step"], train_df["train/loss_running"], color="tab:blue", linewidth=1.2, label="train loss (running mean)")
    if val_df is not None and "val/loss" in val_df.columns:
        ax.plot(val_df["global_step"], val_df["val/loss"], "o-", color="tab:red", markersize=4, label="val loss")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.set_title("loss curves")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out / "loss_curves.png")


def plot_lr(train_df: pd.DataFrame | None, out: Path, dpi: int) -> None:
    if train_df is None or "train/lr" not in train_df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4), dpi=dpi)
    ax.plot(train_df["global_step"], train_df["train/lr"], color="tab:green", linewidth=1.2)
    ax.set_xlabel("global step")
    ax.set_ylabel("learning rate")
    ax.set_title("learning rate schedule")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    _save(fig, out / "lr.png")


def plot_grad_norm(train_df: pd.DataFrame | None, out: Path, dpi: int) -> None:
    if train_df is None or "train/grad_norm" not in train_df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4), dpi=dpi)
    gn = train_df["train/grad_norm"].astype(float)
    finite = np.isfinite(gn)
    ax.plot(train_df.loc[finite, "global_step"], gn[finite], color="tab:purple", alpha=0.3, linewidth=0.6, label="grad norm")
    if finite.sum() >= 50:
        rolling = gn.where(finite).rolling(50, min_periods=1).median()
        ax.plot(train_df["global_step"], rolling, color="tab:purple", linewidth=1.4, label="median (window 50)")
    ax.set_xlabel("global step")
    ax.set_ylabel("grad norm (pre-clip)")
    ax.set_title("gradient norm")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    _save(fig, out / "grad_norm.png")


def plot_scaler_scale(train_df: pd.DataFrame | None, out: Path, dpi: int) -> None:
    if train_df is None or "train/scaler_scale" not in train_df.columns:
        return
    scale = train_df["train/scaler_scale"].astype(float)
    if not np.isfinite(scale).any():
        return
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=dpi)
    ax.plot(train_df["global_step"], scale, color="tab:orange", linewidth=1.0)
    ax.set_xlabel("global step")
    ax.set_ylabel("AMP scaler scale")
    ax.set_title("GradScaler scale (drops indicate NaN/inf in grads)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    _save(fig, out / "scaler_scale.png")


def plot_val_summary(val_df: pd.DataFrame | None, metric: str, out: Path, dpi: int) -> None:
    if val_df is None:
        return
    needed = [f"val/{metric}", f"val/{metric}_head", f"val/{metric}_medium", f"val/{metric}_tail"]
    if not all(c in val_df.columns for c in needed):
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    x = val_df["global_step"]
    ax.plot(x, val_df[f"val/{metric}"], "o-", label=f"mean {metric.upper()}", linewidth=2)
    ax.plot(x, val_df[f"val/{metric}_head"], "s--", label=f"head {metric.upper()}", alpha=0.8)
    ax.plot(x, val_df[f"val/{metric}_medium"], "d--", label=f"medium {metric.upper()}", alpha=0.8)
    ax.plot(x, val_df[f"val/{metric}_tail"], "v--", label=f"tail {metric.upper()}", alpha=0.8)
    ax.set_xlabel("global step")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"validation {metric.upper()} (mean and head/medium/tail)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out / f"val_{metric}.png")


def plot_per_class(val_df: pd.DataFrame | None, metric: str, out: Path, dpi: int) -> None:
    if val_df is None:
        return
    prefix = f"val/{metric}/"
    cols = [c for c in val_df.columns if c.startswith(prefix)]
    if not cols:
        return
    classes = [c[len(prefix):] for c in cols]
    n = len(classes)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.2 * nrows), dpi=dpi, sharex=True)
    axes = np.array(axes).reshape(-1)
    x = val_df["global_step"]
    for ax, name in zip(axes, classes):
        ax.plot(x, val_df[f"{prefix}{name}"], linewidth=1.2)
        ax.set_title(name, fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"per-class {metric.upper()} vs step", y=1.005)
    _save(fig, out / f"per_class_{metric}.png")


def plot_run(run_dir: Path, output_dir: Path | None, smooth: int, dpi: int) -> None:
    logs_dir = run_dir / "logs"
    out = output_dir if output_dir is not None else run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[plot] {run_dir} -> {out}")
    train_df = _read_csv(logs_dir / "train_steps.csv")
    val_df = _read_csv(logs_dir / "val_epochs.csv")
    plot_loss(train_df, val_df, smooth, out, dpi)
    plot_lr(train_df, out, dpi)
    plot_grad_norm(train_df, out, dpi)
    plot_scaler_scale(train_df, out, dpi)
    plot_val_summary(val_df, "ap", out, dpi)
    plot_val_summary(val_df, "auroc", out, dpi)
    plot_per_class(val_df, "ap", out, dpi)
    plot_per_class(val_df, "auroc", out, dpi)


def main() -> None:
    args = parse_args()
    for run_dir in args.run_dirs:
        if not run_dir.exists():
            print(f"[plot] skip: {run_dir} does not exist")
            continue
        plot_run(run_dir, args.output_dir, args.smooth, args.dpi)


if __name__ == "__main__":
    main()
