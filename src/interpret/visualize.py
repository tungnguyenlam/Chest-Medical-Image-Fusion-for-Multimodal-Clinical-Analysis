"""Render an AttributionResult as separate, inspect-by-hand images.

Three files per class (each gets its own room instead of one cramped figure):

    <class>/image.png    one row per CXR view: original | Grad-CAM overlay
    <class>/text.png     indication tokens coloured by grad x embedding
    <class>/vitals.png   signed grad x value per vital + modality share

``render_attribution_split(result, out_dir)`` writes all three and returns the paths.
``render_attribution(result, path)`` still produces the old single combined figure if wanted.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from src.interpret.attribution import VIEW_NAMES, AttributionResult


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def render_attribution_split(result: AttributionResult, out_dir: str | Path, tokens_per_row: int = 10) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        render_images(result, out_dir / "image.png"),
        render_text(result, out_dir / "text.png", tokens_per_row=tokens_per_row),
        render_vitals(result, out_dir / "vitals.png"),
    ]
    return paths


def _header(result: AttributionResult) -> str:
    label = "n/a" if result.label is None else str(int(result.label))
    head = (
        f"{result.class_name}   |   study={result.study_id}   |   "
        f"p={result.prob:.3f}   |   logit={result.logit:+.2f}   |   true={label}"
    )
    others = [c for c in result.true_labels if c != result.class_name]
    if others:
        head += f"\nalso positive: {', '.join(others)}"
    return head


# --------------------------------------------------------------------------- #
# image: one row per view -> original | Grad-CAM overlay
# --------------------------------------------------------------------------- #
def render_images(result: AttributionResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    views = result.views or []
    n = max(1, len(views))
    total = sum(v.contribution for v in views) + 1e-8

    fig, axes = plt.subplots(n, 2, figsize=(8.5, 4.2 * n), squeeze=False)
    fig.suptitle(_header(result), fontsize=13, fontweight="bold")

    if not views:
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, "(no images)", ha="center", va="center")

    for row, v in enumerate(views):
        name = VIEW_NAMES.get(v.view_position, "?")
        ax_o, ax_h = axes[row, 0], axes[row, 1]
        for ax in (ax_o, ax_h):
            ax.set_xticks([])
            ax.set_yticks([])

        ax_o.imshow(v.image, cmap="gray", vmin=0, vmax=1)
        ax_o.set_title(f"{name} — original", fontsize=11)

        ax_h.imshow(v.image, cmap="gray", vmin=0, vmax=1)
        if v.encoded:
            hm = ax_h.imshow(v.cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            share = 100.0 * v.contribution / total
            ax_h.set_title(f"{name} — Grad-CAM ({share:.0f}% of image signal)", fontsize=11)
            cbar = fig.colorbar(hm, ax=ax_h, fraction=0.046, pad=0.04)
            cbar.set_label("relevance", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        else:
            ax_h.set_title(f"{name} — not encoded (view=0)", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# text: token chips coloured by grad x embedding
# --------------------------------------------------------------------------- #
def render_text(result: AttributionResult, out_path: str | Path, tokens_per_row: int = 10) -> Path:
    out_path = Path(out_path)
    tokens, scores = result.tokens, result.token_scores
    n_rows = max(1, math.ceil(len(tokens) / tokens_per_row)) if tokens else 1

    fig = plt.figure(figsize=(1.25 * tokens_per_row + 1.0, 1.4 + 0.75 * n_rows))
    gs = gridspec.GridSpec(2, 1, height_ratios=[max(1, n_rows), 0.4], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax.set_title(f"{result.class_name} — clinical indication (grad × embedding)", fontsize=12, loc="left")
    ax.axis("off")

    if not tokens:
        ax.text(0.5, 0.5, "(no text)", ha="center", va="center")
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path

    vmax = float(np.abs(scores).max()) + 1e-8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    for i, (tok, sc) in enumerate(zip(tokens, scores)):
        row, col = divmod(i, tokens_per_row)
        x = (col + 0.5) / tokens_per_row
        y = 1.0 - (row + 0.5) / n_rows
        color = cmap(norm(sc))
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        ax.text(
            x, y, tok.replace("##", ""),
            fontsize=12, va="center", ha="center",
            color="white" if lum < 0.5 else "black",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="0.6", lw=0.5),
            transform=ax.transAxes,
        )

    cax = fig.add_subplot(gs[1])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("← pushes away from class            pushes toward class →", fontsize=9)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# vitals: signed grad x value + modality share
# --------------------------------------------------------------------------- #
def render_vitals(result: AttributionResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig = plt.figure(figsize=(11, 4.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.0], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    names, scores = result.vital_names, result.vital_scores
    y = np.arange(len(names))
    colors = ["#d62728" if s >= 0 else "#1f77b4" for s in scores]
    ax.barh(y, scores, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{result.class_name} — vitals (grad × value, red = toward class)", fontsize=12, loc="left")
    span = float(np.abs(scores).max()) + 1e-8
    for yi, (sc, disp, miss) in enumerate(zip(scores, result.vital_display, result.vital_missing)):
        ax.text(
            sc + math.copysign(0.02 * span, sc or 1.0), yi,
            f"{disp}{'  (missing)' if miss else ''}",
            va="center", ha="left" if sc >= 0 else "right", fontsize=9,
            color="gray" if miss else "black",
        )
    ax.margins(x=0.28)

    ax2 = fig.add_subplot(gs[1])
    mc = result.modality_contrib
    keys = ["image", "text", "vitals"]
    vals = [100.0 * mc[k] for k in keys]
    ax2.bar(keys, vals, color=["#2ca02c", "#9467bd", "#ff7f0e"])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% of |grad × input|")
    ax2.set_title("Modality share (heuristic)", fontsize=11, loc="left")
    for i, val in enumerate(vals):
        ax2.text(i, val + 1, f"{val:.0f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# legacy combined figure (kept for convenience)
# --------------------------------------------------------------------------- #
def render_attribution(result: AttributionResult, out_path: str | Path, tokens_per_row: int = 12) -> Path:
    """Single combined panel (image + text + vitals). Prefer render_attribution_split."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_token_rows = max(1, math.ceil(len(result.tokens) / tokens_per_row)) if result.tokens else 1
    fig = plt.figure(figsize=(16, 9 + 0.25 * n_token_rows), constrained_layout=True)
    outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3.0, 0.7 + 0.35 * n_token_rows, 2.2])
    fig.suptitle(_header(result), fontsize=15, fontweight="bold")

    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.05)
    total = sum(v.contribution for v in result.views) + 1e-8
    for col in range(4):
        ax = fig.add_subplot(inner[col])
        ax.set_xticks([]); ax.set_yticks([])
        if col >= len(result.views):
            ax.axis("off"); continue
        v = result.views[col]
        ax.imshow(v.image, cmap="gray", vmin=0, vmax=1)
        if v.encoded:
            ax.imshow(v.cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
            ax.set_title(f"{VIEW_NAMES.get(v.view_position, '?')} ({100*v.contribution/total:.0f}%)", fontsize=10)
        else:
            ax.set_title(f"{VIEW_NAMES.get(v.view_position, '?')} (not encoded)", fontsize=10)

    ax = fig.add_subplot(outer[1])
    ax.set_title("Clinical indication — grad × embedding (red = toward class)", fontsize=11, loc="left")
    ax.axis("off")
    if result.tokens:
        vmax = float(np.abs(result.token_scores).max()) + 1e-8
        cmap = plt.get_cmap("coolwarm")
        for i, (tok, sc) in enumerate(zip(result.tokens, result.token_scores)):
            row, col = divmod(i, tokens_per_row)
            color = cmap(0.5 + 0.5 * sc / vmax)
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            ax.text(col / tokens_per_row, 1.0 - (row + 0.5) / n_token_rows, tok.replace("##", ""),
                    fontsize=11, va="center", ha="left", color="white" if lum < 0.5 else "black",
                    bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"), transform=ax.transAxes)

    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], width_ratios=[2.0, 1.0], wspace=0.25)
    ax = fig.add_subplot(inner[0])
    scores = result.vital_scores
    y = np.arange(len(result.vital_names))
    ax.barh(y, scores, color=["#d62728" if s >= 0 else "#1f77b4" for s in scores])
    ax.set_yticks(y); ax.set_yticklabels(result.vital_names); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8); ax.margins(x=0.25)
    ax.set_title("Vitals — grad × value", fontsize=11, loc="left")
    ax2 = fig.add_subplot(inner[1])
    vals = [100.0 * result.modality_contrib[k] for k in ("image", "text", "vitals")]
    ax2.bar(["image", "text", "vitals"], vals, color=["#2ca02c", "#9467bd", "#ff7f0e"])
    ax2.set_ylim(0, 100); ax2.set_title("Modality share (heuristic)", fontsize=11, loc="left")

    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
