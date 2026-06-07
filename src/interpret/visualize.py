"""Render an AttributionResult as separate, inspect-by-hand images.

Four files per class (each gets its own room instead of one cramped figure):

    <class>/image.png     one row per CXR view: original | Grad-CAM overlay
    <class>/text.png      indication tokens coloured by grad x embedding
    <class>/vitals.png    signed grad x value per vital (name + value on each label)
    <class>/modality.png  % of signal from image / text / vitals (heuristic)

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
        render_modality(result, out_dir / "modality.png"),
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
    return _render_image_rows(result.views or [], _header(result), out_path)


def _render_image_rows(views: list, header: str, out_path: str | Path,
                       empty_msg: str = "(no images)") -> Path:
    """One row per view: original (or per-channel planes) | Grad-CAM overlay.

    Factored out of ``render_images`` so the prior branch can render its own
    image panel with a different header without duplicating the layout logic.
    """
    out_path = Path(out_path)
    n = max(1, len(views))
    total = sum(v.contribution for v in views) + 1e-8

    # When the model was fed deterministic multi-channel inputs (e.g.
    # raw_clahe_histeq), show every input plane, then the Grad-CAM overlay. The
    # heatmap is per-view (identical across planes), so it is drawn once over the
    # raw plane. Legacy RGB inputs keep the 2-column original | overlay layout.
    chan_names = next((v.channel_names for v in views if v.channels is not None), None)
    n_chan = len(chan_names) if chan_names else 0
    ncols = n_chan + 1 if n_chan else 2

    fig, axes = plt.subplots(n, ncols, figsize=(4.2 * ncols, 4.2 * n), squeeze=False)
    fig.suptitle(header, fontsize=13, fontweight="bold")

    if not views:
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, empty_msg, ha="center", va="center")

    for row, v in enumerate(views):
        name = VIEW_NAMES.get(v.view_position, "?")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

        if v.channels is not None and n_chan:
            for c in range(n_chan):
                axes[row, c].imshow(v.channels[..., c], cmap="gray", vmin=0, vmax=1)
                axes[row, c].set_title(f"{name} — {chan_names[c]}", fontsize=11)
        else:
            axes[row, 0].imshow(v.image, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"{name} — original", fontsize=11)

        ax_h = axes[row, ncols - 1]
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
    """Token chips packed left-to-right like running text (width-aware), then wrapped.

    ``tokens_per_row`` only sets the figure width (how many average tokens fit per
    line); chips themselves are placed by their measured width so short words sit
    close together instead of each occupying an equal-width column.
    """
    title = f"{result.class_name} — clinical indication (grad × embedding)"
    return _render_token_chips(result.tokens, result.token_scores, title, out_path,
                               tokens_per_row=tokens_per_row, empty_msg="(no text)")


def _render_token_chips(raw_tokens, scores, title: str, out_path: str | Path,
                        tokens_per_row: int = 10, empty_msg: str = "(no text)") -> Path:
    """Token chips coloured by signed score, packed left-to-right then wrapped.

    Factored out of ``render_text`` so the prior clinical / prior report panels
    reuse the exact same packing + colour mapping with their own title.
    """
    out_path = Path(out_path)
    tokens = [t.replace("##", "") for t in (raw_tokens or [])]
    fig_w = 1.0 * tokens_per_row + 1.0
    fontsize = 13

    if not tokens:
        fig, ax = plt.subplots(figsize=(fig_w, 1.8))
        ax.axis("off")
        ax.set_title(title, fontsize=12, loc="left")
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path

    vmax = float(np.abs(scores).max()) + 1e-8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    # pass 1: measure each chip's width as a fraction of the figure width
    meas = plt.figure(figsize=(fig_w, 2.0))
    meas.canvas.draw()
    renderer = meas.canvas.get_renderer()
    inv = meas.transFigure.inverted()
    box_pad = 0.6 * fontsize / 72.0 / fig_w  # round-box padding, both sides
    widths = []
    for tok in tokens:
        t = meas.text(0, 0, tok, fontsize=fontsize)
        bb = t.get_window_extent(renderer)
        widths.append(inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0] + box_pad)
        t.remove()
    plt.close(meas)

    # pack chips left-to-right, wrapping when the current row is full
    x_lo, x_hi, gap = 0.02, 0.98, 0.012
    placed, x, row = [], x_lo, 0
    for tok, sc, w in zip(tokens, scores, widths):
        if x + w > x_hi and x > x_lo:
            row += 1
            x = x_lo
        placed.append((tok, sc, row, x + w / 2.0))
        x += w + gap
    n_rows = row + 1

    # pass 2: lay out for real, figure height scaled to the packed rows
    row_h, title_h, cbar_h = 0.42, 0.55, 0.62
    H = title_h + n_rows * row_h + cbar_h
    fig = plt.figure(figsize=(fig_w, H))
    fig.text(0.02, 1.0 - 0.32 / H, title, fontsize=12, ha="left", va="center", fontweight="bold")

    band_top = (cbar_h + n_rows * row_h) / H
    for tok, sc, rr, xc in placed:
        y = band_top - (rr + 0.5) * row_h / H
        color = cmap(norm(sc))
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        fig.text(
            xc, y, tok, fontsize=fontsize, ha="center", va="center",
            color="white" if lum < 0.5 else "black",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="0.6", lw=0.5),
        )

    cax = fig.add_axes([0.2, 0.45 * cbar_h / H, 0.6, 0.16 * cbar_h / H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("← pushes away from class            pushes toward class →", fontsize=9)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# vitals: signed grad x value + modality share
# --------------------------------------------------------------------------- #
def render_vitals(result: AttributionResult, out_path: str | Path) -> Path:
    title = f"{result.class_name} — vitals (grad × value, red = toward class)"
    return _render_vitals_barh(
        result.vital_names, result.vital_scores, result.vital_display,
        result.vital_missing, title, out_path,
    )


def _render_vitals_barh(names, scores, displays, missing, title: str, out_path: str | Path) -> Path:
    """Signed horizontal bar of grad × value per vital, name + value on each tick.

    Factored out of ``render_vitals`` so the prior-branch vitals panel reuses it.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(9, 4.6))

    y = np.arange(len(names))
    colors = ["#d62728" if s >= 0 else "#1f77b4" for s in scores]
    ax.barh(y, scores, color=colors)

    # each tick label shows the vital name plus its actual (de-normalized) value
    ax.set_yticks(y)
    labels = ax.set_yticklabels(
        [f"{n}\n{d}{'  (missing)' if m else ''}"
         for n, d, m in zip(names, displays, missing)]
    )
    for lbl, miss in zip(labels, missing):
        lbl.set_fontsize(9)
        if miss:
            lbl.set_color("gray")

    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("signed grad × value")
    ax.margins(x=0.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# modality share: where did the signal come from (own plot, with headroom)
# --------------------------------------------------------------------------- #
def render_modality(result: AttributionResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))

    mc = result.modality_contrib
    keys = ["image", "text", "vitals"]
    vals = [100.0 * mc[k] for k in keys]
    ax.bar(keys, vals, color=["#2ca02c", "#9467bd", "#ff7f0e"])
    ax.set_ylim(0, 110)  # headroom so a ~99% bar (and its label) don't hit the ceiling
    ax.set_ylabel("% of |grad × input|")
    ax.set_title(f"{result.class_name} — modality share (heuristic)", fontsize=12, loc="left")
    for i, val in enumerate(vals):
        ax.text(i, val + 2, f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")

    fig.text(0.5, 0.01,
             "Rough share of total |grad × input| at each modality's native input — comparable-ish, not exact.",
             ha="center", fontsize=8, color="0.45")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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


# --------------------------------------------------------------------------- #
# prior-aware models: current panels (reused) + prior-branch panels
# --------------------------------------------------------------------------- #
def _prior_header(result) -> str:
    from src.interpret.prior_attribution import DELTA_BUCKET_NAMES

    label = "n/a" if result.label is None else str(int(result.label))
    head = (
        f"{result.class_name}   |   study={result.study_id}   |   "
        f"p={result.prob:.3f}   |   logit={result.logit:+.2f}   |   true={label}"
    )
    if result.has_prior:
        bucket = DELTA_BUCKET_NAMES.get(result.delta_bucket, "?")
        head += f"\nprior: yes  |  Δt={result.days_since_prior:.0f}d ({bucket})"
    else:
        head += "\nprior: none (prior tokens masked out)"
    others = [c for c in result.true_labels if c != result.class_name]
    if others:
        head += f"\nalso positive: {', '.join(others)}"
    return head


def render_prior_attribution_split(result, out_dir: str | Path, tokens_per_row: int = 10) -> list[Path]:
    """Write the full per-class panel set for a prior-aware model.

    Current-branch panels reuse the single-study renderers; prior-branch panels
    (image / clinical / report / label / time-delta) get their own files. ``modality``
    shows the current-vs-prior contribution breakdown across all token groups.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = _prior_header(result)
    paths = [
        _render_image_rows(result.cur_views, header, out_dir / "image.png"),
        _render_image_rows(result.prv_views, header, out_dir / "prior_image.png",
                           empty_msg="(no prior study)"),
        render_prior_modality(result, out_dir / "modality.png"),
        render_time_delta(result, out_dir / "time_delta.png"),
        render_prior_label(result, out_dir / "prior_label.png"),
    ]
    # text streams: one PNG per stream, named by its key (current_*/prior_*).
    for st in result.cur_texts + result.prv_texts:
        name = "text.png" if st.key == "cur_clin" else f"{st.key}.png"
        paths.append(_render_token_chips(st.tokens, st.scores, st.title, out_dir / name,
                                         tokens_per_row=tokens_per_row, empty_msg="(no text)"))
    # vitals (Nano variants only; the base model uses obs text streams instead).
    if result.has_vitals:
        paths.append(_render_vitals_barh(
            result.cur_vital_names, result.cur_vital_scores, result.cur_vital_display,
            result.cur_vital_missing, f"{result.class_name} — current vitals (grad × value)",
            out_dir / "vitals.png"))
        paths.append(_render_vitals_barh(
            result.cur_vital_names, result.prv_vital_scores, result.prv_vital_display,
            result.prv_vital_missing, f"{result.class_name} — prior vitals (grad × value)",
            out_dir / "prior_vitals.png"))
    return paths


def render_prior_label(result, out_path: str | Path) -> Path:
    """Per-class grad × value over the prior study's 26-dim CheXpert label vector.

    This answers 'which findings in the prior study drove the current prediction?' —
    a signal unique to the prior-aware models (the Linear(26→768) prior-label token)."""
    out_path = Path(out_path)
    names = list(result.class_names) or [str(i) for i in range(len(result.prior_label_scores))]
    scores = np.asarray(result.prior_label_scores, dtype=float)
    values = np.asarray(result.prior_label_values, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.32 * len(names))))
    y = np.arange(len(names))
    colors = ["#d62728" if s >= 0 else "#1f77b4" for s in scores]
    ax.barh(y, scores, color=colors)
    ax.set_yticks(y)
    # bold the labels that were actually positive in the prior study.
    labels = ax.set_yticklabels(names)
    for lbl, on in zip(labels, values):
        lbl.set_fontsize(8)
        if on >= 0.5:
            lbl.set_fontweight("bold")
            lbl.set_color("#b22222")
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.margins(x=0.1)
    title = f"{result.class_name} — prior label token (grad × value)"
    if not result.has_prior:
        title += "  [no prior — masked]"
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("signed grad × value   (red bars push toward class)")
    fig.text(0.5, 0.005, "Bold/red ticks = findings positive in the PRIOR study.",
             ha="center", fontsize=8, color="0.45")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_time_delta(result, out_path: str | Path) -> Path:
    """The time-delta bucket token: its signed grad × embedding and magnitude."""
    from src.interpret.prior_attribution import DELTA_BUCKET_NAMES

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bucket = DELTA_BUCKET_NAMES.get(result.delta_bucket, "?")
    color = "#d62728" if result.delta_score >= 0 else "#1f77b4"
    ax.bar(["time-delta token"], [result.delta_score], color=color, width=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("signed grad × embedding")
    sub = (f"bucket {result.delta_bucket} ({bucket}) | Δt={result.days_since_prior:.0f}d | "
           f"|grad×emb|={result.delta_mag:.3g}")
    if not result.has_prior:
        sub = "no prior study — delta bucket 0 (masked)"
    ax.set_title(f"{result.class_name} — time gap to prior\n{sub}", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_prior_modality(result, out_path: str | Path) -> Path:
    """Current-vs-prior contribution breakdown across every token group (heuristic)."""
    out_path = Path(out_path)
    mc = result.modality_contrib
    keys = ["cur_image", "cur_clin", "cur_vitals",
            "prv_image", "prv_clin", "prv_report", "prv_vitals", "prv_label", "time_delta"]
    labels = ["cur img", "cur clin", "cur vit/obs",
              "prv img", "prv clin", "prv report", "prv vit/obs", "prv label", "Δt"]
    vals = [100.0 * mc.get(k, 0.0) for k in keys]
    # current group greens/purples, prior group warm — visually split the two branches.
    colors = ["#2ca02c", "#9467bd", "#ff7f0e",
              "#1f9e89", "#7b68ee", "#ff9896", "#ffbb78", "#8c564b", "#c7c7c7"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, max(110, max(vals) + 10))
    ax.set_ylabel("% of |grad × input|")
    ax.set_title(f"{result.class_name} — current vs prior contribution (heuristic)", fontsize=12, loc="left")
    cur_share = sum(mc.get(k, 0.0) for k in ("cur_image", "cur_clin", "cur_vitals"))
    ax.axvline(2.5, color="0.6", ls="--", lw=1.0)
    ax.text(1.0, ax.get_ylim()[1] * 0.93, f"current {100*cur_share:.0f}%", ha="center", fontsize=10, color="0.3")
    ax.text(5.5, ax.get_ylim()[1] * 0.93, f"prior {100*(1-cur_share):.0f}%", ha="center", fontsize=10, color="0.3")
    for i, val in enumerate(vals):
        if val > 0.5:
            ax.text(i, val + 1.5, f"{val:.0f}", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    fig.text(0.5, 0.005,
             "Rough share of total |grad × input| at each token group's native input — comparable-ish, not exact.",
             ha="center", fontsize=8, color="0.45")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
