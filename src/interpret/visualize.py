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
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb

from src.interpret.attribution import VIEW_NAMES, AttributionResult


# --------------------------------------------------------------------------- #
# house style: one font scale + one muted palette for every panel
# --------------------------------------------------------------------------- #
# Muted, consistent palette. Hue *meanings* are unchanged from the original
# (red = toward class, blue = away, green = image, purple = text, orange =
# vitals) -- only the saturation is dialled down so nothing shouts.
C_POS = "#b5524b"      # toward class / positive          (muted red)
C_NEG = "#4d7ea8"      # away from class / negative        (muted slate blue)
C_IMAGE = "#6f9e6f"    # image modality                    (muted green)
C_TEXT = "#8d7aa8"     # text modality                     (muted purple)
C_VITALS = "#cc8a52"   # vitals modality                   (muted ochre)
C_TARGET = "#d98a3d"   # the targeted class                (muted orange)
C_MUTED = "#a9bcce"    # neutral / below-threshold bar
C_ZERO = "#888888"     # zero / threshold reference lines
C_INK = "#222222"      # primary text (titles)
C_FAINT = "0.45"       # captions / secondary annotation
# Grad-CAM overlay style. "turbo" (Google's perceptual fix for "jet") gives an
# unmistakable cold->hot ramp so salient regions read at a glance. Swap to
# "viridis"/"magma" here if colour-blind-safety matters more than hot/cold punch.
# The key change vs a flat overlay is the *per-pixel* alpha below: low-relevance
# pixels (incl. background) stay transparent so the CXR shows through, instead of
# the whole frame being washed in the colormap's low colour (the old viridis-purple).
GRADCAM_CMAP = "turbo"
_CAM_MAX_ALPHA = 0.78   # opacity at the hottest pixel
_CAM_ALPHA_GAMMA = 0.7  # <1 lifts mid-relevance so warm regions ink in; near-zero stays clear


def _overlay_cam(ax, cam, cmap_name: str = GRADCAM_CMAP):
    """Overlay a [0,1] Grad-CAM with per-pixel alpha proportional to relevance.

    A constant-alpha overlay tints every pixel — including empty background — with
    the colormap, so you cannot tell where the model actually looked or whether it
    leans on background. Here alpha = (relevance**gamma) * max_alpha, so background
    and other low-relevance pixels fall away to transparent and only salient regions
    are inked. Returns a ScalarMappable so the caller can draw a relevance colorbar.
    """
    cam = np.clip(np.asarray(cam, dtype=float), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(cam)
    rgba[..., 3] = (cam ** _CAM_ALPHA_GAMMA) * _CAM_MAX_ALPHA
    ax.imshow(rgba, interpolation="bilinear")
    return ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlecolor": C_INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 8.0,
    "axes.labelsize": 9.5,
    "axes.labelcolor": "0.2",
    "axes.edgecolor": "0.7",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "0.35",
    "ytick.color": "0.35",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


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
        render_class_distribution(result.class_names, result.all_probs, result.true_labels,
                                  result.class_name, out_dir / "class_distribution.png",
                                  logits=result.all_logits),
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
    fig.suptitle(header, fontsize=13, fontweight="semibold", color=C_INK)

    # image panels read cleaner with no frame at all (ticks already blank)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    if not views:
        axes[0, 0].text(0.5, 0.5, empty_msg, ha="center", va="center", color=C_FAINT)

    for row, v in enumerate(views):
        name = VIEW_NAMES.get(v.view_position, "?")

        if v.channels is not None and n_chan:
            for c in range(n_chan):
                axes[row, c].imshow(v.channels[..., c], cmap="gray", vmin=0, vmax=1)
                axes[row, c].set_title(f"{name} — {chan_names[c]}", fontsize=10)
        else:
            axes[row, 0].imshow(v.image, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"{name} — original", fontsize=10)

        ax_h = axes[row, ncols - 1]
        ax_h.imshow(v.image, cmap="gray", vmin=0, vmax=1)
        if v.encoded:
            sm = _overlay_cam(ax_h, v.cam)
            share = 100.0 * v.contribution / total
            ax_h.set_title(f"{name} — Grad-CAM ({share:.0f}% of image signal)", fontsize=10)
            cbar = fig.colorbar(sm, ax=ax_h, fraction=0.046, pad=0.04)
            cbar.set_label("relevance", fontsize=8)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=7, length=0)
        else:
            ax_h.set_title(f"{name} — not encoded (view=0)", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
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


# highlighter palette (RRR "20 Newsgroups" style): black text on a coloured
# highlight whose intensity scales with |score|. Salient words pop; the rest
# stay near-white so the indication still reads as a running sentence.
_HL_RED = np.array(to_rgb(C_POS))    # toward class (muted red, matches the bars)
_HL_BLUE = np.array(to_rgb(C_NEG))   # away from class (muted slate, matches the bars)
_HL_CMAP = LinearSegmentedColormap.from_list(
    "rrr_highlight", [tuple(_HL_BLUE), (1.0, 1.0, 1.0), tuple(_HL_RED)]
)


def _highlight_rgb(score: float, vmax: float, gamma: float = 0.8):
    """Blend white -> red/blue by |score|/vmax (gamma-shaped so weak words fade out)."""
    m = (min(1.0, abs(float(score)) / vmax)) ** gamma
    base = _HL_RED if score >= 0 else _HL_BLUE
    return tuple(1.0 - m * (1.0 - base))


def _render_token_chips(raw_tokens, scores, title: str, out_path: str | Path,
                        tokens_per_row: int = 10, empty_msg: str = "(no text)") -> Path:
    """Inline highlighted running text (RRR-style), packed left-to-right then wrapped.

    Black text on a per-word highlight (white -> red toward class / blue away),
    intensity scaled by |grad x embedding|. Factored out of ``render_text`` so the
    prior clinical / prior report panels reuse the same mapping with their own title.
    """
    out_path = Path(out_path)
    tokens = [t.replace("##", "") for t in (raw_tokens or [])]
    fig_w = 1.0 * tokens_per_row + 1.0
    fontsize = 13

    if not tokens:
        fig, ax = plt.subplots(figsize=(fig_w, 1.8))
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center", color=C_FAINT, transform=ax.transAxes)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    vmax = float(np.abs(scores).max()) + 1e-8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # pass 1: measure each word's width as a fraction of the figure width
    meas = plt.figure(figsize=(fig_w, 2.0))
    meas.canvas.draw()
    renderer = meas.canvas.get_renderer()
    inv = meas.transFigure.inverted()
    box_pad = 0.32 * fontsize / 72.0 / fig_w  # tight highlight padding, both sides
    widths = []
    for tok in tokens:
        t = meas.text(0, 0, tok, fontsize=fontsize)
        bb = t.get_window_extent(renderer)
        widths.append(inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0] + box_pad)
        t.remove()
    plt.close(meas)

    # pack words left-to-right, wrapping when the current row is full (running text)
    x_lo, x_hi, gap = 0.02, 0.98, 0.006
    placed, x, row = [], x_lo, 0
    for tok, sc, w in zip(tokens, scores, widths):
        if x + w > x_hi and x > x_lo:
            row += 1
            x = x_lo
        placed.append((tok, sc, row, x))  # left edge, ha="left"
        x += w + gap
    n_rows = row + 1

    # pass 2: lay out for real, figure height scaled to the packed rows
    row_h, title_h, cbar_h = 0.40, 0.55, 0.62
    H = title_h + n_rows * row_h + cbar_h
    fig = plt.figure(figsize=(fig_w, H))
    fig.text(0.02, 1.0 - 0.32 / H, title, fontsize=12, ha="left", va="center",
             fontweight="semibold", color=C_INK)

    band_top = (cbar_h + n_rows * row_h) / H
    for tok, sc, rr, xl in placed:
        y = band_top - (rr + 0.5) * row_h / H
        fig.text(
            xl, y, tok, fontsize=fontsize, ha="left", va="center", color=C_INK,
            bbox=dict(boxstyle="square,pad=0.22", fc=_highlight_rgb(sc, vmax), ec="none"),
        )

    cax = fig.add_axes([0.2, 0.45 * cbar_h / H, 0.6, 0.16 * cbar_h / H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=_HL_CMAP), cax=cax, orientation="horizontal")
    cb.set_label("← pushes away from class            pushes toward class →", fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.savefig(out_path)
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
    colors = [C_POS if s >= 0 else C_NEG for s in scores]
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
            lbl.set_color(C_FAINT)

    ax.invert_yaxis()
    ax.axvline(0, color=C_ZERO, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("signed grad × value")
    ax.margins(x=0.1)

    fig.tight_layout()
    fig.savefig(out_path)
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
    ax.bar(keys, vals, color=[C_IMAGE, C_TEXT, C_VITALS])
    ax.set_ylim(0, 110)  # headroom so a ~99% bar (and its label) don't hit the ceiling
    ax.set_ylabel("% of |grad × input|")
    ax.set_title(f"{result.class_name} — modality share (heuristic)")
    for i, val in enumerate(vals):
        ax.text(i, val + 2, f"{val:.0f}%", ha="center", fontsize=11, fontweight="semibold", color=C_INK)

    fig.text(0.5, 0.01,
             "Rough share of total |grad × input| at each modality's native input — comparable-ish, not exact.",
             ha="center", fontsize=8, color=C_FAINT)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# class distribution: per-class sigmoid prob for this one sample
# --------------------------------------------------------------------------- #
def render_class_distribution(class_names, probs, true_labels, target_class,
                              out_path: str | Path, threshold: float = 0.5,
                              logits=None) -> Path:
    """All 26 class probabilities for a single study, sorted high->low.

    This is the model's full last-layer read-out for the sample (multi-label
    sigmoid, so the bars are independent per-class probabilities, NOT a softmax
    that sums to 1). The *targeted* class (the one this folder's Grad-CAM explains)
    is highlighted; ground-truth-positive classes are starred so you can see at a
    glance whether the high bars line up with the truth.
    """
    out_path = Path(out_path)
    names = list(class_names)
    probs = np.asarray(probs, dtype=float)
    logits = None if logits is None else np.asarray(logits, dtype=float)
    truth = set(true_labels or [])

    order = np.argsort(-probs)  # most probable at the top
    names_s = [names[i] for i in order]
    probs_s = probs[order]
    logits_s = None if logits is None else logits[order]

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.30 * len(names))))
    y = np.arange(len(names_s))

    colors, edgecolors, lws = [], [], []
    for nm, p in zip(names_s, probs_s):
        if nm == target_class:
            colors.append(C_TARGET)                        # targeted class
        elif p >= threshold:
            colors.append(C_POS)                           # above threshold
        else:
            colors.append(C_MUTED)                         # below threshold
        if nm in truth:
            edgecolors.append(C_INK); lws.append(1.4)      # ground truth: dark outline
        else:
            edgecolors.append("none"); lws.append(0.0)

    ax.barh(y, probs_s, color=colors, edgecolor=edgecolors, linewidth=lws)
    ax.set_yticks(y)
    # arrow marks the targeted class, star marks ground-truth-positive classes
    ticklabels = [f"{'» ' if nm == target_class else ''}{'★ ' if nm in truth else ''}{nm}"
                  for nm in names_s]
    labels = ax.set_yticklabels(ticklabels)
    for lbl, nm in zip(labels, names_s):
        lbl.set_fontsize(8)
        if nm in truth:
            lbl.set_fontweight("bold")
        if nm == target_class:
            lbl.set_color(C_TARGET)

    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.axvline(threshold, color=C_ZERO, lw=1.0, ls="--")
    ax.text(threshold, -0.8, f" threshold={threshold:g}", color=C_FAINT, fontsize=8, va="bottom")

    # value at each bar end (prob, and raw logit if provided)
    for yi, p in zip(y, probs_s):
        txt = f"{p:.2f}" + (f"  (z={logits_s[yi]:+.1f})" if logits_s is not None else "")
        ax.text(min(p + 0.012, 0.995), yi, txt, va="center", ha="left", fontsize=7, color="0.35")

    ax.set_xlabel("per-class probability  =  sigmoid(logit)")
    ax.set_title(f"{target_class} — class distribution for this study (» targeted, ★ ground truth)")
    fig.tight_layout()
    fig.savefig(out_path)
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
    fig.suptitle(_header(result), fontsize=15, fontweight="semibold", color=C_INK)

    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.05)
    total = sum(v.contribution for v in result.views) + 1e-8
    for col in range(4):
        ax = fig.add_subplot(inner[col])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if col >= len(result.views):
            ax.axis("off"); continue
        v = result.views[col]
        ax.imshow(v.image, cmap="gray", vmin=0, vmax=1)
        if v.encoded:
            _overlay_cam(ax, v.cam)
            ax.set_title(f"{VIEW_NAMES.get(v.view_position, '?')} ({100*v.contribution/total:.0f}%)", fontsize=10)
        else:
            ax.set_title(f"{VIEW_NAMES.get(v.view_position, '?')} (not encoded)", fontsize=10)

    ax = fig.add_subplot(outer[1])
    ax.set_title("Clinical indication — grad × embedding (red = toward class)", fontsize=11)
    ax.axis("off")
    if result.tokens:
        vmax = float(np.abs(result.token_scores).max()) + 1e-8
        for i, (tok, sc) in enumerate(zip(result.tokens, result.token_scores)):
            row, col = divmod(i, tokens_per_row)
            ax.text(col / tokens_per_row, 1.0 - (row + 0.5) / n_token_rows, tok.replace("##", ""),
                    fontsize=11, va="center", ha="left", color=C_INK,
                    bbox=dict(boxstyle="square,pad=0.22", fc=_highlight_rgb(sc, vmax), ec="none"),
                    transform=ax.transAxes)

    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], width_ratios=[2.0, 1.0], wspace=0.25)
    ax = fig.add_subplot(inner[0])
    scores = result.vital_scores
    y = np.arange(len(result.vital_names))
    ax.barh(y, scores, color=[C_POS if s >= 0 else C_NEG for s in scores])
    ax.set_yticks(y); ax.set_yticklabels(result.vital_names); ax.invert_yaxis()
    ax.axvline(0, color=C_ZERO, lw=0.8); ax.margins(x=0.25)
    ax.set_title("Vitals — grad × value", fontsize=11)
    ax2 = fig.add_subplot(inner[1])
    vals = [100.0 * result.modality_contrib[k] for k in ("image", "text", "vitals")]
    ax2.bar(["image", "text", "vitals"], vals, color=[C_IMAGE, C_TEXT, C_VITALS])
    ax2.set_ylim(0, 100); ax2.set_title("Modality share (heuristic)", fontsize=11)

    fig.savefig(out_path)
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
        render_class_distribution(result.class_names, result.all_probs, result.true_labels,
                                  result.class_name, out_dir / "class_distribution.png",
                                  logits=result.all_logits),
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
    colors = [C_POS if s >= 0 else C_NEG for s in scores]
    ax.barh(y, scores, color=colors)
    ax.set_yticks(y)
    # bold the labels that were actually positive in the prior study.
    labels = ax.set_yticklabels(names)
    for lbl, on in zip(labels, values):
        lbl.set_fontsize(8)
        if on >= 0.5:
            lbl.set_fontweight("bold")
            lbl.set_color(C_POS)
    ax.invert_yaxis()
    ax.axvline(0, color=C_ZERO, lw=0.8)
    ax.margins(x=0.1)
    title = f"{result.class_name} — prior label token (grad × value)"
    if not result.has_prior:
        title += "  [no prior — masked]"
    ax.set_title(title)
    ax.set_xlabel("signed grad × value   (red bars push toward class)")
    fig.text(0.5, 0.005, "Bold/red ticks = findings positive in the PRIOR study.",
             ha="center", fontsize=8, color=C_FAINT)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_time_delta(result, out_path: str | Path) -> Path:
    """The time-delta bucket token: its signed grad × embedding and magnitude."""
    from src.interpret.prior_attribution import DELTA_BUCKET_NAMES

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bucket = DELTA_BUCKET_NAMES.get(result.delta_bucket, "?")
    color = C_POS if result.delta_score >= 0 else C_NEG
    ax.bar(["time-delta token"], [result.delta_score], color=color, width=0.5)
    ax.axhline(0, color=C_ZERO, lw=0.8)
    ax.set_ylabel("signed grad × embedding")
    sub = (f"bucket {result.delta_bucket} ({bucket}) | Δt={result.days_since_prior:.0f}d | "
           f"|grad×emb|={result.delta_mag:.3g}")
    if not result.has_prior:
        sub = "no prior study — delta bucket 0 (masked)"
    ax.set_title(f"{result.class_name} — time gap to prior\n{sub}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
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
    # current group = the three base modality hues; prior group = the same hues
    # desaturated, so the two branches read as one family split into "now" vs "before".
    colors = [C_IMAGE, C_TEXT, C_VITALS,
              "#9bb79b", "#bbb1cc", "#e3c4a3", "#d8b48a", "#b9a39c", "#cccccc"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, max(110, max(vals) + 10))
    ax.set_ylabel("% of |grad × input|")
    ax.set_title(f"{result.class_name} — current vs prior contribution (heuristic)")
    cur_share = sum(mc.get(k, 0.0) for k in ("cur_image", "cur_clin", "cur_vitals"))
    ax.axvline(2.5, color=C_ZERO, ls="--", lw=1.0)
    ax.text(1.0, ax.get_ylim()[1] * 0.93, f"current {100*cur_share:.0f}%", ha="center", fontsize=10, color=C_FAINT)
    ax.text(5.5, ax.get_ylim()[1] * 0.93, f"prior {100*(1-cur_share):.0f}%", ha="center", fontsize=10, color=C_FAINT)
    for i, val in enumerate(vals):
        if val > 0.5:
            ax.text(i, val + 1.5, f"{val:.0f}", ha="center", fontsize=9, color=C_INK)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    fig.text(0.5, 0.005,
             "Rough share of total |grad × input| at each token group's native input — comparable-ish, not exact.",
             ha="center", fontsize=8, color=C_FAINT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
