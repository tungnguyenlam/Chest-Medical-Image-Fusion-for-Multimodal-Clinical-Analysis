"""Eyeball the body-vs-background separation before committing to any penalty.

Samples a handful of studies from a split and, for each image, saves a 3-panel
PNG: the raw X-ray, the detected patient silhouette, and the *confident
background* (the region a background-attention penalty would punish). If these
overlays look sane -- corners / text / outside-patient shaded, lungs + chest wall
+ mediastinum left alone -- the separation is trustworthy and we can wire the
penalty in. If they look dumb, we caught it here for free.

Example:
    python -m scripts.visualize_body_mask \
        --config training/camchex_v2nano_vitals/config.yaml \
        --split val --num 16 --out output/body_mask_check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.body_mask import BodyMaskConfig, body_and_background
from src.dataloader.utils import _safe_decode_jpeg, resolve_preferred_image_path
from training.common import load_config, read_dataframe, resolve_path

SPLIT_KEY = {"val": "devel_df_path", "test": "pred_df_path", "train": "train_df_path"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize the CXR body/background separation.")
    p.add_argument("--config", required=True, help="A training config.yaml (for the split paths).")
    p.add_argument("--split", choices=["val", "test", "train"], default="val")
    p.add_argument("--num", type=int, default=16, help="Number of images to render.")
    p.add_argument("--out", default="output/body_mask_check", help="Output directory for PNGs.")
    p.add_argument("--seed", type=int, default=0)
    # Mask knobs (defaults match BodyMaskConfig); expose the ones that matter most.
    p.add_argument("--band-frac", type=float, default=None, help="Outer-edge band where background is allowed.")
    p.add_argument("--dark-frac", type=float, default=None, help="<= this*255 = collimation black.")
    p.add_argument("--bright-frac", type=float, default=None, help=">= this*255 = direct-exposure white.")
    p.add_argument("--feather-frac", type=float, default=None, help="Background-edge feather sigma.")
    return p.parse_args()


def _gray_uint8(path: str) -> np.ndarray | None:
    rgb = _safe_decode_jpeg(resolve_preferred_image_path(path))
    if rgb is None:
        return None
    if rgb.ndim == 3:
        return rgb[..., 0]  # decoded as gray-replicated RGB; any channel works
    return rgb


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    key = SPLIT_KEY[args.split]
    data = cfg.get("data", {})
    # df paths live under data.datamodule_cfg; fall back to data.* and top-level.
    df_path = data.get("datamodule_cfg", {}).get(key) or data.get(key) or cfg.get(key)
    if df_path is None:
        raise SystemExit(f"Could not find {key} in config {args.config} (looked under data.datamodule_cfg)")
    df = read_dataframe(resolve_path(df_path))

    mask_cfg = BodyMaskConfig()
    if args.band_frac is not None:
        mask_cfg.band_frac = args.band_frac
    if args.dark_frac is not None:
        mask_cfg.dark_frac = args.dark_frac
    if args.bright_frac is not None:
        mask_cfg.bright_frac = args.bright_frac
    if args.feather_frac is not None:
        mask_cfg.feather_frac = args.feather_frac

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    paths = df["path"].astype(str).tolist()
    pick = rng.choice(len(paths), size=min(args.num, len(paths)), replace=False)

    saved = 0
    for i in pick:
        path = paths[int(i)]
        gray = _gray_uint8(path)
        if gray is None:
            print(f"  skip (unreadable): {path}")
            continue
        body, bg = body_and_background(gray, mask_cfg)
        bg_frac = float(bg.mean())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("raw CXR")
        axes[1].imshow(gray, cmap="gray")
        axes[1].imshow(body, cmap="Greens", alpha=0.35)
        axes[1].set_title("kept (anatomy + body edge)")
        axes[2].imshow(gray, cmap="gray")
        axes[2].imshow(bg, cmap="Reds", alpha=0.5)
        axes[2].set_title(f"confident background ({bg_frac:.0%} punished)")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"mask_{saved:03d}.png", dpi=90)
        plt.close(fig)
        saved += 1
        print(f"  [{saved:2d}] bg={bg_frac:.0%}  {path}")

    print(f"\nSaved {saved} overlays to {out_dir}")
    print("Check: are corners/text/outside-patient red, and lungs+chest-wall+mediastinum NOT red?")


if __name__ == "__main__":
    main()
