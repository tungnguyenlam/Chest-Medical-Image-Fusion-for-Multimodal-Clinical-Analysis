"""Ablation grid + Grad-CAM raw-asset driver for one prior-aware checkpoint.

This merges the old ``eval_no_cache.py`` / ``eval_no_cache_gray.py`` wrappers into a
single self-contained driver. Given ONE checkpoint (``best.pt``; ``last.pt`` is
ignored), it runs a *deterministic* eval-time ablation grid and a best-per-class
Grad-CAM pass, writing everything into the checkpoint's own run directory under
``<run>/ablation/``. The run is deterministic, so re-running overwrites cleanly.

What it produces
----------------
1. Ablation grid (all on the SAME checkpoint, eval-time input ablations):

   * ``full``        — native 3-channel build, every modality on.
   * ``gray3``       — 1-channel grayscale replicated x3 (the old eval_no_cache_gray
                       behaviour). NB this is a train/eval *mismatch* probe on a model
                       trained on the engineered channels, not a channel-design proof.
   * ``drop_vitals`` — current + prior vitals zeroed and flagged missing.
   * ``drop_report`` — current clinical indication blanked (in-distribution placeholder).
   * ``drop_prior``  — has_prior=False; the whole prior memory branch is masked out.

   Per arm: ``predictions_<arm>.csv`` + ``metrics_<arm>.json``, a console summary, and
   a delta vs ``full``. ``ablation_summary.csv`` collects macro/head/medium/tail
   mAP+AUROC for every arm for direct report-table use.

2. Grad-CAM raw assets (full pipeline only): one best (highest-confidence true
   positive) study per class, dumped to ``ablation/gradcam/<Class>/`` as raw,
   single-channel PNGs (the 3 input planes split out individually + the Grad-CAM
   heatmap, current and prior) plus the indication / report / vitals text and a
   ``meta.json`` — the raw materials for hand-built report figures.

There is intentionally no graph-head arm: the v6-vs-v8 (no-graph-vs-graph)
comparison comes from running this script on each checkpoint separately.

Two patches are kept from the old wrappers:

* The on-disk channel cache is disabled (channels decode fresh from JPEG).
* CXR-BERT is loaded with ``low_cpu_mem_usage=False`` so its tied MLM head does not
  stay on the meta device (``model.to('mps')`` otherwise crashes). A leftover-meta
  materialization guard is kept as belt-and-suspenders.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.CaMCheXVitalsDataset import DEFAULT_VITAL_STATS, VITAL_FIELDS
from src.dataloader.PriorAwareDataset import PriorAwareDataset
from src.dataloader.utils import get_transforms
from src.interpret.attribution import VIEW_NAMES
from src.interpret.prior_attribution import DELTA_BUCKET_NAMES, PriorAwareAttributor
from training.common import (
    add_common_args,
    classes_from_config,
    compute_metrics,
    data_cfg_from_config,
    predict_dataframe,
    print_validation_summary,
    resolve_eval_config,
    resolve_path,
    select_device,
    set_seed,
)
from training.utils.data import (
    _blank_prior_aware_current_indication,
    _prior_aware_tokenizer,
    dataloader_args_from_config,
)

# run_prior_gradcam owns the arch->class map and the grad-enabled model/dataset
# builders + per-class study selection; reuse them so this driver stays arch-generic
# (it also runs on a v6 checkpoint) and the attribution wiring is shared.
import src.interpret.run_prior_gradcam as gc

# The src.encoder package re-exports the BioBertEncoder *class* under the same name,
# shadowing the submodule, so import the module object explicitly.
_bb = importlib.import_module("src.encoder.BioBertEncoder")

ALL_ARMS = ("full", "gray3", "drop_vitals", "drop_report", "drop_prior")


# --------------------------------------------------------------------------- #
# patches carried over from the old wrappers
# --------------------------------------------------------------------------- #
def _install_no_meta_patch() -> None:
    """Force low_cpu_mem_usage=False when loading CXR-BERT (avoid meta tensors)."""
    orig = _bb.from_pretrained_best_attention

    def _no_meta(model_cls, model_name, **kwargs):
        kwargs.setdefault("low_cpu_mem_usage", False)
        return orig(model_cls, model_name, **kwargs)

    _bb.from_pretrained_best_attention = _no_meta


def _materialize_meta(model: torch.nn.Module) -> None:
    """Replace any leftover meta param/buffer with zeros (the unused LM head)."""
    n = 0
    for mod in model.modules():
        for name, p in list(mod._parameters.items()):
            if p is not None and p.is_meta:
                mod._parameters[name] = torch.nn.Parameter(
                    torch.zeros(p.shape, dtype=p.dtype), requires_grad=False
                )
                n += 1
        for name, b in list(mod._buffers.items()):
            if b is not None and b.is_meta:
                mod._buffers[name] = torch.zeros(b.shape, dtype=b.dtype)
                n += 1
    if n:
        print(f"[ablation] materialized {n} leftover meta tensor(s) (unused LM head)", flush=True)


# --------------------------------------------------------------------------- #
# ablation dataset wrapper
# --------------------------------------------------------------------------- #
class _AblatedDataset(Dataset):
    """Apply an input ablation to each sample of a base PriorAwareDataset.

    ``drop_report`` is handled upstream by blanking the dataset's ``clin_text``
    column (so the in-distribution placeholder is tokenized), not here. This
    wrapper handles the per-sample tensor edits: ``drop_vitals`` (zero + flag both
    branches' vitals as missing) and ``drop_prior`` (mirror the dataset's own
    no-prior emission so the model masks the entire prior memory branch).
    """

    def __init__(self, base: PriorAwareDataset, *, drop_vitals: bool = False, drop_prior: bool = False):
        self.base = base
        self.drop_vitals = drop_vitals
        self.drop_prior = drop_prior

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        data, label = self.base[index]
        if self.drop_vitals:
            data["vital_values"] = np.zeros_like(data["vital_values"])
            data["vital_missing_mask"] = np.ones_like(data["vital_missing_mask"], dtype=np.bool_)
            data["prior_vital_values"] = np.zeros_like(data["prior_vital_values"])
            data["prior_vital_missing_mask"] = np.ones_like(data["prior_vital_missing_mask"], dtype=np.bool_)
        if self.drop_prior:
            # Mirror PriorAwareDataset.__getitem__'s no-prior branch so the model's
            # has_prior-gated padding masks zero out every prior token.
            data["has_prior"] = False
            data["prior_img"] = np.zeros_like(data["prior_img"])
            data["prior_view_positions"] = np.zeros_like(data["prior_view_positions"])
            data["prior_label"] = np.zeros_like(data["prior_label"])
            data["days_since_prior"] = 0.0
            data["prior_vital_values"] = np.zeros_like(data["prior_vital_values"])
            data["prior_vital_missing_mask"] = np.ones_like(data["prior_vital_missing_mask"], dtype=np.bool_)
        return data, label


def _make_base_dataset(cfg, args, channel_mode: str | None):
    """Token-path PriorAwareDataset on the test split, channel cache OFF.

    ``channel_mode=None`` keeps the config's native build; pass ``"gray3"`` for the
    1-channel arm. Live-tokenizer path (no text-embedding cache) to match the
    grad-enabled model (live CXR-BERT)."""
    data_cfg = data_cfg_from_config(cfg, args)
    data_cfg["image_channel_cache_dir"] = None  # decode fresh, no .npy cache
    data_cfg["compute_bg_mask"] = False
    if channel_mode is not None:
        data_cfg["channel_mode"] = channel_mode
    # Force the live tokenizer branch (no precomputed text embeddings).
    data_cfg.pop("text_embedding_cache", None)
    data_cfg["use_text_embedding_cache"] = False
    _, transforms_val = get_transforms(data_cfg["size"], data_cfg.get("channel_mode"))
    ds = PriorAwareDataset(
        parquet_path=str(resolve_path(data_cfg["pred_df_path"])),
        image_size=data_cfg["size"],
        transform=transforms_val,
        label_dropout_p=0.0,
        cfg=data_cfg,
        tokenizer=_prior_aware_tokenizer(cfg, data_cfg, args),
    )
    ds.text_embedding_cache = None
    return ds


def _build_arm_loader(cfg, args, arm: str) -> DataLoader:
    channel_mode = "gray3" if arm == "gray3" else None
    base = _make_base_dataset(cfg, args, channel_mode)
    if arm == "drop_report":
        _blank_prior_aware_current_indication(base)
    ds = _AblatedDataset(base, drop_vitals=(arm == "drop_vitals"), drop_prior=(arm == "drop_prior"))
    dl_args = dataloader_args_from_config(cfg, args, shuffle=False, for_eval=True)
    return DataLoader(ds, **dl_args)


# --------------------------------------------------------------------------- #
# ablation grid
# --------------------------------------------------------------------------- #
_SUMMARY_KEYS = [
    ("mAP", "val_ap"),
    ("AUROC", "val_auroc"),
    ("head_mAP", "val/ap_head"),
    ("med_mAP", "val/ap_medium"),
    ("tail_mAP", "val/ap_tail"),
    ("head_AUROC", "val/auroc_head"),
    ("med_AUROC", "val/auroc_medium"),
    ("tail_AUROC", "val/auroc_tail"),
]


def _print_delta(full: dict[str, float], arm: dict[str, float], arm_name: str) -> None:
    print(f"\n--- delta: {arm_name} vs full (full - {arm_name}) ---", flush=True)
    for label, key in _SUMMARY_KEYS:
        f = full.get(key, float("nan"))
        a = arm.get(key, float("nan"))
        print(f"  {label:<12}{f:>9.4f}{a:>9.4f}   Δ={f - a:>+8.4f}", flush=True)


def run_ablation_grid(cfg, model, classes, device, args, arms, out_dir: Path) -> dict[str, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    for arm in arms:
        print(f"\n========== ablation arm: {arm} ==========", flush=True)
        loader = _build_arm_loader(cfg, args, arm)
        out_df, preds, labels = predict_dataframe(model, loader, classes, device)
        out_df.to_csv(out_dir / f"predictions_{arm}.csv", index=False)
        metrics = compute_metrics(preds, labels, classes)
        with open(out_dir / f"metrics_{arm}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print_validation_summary(metrics, classes, header=f"ablation | {arm}")
        results[arm] = metrics
        if arm != "full" and "full" in results:
            _print_delta(results["full"], metrics, arm)

    summary_path = out_dir / "ablation_summary.csv"
    with open(summary_path, "w") as f:
        f.write("arm," + ",".join(label for label, _ in _SUMMARY_KEYS) + "\n")
        for arm in arms:
            m = results.get(arm, {})
            f.write(arm + "," + ",".join(f"{m.get(key, float('nan')):.6f}" for _, key in _SUMMARY_KEYS) + "\n")
    print(f"\n[ablation] wrote summary -> {summary_path}", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Grad-CAM raw-asset dump
# --------------------------------------------------------------------------- #
def _to_u8(plane: np.ndarray) -> np.ndarray:
    """[0,1] float plane -> uint8 grayscale for a raw single-channel PNG."""
    return (np.clip(np.asarray(plane, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def _stream_text(streams, key: str) -> str:
    for st in streams:
        if st.key == key:
            return st.text
    return ""


def _write_text(path: Path, text: str) -> None:
    path.write_text((text or "").strip() + "\n")


def _write_vitals(path: Path, names, displays, missing) -> None:
    lines = []
    for i, name in enumerate(names):
        disp = displays[i] if i < len(displays) else ""
        miss = bool(missing[i]) if i < len(missing) else False
        lines.append(f"{name}: {disp}{'  (missing)' if miss else ''}")
    path.write_text("\n".join(lines) + "\n")


def _dump_branch(out_dir: Path, branch: str, views) -> None:
    counters: dict[str, int] = {}
    for v in views:
        name = VIEW_NAMES.get(v.view_position, "view")
        counters[name] = counters.get(name, 0) + 1
        idx = counters[name]
        base = f"image-{branch}-{name}-{idx}"
        if v.channels is not None and v.channel_names:
            for ci, cname in enumerate(v.channel_names):
                cv2.imwrite(str(out_dir / f"{base}-{cname}.png"), _to_u8(v.channels[..., ci]))
        else:
            cv2.imwrite(str(out_dir / f"{base}-gray.png"), _to_u8(v.image))
        if v.encoded:
            cv2.imwrite(str(out_dir / f"gradcam-{branch}-{name}-{idx}.png"), _to_u8(v.cam))


def _dump_gradcam_assets(result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _dump_branch(out_dir, "current", result.cur_views)
    _dump_branch(out_dir, "prior", result.prv_views)

    # Text: the current study has no radiology report (withheld to avoid leakage),
    # so its text is the clinical indication. The prior has both indication + report.
    _write_text(out_dir / "indication-current.txt", _stream_text(result.cur_texts, "cur_clin"))
    _write_text(out_dir / "indication-prior.txt", _stream_text(result.prv_texts, "prv_clin"))
    _write_text(out_dir / "report-prior.txt", _stream_text(result.prv_texts, "prv_report"))

    if result.has_vitals:
        _write_vitals(out_dir / "vitals-current.txt", result.cur_vital_names,
                      result.cur_vital_display, result.cur_vital_missing)
        _write_vitals(out_dir / "vitals-prior.txt", result.cur_vital_names,
                      result.prv_vital_display, result.prv_vital_missing)

    meta = {
        "study_id": result.study_id,
        "class_name": result.class_name,
        "class_index": result.class_index,
        "prob": result.prob,
        "logit": result.logit,
        "label": result.label,
        "has_prior": result.has_prior,
        "days_since_prior": result.days_since_prior,
        "delta_bucket": result.delta_bucket,
        "delta_bucket_name": DELTA_BUCKET_NAMES.get(result.delta_bucket, "?"),
        "true_labels": result.true_labels,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_gradcam_assets(cfg, model, classes, device, args, out_dir: Path) -> None:
    print(f"\n========== Grad-CAM raw assets (split={args.split}) ==========", flush=True)
    dataset, tokenizer = gc.build_dataset(cfg, args)
    vital_stats = {**DEFAULT_VITAL_STATS, **dict(cfg.get("data", {}).get("datamodule_cfg", {}).get("vital_stats", {}) or {})}
    channel_mode = data_cfg_from_config(cfg, args).get("channel_mode")
    out_dir.mkdir(parents=True, exist_ok=True)

    with PriorAwareAttributor(model, tokenizer, classes, device, VITAL_FIELDS, vital_stats, channel_mode) as attributor:
        print(f"[gradcam] selecting 1 best study/class over {len(dataset)} studies "
              f"({'all' if args.gradcam_scan_limit <= 0 else args.gradcam_scan_limit})...", flush=True)
        best_idx = gc.select_studies(attributor, dataset, classes, args.gradcam_scan_limit)["best"]
        saved, missing = 0, []
        for c, name in enumerate(classes):
            if best_idx[c] < 0:
                missing.append(name)
                continue
            set_seed(best_idx[c])  # deterministic >4-view sampling, matches run_prior_gradcam
            result = attributor.attribute(dataset[best_idx[c]], c)
            stem = name.replace(" ", "_").replace("/", "-")
            _dump_gradcam_assets(result, out_dir / stem)
            saved += 1
            flag = "" if result.has_prior else "  (no prior)"
            print(f"  [{saved:2d}] {name:28s} p={result.prob:.3f}{flag}", flush=True)
    print(f"[gradcam] wrote {saved} per-class asset folders -> {out_dir}", flush=True)
    if missing:
        print(f"[gradcam]   no best study found for: {', '.join(missing)}", flush=True)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation grid + Grad-CAM raw-asset dump for one prior-aware checkpoint."
    )
    add_common_args(parser, model_name="prior_aware_v8nano", mode="eval")
    parser.add_argument(
        "--ablations",
        default=",".join(ALL_ARMS),
        help=f"Comma list of ablation arms to run (subset of {list(ALL_ARMS)}; 'full' is always "
             "included as the delta baseline). Default: all.",
    )
    parser.add_argument("--skip-gradcam", action="store_true", help="Skip the Grad-CAM raw-asset pass.")
    parser.add_argument("--skip-ablations", action="store_true", help="Skip the ablation grid (Grad-CAM only).")
    parser.add_argument("--split", choices=["val", "test", "train"], default="test",
                        help="Split for the Grad-CAM study selection (default test).")
    parser.add_argument("--gradcam-scan-limit", type=int, default=0,
                        help="Cap studies scanned during best-per-class selection (0 = all).")
    return parser.parse_args()


def _resolve_arms(raw: str) -> list[str]:
    requested = [a.strip() for a in raw.split(",") if a.strip()]
    unknown = [a for a in requested if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown ablation arm(s) {unknown}; choose from {list(ALL_ARMS)}")
    # 'full' first (delta baseline), then the rest in canonical order, de-duplicated.
    ordered = ["full"] + [a for a in ALL_ARMS if a != "full" and a in requested]
    return ordered


def main() -> None:
    _install_no_meta_patch()
    args = parse_args()
    if not args.checkpoint_path:
        raise SystemExit("--checkpoint-path is required (point it at the run's best.pt)")
    set_seed(args.seed if args.seed is not None else 0)

    cfg = resolve_eval_config(args)
    classes = classes_from_config(cfg)
    device = select_device(args.accelerator)

    # Output root = the checkpoint's own run dir (one run per checkpoint, deterministic).
    run_dir = resolve_path(args.checkpoint_path).resolve().parents[1]
    out_dir = run_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ablation] checkpoint={args.checkpoint_path}\n[ablation] out={out_dir}\n"
          f"[ablation] arch={gc._arch_from_config(cfg)}  device={device}", flush=True)

    # One grad-enabled model (live CXR-BERT, penalties stripped -> logits-only forward),
    # reused for both the forward-only ablation passes and the Grad-CAM backward pass.
    model = gc.build_model(cfg, args, device)
    _materialize_meta(model)
    model.to(device)

    if not args.skip_ablations:
        run_ablation_grid(cfg, model, classes, device, args, _resolve_arms(args.ablations), out_dir)
    if not args.skip_gradcam:
        run_gradcam_assets(cfg, model, classes, device, args, out_dir / "gradcam")


if __name__ == "__main__":
    main()
