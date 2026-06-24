from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.PriorAwareV8NanoModel import PriorAwareV8NanoModel
from training.common import (
    add_common_args,
    classes_from_config,
    data_cfg_from_config,
    image_norm_stats,
    load_config,
    log_rss,
    loss_args_from_config,
    lr_from_config,
    make_prior_aware_loaders,
    model_init_args_from_config,
    prepare_run_dir,
    resolve_path,
    timm_args_from_config,
    train_model,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prior-Aware v8 Nano (v6 stack + noise-aware label-correlation graph head).")
    add_common_args(parser, model_name="prior_aware_v8nano")
    parser.add_argument("--frontal-pretrained-path", help="Stage-1 frontal timm backbone state_dict.")
    parser.add_argument("--lateral-pretrained-path", help="Stage-1 lateral timm backbone state_dict.")
    parser.add_argument("--text-model", help="Override model.text_model from config.")
    parser.add_argument("--freeze-text-encoder", action="store_true", help="Freeze BioBERT/CXR-BERT if token ids are used.")
    parser.add_argument("--use-precomputed-text-embeddings", action="store_true", help="Use the shared frozen text embedding cache and do not load CXR-BERT.")
    parser.add_argument("--text-embedding-cache-dir", help="Override the shared text embedding cache root.")
    parser.add_argument(
        "--text-embeddings-gpu-resident",
        action="store_true",
        help="(opt-in) Keep the precomputed text embeddings as one frozen table inside the model "
        "(moved to the training device) and have the dataset emit integer row indices instead of "
        "per-sample float vectors. Requires --use-precomputed-text-embeddings.",
    )
    parser.add_argument(
        "--uint8-image-pipeline",
        action="store_true",
        help="(opt-in) Ship images through the DataLoader as uint8 [0,255] and dequantize + "
        "normalize on-device in the model. Requires a channel mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = prepare_run_dir(args)
    cfg = load_config(args.config)
    write_resolved_config(run_dir, args, cfg)

    train_loader, val_loader = make_prior_aware_loaders(cfg, args)
    frontal_pretrained_path = str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None
    lateral_pretrained_path = str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None
    text_model = args.text_model or cfg.get("model", {}).get("text_model") or "microsoft/BiomedVLP-CXR-BERT-specialized"
    model_init_args = model_init_args_from_config(cfg)
    if args.freeze_text_encoder:
        model_init_args["freeze_text_encoder"] = True
    data_cfg = cfg.get("data", {}).get("datamodule_cfg", {}) or {}
    if args.use_precomputed_text_embeddings or data_cfg.get("use_text_embedding_cache", False):
        model_init_args["use_precomputed_text_embeddings"] = True
        model_init_args["freeze_text_encoder"] = True
    model = PriorAwareV8NanoModel(
        timm_init_args=timm_args_from_config(cfg, args),
        frontal_pretrained_path=frontal_pretrained_path,
        lateral_pretrained_path=lateral_pretrained_path,
        text_model=text_model,
        **model_init_args,
    )
    if args.text_embeddings_gpu_resident:
        import torch

        if not model_init_args.get("use_precomputed_text_embeddings"):
            raise SystemExit("--text-embeddings-gpu-resident requires --use-precomputed-text-embeddings.")
        cache = getattr(train_loader.dataset, "text_embedding_cache", None)
        if cache is None:
            raise SystemExit("--text-embeddings-gpu-resident: no text embedding cache was built.")
        table = cache.build_index_table()
        model.attach_text_embedding_table(torch.from_numpy(table))
        print(
            f"[train] text embeddings GPU-resident: table {tuple(table.shape)} "
            f"({table.nbytes / 1e6:.0f} MB), dataset emits row indices"
        )
        print("[train] precomputing text-stream row indices and dropping raw-text columns ...", flush=True)
        for loader in (train_loader, val_loader):
            loader.dataset.precompute_text_indices()
        del table
        log_rss("after build_index_table (RAM dict freed; host table awaits model->device)")
    if args.uint8_image_pipeline:
        mean, std = image_norm_stats(data_cfg_from_config(cfg, args))
        model.enable_input_normalization(mean, std)
        print(f"[train] uint8 image pipeline: model normalizes on-device (mean={mean}, std={std})")
    import gc

    gc.freeze()
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        run_dir=run_dir,
        lr=lr_from_config(cfg, args),
        classes=classes_from_config(cfg),
        loss_init_args=loss_args_from_config(cfg),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
