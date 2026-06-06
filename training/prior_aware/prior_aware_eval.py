from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.PriorAwareCaMCheXModel import PriorAwareCaMCheXModel
from training.common import (
    add_common_args,
    classes_from_config,
    load_config,
    load_weights,
    make_prior_aware_eval_loader,
    predict_dataframe,
    prepare_run_dir,
    resolve_path,
    select_device,
    timm_args_from_config,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate / predict with the prior-aware CaMCheX model.")
    add_common_args(parser, model_name="prior_aware")
    parser.add_argument("--frontal-pretrained-path")
    parser.add_argument("--lateral-pretrained-path")
    parser.add_argument("--text-model", help="Override model.text_model from config.")
    parser.add_argument("--freeze-text-encoder", action="store_true", help="Freeze BioBERT/CXR-BERT if token ids are used.")
    parser.add_argument("--use-precomputed-text-embeddings", action="store_true", help="Use the shared frozen text embedding cache and do not load BioBERT/CXR-BERT.")
    parser.add_argument("--text-embedding-cache-dir", help="Override the shared text embedding cache root.")
    parser.add_argument("--output-csv", default="predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = prepare_run_dir(args)
    write_resolved_config(run_dir, args, cfg)

    loader, _ = make_prior_aware_eval_loader(cfg, args)
    text_model = args.text_model or cfg.get("model", {}).get("text_model") or "dmis-lab/biobert-v1.1"
    model_init_args = dict(cfg.get("model", {}).get("model_init_args", {}) or {})
    if args.freeze_text_encoder:
        model_init_args["freeze_text_encoder"] = True
    data_cfg = cfg.get("data", {}).get("datamodule_cfg", {}) or {}
    if args.use_precomputed_text_embeddings or data_cfg.get("use_text_embedding_cache", False):
        model_init_args["use_precomputed_text_embeddings"] = True
        model_init_args["freeze_text_encoder"] = True
    model = PriorAwareCaMCheXModel(
        timm_init_args=timm_args_from_config(cfg, args),
        frontal_pretrained_path=str(resolve_path(args.frontal_pretrained_path)) if args.frontal_pretrained_path else None,
        lateral_pretrained_path=str(resolve_path(args.lateral_pretrained_path)) if args.lateral_pretrained_path else None,
        text_model=text_model,
        **model_init_args,
    )
    if args.checkpoint_path:
        load_weights(model, args.checkpoint_path)
    device = select_device(args.accelerator)
    out, _, _ = predict_dataframe(model, loader, classes_from_config(cfg), device)
    out_path = run_dir / args.output_csv
    out.to_csv(out_path, index=False)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
