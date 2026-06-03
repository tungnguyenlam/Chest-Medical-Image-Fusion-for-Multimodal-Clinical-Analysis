from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def clinical_text(row) -> str:
    text = row.get("clinical_indication", "")
    if pd.isna(text) or str(text).strip() == "":
        return "No clinical history available."
    return str(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute frozen clinical indication embeddings for the non-prior "
            "camchex_v2nano_vitals training path. For prior-aware parquet files, "
            "use src/prepare/04_build_prior_aware_dataset.py --precompute-text-embeddings."
        )
    )
    parser.add_argument("--input-csv", action="append", required=True, help="CSV to read. Repeat for train/val/test.")
    parser.add_argument("--output-path", required=True, help="Output .pt file containing study_id -> CLS embedding.")
    parser.add_argument("--text-model", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.text_model, trust_remote_code=True).to(device)
    model.eval()

    rows = []
    seen = set()
    for csv_path in args.input_csv:
        df = pd.read_csv(resolve_path(csv_path), low_memory=False)
        for study_id, group in df.groupby("study_id"):
            key = str(study_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append((key, clinical_text(group.iloc[0])))

    embeddings = {}
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="clinical embeddings"):
            batch = rows[start:start + args.batch_size]
            tokens = tokenizer(
                [text for _, text in batch],
                padding="max_length",
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            cls = model(**tokens).last_hidden_state[:, 0, :].detach().cpu()
            for (study_id, _), emb in zip(batch, cls):
                embeddings[study_id] = emb.float()

    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_model": args.text_model,
            "embedding_dim": next(iter(embeddings.values())).numel() if embeddings else 0,
            "embeddings": embeddings,
        },
        output_path,
    )
    print(f"saved {len(embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    main()
