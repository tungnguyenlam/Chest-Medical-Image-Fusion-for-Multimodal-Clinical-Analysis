"""Build the noise-aware label-correlation graph artifact for Prior-Aware v8.

Train-split only (leakage-safe). Produces ``label_graph.pt`` holding the frozen matrices
the v8 graph head consumes, plus CXR-BERT class-name node features. The artifact stores the
*shrunk* lift / conditional / significance / curated matrices (not a single pre-built
adjacency) so the operative sparsifier knobs -- ``lift_threshold``, ``top_k``, ``graph_dir``,
``reweight_p``, ``use_hierarchy_edges`` -- stay config-only at train time (no rebuild).

Pipeline (mirrors ``data/00-examine-data/v8_label_graph_gate.ipynb`` and
``docs/prior_aware_v8_label_graph.md`` §3):
  1. counts N(i), N(i,j) and marginals P(j) from the train parquet labels;
  2. Bayesian shrinkage toward base rate (§3.1): P̂(j|i)=(N(i,j)+α·P(j))/(N(i)+α);
  3. BH-corrected one-sided Fisher significance "P(j|i) > P(j)" (§3.2);
  4. curated clinical-hierarchy whitelist mask (§3.3);
  5. CXR-BERT class-name node features Z0 ∈ R^{26×768} (names only -- no labels, no leakage).

Usage:
  python src/prepare/05_build_label_graph.py \
      --train-parquet data/data-camchex/prior_aware_train.parquet \
      --out data/data-camchex/label_graph.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSES = [
    "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Consolidation",
    "Edema", "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture",
    "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity", "Mass", "No Finding",
    "Nodule", "Pleural Effusion", "Pleural Other", "Pleural Thickening",
    "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax",
    "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta",
]

# Curated clinical-hierarchy / air-leak edges (unordered) allowed past the deterministic
# prune and unioned in when use_hierarchy_edges is on. Real ontology relations, not
# co-mention artifacts (doc §3.3).
CURATED_PAIRS = [
    ("Enlarged Cardiomediastinum", "Cardiomegaly"),
    ("Pneumothorax", "Pneumomediastinum"),
    ("Pneumomediastinum", "Subcutaneous Emphysema"),
    ("Pneumothorax", "Subcutaneous Emphysema"),
    ("Pneumoperitoneum", "Pneumomediastinum"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the v8 label-correlation graph artifact (train split only).")
    p.add_argument("--train-parquet", default="data/data-camchex/prior_aware_train.parquet")
    p.add_argument("--out", default="data/data-camchex/label_graph.pt")
    p.add_argument("--alpha-shrink", type=float, default=10.0, help="Bayesian pseudocount (§3.1).")
    p.add_argument("--bh-q", type=float, default=0.05, help="BH FDR for the Fisher significance test (§3.2).")
    p.add_argument("--text-model", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    p.add_argument("--device", default="cpu", help="Device for CXR-BERT node-feature encoding.")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def compute_cooccurrence(train_parquet: Path, alpha: float, bh_q: float):
    """Counts -> shrunk conditional / lift -> BH-significant mask (train split only)."""
    df = pd.read_parquet(train_parquet, columns=["label"])
    y = np.stack(df["label"].to_numpy()).astype(np.int64)  # (N, K)
    n, k = y.shape
    assert k == len(CLASSES), (k, len(CLASSES))

    n_i = y.sum(axis=0)
    p_j = n_i / n
    co = y.T @ y
    np.fill_diagonal(co, 0)

    p_hat = (co + alpha * p_j[None, :]) / (n_i[:, None] + alpha)   # shrunk P(j|i)
    lift = p_hat / p_j[None, :]
    np.fill_diagonal(p_hat, 0.0)
    np.fill_diagonal(lift, 0.0)

    pairs, pvals = [], []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            a = co[i, j]
            b = n_i[i] - a
            c = n_i[j] - a
            d = n - n_i[i] - n_i[j] + a
            _, pv = fisher_exact([[a, b], [c, d]], alternative="greater")
            pairs.append((i, j))
            pvals.append(pv)
    reject, _, _, _ = multipletests(np.asarray(pvals), alpha=bh_q, method="fdr_bh")
    sig = np.zeros((k, k), dtype=bool)
    for (i, j), rj in zip(pairs, reject):
        sig[i, j] = rj

    curated = np.zeros((k, k), dtype=bool)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for a_name, b_name in CURATED_PAIRS:
        ia, ib = idx[a_name], idx[b_name]
        curated[ia, ib] = curated[ib, ia] = True

    return {
        "n_train": int(n), "n_i": n_i, "p_j": p_j,
        "pcond": p_hat.astype(np.float32), "lift": lift.astype(np.float32),
        "sig": sig, "curated": curated,
    }


def encode_node_features(text_model: str, device: str) -> torch.Tensor:
    """CXR-BERT CLS for each class *name* -> Z0 ∈ R^{K×768}. Names only: no leakage."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(text_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(text_model, trust_remote_code=True).to(device).eval()
    feats = []
    with torch.no_grad():
        for name in CLASSES:
            enc = tok(name, return_tensors="pt", truncation=True, max_length=16).to(device)
            cls = model(**enc).last_hidden_state[:, 0, :]   # (1, 768)
            feats.append(cls.squeeze(0).cpu())
    return torch.stack(feats, dim=0).float()


def main() -> None:
    args = parse_args()
    train_parquet = _resolve(args.train_parquet)
    out_path = _resolve(args.out)
    print(f"[graph] train parquet: {train_parquet}")

    stats = compute_cooccurrence(train_parquet, args.alpha_shrink, args.bh_q)
    print(f"[graph] N={stats['n_train']:,}  significant pairs={int(stats['sig'].sum())}/650")

    print(f"[graph] encoding {len(CLASSES)} class-name node features with {args.text_model} ...")
    z0 = encode_node_features(args.text_model, args.device)
    print(f"[graph] node features Z0: {tuple(z0.shape)}")

    artifact = {
        "classes": CLASSES,
        "node_features": z0,                                  # (K, 768) float
        "lift": torch.from_numpy(stats["lift"]),              # (K, K) shrunk lift
        "pcond": torch.from_numpy(stats["pcond"]),            # (K, K) shrunk P(j|i)
        "sig": torch.from_numpy(stats["sig"]),                # (K, K) bool BH-significant
        "curated_mask": torch.from_numpy(stats["curated"]),   # (K, K) bool curated hierarchy
        "meta": {
            "n_train": stats["n_train"],
            "alpha_shrink": args.alpha_shrink,
            "bh_q": args.bh_q,
            "text_model": args.text_model,
            "n_classes": len(CLASSES),
            "note": "train-split only; sparsifier (lift_threshold/top_k/reweight_p) applied in the model",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    print(f"[graph] wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
