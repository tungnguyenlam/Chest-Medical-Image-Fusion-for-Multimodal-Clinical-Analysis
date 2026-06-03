from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_cache_dir(cache_root: str | Path, text_model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text_model).strip("_") or "text_model"
    digest = hashlib.sha1(text_model.encode("utf-8")).hexdigest()[:10]
    return resolve_path(cache_root) / f"{safe}-{digest}"


class TextEmbeddingCache:
    def __init__(
        self,
        text_model: str,
        cache_root: str | Path = "data/text_embeddings",
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.text_model = text_model
        self.batch_size = max(1, int(batch_size))
        self.device = select_device(device)
        self.cache_dir = model_cache_dir(cache_root, text_model)
        self.embedding_dir = self.cache_dir / "embeddings"
        self.metadata_path = self.cache_dir / "metadata.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._model = None
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self.metadata_path.exists():
            return
        self.metadata_path.write_text(
            json.dumps(
                {
                    "text_model": self.text_model,
                    "cache_key": "sha1(model|max_length|text)",
                    "storage": "one float32 numpy vector per cache key under embeddings/<key-prefix>/<key>.npy",
                },
                indent=2,
                sort_keys=True,
            )
        )

    def _key(self, text: str, max_length: int) -> str:
        payload = json.dumps(
            {"model": self.text_model, "max_length": int(max_length), "text": text},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _embedding_path(self, key: str) -> Path:
        return self.embedding_dir / key[:2] / f"{key}.npy"

    def _save_embedding(self, key: str, embedding: torch.Tensor) -> None:
        path = self._embedding_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as f:
            np.save(f, embedding.detach().cpu().float().numpy().astype(np.float32, copy=False))
        tmp_path.replace(path)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.text_model, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.text_model, trust_remote_code=True).to(self.device)
        self._model.eval()

    def unload_model(self) -> None:
        self._tokenizer = None
        self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def ensure_texts(self, texts: Iterable[str], max_length: int, desc: str = "text embeddings") -> None:
        texts = ["" if text is None else str(text) for text in texts]
        keys = [self._key(text, max_length) for text in texts]
        missing_by_key = {
            key: text
            for key, text in zip(keys, texts)
            if not self._embedding_path(key).exists()
        }
        missing = list(missing_by_key.items())

        if missing:
            self._load_model()
            with torch.inference_mode():
                for start in tqdm(range(0, len(missing), self.batch_size), desc=desc, dynamic_ncols=True):
                    batch = missing[start:start + self.batch_size]
                    batch_texts = [text for _, text in batch]
                    enc = self._tokenizer(
                        batch_texts,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    cls = self._model(**enc).last_hidden_state[:, 0, :].detach().cpu().float()
                    for (key, _), emb in zip(batch, cls):
                        self._save_embedding(key, emb)

    def get_embedding(self, text: str, max_length: int) -> np.ndarray:
        text = "" if text is None else str(text)
        key = self._key(text, max_length)
        path = self._embedding_path(key)
        if not path.exists():
            raise KeyError(f"Text embedding cache miss for key {key} in {self.cache_dir}")
        return np.load(path).astype(np.float32, copy=False)

    def embed_texts(self, texts: list[str], max_length: int, desc: str = "text embeddings") -> np.ndarray:
        texts = ["" if text is None else str(text) for text in texts]
        self.ensure_texts(texts, max_length=max_length, desc=desc)
        return np.stack([self.get_embedding(text, max_length=max_length) for text in texts], axis=0)
