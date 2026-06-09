import torch.nn as nn

from src.utils.attention import from_pretrained_best_attention


class BioBertEncoder(nn.Module):
    def __init__(self, text_model="dmis-lab/biobert-v1.1"):
        super().__init__()
        # Imported lazily so that importing src.encoder (which re-exports this
        # class) does not pull in transformers. With precomputed text embeddings
        # the encoder is never constructed, so transformers stays unimported.
        from transformers import AutoModel

        self.text_encoder = from_pretrained_best_attention(
            AutoModel, text_model, trust_remote_code=True
        )

    def forward(self, input_ids, attention_mask):
        return self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
