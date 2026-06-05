import torch.nn as nn
from transformers import AutoModel

from src.utils.attention import from_pretrained_best_attention


class BioBertEncoder(nn.Module):
    def __init__(self, text_model="dmis-lab/biobert-v1.1"):
        super().__init__()
        self.text_encoder = from_pretrained_best_attention(
            AutoModel, text_model, trust_remote_code=True
        )

    def forward(self, input_ids, attention_mask):
        return self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
