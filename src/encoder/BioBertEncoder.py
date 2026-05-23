import torch.nn as nn
from transformers import AutoModel


class BioBertEncoder(nn.Module):
    def __init__(self, text_model="dmis-lab/biobert-v1.1"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)

    def forward(self, input_ids, attention_mask):
        return self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
