import torch
import torch.nn as nn
from transformers import AutoModel


class BioBertTextEncoder(nn.Module):
    """Hugging Face ``AutoModel`` wrapped to return the [CLS] embedding.

    Defaults to BioBERT v1.1. Used by CaMCheX for clinical indication and
    observations streams. ``name`` is the human-readable component label.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        name: str = "text_encoder",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.component_name = name
        self.model = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]
