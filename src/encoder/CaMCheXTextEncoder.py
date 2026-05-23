import torch.nn as nn

from src.encoder.BioBertEncoder import BioBertEncoder


class CaMCheXTextEncoder(nn.Module):
    def __init__(self, text_model="dmis-lab/biobert-v1.1"):
        super().__init__()
        self.biobert_encoder = BioBertEncoder(text_model=text_model)

    def forward(
        self,
        clinical_input_ids,
        clinical_attention_mask,
        clinical_obs_input_ids,
        clinical_obs_attention_mask,
    ):
        clinical_cls = self.biobert_encoder(
            input_ids=clinical_input_ids,
            attention_mask=clinical_attention_mask,
        )
        obs_cls = self.biobert_encoder(
            input_ids=clinical_obs_input_ids,
            attention_mask=clinical_obs_attention_mask,
        )
        return clinical_cls, obs_cls
