import einops
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from src.decoder.MLDecoder import MLDecoder
from src.encoder import CaMCheXTextEncoder, TimmImageEncoder


class ConvNeXtV2NanoImageEncoder(nn.Module):
    def __init__(
        self,
        timm_init_args,
        frontal_pretrained_path=None,
        lateral_pretrained_path=None,
        feature_dim: int = 640,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.frontal_encoder = TimmImageEncoder(
            timm_init_args=timm_init_args,
            pretrained_path=frontal_pretrained_path,
        )
        self.lateral_encoder = TimmImageEncoder(
            timm_init_args=timm_init_args,
            pretrained_path=lateral_pretrained_path,
        )

    def forward(self, x, view_positions):
        b, s, _, h, w = x.shape
        x = x.reshape(b * s, *x.shape[2:])
        view_positions = view_positions.reshape(b * s)

        nonzero_mask = x.sum(dim=(1, 2, 3)) != 0
        x_nonzero = x[nonzero_mask]
        view_positions_nonzero = view_positions[nonzero_mask]

        feats = torch.zeros(
            (x_nonzero.shape[0], self.feature_dim, h // 32, w // 32),
            device=x.device,
            dtype=x.dtype,
        )
        frontal_mask = view_positions_nonzero == 1
        lateral_mask = view_positions_nonzero == 2
        if frontal_mask.any():
            feats[frontal_mask] = self.frontal_encoder(
                x_nonzero[frontal_mask]
            ).to(feats.dtype)
        if lateral_mask.any():
            feats[lateral_mask] = self.lateral_encoder(
                x_nonzero[lateral_mask]
            ).to(feats.dtype)
        return feats, nonzero_mask


class VitalsTokenProjector(nn.Module):
    def __init__(
        self,
        num_vitals: int = 7,
        d_model: int = 768,
        grid_size: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        vital_dropout_p: float = 0.0,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.num_vitals = num_vitals
        self.d_model = d_model
        self.grid_size = grid_size
        self.vital_dropout_p = vital_dropout_p
        self.mask_value = mask_value
        self.proj = nn.Sequential(
            nn.Linear(num_vitals * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, grid_size * grid_size * d_model),
        )

    def forward(self, values: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        values = values.float()
        missing_mask = missing_mask.bool()
        if self.training and self.vital_dropout_p > 0:
            drop_mask = torch.rand_like(values) < self.vital_dropout_p
            values = values.masked_fill(drop_mask, self.mask_value)
            missing_mask = missing_mask | drop_mask
        x = torch.cat([values, missing_mask.float()], dim=-1)
        x = self.proj(x)
        return x.view(values.size(0), self.grid_size * self.grid_size, self.d_model)


class CaMCheXV2NanoVitalsModel(nn.Module):
    gradcam_runner_module = "src.interpret.run_gradcam"

    def __init__(
        self,
        timm_init_args,
        frontal_pretrained_path=None,
        lateral_pretrained_path=None,
        text_model="microsoft/BiomedVLP-CXR-BERT-specialized",
        freeze_text_encoder: bool = False,
        use_precomputed_text_embeddings: bool = False,
        vital_dropout_p: float = 0.0,
        vitals_dropout: float = 0.1,
        vitals_hidden_dim: int = 256,
        d_model: int = 768,
    ):
        super().__init__()
        self.freeze_text_encoder = freeze_text_encoder
        self.use_precomputed_text_embeddings = use_precomputed_text_embeddings
        self.d_model = d_model

        self.image_encoder = ConvNeXtV2NanoImageEncoder(
            timm_init_args=timm_init_args,
            frontal_pretrained_path=frontal_pretrained_path,
            lateral_pretrained_path=lateral_pretrained_path,
        )
        self.text_encoder = None
        if not use_precomputed_text_embeddings:
            self.text_encoder = CaMCheXTextEncoder(text_model=text_model)
            if freeze_text_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                self.text_encoder.eval()

        self.image_proj = nn.Conv2d(640, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(d_model))
        self.vitals_projector = VitalsTokenProjector(
            num_vitals=7,
            d_model=d_model,
            grid_size=8,
            hidden_dim=vitals_hidden_dim,
            dropout=vitals_dropout,
            vital_dropout_p=vital_dropout_p,
        )

        self.padding_token = nn.Parameter(torch.randn(1, d_model, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(6, d_model, 1, 1))
        with torch.no_grad():
            self.segment_embedding.data = self.segment_embedding.data.clamp(-1.0, 1.0)

        self.head = MLDecoder(num_classes=26, initial_num_features=d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=0.1, batch_first=True),
            num_layers=2,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder and self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def _encode_text(self, clinical_input_ids, clinical_attention_mask):
        if clinical_input_ids.is_floating_point() and clinical_input_ids.ndim == 2:
            return clinical_input_ids
        if self.use_precomputed_text_embeddings:
            raise TypeError("use_precomputed_text_embeddings=True requires float clinical embedding tensors")
        if self.text_encoder is None:
            raise RuntimeError("text_encoder is not initialized; disable use_precomputed_text_embeddings for token batches")
        if self.freeze_text_encoder:
            with torch.no_grad():
                clinical_cls = self.text_encoder.biobert_encoder(
                    input_ids=clinical_input_ids,
                    attention_mask=clinical_attention_mask,
                )
            return clinical_cls
        return self.text_encoder.biobert_encoder(
            input_ids=clinical_input_ids,
            attention_mask=clinical_attention_mask,
        )

    def forward(self, data):
        (
            _,
            x,
            view_positions,
            clinical_input_ids,
            clinical_attention_mask,
            vital_values,
            vital_missing_mask,
        ) = data
        b, s = x.shape[:2]

        feats, nonzero_mask = self.image_encoder(x, view_positions)
        feats = self.image_proj(feats)
        feats = self.pos_encoding(feats)
        _, cdim, h2, w2 = feats.shape

        pad_tokens = einops.repeat(
            self.padding_token,
            "1 c 1 1 -> (b s) c h w",
            b=b,
            s=s,
            h=h2,
            w=w2,
        ).type_as(feats).clone()
        segment_embedding = einops.repeat(
            self.segment_embedding[:4],
            "s c 1 1 -> (b s) c h w",
            b=b,
            h=h2,
            w=w2,
        ).type_as(feats)
        pad_tokens[nonzero_mask] = feats + segment_embedding[nonzero_mask]
        img_block = einops.rearrange(pad_tokens, "(b s) c h w -> b s c h w", b=b, s=s)
        img_tokens = einops.rearrange(img_block, "b s c h w -> b (s h w) c")

        clinical_cls = self._encode_text(clinical_input_ids, clinical_attention_mask)
        clin_seg = self.segment_embedding[4].view(cdim).to(clinical_cls.device)
        clin_feats = (clinical_cls + clin_seg).unsqueeze(1).expand(-1, h2 * w2, -1)

        vital_tokens = self.vitals_projector(
            vital_values.to(x.device),
            vital_missing_mask.to(x.device),
        )
        vital_seg = self.segment_embedding[5].view(cdim).to(vital_tokens.device)
        vital_tokens = vital_tokens + vital_seg

        tokens = torch.cat([img_tokens, clin_feats, vital_tokens], dim=1)
        img_slots_valid = nonzero_mask.view(b, s)
        img_valid_tokens = (
            img_slots_valid.unsqueeze(-1).unsqueeze(-1)
            .expand(b, s, h2, w2)
            .reshape(b, s * h2 * w2)
        )
        text_valid_tokens = torch.ones(b, 2 * h2 * w2, dtype=torch.bool, device=tokens.device)
        valid_tokens = torch.cat([img_valid_tokens, text_valid_tokens], dim=1)
        mask = ~valid_tokens

        tokens = self.transformer_encoder(tokens, src_key_padding_mask=mask)
        return self.head(tokens, mask)
