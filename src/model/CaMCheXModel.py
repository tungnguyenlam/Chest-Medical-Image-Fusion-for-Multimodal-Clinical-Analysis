import einops
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from src.decoder.MLDecoder import MLDecoder
from src.encoder import CaMCheXImageEncoder, CaMCheXTextEncoder


class CaMCheXModel(nn.Module):
    def __init__(self, timm_init_args, frontal_pretrained_path=None, lateral_pretrained_path=None, text_model="dmis-lab/biobert-v1.1"):
        super().__init__()

        self.image_encoder = CaMCheXImageEncoder(
            timm_init_args=timm_init_args,
            frontal_pretrained_path=frontal_pretrained_path,
            lateral_pretrained_path=lateral_pretrained_path,
        )
        self.text_encoder = CaMCheXTextEncoder(text_model=text_model)
            
        self.conv2d = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)

        self.pos_encoding = Summer(PositionalEncoding2D(768))

        self.padding_token = nn.Parameter(torch.randn(1, 768, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(6, 768, 1, 1)) 
        with torch.no_grad():
            self.segment_embedding.data = self.segment_embedding.data.clamp(-1.0, 1.0)

        self.head = MLDecoder(num_classes=26, initial_num_features=768)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.1, batch_first=True),
            num_layers=2
        )

    def forward(self, data):
        _, x, view_positions, clinical_input_ids, clinical_attention_mask, clinical_obs_input_ids, clinical_obs_attention_mask = data
        b, s = x.shape[:2]

        feats, nonzero_mask = self.image_encoder(x, view_positions)
        feats = self.conv2d(feats)
        feats = self.pos_encoding(feats)

        pad_tokens = einops.repeat(
            self.padding_token, '1 c 1 1 -> (b s) c h w', b=b, s=s, h=feats.shape[2], w=feats.shape[3]
        ).type_as(feats).clone()

        segment_embedding = einops.repeat(
            self.segment_embedding[:4], 's c 1 1 -> (b s) c h w', b=b, h=feats.shape[2], w=feats.shape[3]
        ).type_as(feats)

        pad_tokens[nonzero_mask] = feats + segment_embedding[nonzero_mask]
        pad_tokens = einops.rearrange(pad_tokens, '(b s) c h w -> b s c h w', b=b, s=s)

        feats_img = pad_tokens
        b, s_img, cdim, h2, w2 = feats_img.shape

        clin_cls, obs_cls = self.text_encoder(
            clinical_input_ids=clinical_input_ids,
            clinical_attention_mask=clinical_attention_mask,
            clinical_obs_input_ids=clinical_obs_input_ids,
            clinical_obs_attention_mask=clinical_obs_attention_mask,
        )

        clin_seg = self.segment_embedding[4].view(cdim).to(clin_cls.device)  
        obs_seg  = self.segment_embedding[5].view(cdim).to(obs_cls.device)   

        clin_feats = (clin_cls + clin_seg).unsqueeze(-1).unsqueeze(-1)       
        clin_feats = clin_feats.expand(-1, -1, h2, w2).unsqueeze(1)      

        obs_feats  = (obs_cls + obs_seg).unsqueeze(-1).unsqueeze(-1)         
        obs_feats  = obs_feats.expand(-1, -1, h2, w2).unsqueeze(1)      
        feats_cat = torch.cat([feats_img, clin_feats, obs_feats], dim=1)     
        x = einops.rearrange(feats_cat, "b s c h w -> b (s h w) c")        

        img_slots_valid = nonzero_mask.view(b, s_img)                       
        img_valid_tokens = img_slots_valid.unsqueeze(-1).unsqueeze(-1) \
                                        .expand(b, s_img, h2, w2) \
                                        .reshape(b, s_img * h2 * w2)         
        text_valid_tokens = torch.ones(b, 2 * h2 * w2, dtype=torch.bool, device=x.device)
        valid_tokens = torch.cat([img_valid_tokens, text_valid_tokens], dim=1)
        mask = ~valid_tokens

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.head(x, mask)
        return x
