import torch
import timm
import torch.nn as nn
import copy    
import einops
from transformers import AutoModel
from model.ml_decoder import MLDecoder
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer


class SingleViewModel(nn.Module):
    def __init__(self, timm_init_args):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.head = MLDecoder(num_classes=26, initial_num_features=768)

    def forward(self, x):
        x = self.model(x)
        x = self.pos_encoding(x)
        x = self.head(x)
        return x

class CaMCheXModel(nn.Module):
    def __init__(self, timm_init_args, frontal_pretrained_path=None, lateral_pretrained_path=None, text_model="dmis-lab/biobert-v1.1"):
        super().__init__()

        self.frontal_model = timm.create_model(**timm_init_args)
        self.lateral_model = timm.create_model(**timm_init_args)
        self.text_encoder = AutoModel.from_pretrained(text_model)

        self.frontal_model.head = nn.Identity()
        self.lateral_model.head = nn.Identity()

        if frontal_pretrained_path is not None:
            state_dict = torch.load(frontal_pretrained_path, map_location='cpu')
            self.frontal_model.load_state_dict(state_dict, strict=False)

        if lateral_pretrained_path is not None:
            state_dict = torch.load(lateral_pretrained_path, map_location='cpu')
            self.lateral_model.load_state_dict(state_dict, strict=False)
            
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
        b, s, c, h, w = x.shape

        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        view_positions = einops.rearrange(view_positions, 'b s -> (b s)')

        nonzero_mask = (x.sum(dim=(1, 2, 3)) != 0)
        x_nonzero = x[nonzero_mask]
        view_positions_nonzero = view_positions[nonzero_mask]

        frontal_mask = (view_positions_nonzero == 1)
        lateral_mask = (view_positions_nonzero == 2)

        feats = torch.zeros((x_nonzero.shape[0], 768, h // 32, w // 32), device=x.device)
        
        if frontal_mask.any():
            feats[frontal_mask] = self.frontal_model(x_nonzero[frontal_mask])
        if lateral_mask.any():
            feats[lateral_mask] = self.lateral_model(x_nonzero[lateral_mask])

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

        #with torch.no_grad():
        clin_cls = self.text_encoder(
            input_ids=clinical_input_ids, attention_mask=clinical_attention_mask
        ).last_hidden_state[:, 0, :]  # [B, C]

        obs_cls = self.text_encoder(
            input_ids=clinical_obs_input_ids, attention_mask=clinical_obs_attention_mask
        ).last_hidden_state[:, 0, :]  # [B, C]

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