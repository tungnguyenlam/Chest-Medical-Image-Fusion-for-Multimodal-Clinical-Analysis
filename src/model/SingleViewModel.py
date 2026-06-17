import timm
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from src.decoder.MLDecoder import MLDecoder


class SingleViewModel(nn.Module):
    def __init__(self, timm_init_args, n_classes: int = 26):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()
        self.pos_encoding = Summer(PositionalEncoding2D(768))
        self.head = MLDecoder(num_classes=n_classes, initial_num_features=768)

    def forward(self, x):
        x = self.model(x)
        x = self.pos_encoding(x)
        x = self.head(x)
        return x
