import warnings

import numpy as np
from torch.utils.data import Dataset

from src.dataloader.utils import (
    _safe_decode_jpeg,
    load_or_build_channels,
    make_preprocess_config,
    resolve_preferred_image_path,
)


class SingleViewDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform
        self.channel_mode = cfg.get("channel_mode")
        self.channel_cache_dir = cfg.get("image_channel_cache_dir")
        self.channel_cfg = make_preprocess_config(cfg) if self.channel_mode else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if all([c in self.df.columns for c in self.cfg['classes']]):
            label = self.df.iloc[index][self.cfg['classes']].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        path = self.df.iloc[index]["path"]

        if self.channel_mode:
            # Raw path -> cache hit is string-keyed, no source-FS stat.
            img = load_or_build_channels(path, self.channel_mode, self.channel_cfg, self.channel_cache_dir)
        else:
            img = _safe_decode_jpeg(resolve_preferred_image_path(path))
        if img is None:
            warnings.warn(f"Skipping unreadable image at index {index} ({path}); using neighbor sample")
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label 
