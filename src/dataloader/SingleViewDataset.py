import os
import warnings

import numpy as np
from torch.utils.data import Dataset

from src.dataloader.utils import _safe_decode_jpeg


class SingleViewDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if all([c in self.df.columns for c in self.cfg['classes']]):
            label = self.df.iloc[index][self.cfg['classes']].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        path = self.df.iloc[index]["path"]
        path_resized = path.replace(".jpg", "_resized_1024.jpg")
        path = path_resized if os.path.exists(path_resized) else path

        img = _safe_decode_jpeg(path)
        if img is None:
            warnings.warn(f"Skipping unreadable image at index {index} ({path}); using neighbor sample")
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label 
