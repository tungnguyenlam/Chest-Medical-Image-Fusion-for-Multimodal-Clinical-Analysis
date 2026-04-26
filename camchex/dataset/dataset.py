import os
import cv2
import pandas as pd
import numpy as np
import jpeg4py as jpeg
from torch import from_numpy
from torch.utils.data import Dataset


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
        
        img = jpeg.JPEG(path).decode()

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label 

class CaMCheXDataset(Dataset):
    def __init__(self, cfg, df, transform=None, tokenizer=None):
        self.cfg = cfg
        self.transform = transform
        assert tokenizer is not None, "Tokenizer must be provided for this dataset."
        self.tokenizer = tokenizer
        self.df = df.groupby("study_id")
        self.study_ids = list(self.df.groups.keys())

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, index):
        df = self.df.get_group(self.study_ids[index])
        study_id = self.study_ids[index]
        if len(df) > 4:
            df = df.sample(4)

        if all([c in df.columns for c in self.cfg['classes']]):
            label = df[self.cfg['classes']].iloc[0].to_numpy().astype(np.float32)
        else:
            label = np.zeros(len(self.cfg['classes']))

        imgs = []
        view_positions = []
        for i in range(len(df)):
            path = df.iloc[i]["path"]
            path_resized = path.replace(".jpg", "_resized_1024.jpg")
            path = path_resized if os.path.exists(path_resized) else path

            if os.path.exists(path):
                img = jpeg.JPEG(path).decode()
            else:
                raise FileNotFoundError(f"Neither resized nor original image found: {path}")

            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
                img = np.moveaxis(img, -1, 0)

            imgs.append(img)

            vp = df.iloc[i].get("ViewPosition", "").upper()
            if vp in ["AP", "PA", "FRONTAL"]:
                view_positions.append(1)
            elif vp in ["LATERAL", "LL"]:
                view_positions.append(2)
            else:
                view_positions.append(0)

        img = np.stack(imgs, axis=0)
        img = np.concatenate([img, np.zeros((4-len(df), 3, self.cfg['size'], self.cfg['size']))], axis=0).astype(np.float32)
        view_positions = np.array(view_positions + [0]*(4-len(df)), dtype=np.int64)

        clinical_text = df.iloc[0].get("clinical_indication", "")
        if pd.isna(clinical_text) or clinical_text.strip() == "":
            clinical_text = "No clinical history available."
        clinical_tokens = self.tokenizer(
            clinical_text,
            padding="max_length",
            truncation=True,
            max_length=384,
            return_tensors="pt"
        )

        clinical_input_ids = clinical_tokens["input_ids"].squeeze(0)
        clinical_attention_mask = clinical_tokens["attention_mask"].squeeze(0)

        obs_fields = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "gender"]
        obs_values = {
            field: str(df.iloc[0].get(field)) if not pd.isna(df.iloc[0].get(field)) else "NA"
            for field in obs_fields
        }
        obs_text = " | ".join([
            f"Temperature: {obs_values['temperature']}",
            f"Heart rate: {obs_values['heartrate']}",
            f"Respiratory rate: {obs_values['resprate']}",
            f"O2 Saturation: {obs_values['o2sat']}",
            f"Systolic BP: {obs_values['sbp']}",
            f"Diastolic BP: {obs_values['dbp']}",
            f"Gender: {obs_values['gender']}"
        ])
                
        clinical_obs_tokens = self.tokenizer(
            obs_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        clinical_obs_input_ids = clinical_obs_tokens["input_ids"].squeeze(0)
        clinical_obs_attention_mask = clinical_obs_tokens["attention_mask"].squeeze(0)

        return (
            study_id,
            img,  
            view_positions,  
            clinical_input_ids,
            clinical_attention_mask,
            clinical_obs_input_ids,
            clinical_obs_attention_mask
        ),  label
