import os
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, ConcatDataset
from dataset.dataset import SingleViewDataset, CaMCheXDataset
from dataset.transforms import get_transforms


class CaMCheXDataModule(pl.LightningDataModule):
    def __init__(self, datamodule_cfg, dataloader_init_args):
        super(CaMCheXDataModule, self).__init__()
        self.cfg = datamodule_cfg
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.train_df = pd.read_csv(self.cfg["train_df_path"])
        self.devel_df = pd.read_csv(self.cfg["devel_df_path"])
        self.test_df = pd.read_csv(self.cfg["pred_df_path"])
        self.dataloader_init_args = dataloader_init_args
        if self.cfg["use_pseudo_label"]:
            print("Using pseudo label")
            self.vin_df = pd.read_csv(self.cfg["vinbig_train_df_path"])
            self.nih_df = pd.read_csv(self.cfg["nih_train_df_path"])
            self.chexpert_df = pd.read_csv(self.cfg["chexpert_train_df_path"])

    def setup(self, stage):
        transforms_train, transforms_val = get_transforms(self.cfg["size"])
        if stage in ('fit', 'validate'):
            train_df = self.train_df
            val_df = self.devel_df

            self.train_dataset = CaMCheXDataset(self.cfg, train_df, transforms_train, self.tokenizer)
            self.val_dataset = CaMCheXDataset(self.cfg, val_df, transforms_val, self.tokenizer)       
            # self.train_dataset = SingleViewDataset(self.cfg, train_df, transforms_train)
            # self.val_dataset = SingleViewDataset(self.cfg, val_df, transforms_val)
            
            if self.cfg["use_pseudo_label"]:
                vin_dataset = SingleViewDataset(self.cfg, self.vin_df, transforms_train)
                nih_dataset = SingleViewDataset(self.cfg, self.nih_df, transforms_train)
                chexpert_dataset = SingleViewDataset(self.cfg, self.chexpert_df, transforms_train)
                print(f"vin len: {len(vin_dataset)}")
                print(f"nih len: {len(nih_dataset)}")
                print(f"chexpert len: {len(chexpert_dataset)}")
                self.train_dataset = ConcatDataset([self.train_dataset, vin_dataset, nih_dataset, chexpert_dataset])

            print(f"train len: {len(self.train_dataset)}")
            print(f"val len: {len(self.val_dataset)}")

        elif stage == 'predict':
            if self.cfg["predict_pseudo_label"] == "vinbig":
                print("predicting with vinbig dataset")
                pred_df = pd.read_csv(self.cfg["train_df_path"])
                self.pred_dataset = SingleViewDataset(self.cfg, pred_df, transforms_val)
            elif self.cfg["predict_pseudo_label"] == "nih":
                print("predicting with nih dataset")
                pred_df = pd.read_csv(self.cfg["train_df_path"])
                self.pred_dataset = SingleViewDataset(self.cfg, pred_df, transforms_val)
            elif self.cfg["predict_pseudo_label"] == "chexpert":
                print("predicting with chexpert dataset")
                pred_df = pd.read_csv(self.cfg["train_df_path"])
                self.pred_dataset = SingleViewDataset(self.cfg, pred_df, transforms_val)
            else:
                pred_df = self.test_df
                self.pred_dataset = CaMCheXDataset(self.cfg, pred_df, transforms_val, self.tokenizer)  
                # self.pred_dataset = SingleViewDataset(self.cfg, pred_df, transforms_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_init_args, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_init_args, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, **self.dataloader_init_args, shuffle=False)