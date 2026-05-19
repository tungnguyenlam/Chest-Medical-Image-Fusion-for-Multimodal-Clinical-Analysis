from pathlib import Path
from typing import Optional, Sequence

import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.modules.dataloaders.mimic_multiview_dataset import MimicMultiViewDataset
from src.modules.dataloaders.transforms import get_transforms


class MimicMultiViewDataModule(pl.LightningDataModule):
    """DataModule that loads three CSVs (train/val/test) and wraps them with
    MimicMultiViewDataset. All image paths in the CSVs are resolved against
    ``image_root`` (typically ``data/<subset_name>/MIMIC-CXR-JPG/files``).
    """

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        image_root: str,
        classes: Sequence[str],
        image_size: int,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        text_model: str = "dmis-lab/biobert-v1.1",
        max_views: int = 4,
        clinical_max_length: int = 384,
        obs_max_length: int = 128,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.image_root = str(Path(image_root))
        self.classes = list(classes)
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.text_model = text_model
        self.max_views = int(max_views)
        self.clinical_max_length = int(clinical_max_length)
        self.obs_max_length = int(obs_max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(text_model)

        self.train_dataset: Optional[MimicMultiViewDataset] = None
        self.val_dataset: Optional[MimicMultiViewDataset] = None
        self.pred_dataset: Optional[MimicMultiViewDataset] = None

    def _make_dataset(self, df: pd.DataFrame, transform) -> MimicMultiViewDataset:
        return MimicMultiViewDataset(
            df=df,
            classes=self.classes,
            image_size=self.image_size,
            transform=transform,
            tokenizer=self.tokenizer,
            image_root=self.image_root,
            max_views=self.max_views,
            clinical_max_length=self.clinical_max_length,
            obs_max_length=self.obs_max_length,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        transforms_train, transforms_val = get_transforms(self.image_size)

        if stage in (None, "fit", "validate"):
            train_df = pd.read_csv(self.train_csv)
            val_df = pd.read_csv(self.val_csv)
            self.train_dataset = self._make_dataset(train_df, transforms_train)
            self.val_dataset = self._make_dataset(val_df, transforms_val)
            print(f"train studies: {len(self.train_dataset)}")
            print(f"val studies:   {len(self.val_dataset)}")

        if stage in (None, "predict", "test"):
            pred_df = pd.read_csv(self.test_csv)
            self.pred_dataset = self._make_dataset(pred_df, transforms_val)
            print(f"pred studies:  {len(self.pred_dataset)}")

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.pred_dataset, shuffle=False)
