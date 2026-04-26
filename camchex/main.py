import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger
from model.wrapper import CaMCheX
from dataset.datamodule import CaMCheXDataModule
import sys
import os

class MyLightningCLI(LightningCLI):
    def before_fit(self):
        # Optionally override things here
        pass

def cli_main():
    torch.set_float32_matmul_precision('high')
    MyLightningCLI(CaMCheX, CaMCheXDataModule, save_config_callback=None)

if __name__ == "__main__":
    cli_main()