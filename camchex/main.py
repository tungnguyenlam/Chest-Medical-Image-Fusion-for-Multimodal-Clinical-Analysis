import os
import sys

# All config paths (../data/...) and CSV image paths (images/p1x/...) are
# relative to this directory. Resolve any --config paths against the original
# cwd first, then chdir here so the command works regardless of where it was
# invoked from.
_ORIG_CWD = os.getcwd()
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        p = sys.argv[i + 1]
        if not os.path.isabs(p):
            sys.argv[i + 1] = os.path.abspath(os.path.join(_ORIG_CWD, p))
    elif arg.startswith("--config="):
        p = arg[len("--config="):]
        if p and not os.path.isabs(p):
            sys.argv[i] = "--config=" + os.path.abspath(os.path.join(_ORIG_CWD, p))
os.chdir(_SCRIPT_DIR)

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger
from model.wrapper import CaMCheX
from dataset.datamodule import CaMCheXDataModule

class MyLightningCLI(LightningCLI):
    def before_fit(self):
        # Optionally override things here
        pass

def cli_main():
    torch.set_float32_matmul_precision('high')
    MyLightningCLI(CaMCheX, CaMCheXDataModule, save_config_callback=None)

if __name__ == "__main__":
    cli_main()