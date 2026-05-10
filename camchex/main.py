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


def _apply_cpu_threads(n: int) -> None:
    n = max(1, int(n))
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--cpu_threads",
            type=int,
            default=None,
            help="CPU threads for torch/OMP/MKL. null = half of os.cpu_count().",
        )

    def before_instantiate_classes(self):
        cfg = self.config
        # LightningCLI nests under the subcommand name (fit/validate/...) when subcommands are used.
        sub = getattr(self, "subcommand", None)
        node = cfg[sub] if sub is not None and sub in cfg else cfg
        n = node.get("cpu_threads") if hasattr(node, "get") else getattr(node, "cpu_threads", None)
        if n is None:
            n = max(1, (os.cpu_count() or 2) // 2)
        _apply_cpu_threads(n)


def cli_main():
    torch.set_float32_matmul_precision('high')
    MyLightningCLI(CaMCheX, CaMCheXDataModule, save_config_callback=None)

if __name__ == "__main__":
    cli_main()