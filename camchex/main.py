import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

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
import yaml
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger
from model.wrapper import CaMCheX
from dataset.datamodule import CaMCheXDataModule


def _apply_cpu_threads(n: int) -> None:
    n = max(1, int(n))
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _cfg_set(node, key, value):
    if isinstance(node, dict):
        node[key] = value
    else:
        setattr(node, key, value)


def _ensure_child(node, key):
    child = _cfg_get(node, key)
    if child is None:
        child = {}
        _cfg_set(node, key, child)
    return child


def _plain_cfg(obj):
    if hasattr(obj, "as_dict"):
        return _plain_cfg(obj.as_dict())
    if isinstance(obj, dict):
        return {k: _plain_cfg(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain_cfg(v) for v in obj]
    return obj


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def _unique_run_dir(output_root: str, name: str, run_id: Optional[str]) -> str:
    root = Path(output_root)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = root / f"{run_id}-{_slugify(name)}"
    path = base
    i = 2
    while path.exists():
        path = Path(f"{base}-{i}")
        i += 1
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


def _update_class_path_callback(callbacks, class_path: str, init_updates: dict):
    if callbacks is None:
        return
    for cb in callbacks:
        if _cfg_get(cb, "class_path") != class_path:
            continue
        init_args = _ensure_child(cb, "init_args")
        for key, value in init_updates.items():
            _cfg_set(init_args, key, value)


def _configure_run_paths(cfg_node):
    run_cfg = _ensure_child(cfg_node, "run")
    model_name = _slugify(str(_cfg_get(run_cfg, "model_name", "camchex")))
    run_name = _cfg_get(run_cfg, "name", "baseline")
    output_root = _cfg_get(run_cfg, "output_root", None)
    if output_root is None:
        output_root = f"../output/{model_name}/runs"
    run_id = _cfg_get(run_cfg, "id", None)
    log_every_n_steps = int(_cfg_get(run_cfg, "log_every_n_steps", 50))
    run_dir = _unique_run_dir(output_root, run_name, run_id)

    _cfg_set(run_cfg, "model_name", model_name)
    _cfg_set(run_cfg, "output_root", output_root)
    _cfg_set(run_cfg, "dir", run_dir)

    trainer_cfg = _ensure_child(cfg_node, "trainer")
    _cfg_set(trainer_cfg, "default_root_dir", run_dir)
    if _cfg_get(trainer_cfg, "log_every_n_steps", None) is None:
        _cfg_set(trainer_cfg, "log_every_n_steps", log_every_n_steps)

    loggers = _cfg_get(trainer_cfg, "logger", None)
    if loggers is not None:
        for logger_cfg in loggers:
            if _cfg_get(logger_cfg, "class_path") != "lightning.pytorch.loggers.CSVLogger":
                continue
            init_args = _ensure_child(logger_cfg, "init_args")
            _cfg_set(init_args, "save_dir", run_dir)
            _cfg_set(init_args, "name", "logs")
            _cfg_set(init_args, "version", "")

    callbacks = _cfg_get(trainer_cfg, "callbacks", None)
    checkpoint_dir = str(Path(run_dir) / "checkpoints")
    _update_class_path_callback(
        callbacks,
        "lightning.pytorch.callbacks.ModelCheckpoint",
        {"dirpath": checkpoint_dir},
    )
    _update_class_path_callback(
        callbacks,
        "callbacks.prediction_callback.PredictionWriter",
        {"output_path": str(Path(run_dir) / "predictions" / "predictions.csv")},
    )
    _update_class_path_callback(
        callbacks,
        "callbacks.run_logger.RunLoggerCallback",
        {
            "run_dir": run_dir,
            "grad_norm_every_n_steps": log_every_n_steps,
            "log_module_grad_norms": bool(_cfg_get(run_cfg, "log_module_grad_norms", True)),
            "save_git_diff": bool(_cfg_get(run_cfg, "save_git_diff", True)),
            "save_pip_freeze": bool(_cfg_get(run_cfg, "save_pip_freeze", True)),
        },
    )

    data_cfg = _cfg_get(cfg_node, "data", None)
    data_module_cfg = _cfg_get(data_cfg, "datamodule_cfg", None)
    if data_module_cfg is not None:
        _cfg_set(data_module_cfg, "save_dir", run_dir)

    Path(run_dir, "metadata").mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(_plain_cfg(cfg_node), f, sort_keys=False)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--cpu_threads",
            type=int,
            default=None,
            help="CPU threads for torch/OMP/MKL. null = half of os.cpu_count().",
        )
        parser.add_argument("--run.name", default="baseline")
        parser.add_argument("--run.model_name", default="camchex")
        parser.add_argument("--run.output_root", default=None)
        parser.add_argument("--run.id", default=None)
        parser.add_argument("--run.dir", default=None)
        parser.add_argument("--run.log_every_n_steps", type=int, default=50)
        parser.add_argument("--run.log_module_grad_norms", type=bool, default=True)
        parser.add_argument("--run.save_git_diff", type=bool, default=True)
        parser.add_argument("--run.save_pip_freeze", type=bool, default=True)

    def before_instantiate_classes(self):
        cfg = self.config
        # LightningCLI nests under the subcommand name (fit/validate/...) when subcommands are used.
        sub = getattr(self, "subcommand", None)
        node = cfg[sub] if sub is not None and sub in cfg else cfg
        n = node.get("cpu_threads") if hasattr(node, "get") else getattr(node, "cpu_threads", None)
        if n is None:
            n = max(1, (os.cpu_count() or 2) // 2)
        _apply_cpu_threads(n)
        _configure_run_paths(node)


def cli_main():
    torch.set_float32_matmul_precision('high')
    MyLightningCLI(CaMCheX, CaMCheXDataModule, save_config_callback=None)

if __name__ == "__main__":
    cli_main()
