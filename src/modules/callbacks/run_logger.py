import json
import math
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import lightning
import numpy
import pandas
import timm
import torch
import torchmetrics
import transformers
from lightning.pytorch.callbacks import Callback


class RunLoggerCallback(Callback):
    def __init__(
        self,
        run_dir: Optional[str] = None,
        grad_norm_every_n_steps: int = 50,
        log_module_grad_norms: bool = True,
        save_git_diff: bool = True,
        save_pip_freeze: bool = True,
    ):
        super().__init__()
        self.run_dir = run_dir
        self.grad_norm_every_n_steps = int(grad_norm_every_n_steps)
        self.log_module_grad_norms = bool(log_module_grad_norms)
        self.save_git_diff = bool(save_git_diff)
        self.save_pip_freeze = bool(save_pip_freeze)

    def setup(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        if self.run_dir is None:
            self.run_dir = trainer.default_root_dir
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)

    def on_fit_start(self, trainer, pl_module) -> None:
        run_dir = Path(self.run_dir or trainer.default_root_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata").mkdir(parents=True, exist_ok=True)

        self._write_json(run_dir / "metadata" / "environment.json", self._environment())
        self._write_text(run_dir / "metadata" / "command.txt", " ".join(sys.argv) + "\n")
        self._write_git_metadata(run_dir / "metadata")

        if self.save_pip_freeze:
            res = self._run([sys.executable, "-m", "pip", "freeze"])
            if res is not None:
                self._write_text(run_dir / "metadata" / "pip-freeze.txt", res)

        if trainer.logger is not None:
            trainer.logger.log_hyperparams({"run/log_every_n_steps": float(self.grad_norm_every_n_steps)})

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if not self.log_module_grad_norms or self.grad_norm_every_n_steps <= 0:
            return

        step = trainer.global_step + 1
        if step % self.grad_norm_every_n_steps != 0:
            return

        param_to_group = self._build_param_group_index(pl_module)
        sums: Dict[str, float] = {}
        total_sq = 0.0
        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                continue
            norm = float(torch.linalg.vector_norm(grad.float(), ord=2).item())
            sq = norm * norm
            total_sq += sq
            group = param_to_group.get(name, self._fallback_group(name))
            sums[group] = sums.get(group, 0.0) + sq

        metrics = {"grad_norm/global": math.sqrt(total_sq)}
        for group, sq in sorted(sums.items()):
            metrics[f"grad_norm/{group}"] = math.sqrt(sq)

        pl_module.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def _build_param_group_index(self, pl_module) -> Dict[str, str]:
        # Map each parameter's qualified name to the nearest ancestor module
        # that declares ``component_name``. Lets RunLogger group grad norms
        # by human-readable component label (e.g. "image_encoder_frontal")
        # rather than by Python attribute path.
        index: Dict[str, str] = {}
        for module_name, module in pl_module.named_modules():
            component_name = getattr(module, "component_name", None)
            if component_name is None:
                continue
            prefix = f"{module_name}." if module_name else ""
            for param_name, _ in module.named_parameters(recurse=True):
                full = f"{prefix}{param_name}"
                index.setdefault(full, component_name)
        return index

    def _fallback_group(self, param_name: str) -> str:
        parts = param_name.split(".")
        if len(parts) >= 2 and parts[0] == "backbone":
            return f"backbone.{parts[1]}"
        return parts[0]

    def _environment(self) -> Dict:
        cuda_devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cuda_devices.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    }
                )

        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": {
                "executable": sys.executable,
                "version": sys.version,
            },
            "cwd": os.getcwd(),
            "versions": {
                "lightning": lightning.__version__,
                "numpy": numpy.__version__,
                "pandas": pandas.__version__,
                "timm": timm.__version__,
                "torch": torch.__version__,
                "torchmetrics": torchmetrics.__version__,
                "transformers": transformers.__version__,
            },
            "torch": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "num_threads": torch.get_num_threads(),
            },
            "cuda_devices": cuda_devices,
        }

    def _write_git_metadata(self, out_dir: Path) -> None:
        commands = {
            "git-commit.txt": ["git", "rev-parse", "HEAD"],
            "git-branch.txt": ["git", "branch", "--show-current"],
            "git-status.txt": ["git", "status", "--short"],
        }
        if self.save_git_diff:
            commands["git-diff.patch"] = ["git", "diff", "--no-ext-diff"]

        for filename, cmd in commands.items():
            res = self._run(cmd)
            if res is not None:
                self._write_text(out_dir / filename, res)

    def _run(self, cmd):
        try:
            res = subprocess.run(cmd, check=False, text=True, capture_output=True)
        except OSError as exc:
            return f"Failed to run {' '.join(cmd)}: {exc}\n"
        text = res.stdout
        if res.stderr:
            text += "\nSTDERR:\n" + res.stderr
        return text

    def _write_json(self, path: Path, data: Dict) -> None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def _write_text(self, path: Path, data: str) -> None:
        path.write_text(data)
