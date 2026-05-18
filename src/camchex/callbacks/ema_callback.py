# callbacks/ema_callback.py
from typing import Dict, Iterator, Tuple, Optional
import torch
from torch import nn
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.9999,
        device: Optional[str] = None,
        use_buffers: bool = False,             
        start_after_steps: Optional[int] = None,
        warmup_ratio_of_epoch: float = 0.05,
        min_warmup_steps: int = 100,
    ):
        super().__init__()
        self.decay = float(decay)
        self.device = device
        self.use_buffers = bool(use_buffers)
        self.user_start_after_steps = start_after_steps
        self.warmup_ratio_of_epoch = float(warmup_ratio_of_epoch)
        self.min_warmup_steps = int(min_warmup_steps)

        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._step: int = 0
        self._start_after_steps: int = 0

    def _target_device(self, pl_module: nn.Module) -> torch.device:
        return pl_module.device if self.device is None else torch.device(self.device)

    def _named_tensors(self, pl_module: nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
        for n, p in pl_module.named_parameters():
            if p.requires_grad:
                yield n, p
        if self.use_buffers:
            for n, b in pl_module.named_buffers():
                yield f"[buffer]{n}", b

    def _ensure_shadow_on_target(self, pl_module: nn.Module) -> None:
        tgt = self._target_device(pl_module)
        for n, t in self._named_tensors(pl_module):
            s = self.shadow.get(n)
            if s is None or s.shape != t.shape:
                self.shadow[n] = t.detach().clone().to(device=tgt, dtype=t.dtype)
            else:
                if s.device != tgt or s.dtype != t.dtype:
                    self.shadow[n] = s.to(device=tgt, dtype=t.dtype)

    @torch.no_grad()
    def _ema_update(self, pl_module: nn.Module) -> None:
        tgt = self._target_device(pl_module)
        for n, t in self._named_tensors(pl_module):
            s = self.shadow.get(n)
            if s is None or s.shape != t.shape:
                self.shadow[n] = t.detach().clone().to(device=tgt, dtype=t.dtype)
                continue
            if not t.is_floating_point():
                s.data.copy_(t.detach().to(device=s.device, dtype=s.dtype))
                continue
            if s.device != t.device or s.dtype != t.dtype:
                t_cur = t.detach().to(device=s.device, dtype=s.dtype)
            else:
                t_cur = t.detach()
            s.mul_(self.decay).add_(t_cur, alpha=1.0 - self.decay)

    def _apply_shadow(self, pl_module: nn.Module) -> None:
        self.backup.clear()
        for n, t in self._named_tensors(pl_module):
            s = self.shadow.get(n)
            if s is None or s.shape != t.shape:
                continue
            self.backup[n] = t.detach().clone()
            t.data.copy_(s.data.to(device=t.device, dtype=t.dtype))

    def _restore(self, pl_module: nn.Module) -> None:
        if not self.backup:
            return
        for n, t in self._named_tensors(pl_module):
            if n in self.backup:
                t.data.copy_(self.backup[n].data.to(device=t.device, dtype=t.dtype))
        self.backup.clear()

    def on_fit_start(self, trainer, pl_module) -> None:
        total_steps_all = trainer.estimated_stepping_batches or 0
        max_epochs = max(1, trainer.max_epochs or 1)
        steps_per_epoch = max(1, total_steps_all // max_epochs)
        warmup_steps = max(self.min_warmup_steps, int(self.warmup_ratio_of_epoch * steps_per_epoch))
        self._start_after_steps = (
            int(self.user_start_after_steps) if self.user_start_after_steps is not None else int(warmup_steps)
        )
        self._ensure_shadow_on_target(pl_module)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._step += 1
        if self._step < self._start_after_steps:
            return
        self._ensure_shadow_on_target(pl_module)
        self._ema_update(pl_module)

    def on_validation_start(self, trainer, pl_module) -> None:
        self._ensure_shadow_on_target(pl_module)
        self._apply_shadow(pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._restore(pl_module)

    def on_test_start(self, trainer, pl_module) -> None:
        self._ensure_shadow_on_target(pl_module)
        self._apply_shadow(pl_module)

    def on_test_end(self, trainer, pl_module) -> None:
        self._restore(pl_module)

    def on_predict_start(self, trainer, pl_module) -> None:
        self._ensure_shadow_on_target(pl_module)
        self._apply_shadow(pl_module)

    def on_predict_end(self, trainer, pl_module) -> None:
        self._restore(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict) -> None:
        checkpoint["ema_state_dict"] = {k: v.detach().cpu() for k, v in self.shadow.items()}
        checkpoint["ema_decay"] = self.decay
        checkpoint["ema_step"] = self._step
        checkpoint["ema_start_after_steps"] = self._start_after_steps
        checkpoint["ema_use_buffers"] = self.use_buffers
        checkpoint["ema_device"] = self.device

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: Dict) -> None:
        sd = checkpoint.get("ema_state_dict", None)
        if sd is not None:
            self.shadow = {k: v for k, v in sd.items()}
        self.decay = float(checkpoint.get("ema_decay", self.decay))
        self._step = int(checkpoint.get("ema_step", self._step))
        self._start_after_steps = int(checkpoint.get("ema_start_after_steps", self._start_after_steps))
        self.use_buffers = bool(checkpoint.get("ema_use_buffers", self.use_buffers))
        self.device = checkpoint.get("ema_device", self.device)