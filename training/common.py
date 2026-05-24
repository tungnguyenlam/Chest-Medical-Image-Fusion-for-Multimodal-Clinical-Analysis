from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchmetrics import AveragePrecision, AUROC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.CaMCheXDataset import CaMCheXDataset
from src.dataloader.SingleViewDataset import SingleViewDataset
from src.dataloader.utils import get_transforms
from src.loss.AsymetricLoss import AsymetricLoss
from src.optimizer import build_adamw_optimizer
from src.scheduler import build_warmup_cosine_scheduler


VIEW_ALIASES = {
    "frontal": {"AP", "PA", "FRONTAL"},
    "lateral": {"LATERAL", "LL"},
}


class MultiLabelModule(pl.LightningModule):
    def __init__(self, model, lr: float, classes: list[str], loss_init_args: dict[str, Any]):
        super().__init__()
        self.model = model
        self.lr = lr
        self.classes = classes
        self.criterion = AsymetricLoss(**loss_init_args)
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []
        self.val_ap = AveragePrecision(task="binary")
        self.val_auc = AUROC(task="binary")

    def forward(self, data):
        return self.model(data)

    def _shared_step(self, batch):
        data, label = batch
        label = label.float()
        pred = self(data)
        loss = self.criterion(pred, label)
        return {"loss": loss, "pred": torch.sigmoid(pred.detach()), "label": label.detach()}

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.log("loss", out["loss"], prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/loss", out["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.validation_step_outputs.append(out)
        self.log("val/loss_step", out["loss"], prog_bar=False, on_step=True, on_epoch=False)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        preds = torch.cat([x["pred"] for x in self.validation_step_outputs])
        labels = torch.cat([x["label"] for x in self.validation_step_outputs])
        val_loss = torch.stack([x["loss"].detach() for x in self.validation_step_outputs]).mean()
        metrics = compute_metrics(preds, labels, self.classes)
        metrics["val/loss"] = val_loss
        metrics["val_loss"] = val_loss
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = build_adamw_optimizer(self.parameters(), lr=self.lr)
        total_steps = max(1, int(getattr(self.trainer, "estimated_stepping_batches", 1)))
        steps_per_epoch = max(1, total_steps // max(1, self.trainer.max_epochs))
        warmup_steps = max(1, min(int(0.05 * steps_per_epoch), total_steps))
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            lr=self.lr,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=warmup_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def add_common_args(parser: argparse.ArgumentParser, model_name: str) -> None:
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--train-df-path")
    parser.add_argument("--val-df-path")
    parser.add_argument("--test-df-path")
    parser.add_argument("--output-dir", default=f"output/{model_name}/runs")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument("--val-check-interval", type=float)
    parser.add_argument("--backbone-name")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    candidate = ROOT / p
    if candidate.exists():
        return candidate
    if str(path).startswith("../"):
        return ROOT / str(path)[3:]
    return ROOT / p


def make_run_dir(base_dir: str | Path, run_name: str, run_id: str | None) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve_path(base_dir) or Path(base_dir)
    run_dir = run_dir / f"{run_id}-{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(run_dir: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    payload = {"args": vars(args), "config": cfg}
    with open(run_dir / "config.resolved.json", "w") as f:
        json.dump(payload, f, indent=2)


def timm_args_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    timm_args = dict(cfg["model"]["timm_init_args"])
    if args.backbone_name:
        timm_args["model_name"] = args.backbone_name
    if args.no_pretrained:
        timm_args["pretrained"] = False
    return timm_args


def classes_from_config(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["model"]["classes"])


def loss_args_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["model"]["loss_init_args"])


def lr_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> float:
    return args.lr if args.lr is not None else float(cfg["model"]["lr"])


def data_cfg_from_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data_cfg = dict(cfg["data"]["datamodule_cfg"])
    data_cfg["classes"] = classes_from_config(cfg)
    if args.image_size is not None:
        data_cfg["size"] = args.image_size
    if args.train_df_path:
        data_cfg["train_df_path"] = args.train_df_path
    if args.val_df_path:
        data_cfg["devel_df_path"] = args.val_df_path
    if args.test_df_path:
        data_cfg["pred_df_path"] = args.test_df_path
    return data_cfg


def dataloader_args_from_config(cfg: dict[str, Any], args: argparse.Namespace, shuffle: bool) -> dict[str, Any]:
    dl_args = dict(cfg["data"]["dataloader_init_args"])
    if args.batch_size is not None:
        dl_args["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dl_args["num_workers"] = args.num_workers
    if dl_args.get("num_workers", 0) == 0:
        dl_args["persistent_workers"] = False
    dl_args["shuffle"] = shuffle
    return dl_args


def read_dataframe(path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(path)
    if resolved is None:
        raise FileNotFoundError(path)
    return pd.read_csv(resolved)


def filter_single_view(df: pd.DataFrame, view_position: str) -> pd.DataFrame:
    if view_position == "all" or "ViewPosition" not in df.columns:
        return df.reset_index(drop=True)
    valid = VIEW_ALIASES[view_position]
    mask = df["ViewPosition"].fillna("").astype(str).str.upper().isin(valid)
    return df.loc[mask].reset_index(drop=True)


def make_single_view_loaders(cfg: dict[str, Any], args: argparse.Namespace, view_position: str):
    data_cfg = data_cfg_from_config(cfg, args)
    transforms_train, transforms_val = get_transforms(data_cfg["size"])
    train_df = filter_single_view(read_dataframe(data_cfg["train_df_path"]), view_position)
    val_df = filter_single_view(read_dataframe(data_cfg["devel_df_path"]), view_position)
    train_ds = SingleViewDataset(data_cfg, train_df, transforms_train)
    val_ds = SingleViewDataset(data_cfg, val_df, transforms_val)
    train_loader = DataLoader(train_ds, **dataloader_args_from_config(cfg, args, shuffle=True))
    val_loader = DataLoader(val_ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    return train_loader, val_loader


def make_camchex_loaders(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    transforms_train, transforms_val = get_transforms(data_cfg["size"])
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.get("tokenizer") or "dmis-lab/biobert-v1.1")
    train_ds = CaMCheXDataset(data_cfg, read_dataframe(data_cfg["train_df_path"]), transforms_train, tokenizer)
    val_ds = CaMCheXDataset(data_cfg, read_dataframe(data_cfg["devel_df_path"]), transforms_val, tokenizer)
    train_loader = DataLoader(train_ds, **dataloader_args_from_config(cfg, args, shuffle=True))
    val_loader = DataLoader(val_ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    return train_loader, val_loader


def make_single_view_eval_loader(cfg: dict[str, Any], args: argparse.Namespace, view_position: str):
    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"])
    df = filter_single_view(read_dataframe(data_cfg["pred_df_path"]), view_position)
    ds = SingleViewDataset(data_cfg, df, transforms_val)
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    ids = df["path"].tolist() if "path" in df.columns else list(range(len(df)))
    labels_available = all(c in df.columns for c in data_cfg["classes"])
    return loader, ids, labels_available


def make_camchex_eval_loader(cfg: dict[str, Any], args: argparse.Namespace):
    from transformers import AutoTokenizer

    data_cfg = data_cfg_from_config(cfg, args)
    _, transforms_val = get_transforms(data_cfg["size"])
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.get("tokenizer") or "dmis-lab/biobert-v1.1")
    df = read_dataframe(data_cfg["pred_df_path"])
    ds = CaMCheXDataset(data_cfg, df, transforms_val, tokenizer)
    loader = DataLoader(ds, **dataloader_args_from_config(cfg, args, shuffle=False))
    labels_available = all(c in df.columns for c in data_cfg["classes"])
    return loader, labels_available


def trainer_from_args(args: argparse.Namespace, run_dir: Path) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="last", save_last=True, save_top_k=0),
        ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="{epoch:02d}-{val_ap:.5f}", monitor="val_ap", mode="max", save_top_k=1),
    ]
    trainer_kwargs = {
        "logger": CSVLogger(save_dir=str(run_dir), name="logs"),
        "callbacks": callbacks,
        "max_epochs": args.max_epochs if args.max_epochs is not None else 1000,
        "fast_dev_run": args.fast_dev_run,
        "accelerator": args.accelerator or "auto",
        "devices": args.devices or "auto",
        "precision": args.precision or "16-mixed",
        "accumulate_grad_batches": args.accumulate_grad_batches or 1,
        "log_every_n_steps": 50,
        "num_sanity_val_steps": 2,
    }
    if args.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = args.val_check_interval
    return pl.Trainer(**trainer_kwargs)


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> dict[str, torch.Tensor]:
    val_ap = []
    val_auc = []
    metrics = {}
    for idx, name in enumerate(classes):
        ap = AveragePrecision(task="binary").to(preds.device)(preds[:, idx], labels[:, idx].long())
        auc = AUROC(task="binary").to(preds.device)(preds[:, idx], labels[:, idx].long())
        val_ap.append(ap)
        val_auc.append(auc)
        metrics[f"val/ap/{name}"] = ap
        metrics[f"val/auroc/{name}"] = auc

    head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
    medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
    tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]
    metrics.update(
        {
            "val_ap": sum(val_ap) / len(val_ap),
            "val_auroc": sum(val_auc) / len(val_auc),
            "val/ap": sum(val_ap) / len(val_ap),
            "val/auroc": sum(val_auc) / len(val_auc),
            "val/ap_head": sum(val_ap[i] for i in head_idx) / len(head_idx),
            "val/ap_medium": sum(val_ap[i] for i in medium_idx) / len(medium_idx),
            "val/ap_tail": sum(val_ap[i] for i in tail_idx) / len(tail_idx),
        }
    )
    return metrics


def load_weights(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    candidates = [
        state_dict,
        {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")},
        {k.removeprefix("model.model."): v for k, v in state_dict.items() if k.startswith("model.model.")},
    ]
    errors = []
    model_keys = set(model.state_dict().keys())
    for candidate in candidates:
        if not model_keys.intersection(candidate.keys()):
            continue
        try:
            model.load_state_dict(candidate, strict=False)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "no checkpoint keys matched the model"
    raise RuntimeError(f"Could not load checkpoint {checkpoint_path}: {detail}")


def save_single_view_encoder(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.model.state_dict(), path)


def select_device(requested: str | None = None) -> torch.device:
    if requested and requested not in {"auto", "gpu"}:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


@torch.inference_mode()
def predict_dataframe(model, loader, classes: list[str], device: torch.device, ids: list[Any] | None = None) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    model.to(device)
    model.eval()
    preds = []
    labels = []
    batch_ids = []
    for batch in loader:
        data, label = batch
        if isinstance(data, (tuple, list)) and data and not torch.is_tensor(data[0]):
            batch_ids.extend(list(data[0]))
        data = move_to_device(data, device)
        pred = torch.sigmoid(model(data)).cpu()
        preds.append(pred)
        labels.append(label.cpu().float())

    pred_tensor = torch.cat(preds)
    label_tensor = torch.cat(labels)
    out = pd.DataFrame(pred_tensor.numpy(), columns=[f"pred_{c}" for c in classes])
    if batch_ids:
        out.insert(0, "study_id", batch_ids)
    elif ids is not None:
        out.insert(0, "sample_id", ids[: len(out)])
    for idx, name in enumerate(classes):
        out[f"label_{name}"] = label_tensor[:, idx].numpy()
    return out, pred_tensor, label_tensor
