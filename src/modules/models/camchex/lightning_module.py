from typing import Optional

import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torchmetrics import AUROC, AveragePrecision

from src.modules.decoders.ml_decoder import MLDecoder
from src.modules.encoders.biobert_text import BioBertTextEncoder
from src.modules.encoders.timm_image import TimmImageEncoder
from src.modules.models.camchex.camchex import CaMCheXModel
from src.modules.models.camchex.fusion import TransformerFusion
from src.modules.models.camchex.loss import ASL


class CaMCheX(pl.LightningModule):
    """LightningModule for the CaMCheX 26-class multi-label task.

    Owns the assembly: instantiates frontal/lateral image encoders, the text
    encoder, the fusion block, and the MLDecoder head. Each component carries
    a ``component_name`` so RunLoggerCallback groups grad norms per component.
    """

    NUM_CLASSES = 26
    HEAD_IDX = [0, 2, 4, 12, 14, 16, 20, 24]
    MEDIUM_IDX = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
    TAIL_IDX = [7, 11, 17, 18, 19, 21, 23, 25]

    def __init__(
        self,
        lr: float,
        classes: list,
        loss_init_args: dict,
        timm_init_args: dict,
        text_model: str = "dmis-lab/biobert-v1.1",
        frontal_pretrained_path: Optional[str] = None,
        lateral_pretrained_path: Optional[str] = None,
        feature_dim: int = 768,
        num_views: int = 4,
        num_text_streams: int = 2,
        fusion_num_layers: int = 2,
        fusion_nhead: int = 8,
        fusion_dropout: float = 0.1,
        decoder_embedding: int = 768,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.lr = lr
        self.classes = classes
        self.weight_decay = weight_decay

        frontal_encoder = TimmImageEncoder(
            timm_init_args, pretrained_path=frontal_pretrained_path,
            name="image_encoder_frontal",
        )
        lateral_encoder = TimmImageEncoder(
            timm_init_args, pretrained_path=lateral_pretrained_path,
            name="image_encoder_lateral",
        )
        text_encoder = BioBertTextEncoder(model_name=text_model, name="text_encoder")
        fusion = TransformerFusion(
            feature_dim=feature_dim,
            num_views=num_views,
            num_text_streams=num_text_streams,
            num_layers=fusion_num_layers,
            nhead=fusion_nhead,
            dropout=fusion_dropout,
            name="fusion",
        )
        head = MLDecoder(
            num_classes=self.NUM_CLASSES,
            initial_num_features=feature_dim,
            decoder_embedding=decoder_embedding,
            name="ml_decoder",
        )

        self.backbone = CaMCheXModel(
            frontal_encoder=frontal_encoder,
            lateral_encoder=lateral_encoder,
            text_encoder=text_encoder,
            fusion=fusion,
            head=head,
        )

        self.validation_step_outputs = []
        self.val_ap = AveragePrecision(task="binary")
        self.val_auc = AUROC(task="binary")

        self.criterion_cls = ASL(**loss_init_args)

    def forward(self, image):
        return self.backbone(image)

    def shared_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)
        loss = self.criterion_cls(pred, label)
        pred = torch.sigmoid(pred).detach()
        return {"loss": loss, "pred": pred, "label": label}

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log("loss", res["loss"].detach(), prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/loss_step", res["loss"].detach(), prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/loss_epoch", res["loss"].detach(), prog_bar=True, on_step=False, on_epoch=True)
        return res["loss"]

    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log("val/loss_step", res["loss"].detach(), prog_bar=False, on_step=True, on_epoch=False)
        self.validation_step_outputs.append(res)

    def on_validation_epoch_end(self):
        preds = torch.cat([x["pred"] for x in self.validation_step_outputs])
        labels = torch.cat([x["label"] for x in self.validation_step_outputs])
        val_loss = torch.stack([x["loss"].detach() for x in self.validation_step_outputs]).mean()

        val_ap, val_auroc = [], []
        class_metrics = {}
        for i in range(self.NUM_CLASSES):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            class_metrics[f"val/ap/{self.classes[i]}"] = ap
            class_metrics[f"val/auroc/{self.classes[i]}"] = auroc
            print(f"{self.classes[i]}_ap: {ap}")

        n = float(self.NUM_CLASSES)
        summary_metrics = {
            "val_loss": val_loss,
            "val/loss": val_loss,
            "val_ap": sum(val_ap) / n,
            "val_auroc": sum(val_auroc) / n,
            "val_head_ap": sum(val_ap[i] for i in self.HEAD_IDX) / len(self.HEAD_IDX),
            "val_medium_ap": sum(val_ap[i] for i in self.MEDIUM_IDX) / len(self.MEDIUM_IDX),
            "val_tail_ap": sum(val_ap[i] for i in self.TAIL_IDX) / len(self.TAIL_IDX),
            "val/ap": sum(val_ap) / n,
            "val/auroc": sum(val_auroc) / n,
            "val/ap_head": sum(val_ap[i] for i in self.HEAD_IDX) / len(self.HEAD_IDX),
            "val/ap_medium": sum(val_ap[i] for i in self.MEDIUM_IDX) / len(self.MEDIUM_IDX),
            "val/ap_tail": sum(val_ap[i] for i in self.TAIL_IDX) / len(self.TAIL_IDX),
        }
        self.log_dict(class_metrics, prog_bar=False, on_step=False, on_epoch=True)
        self.log_dict(summary_metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        data, _label = batch

        if isinstance(data, torch.Tensor):  # SingleViewDataset
            study_id = None
            image = data
            pred = torch.sigmoid(self(image)).detach()
            pred_flip = torch.sigmoid(self(image.flip(-1))).detach()
            pred = (pred + pred_flip) / 2
        else:  # CaMCheXDataset
            study_id = data[0]
            pred = torch.sigmoid(self(data)).detach()
            image_flip = data[1].flip(-1)
            inputs_flip = (data[0], image_flip) + tuple(data[2:])
            pred_flip = torch.sigmoid(self(inputs_flip)).detach()
            pred = (pred + pred_flip) / 2

        return {"study_id": study_id, "pred": pred}

    def configure_optimizers(self):
        base_lr = self.lr
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n for k in ["bias", "norm", "LayerNorm", "bn", "running_mean", "running_var"]):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay, "lr": base_lr},
                {"params": no_decay, "weight_decay": 0.0, "lr": base_lr},
            ],
            betas=(0.9, 0.999), eps=1e-8,
        )

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(1, total_steps // max(1, self.trainer.max_epochs))
        warmup_steps = max(100, int(0.05 * steps_per_epoch))

        warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine_r = CosineAnnealingWarmRestarts(
            optimizer, T_0=steps_per_epoch, T_mult=1, eta_min=base_lr * 0.1,
        )
        sched = SequentialLR(optimizer, [warmup, cosine_r], milestones=[warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }
