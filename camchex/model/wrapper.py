import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from model.models import SingleViewModel, CaMCheXModel
from model.loss import ASL
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR

class CaMCheX(pl.LightningModule):
    def __init__(self, lr, classes, loss_init_args, timm_init_args):
        super(CaMCheX, self).__init__()
        self.lr = lr
        self.classes = classes
        
        # self.backbone = SingleViewModel(timm_init_args)
        self.backbone = CaMCheXModel(timm_init_args)
        
        self.validation_step_outputs = []
        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")
        
        self.criterion_cls = ASL(**loss_init_args)

    def forward(self, image):
        return self.backbone(image)
    
    def shared_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)

        loss = self.criterion_cls(pred, label)

        pred=torch.sigmoid(pred).detach()

        return dict(
            loss=loss,
            pred=pred,
            label=label,
        )

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'loss': res['loss'].detach()}, prog_bar=True)
        self.log_dict({'train_loss': res['loss'].detach()}, prog_bar=True, on_step=False, on_epoch=True)
        return res['loss']
        
    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'val_loss': res['loss'].detach()}, prog_bar=True)
        self.validation_step_outputs.append(res)

    def on_validation_epoch_end(self):
        preds = torch.cat([x['pred'] for x in self.validation_step_outputs])
        labels = torch.cat([x['label'] for x in self.validation_step_outputs])

        val_ap = []
        val_auroc = []
        for i in range(26):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            print(f'{self.classes[i]}_ap: {ap}')
        
        head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]

        self.log_dict({'val_ap': sum(val_ap)/26}, prog_bar=True)
        self.log_dict({'val_auroc': sum(val_auroc)/26}, prog_bar=True)
        self.log_dict({'val_head_ap': sum([val_ap[i] for i in head_idx]) / len(head_idx)}, prog_bar=True)
        self.log_dict({'val_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx)}, prog_bar=True)
        self.log_dict({'val_tail_ap': sum([val_ap[i] for i in tail_idx]) / len(tail_idx)}, prog_bar=True)
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        data, label = batch
        
        if isinstance(data, torch.Tensor): # SingleViewDataset
            study_id = None
            image = data
            pred = torch.sigmoid(self(image)).detach()

            image_flip = image.flip(-1)
            pred_flip = torch.sigmoid(self(image_flip)).detach()
            pred = (pred + pred_flip) / 2
        
        else: # CaMCheXDataset
            study_id = data[0]
            pred = torch.sigmoid(self(data)).detach()

            image_flip = data[1].flip(-1)
            inputs_flip = (data[0], image_flip) + tuple(data[2:])
            pred_flip = torch.sigmoid(self(inputs_flip)).detach()
            pred = (pred + pred_flip) / 2

        return {"study_id": study_id, "pred": pred}

    def configure_optimizers(self):
        wd = 0.01
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
                {"params": decay,    "weight_decay": wd,  "lr": base_lr},
                {"params": no_decay, "weight_decay": 0.0, "lr": base_lr},
            ],
            betas=(0.9, 0.999), eps=1e-8
        )

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(1, total_steps // max(1, self.trainer.max_epochs))
        warmup_steps = max(100, int(0.05 * steps_per_epoch))

        warmup  = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosineR = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch,  
            T_mult=1,
            eta_min=base_lr * 0.1  
        )

        sched = SequentialLR(optimizer, [warmup, cosineR], milestones=[warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }