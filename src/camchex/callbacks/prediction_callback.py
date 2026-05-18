import pandas as pd
import numpy as np
import torch
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter
from torchmetrics import AveragePrecision, AUROC

class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_path, pred_df_path, write_interval="epoch", prediction_level="study"):
        super().__init__(write_interval=write_interval)
        self.output_path = output_path
        self.pred_df_path = pred_df_path
        self.prediction_level = prediction_level.lower().strip()

        self.class_names = [
            "Atelectasis","Calcification of the Aorta","Cardiomegaly","Consolidation",
            "Edema","Emphysema","Enlarged Cardiomediastinum","Fibrosis","Fracture",
            "Hernia","Infiltration","Lung Lesion","Lung Opacity","Mass","No Finding",
            "Nodule","Pleural Effusion","Pleural Other","Pleural Thickening",
            "Pneumomediastinum","Pneumonia","Pneumoperitoneum","Pneumothorax",
            "Subcutaneous Emphysema","Support Devices","Tortuous Aorta"
        ]

        self.head_idx   = [0, 2, 4, 12, 14, 16, 20, 24]
        self.medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        self.tail_idx   = [7, 11, 17, 18, 19, 21, 23, 25]

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        flat = []
        for b in predictions:
            flat.extend(b if isinstance(b, (list, tuple)) else [b])

        preds_list, study_ids_list = [], []
        for item in flat:
            if isinstance(item, dict):
                sid = item.get("study_id", None)
                pred = item["pred"]
            else:
                sid = None
                pred = item

            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            preds_list.append(pred.detach().float().cpu())

            if sid is None:
                sid_arr = np.array([None] * pred.size(0))
            elif isinstance(sid, torch.Tensor):
                sid_arr = sid.detach().cpu().numpy().reshape(-1)
            else:
                sid_arr = np.asarray(sid)
                if sid_arr.ndim == 0:
                    sid_arr = sid_arr.reshape(1)

            if sid_arr.size == 1:
                sid_arr = np.repeat(sid_arr, pred.size(0))
            elif sid_arr.size != pred.size(0):
                raise ValueError(
                    f"study_id length {sid_arr.size} doesn't match batch size {pred.size(0)}"
                )

            study_ids_list.append(sid_arr)

        preds = torch.cat(preds_list, dim=0)

        df_img = pd.read_csv(self.pred_df_path)
        labels_img = torch.tensor(
            df_img[self.class_names].values, dtype=torch.float32, device=preds.device
        )

        if self.prediction_level == "image":
            preds_img = preds
            self._per_class_metrics(preds_img, labels_img, tag="Image-level")
            self._save_image_level_csv(preds_img, df_img)
            return

        study_ids = np.concatenate(study_ids_list, axis=0).astype("int64")
        if preds.shape[0] == len(df_img):
            preds_img = preds
        else:
            preds_img = self._broadcast_study_preds_to_images(preds, study_ids, df_img)

        self._per_class_metrics(preds_img, labels_img, tag="Image-level")
        self._save_image_level_csv(preds_img, df_img)

        sid_series = df_img["study_id"].to_numpy()
        uniq_sids = np.unique(sid_series)
        preds_study = torch.cat(
            [preds_img[sid_series == sid].mean(dim=0, keepdim=True) for sid in uniq_sids], dim=0
        )
        labels_study = torch.cat(
            [labels_img[sid_series == sid].mean(dim=0, keepdim=True) for sid in uniq_sids], dim=0
        )
        self._per_class_metrics(preds_study, labels_study, tag="Study-level")

    def _save_image_level_csv(self, preds_img: torch.Tensor, df_img: pd.DataFrame):
        out_df = pd.DataFrame()
        if "path" in df_img.columns:
            out_df["path"] = df_img["path"].astype(str)
        if "study_id" in df_img.columns:
            out_df["study_id"] = df_img["study_id"].astype("int64", errors="ignore")
        for i, name in enumerate(self.class_names):
            out_df[name] = preds_img[:, i].detach().cpu().numpy()

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
        print(f"Saved predictions to {self.output_path}")

    def _per_class_metrics(self, preds_t: torch.Tensor, labels_t: torch.Tensor, tag: str):
        val_ap = AveragePrecision(task="binary")
        val_auc = AUROC(task="binary")
        ap_scores, auc_scores = [], []

        print(f"\n=== {tag} metrics ===")
        for i, cname in enumerate(self.class_names):
            ap = val_ap(preds_t[:, i], labels_t[:, i].int()).item()
            auc = val_auc(preds_t[:, i], labels_t[:, i].int()).item()
            ap_scores.append(ap)
            auc_scores.append(auc)
            print(f"{cname:<25} AP: {ap:.4f}, AUROC: {auc:.4f}")

        ap_scores = np.asarray(ap_scores)
        auc_scores = np.asarray(auc_scores)

        print(f"\n--- Summary ({tag}) ---")
        print(f"Total mAP:   {ap_scores.mean():.4f}")
        print(f"Total AUROC: {auc_scores.mean():.4f}")
        print(f"Head mAP:    {ap_scores[self.head_idx].mean():.4f}")
        print(f"Medium mAP:  {ap_scores[self.medium_idx].mean():.4f}")
        print(f"Tail mAP:    {ap_scores[self.tail_idx].mean():.4f}")

    def _broadcast_study_preds_to_images(self, preds: torch.Tensor, study_ids: np.ndarray, df_img: pd.DataFrame):
        C = preds.shape[1]
        pred_df = pd.DataFrame(preds.detach().cpu().numpy(), columns=[f"pred_{i}" for i in range(C)])
        pred_df.insert(0, "study_id", study_ids.astype("int64"))
        pred_df = pred_df.groupby("study_id", as_index=False).mean()

        df_img = df_img.copy()
        df_img["study_id"] = df_img["study_id"].astype("int64", copy=False)
        df_aligned = df_img.merge(pred_df, on="study_id", how="left")

        return torch.tensor(
            df_aligned[[f"pred_{i}" for i in range(C)]].to_numpy(),
            dtype=torch.float32,
            device=preds.device
        )