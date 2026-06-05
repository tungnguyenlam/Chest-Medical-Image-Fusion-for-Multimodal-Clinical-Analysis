"""Gradient-based attribution for CaMCheXV2NanoVitalsModel.

A single backward pass from one class logit gives a coherent, per-modality story:

* image  -> Grad-CAM on each view's ConvNeXtV2 feature map (640, 16, 16)
* text   -> grad x input-embedding per BERT token (the only place per-word info exists,
            because the model collapses the report to a single CLS vector before fusion)
* vitals -> grad x input on the 7 raw vital values

The three live at different points in the graph, so we hook each one separately and
read everything off the same ``logit.backward()``.

IMPORTANT: this requires the *live* CXR-BERT path (token ids, not cached embeddings) and
gradients flowing through it. The CLI runner forces ``use_precomputed_text_embeddings=False``
and ``freeze_text_encoder=False`` when constructing the model so the embedding hook fires.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.dataloader.image_channel_preprocessing import CHANNEL_MODES, CHANNEL_STATS

# albumentations A.Normalize() defaults (ImageNet) -- used to de-normalize for display
# on the legacy RGB path (channel_mode is None).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VIEW_NAMES = {0: "unknown", 1: "frontal", 2: "lateral"}


@dataclasses.dataclass
class ViewAttribution:
    """One image slot: the (de-normalized) picture plus its Grad-CAM heatmap."""

    slot: int
    view_position: int          # 0 unknown / 1 frontal / 2 lateral
    image: np.ndarray           # (H, W) grayscale, 0..1 -- Grad-CAM overlay background
    cam: np.ndarray             # (H, W) heatmap, 0..1
    encoded: bool               # False when the view was not fed to an encoder (vp == 0)
    contribution: float         # raw Grad-CAM mass, used for the per-view share
    channels: Optional[np.ndarray] = None       # (H, W, C) de-normalized input planes, 0..1
    channel_names: Optional[list[str]] = None    # per-plane labels (e.g. raw/clahe/hist_eq)


@dataclasses.dataclass
class AttributionResult:
    study_id: str
    class_name: str
    class_index: int
    prob: float
    logit: float
    label: Optional[float]                  # ground-truth 0/1 for the *targeted* class
    true_labels: list[str]                  # all classes positive for this study (multilabel)

    views: list[ViewAttribution]

    tokens: list[str]                        # decoded subword tokens (pad stripped)
    token_scores: np.ndarray                 # (n_tokens,) signed grad x embedding
    text: str

    vital_names: list[str]
    vital_display: list[str]                 # de-normalized values rendered for humans
    vital_missing: np.ndarray                # (7,) bool
    vital_scores: np.ndarray                 # (7,) signed grad x value

    modality_contrib: dict                   # {"image":.., "text":.., "vitals":..} sums to 1


class CaMCheXAttributor:
    """Wraps a (loaded, in-eval) model and produces an AttributionResult per sample."""

    def __init__(self, model, tokenizer, classes, device, vital_fields, vital_stats, channel_mode=None):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.classes = list(classes)
        self.device = device
        self.vital_fields = list(vital_fields)
        self.vital_stats = vital_stats

        # De-normalization must mirror the training transform. With channel_mode set
        # the dataset used A.Normalize(CHANNEL_STATS[mode], max_pixel_value=1.0); the
        # legacy RGB path used ImageNet stats. Same inverse formula, different stats.
        self.channel_mode = channel_mode
        if channel_mode:
            stats = CHANNEL_STATS[channel_mode]
            self._denorm_mean = np.asarray(stats["mean"], dtype=np.float32)
            self._denorm_std = np.asarray(stats["std"], dtype=np.float32)
            self._channel_names = list(CHANNEL_MODES[channel_mode])
        else:
            self._denorm_mean = IMAGENET_MEAN
            self._denorm_std = IMAGENET_STD
            self._channel_names = None

        enc = model.image_encoder
        self._modules = {
            "frontal": enc.frontal_encoder,
            "lateral": enc.lateral_encoder,
            "text": _find_embeddings(model.text_encoder.biobert_encoder.text_encoder),
        }
        self._acts: dict[str, torch.Tensor] = {}
        self._grads: dict[str, torch.Tensor] = {}
        self._handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in self._modules.items()
        ]

    # -- hook plumbing ---------------------------------------------------------
    def _make_hook(self, name):
        def hook(_module, _inp, out):
            self._acts[name] = out
            if out.requires_grad:  # skip the no_grad selection pass
                out.register_hook(lambda g, n=name: self._grads.__setitem__(n, g.detach()))
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- main entry point ------------------------------------------------------
    @torch.no_grad()
    def predict_probs(self, sample) -> np.ndarray:
        """Cheap forward used by the selection pass (no backward, no hooks needed)."""
        data_batch, _ = self._to_batch(sample, vitals_require_grad=False)
        logits = self.model(data_batch)
        return torch.sigmoid(logits)[0].detach().cpu().numpy()

    def attribute(self, sample, class_index: int) -> AttributionResult:
        (study_id, img_np, vp_np, _clin_ids, _clin_mask, _vit_vals, _vit_miss), label = sample
        data_batch, vitals = self._to_batch(sample, vitals_require_grad=True)
        _, x, view_positions, clinical_ids, clinical_mask, _, _ = data_batch

        self._acts.clear()
        self._grads.clear()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(data_batch)
        prob = torch.sigmoid(logits)[0, class_index].item()
        logit = logits[0, class_index].item()
        logits[0, class_index].backward()

        views = self._image_attribution(img_np, vp_np)
        tokens, token_scores = self._text_attribution(clinical_ids, clinical_mask)
        vital_scores = (vitals.grad[0] * vitals[0]).detach().cpu().numpy()
        vital_display = self._format_vitals(_vit_vals, _vit_miss)

        modality_contrib = self._modality_contrib(views, token_scores, vital_scores)

        label_val = float(label[class_index]) if label is not None else None
        true_labels = (
            [self.classes[i] for i in range(len(self.classes)) if float(label[i]) == 1.0]
            if label is not None else []
        )
        return AttributionResult(
            study_id=str(study_id),
            class_name=self.classes[class_index],
            class_index=class_index,
            prob=prob,
            logit=logit,
            label=label_val,
            true_labels=true_labels,
            views=views,
            tokens=tokens,
            token_scores=token_scores,
            text=self.tokenizer.decode(
                clinical_ids[0][clinical_mask[0].bool()], skip_special_tokens=True
            ),
            vital_names=list(self.vital_fields),
            vital_display=vital_display,
            vital_missing=np.asarray(_vit_miss, dtype=bool),
            vital_scores=vital_scores,
            modality_contrib=modality_contrib,
        )

    # -- per-modality helpers --------------------------------------------------
    def _image_attribution(self, img_np, vp_np) -> list[ViewAttribution]:
        h = w = img_np.shape[-1]
        nonzero = img_np.reshape(img_np.shape[0], -1).sum(axis=1) != 0
        # Encoder inputs are grouped by view; reconstruct the original slot order.
        order = {
            "frontal": [i for i in range(len(vp_np)) if nonzero[i] and vp_np[i] == 1],
            "lateral": [i for i in range(len(vp_np)) if nonzero[i] and vp_np[i] == 2],
        }
        cams: dict[int, np.ndarray] = {}
        for name, slots in order.items():
            if not slots or name not in self._acts:
                continue
            act = self._acts[name]            # (n, 640, 16, 16)
            grad = self._grads[name]
            weights = grad.mean(dim=(2, 3), keepdim=True)   # GAP over spatial dims
            cam = F.relu((weights * act).sum(dim=1))        # (n, 16, 16)
            cam = F.interpolate(cam.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)
            cam = cam.squeeze(1).detach().cpu().numpy()
            for k, slot in enumerate(slots):
                cams[slot] = cam[k]

        views = []
        for slot in range(len(vp_np)):
            if not nonzero[slot]:
                continue
            channels = self._denorm_channels(img_np[slot])
            gray = self._overlay_background(channels)
            cam = cams.get(slot)
            encoded = cam is not None
            if cam is None:
                cam = np.zeros_like(gray)
            contribution = float(cam.sum())
            norm = cam / (cam.max() + 1e-8) if cam.max() > 0 else cam
            views.append(
                ViewAttribution(
                    slot=slot,
                    view_position=int(vp_np[slot]),
                    image=gray,
                    cam=norm,
                    encoded=encoded,
                    contribution=contribution,
                    channels=channels if self.channel_mode else None,
                    channel_names=self._channel_names,
                )
            )
        return views

    def _text_attribution(self, clinical_ids, clinical_mask):
        act = self._acts["text"]              # (1, seq, 768)
        grad = self._grads["text"]
        scores = (grad * act).sum(dim=-1)[0]  # (seq,)
        keep = clinical_mask[0].bool()
        ids = clinical_ids[0][keep].detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens, scores[keep].detach().cpu().numpy()

    def _modality_contrib(self, views, token_scores, vital_scores):
        # Rough "where did signal come from" heuristic: total |grad x input| at each
        # modality's native input. Comparable-ish because all feed the same logit.
        text = float(np.abs(token_scores).sum())
        vit = float(np.abs(vital_scores).sum())
        img = float(self._image_grad_mass())
        total = img + text + vit + 1e-8
        return {"image": img / total, "text": text / total, "vitals": vit / total}

    def _image_grad_mass(self) -> float:
        mass = 0.0
        for name in ("frontal", "lateral"):
            if name in self._acts and name in self._grads:
                mass += (self._grads[name] * self._acts[name]).abs().sum().item()
        return mass

    # -- formatting ------------------------------------------------------------
    def _denorm_channels(self, chw):
        """Invert the training normalization -> HWC planes in [0, 1]."""
        hwc = np.transpose(chw, (1, 2, 0)) * self._denorm_std + self._denorm_mean
        return np.clip(hwc, 0.0, 1.0)

    def _overlay_background(self, channels):
        """Single (H, W) grayscale background for the Grad-CAM overlay.

        Channel mode -> the raw plane (channel 0, most anatomically readable);
        legacy RGB -> the channel mean (the old behavior).
        """
        if self.channel_mode:
            return channels[..., 0]
        return channels.mean(axis=-1)

    def _format_vitals(self, values, missing):
        out = []
        for field, val, miss in zip(self.vital_fields, np.asarray(values), np.asarray(missing)):
            if miss:
                out.append("n/a")
                continue
            stats = self.vital_stats.get(field, {"mean": 0.0, "std": 1.0})
            raw = float(val) * float(stats.get("std", 1.0) or 1.0) + float(stats.get("mean", 0.0))
            if field == "gender":
                out.append("M" if raw >= 0.5 else "F")
            else:
                out.append(f"{raw:.1f}")
        return out

    # -- batch construction ----------------------------------------------------
    def _to_batch(self, sample, vitals_require_grad: bool):
        (study_id, img_np, vp_np, clin_ids, clin_mask, vit_vals, vit_miss), label = sample
        x = torch.as_tensor(img_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        vp = torch.as_tensor(vp_np, dtype=torch.long, device=self.device).unsqueeze(0)
        clinical_ids = torch.as_tensor(clin_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        clinical_mask = torch.as_tensor(clin_mask, dtype=torch.long, device=self.device).unsqueeze(0)
        vitals = torch.as_tensor(vit_vals, dtype=torch.float32, device=self.device).unsqueeze(0)
        if vitals_require_grad:
            vitals = vitals.clone().requires_grad_(True)
        vital_missing = torch.as_tensor(vit_miss, dtype=torch.bool, device=self.device).unsqueeze(0)
        data = ([str(study_id)], x, vp, clinical_ids, clinical_mask, vitals, vital_missing)
        return data, vitals


def _find_embeddings(hf_model):
    """Locate a BERT-style embeddings submodule to hook for token attribution."""
    if hasattr(hf_model, "embeddings"):
        return hf_model.embeddings
    for module in hf_model.modules():
        if module.__class__.__name__.endswith("Embeddings"):
            return module
    raise AttributeError(
        "Could not find an embeddings submodule on the text encoder; "
        "token-level attribution needs the live BERT path."
    )
