"""Gradient-based attribution for the prior-aware CaMCheX models.

The prior-aware models are a *superset* of the single-study CaMCheX Nano models:
they run the same shared image backbone and CXR-BERT text encoder, but call them
twice — once for the *current* study and once for a *prior* study — and add tokens
the single-study model never has (a prior radiology report, a projected prior-label
vector, and a time-delta bucket embedding broadcast onto every prior token).

So this attributor reuses :class:`CaMCheXAttributor`'s de-normalization, Grad-CAM,
text grad×embedding, and vital grad×value machinery, and adds the prior branch on
top. One backward pass from a single class logit yields, per modality:

* current image / clinical text / vitals (or obs text) — identical to CaMCheX
* prior image (own Grad-CAM), prior clinical text, prior report text
* prior-label token — grad×value over the 26-dim prior CheXpert vector (per-class)
* time-delta token — grad×embedding on the looked-up bucket (a single scalar)

Two hooking differences vs the single-study attributor:

* The image encoder is hooked at the ``image_encoder`` module level (not the per-view
  sub-encoders) because it is called exactly twice per forward — call 0 is current,
  call 1 is prior — whereas the per-view sub-encoders fire a data-dependent number of
  times across the two branches.
* The shared BERT embeddings module fires once per text stream, in a fixed order that
  depends on the model family (see ``_text_slots``); we accumulate every call and map
  by index.

Requires the *live* CXR-BERT path (token ids, grads on); the runner forces
``use_precomputed_text_embeddings=False`` and ``freeze_text_encoder=False``.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.dataloader.PriorAwareDataset import bucket_days
from src.interpret.attribution import (
    CaMCheXAttributor,
    ViewAttribution,
    _find_embeddings,
)

# Human-readable label for each time-delta bucket index (see bucket_days).
DELTA_BUCKET_NAMES = {
    0: "no prior / unknown",
    1: "≤ 1 day",
    2: "2–7 days",
    3: "8–30 days",
    4: "1–6 months",
    5: "6–12 months",
    6: "1–3 years",
    7: "> 3 years",
}


@dataclasses.dataclass
class TextStream:
    """One attributed text stream (clinical / obs / report) for either branch."""

    key: str                       # e.g. "cur_clin", "prv_report"
    title: str                     # panel title
    tokens: list[str]
    scores: np.ndarray             # (n_tokens,) signed grad × embedding
    text: str


@dataclasses.dataclass
class PriorAttributionResult:
    study_id: str
    class_name: str
    class_index: int
    prob: float
    logit: float
    label: Optional[float]
    true_labels: list[str]

    has_prior: bool
    days_since_prior: float
    delta_bucket: int
    delta_score: float             # signed grad × embedding summed over the delta vector
    delta_mag: float               # |grad × embedding| summed (unsigned magnitude)

    # current branch
    cur_views: list[ViewAttribution]
    cur_texts: list[TextStream]
    has_vitals: bool
    cur_vital_names: list[str]
    cur_vital_display: list[str]
    cur_vital_missing: np.ndarray
    cur_vital_scores: np.ndarray

    # prior branch
    prv_views: list[ViewAttribution]
    prv_texts: list[TextStream]
    prv_vital_display: list[str]
    prv_vital_missing: np.ndarray
    prv_vital_scores: np.ndarray
    prior_label_scores: np.ndarray  # (26,) signed grad × value
    prior_label_values: np.ndarray  # (26,) the raw multi-hot prior label
    class_names: list[str]          # the 26 class names (y-axis of the prior-label panel)

    # {"cur_image":.., "cur_text":.., .., "prv_label":.., "time_delta":..} sums to 1
    modality_contrib: dict


class PriorAwareAttributor(CaMCheXAttributor):
    """Attribution for PriorAware{CaMCheX,V2Nano,V3Nano}Model (dict-batch forward)."""

    def __init__(self, model, tokenizer, classes, device, vital_fields, vital_stats, channel_mode=None):
        # NB: deliberately does NOT call CaMCheXAttributor.__init__ — that hooks the
        # per-view sub-encoders and a tuple-style batch. We reuse only its helpers.
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.classes = list(classes)
        self.device = device
        self.vital_fields = list(vital_fields)
        self.vital_stats = vital_stats
        self._init_denorm(channel_mode)

        # Family: the Nano variants carry a numeric vitals projector + clinical-only
        # text; the base prior_aware model has no vitals and feeds an obs text stream.
        self.has_vitals = hasattr(model, "vitals_projector")

        # Hooks. image_encoder fires twice/forward (current, prior); the shared BERT
        # embeddings and delta_embedding fire once per call — accumulate lists and
        # disambiguate by call order in `_text_slots` / `attribute`.
        self._img_acts: list[torch.Tensor] = []
        self._img_grads: list[Optional[torch.Tensor]] = []
        self._txt_acts: list[torch.Tensor] = []
        self._txt_grads: list[Optional[torch.Tensor]] = []
        self._delta_act: list[torch.Tensor] = []
        self._delta_grad: list[Optional[torch.Tensor]] = []

        embeddings = _find_embeddings(model.text_encoder.biobert_encoder.text_encoder)
        self._handles = [
            model.image_encoder.register_forward_hook(self._img_hook),
            embeddings.register_forward_hook(self._txt_hook),
            model.delta_embedding.register_forward_hook(self._delta_hook),
        ]

    # -- hook plumbing ---------------------------------------------------------
    @staticmethod
    def _stash(acts: list, grads: list):
        """Append `out` to `acts`; register a grad hook writing into `grads[idx]`."""
        def hook(out):
            idx = len(acts)
            acts.append(out)
            if isinstance(out, torch.Tensor) and out.requires_grad:
                def grad_hook(g, i=idx):
                    while len(grads) <= i:
                        grads.append(None)
                    grads[i] = g.detach()
                out.register_hook(grad_hook)
        return hook

    def _img_hook(self, _module, _inp, out):
        feats = out[0] if isinstance(out, (tuple, list)) else out
        self._stash(self._img_acts, self._img_grads)(feats)

    def _txt_hook(self, _module, _inp, out):
        self._stash(self._txt_acts, self._txt_grads)(out)

    def _delta_hook(self, _module, _inp, out):
        self._stash(self._delta_act, self._delta_grad)(out)

    def _clear(self):
        for buf in (self._img_acts, self._img_grads, self._txt_acts,
                    self._txt_grads, self._delta_act, self._delta_grad):
            buf.clear()

    # -- text-stream layout per family ----------------------------------------
    def _text_slots(self, batch):
        """Ordered (key, title, input_ids, attn_mask) per BERT embeddings call.

        The order mirrors each model's forward(): the Nano variants encode
        cur_clin, prv_clin, prv_report; the base prior_aware model encodes the
        current (clin, obs) pair, then the prior (clin, obs) pair, then the report.
        """
        clin = ("clin_input_ids", "clin_attn_mask")
        obs = ("obs_input_ids", "obs_attn_mask")
        pclin = ("prior_clin_input_ids", "prior_clin_attn_mask")
        pobs = ("prior_obs_input_ids", "prior_obs_attn_mask")
        prep = ("prior_report_input_ids", "prior_report_attn_mask")

        def s(key, title, pair):
            ids, mask = pair
            return (key, title, batch[ids], batch[mask])

        cn = self.class_name_for_title
        if self.has_vitals:
            return [
                s("cur_clin", f"{cn} — current clinical indication", clin),
                s("prv_clin", f"{cn} — prior clinical indication", pclin),
                s("prv_report", f"{cn} — prior radiology report", prep),
            ]
        return [
            s("cur_clin", f"{cn} — current clinical indication", clin),
            s("cur_obs", f"{cn} — current observation/vitals text", obs),
            s("prv_clin", f"{cn} — prior clinical indication", pclin),
            s("prv_obs", f"{cn} — prior observation/vitals text", pobs),
            s("prv_report", f"{cn} — prior radiology report", prep),
        ]

    # -- main entry points -----------------------------------------------------
    @torch.no_grad()
    def predict_probs(self, sample) -> np.ndarray:
        self._clear()
        try:
            batch, _, _ = self._to_dict_batch(sample, with_grad=False)
            logits = self.model(batch)
            return torch.sigmoid(logits)[0].detach().cpu().numpy()
        finally:
            self._clear()

    def attribute(self, sample, class_index: int) -> PriorAttributionResult:
        data, label = sample
        self.class_name_for_title = self.classes[class_index]
        self._clear()
        self.model.zero_grad(set_to_none=True)

        try:
            batch, grad_inputs, _ = self._to_dict_batch(sample, with_grad=True)
            logits = self.model(batch)
            prob = torch.sigmoid(logits)[0, class_index].item()
            logit = logits[0, class_index].item()
            logits[0, class_index].backward()

            has_prior = bool(data["has_prior"])

            # images: call 0 = current, call 1 = prior.
            cur_views = self._block_views(data["img"], data["view_positions"], 0)
            prv_views = self._block_views(data["prior_img"], data["prior_view_positions"], 1) if has_prior else []

            # text streams, mapped by fixed call order.
            slots = self._text_slots(batch)
            cur_texts, prv_texts = [], []
            for i, (key, title, ids, mask) in enumerate(slots):
                toks, scores = self._text_attr(i, ids, mask)
                stream = TextStream(key=key, title=title, tokens=toks, scores=scores,
                                    text=self._decode(ids, mask))
                (cur_texts if key.startswith("cur") else prv_texts).append(stream)

            # vitals (Nano only): grad x value on the raw normalized vitals.
            if self.has_vitals:
                cur_vital_scores = self._grad_x_input(grad_inputs["cur_vitals"])
                prv_vital_scores = self._grad_x_input(grad_inputs["prior_vitals"]) if has_prior else np.zeros(len(self.vital_fields))
                cur_vital_display = self._format_vitals(data["vital_values"], data["vital_missing_mask"])
                prv_vital_display = self._format_vitals(data["prior_vital_values"], data["prior_vital_missing_mask"])
            else:
                cur_vital_scores = prv_vital_scores = np.zeros(0)
                cur_vital_display = prv_vital_display = []

            # prior label token: grad x value over the 26-dim multi-hot vector.
            prior_label_scores = self._grad_x_input(grad_inputs["prior_label"])
            prior_label_values = np.asarray(data["prior_label"], dtype=np.float32)

            # time-delta token: one embedding lookup -> signed/abs grad x embedding.
            delta_idx = int(bucket_days(
                torch.tensor([float(data["days_since_prior"])]),
                torch.tensor([has_prior]),
            ).item())
            delta_score, delta_mag = self._delta_attr()

            modality_contrib = self._modality_contrib(
                cur_views, prv_views, cur_texts, prv_texts,
                cur_vital_scores, prv_vital_scores, prior_label_scores, delta_mag,
            )

            label_val = float(label[class_index]) if label is not None else None
            true_labels = (
                [self.classes[i] for i in range(len(self.classes)) if float(label[i]) == 1.0]
                if label is not None else []
            )
            return PriorAttributionResult(
                study_id=str(data["study_id"]),
                class_name=self.classes[class_index],
                class_index=class_index,
                prob=prob,
                logit=logit,
                label=label_val,
                true_labels=true_labels,
                has_prior=has_prior,
                days_since_prior=float(data["days_since_prior"]),
                delta_bucket=delta_idx,
                delta_score=delta_score,
                delta_mag=delta_mag,
                cur_views=cur_views,
                cur_texts=cur_texts,
                has_vitals=self.has_vitals,
                cur_vital_names=list(self.vital_fields) if self.has_vitals else [],
                cur_vital_display=cur_vital_display,
                cur_vital_missing=np.asarray(data["vital_missing_mask"], dtype=bool) if self.has_vitals else np.zeros(0, bool),
                cur_vital_scores=cur_vital_scores,
                prv_views=prv_views,
                prv_texts=prv_texts,
                prv_vital_display=prv_vital_display,
                prv_vital_missing=np.asarray(data["prior_vital_missing_mask"], dtype=bool) if self.has_vitals else np.zeros(0, bool),
                prv_vital_scores=prv_vital_scores,
                prior_label_scores=prior_label_scores,
                prior_label_values=prior_label_values,
                class_names=list(self.classes),
                modality_contrib=modality_contrib,
            )
        finally:
            self._clear()
            self.model.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # -- per-modality helpers --------------------------------------------------
    def _block_views(self, img_np, vp_np, call_index: int) -> list[ViewAttribution]:
        """Grad-CAM for one image block, hooked at the image_encoder level.

        The encoder packs its nonzero slots in slot order, so feature row k maps to
        the k-th nonzero slot of this block (matches the model's own gather).
        """
        img_np = np.asarray(img_np)
        vp_np = np.asarray(vp_np)
        h = w = img_np.shape[-1]
        nonzero = img_np.reshape(img_np.shape[0], -1).sum(axis=1) != 0
        slots = [i for i in range(len(vp_np)) if nonzero[i]]

        cams: dict[int, np.ndarray] = {}
        act = self._img_acts[call_index] if call_index < len(self._img_acts) else None
        grad = self._img_grads[call_index] if call_index < len(self._img_grads) else None
        if act is not None and grad is not None and act.shape[0] == len(slots) and len(slots):
            weights = grad.mean(dim=(2, 3), keepdim=True)          # GAP over spatial dims
            cam = F.relu((weights * act).sum(dim=1))               # (n, h2, w2)
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
            views.append(ViewAttribution(
                slot=slot, view_position=int(vp_np[slot]), image=gray, cam=norm,
                encoded=encoded, contribution=contribution,
                channels=channels if self.channel_mode else None,
                channel_names=self._channel_names,
            ))
        return views

    def _text_attr(self, call_index: int, ids, mask):
        if call_index >= len(self._txt_acts) or call_index >= len(self._txt_grads):
            return [], np.zeros(0)
        act = self._txt_acts[call_index]
        grad = self._txt_grads[call_index]
        if grad is None:
            return [], np.zeros(0)
        scores = (grad * act).sum(dim=-1)[0]                       # (seq,)
        keep = torch.as_tensor(np.asarray(mask)[0] if np.asarray(mask).ndim == 2 else np.asarray(mask)).bool()
        keep = keep.to(scores.device)
        ids_t = torch.as_tensor(np.asarray(ids))
        ids_t = ids_t[0] if ids_t.ndim == 2 else ids_t
        kept_ids = ids_t[keep.cpu()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(kept_ids)
        return tokens, scores[keep].detach().cpu().numpy()

    def _decode(self, ids, mask) -> str:
        ids_t = torch.as_tensor(np.asarray(ids))
        ids_t = ids_t[0] if ids_t.ndim == 2 else ids_t
        mask_t = torch.as_tensor(np.asarray(mask))
        mask_t = (mask_t[0] if mask_t.ndim == 2 else mask_t).bool()
        return self.tokenizer.decode(ids_t[mask_t], skip_special_tokens=True)

    def _delta_attr(self):
        if not self._delta_act or not self._delta_grad or self._delta_grad[0] is None:
            return 0.0, 0.0
        act, grad = self._delta_act[0], self._delta_grad[0]
        prod = (grad * act)[0]
        return float(prod.sum().item()), float(prod.abs().sum().item())

    @staticmethod
    def _grad_x_input(tensor) -> np.ndarray:
        if tensor.grad is None:
            return np.zeros(tensor.shape[-1], dtype=np.float32)
        return (tensor.grad[0] * tensor[0]).detach().cpu().numpy()

    def _modality_contrib(self, cur_views, prv_views, cur_texts, prv_texts,
                          cur_vital_scores, prv_vital_scores, prior_label_scores, delta_mag):
        def img_mass(views):
            return float(sum(v.contribution for v in views))

        def txt_mass(streams, key):
            for st in streams:
                if st.key == key:
                    return float(np.abs(st.scores).sum())
            return 0.0

        parts = {
            "cur_image": img_mass(cur_views),
            "cur_clin": txt_mass(cur_texts, "cur_clin"),
            "cur_vitals": float(np.abs(cur_vital_scores).sum()) if self.has_vitals else txt_mass(cur_texts, "cur_obs"),
            "prv_image": img_mass(prv_views),
            "prv_clin": txt_mass(prv_texts, "prv_clin"),
            "prv_report": txt_mass(prv_texts, "prv_report"),
            "prv_vitals": float(np.abs(prv_vital_scores).sum()) if self.has_vitals else txt_mass(prv_texts, "prv_obs"),
            "prv_label": float(np.abs(prior_label_scores).sum()),
            "time_delta": float(delta_mag),
        }
        total = sum(parts.values()) + 1e-8
        return {k: v / total for k, v in parts.items()}

    # -- batch construction ----------------------------------------------------
    def _to_dict_batch(self, sample, with_grad: bool):
        data, label = sample

        def t(key, dtype):
            return torch.as_tensor(np.asarray(data[key]), dtype=dtype, device=self.device).unsqueeze(0)

        batch = {
            "study_id": [str(data["study_id"])],
            "img": t("img", torch.float32),
            "view_positions": t("view_positions", torch.long),
            "clin_input_ids": t("clin_input_ids", torch.long),
            "clin_attn_mask": t("clin_attn_mask", torch.long),
            "obs_input_ids": t("obs_input_ids", torch.long),
            "obs_attn_mask": t("obs_attn_mask", torch.long),
            "has_prior": torch.as_tensor([bool(data["has_prior"])], device=self.device),
            "prior_img": t("prior_img", torch.float32),
            "prior_view_positions": t("prior_view_positions", torch.long),
            "prior_clin_input_ids": t("prior_clin_input_ids", torch.long),
            "prior_clin_attn_mask": t("prior_clin_attn_mask", torch.long),
            "prior_obs_input_ids": t("prior_obs_input_ids", torch.long),
            "prior_obs_attn_mask": t("prior_obs_attn_mask", torch.long),
            "prior_report_input_ids": t("prior_report_input_ids", torch.long),
            "prior_report_attn_mask": t("prior_report_attn_mask", torch.long),
            "days_since_prior": t("days_since_prior", torch.float32),
            "vital_missing_mask": t("vital_missing_mask", torch.bool),
            "prior_vital_missing_mask": t("prior_vital_missing_mask", torch.bool),
        }
        cur_vit = t("vital_values", torch.float32)
        prv_vit = t("prior_vital_values", torch.float32)
        prior_label = t("prior_label", torch.float32)
        grad_inputs = {}
        if with_grad:
            prior_label = prior_label.clone().requires_grad_(True)
            grad_inputs["prior_label"] = prior_label
            if self.has_vitals:
                cur_vit = cur_vit.clone().requires_grad_(True)
                prv_vit = prv_vit.clone().requires_grad_(True)
                grad_inputs["cur_vitals"] = cur_vit
                grad_inputs["prior_vitals"] = prv_vit
        batch["vital_values"] = cur_vit
        batch["prior_vital_values"] = prv_vit
        batch["prior_label"] = prior_label
        return batch, grad_inputs, label
