from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW, Optimizer

# Global discriminative-LR default: the pretrained image backbone is trained
# slower than the freshly-initialised fusion/decoder, since its features are
# already useful and a high LR would smash them while the fresh head still needs
# to move fast. Applied to every model unless a config overrides it (see
# build_adamw_optimizer). The multiplier scales the base LR, so it composes with
# whatever base ``lr`` a config sets and stays correct across model variants.
DEFAULT_BACKBONE_PREFIXES: tuple[str, ...] = ("image_encoder.",)
DEFAULT_BACKBONE_LR_MULT: float = 0.3


def _match_prefix(name: str, param_group_lrs: dict[str, float]) -> str | None:
    """Longest matching prefix in ``param_group_lrs`` for parameter ``name`` (a more
    specific prefix like ``"image_encoder.frontal_encoder."`` overrides a broader
    ``"image_encoder."``), or None if nothing matches."""
    best_prefix: str | None = None
    for prefix in param_group_lrs:
        if name.startswith(prefix) and (best_prefix is None or len(prefix) > len(best_prefix)):
            best_prefix = prefix
    return best_prefix


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    param_group_lrs: dict[str, float] | None = None,
) -> list[dict]:
    """Split trainable params into AdamW groups by (learning rate, decay-eligibility).

    Decay split: bias and 1-D params (LayerNorm/BatchNorm gain, embeddings' scaling, etc.)
    get weight_decay=0; everything else gets ``weight_decay``.

    Discriminative LR (optional): ``param_group_lrs`` maps a parameter-name *prefix* to a
    learning rate; matching params use that LR instead of ``base_lr`` (longest-prefix wins).
    When it is empty/None every param uses ``base_lr`` and exactly two groups are produced
    (decay, no_decay) -- identical to the historical behaviour, so existing optimizer
    checkpoints stay compatible.
    """
    param_group_lrs = param_group_lrs or {}
    # Insertion-ordered buckets keyed by (component label, no_decay); named_parameters()
    # order is stable, so group ordering is deterministic and reproducible. The label is
    # the matched prefix (or "base" for unmatched params) and is stored as ``name`` on the
    # group so logging/plotting can show each component's LR separately.
    buckets: dict[tuple[str, bool], dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        prefix = _match_prefix(name, param_group_lrs)
        label = "base" if prefix is None else prefix
        lr = base_lr if prefix is None else float(param_group_lrs[prefix])
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        bucket = buckets.setdefault((label, no_decay), {"lr": lr, "params": []})
        bucket["params"].append(p)

    groups: list[dict] = []
    for (label, no_decay), g in buckets.items():
        groups.append(
            {
                "params": g["params"],
                "lr": g["lr"],
                "name": label,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return groups


# Back-compat alias: the historical name, now a thin wrapper (no discriminative LR).
def split_decay_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    return build_param_groups(model, base_lr=0.0, weight_decay=weight_decay)


def build_adamw_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    param_group_lrs: dict[str, float] | None = None,
    backbone_lr_mult: float | None = None,
    backbone_prefixes: tuple[str, ...] = DEFAULT_BACKBONE_PREFIXES,
) -> Optimizer:
    """AdamW with decay/no-decay split and discriminative (per-component) LR.

    Two ways to set per-component LRs, in precedence order:

    1. ``param_group_lrs`` (from ``optimizer_init_args.param_group_lrs`` in config) maps a
       parameter-name prefix to an *absolute* LR; unmatched params use ``lr``. If given, it
       takes full control and the backbone default below is skipped.
    2. Otherwise the **global backbone default** applies: every param under a
       ``backbone_prefixes`` entry (default ``image_encoder.``) gets ``lr * backbone_lr_mult``
       (default :data:`DEFAULT_BACKBONE_LR_MULT`). Override the multiplier per-config via
       ``optimizer_init_args.backbone_lr_mult``; set it to ``1.0`` (or ``None`` -> default,
       so pass ``1.0`` explicitly) to disable and train everything at ``lr``.

    Per-group LRs compose with the LR schedulers, which scale each group relative to its own
    initial LR -- so a lower backbone LR stays proportionally lower through the whole schedule.
    """
    mult = DEFAULT_BACKBONE_LR_MULT if backbone_lr_mult is None else float(backbone_lr_mult)
    # Explicit param_group_lrs wins outright; otherwise synthesise the backbone default
    # (skip when mult == 1.0, which means "no discriminative LR").
    if not param_group_lrs and mult != 1.0:
        param_group_lrs = {prefix: lr * mult for prefix in backbone_prefixes}

    groups = build_param_groups(model, base_lr=lr, weight_decay=weight_decay, param_group_lrs=param_group_lrs)
    matched_labels = {g["name"] for g in groups}
    if len(matched_labels) > 1:
        summary = ", ".join(
            f"{g['name']} lr={g['lr']:.2e}/wd={g['weight_decay']:g}: {sum(p.numel() for p in g['params'])} params"
            for g in groups
        )
        print(f"[optimizer] discriminative LR groups -> {summary}")
    elif param_group_lrs:
        # We asked for a split but nothing matched -- usually a renamed/absent backbone.
        # Surface it so a silently-uniform LR isn't mistaken for the intended split.
        wanted = ", ".join(param_group_lrs)
        print(
            f"[optimizer] no params matched discriminative-LR prefixes ({wanted}); "
            f"all {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable "
            f"params train at base lr={lr:.2e}"
        )
    return AdamW(
        groups,
        lr=lr,
        betas=betas,
        eps=eps,
    )
