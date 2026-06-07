from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW, Optimizer


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
) -> Optimizer:
    """AdamW with decay/no-decay split and optional per-component (discriminative) LR.

    ``param_group_lrs`` (from ``optimizer_init_args.param_group_lrs`` in config) maps a
    parameter-name prefix to its LR; unmatched params use ``lr``. Default None -> all
    components share ``lr``. Per-group LRs compose with the LR schedulers, which scale
    each group relative to its own initial LR.
    """
    groups = build_param_groups(model, base_lr=lr, weight_decay=weight_decay, param_group_lrs=param_group_lrs)
    if param_group_lrs:
        summary = ", ".join(
            f"lr={g['lr']:.2e}/wd={g['weight_decay']:g}: {sum(p.numel() for p in g['params'])} params"
            for g in groups
        )
        print(f"[optimizer] discriminative LR groups -> {summary}")
    return AdamW(
        groups,
        lr=lr,
        betas=betas,
        eps=eps,
    )
