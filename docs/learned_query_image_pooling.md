# Learned-Query Current-Image Pooling (design note, built in v7)

**Status:** built in the v7 line, with the resolution caveat below baked in as a
runtime gate. v7 keeps v6's 2×2 max-pool and **skips the current pooler at 512px**
(post-max-pool grid already 8×8 → current path ≡ v6); the pooler only activates
above 512, where it both carries more signal than a fixed pool and holds the fusion
token count constant. This is the direct implementation of risk #1's conclusion
("only nets positive if paired with a resolution bump"). See
[`training/prior_aware_v7nano/PROPOSAL.md`](../training/prior_aware_v7nano/PROPOSAL.md)
for the v7 plan and `src/model/PriorAwareV7NanoModel.py` (`SKIP_CUR_POOLER_INPUT_SIZE`)
for the gate. The label-graph head is the separate v8 line
([`docs/prior_aware_v8_label_graph.md`](prior_aware_v8_label_graph.md)).

## The idea

v5/v6 already use a Perceiver-style learned-query pooler for the **prior** memory:
`n_prior_latents` (default 16) learned query tokens cross-attend to the full prior token
set through one `TransformerDecoder` block → 16 fixed prior latents
([`PriorAwareV6NanoModel.py:220-236`](../src/model/PriorAwareV6NanoModel.py#L220)).

The **current** image is deliberately *not* pooled this way. It keeps its full spatial
grid (max-pooled to 8×8 per view, [`:170`](../src/model/PriorAwareV6NanoModel.py#L170)),
carries 2D positional + segment embeddings, and enters fusion as the `tgt`. This
asymmetry — *current = full-detail queries, prior = compressed memory* — is the identity
of the v4→v6 line, motivated by the small-finding thesis.

**Proposal:** apply the same learned-query bottleneck to the **current** image too —
query a fixed number K of latent tokens from the current grid before fusion.

## Why this is interesting: it decouples token count from input resolution

This is the main reason to pursue it, and it ties directly to the open
[image-size question](#relation-to-image-size). Today, raising input resolution to
768/1024 explodes fusion attention 5–16× because token count scales with the backbone
grid (ConvNeXt /32 → extra stride-2 → /64; 512→64 tokens/view, 768→144, 1024→256).

A learned-query current pooler makes the **fusion sequence length constant** regardless
of input resolution. The backbone still pays area-scaling conv FLOPs, but the O(N²)
transformer cost stays flat. **This is arguably the only way to afford 768/1024 input on
the current GPU**, and it's a clean thesis story: *a fixed-budget latent bottleneck lets
us feed higher-resolution images at constant fusion cost.*

## Risks (in priority order)

1. **Fights the small-finding thesis head-on.** Current detail was protected precisely
   because focal lesions need spatial resolution. Compressing current to K latents risks
   washing out exactly what the architecture exists to catch. **This only nets positive
   if paired with a resolution bump** — a 16×16 grid (from 768px) pooled to ~64 learned
   latents can carry *more* localized signal than today's raw 8×8=64 grid, because the
   queries learn where to look. Pooling current at 512px would just lose information for
   no gain.

2. **Loses spatial grounding — breaks two things the repo depends on.** Learned latents
   are position-detached, so:
   - **Grad-CAM / [`prior_attribution.py`](../src/interpret/prior_attribution.py)** lose
     the clean spatial-token → pixel mapping the interpretability chapter relies on.
   - **The background-attention penalty**
     ([`PriorAwareV6NanoModel.py:338`](../src/model/PriorAwareV6NanoModel.py#L338),
     [`docs/background_attention_penalty.md`](background_attention_penalty.md)) operates
     directly on `cur_block` spatial tokens to know which cells are background — it can't
     function on permutation-free latents.

   Need a fallback before committing: attribute through the pooler's cross-attention
   weights (query→grid-cell attention) instead of through spatial tokens.

3. **Dissolves the asymmetry.** If both current and prior become latent sets, the model
   is no longer "asymmetric cross-attention" — it's a symmetric two-latent-set fusion.
   Legitimate redesign, but a philosophical departure from the v-series; frame it as such.

## Design choices to settle when we revisit

- **K (current latents).** Keep generous — current is the primary signal, it must not
  share the prior's tiny 16-latent budget. Candidates: per-view latents, or global
  K ≈ 64–96.
- **Per-view vs global pooling.** Pool each of the 4 views separately (preserves view
  identity / segment embedding) or pool the concatenated multi-view grid into one global
  latent set.
- **Resolution pairing.** Treat this as a *resolution-decoupling* pooler, not a
  *compression* pooler — only pursue bundled with 768px input.
- **Interpretability fallback.** Decide the Grad-CAM substitute (pooler cross-attention
  rollout) up front.

## Relation to image size

See the earlier analysis: at 512 the fusion is ~512 image tokens; 768 → ~1150 (~5×
attention), 1024 → ~2050 (~16×). This pooler is what removes that scaling wall. Bundle
the two: *learned current queries + 768px* is the coherent next experiment, not either
one alone.
