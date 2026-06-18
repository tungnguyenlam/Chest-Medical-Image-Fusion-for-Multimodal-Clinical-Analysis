# Prior latents — what they are and why v5 has them

## The problem in one sentence

The prior study contributes **261 fusion tokens** (4 image views + clinical +
report + vitals + a prior-label token + a sentinel). Letting all 261 flow into
cross-attention is both expensive *and* a memorization risk: the prior-label
token in particular is a near-clean copy of "what this patient had last time,"
and the model can learn to lean on it instead of reading the current image.
**Prior latents** are v5's fix — a small, fixed bank of learned tokens
(`n_prior_latents`, default 16) that the 261 prior tokens are *selectively pooled*
down to before fusion.

No prior background needed — this note explains exactly what they are, how they
are computed, and what each knob does. The implementation lives in
[`PriorAwareV5NanoModel`](../src/model/PriorAwareV5NanoModel.py)
(`_pool_prior` and the `prior_pooler` it calls).

---

## 1. The two layouts

**The raw prior memory (261 tokens).** Built in `forward`:

```
memory = [ sentinel | 256 prior-image | prior-clin | prior-report | prior-vitals | prior-label ]
            1          (4 views × 8×8)   1            1              1              1            = 261
```

Each is a `d_model`-wide (768) vector. The sentinel is a learned, always-valid
"I have no prior" token so a no-prior sample never produces an all-masked
cross-attention row (which would NaN).

**The pooled prior latents (K tokens).** After pooling, those 261 tokens are
represented by just `K = n_prior_latents` vectors (default 16):

```
fusion_memory = [ latent_1 | latent_2 | ... | latent_K ]   = K (default 16)
```

The fusion stack's cross-attention then reads from these `K` latents instead of
the 261 raw tokens.

---

## 2. How the latents are computed (the pooler)

The latents are **not** an average or a stride — they are produced by *attention-based
selection*. A bank of `K` learned query vectors (`prior_latent_queries`, a
`nn.Parameter` of shape `(K, d_model)`) is run through one
Perceiver / Q-Former–style block, implemented as a single
`nn.TransformerDecoderLayer` (`prior_pooler`):

```
latents = prior_pooler(
    queries,                         # (B, K, C)  the learned query bank
    memory,                          # (B, 261, C) the raw prior tokens (K/V)
    memory_key_padding_mask=mem_pad, # masks out absent-prior tokens
)                                    # -> (B, K, C)
```

Inside that one block:

1. **self-attention** over the `K` queries — lets the latents specialize and not
   all grab the same thing;
2. **cross-attention** `queries → 261 prior tokens` — each latent reads a learned,
   *soft* selection of the prior; and
3. **FFN**.

Because it is selection (not pooling), a single change-relevant prior patch can
win a query — focal prior evidence (a nodule that was there last time, a line, an
effusion) survives. But the prior **label can no longer arrive un-mixed**: it gets
blended with image/text evidence inside the latents, so it stops being a clean
copyable channel.

---

## 3. Why this is the regularizer (not just a speedup)

Two things happen at once:

**(a) Cheaper.** Fusion cross-attention cost drops from `258 × 261` to `258 × K`.
With `K = 16` that is a ~16× reduction in the prior side of attention.

**(b) An information bottleneck on the prior, applied where the memorization risk
lives.** v5's whole thesis is asymmetry: the *current* image is kept at full
spatial resolution (it holds the small findings), while the *prior* is squeezed.
Forcing all of "what was true last time" through 16 latents means the model
cannot route the 26-dim prior label straight to the classifier — it has to be
reconstructed from a lossy, mixed representation, which is exactly the copy
shortcut we want to break.

Setting `n_prior_latents: 0` **disables pooling** and falls back to v4's full
261-token memory. That is the "baseline" vs "pooled" ablation cell — both live
inside the same class.

---

## 4. Prior-latent dropout

`prior_latent_dropout` (default 0.1) is a second, complementary regularizer
applied **only during training**, in `_pool_prior`:

```
keep = rand(B, K) >= prior_latent_dropout   # randomly drop whole latents
keep[:, 0] = True                            # always keep ≥1 (no all-masked row → NaN)
```

Dropped latents are masked out of fusion. Because any individual latent may
vanish on a given step, no single latent can become "the label channel" — the
information has to be carried redundantly across the bank. At eval time nothing
is dropped (all `K` latents are used).

---

## 5. The knobs

| Config key | Default | What it does |
|---|---|---|
| `n_prior_latents` | `16` | Number of learned latents `K`. `0` disables pooling (→ v4 full 261-token memory). Smaller `K` = tighter prior bottleneck = stronger regularization. |
| `pooler_nhead` | `8` | Attention heads inside the pooler block. |
| `prior_latent_dropout` | `0.1` | Probability of dropping a whole latent during training (≥1 always kept). |

These set in the model section of each `training/prior_aware_v5nano*/config.yaml`.

---

## 6. Where it sits in the forward pass

```
prior tokens (261) ──► prior_pooler (K learned queries) ──► K latents
                                                              │
                              (training: drop whole latents)  │
                                                              ▼
current tokens (258) ──────────────► fusion cross-attention ◄─┘
                                              │
                                              ▼
                                       MLDecoder head
```

The current branch (`tgt`) is the residual stream; the prior latents are
read-only `memory`. That asymmetry plus the latent bottleneck is what
distinguishes v5 from v4 — see the module docstring in
[`PriorAwareV5NanoModel.py`](../src/model/PriorAwareV5NanoModel.py) for the full
design, and [`background_attention_penalty.md`](background_attention_penalty.md)
for the related anti-shortcut loss term.
