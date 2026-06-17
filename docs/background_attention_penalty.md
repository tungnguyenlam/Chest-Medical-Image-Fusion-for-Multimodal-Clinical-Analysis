# Background-attention penalty — the math

## The problem in one sentence

Some Grad-CAMs show the model drawing evidence from outside the patient
(collimation bars, corners, burned-in text). We want a loss term that makes the
model **pay a small price** for putting signal there, without ever punishing it
for looking at real anatomy.

This note explains exactly what that term is, where every number comes from, and
why it does what we want. No prior background needed.

---

## 1. The two things we already have

**(a) The background mask.** For each image we precompute a map

$$
M \in [0,1]^{H_{\text{img}} \times W_{\text{img}}}, \qquad
M(p) = \begin{cases}
\to 1 & \text{pixel } p \text{ is confident outside-patient background} \\
0 & \text{pixel } p \text{ is anatomy (or ambiguous body edge)}
\end{cases}
$$

This is the `confident_background()` output you've been eyeballing: 1 (red) only
on border-connected, extreme-valued pixels inside a thin outer band; 0 (green)
everywhere else. It's *soft* (feathered), so an edge pixel can be, say, $0.4$.

**(b) The image features.** The conv encoder turns each view into a feature grid

$$
F \in \mathbb{R}^{(B\cdot S)\times C \times H \times W}
$$

— in [`CaMCheXV2NanoVitalsModel.forward`](../src/model/CaMCheXV2NanoVitalsModel.py),
this is the `feats` tensor right after `image_encoder`. Here $B$ = batch,
$S$ = views per study, $C$ = channels (feature dimension), and $H\times W$ is a
small spatial grid (e.g. $8\times 8$). **Crucially, each of the $H\times W$ cells
still corresponds to a patch of the original image** — cell $(i,j)$ summarizes a
roughly $\tfrac{H_{\text{img}}}{H}\times\tfrac{W_{\text{img}}}{W}$ region. That
spatial correspondence is the whole reason this works.

The number we care about per cell is its **activation energy**

$$
E_{i,j} \;=\; \lVert F_{:,\,i,j}\rVert^2 \;=\; \sum_{c=1}^{C} F_{c,i,j}^2 .
$$

Read $E_{i,j}$ as *"how loudly is the encoder responding at this location?"* If a
cell is near-silent ($E_{i,j}\approx 0$), nothing downstream — no transformer
token, no attention head — has anything to attend to there. So **suppressing the
energy of a cell ≈ telling the model to stop using that location.**

---

## 2. Putting the mask on the same grid as the features

$M$ lives at image resolution ($H_{\text{img}}\times W_{\text{img}}$); $E$ lives
on the $H\times W$ grid. We bring the mask down to the feature grid by
**average-pooling**:

$$
\widetilde{M}_{i,j} \;=\; \frac{1}{|R_{i,j}|}\sum_{p \in R_{i,j}} M(p)
\;\in\; [0,1],
$$

where $R_{i,j}$ is the image region that cell $(i,j)$ covers. So
$\widetilde{M}_{i,j}$ is simply **the fraction of cell $(i,j)$ that is background.**

- A cell entirely in the black collimation bar → $\widetilde{M}\approx 1$.
- A cell straddling the body edge → $\widetilde{M}\approx 0.4$ (partial).
- A cell deep in the lung → $\widetilde{M}=0$.

This is the answer to your earlier objection — *"you can't say a token is
background or not."* Correct: we never make that binary call. A boundary cell gets
a **partial weight**, proportional to how much background it actually contains.

---

## 3. The penalty term

Penalize activation energy **weighted by how much background each cell holds**:

$$
\boxed{\;
\mathcal{L}_{\text{bg}}
\;=\;
\frac{\displaystyle\sum_{i,j}\widetilde{M}_{i,j}\,E_{i,j}}
     {\displaystyle\sum_{i,j}\widetilde{M}_{i,j} \;+\; \varepsilon}
\;}
$$

(averaged over the batch; $\varepsilon$ is a tiny constant so an image with **no**
background — no bars, nothing flagged — gives $\mathcal{L}_{\text{bg}}=0$ instead
of dividing by zero).

Term by term:

| symbol | meaning | effect in the sum |
|---|---|---|
| $E_{i,j}$ | encoder's response at cell $(i,j)$ | the thing we want small in background |
| $\widetilde{M}_{i,j}$ | fraction of that cell that is background | gates the penalty: $0$ over anatomy ⇒ **anatomy contributes nothing** |
| denominator | total background weight | normalizes, so the value is "mean energy *per unit of background*", not "more bars ⇒ bigger number" |

Because $\widetilde{M}_{i,j}=0$ on every anatomy cell, those cells are
**multiplied out of the sum entirely**. The penalty literally cannot see, reward,
or punish what happens over the lungs/heart/chest wall. The mask's guarantee
(never flags anatomy) is inherited directly by the loss.

---

## 4. How it joins training

The total loss is the usual classification loss plus this term, scaled by a
single knob $\lambda$:

$$
\mathcal{L} \;=\; \mathcal{L}_{\text{cls}} \;+\; \lambda\,\mathcal{L}_{\text{bg}} .
$$

- $\lambda = 0$ → the term vanishes; training is exactly as before (this is the
  default — the feature is opt-in).
- $\lambda > 0$ → at every step the optimizer gets a gradient that nudges the
  encoder's weights to **lower the energy it produces in background cells.**

### What the gradient actually does

For any feature value $F_{c,i,j}$ sitting in a background cell, the term's
gradient is

$$
\frac{\partial \mathcal{L}_{\text{bg}}}{\partial F_{c,i,j}}
\;\propto\; \widetilde{M}_{i,j}\,\cdot\, 2\,F_{c,i,j}.
$$

Plain reading: **push that feature toward zero, with a strength proportional to
how much background the cell contains.** Over anatomy ($\widetilde{M}=0$) the push
is exactly zero. At a half-background edge cell the push is half-strength. In the
black bar it's full-strength. This is the "soft, fractional, asymmetric"
behaviour we wanted, falling straight out of the formula.

---

## 5. Why $\lambda$ must be small

The mask is excellent on the collimation/black-bar side but has **rare false
positives** on the saturated-white side (it occasionally clips real tissue —
chin, neck, upper abdomen — sitting right at the image edge). On those rare cells
$\widetilde{M}>0$ even though it's really anatomy, so the penalty *would* push
down a legitimate feature.

A small $\lambda$ is the safety margin:

- **Where the mask is right** (the common case): a steady, gentle pressure across
  thousands of steps still adds up to "stop using the bars."
- **Where the mask is wrong** (rare edge tissue): the push is too weak to teach
  the model to ignore that tissue, so we don't trade the background bias for a
  *new* "ignore the lower chest" bias.

In other words: we trust the mask enough to nudge, not enough to shove. We'll
start around $\lambda = 0.01$ and tune by watching (a) val AUC doesn't drop,
especially on off-lung classes, and (b) the Grad-CAMs stop lighting up the
corners.

---

## 6. Why this form and not the alternatives

- **Why energy $\lVert F\rVert^2$ and not Grad-CAM directly?** Grad-CAM is a
  *diagnostic* (noisy, post-hoc, one class at a time). Penalizing raw feature
  energy is cheap, class-agnostic, and acts on the same quantity Grad-CAM
  visualizes — if the encoder is silent in the background, every downstream
  class-CAM is too.
- **Why not just delete the background pixels from the input?** Hard-masking the
  image would erase evidence for the off-lung classes (devices, fractures,
  mediastinal/pleural findings) and force a hard binary call on ambiguous edge
  cells. The soft penalty leaves the pixels in and only discourages *over-relying*
  on the confident-background ones.
- **Why not the input-gradient (RRR) version?** That penalizes
  $\lVert(1-\text{anatomy})\odot \partial \mathcal{L}/\partial x\rVert^2$ and is
  arguably more faithful, but it needs a **second backward pass per step** —
  expensive on the current GPU. The activation-energy term rides the forward pass
  we already do, so it's near-free. (If we ever want the stronger version, the
  math above carries over; only $E_{i,j}$ changes.)

---

## TL;DR

1. Mask $M$ = where the background is (you've verified it).
2. Average-pool it to the feature grid → $\widetilde{M}$ = *fraction of each cell
   that is background*.
3. Penalize $\widetilde{M}\cdot(\text{feature energy})$, normalized.
4. Add $\lambda$ × that to the loss; $\lambda$ small.
5. Anatomy cells have $\widetilde{M}=0$, so they're untouchable by construction;
   only confident-background cells feel a gentle "go quiet" pressure.
