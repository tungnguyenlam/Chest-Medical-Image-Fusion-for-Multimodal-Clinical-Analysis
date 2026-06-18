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

— in [`PriorAwareV5NanoModel._encode_image_block`](../src/model/PriorAwareV5NanoModel.py),
this is the `content` tensor: the `image_proj` output taken **before the positional
encoding and segment embedding are added**. Those two are a large,
content-independent energy floor (the same for every cell no matter what the image
shows); penalizing them would just punish position, so we exclude them and act only
on the image-feature energy the model controls. Here $B$ = batch,
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

Penalize the **fraction of total feature energy that sits in background cells**:

$$
\boxed{\;
\mathcal{L}_{\text{bg}}
\;=\;
\frac{\displaystyle\sum_{i,j}\widetilde{M}_{i,j}\,E_{i,j}}
     {\displaystyle\sum_{i,j} E_{i,j} \;+\; \varepsilon}
\;\in\;[0,1]
\;}
$$

(sums run over all *valid* cells of the batch; $\varepsilon$ is a tiny constant so
an all-silent grid gives $0$ instead of dividing by zero).

The denominator changed from "total background weight" to **total energy**. That one
change is what makes $\mathcal{L}_{\text{bg}}$ a **dimensionless ratio in $[0,1]$**:
*"of all the energy the encoder produces, what fraction is it spending on
background?"* This matters because it is now **scale-free** — independent of the
feature magnitude, the channel count $C$, the grid size, and the batch size. (The
earlier form normalized by background weight only, so its value scaled with the raw
feature magnitude — which on this model is dominated by an arbitrary, content-free
positional/segment floor of order several hundred. That made the absolute penalty
huge and impossible to weight intuitively; see §5.)

Term by term:

| symbol | meaning | effect in the sum |
|---|---|---|
| $E_{i,j}$ | encoder's response at cell $(i,j)$ (content features only) | the thing we want small in background |
| $\widetilde{M}_{i,j}$ | fraction of that cell that is background | gates the numerator: $0$ over anatomy ⇒ **anatomy contributes nothing to the penalty** |
| denominator | total energy over all valid cells | turns the term into a *fraction*, so the value answers "how much of my energy is background-bound" |

Because $\widetilde{M}_{i,j}=0$ on every anatomy cell, those cells are
**multiplied out of the numerator entirely**. The penalty cannot reward or punish
what happens over the lungs/heart/chest wall; it only measures how much of the
total energy leaked into the confident-background frame. The mask's guarantee
(never flags anatomy) is inherited directly by the loss.

---

## 4. How it joins training

The total loss is the classification loss plus this term, and we want the penalty
to stay a **fixed fraction of the classification loss** $\mathcal{L}_{\text{cls}}$
throughout training. So we scale it by the *detached* classification loss:

$$
\mathcal{L} \;=\; \mathcal{L}_{\text{cls}} \;+\;
\lambda\,\operatorname{sg}[\mathcal{L}_{\text{cls}}]\,\mathcal{L}_{\text{bg}},
$$

where $\operatorname{sg}[\cdot]$ is stop-gradient (`.detach()` in
[`train_step`](../training/utils/train.py)). Read it as: the penalty contributes at
most $\lambda$ of the classification loss (it reaches exactly $\lambda\,\mathcal{L}_{\text{cls}}$
only if *all* energy is background-bound, $\mathcal{L}_{\text{bg}}=1$), and
proportionally less when the model already keeps background quiet.

- $\lambda = 0$ → the term vanishes; training is exactly as before (this is the
  default — the feature is opt-in).
- $\lambda > 0$ → at every step the optimizer gets a gradient that nudges the
  encoder's weights to **lower the energy it produces in background cells.**

**Why detach the classification loss?** Two reasons. (1) Without it, a fixed
$\lambda$ does *not* give a fixed ratio: $\mathcal{L}_{\text{cls}}$ falls as the
model converges (e.g. $0.1 \to 0.03$) while $\mathcal{L}_{\text{bg}}$ does not, so an
uncoupled penalty would grow to *dominate* the converged tail — exactly when you
want it to fade. (2) The stop-gradient means the coupling only sets the penalty's
*scale*; it injects no gradient into the classifier through that $\mathcal{L}_{\text{cls}}$
factor, so the classification objective itself is untouched.

### What the gradient actually does

Because $\mathcal{L}_{\text{bg}}$ is now a ratio (numerator $N=\sum\widetilde{M}E$
over denominator $D=\sum E$), the gradient for a feature $F_{c,i,j}$ is

$$
\frac{\partial \mathcal{L}_{\text{bg}}}{\partial F_{c,i,j}}
\;=\; \frac{2\,F_{c,i,j}}{D}\,\bigl(\widetilde{M}_{i,j} - \mathcal{L}_{\text{bg}}\bigr).
$$

Plain reading: a cell is pushed **quieter** when its background fraction
$\widetilde{M}_{i,j}$ exceeds the current overall background fraction
$\mathcal{L}_{\text{bg}}$ (the black bars, the corners), and is left alone — or
gently *encouraged* — where $\widetilde{M}_{i,j}<\mathcal{L}_{\text{bg}}$ (anatomy,
$\widetilde{M}=0$). So the model minimizes the term by **moving energy out of the
background and into the body**, not merely by going globally silent. Over pure
anatomy the only effect is the mild "use me instead" pull from the denominator;
the numerator still cannot single out or punish any anatomy cell.

---

## 5. Why $\lambda$ stays modest

The mask is excellent on the collimation/black-bar side but has **rare false
positives** on the saturated-white side (it occasionally clips real tissue —
chin, neck, upper abdomen — sitting right at the image edge). On those rare cells
$\widetilde{M}>0$ even though it's really anatomy, so the penalty *would* push
down a legitimate feature.

A modest $\lambda$ is the safety margin. Now that $\lambda$ is the penalty's size
**as a fraction of the classification loss** (§4), it reads directly as a budget:

- **Where the mask is right** (the common case): a steady pressure capped at
  $\lambda\cdot\mathcal{L}_{\text{cls}}$ across thousands of steps still adds up to
  "stop using the bars."
- **Where the mask is wrong** (rare edge tissue): the push is too weak to teach
  the model to ignore that tissue, so we don't trade the background bias for a
  *new* "ignore the lower chest" bias.

In other words: we trust the mask enough to nudge, not enough to shove. We start at
$\lambda = 0.1$ (the penalty is at most ~10% of the ASL loss, and proportionally
less once little energy is background-bound) and tune by watching (a) val AUC
doesn't drop, especially on off-lung classes, and (b) the Grad-CAMs stop lighting
up the corners.

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
3. Penalty = **fraction of total feature energy that sits in background**:
   $\mathcal{L}_{\text{bg}} = \sum\widetilde{M}E / \sum E \in [0,1]$, using only the
   content features (pre positional/segment), so it's scale-free.
4. Add $\lambda\cdot\operatorname{sg}[\mathcal{L}_{\text{cls}}]\cdot\mathcal{L}_{\text{bg}}$
   to the loss: the penalty is pinned at $\le\lambda$ of the classification loss
   throughout training. Start $\lambda = 0.1$.
5. Anatomy cells have $\widetilde{M}=0$, so they can never be *punished*; the model
   minimizes the term by moving energy out of the background frame into the body.
