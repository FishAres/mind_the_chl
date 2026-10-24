# Learned kriging stem: design (draft, 2026-10-24; revised after bridge #21–#22)

Status: design only, not built. Discussed on the CHL bridge (UI #15 proposal; CC design #19; UI review #21, agreed by
the User; CC amendments #22, marked "CC amendment" below and pending the User). Explainer figure:
[analyses/kriging_explainer/](../kriging_explainer/).

## Revision 2026-10-24 (#21, #22)

- **Lags:** the first arm (K1) uses |Δt| ≤ 4 only; longer lags are left to the tokens, whose pooling already puts 0.26
  of its weight on |offset| > 4 when the blend is present (#17). (#21 change 3)
- **Stencil:** full disc of radius 5 px at |Δt| = 1–2, closing the unsampled radii 3–5 (#21 change 4). Same-day
  neighbours stay in the stencil (#21).
- **Back-advected centring** of each lag's stencil at p − s·u(p,t)·Δt (#21 change 1). CC amendment: sample with
  nearest-neighbour rounding, not bilinear grid_sample, so observed values and the visibility mask stay exact; s is then
  fixed (0.25), not learned through positions. On IO, tracked features are expected to move ~0.25–0.45 × the
  reanalysis currents (≈ 0.25–0.5 px/day), so the shift is ≤ ~2 px within |Δt| ≤ 4: an opt-in flag, off for the
  first IO arm, on for an ablation and for the Black Sea (4 km), where it matters more.
- **Growing spacing with lag** (#21 change 2): only for a K0 arm without tokens, which needs the longer lags.
- CC amendment: add a learned relative-position bias per (offset, head), the Swin-style scalar, alongside the offset
  features. It is the direct "learned variogram" and costs one scalar per offset per head.
- Key–key declustering: skipped. A signed second head (difference of two softmaxes, to allow kriging-like negative
  weights) is a low-priority option (#21).

## Idea in one paragraph

Each ocean pixel p on day t attends over individual visible observations (q, d) near it in space and time: same-day
neighbours, its own past/future values, and nearby pixels on other days. Attention weights play the role of a learned,
anisotropic, state-dependent kriging covariance; the values are the observed CHL anomalies, passed through exactly. The
output (an embedding, the weighted estimate and its spread) goes into the existing full-resolution dilated backbone next
to day t's own inputs. Arm A's own-pixel blend is the special case "stencil = own pixel on other days".

## Shapes (IO, 0.25°)

B = 8, T = 17 (t−8..t+8), H × W = 149 × 181 (16,695 ocean px), candidate stencil K ≈ 256 offsets, attention width D = 32,
4 heads.

## 1. Candidate keys: a fixed sparse space–time stencil

Dense top-k over the full ±8-day, ±16-px neighbourhood would be ~10⁴ candidates per pixel. Instead a fixed stencil of
offsets (Δt, Δy, Δx), denser near the target, sparser farther out (tunable; first version):

Revised stencil for K1 (#21/#22):

| |Δt| | spatial offsets (centred on p, or on p − s·u·Δt when back-advection is on) | count |
|---|---|---|
| 0 | disc radius 3 px, excluding the pixel itself | 28 |
| 1–2 | full disc radius 5 px | 81 each → 324 |
| 3–4 | 5 × 5 block + 8 points on a radius-6 ring | 33 each → 132 |
| total | | ≈ 484 |

Original draft (kept for K0, with spacing growing with lag as in #21 change 2):

| |Δt| | spatial offsets | count |
|---|---|---|
| 0 | disc radius 3 px, excluding the pixel itself | 28 |
| 1–2 | 5 × 5 block + 8 points on a radius-6 ring | 33 each → 132 |
| 3–8 | 3 × 3 block + 8 points on a radius-4 ring | 17 each → 204 |
| total | | ≈ 364 |

Masked softmax over the stencil's **visible** candidates (after donor masking; day t's hidden pixels are never visible),
plus one learned "no key" slot. No hard top-k, so everything is differentiable and the selection is learned through the
weights. The pixel itself at Δt = 0 is a key only where it is visible (no leak; output there is not scored).

## 2. Features

- **Query** at (p, t): the 9 physics fields and 7 static channels at p, plus a 3 × 3-conv summary of day t's visible CHL
  and mask around p → MLP → D.
- **Key** per candidate (q, d):
  - the offset (Δx, Δy, Δt), normalised;
  - the advected offset (Δx − s·u(p,t)·Δt, Δy − s·v(p,t)·Δt), with s a learned scalar initialised at 0.25 (the current
    scale from the tracer work);
  - the 9 physics fields at (q, d), and |SST(q,d) − SST(p,t)|, |SSS(q,d) − SSS(p,t)|;
  - the local visible density at (q, d) (fraction of its 3 × 3 neighbours visible), a cheap stand-in for kriging's
    declustering;
  → MLP → D.
- **Value** per candidate: [CHL z-anomaly at (q, d), offset features] → MLP → D_v = 16, plus the raw anomaly itself.

## 3. Outputs to the backbone (per pixel, full resolution)

- the attention output (D_v = 16 channels, after merging heads);
- the kriging estimate Σ w·a(q,d) and the spread √Σ w·(a − estimate)², from the raw anomalies (exact values);
- the weight mass on real keys (1 − w_no-key) and the weight entropy.

These 20 channels are concatenated with day t's 11 full-resolution channels and the 7 static channels. Same dilated
backbone, loss and recipe as io4 (L1 hidden + 0.1 visible + gradient loss 100 + the t±1 donor-hidden term, AdamW 1e-3,
cosine over 60 epochs, batch 8).

## 4. Making it fit (A10G, batch 8)

Gathering every candidate at once is ~B·H·W·K·F ≈ 8 · 27 k · 364 · 24 ≈ 1.9 G values: too much with backward. Instead
stream over stencil offsets in chunks of ~32 with an online (flash-style) softmax: keep a running max, normaliser and
weighted value sum per pixel, so memory is O(B·H·W·D) rather than O(B·H·W·K·D). Each chunk is a shifted view of the
(T, H, W) feature volumes (no index gather), wrapped in activation checkpointing. Target: ≤ ~150 s/epoch, ≤ 16 GB.

## 5. Leakage checks

As for io3/io4: perturbing day t's hidden values or the donor-hidden observed pixels of a context day must change
nothing (keys, weights, outputs, predictions); a positive control (all observed pixels of t+2) must change predictions.
Plus a check that the stencil never reads across the padded border or land as a visible key.

## 6. Arms (one seed each, 60 epochs)

| run | context sources | question |
|---|---|---|
| K1 `io6_tokdec_krig` (first) | io4 tokens + decoder, with kriging (|Δt| ≤ 4, revised stencil, no back-advection) replacing the own-pixel blend | does it beat tokens + blend (0.1276)? |
| K1-adv | K1 with back-advected centring (nearest-neighbour, s = 0.25) | does the shift matter at 0.25°? |
| K0 `io6_krig` (if cheap) | kriging only (no tokens), ±8 days, spacing growing with lag | exact-value space–time attention vs arm A and tokens-only |

Later, only if K0/K1 are interesting: no advected-offset feature; own-pixel-only stencil (should reproduce arm A's
blend); a Lagrangian stencil centred on s·ū·Δt; the Black Sea at 4 km, where motion and fronts are resolved.

## 7. Diagnostics

- the usual scores: val/test MAE, gradient ratio/corr at 1/2/4 px, distance and two-sided age bins; expected gains
  where the old per-pixel attention still beat arm A (2–15 px, 4–8 d and "none"). The distance bins measure distance to
  day t's visible pixels, so a pixel 16+ px from any day-t observation can still have other-day keys in the stencil:
  report the gain split by "has ≥ 1 visible key" vs "no key" as well;
- the learned "variogram": mean weight as a function of (Δt, Δx, Δy), and its anisotropy relative to the current
  direction and the SST gradient;
- weight vs |SST difference|, and the learned s;
- spread vs absolute error by bin (is the spread a usable uncertainty?);
- per-pixel weight maps for the attention explorer notebook.

## Open questions

- Stencil size and shape: ~256 vs ~364 candidates; whether same-day neighbours (Δt = 0) belong in the stencil or are
  left to the backbone's convolutions.
- Whether keys should attend to each other (true declustering): cost K² per pixel, so only as a small second stage.
