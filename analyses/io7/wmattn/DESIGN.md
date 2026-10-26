# Water-mass attention (io7_wmattn; bridge #25 Phase 1 run 2), 2026-10-24

Status: built and CPU-tested; queued (io7 jobs leak_wmattn → smoke_wmattn → train_wmattn → post_wmattn, prio 40). Not run.
Code: `chl_cnn/wmattn.py`; flags `--wm-attn` (+ `--wm-keys 1024 --wm-lags 2 --wm-dim 64 --wm-heads 4 --wm-dv 16 --wm-sst-bins 8 --wm-qchunk 0`).
Run: `train io7_wmattn $SPATIAL --wm-attn` (io6_tokdec_blend_spatial + this stem; the own-pixel blend stays).

## Model
- Each ocean px p of day t attends over a pool of 1,024 visible observations drawn from the whole basin on days t−2..t+2.
- Keys: MLP(SST, SSS, MLD, SLA, CHL climatology, local visible density) at (q, d). Three fixed-structure terms are added as extra
  q/k dimensions, so one fused SDPA call computes them exactly:
  - a learned stationary kernel Σ α·cos(ω·Δ) in ΔSST, ΔSSS and Δclim (periods 0.5–8 z units);
  - a weak position bias of the same form in grid (Δy, Δx) at basin and half-basin periods;
  - a bias per (day offset, head); a "no key" slot. All four start at zero.
- Query: MLP(day-t physics, climatology, static channels, 3×3 conv of day t's visible CHL), the same as the kriging stem.
- Values: the exact anomaly a, plus a small embedding MLP(a, density, day).
- Outputs: dv + 4 = 20 channels, wired as in the kriging stem: attention output, Σw·a, spread, mass, entropy.
- Entropy uses log Z = none_logit − log w_none and Σw·s = q′·Σw·k′. These are exact rewrites, so no N × K tensor is stored.

## Key pool
- Candidates are taken only from px visible after donor masking: x3d's visible mask × day_ok × ocean. Day t's fake-cloud px and the donor-hidden
  px of context days are never candidates, and their CHL values are never read.
- The draw is stratified by day × 8 equal-width SST bins, spanning the sample's 0.5–99.5 % SST range. It is weighted sampling without
  replacement with weight 1/n_stratum (Efraimidis–Spirakis top-K).
- Why not uniform: a uniform draw follows cloud cover and the dominant water mass, so rare water masses (upwelling, fronts, SST tails) would get
  few keys. Equal-count (quantile) bins reduce to a uniform draw (checked in test 6). With equal-width bins each (day, SST band) gets
  ~K/40 keys while it has px.
- Region is not a stratum (position should enter weakly). Fewer than K visible candidates: extra slots get score −1e4, weight exactly 0.
- Training draws use torch.rand. Eval uses a generator per sample, seeded from an integer hash of day t's 9 physics fields. The seed is fixed
  per day, does not depend on batch order, and no CHL value enters it.

## Cost (estimates; no GPU measurement yet)
- CUDA uses memory-efficient SDPA only, in fp32 (q/k dim 64, v dim 88), at B 8, N 16,663, K 1,025.
- Stem activations are ≲ 1.5 GB, so the expected total is ~13 GB allocated (io6 spatial: 11.2 GB).
- The attention is ~0.2 TFLOP per batch, forward + backward. Expected ~140–170 s/epoch (io6 spatial: 128), ≈ 2.5 h for 60 epochs.
- Fallback `--wm-qchunk 2048` keeps the same model. It processes queries in chunks with math SDPA, checkpointed per chunk, at ~1.3 GB transient.
  It is used automatically if the efficient kernel refuses the forward, and by the smoke retry on OOM or an SDPA kernel error.

## Tests
`python -m analyses.io7.wmattn.test_wmattn` runs on the CPU with tiny shapes. All pass:
- the pool holds only visible candidates (50 draws); garbage CHL on non-visible px leaves outputs and pool bit-identical (control changes them);
- eval forwards are reproducible and independent of batch order;
- estimate, spread, mass and entropy match an explicit softmax (max error 7e-7);
- chunked output equals fused output; every parameter gets a gradient; WmTokDecNet runs with the blend and spatial blocks on.

The queue leak test (`analyses/io7/wmattn/leak_wmattn.py`, the io3/io6 four cases) also compares the key pool, and exits non-zero on a leak.

## Open
- The GPU kernel choice (efficient attention in fp32 with dqk ≠ dv on sm86) is untested here. The smoke run shows which path ran
  (`wmattn:` log line).
- The pool is random during training. Its eval seed is a content hash, so two days with identical physics fields would share a draw.
