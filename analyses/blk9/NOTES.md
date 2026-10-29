# blk9: two token levels on the Black Sea (q9 queue; bridge CHL Log #33; built 2026-10-28)

One run, queue `~/mind_the_chl_gap/logs/q9/` (not started by the builder). Source: #33 "two token levels, P = 4 for decoding and pooled
P = 12 for spatial attention". blk8_spatial_p12 (P 12 everywhere) scored val 0.14913 vs 0.14119 for blk4_tokdec_blend_spatial (P 4).

| run | flags (logs/q9/common.sh) | what is new |
|---|---|---|
| blk9_spatial_pool3 | `$BLK $SPATIAL --tokdec-spatial-pool 3` | the 2 spatial blocks attend over 3 x 3 pooled P 4 tokens (12 px, ~54 km) |

Base = blk4_tokdec_blend_spatial: `$BLK $SPATIAL` parse to its config.json apart from `dilations` (saved resolved, default None) and the
args added later (all at their defaults). Recipe: batch 8 (OOM fallback 6 -> 4 via `train_oom`), 60-epoch cosine, lr 1e-3, wd 1e-4, λ 0,
ckpt-from 10, seed 0, `--preds-f32`. Post: `logs/blk7/postrun.sh` (blk4 profile) -> analyses/blk7/results/, `## blk7 postrun:` FINDINGS entry.

## Design (chl_cnn/tokdec.py `_PooledSpatialBlock`)
- Temporal attention, pooling over days, decoder (Linear -> P² C, pixel shuffle), blend: unchanged at P 4 (1,630 ocean patches, 45 x 69 grid).
- Each spatial block, given the pooled P 4 tokens x (B, Np, D):
  1. Coarse token = unweighted mean of x over the ocean P 4 patches of its 3 x 3 block (mask-aware: land patches are not tokens, so they never
     enter the mean; weights are not by ocean-px count). Coarse cells with no ocean patch are not tokens. The patch grid is padded with land
     patches to a multiple of 3 (45 x 69 already is: 15 x 23 coarse grid; coarse ocean cells == blk8_spatial_p12's 217 ocean patches).
  2. LayerNorm, global self-attention over the coarse tokens (heads 4, width 128, fused SDPA) with a learned relative-position bias per
     (coarse offset, head) (zero init), output projection (zero init).
  3. Upsampling: nearest (each P 4 patch gets its coarse cell's attention output; no learned upsampler), added residually to x.
  4. MLP (pre-norm, output zero init) on the P 4 tokens, as in the P 4 block.
- At init the net equals TokDecNet without spatial blocks (zero-init projections), as the P 4 spatial run.
- Assumption: per-block pooling of the current tokens (each block re-pools its own input), not one pooling before both blocks.

## Code changes (backups `*.pre_blk9.bak` next to each edited file)
- chl_cnn/tokdec.py: `TokDecNet(..., spatial_pool=1)`; with spatial and spatial_pool > 1: buffers sp_parent / sp_count / sp_cell, coarse
  rel_idx, `_PooledSpatialBlock` (subclass of `_SpatialBlock`, same parameter names; rel table (2 nCy - 1)(2 nCx - 1)). Docstring section.
- chl_cnn/train_pinn.py: `--tokdec-spatial-pool S` (default 1 = off); parse error unless --tokdec-spatial N and the plain token decoder (no
  krig / wm-attn / physmem / bsmem / analog); RESUME_NEW_DEFAULTS; passed only when != 1; stem print line; docstring.
- analyses/io7/postrun.py: cfg_note adds tokdec_spatial_pool when != 1 (other runs unchanged).
- New analyses/blk9/test_blk9.py (CPU).

## Checks
- CPU: analyses/blk9/test_blk9.py ALL PASS: default path (spatial 2, pool 1) == pre-blk9 tokdec.py output with the same weights; 177 x 273:
  45 x 69 -> 15 x 23, coarse cells == P 12 ocean patches, forward / backward, every param gets a grad; pooling == brute-force loop mean on
  padded grids (S 3: 6 x 7 -> 6 x 9; S 4: -> 8 x 8) with all-land cells dropped; block == float64 brute-force reference; nearest upsampling;
  zero init == no-spatial net; day_ok masking; CLI. analyses/io8/test_io8 (--no-store) and analyses/io6_kriging/test_krig ALL PASS. Peak RSS 1.2 GB.
- IO regression: logs/io7/regcheck.sh OK (default conv; io6_tokdec_blend_spatial val 0.1270596 / test 0.1275163). Eval-only of
  blk4_tokdec_blend_spatial with `$BLK $SPATIAL` reproduces its saved val 0.1411894 / test 0.1339807 (|diff| 0, 0). Output: logs/q9/regcheck*.out.
- GPU smoke (logs/q9/smoke.sh: 2 epochs of the 60-epoch schedule, --fig-every 1, final val / test scoring, preds.npz), batch 8, no OOM. Stem line:
  "P 4 -> 1630 ocean patches of 45 x 69 ... spatial blocks 2 (pooled 3 x 3 -> 217 coarse tokens of 15 x 23)". Trainer max RSS 9.8 GB.

| run | batch | peak GB alloc / reserved | s / epoch (ep 1, 2) | est. hours (60 ep + eval) |
|---|---|---|---|---|
| blk9_spatial_pool3 | 8 | 11.9 / 13.0 | 76, 74 | 1.3-1.4 |
| (blk4_tokdec_blend_spatial, blk7 smoke) | 8 | 12.7 / 14.4 | 92, 90 | 1.6 |

  Queue estimate ~1.25 h training + ~5 min load / final eval + ~3 min postrun: ~1.4 h. Smoke and eval-only run dirs were deleted.
- Runner (logs/q9/runner.sh, copy of q8): keepalive at start, then a parallel pre-read of ~/mind_the_chl_gap/data/{blk4_store_2014_2024,
  static_bases} (files > 100 kB, 32 parallel dd; 20 files, 7.7 GB; lazy EBS reads after a restart), log "warm done"; tmux autostop when done.
- Launch: `~/.local/bin/tmux new -d -s q9_queue 'bash ~/mind_the_chl_gap/logs/q9/runner.sh'`.

## Result (q9, 2026-10-28 23:46 UTC; bridge #35)
blk9_spatial_pool3 val 0.14110 / test 0.13416 vs blk4_tokdec_blend_spatial 0.14119 / 0.13398 (n val 1,062,322 px, test 1,101,920).
No key val 0.25000 vs 0.25408; rw val 0.16420 vs 0.16460; lag-1 grad ratio/corr equal to 3 dp. Single seed.
