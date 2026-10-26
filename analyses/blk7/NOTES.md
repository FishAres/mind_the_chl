# blk7: IO attention / kriging / physics-memory configs on the Black Sea 4 km store (2026-10-26)

Five IO runs re-run on `--store blk4` with the IO attention recipe. Queue: `~/mind_the_chl_gap/logs/blk7/` (not started by the
builder). Post-processing: `analyses/io7/postrun.py` (blk4 profile) -> `analyses/blk7/results/`, `analyses/blk7/figures/blk7_runs.png`.

| blk7 run | IO source | stem flags (common.sh) |
|---|---|---|
| blk4_tokdec_blend_spatial | io6_tokdec_blend_spatial | `$SPATIAL` = tokdec ±8, blend, 2 spatial blocks |
| blk4_physmem | io7_physmem | `$SPATIAL --context 15 --feat-context 15 --physmem` |
| blk4_tokdec_blend | io4_tokdec_blend | `$BLEND` = tokdec ±8, blend |
| blk4_tokdec_krig | io6_tokdec_krig | `$KRIG` = tokdec ±8, learned kriging (triton, tf32) |
| blk4_tokdec_krig_signed | io6_tokdec_krig_signed | `$KRIGS` = `$KRIG --krig-signed` |

## Assumptions
- Recipe (user's choice): IO attention recipe: batch 8, 60-epoch cosine, lr 1e-3, wd 1e-4, λ 0, seed 0, ckpt-from 10, same tokdec
  settings (P 4, dim 128, ch 32, heads 4, layers 2), grad-weight 100, vis-weight 0.1, donor-window 15, `--preds-f32`.
- Features: `sst so mlotst sla u_curr v_curr u_wind v_wind` (8; no air_temp). sst -> store variable sst32 (trainer's sname).
- Store-specific settings kept from blk4_t3d_c4_gdl100_lam0_b16_50ep: `--crop` (177 × 273, 23,606 ocean px), `--val-stride 4`
  (182 val / 183 test scored days), `--eval-batch 8`, `--phys semi-lagrangian` (only the diagnostic residual at λ 0; upwind is not
  usable at Black Sea Courant numbers), `--feat-device gpu`, `--context-clouds persist`, `--lag-max` store default 10, split val
  2017 2021 / test 2019 2023, lat ascending (handled by the trainer's signed grid metrics).
- Batch: 8 unless the smoke OOMs (then fallback flag, then 6, then 4; user's decision 2026-10-26). Queue: a mid-run CUDA OOM retries
  once at the next smaller batch as `<run>_b<k>` (common.sh `train_oom`). Batch per run: see the smoke table below and the run's
  FINDINGS entry (cfg line).
- physmem on blk4: u_wind / v_wind are wind stress (Pa, CMEMS tauuo / tauvo), used directly as tau (no drag law); w_E from that tau
  with the physics.ekman formula. Pixel constants unchanged: coastal band 4 px (~18 km at 4.5 km; IO ~100 km), land smoothing
  sigma 2 px, patch P 4 (~18 km patches; IO ~100 km). w_E equatorial cut (|lat| < 5°) never applies (40-47° N). Every val / test
  day has the full 60-day history (record starts 2014-01-01; val 2017).
- Kriging stencil (radius 3-6 px, |dt| ≤ 4) unchanged in pixels, so it spans ~4.5x less distance than on IO.
- Postrun: distance bins and key bins in pixels as on IO. Real-cloud reweighting target = real-cloud px (ocean, not observed) of
  the scored val days (every 4th). dist_gain reference = blk4_t3d_c4_gdl100_lam0_b16_50ep (conv ±4, batch 16, 50 epochs: recipe
  differs, noted in each entry). Δ references: the other blk7 runs that have preds.npz, plus the two blk4 conv runs.

## Code changes (backups `*.pre_blk4.bak` next to each edited file)
- chl_cnn/physmem.py: `ekman_from_stress()`; `daily_fields(..., wind_is_stress=False)` and `MemTable(..., wind_is_stress=False)`:
  with True, tau = (u_wind, v_wind). Cache key unchanged for the IO default (" stress" appended otherwise). Docstring section.
- chl_cnn/train_pinn.py (one call): MemTable gets `rd(sname(f), t)` (blk "sst" -> "sst32") and `wind_is_stress=store in (blk, blk4)`.
  IO: sname is the identity, so the call is unchanged.
- chl_cnn/krig.py: KrigStem accepts n_feat ≤ 9; with < 9 the gather volume's physics block is zero-padded to 9 channels (Triton /
  torch paths and the 17 key features unchanged; the padded key column is always 0). Before, n_feat 8 failed the assert (and
  `xf[:, :9]` would have taken the CHL channel as a 9th physics field). IO (9) unchanged.
- analyses/io7/physmem/leak_physmem.py: store / crop / split from `--base-args` via analyses/blk7/storectx.py (IO default unchanged).
- analyses/io7/keys.py: `compute(c=...)`, `load(cache, c=...)` for a strided context; no-argument IO behaviour unchanged.
- analyses/io7/postrun.py: store profiles (io / blk4) picked from the run's config.json; `--res-dir`; blk4 profile as above, FINDINGS
  header `## blk7 postrun: <run> (Black Sea 4 km; ...)` with a recipe note line, overview figure. IO profile unchanged.
- New: analyses/blk7/storectx.py (store context, trainer-equivalent crop / split / hidden masks; strided scoring context with a dist /
  age cache in ~/mind_the_chl_gap/data/blk7/), analyses/blk7/figures.py (overview figure, Okabe-Ito), analyses/blk7/test_blk7.py
  (CPU: 8-feature kriging == 9 with a zero feature, stress Ekman == physics.ekman, physmem fields invariant to latitude order).

## Checks
- CPU: analyses/blk7/test_blk7.py ALL PASS; analyses/io7/physmem/test_physmem, analyses/io7/wmattn/test_wmattn,
  analyses/io6_kriging/test_krig ALL PASS after the edits.
- IO regression: logs/io7/regcheck.sh OK (default conv val 0.1301648 / test 0.1305480; io6_tokdec_blend_spatial 0.1270596 /
  0.1275163); logs/blk7/regcheck_extra.sh: eval-only io6_tokdec_krig and io7_physmem reproduce their saved val / test MAE (|diff| 0).
- IO postrun identity: `postrun io6_tokdec_blend_spatial --force --no-findings --res-dir <tmp>` is byte-identical to
  analyses/io7/results/io6_tokdec_blend_spatial.json (references recomputed too).
- blk4 postrun: reference runs re-scored from preds.npz match their trainer val MAE (0.156113 vs 0.1561123; float16 preds); tested on
  the spatial smoke run (val 0.215278 = trainer 0.2152783; entry and figure rendered in a temp dir).
- Leak test (logs/blk7/physmem_leak.sh, batch 8, ±15 physmem): hidden, masked t+2 / t+12 / t-12 change nothing (all recorded tensors
  and the prediction exactly 0); obs t+2 / t+12 change the prediction and the tokens: NO LEAK, controls OK
  (analyses/io7/physmem/results/leak_blk7_leakinit_physmem.json; day 2021-05-20).
- physmem summaries on blk4 (train days, ocean patches): curl / w_E / SSTA / dMLD valid ~0.80-0.83 (patches without physics, e.g. Azov /
  Marmara, are 0), alongshore stress 0.23 (coastal patches).
- GPU smokes (logs/blk7/smoke.sh: 2 epochs of the 60-epoch schedule, --fig-every 1, final val / test scoring, preds.npz), batch 8:

| run | batch | fallback | peak GB alloc / reserved | s / epoch (ep 1, 2) | est. hours (60 ep) |
|---|---|---|---|---|---|
| blk4_tokdec_blend_spatial | 8 | none | 12.7 / 14.4 | 92, 90 | 1.6 |
| blk4_physmem | 8 | none (±15 fits; no --physmem-stride2) | 15.2 / 17.5 | 119, 118 | 2.0 |
| blk4_tokdec_blend | 8 | none | 11.8 / 12.9 | 74, 73 | 1.3 |
| blk4_tokdec_krig | 8 | none | 11.7 / 12.8 | 102, 100 | 1.7 |
| blk4_tokdec_krig_signed | 8 | none | 11.7 / 12.9 | 111, 110 | 1.9 |

  Queue estimate ~8.5 h training + ~5 min load / final eval per run + ~3 min postrun each: ~9 h. Smoke runs were deleted.
- Postrun Δ lines: each entry compares with the blk7 runs finished before it (queue order spatial, physmem, blend, krig, krig_signed),
  so the last entry (krig_signed) carries all of them; `bash logs/blk7/postrun.sh <run> --force` redoes an earlier one.
