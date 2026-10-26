# io8 / blk8: q8 queue (bridge CHL Log #29 steps 2-4; built 2026-10-26)

Four runs, queue `~/mind_the_chl_gap/logs/q8/` (not started by the builder). Recipe for all: batch 8 (OOM fallback 6 -> 4 via `train_oom`),
60-epoch cosine, lr 1e-3, wd 1e-4, λ 0, ckpt-from 10, `--preds-f32`. Flags: `logs/q8/common.sh` ($IO = io7 BASE, $BLK = blk7 BASE).

| run | flags | what is new |
|---|---|---|
| io7_physmem_s1 | `$IO $SPATIAL $PHYSMEM --seed 1` | nothing (seed 1 of io7_physmem; config equal apart from the seed, checked) |
| io8_physmem_rwloss | `$IO $SPATIAL $PHYSMEM --rw-loss` | real-cloud loss reweighting (Phase 1 run 4, #25) |
| blk8_spatial_p12 | `$BLK $SPATIAL --ptok-patch 12` | token patch 12 px (~54 km) instead of 4 |
| blk8_spatial_bsmem | `$BLK $SPATIAL --bsmem` | Black Sea physics memory |

Order: s1 -> rwloss -> p12 -> bsmem, each train then post. Postrun: `analyses/io7/postrun.py` (IO profile for io*, blk4 profile for blk8_*,
results in analyses/io7/results/ and analyses/blk7/results/, FINDINGS headers `## io7 postrun:` / `## blk7 postrun:`). IO posts add seed
context lines (#29 rule): `--seed-group io7_physmem=io7_physmem,io7_physmem_s1` and `--seed-group io6_spatial=io6_tokdec_blend_spatial{,_s1,_s2}`
(mean and range of each group's val / test MAE and real-cloud rw MAE, run − mean, inside / outside the range). No seed group exists for the
Black Sea runs (blk4_tokdec_blend_spatial has one seed); their entries compare against single runs only.

## Assumptions
- rw-loss target: real-cloud px (ocean, not observed) of the TRAIN days, with cells computed under the same input visibility as that day's
  training fake clouds (fake-hidden px removed from the visible set), as postrun.py does for val / test. Alternative not used: real-cloud
  cells with no fake clouds applied (the deployment setting).
- rw-loss distributions: 3 training donor draws (seasonal donors ±15 DOY, as train_epoch) from a separate generator (seed + 1000003), so
  the training draws and batch order equal those of io7_physmem (seed 0). Weights are fixed for the whole run.
- rw-loss weight cap: 10 (clip, renormalise to mean 1 under the fake distribution, repeat until max ≤ 10). Cells with no fake px get 0.
- rw-loss scope: only the day-t hidden-px pixel term is weighted. Visible px (--vis-weight 0.1), the t±1 heads and the gradient-difference
  term are unchanged. Denominator = sum of the weights (weighted mean). Training only; val selection and scoring unchanged.
- Distance bins on days with no visible px: the GPU binner gives 32+ everywhere; scipy's EDT (postrun) measures to a phantom corner
  there. IO val / test: 0 such days (2,921 days). blk4 scored days: 7 of 365 (19,505 of 2,164,242 hidden px) - affects only the blk7
  postrun's distance / rw numbers, not the runs.
- Patch 12: TokDecNet needs P even; 12 is supported. Grid 177 x 273 padded to 180 x 276 -> 15 x 23 patches (217 ocean patches; P 4: 1,630).
  12 px x ~4.5 km ≈ 54 km. Everything else as blk4_tokdec_blend_spatial (patch conv kernel 2P = 24, stride 12; decoder P² C outputs).
- bsmem (chl_cnn/bsmem.py): physmem's token net (PhysMemTokDecNet, zero-init memory conv) with 12 other channels, window ±8 (blend on all
  17 days; at init equal to TokDecNet blend + spatial, tested). Windows: W30 = [t-30, t-1], W60 = [t-60, t-1] (nested; "30- and 60-day past
  means"). Channels:
  - river discharge × proximity (6): Danube, Dniester, Dnieper-Bug; Q_r(day) = Σ friver × px area over the river's mouth px (px with
    friver > 0 on any day whose nearest of the 8 store mouths is r, within 40 km: 5 / 3 / 3 px; mean Q 5.86e6 / 0.35e6 / 1.48e6 kg s-1).
    Channel = z(log mean Q over W; z over train days) × patch mean of exp(-river_dist_r / 100 km). Choices: 40 km mouth radius, 100 km
    proximity e-folding scale.
  - MLD deepening (2): W30 mean[t-15, t-1] − mean[t-30, t-16]; W60 mean[t-30, t-1] − mean[t-60, t-31] (mlotst, + = deepening).
  - surface heat flux (2): rsntds (net downward shortwave) anomaly vs the train-day DOY climatology (31-day smoothed), mean over W.
  - coverage (2): fraction of each window's calendar days missing from the store.
  A mean needs ≥ half the window's days (half-windows ≥ half), else 0; dmld / swr z-scored over train days and ocean patches, all clipped ±6.
  Missing: GloFAS (not on disk; store friver = CMEMS river flux used instead); net heat flux hfds and its longwave / latent / sensible parts
  (store has only rsntds). rsntds: 5 all-NaN days. Patches without mlotst / rsntds (e.g. Azov, Marmara) are 0 (valid frac ~0.82).

## Code changes (backups `*.pre_io8.bak` next to each edited file)
- New chl_cnn/rwloss.py: `--rw-loss [--rw-cap 10] [--rw-draws 3]`; GPU binner (disk convolutions for the distance bins, stencil
  convolutions for the key counts), weight capping, report. Writes `<run>/rw_loss.json`; config.json `rw_loss_info`.
- New chl_cnn/bsmem.py: `--bsmem` (store blk4 only), BSMemTable, daily fields cache in ~/mind_the_chl_gap/data/blk8/; config.json `bsmem_info`.
- chl_cnn/train_pinn.py: flags / checks / resume defaults of both modules; NB also holds ±1..4 with --rw-loss; rw precompute before
  training; data_loss uses the weights with --rw-loss in training; model build `physmem or bsmem` (mem_ch passed); predict_z sets mem_in for
  both; docstring lines. Default path: unchanged code paths (regcheck below).
- analyses/io7/physmem/leak_physmem.py: `--far K` (default 12 = unchanged); bsmem leak test uses 6 (inside its ±8 window).
- analyses/io7/postrun.py: `--seed-group NAME=run1,...` (default none = unchanged; identity checked on io7_physmem: results JSON equal apart
  from the added seed_groups); cfg_note adds physmem / rw_loss / rw_cap / bsmem / ptok_patch for runs that use them.
- New analyses/io8/test_io8.py (CPU).

## Checks
- CPU: analyses/io8/test_io8.py ALL PASS (binner == postrun EDT / keys.py cells on every non-visible px of days with a visible px; weights;
  bsmem past-only on synthetic and real store data; bsmem net == TokDecNet at init; P 12 forward / backward on 177 x 273);
  analyses/io7/physmem/test_physmem and analyses/blk7/test_blk7 ALL PASS.
- rw-loss weights (IO, 3 draws x 5,540 training days, 49.2 M fake / 201.2 M real-cloud px): max raw ratio 8.29, so the cap of 10 is not
  active; Kish effective sample fraction 0.441; TV(fake, real) 0.314 -> 0.000 after reweighting; uncovered real mass 0. Largest weights:
  32+ px / 0 keys 8.29, 16-31 px / 0 keys 3.91; smallest: 1 px / 60+ keys 0.32. Precompute 6 s.
- IO regression (shared code changed): logs/io7/regcheck.sh OK (default conv val 0.1301648 / test 0.1305480 path, io6_tokdec_blend_spatial
  0.1270596 / 0.1275163); logs/blk7/regcheck_extra.sh: eval-only io7_physmem and io6_tokdec_krig reproduce their saved val / test MAE
  (|diff| 0, 0). Output: logs/q8/regcheck.out, regcheck_extra.out.
- bsmem leak test (logs/q8/bsmem_leak.sh, 20-step init, ±8 window, --far 6; day 2021-05-20): hidden, masked t+2 / t+6 / t-6 change no
  recorded tensor and no prediction (all exactly 0); obs t+2 / t+6 change tokens and predictions: NO LEAK, controls OK
  (analyses/io7/physmem/results/leak_q8_leakinit_bsmem.json). Physics memory past-only: CPU tests (synthetic and real store).
- rw weights are reproducible: smoke run and an earlier 5-batch run gave identical weight tables.
- GPU smokes (logs/q8/smoke.sh: 2 epochs of the 60-epoch schedule, final val / test scoring, preds.npz), batch 8, no OOM:

| run | batch | peak GB alloc / reserved | s / epoch (ep 1, 2) | est. hours (60 ep + eval) |
|---|---|---|---|---|
| io7_physmem_s1 | 8 | (io7_physmem: 14.5 / 21.4) | (io7_physmem: 173) | 2.9 (io7_physmem wall time) |
| io8_physmem_rwloss | 8 | 14.5 / 21.3 | 177, 176 | 3.0 |
| blk8_spatial_p12 | 8 | 9.8 / 10.3 | 54, 52 | 0.9-1.0 |
| blk8_spatial_bsmem | 8 | 13.0 / 14.7 | 94, 92 | 1.6 |

  io7_physmem / io8_physmem_rwloss reserve 21.3-21.4 GB at batch 8 (rule: ≤ 21.5); a mid-run OOM retries at batch 6 (`train_oom`).
  Queue estimate: ~8.5 h training + ~3 min postrun per run: ~9 h. Smoke, leak and regcheck run dirs were deleted.
- Fixed during the build: leak_physmem.py's new case lists first reused the name `pos` (the scored position): TypeError after all 7 eval
  runs; renamed, rerun passed.
