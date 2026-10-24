# IO gap-filler improvement track (io2_*): plan and pre-registered decision rules

Written 2026-10-21, before any io2_ run was launched. The rules in sections 3 and 4 are fixed; results do not change them.
Anything decided later is logged in section 6 with its date and reason.

## 1. Protocol (every run)

- Store `io`; donor fake clouds (`--donor-window 15`); context days carry the donor's day d+k mask (`--context-clouds persist`).
- Split: val years 2001/07/13/18, test 2004/10/16/20; `--val-stride 1` (1458 val / 1463 test days); fixed hidden pixels (`--eval-seed 0`).
- Physics loss off (`--lam 0`; `--phys upwind` only for the diagnostic residual, as in the baseline).
- Baseline `t3d_dil_c4_60ep_gdl100` (seed 0): val 0.130165 / test 0.130548, best epoch 54 of a 60-epoch cosine schedule.
  It is not retrained. Its flags (from its config.json) are `BASE` below; every io2_ run starts from `BASE`.

```
BASE = --store io --features sst so air_temp u_wind v_wind sla mlotst u_curr v_curr --model temporal3d --t3d-backbone dilated
       --t3d-width 32 --t3d-depth 2 --t3d-kernel-t 3 --width 64 --context 4 --feat-context 4 --grad-weight 100 --clim --coords
       --residual --vis-weight 0.1 --year-input --donor-window 15 --batch 8 --lr 1e-3 --wd 1e-4 --sched cosine --epochs 60
       --ckpt-from 10 --phys upwind --lam 0 --fig-every 10 --preds-f32
```

`--preds-f32` (new, opt-in) only stores float32 predictions in preds.npz (the default float16 is unchanged).

## 2. What every run reports (analyses/io_improve/score.py, from preds.npz and the IO store)

- val / test hidden-pixel MAE, RMSE, bias (pred − truth), ln CHL;
- gradient ratio (Σ|Δpred| / Σ|Δtruth|) and gradient correlation (uncentred, as the trainer) at lags 1, 2, 4 px along both axes,
  on pixel pairs with truth at both ends and at least one hidden end (the trainer's lag-1 definition, extended);
- MAE by Euclidean distance of the hidden pixel to the nearest visible (input) pixel of day t: 1, 2-3, 4-7, 8-15, 16+ px;
- MAE by age of the last real observation within t−8..t−1 (1 d, 2-3 d, 4-8 d, none; analyses/forecast/forecast_eval.persistence);
- n hidden pixels, n days, n gradient pairs, n non-finite predictions on scored pixels.

## 3. Comparison quantities (fixed)

- **Val at epoch N** of a run = the lowest val hidden MAE over epochs 11..N (the model `--ckpt-from 10` keeps: model.pt).
  The baseline's value at epoch N is computed the same way from its history.json: 0.132826 at N = 30 (epoch 28), 0.130165 at N = 60.
  The raw val at epoch N is reported alongside (baseline: 0.134145 at 30).
- **Δ** = option − baseline at the same N. For warm restarts: restart best val − 0.130165 (the source's score).
- **Seed band** (item 2). Seeds 0 (= the baseline), 1, 2 of the baseline recipe:
  - S30 = max − min over the 3 seeds of the val at epoch 30 (definition above, from each history.json);
  - S60 = the same at epoch 60 (the 60-epoch best val); test spread reported alongside;
  - SR = max − min of the restarted seeds' best val (after the item-1 winner restart).
- **"Helps"**: Δ < −S at the matched comparison: S30 for 30-epoch runs, S60 for full-length runs, SR for restarted runs.
  Provisional S = 0.001 only if the seed runs have not finished when a decision is due (by queue order they finish first);
  any provisional decision is re-checked against the final band and reported.

## 4. Decision rules (fixed)

1. **Item 1 winner** = the restart variant (1a or 1b) with the lower best val. It is applied in stage B whatever its Δ;
   whether it "helps" (Δ < −SR, once SR exists; −S60 before) is reported.
2. **Full-length runs.** Each tier-2 option (items 3, 5, 6, 7) that helps at epoch 30 gets one full-length run
   (`--epochs 60`, no stop) and then the item-1 winner restart. If both 5 and 6 help, only the lower-val one goes to full length
   (6 contains 5).
3. **Combination.** Only if at least two of {3, the better of 5/6, 7} help alone: one combined full run (union of their flags;
   7 already carries 5's memory features if 5 helped), then the item-1 restart. Nothing is combined otherwise.
4. **Item 7 inputs.** The attention stem's keys/query use the item-5 memory features only if item 5 helped (rule 3 above).
   ±15 days is run (30 epochs) only if the ±8 run fits in 22 GB reserved at batch 8 and runs at ≤ 2× the baseline's epoch time.
5. **Item 4 recipe.** Candidates (all after their item-1 restart, val after restart):
   the baseline recipe (seed 0 restart), each full-length option run, the combined run, and the 3-seed ensemble of the baseline
   recipe (mean of the restarted outputs). The recipe with the lowest val wins; the ensemble wins only if its val is lower than
   every single candidate's.
6. **Item 4 fit.** Train on train + val (`--train-on-val`), score test only, no val-based selection (`--no-select`: model.pt = last
   epoch). Epoch counts are fixed from the chosen recipe's val-selected runs: stage 1 = `--stop-epoch E1` on the 60-epoch schedule
   (E1 = its best epoch), stage 2 = the restart with `--stop-epoch E2` (E2 = the restart's best epoch), the restart's lr schedule
   replayed epoch by epoch from its history (`--lr-replay`, needed only if plateau wins). For the ensemble recipe each seed uses its own
   E1 / E2. Reported only as "train+val fit, test".
7. **Item 9 (diffusion)** is not built. Item 8's outcome is reported; the decision goes to the main session.

## 5. Items and flags

All runs: `BASE`, `--run io2_<name>`, seed 0 unless stated. Restarts use `--continue-from SRC --continue-weights-only`
(new, opt-in: SRC's model.pt weights, fresh AdamW / GradScaler; needed because the baseline predates resume.pt, and used for every
restart so all restarts start the same way) and `--ckpt-from 0`.

| item | run | flags beyond BASE |
|---|---|---|
| 1a | io2_r_plat | `--continue-from t3d_dil_c4_60ep_gdl100 --continue-weights-only --ckpt-from 0 --lr 1e-3 --sched plateau --plateau-patience 5 --patience 15 --epochs 60` |
| 1b | io2_r_cos | `--continue-from t3d_dil_c4_60ep_gdl100 --continue-weights-only --ckpt-from 0 --lr 3e-4 --sched cosine --epochs 20` |
| 2 | io2_seed1, io2_seed2 | `--seed 1` / `--seed 2` (full 60 epochs); then the item-1 winner restart of each (io2_seed{1,2}_r) |
| 2 | ensemble | mean (and median) of io2_<item-1 winner>, io2_seed1_r, io2_seed2_r (analysis only) |
| 3 | io2_w96_s30 | `--width 96 --stop-epoch 30` |
| 5 | io2_mem_s30 | `--memory --stop-epoch 30` |
| 6 | io2_anom_s30 | `--memory --residual-base anom --stop-epoch 30` |
| 7 | io2_tattn8_s30 | `--t3d-stem attn --context 8 --feat-context 8 --stop-epoch 30` (+ `--memory` if 5 helps) |
| 8 | io2_stoch | `--continue-from t3d_dil_c4_60ep_gdl100 --continue-weights-only --ckpt-from 0 --stoch --lr 3e-4 --sched cosine --epochs 20` |

**Item 5, memory channels** (`--memory`, chl_cnn/memory.py), 8 static channels after the stem, per pixel and day t:
- last valid ln CHL (z) among days t−5..t−60 and its age in days / 60 (no observation: value 0, age 1);
- first valid ln CHL among t+5..t+60 and its age / 60 (same fill);
- median ln CHL (z) over t−4..t+4 excluding t (8-day) and over t−15..t+15 excluding t (30-day), and the observation counts
  divided by 8 / 30 (no observation: median 0).
- "Valid" on day t+k = observed on t+k AND observed on the donor's day d+k (d = the donor of day t; if d+k is not in the store,
  the donor day d itself, exactly the `persist` rule of nb_masks). Days missing from the store count as missing.
- Leak test (analyses/io_improve/leak_test.py): shift day t's hidden truth by +1.0 ln CHL; the memory channels and the prediction at
  hidden pixels of day t must not change (max |Δ| = 0).

**Item 6, anomaly residual base** (`--residual-base anom`, default `clim` = unchanged): output = network + clim(day) + a(t), where
a(t) = w · median over t−15..t+15, k ≠ 0, of (ln CHL(t+k) − clim(t+k)) on the valid (donor-masked) days, z units, with
w = n / (n + n0), n the count of valid days, n0 = 3 (`--anom-n0`); a = 0 when n = 0. The same a(t) is added to the t−1 and t+1 heads.

**Item 7, temporal attention stem** (`--t3d-stem attn`, chl_cnn/tattn.py): per-day 2D conv embedding (shared across days, 3x3, two
layers) of [features, CHL value, visible mask] + learned day-offset embedding → per-pixel multi-head self-attention over the days
(missing days masked out) → attention pooling with a query from day t's token (plus the memory channels if 5 helps) → the same
static channels and dilated backbone as the baseline. Peak memory is checked in the smoke test; batch or width is reduced if needed
(and stated).

**Item 8, stochastic output** (`--stoch`, chl_cnn/stoch.py): 8 channels of N(0,1) noise per pixel enter the backbone's input conv and
every residual block through zero-initialised 1x1 convs, so the warm-started model equals the baseline at step 0. Loss: fair CRPS
with M = 2 samples per example, on the same pixels and weights as the baseline's L1 terms (hidden, 0.1 × visible, t±1 heads) and on the
gradient-difference pairs (same weight 100). Per-epoch val selection: fair CRPS from K = 4 samples (fixed noise seed). Final scoring
with K = 8 samples: preds.npz `pred` = the sample median; samples saved for the analysis (single-sample gradient ratio / corr at lags
1/2/4, MAE of sample median and mean, CRPS, spread–skill by distance bin).

## 6. Log of decisions taken during the run (dated)

- 2026-10-21 21:25, before any io2 result: if both the ±8 and ±15 attention runs help at epoch 30, only the lower-val one goes to
  full length (same rule as 5 vs 6). Rules 4.1-4.6 are applied by `analyses/io_improve/decide.py`, which logs every decision with its
  inputs to `results/decisions.json`.
- 2026-10-21, before any io2 result: the weights-only restart smoke (lr 3e-4, 30 batches) moved val (every 20th day) to 0.143; restarts
  start from a perturbed state, as on the upstream track (epoch-1 val above the source there too).
- 2026-10-21 22:55 (supervising agent, after the previous agent stopped at ~22:15): the item-7 smokes at batch 8 ran out of GPU
  memory (io2_smoke_tattn8: OOM in the self-attention block's forward with 19.4 GB allocated of 22 GB; io2_smoke_tattn15: OOM too).
  - Rule 4.4: the ±8 run does not fit at batch 8, so ±15 is not run (decide.py tattn15 logs "smoke missing" -> not allowed).
  - Section 5 allows reducing batch or width. Batch is reduced, width kept: the T×T attention maps do not shrink with the token
    width. Job 020a_smoke_tattn8_batch (inserted after io2_seed1) smokes ±8 at batch 4, then batch 2, with `--eval-batch 8`
    (eval chunking only). io2_tattn8_s30 (and any stage-B attention run) uses the largest of these that fits; lr and epochs unchanged,
    so the attention run takes 2× (batch 4) or 4× (batch 2) the optimiser steps of the baseline per epoch. This is a confound of the
    epoch-30 comparison and is reported with it.
- 2026-10-22 00:05: smoke results (60 training batches, `--memory --attn-query-memory`, `--eval-batch 8`): ±8 at batch 4 peaks at 14.5 GB
  reserved, 15.1 s per 60 batches (240 days); batch 2 13.4 GB, 9.7 s (120 days); baseline smoke at batch 8 13.4 GB, 6.8 s (480 days).
  Batch 4 is used: job 060 and decide.py OPTIONS (io2_tattn8 / io2_tattn15 full-length runs, so also a combination or item-4 fit that
  includes them) carry `--batch 4 --eval-batch 8`. Estimated epoch time ≈ 4.4× the baseline's.
- 2026-10-22 ~06:22 UTC: the instance idle-stopped during epoch 27 of io2_tattn8_s30. Cause: keepalive.sh's process pattern missed
  `python -W ignore -m ...` (the queue's launch form), so no activity was written during the io2 jobs. The main session fixed the
  pattern at 19:12 UTC (anchored, allows interpreter flags; tested) and relaunched the `keepalive` and `io2_queue` tmux sessions.
  io2_tattn8_s30 resumed from its resume.pt at its best epoch so far (24, val 0.1306); epochs 25-27 were rerun after the resume,
  so the logged epochs 25-27 before the stop are superseded. Nothing else changed (recorded by the supervising agent, 19:15 UTC).
- 2026-10-23 ~05:30 UTC: the user paused the io2 queue to free the GPU (io2_queue tmux session and job 200b_item4_io2_final_s0_r killed by the
  main session; io2_final_s0_r stopped after restart epoch 3 of 17, resume.pt kept). Not run: 200b (resumes), 201a/b, 202a/b, 209_item4_ensemble,
  990_report (summary, figures, runs_index, END). Relaunching with the queue header command resumes them in order; nothing else changed.
