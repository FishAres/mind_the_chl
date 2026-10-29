"""PINN: the train2.py dilated CNN with a 3-day output head and a transport-residual loss.

Model output: ln CHL (z-scored) for days t-1, t, t+1 (3 channels). Inputs exactly as train2.py.

Losses
  data     hidden-pixel loss on day t (as train2.py) + --nb-weight x the same on days t+-1
           (their hidden pixels are those the donor's cloud sequence hid from the input)
  physics  r = (C[t+1] - C[t-1]) / 2dt + u.grad C[t] - K lap C[t] - S      [ln CHL per day]
           on every ocean pixel with a full 5-point ocean stencil, finite currents and both
           neighbour days present, including pixels under real clouds. Observed pixels that were
           visible in the input are pasted back into C before the residual (hard constraint).
           Upwind first-order advection, index-space velocities (cells per day). --co-max drops
           pixel-days whose Courant number exceeds it (0 = keep all).
  total    data + lambda(ep) * weighted mean|r| ;  lambda ramps 0 -> --lam over --phys-warmup epochs.

Physics-only inputs
  --no-chl-input  drop the CHL value channels (day t and context days): the model sees only the physical
                  features, climatology and coordinates. The data loss is then taken on all observed pixels
                  of t-1, t, t+1 (no visible/hidden split); scoring stays on the fixed hidden pixels.
  --cloud-input   keep the 0/1 observation-mask channels (day t and context days) as inputs
  --no-chl-today  drop only day t's CHL channels; the CHL context days t+-1..K stay as inputs
                  (data loss on all observed pixels of day t, as with --no-chl-input)
  --feat-context K  add the physical features of days t+-1..K as extra channels

3D-convolution front-end (2026-10-14; --model temporal3d, models.TemporalStem3D / Temporal3DNet)
  The window t-K..t+K (--context K, with --feat-context K required equal) is stacked on a day axis: every day carries the
  same channels [features (F), CHL value (0 where not visible), visible mask], missing neighbour days are zero.
  --t3d-depth x [Conv3d (kernel (--t3d-kernel-t, 3, 3), --t3d-width channels) -> GroupNorm -> GELU], then a learned
  (T, 1, 1) conv collapses the day axis; the static 2D channels (climatology of day t, coords, doy, year) are
  concatenated and the result is fed to the usual --t3d-backbone {dilated, unet} (--width, --unet-depth as before,
  3-day output head). Losses, fake-cloud protocol and scoring unchanged; context-day masks follow --context-clouds.
  Runs land in models/pinn/t3d/<run>. --no-chl-input / --cloud-input / --diff-fill are not supported.

Causal ("forecast") window for temporal3d (2026-10-15)
  --context-before B / --context-after A  the window is t-B..t+A (both default to --context: the symmetric window above,
                  unchanged); --feat-context must equal max(B, A). Example: days t-8..t-1 as a forecast of day t =
                  --context-before 8 --context-after 0 --feat-context 8 --no-chl-today --context-clouds none.
  --no-chl-today  day t keeps its feature channels; its CHL value and visible-mask channels are zero (uniform per-day layout).
                  The data loss (and the gradient loss) then covers all observed pixels of day t (data_loss / grad_terms);
                  metrics.json "cnn_observed" scores all observed day-t pixels, "cnn_hidden" the usual fixed hidden pixels.
  --context-clouds none   context days carry their real observation mask (nb_masks: m = observed & day exists; no donor
                  clouds), so past days are seen as they were observed.
  3-day output head: kept. A neighbour head (t-1 / t+1) gets a data loss only when its day is inside the input window, and
                  then only on that day's observed pixels not visible in the input (the existing rule). With
                  --context-clouds none every observed t-1 pixel is visible, so the t-1 term is empty; t+1 lies outside the
                  window when A = 0 and its term is skipped. Neither head can train on a pixel it sees. The t+1 head only
                  enters the physics residual (which pastes the observed t+1 values; a loss, never an input).
  Baselines for this set-up: analyses/forecast/forecast_eval.py (persistence within t-8..t-1 and DOY climatology).

Donor exclusion and eval-only scoring (2026-10-16)
  --donor-exclude D  no fake-cloud donor day within the calendar window [t-D, t+D] of its target day t (dates from TIMES,
                  not indices; clouds.py). Applies to the training donors (uniform draw and the --donor-window seasonal draw)
                  and to the fixed val/test donors (C.fixed_hidden), so D > 0 is a different hidden-pixel benchmark from
                  every run with the default D = 0 (unchanged draws, bit for bit). Recorded in config.json (donor_exclude).
                  Nowcast example (window t-8..t, day t with its visible pixels, past days with their real observations):
                  --context-before 8 --context-after 0 --feat-context 8 --context-clouds none --donor-exclude 8.
  --eval-only --init-from RUN   no training: load models/.../RUN/model.pt and score val/test (metrics.json, preds.npz) under
                  this run's arguments (repeat RUN's architecture / input arguments; --donor-exclude etc. may differ).
                  --epochs is ignored for training; config.json best_epoch 0, best_val_mae NaN.
  --continue-from RUN   warm restart (2026-10-17): load RUN's resume.pt (model, AdamW moments, GradScaler, src_ekman at RUN's
                  best val epoch), then train --epochs more epochs under a fresh lr schedule starting again at --lr, with a
                  fresh history. Repeat RUN's architecture / input arguments.

Learning-rate schedules (--sched; 2026-10-21 added plateau, every default unchanged)
  cosine   CosineAnnealingLR over --epochs (default without the recipe); none = constant --lr (recipe default).
  plateau  ReduceLROnPlateau(mode min, factor --plateau-factor 0.5, patience --plateau-patience 5, min_lr --min-lr 1e-6, torch's
           default relative threshold 1e-4) stepped after every val pass with the model-selection score: val all-pixel z MSE with
           --recipe up2024, else val hidden-pixel MAE. Its state is in resume.pt (--return restores it). --patience stays the
           early-stopping patience. history.json "lr" = lr after the epoch's scheduler step (as before), "lr_used" = lr of the
           epoch's updates; config.json final_lr and n_lr_drops (plateau only).

Returning after an interruption (2026-10-15; --return, default on; --no-return starts fresh)
  Each time val MAE improves on the best since epoch 1 (independent of --ckpt-from), <run>/resume.pt is written atomically
  (resume.pt.tmp + os.replace) with the full training state at that epoch: model, AdamW, scheduler, GradScaler, src_ekman,
  epoch, best val / best epoch of the --ckpt-from logic, history up to that epoch, numpy / torch / cuda RNG states, vars(a).
  model.pt is unchanged (best epoch after --ckpt-from, final weights at the end).
  On start with --return: metrics.json present -> print "complete" and exit 0. resume.pt present -> compare its arguments
  with the current ones (all except RESUME_IGNORED = --return, --fig-every, --eval-batch; --epochs, --max-batches,
  --val-stride, --ckpt-from, --patience are checked), refuse on any difference, else load and continue from the epoch after
  the stored best epoch ("returning to best checkpoint at epoch N"). Later epochs of the interrupted run are discarded.
  Legacy fallback (runs from before this feature: model.pt + history.json, no resume.pt): model weights from model.pt,
  scheduler set to its epoch, history truncated to it, fresh optimizer / GradScaler / RNG, arguments not checked; prints
  "LEGACY RESUME: optimizer state reset at epoch N". config.json records resume_events, legacy_resume, legacy_resume_epoch.

2024 hackweek ("upstream") protocol (2026-10-20; every default unchanged; analyses/upstream/compare.py scores the runs)
  --store io_up   store_io_up.py: their box 32N-11.75S, 42-101.75E (176 x 240), their record days, split train 2015-01-01..
                  2017-12-31, val 2018-01-01..2019-01-01, test 2019-01-01..2020-01-01 (their inclusive slices; val and test
                  share 2019-01-01). Runs land in models/pinn/upstream/<run>; preds.npz holds float32 raw output on every pixel.
  --fake-mask shift10   their fixed mask for train, val and test: hidden = finite L3 at t & cloud code 0 at record t+10 (& our
                  ocean mask). No donor draws. Needs --context-clouds none (neighbour days unmasked, as theirs).
  --feat-today-only     temporal3d: features only on day t (zeros on the other window days).
  --doy-input / --land-input   static sin/cos(day of year) channels / a land (not-ocean) channel.

2024 hackweek training recipe (2026-10-20; --recipe up2024; default --recipe none leaves everything above unchanged)
  Trains a model the way geo-smart/mind-the-chl-gap U-Net_Tutorial.ipynb (cells 9, 12, 15, 18, 28-29) trained their U-Net.
  Needs --store io_up --fake-mask shift10 --context-clouds none and --model dilated, or --model temporal3d with
  --context 1 --feat-context 1. Rejects --clim / --coords / --residual / --year-input / --doy-input / --land-input / --geo-static /
  --grad-weight > 0 and the other input options. --features is ignored (their u_wind v_wind sst air_temp are used).
  inputs   their 13 channels from store_io_up.recipe_up2024: u/v wind, SST, air temp (float32 IO.zarr values), sin / cos of
           (days since 1900-01-01) / 365 * 2 pi, masked CHL (their hidden px NaN, no ocean cut), CHL of records t-1 / t+1 unmasked,
           0/1 land / real-cloud / valid-CHL / fake-cloud flags; numeric channels z-scored over train records [2015-01-01,
           2018-01-01), NaN -> 0. dilated: the 13 channels as is. temporal3d: 3-day window with 8 channels per day [4 features,
           masked CHL, real cloud, valid, fake cloud] filled on day t only, CHL t-1 / t+1 in the CHL slot of days t-1 / t+1 (other
           slots 0), and sin, cos, land flag as static channels after the stem (no information beyond the 13 channels).
  target   ln L3 z-scored with their all-record mean / std (up_norm.json), NaN -> 0; the model's ln output is z * std + mean.
  loss     plain MSE in z over every pixel of the 176 x 240 frame (land and real cloud with target 0 included), day-t head only;
           + lambda * the usual physics residual when --lam > 0 (unchanged, including its --phys-warmup ramp).
  training split train 2015-01-01..2018-01-01 inclusive (1097 days; 2018-01-01 is also a val day), val / test as io_up.
           Adam (eps 1e-7 as Keras, no weight decay), lr --lr constant (no scheduler), batch 4, max 50 epochs, train order from a
           tf.data-style shuffle buffer of 512 days refilled in time order (their shuffle(buffer_size=512), redrawn each epoch),
           early stopping on val all-pixel z MSE with patience 10, best-val-loss weights restored, no --ckpt-from delay.
           Forced: --batch 4 --ckpt-from 0 --wd 0 --val-stride 1 (--stop-epoch still works). Recipe defaults --epochs 50 --patience 10
           --sched none; explicit values of these three are kept (2026-10-21), e.g. --sched plateau / cosine --epochs 200 --patience 15.
           --continue-from RUN works with the recipe (2026-10-21): RUN's best-epoch resume.pt (model, Adam moments), fresh schedule
           from --lr; history.json "source" and config.json continue_from_source record RUN, its best epoch and best val z MSE.
  scoring  history.json val_up_mse / val_up_mae (all-pixel z, every val px); metrics.json scores.<split>.up_allpx (their Keras
           evaluate MSE / MAE pooled over all days x pixels, and the all-zero-prediction values); preds.npz raw ln output on
           every pixel. analyses/upstream/compare_recipe.py scores the runs.

IO improvement track (2026-10-21; io2_* runs, analyses/io_improve/PLAN.md; every default unchanged: --eval-only --init-from
t3d_dil_c4_60ep_gdl100 still gives val 0.130165 / test 0.130548)
  --continue-weights-only  with --continue-from RUN: load RUN's model.pt weights only (fresh AdamW / GradScaler), for sources without resume.pt
                  (runs from before --return). New parameters of this run's model that RUN lacks (--stoch noise convs) stay at their zero init.
  --preds-f32     preds.npz "pred" in float32 (default float16).
  --train-on-val  final fit: the val days join the training days (climatology, normalisation and donor pools follow tr). Val scores become
                  in-sample; the fixed val hidden masks are restored before every val pass. Use with --no-select.
  --no-select     no val-based selection: every epoch counts as the best (model.pt = last epoch), no early stop; not with --sched plateau.
  --lr-replay RUN  the lr of epoch e is RUN's history lr_used[e] (replaces --sched; for replaying a plateau schedule without val).
  --memory        temporal3d: 8 static channels from chl_cnn/memory.py (last / next valid observation outside +-4 days up to 60 days with
                  their ages, 8- and 30-day medians excluding day t with counts), "valid" under the --context-clouds persist donor rule.
  --residual-base anom  (with --memory --residual) output base = DOY climatology + w x 30-day median anomaly, w = n / (n + --anom-n0).
  --t3d-stem attn  per-pixel temporal attention stem (chl_cnn/tattn.py) in place of the Conv3d stem; --attn-dim / --attn-heads /
                  --attn-layers; --attn-query-memory conditions the pooling query on the --memory channels.
  --stoch         stochastic output (chl_cnn/stoch.py): --stoch-ch noise channels injected through zero-initialised 1x1 convs; loss = fair
                  CRPS of --stoch-train-samples samples on the pixel and gradient terms; model selection on val fair CRPS (--stoch-val-samples,
                  fixed noise seed); the point prediction (metrics, preds.npz) is the sample median; samples.npz holds --stoch-final-samples
                  samples of day t for every scored day. history.json gains val_crps.
  --t3d-stem patch  gridded patch-token stem (chl_cnn/ptok.py; io3 track, 2026-10-23): one token per ocean P x P patch per window day from
                  mask-aware patch statistics (CHL anomaly mean / std, visible fraction, physics means), self-attention over the days within each
                  patch, per-pixel softmax day weights over the visible context days + a "no day" slot; the backbone gets the weighted blend of
                  the visible anomalies, the weight mass and entropy, day t's own channels and the static channels, with FiLM from the day-t
                  patch token. --ptok-patch / --ptok-dim / --ptok-heads / --ptok-layers / --ptok-arm {interp, pixq} / --ptok-value {own, lavg}
                  (lavg: each context day's value is a Gaussian local masked average, chl_cnn/ptok.py). Dilated backbone only.
  --t3d-stem tokdec  token-decoder stem (chl_cnn/tokdec.py; io4 track, 2026-10-23): per-day 12-channel input (CHL anomaly x visible, visible
                  mask, physics x ocean, ocean mask) -> shared ViT patch embedding Conv2d(12, --tokdec-dim, 2P, stride P, padding P/2) on ptok's
                  P x P patch grid (--ptok-patch) + day / position embeddings; --tokdec-layers self-attention blocks over the days of each ocean
                  patch (--tokdec-heads); pooling with day t's token as query over all T tokens (day t included); [pooled, day-t token] decoded
                  by Linear -> P^2 x --tokdec-ch per patch, pixel shuffle, Conv 3x3 + GELU; the backbone gets the decoded channels, day t's own
                  channels and the static channels (no blend, no FiLM). --tokdec-blend adds ptok's arm-A blend / mass / entropy from the same
                  tokens as 3 more backbone channels. Dilated backbone only; same restrictions as --t3d-stem patch.
  --leak-day / --leak-offset / --leak-what / --leak-delta   leak test with --eval-only (analyses/io_improve/leak_test.py): the input copy of
                  ln CHL of day t+offset is shifted on day t's hidden (or all observed) pixels; the --memory channels of t go to leak_memory.npz.

Input ablation (io5 track, 2026-10-24; chl_cnn/ablate.py; default off, every default unchanged)
  --ablate GROUP --ablate-mode {mean, doyclim}   with --eval-only (temporal3d, --t3d-stem conv / patch / tokdec): one input group is replaced
                  in copies of the model inputs right before the forward pass, on every input path (the per-day stack incl. day t, the static
                  channels, and the per-day climatology the token stems subtract). Physics groups (sst, so, mlotst, sla, currents, winds, air_temp,
                  physics = all 9) on every window day: mean = 0 in z (training mean), doyclim = per-pixel DOY climatology of the training days
                  (31-day window). ctx_chl / day_chl: CHL value and visible mask 0 on the context days / on day t. clim, clim_static, latlon, depth,
                  year: static channel(s) -> mean over ocean px of the training days (clim also replaces the token stems' clim3d). Targets, masks,
                  the --residual base and the residual diagnostics are unchanged. The first batch is checked (config.json ablate_info.check).

Learned kriging and spatial token attention (io6 track, 2026-10-24; analyses/io6_kriging/DESIGN.md; default off, every default unchanged)
  --tokdec-krig    with --t3d-stem tokdec (not with --tokdec-blend, which it replaces): the learned-kriging stem (chl_cnn/krig.py, Triton kernels
                  chl_cnn/krig_triton.py). Each ocean px of day t attends over the visible observations of a fixed 484-offset space-time stencil
                  (|dt| <= 4: dt 0 disc r3 excluding the px, |dt| 1-2 disc r5, |dt| 3-4 5 x 5 block + r6 ring), masked softmax + a "no key" slot,
                  query / key / value features of DESIGN.md section 2 (advected-offset key feature with learned s, init --krig-s0), a learned
                  relative-position bias per (offset, head). The backbone gets dv + 4 extra channels: attention output (--krig-dv), kriging estimate
                  sum w a and spread from the exact anomalies, weight mass on real keys, weight entropy. --krig-dim / --krig-heads / --krig-dv.
                  --krig-signed: + a signed head pair, w = softmax(s+) - lambda softmax(s-), lambda = sigmoid (init 0.1): 3 more channels (signed
                  estimate, sum softmax(s+), lambda sum softmax(s-)). --krig-advect (opt-in): each lag's stencil centred at the nearest-neighbour
                  position p + s0 u(p,t) dt (s0 = --krig-s0 fixed; PyTorch path). --krig-impl triton (default; fused kernels, tl.dot precision
                  --krig-prec, default tf32) or torch (streamed chunks of --krig-chunk offsets, checkpointed, torch.compile).
  --tokdec-spatial N   with --t3d-stem tokdec: N pre-norm spatial self-attention blocks over the pooled ocean-patch tokens of each sample (global,
                  fused SDPA, --tokdec-heads heads, width --tokdec-dim, learned 2D relative-position bias per (patch offset, head); output
                  projections zero-initialised) between the temporal pooling and the decoder (chl_cnn/tokdec.py).
  --tokdec-spatial-pool S   blk9 (default 1 = off): the spatial blocks attend over coarse tokens = mean of the ocean-patch tokens of S x S
                  patches (P 4, S 3 -> 12 px); attention output copied back to the patches (nearest) and added; MLP and decoder stay at P.
  --physmem [--physmem-blend-k 8] [--physmem-stride2]   io7 (chl_cnn/physmem.py): with --t3d-stem tokdec --tokdec-blend; tokens over the whole
                  window (e.g. --context 15 --feat-context 15), blend on +-K only; past-only physics summaries of day t (wind-stress curl, Ekman
                  pumping, alongshore stress, SSTA, dMLD over [t-30, t-1] and [t-60, t-31]) added to every day token. --physmem-stride2: even
                  offsets only beyond +-K.
  --rw-loss [--rw-cap 10] [--rw-draws 3]   io8 (chl_cnn/rwloss.py): day-t hidden-px loss weighted per (distance to visible x key-count) cell so the
                  training fake-cloud px match the training days' real-cloud px (postrun.py bins); training only, scoring unchanged.
  --bsmem         blk8 (chl_cnn/bsmem.py; --store blk4, tokdec + blend): physmem's token net with a Black Sea memory (Danube / Dniester /
                  Dnieper discharge x mouth proximity, MLD deepening, shortwave anomaly; 30- / 60-day past means; coverage).

Prior-knowledge terms (v1), each off by default
  --src-clim      S += d/dt of the day-of-year ln CHL climatology (seasonal source, no parameters)
  --src-ekman     S += alpha * sum_k kappa_k * wE+_k(t) / h0 * taper(lat): upward Ekman pumping from the
                  winds averaged over lag windows 0-3, 4-10, 11-20, 21-30 days (priors.py); alpha >= 0
                  (softplus) and kappa a softmax over the 4 windows are learned; taper = 0 for |lat| < 3,
                  1 beyond 5 deg (Coriolis -> 0 at the equator); h0 = 30 m
  --phys-weight courant   residual weight 1 / max(1, Co_clim(x, y)) (boundary currents downweighted)

Tracked every epoch (history.json): train data loss, train physics loss, val hidden-pixel MAE, val mean|r|
under real clouds / observed pixels (full residual, and the transport-only residual 'base' for comparison
across source definitions), alpha and kappa. Every --fig-every epochs a diagnostic figure for a fixed
val day -> models/pinn/<cnn|unet>/<run>/figures/.

Black Sea store (2026-10-09; every IO default unchanged)
  --store blk     store_blk.py (2 km, 2014-2024); feature "sst" maps to "sst32"; u_wind/v_wind are wind stress (Pa, z-scored
                  like any feature); topo = minus distance to coast (the "depth" channel becomes log distance to coast);
                  no air_temp. Latitude ascends in this store: grid_metrics gives dy > 0 (currents in index space stay
                  consistent) and the diagnostic figure uses origin="lower". Default split val 2017 2021 / test 2019 2023
                  (--val-years / --test-years). --crop cuts the grid to the ocean bounding box + 1 px (as train_spatial.py).
  --store blk4    store_blk4.py: the same data 2x2 block-averaged to ~4.5 km (197 x 275; CHL = exp(mean ln CHL) where >= 2 of 4
                  fine pixels are observed). Behaves exactly like blk (sst -> sst32, split, latitude handling) except
                  --lag-max defaults to 10 cells (the same physical distance as blk's 20). Scores are on the 4 km target.
  --geo-static    (2026-10-17) 10 extra static channels from the store: log10(max(depth.npy, 5 m)), log10(1 + distance km)
                  to each of the 8 river mouths of river_dist.npy (river_names.json order), and a physics-availability mask
                  (1 where salinity and currents are finite on > 50 % of days; 0 in most of Azov / Marmara, where those
                  features are 0-filled). Each z-scored over ocean pixels, 0 on land. Needs --store blk / blk4.
  --feat-device cpu   features and currents are not kept on the GPU (8 x 1.7 GB on blk): each batch is read from the
                  memory-mapped store, pinned and copied (z-scoring on the GPU). gpu = the IO behaviour.
  --phys semi-lagrangian   residual r = [C(t+1)(x) - C(t-1)(x - 2 u dt)] / 2dt - K lap C(t) - S: the day-(t-1) field is
                  sampled at the departure point x - 2 u dt (bilinear grid_sample, zero padding; displacement in cells
                  = u 86400 / dx from the grid metrics, clipped at +---lag-max cells; the ocean mask is sampled the same
                  way and the pixel is valid only where it is >= 0.99). The default --phys upwind is the old residual,
                  which is not usable at the Black Sea Courant numbers (~6 cells/day). Both forms are scored every epoch.
  --max-batches N, --val-stride S, --eval-batch B   debug / speed: N training batches per epoch, every S-th val (and
                  test) day scored, eval chunk size.

Example
  ALL="sst so air_temp u_wind v_wind sla mlotst u_curr v_curr"
  python -m chl_cnn.train_pinn --run pinn_lam0p3 --lam 0.3 --features $ALL --clim --coords --residual --context 2 --vis-weight 0.1 --epochs 12
  python -m chl_cnn.train_pinn --store blk --crop --feat-device cpu --val-stride 4 --eval-batch 8 --phys semi-lagrangian --lam 0.3 \
      --run cnn_blk_sl_lam0p3 --features sst so mlotst sla u_curr v_curr u_wind v_wind --clim --coords --residual --context 2 \
      --vis-weight 0.1 --year-input --donor-window 15 --epochs 12
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data as D
from . import clouds as C
from . import store as S
from . import train2 as T2
from . import priors as PR
from . import spatial_node as SN
from . import deq as DQ
from . import memory as MEM
from .evaluate import metrics, per_day_metrics, climatology_reference
from .models import build_model, count_params, Temporal3DNet
from .physics import grid_metrics
from .plotting import OKABE_ITO, MAP_CMAP, ERR_CMAP

from .runs import run_dir, CATEGORIES
DT = 86400.0
H0 = 30.0   # reference mixed-layer depth (m) for the Ekman source


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--store", choices=["io", "blk", "blk4", "io_up"], default="io", help="io: Indian Ocean 0.25 deg store (store.py); blk: Black Sea 2 km store (store_blk.py); blk4: Black Sea 4 km store (store_blk4.py); io_up: 2024 hackweek box / days / split (store_io_up.py)")
    p.add_argument("--fake-mask", choices=["donor", "shift10"], default="donor", help="donor: fake clouds from a donor day (clouds.py); shift10: the 2024 hackweek mask, fixed per day for train/val/test (needs --store io_up, --context-clouds none)")
    p.add_argument("--feat-today-only", action="store_true", help="temporal3d: physical features only on day t (zero on the other window days)")
    p.add_argument("--doy-input", action="store_true", help="add static sin/cos(day of year) channels (without the lat/lon/depth of --coords)")
    p.add_argument("--land-input", action="store_true", help="add a static land channel (1 where not ocean)")
    p.add_argument("--recipe", choices=["none", "up2024"], default="none", help="up2024: train the 2024 hackweek way (their 13 inputs, all-pixel MSE in z, Adam constant lr, batch 4, early stopping on val loss; see the docstring)")
    p.add_argument("--crop", action="store_true", help="crop the grid to the bounding box of the ocean mask (+1 pixel margin)")
    p.add_argument("--val-years", type=int, nargs="*", default=None, help="val years (default: data.VAL_YEARS for io; 2017 2021 for blk)")
    p.add_argument("--test-years", type=int, nargs="*", default=None, help="test years (default: data.TEST_YEARS for io; 2019 2023 for blk)")
    p.add_argument("--feat-device", choices=["gpu", "cpu"], default="gpu", help="gpu: all features/currents on the GPU in float16 (IO); cpu: read each batch from the memory-mapped store")
    p.add_argument("--max-batches", type=int, default=0, help="debug: train only the first N batches per epoch (0 = all)")
    p.add_argument("--val-stride", type=int, default=1, help="score every S-th val (and test) day")
    p.add_argument("--eval-batch", type=int, default=32, help="days per eval chunk")
    p.add_argument("--model", choices=["dilated", "unet", "deq", "temporal3d"], default="dilated")
    p.add_argument("--t3d-backbone", choices=["dilated", "unet"], default="dilated", help="temporal3d: 2D backbone after the 3D stem")
    p.add_argument("--t3d-width", type=int, default=32, help="temporal3d: stem channels")
    p.add_argument("--t3d-depth", type=int, default=2, help="temporal3d: number of Conv3d layers in the stem")
    p.add_argument("--t3d-kernel-t", type=int, default=3, help="temporal3d: temporal kernel size of the stem convolutions")
    p.add_argument("--deq-solver", choices=["unroll", "fixed", "anderson"], default="anderson", help="deq: forward solver (see deq.py)")
    p.add_argument("--deq-steps", type=int, default=24, help="deq: unroll steps / max solver iterations")
    p.add_argument("--deq-tol", type=float, default=1e-3, help="deq: solver stopping tolerance on the relative residual")
    p.add_argument("--deq-phantom-steps", type=int, default=1, help="deq (fixed/anderson): phantom-gradient steps after the solve (1 with tau 1 = Jacobian-free backprop)")
    p.add_argument("--deq-phantom-tau", type=float, default=1.0, help="deq: phantom-gradient damping")
    p.add_argument("--deq-anderson-m", type=int, default=5)
    p.add_argument("--deq-anderson-beta", type=float, default=1.0)
    p.add_argument("--deq-eta", type=float, default=1.0, help="deq: initial step size of the cell update (learned)")
    p.add_argument("--deq-fixed-eta", action="store_true", help="deq: keep eta fixed")
    p.add_argument("--cell-dilations", default="1,2,4,8", help="deq: dilations of the cell's residual blocks")
    p.add_argument("--deq-spectral", action="store_true", help="deq: spectral-normalise the cell convolutions")
    p.add_argument("--deq-ckpt", action="store_true", help="deq unroll: checkpoint each step (memory ~ 1 step instead of K)")
    p.add_argument("--deq-no-project", action="store_true", help="deq: do not overwrite visible pixels after each cell application")
    p.add_argument("--deq-eval-steps", type=int, default=0, help="deq: solver iterations at eval (0 = --deq-steps)")
    p.add_argument("--deq-curve-days", type=int, default=64, help="deq: val days used for the MAE-vs-iterations curve at the end")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--unet-depth", type=int, default=4, help="U-Net pooling levels (2 = 64/128/256 like the mind-the-chl-gap U-Net with --width 64)")
    p.add_argument("--spectral", type=int, default=0, help="dilated: number of leading blocks that get an FNO spectral path")
    p.add_argument("--spectral-modes", type=int, default=16, help="Fourier modes kept per axis in the spectral path")
    p.add_argument("--spectral-ch", type=int, default=16, help="channels of the spectral bottleneck")
    p.add_argument("--dilations", default=None, help="dilated backbone (--model dilated, temporal3d dilated): comma-separated residual-block dilations (default None = DilatedCNN default 1,2,4,8,16,1,2,4)")
    p.add_argument("--grad-weight", type=float, default=0.0, help="weight of the gradient-difference loss on day t (L1 on d/dlat, d/dlon of z-scored ln CHL where both pixels have truth and at least one is hidden)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--features", nargs="*", default=["sst", "so"])
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--context-before", type=int, default=None, help="temporal3d: days before t in the window (default --context)")
    p.add_argument("--context-after", type=int, default=None, help="temporal3d: days after t in the window (default --context)")
    p.add_argument("--context-clouds", choices=["persist", "none"], default="persist")
    p.add_argument("--clim", action="store_true")
    p.add_argument("--coords", action="store_true")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--loss", choices=["l1", "huber", "mse"], default="l1")
    p.add_argument("--huber-delta", type=float, default=0.5)
    p.add_argument("--vis-weight", type=float, default=0.0)
    p.add_argument("--nb-weight", type=float, default=1.0, help="weight of the hidden-pixel loss on days t+-1")
    p.add_argument("--no-chl-input", action="store_true", help="physics-only inputs: drop the CHL value channels")
    p.add_argument("--cloud-input", action="store_true", help="keep the observation-mask channels as inputs (with --no-chl-input)")
    p.add_argument("--no-chl-today", action="store_true", help="drop day t's CHL channels only; CHL context days stay")
    p.add_argument("--geo-static", action="store_true", help="blk/blk4: add bathymetry, 8 river-mouth distances and a salinity/currents availability mask as static channels")
    p.add_argument("--year-input", action="store_true", help="add a constant channel (year - 2009) / 7 (calendar-year drift)")
    p.add_argument("--donor-window", type=int, default=0, help="training fake-cloud donors drawn from days within +-W days of year (0 = any day); val/test donors unchanged")
    p.add_argument("--donor-exclude", type=int, default=0, help="no fake-cloud donor within D calendar days of its target day (training donors, uniform and "
                   "--donor-window draws, and the fixed val/test donors, i.e. the hidden-pixel benchmark changes when D > 0); 0 = unchanged")
    p.add_argument("--continue-from", default=None, help="run name whose resume.pt (best-epoch training state) starts this run; fresh lr schedule from --lr over --epochs")
    p.add_argument("--eval-only", action="store_true", help="no training: score the weights of --init-from on the val/test hidden pixels of this run's settings")
    p.add_argument("--init-from", default=None, help="run name whose model.pt is loaded (with --eval-only); its architecture arguments must be repeated")
    p.add_argument("--feat-context", type=int, default=0, help="physical features of days t+-1..K as extra channels")
    p.add_argument("--diff-fill", default=None, help="spatial run name (models/spatial/<run>/model.pt, --dmodel sst) whose steady-state anisotropic-diffusion fill of today's visible pixels (+ neighbour-day soft terms) is added as an input channel, z-scored")
    p.add_argument("--diff-fill-ctx", type=int, default=2, help="neighbour days used by the fill's soft terms (<= --context)")
    # physics
    p.add_argument("--lam", type=float, default=0.0, help="physics loss weight (0 = data-only control)")
    p.add_argument("--phys-warmup", type=int, default=3, help="epochs over which lambda ramps from 0")
    p.add_argument("--phys-loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--K", type=float, default=500.0, help="eddy diffusivity m2/s")
    p.add_argument("--co-max", type=float, default=0.0, help="drop residual pixels with Courant > co-max (0 = off)")
    p.add_argument("--phys", choices=["upwind", "semi-lagrangian"], default="upwind", help="residual form: upwind finite differences (IO) or semi-Lagrangian back-trajectory over 2 days")
    p.add_argument("--lag-max", type=float, default=None, help="semi-lagrangian: clip the 2-day displacement to +-lag-max cells (default 20 for blk, 10 for blk4, 5 for io)")
    p.add_argument("--no-paste", action="store_true", help="do not paste visible pixels back before the residual")
    # priors
    p.add_argument("--src-clim", action="store_true", help="seasonal source: d/dt of the day-of-year climatology")
    p.add_argument("--src-ekman", action="store_true", help="lagged Ekman upwelling source, equator-tapered")
    p.add_argument("--phys-weight", choices=["none", "courant"], default="none")
    # training
    p.add_argument("--epochs", type=int, default=None, help="max training epochs (default 12; 50 with --recipe up2024)")
    p.add_argument("--stop-epoch", type=int, default=0, help="stop training after this many epochs while the lr schedule still spans --epochs (0 = off). For quick architecture comparisons against an existing --epochs run at the same epoch, without retraining that run.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--sched", choices=["none", "cosine", "plateau"], default=None,
                   help="lr schedule (default cosine; none with --recipe up2024). plateau: ReduceLROnPlateau on the model-selection score "
                        "(val hidden MAE, or val all-pixel z MSE with --recipe up2024)")
    p.add_argument("--plateau-factor", type=float, default=0.5, help="--sched plateau: lr multiplier at each drop")
    p.add_argument("--plateau-patience", type=int, default=5, help="--sched plateau: epochs without improvement before an lr drop")
    p.add_argument("--min-lr", type=float, default=1e-6, help="--sched plateau: lower bound of the lr")
    p.add_argument("--patience", type=int, default=None, help="early-stopping patience in epochs (default 100; 10 with --recipe up2024)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-channels-last", action="store_true", help="keep NCHW tensors (channels_last is ~35%% faster for the dilated CNN on A10G)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--fig-every", type=int, default=5)
    p.add_argument("--ckpt-from", type=int, default=0, help="track/checkpoint the best val model only after this epoch (model.pt written on every improvement)")
    p.add_argument("--return", dest="resume_best", action=argparse.BooleanOptionalAction, default=True,
                   help="return to the best checkpointed training state (resume.pt) if the run was interrupted; --no-return starts fresh")
    # io2 track (2026-10-21; every default unchanged, see the docstring section "IO improvement track")
    p.add_argument("--continue-weights-only", action="store_true", help="--continue-from: load the source's model.pt weights only (fresh AdamW / GradScaler); "
                   "for sources without resume.pt")
    p.add_argument("--preds-f32", action="store_true", help="store float32 predictions in preds.npz (default float16)")
    p.add_argument("--train-on-val", action="store_true", help="train on train + val days (final fit; val scores become in-sample)")
    p.add_argument("--no-select", action="store_true", help="no val-based selection: every epoch counts as best, model.pt = last epoch, no early stop")
    p.add_argument("--lr-replay", default=None, help="run name whose history.json lr_used sets the lr of each epoch (replaces --sched)")
    p.add_argument("--memory", action="store_true", help="temporal3d: 8 memory channels (last/next valid obs outside +-4 d with age, 8-/30-day medians "
                   "with counts; donor-masked like --context-clouds persist; chl_cnn/memory.py)")
    p.add_argument("--residual-base", choices=["clim", "anom"], default="clim", help="--residual base: clim (default) or clim + shrunk 30-day "
                   "anomaly median (needs --memory)")
    p.add_argument("--anom-n0", type=float, default=3.0, help="--residual-base anom: shrinkage w = n / (n + n0)")
    p.add_argument("--t3d-stem", choices=["conv", "attn", "patch", "tokdec"], default="conv", help="temporal3d stem: conv (Conv3d, default) or attn (per-pixel temporal "
                   "attention, chl_cnn/tattn.py)")
    p.add_argument("--attn-dim", type=int, default=32, help="--t3d-stem attn: token width (= stem output channels)")
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-layers", type=int, default=1, help="--t3d-stem attn: self-attention layers before the pooling")
    p.add_argument("--attn-query-memory", action="store_true", help="--t3d-stem attn: condition the pooling query on the --memory channels")
    p.add_argument("--ptok-patch", type=int, default=4, help="--t3d-stem patch: patch size P (px)")
    p.add_argument("--ptok-dim", type=int, default=64, help="--t3d-stem patch: token width")
    p.add_argument("--ptok-heads", type=int, default=4)
    p.add_argument("--ptok-layers", type=int, default=2, help="--t3d-stem patch: self-attention layers over the days of each patch")
    p.add_argument("--ptok-arm", choices=["interp", "pixq"], default="interp", help="--t3d-stem patch: per-pixel day-score arm")
    p.add_argument("--ptok-value", choices=["own", "lavg"], default="own", help="--t3d-stem patch: context-day value at a pixel: own visible "
                   "anomaly (default) or the Gaussian (sigma 1.5 px, r 5 px) normalised-convolution local average of the visible anomalies")
    p.add_argument("--tokdec-dim", type=int, default=128, help="--t3d-stem tokdec: token width D")
    p.add_argument("--tokdec-ch", type=int, default=32, help="--t3d-stem tokdec: decoded context channels C")
    p.add_argument("--tokdec-heads", type=int, default=4, help="--t3d-stem tokdec: heads (self-attention and pooling)")
    p.add_argument("--tokdec-layers", type=int, default=2, help="--t3d-stem tokdec: self-attention layers over the days of each patch")
    p.add_argument("--tokdec-blend", action="store_true", help="--t3d-stem tokdec: add ptok's arm-A blend / mass / entropy channels (own-pixel skip)")
    p.add_argument("--stoch", action="store_true", help="stochastic output: zero-initialised noise injection (chl_cnn/stoch.py), fair-CRPS loss")
    p.add_argument("--stoch-ch", type=int, default=8, help="--stoch: noise channels")
    p.add_argument("--stoch-train-samples", type=int, default=2, help="--stoch: samples per example in the fair-CRPS loss (>= 2)")
    p.add_argument("--stoch-val-samples", type=int, default=4, help="--stoch: samples for the per-epoch val score (fair CRPS)")
    p.add_argument("--stoch-final-samples", type=int, default=8, help="--stoch: samples for the final scoring (saved in samples.npz)")
    p.add_argument("--leak-day", type=int, default=None, help="leak test (with --eval-only): store index t of a val day; ln CHL of day t+--leak-offset is "
                   "shifted by --leak-delta on the pixels of --leak-what before anything is computed; the --memory channels of target t are saved")
    p.add_argument("--leak-offset", type=int, default=0)
    p.add_argument("--leak-delta", type=float, default=1.0)
    p.add_argument("--leak-what", choices=["hidden", "observed", "masked"], default="hidden", help="hidden: day t's fixed hidden px (offset 0); observed: every observed px; "
                   "masked: the observed px of day t+offset that the --context-clouds persist donor rule hides from target t (io3, 2026-10-23)")
    # io5 input ablation (2026-10-24; chl_cnn/ablate.py; default off = unchanged)
    p.add_argument("--ablate", default=None, choices=["sst", "so", "mlotst", "sla", "currents", "winds", "air_temp", "physics", "ctx_chl", "day_chl",
                                                      "clim", "clim_static", "latlon", "depth", "year"],
                   help="with --eval-only: replace one input group in the model inputs (chl_cnn/ablate.py)")
    p.add_argument("--ablate-mode", choices=["mean", "doyclim"], default="mean", help="--ablate physics groups: training mean (0 in z) or the "
                   "per-pixel DOY climatology of the training days (31-day window)")
    # io6 learned kriging / spatial token attention (2026-10-24; chl_cnn/krig.py, chl_cnn/tokdec.py; default off = unchanged)
    p.add_argument("--tokdec-krig", action="store_true", help="--t3d-stem tokdec: learned-kriging stem channels in place of the blend (chl_cnn/krig.py)")
    p.add_argument("--krig-dim", type=int, default=32, help="--tokdec-krig: query / key width D")
    p.add_argument("--krig-heads", type=int, default=4, help="--tokdec-krig: heads")
    p.add_argument("--krig-dv", type=int, default=16, help="--tokdec-krig: value embedding / attention output channels")
    p.add_argument("--krig-chunk", type=int, default=32, help="--tokdec-krig --krig-impl torch: stencil offsets per streamed chunk")
    p.add_argument("--krig-signed", action="store_true", help="--tokdec-krig: + a signed head pair (difference of two softmaxes)")
    p.add_argument("--krig-advect", action="store_true", help="--tokdec-krig: back-advected stencil centring (nearest neighbour, s fixed --krig-s0)")
    p.add_argument("--krig-s0", type=float, default=0.25, help="--tokdec-krig: initial advected-offset scale s (and the fixed --krig-advect scale)")
    p.add_argument("--krig-impl", choices=["triton", "torch"], default="triton", help="--tokdec-krig: fused Triton kernels or the PyTorch path")
    p.add_argument("--krig-prec", choices=["tf32", "ieee", "tf32x3"], default="tf32", help="--krig-impl triton: tl.dot input precision")
    p.add_argument("--tokdec-spatial", type=int, default=0, help="--t3d-stem tokdec: spatial self-attention blocks over the pooled patch tokens")
    p.add_argument("--tokdec-spatial-pool", type=int, default=1, help="blk9 --tokdec-spatial: attend over S x S pooled patch tokens (1 = off)")
    from . import analog as AN                                 # io7 analog tokens (--analog*, default off = unchanged)
    AN.add_args(p)
    from . import wmattn as WMA                                # io7 water-mass attention (--wm-*, default off = unchanged)
    WMA.add_args(p)
    from . import physmem as PMEM                              # io7 physics memory (--physmem*, default off = unchanged)
    PMEM.add_args(p)
    from . import rwloss as RWL                                # io8 real-cloud loss reweighting (--rw-loss*, default off = unchanged)
    RWL.add_args(p)
    from . import bsmem as BSM                                 # blk8 Black Sea physics memory (--bsmem, default off = unchanged)
    BSM.add_args(p)
    a = p.parse_args(argv)
    AN.check_args(p, a)
    if a.width is None:
        a.width = 32 if a.model == "unet" or (a.model == "temporal3d" and a.t3d_backbone == "unet") else 64
    a.context_before = a.context if a.context_before is None else a.context_before
    a.context_after = a.context if a.context_after is None else a.context_after
    if a.model == "temporal3d":
        if a.context_before < 0 or a.context_after < 0:
            p.error("--context-before / --context-after must be >= 0")
        if a.feat_context != max(a.context_before, a.context_after):
            p.error("--model temporal3d needs --feat-context equal to max(--context-before, --context-after) "
                    "(features on every day of the window t-B..t+A; with the defaults B = A = --context)")
        if a.no_chl_input or a.cloud_input or a.diff_fill:
            p.error("--model temporal3d does not support --no-chl-input / --cloud-input / --diff-fill")
    elif (a.context_before, a.context_after) != (a.context, a.context):
        p.error("--context-before / --context-after are only supported with --model temporal3d")
    if a.residual and not a.clim:
        p.error("--residual needs --clim")
    if a.src_clim and not a.clim:
        p.error("--src-clim needs --clim")
    if a.context < 1:
        p.error("--context must be >= 1 (the residual needs days t+-1 as input)")
    if a.eval_only and not a.init_from:
        p.error("--eval-only needs --init-from")
    if a.init_from and not a.eval_only:
        p.error("--init-from is only supported with --eval-only")
    if a.fake_mask == "shift10" and (a.store != "io_up" or a.context_clouds != "none"):
        p.error("--fake-mask shift10 needs --store io_up and --context-clouds none")
    if a.feat_today_only and a.model != "temporal3d":
        p.error("--feat-today-only is only supported with --model temporal3d")
    if a.doy_input and a.coords:
        p.error("--doy-input duplicates the sin/cos channels of --coords")
    if a.lag_max is None:
        a.lag_max = {"blk": 20.0, "blk4": 10.0}.get(a.store, 5.0)
    if a.continue_weights_only and not a.continue_from:
        p.error("--continue-weights-only needs --continue-from")
    if (a.memory or a.t3d_stem != "conv" or a.stoch) and a.model != "temporal3d":
        p.error("--memory / --t3d-stem attn / --stoch need --model temporal3d")
    if a.residual_base != "clim" and not (a.memory and a.residual):
        p.error("--residual-base anom needs --memory and --residual")
    if a.t3d_stem in ("patch", "tokdec") and (a.memory or a.stoch or a.t3d_backbone != "dilated" or a.no_chl_input or a.no_chl_today or a.feat_today_only
                                  or a.recipe != "none" or not a.clim):
        p.error("--t3d-stem patch / tokdec: needs --clim and the dilated backbone; no --memory / --stoch / --no-chl-* / --feat-today-only / --recipe")
    if a.tokdec_blend and a.t3d_stem != "tokdec":
        p.error("--tokdec-blend needs --t3d-stem tokdec")
    if (a.tokdec_krig or a.tokdec_spatial) and a.t3d_stem != "tokdec":
        p.error("--tokdec-krig / --tokdec-spatial need --t3d-stem tokdec")
    if a.tokdec_krig and a.tokdec_blend:
        p.error("--tokdec-krig replaces --tokdec-blend; use one of them")
    if a.tokdec_spatial_pool != 1 and (a.tokdec_spatial_pool < 1 or not a.tokdec_spatial or a.tokdec_krig or a.wm_attn or a.physmem or a.bsmem or a.analog):
        p.error("--tokdec-spatial-pool S (S >= 2) needs --tokdec-spatial N and the plain token decoder (no --tokdec-krig / --wm-attn / --physmem / --bsmem / --analog)")
    if (a.krig_signed or a.krig_advect) and not a.tokdec_krig:
        p.error("--krig-signed / --krig-advect need --tokdec-krig")
    if a.tokdec_krig and (a.context_before < 4 or a.context_after < 4 or a.krig_dim % a.krig_heads):
        p.error("--tokdec-krig needs a window of at least t-4..t+4 and --krig-dim divisible by --krig-heads")
    WMA.check_args(p, a)                                       # io7 --wm-attn (no-op without it)
    PMEM.check_args(p, a)                                      # io7 --physmem (no-op without it)
    RWL.check_args(p, a)                                       # io8 --rw-loss (no-op without it)
    BSM.check_args(p, a)                                       # blk8 --bsmem (no-op without it)
    if a.attn_query_memory and not (a.memory and a.t3d_stem == "attn"):
        p.error("--attn-query-memory needs --memory and --t3d-stem attn")
    if a.stoch and (a.stoch_train_samples < 2 or a.recipe != "none"):
        p.error("--stoch needs --stoch-train-samples >= 2 and no --recipe")
    if a.no_select and a.sched == "plateau":
        p.error("--no-select cannot drive --sched plateau (use --lr-replay)")
    if a.ablate:
        from .ablate import FEAT_GROUPS
        if not a.eval_only or a.model != "temporal3d" or a.t3d_stem not in ("conv", "patch", "tokdec") or a.stoch or a.memory or a.recipe != "none":
            p.error("--ablate needs --eval-only and --model temporal3d with --t3d-stem conv / patch / tokdec (no --stoch / --memory / --recipe)")
        if a.ablate_mode == "doyclim" and a.ablate not in FEAT_GROUPS:
            p.error("--ablate-mode doyclim applies to the physics groups only")
        if a.ablate in FEAT_GROUPS and not set(FEAT_GROUPS[a.ablate]) <= set(a.features):
            p.error(f"--ablate {a.ablate}: features {FEAT_GROUPS[a.ablate]} not all in --features")
        need = {"clim": a.clim, "clim_static": a.clim, "latlon": a.coords, "depth": a.coords, "year": a.year_input}
        if not need.get(a.ablate, True):
            p.error(f"--ablate {a.ablate}: the model has no such channel")
    if a.recipe == "up2024":
        if a.store != "io_up" or a.fake_mask != "shift10" or a.context_clouds != "none":
            p.error("--recipe up2024 needs --store io_up --fake-mask shift10 --context-clouds none")
        if a.model not in ("dilated", "temporal3d") or (a.model == "temporal3d" and (a.context, a.feat_context, a.context_before, a.context_after) != (1, 1, 1, 1)):
            p.error("--recipe up2024 supports --model dilated, or --model temporal3d with --context 1 --feat-context 1")
        bad = [f for f in ("clim", "coords", "residual", "year_input", "doy_input", "land_input", "geo_static", "feat_today_only", "no_chl_input",
                           "cloud_input", "no_chl_today", "src_clim", "src_ekman", "crop") if getattr(a, f)] + (["diff_fill"] if a.diff_fill else []) \
            + (["grad_weight"] if a.grad_weight > 0 else [])
        if bad:
            p.error(f"--recipe up2024 takes only their 13 inputs and their loss; remove {bad}")
        a.features = ["u_wind", "v_wind", "sst", "air_temp"]
        a.batch, a.ckpt_from, a.wd, a.val_stride, a.loss = 4, 0, 0.0, 1, "mse"
    rec = a.recipe == "up2024"                     # defaults that differ under the recipe (explicit values are kept)
    a.epochs = a.epochs if a.epochs is not None else (50 if rec else 12)
    a.patience = a.patience if a.patience is not None else (10 if rec else 100)
    a.sched = a.sched if a.sched is not None else ("none" if rec else "cosine")
    return a


def buffer_shuffle(x, buf, rng):
    """tf.data Dataset.shuffle(buffer_size=buf) order of x: fill a buffer with the first buf items (in order), then repeatedly emit
    a uniformly drawn buffer slot and refill it with the next item; once x is exhausted the emitted slot is removed."""
    x = list(x); b = x[:buf]; i = len(b); out = []
    while b:
        k = int(rng.integers(len(b)))
        out.append(b[k])
        if i < len(x):
            b[k] = x[i]; i += 1
        else:
            b[k] = b[-1]; b.pop()
    return np.asarray(out)


class EkmanSource(nn.Module):
    """S_up = softplus(a) * softmax(k) . wE+_lags / H0  (ln CHL per day per (m/day)/m)."""

    def __init__(self, n_lags):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(-3.0))          # alpha = softplus(-3) ~ 0.05 at init
        self.k = nn.Parameter(torch.zeros(n_lags))

    def alpha(self):
        return F.softplus(self.a)

    def kappa(self):
        return torch.softmax(self.k, 0)

    def forward(self, lags):                                 # lags (B, n_lags, H, W) in m/day
        return self.alpha() * torch.einsum("k,bkhw->bhw", self.kappa(), lags) / H0


# arguments that may differ between an interrupted run and its --return continuation (they do not change training or
# model selection): the flag itself, the diagnostic-figure cadence and the eval chunk size
RESUME_IGNORED = {"resume_best", "fig_every", "eval_batch"}
# arguments added after --return: a resume.pt written before them is compared as if they had their default value
RESUME_NEW_DEFAULTS = {"donor_exclude": 0, "eval_only": False, "init_from": None, "dilations": None, "continue_from": None, "geo_static": False,
                       "fake_mask": "donor", "feat_today_only": False, "doy_input": False, "land_input": False, "recipe": "none",
                       "plateau_factor": 0.5, "plateau_patience": 5, "min_lr": 1e-6,
                       "continue_weights_only": False, "preds_f32": False, "train_on_val": False, "no_select": False, "lr_replay": None,
                       "memory": False, "residual_base": "clim", "anom_n0": 3.0, "t3d_stem": "conv", "attn_dim": 32, "attn_heads": 4,
                       "attn_layers": 1, "attn_query_memory": False, "ptok_patch": 4, "ptok_dim": 64, "ptok_heads": 4, "ptok_layers": 2,
                       "ptok_arm": "interp", "ptok_value": "own",
                       "tokdec_dim": 128, "tokdec_ch": 32, "tokdec_heads": 4, "tokdec_layers": 2, "tokdec_blend": False, "stoch": False, "stoch_ch": 8, "stoch_train_samples": 2,
                       "stoch_val_samples": 4, "stoch_final_samples": 8,
                       "leak_day": None, "leak_offset": 0, "leak_delta": 1.0, "leak_what": "hidden", "ablate": None, "ablate_mode": "mean",
                       "tokdec_krig": False, "krig_dim": 32, "krig_heads": 4, "krig_dv": 16, "krig_chunk": 32, "krig_signed": False,
                       "krig_advect": False, "krig_s0": 0.25, "krig_impl": "triton", "krig_prec": "tf32", "tokdec_spatial": 0,
                       "tokdec_spatial_pool": 1, "physmem": False, "physmem_blend_k": 8, "physmem_stride2": False}
from .analog import RESUME_DEFAULTS as _ANALOG_DEFAULTS       # io7 analog tokens
RESUME_NEW_DEFAULTS.update(_ANALOG_DEFAULTS)
from .wmattn import RESUME_DEFAULTS as _WM_DEFAULTS           # io7 water-mass attention
RESUME_NEW_DEFAULTS.update(_WM_DEFAULTS)
from .rwloss import RESUME_DEFAULTS as _RW_DEFAULTS          # io8 real-cloud loss reweighting
RESUME_NEW_DEFAULTS.update(_RW_DEFAULTS)
from .bsmem import RESUME_DEFAULTS as _BSM_DEFAULTS          # blk8 Black Sea physics memory
RESUME_NEW_DEFAULTS.update(_BSM_DEFAULTS)


def main(argv=None):
    a = parse(argv)
    STOCH = None
    if a.stoch:
        from . import stoch as STOCH
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = run_dir(a.run, "pinn", a.model) if a.store != "io_up" else CATEGORIES["pinn/upstream"] / a.run
    if a.resume_best and (out / "metrics.json").exists():
        print(f"run {a.run} is complete ({out / 'metrics.json'} exists); nothing to do", flush=True)
        return
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if a.store in ("blk", "blk4"):
        if a.store == "blk":
            from . import store_blk as SB
        else:
            from . import store_blk4 as SB
        d = SB.load_store()
        val_years = tuple(a.val_years) if a.val_years else (2017, 2021); test_years = tuple(a.test_years) if a.test_years else (2019, 2023)
    elif a.store == "io_up":
        from . import store_io_up as SU
        d = SU.load_store()
        val_years, test_years = (2018,), (2019,)          # labels only: the split is SU.split_up (their inclusive date slices)
    else:
        d = S.load_store()
        val_years = tuple(a.val_years) if a.val_years else D.VAL_YEARS; test_years = tuple(a.test_years) if a.test_years else D.TEST_YEARS
    sname = lambda f: "sst32" if (a.store in ("blk", "blk4") and f == "sst") else f      # store variable name of a feature
    TIMES = d["time"]; T2.TIMES = TIMES
    if a.crop:
        oc = d["ocean"]; rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
        ys, xs = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])), slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
        print(f"crop: rows {ys.start}:{ys.stop} cols {xs.start}:{xs.stop} of {oc.shape}", flush=True)
    else:
        ys, xs = slice(None), slice(None)
    rd = lambda name, t=slice(None): np.asarray(d[name][t, ys, xs])          # cropped read of a (time, lat, lon) array
    n = len(TIMES)
    ocean = d["ocean"][ys, xs]
    H, W = ocean.shape
    tr, va, te = D.split_by_years(TIMES, val_years, test_years) if a.store != "io_up" else SU.split_up(TIMES)
    recipe = a.recipe == "up2024"
    if recipe:
        tr, va, te = SU.split_up2024(TIMES)                   # their inclusive train slice (adds 2018-01-01)
    if a.train_on_val:                                         # final fit: val days join the training days (val scores are in-sample)
        tr = np.sort(np.r_[tr, va])
        print(f"train-on-val: {len(tr)} training days (train + val); val scores below are in-sample", flush=True)
    y = D.log_chl(rd("chl_l3"))
    obs = np.isfinite(y) & ocean
    obs_frac = obs.reshape(n, -1).sum(1) / ocean.sum()
    chl_mu, chl_sd = float(np.nanmean(y[tr])), float(np.nanstd(y[tr]))
    if recipe:                                                 # their label z-score over all 9193 records (up_norm.json)
        chl_mu, chl_sd = float(d["up"]["label_mean"]), float(d["up"]["label_std"])
    print(f"store {a.store}: {n} days  train {len(tr)} val {len(va)} test {len(te)}  grid {H} x {W}  ocean px {int(ocean.sum())}  "
          f"val years {val_years} test years {test_years}", flush=True)

    if a.fake_mask == "shift10":     # their fixed mask (store_io_up.py): finite L3 at t & cloud code 0 at record t+10; donor unused
        HUP = rd("hidden_up") if recipe else rd("hidden_up") & obs      # recipe: their mask without the ocean cut
        print(f"shift10 mask: their hidden px outside our ocean mask (dropped): val {int((rd('hidden_up', va) & ~ocean).sum())} "
              f"test {int((rd('hidden_up', te) & ~ocean).sum())} train {int((rd('hidden_up', tr) & ~ocean).sum())}", flush=True)
        hid_va, don_va, hid_te, don_te = HUP[va], va.copy(), HUP[te], te.copy()
    else:
        hid_va, don_va = C.fixed_hidden(obs, va, seed=a.eval_seed, times=TIMES, exclude=a.donor_exclude)
        hid_te, don_te = C.fixed_hidden(obs, te, seed=a.eval_seed + 1, times=TIMES, exclude=a.donor_exclude)
    if a.donor_exclude > 0:
        dd = lambda i, j: np.abs((TIMES[j] - TIMES[i]).astype("timedelta64[D]").astype(int))
        print(f"donor-exclude {a.donor_exclude}: min |donor - day| val {dd(va, don_va).min()} d, test {dd(te, don_te).min()} d", flush=True)
    hidden = np.zeros_like(obs); hidden[va], hidden[te] = hid_va, hid_te
    donor = np.zeros(n, dtype=np.int64); donor[va], donor[te] = don_va, don_te
    if a.fake_mask == "shift10":
        hidden[tr] = HUP[tr]; donor[tr] = tr

    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    feat_stats = {}
    sub = tr[::10]
    for c, f in enumerate(a.features):
        xs_ = rd(sname(f), sub).astype("float32")
        mu, sd = float(np.nanmean(xs_)), float(np.nanstd(xs_)) + 1e-6
        feat_stats[f] = dict(mean=mu, std=sd, nan_frac_train=float(np.isnan(xs_).mean()), store_var=sname(f))
        del xs_
    FMUg = T(np.array([feat_stats[f]["mean"] for f in a.features], dtype="float32")).view(1, -1, 1, 1)
    FSDg = T(np.array([feat_stats[f]["std"] for f in a.features], dtype="float32")).view(1, -1, 1, 1)
    if a.feat_device == "gpu":
        Xg = torch.empty((n, len(a.features), H, W), dtype=torch.float16, device=dev)
        for c, f in enumerate(a.features):
            mu, sd = feat_stats[f]["mean"], feat_stats[f]["std"]
            for b0 in range(0, n, 500):
                x = (rd(sname(f), slice(b0, b0 + 500)).astype("float32") - mu) / sd
                Xg[b0:b0 + 500, c] = torch.as_tensor(np.nan_to_num(x, nan=0.0), device=dev).half()
    Yg, OBSg, HIDg, DONg = torch.nan_to_num(T(y), nan=0.0), T(obs), T(hidden), T(donor)
    if a.leak_day is not None:                                 # leak test: perturb the input copy of ln CHL only (scores use the numpy y)
        assert a.eval_only and a.leak_day in set(va.tolist()), "--leak-day needs --eval-only and a val day"
        jl = int(T2.neighbour_index(TIMES, a.leak_offset)[a.leak_day]) if a.leak_offset else a.leak_day
        assert jl >= 0, "--leak-offset day not in the store"
        if a.leak_what == "masked":                            # obs on t+k but not visible under the donor's day d+k (persist rule)
            dn = int(donor[a.leak_day]); jd = int(T2.neighbour_index(TIMES, a.leak_offset)[dn]) if a.leak_offset else dn
            pm = T(obs[jl] & ~obs[jd if jd >= 0 else dn])
        else:
            pm = T(hidden[a.leak_day]) if a.leak_what == "hidden" else OBSg[jl]
        Yg[jl] = Yg[jl] + a.leak_delta * pm
        print(f"LEAK TEST: day {str(TIMES[jl])[:10]} (t{a.leak_offset:+d}, t = {str(TIMES[a.leak_day])[:10]}): ln CHL + {a.leak_delta} on "
              f"{int(pm.sum())} {a.leak_what} px", flush=True)
    chl_z = lambda idx: (Yg[idx] - chl_mu) / chl_sd * OBSg[idx]      # z-scored observed ln CHL (0 where unobserved), per batch
    OCEANg = T(ocean)
    if recipe:                                                 # their 13 channels (store_io_up.recipe_up2024) and target z on the GPU, float32
        REC = SU.recipe_up2024(d)
        RXg, RYg = T(REC["X"]), T(REC["Y"])
        assert REC["X"].shape[1] == 13 and np.array_equal(REC["masked"], HUP) and REC["checks"]["fake_ne_hidden_px"] == 0
        recipe_info = dict(channels=SU.UP24_CHANNELS, stats=REC["stats"], checks=REC["checks"], n_train=len(tr),
                           train_first=str(TIMES[tr[0]])[:10], train_last=str(TIMES[tr[-1]])[:10], target_mean=chl_mu, target_std=chl_sd,
                           zero_pred_allpx=dict({f"{nm}_mae": float(np.abs(REC["Y"][ix]).mean()) for nm, ix in (("val", va), ("test", te))},
                                                **{f"{nm}_mse": float((REC["Y"][ix].astype("float64") ** 2).mean()) for nm, ix in (("val", va), ("test", te))}))
        del REC
        print(f"recipe up2024: 13 input channels {SU.UP24_CHANNELS}; train {recipe_info['train_first']}..{recipe_info['train_last']} ({len(tr)} days); "
              f"target z mean {chl_mu:.4f} std {chl_sd:.4f}; all-zero prediction all-px z: {json.dumps(recipe_info['zero_pred_allpx'])}", flush=True)
        R3_DAY, R3_PREV, R3_NEXT, R3_STATIC = [0, 1, 2, 3, 6, 10, 11, 12], 7, 8, [4, 5, 9]

    # physics inputs: raw currents in index space (cells per day), their validity, stencil mask
    lat, lon = d["lat"][ys].astype("float64"), d["lon"][xs].astype("float64")
    lat_desc = bool(lat[0] > lat[-1])
    dx, dy, _, _ = grid_metrics(lat, lon)                      # signed metres along increasing index (dy < 0 when lat descends)
    DXg, DYg = T(dx.astype("float32")), T(dy.astype("float32"))

    def currents_np(idx):
        """Index-space currents (cells per day along axis -1 / -2) and their validity for days idx, from the store."""
        u = rd("u_curr", idx).astype("float32"); v = rd("v_curr", idx).astype("float32")
        okc = np.isfinite(u) & np.isfinite(v)
        return np.nan_to_num(u * DT / dx[None]), np.nan_to_num(v * DT / dy[None]), okc

    if a.feat_device == "gpu":
        UCg = torch.empty((n, H, W), dtype=torch.float16, device=dev); VCg = torch.empty_like(UCg)
        CURg = torch.empty((n, H, W), dtype=torch.bool, device=dev)
        for b0 in range(0, n, 500):
            u, v, okc = currents_np(slice(b0, b0 + 500))
            UCg[b0:b0 + 500] = torch.as_tensor(u, device=dev).half(); VCg[b0:b0 + 500] = torch.as_tensor(v, device=dev).half()
            CURg[b0:b0 + 500] = torch.as_tensor(okc, device=dev)

    def to_dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).pin_memory().to(dev, non_blocking=True)

    def feats(bi):
        """(B, F, H, W) float32 z-scored features (NaN -> 0) for the day indices bi (GPU tensor)."""
        if a.feat_device == "gpu":
            return Xg[bi].float()
        idx = bi.cpu().numpy()
        x = to_dev(np.stack([rd(sname(f), idx).astype("float32") for f in a.features], 1))
        return torch.nan_to_num((x - FMUg) / FSDg, nan=0.0)

    def currents(bi):
        """(ua, va, ok) for the days bi: cells per day along axis -1 and -2, float32, and the validity mask."""
        if a.feat_device == "gpu":
            return UCg[bi].float(), VCg[bi].float(), CURg[bi]
        u, v, okc = currents_np(bi.cpu().numpy())
        return to_dev(u), to_dev(v), to_dev(okc)

    # semi-Lagrangian departure-point sampling grid (normalised coordinates for F.grid_sample, align_corners=True)
    gy_, gx_ = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
    BASEg = torch.stack([gx_, gy_], -1)
    st = ocean.copy()
    st[1:] &= ocean[:-1]; st[:-1] &= ocean[1:]; st[:, 1:] &= ocean[:, :-1]; st[:, :-1] &= ocean[:, 1:]
    st[0] = st[-1] = False; st[:, 0] = st[:, -1] = False
    STENg = T(st)
    KDX2 = a.K * DT / (DXg ** 2); KDY2 = a.K * DT / (DYg ** 2)   # diffusion in per-day units

    offsets = [k for k in range(-a.context, a.context + 1) if k != 0]
    # temporal3d day window t-B..t+A (B = A = --context by default, i.e. the symmetric window)
    win_offs = list(range(-a.context_before, a.context_after + 1))
    if a.physmem:                                              # io7 --physmem-stride2: even offsets only beyond +-K (default: unchanged)
        from . import physmem as PMEM
        win_offs = PMEM.window_offsets(a.context_before, a.context_after, a.physmem_blend_k, a.physmem_stride2)
    NB = {k: T(T2.neighbour_index(TIMES, k)) for k in sorted(set(offsets) | set(win_offs) | {-1, 1} | (set(range(-4, 5)) if a.rw_loss else set())) if k != 0}
    clim_np, doy_bin = (T2.doy_climatology(y, obs, tr) if a.clim else (None, None))
    CLIMg = (T(clim_np) - chl_mu) / chl_sd if a.clim else None
    BINg = T(doy_bin) if a.clim else None
    if a.coords:
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        depth = np.log10(np.maximum(-d["topo"][ys, xs], 0) + 1)
        stat = np.stack([(la - la.mean()) / la.std(), (lo - lo.mean()) / lo.std(), (depth - depth.mean()) / depth.std()])
        STATg = T(stat.astype("float32"))
        doy = (np.asarray(TIMES) - np.asarray(TIMES).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
        ang = 2 * np.pi * doy / 365.25
        DOYg = T(np.stack([np.sin(ang), np.cos(ang)], 1).astype("float32"))
    if a.doy_input:
        doy = (np.asarray(TIMES) - np.asarray(TIMES).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
        ang = 2 * np.pi * doy / 365.25
        DOYg = T(np.stack([np.sin(ang), np.cos(ang)], 1).astype("float32"))
    if a.land_input:
        LANDg = T((~ocean).astype("float32"))
    n_geo = 0
    if a.geo_static:
        if a.store not in ("blk", "blk4"):
            raise SystemExit("--geo-static needs --store blk or blk4")
        oc_ = ocean.astype(bool)
        dep = np.log10(np.maximum(np.nan_to_num(np.asarray(d["depth"])[ys, xs], nan=5.0), 5.0))
        riv = np.log10(1 + np.nan_to_num(np.asarray(d["river_dist"])[:, ys, xs].astype("float32"), nan=0.0))
        samp = slice(None, None, 10)
        ok = (np.isfinite(rd("so", samp)) & np.isfinite(rd("u_curr", samp))).mean(0) > 0.5
        geo = np.concatenate([dep[None], riv, ok[None].astype("float32")]).astype("float32")
        for k in range(len(geo)):
            g = geo[k]; mu, sd = g[oc_].mean(), g[oc_].std()
            geo[k] = np.where(oc_, (g - mu) / (sd if sd > 0 else 1.0), 0.0)
        GEOg = T(geo); n_geo = len(geo)
        print(f"geo-static: {n_geo} channels (depth, 8 river distances, physics-ok mask: {ok[oc_].mean():.1%} of ocean px ok)", flush=True)
    n_up = (2 if a.doy_input else 0) + (1 if a.land_input else 0)       # --doy-input sin/cos, --land-input
    n_chl = 0 if a.no_chl_input else 1                       # CHL value channels on context days
    n_mask = 1 if (a.cloud_input or not a.no_chl_input) else 0
    n_chl_t = 0 if (a.no_chl_input or a.no_chl_today) else 1   # CHL value channel on day t
    n_mask_t = 1 if (a.cloud_input or n_chl_t) else 0
    foffs = [k for k in range(-a.feat_context, a.feat_context + 1) if k != 0]
    NBF = {k: T(T2.neighbour_index(TIMES, k)) for k in foffs}
    in_ch = len(a.features) * (1 + len(foffs)) + (n_chl_t + n_mask_t) + (n_chl + n_mask) * len(offsets) + (1 if a.clim else 0) + (5 if a.coords else 0) + (1 if a.year_input else 0) + n_geo + n_up
    if a.model == "temporal3d":   # per-day channels [F features, CHL value, visible mask] x T days; static 2D channels after the stem
        c_day, n_days = len(a.features) + 2, len(win_offs)
        n_static = (1 if a.clim else 0) + (5 if a.coords else 0) + (1 if a.year_input else 0) + n_geo + n_up + (MEM.N_CH if a.memory else 0)
        in_ch = (a.attn_dim if a.t3d_stem == "attn" else a.t3d_width) + n_static   # backbone input channels
        if a.t3d_stem == "patch":                              # blend, weight mass, entropy + day t's own channels + static
            in_ch = 3 + c_day + n_static
        if a.t3d_stem == "tokdec":                             # decoded context channels (+ blend, mass, entropy) + day t's own + static
            in_ch = a.tokdec_ch + (3 if a.tokdec_blend else 0) + c_day + n_static
            if a.tokdec_krig:                                  # io6: kriging channels (attention output, estimate, spread, mass, entropy [, signed 3])
                in_ch += a.krig_dv + 4 + (3 if a.krig_signed else 0)
            if a.wm_attn:                                      # io7: water-mass attention channels (chl_cnn/wmattn.py)
                in_ch += a.wm_dv + 4
    if recipe:
        if a.model == "temporal3d":
            c_day, n_days, n_static = 8, 3, 3
            in_ch = a.t3d_width + n_static
        else:
            in_ch = 13
    YEARg = T(((np.asarray(TIMES).astype("datetime64[Y]").astype(int) + 1970 - 2009) / 7.0).astype("float32"))
    if a.diff_fill:
        in_ch += 1
        ck = torch.load(Path(__file__).resolve().parents[1] / "models" / "spatial" / a.diff_fill / "model.pt", map_location=dev)
        fill_D = SN.SSTTensorD().to(dev); fill_D.load_state_dict(ck["dmodel"]); fill_D.requires_grad_(False)
        fill_beta = ck["log_beta"].exp().item(); fill_w = F.softplus(ck["raw_w"]).tolist()
        fill_offs = [k for k in range(-a.diff_fill_ctx, a.diff_fill_ctx + 1) if k != 0]
        assert len(fill_w) == len(fill_offs) and a.context >= a.diff_fill_ctx, "diff-fill context must match the spatial run and be <= --context"
        fill_grid = SN.Grid(ocean, lat, dev)
        sst_name = "sst32" if "sst32" in d else "sst"
        sub = rd(sst_name, tr[::20]).astype("float32"); sst_mu, sst_sd = float(np.nanmean(sub)), float(np.nanstd(sub)); del sub
        SSTFg = torch.empty((n, H, W), dtype=torch.float16, device=dev); SSTFOKg = torch.empty((n, H, W), dtype=torch.bool, device=dev)
        for b0 in range(0, n, 500):
            x = rd(sst_name, slice(b0, b0 + 500)).astype("float32")
            SSTFOKg[b0:b0 + 500] = torch.as_tensor(np.isfinite(x), device=dev)
            SSTFg[b0:b0 + 500] = torch.as_tensor(np.nan_to_num((x - sst_mu) / sst_sd), device=dev).half()
        CLIMLNg = T(clim_np)
        zero_f = torch.zeros((1, H, W), device=dev)
        print(f"diff-fill from {a.diff_fill}: beta {fill_beta:.4g} w {dict(zip(fill_offs, [round(w, 3) for w in fill_w]))} d0 {fill_D.log_d0.exp().item():.3g} g0 {fill_D.log_g0.exp().item():.3g} sst {sst_name}", flush=True)

    def diffusion_fill(bi, vis):
        """Steady-state anisotropic-diffusion fill (ln CHL) of today's visible pixels with neighbour-day soft terms."""
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            B = len(bi); zf = zero_f.expand(B, -1, -1)
            f = SN.corner_features(fill_grid, SSTFg[bi].float(), SSTFOKg[bi], zf, zf, torch.ones_like(vis), zf, zf, zf[0])
            Dc = fill_D(f)
            wsum = fill_beta + torch.zeros((B, H, W), device=dev); tsum = fill_beta * CLIMLNg[BINg[bi]]
            for w, k in zip(fill_w, fill_offs):
                j, ok, m = nb_masks(bi, k)
                wsum = wsum + w * m; tsum = tsum + w * m * Yg[j]
            free = fill_grid.ocean & ~vis
            return SN.steady_state(fill_grid, Dc, wsum, tsum / wsum, Yg[bi] * vis, free, tol=1e-5, maxiter=300)
    no_chl_t = a.no_chl_input or a.no_chl_today

    # ---- io5 input ablation (--ablate; chl_cnn/ablate.py): replacement values, applied by ablate_inputs in predict_z
    ABL_INFO = {}
    if a.ablate:
        from . import ablate as AB
        ab_ti = win_offs.index(0)
        ab_fn = AB.FEAT_GROUPS.get(a.ablate, [])
        ab_f = [a.features.index(f) for f in ab_fn]
        cs = 1 if a.clim else 0                                # static layout: [clim] [lat, lon, depth, sin doy, cos doy] [year] ...
        ab_s, ab_const = [], []
        if a.ablate in ("clim", "clim_static"):                # mean over ocean px of the training days, in the z units the model sees
            zc = (clim_np.astype("float64") - chl_mu) / chl_sd
            wb = np.bincount(doy_bin[tr], minlength=len(zc)).astype("float64")
            ab_s, ab_const = [0], [float((wb[:, None] * zc[:, ocean]).sum() / (wb.sum() * ocean.sum()))]
        elif a.ablate == "latlon":
            ab_s, ab_const = [cs, cs + 1], [float(stat[0][ocean].mean()), float(stat[1][ocean].mean())]
        elif a.ablate == "depth":
            ab_s, ab_const = [cs + 2], [float(stat[2][ocean].mean())]
        elif a.ablate == "year":
            ab_s, ab_const = [cs + (5 if a.coords else 0)], [float(YEARg.cpu().numpy().astype("float64")[tr].mean())]
        if a.ablate_mode == "doyclim":
            ab_dc = AB.build_doyclim(lambda f, idx: rd(sname(f), idx), ab_fn, tr, TIMES)
            assert all(ab_dc[f].shape == (366, H, W) for f in ab_fn), "doyclim cache grid differs from this store"
            DCZg = torch.as_tensor(np.stack([np.nan_to_num((ab_dc[f] - feat_stats[f]["mean"]) / feat_stats[f]["std"], nan=0.0) for f in ab_fn], 1)
                                   .astype("float32"), device=dev).half()               # (366, nf, H, W), z-scored as the feature store
            DOYIg = T(AB.doy_of(TIMES))
        ABL_INFO.update(group=a.ablate, mode=a.ablate_mode, features=ab_fn, feature_idx=ab_f, static_idx=ab_s, const=ab_const, t_index=ab_ti)
        print(f"ABLATE {a.ablate} ({a.ablate_mode}): features {ab_fn} static channels {ab_s} -> {ab_const}", flush=True)

    def ablate_inputs(bi, x3d, xs, cw):
        """--ablate: copies of the model inputs x3d (B, c_day, T, H, W), xs (B, n_static, H, W), cw (B, T, H, W) or None with one input
        group replaced (chl_cnn/ablate.py). The first call checks the result against the originals (ABL_INFO["check"])."""
        x0, s0, c0 = x3d, xs, cw
        x3d = x3d.clone(); xs = xs.clone(); cw = cw.clone() if cw is not None else None
        if ab_f:
            if a.ablate_mode == "mean":
                x3d[:, ab_f] = 0.0
            else:
                for kk, k in enumerate(win_offs):
                    j = bi if k == 0 else NBF[k][bi]
                    x3d[:, ab_f, kk] = DCZg[DOYIg[j.clamp(min=0)]].float() * (j >= 0).view(-1, 1, 1, 1).float()
        elif a.ablate == "ctx_chl":
            x3d[:, -2:, [kk for kk, k in enumerate(win_offs) if k != 0]] = 0.0
        elif a.ablate == "day_chl":
            x3d[:, -2:, ab_ti] = 0.0
        else:
            for c, v in zip(ab_s, ab_const):
                xs[:, c] = v
            if a.ablate == "clim" and cw is not None:
                cw[:] = ab_const[0]
        if "check" not in ABL_INFO:
            ABL_INFO["check"] = ablate_check(bi, x0, s0, c0, x3d, xs, cw)
        return x3d, xs, cw

    def ablate_check(bi, x0, s0, c0, x1, s1, c1):
        """Ablated slots hold the replacement; every other input value is unchanged (exact)."""
        C_, T_ = x0.shape[1], x0.shape[2]
        slot = torch.zeros((C_, T_), dtype=torch.bool, device=dev)
        if ab_f:
            slot[ab_f] = True
        elif a.ablate == "ctx_chl":
            slot[-2:] = True; slot[-2:, ab_ti] = False
        elif a.ablate == "day_chl":
            slot[-2:, ab_ti] = True
        sch = torch.zeros(s0.shape[1], dtype=torch.bool, device=dev); sch[ab_s] = True
        ok_x = bool(torch.equal(x1[:, ~slot], x0[:, ~slot])); ok_s = bool(torch.equal(s1[:, ~sch], s0[:, ~sch]))
        clim_cw = a.ablate == "clim" and c0 is not None
        ok_c = True if c0 is None else bool(torch.equal(c1, torch.full_like(c0, ab_const[0]) if clim_cw else c0))
        if ab_f and a.ablate_mode == "doyclim":                # independent reference for every batch sample and window day
            ref = np.zeros((len(bi), len(ab_f), T_, H, W), dtype="float32"); doy = AB.doy_of(TIMES)
            for kk, k in enumerate(win_offs):
                js = bi.cpu().numpy() if k == 0 else NBF[k][bi].cpu().numpy()
                for i, j in enumerate(js):
                    if j >= 0:
                        ref[i, :, kk] = np.stack([np.nan_to_num((ab_dc[f][doy[j]] - feat_stats[f]["mean"]) / feat_stats[f]["std"], nan=0.0)
                                                  for f in ab_fn]).astype("float32").astype("float16").astype("float32")
            ok_r = bool(np.array_equal(x1[:, ab_f].cpu().numpy(), ref))
        elif ab_f or a.ablate in AB.CHL_GROUPS:
            ok_r = bool((x1[:, slot] == 0).all())
        else:
            ok_r = all(bool((s1[:, c] == v).all()) for c, v in zip(ab_s, ab_const))
        changed = float((x1[:, slot] != x0[:, slot]).float().mean()) if slot.any() else float((s1[:, sch] != s0[:, sch]).float().mean())
        chk = dict(batch_days=[str(TIMES[i])[:10] for i in bi.cpu().numpy()[:3]], n_slots=int(slot.sum()), unchanged_x3d=ok_x, unchanged_static=ok_s,
                   clim3d_ok=ok_c, replacement_ok=ok_r, frac_values_changed_in_ablated_slots=changed, passed=ok_x and ok_s and ok_c and ok_r)
        print(f"ABLATE check (first batch): {json.dumps(chk)}", flush=True)
        if not chk["passed"]:
            raise SystemExit(f"--ablate check failed: {chk}")
        return chk

    # ---- priors
    prior_info = {}
    if a.src_clim:   # d/dt of the 8-day-bin climatology, central difference over bins, ln CHL per day
        DCLIMg = T(((np.roll(clim_np, -1, 0) - np.roll(clim_np, 1, 0)) / 16.0).astype("float32"))
        prior_info["dclim_abs_mean"] = float(np.abs(DCLIMg.cpu().numpy()[:, ocean]).mean())
    if a.src_ekman:
        print("ekman lag channels...", flush=True)
        lagnp = PR.ekman_lag_channels(d)
        LAGg = torch.empty((n, lagnp.shape[1], H, W), dtype=torch.float16, device=dev)
        for b0 in range(0, n, 500):
            LAGg[b0:b0 + 500] = torch.as_tensor(np.asarray(lagnp[b0:b0 + 500]), device=dev)
        TAPERg = T(PR.equator_taper(lat)[:, None] * np.ones((1, W), dtype="float32"))
        src_ekman = EkmanSource(lagnp.shape[1]).to(dev)
        prior_info["wE_pos_lag_mean_m_per_day"] = [float(np.asarray(lagnp[tr[::50], k])[:, ocean].mean()) for k in range(lagnp.shape[1])]
    if a.phys_weight == "courant":
        wco, co_clim = PR.courant_weight(d, tr)
        PWg = T(wco)
        prior_info["courant_clim"] = dict(median=float(np.nanmedian(co_clim[ocean])), frac_gt1=float(np.nanmean(co_clim[ocean] > 1)),
                                          weight_mean=float(wco[ocean].mean()))
        np.save(out / "courant_weight.npy", wco)
    else:
        PWg = torch.ones((H, W), device=dev)
    print("priors:", json.dumps(prior_info), flush=True)

    MEMLAST = {}                                               # --memory: last batch's anomaly / count (for --residual-base anom)
    if a.memory:
        memory = MEM.Memory(TIMES, OBSg, DONg, chl_z, clim_z=(lambda j: CLIMg[BINg[j]]) if a.clim else None, dev=dev)
        print(f"memory: {MEM.N_CH} channels (last/next valid obs outside +-4 d up to {memory.max_age} d with age, 8-/30-day medians with counts; "
              f"donor-masked); residual base {a.residual_base}" + (f" (shrinkage n / (n + {a.anom_n0}))" if a.residual_base == "anom" else ""), flush=True)

    def nb_masks(bi, k):
        """Neighbour day index, its validity, and the mask of its pixels visible in the input."""
        j = NB[k][bi]; ok = (j >= 0).view(-1, 1, 1); j = j.clamp(min=0)
        m = OBSg[j] & ok
        if a.context_clouds == "persist":
            jd = NB[k][DONg[bi]]; jd = torch.where(jd >= 0, jd, DONg[bi])
            m = m & OBSg[jd]
        return j, ok, m

    def up_static(ch, bi):
        """--doy-input (sin/cos day of year) and --land-input channels, appended last (nothing when both are off)."""
        if a.doy_input:
            ch.append(DOYg[bi].view(-1, 2, 1, 1).expand(-1, -1, H, W))
        if a.land_input:
            ch.append(LANDg.view(1, 1, H, W).expand(len(bi), -1, -1, -1))

    def make_inputs(bi, hid):
        vis = OBSg[bi] & ~hid
        ch = [feats(bi)]
        for k in foffs:
            j = NBF[k][bi]; ok = (j >= 0).view(-1, 1, 1, 1); j = j.clamp(min=0)
            ch.append(feats(j) * ok)
        if n_chl_t:
            ch.append((chl_z(bi) * vis).unsqueeze(1))
        if n_mask_t:
            ch.append(vis.float().unsqueeze(1))
        for k in offsets:
            j, ok, m = nb_masks(bi, k)
            if n_chl:
                ch.append((chl_z(j) * m).unsqueeze(1))
            if n_mask:
                ch.append(m.float().unsqueeze(1))
        if a.clim:
            ch.append(CLIMg[BINg[bi]].unsqueeze(1))
        if a.coords:
            ch.append(STATg.unsqueeze(0).expand(len(bi), -1, -1, -1))
            ch.append(DOYg[bi].view(-1, 2, 1, 1).expand(-1, -1, H, W))
        if a.year_input:
            ch.append(YEARg[bi].view(-1, 1, 1, 1).expand(-1, -1, H, W))
        if a.geo_static:
            ch.append(GEOg.unsqueeze(0).expand(len(bi), -1, -1, -1))
        up_static(ch, bi)
        if a.diff_fill:
            ch.append(((diffusion_fill(bi, vis) - chl_mu) / chl_sd).unsqueeze(1))
        return torch.cat(ch, 1)

    def static_inputs(bi):
        """(B, n_static, H, W) static 2D channels of day t (climatology, coords, doy, year), or None when there are none."""
        ch = []
        if a.clim:
            ch.append(CLIMg[BINg[bi]].unsqueeze(1))
        if a.coords:
            ch.append(STATg.unsqueeze(0).expand(len(bi), -1, -1, -1))
            ch.append(DOYg[bi].view(-1, 2, 1, 1).expand(-1, -1, H, W))
        if a.year_input:
            ch.append(YEARg[bi].view(-1, 1, 1, 1).expand(-1, -1, H, W))
        if a.geo_static:
            ch.append(GEOg.unsqueeze(0).expand(len(bi), -1, -1, -1))
        up_static(ch, bi)
        if a.memory:                                           # memory channels last (--attn-query-memory reads the last MEM.N_CH)
            if a.residual_base == "anom":
                mc, an, cn = memory(bi, want_anom=True)
                MEMLAST.update(anom=an * (cn / (cn + a.anom_n0)))
            else:
                mc = memory(bi)
            ch.append(mc)
        return torch.cat(ch, 1) if ch else None

    def make_inputs_3d(bi, hid):
        """temporal3d inputs: (B, c_day, T, H, W) day-ordered window t-B..t+A with per-day channels
        [features, CHL value (0 where not visible), visible mask] (zeros for missing neighbour days), and the static 2D channels.
        --no-chl-today: day t keeps its features, its CHL value and visible-mask channels are zero."""
        vis = OBSg[bi] & ~hid
        days = []
        for k in win_offs:
            if k == 0:
                if no_chl_t:
                    z = torch.zeros((len(bi), 2, H, W), device=dev)
                    days.append(torch.cat([feats(bi), z], 1))
                else:
                    days.append(torch.cat([feats(bi), (chl_z(bi) * vis).unsqueeze(1), vis.float().unsqueeze(1)], 1))
            else:
                jf = NBF[k][bi]; okf = (jf >= 0).view(-1, 1, 1, 1); jf = jf.clamp(min=0)
                j, ok, m = nb_masks(bi, k)
                fk = torch.zeros((len(bi), len(a.features), H, W), device=dev) if a.feat_today_only else feats(jf) * okf
                days.append(torch.cat([fk, (chl_z(j) * m).unsqueeze(1), m.float().unsqueeze(1)], 1))
        return torch.stack(days, 2), static_inputs(bi)

    def deq_state(bi, hid):
        """z0 (B,3,H,W) = climatology of t-1, t, t+1 (zeros without --clim); projection onto the visible pixels of the 3 days."""
        jm, _, mm = nb_masks(bi, -1); jp, _, mp = nb_masks(bi, 1)
        vis = OBSg[bi] & ~hid
        if a.clim:
            z0 = torch.stack([CLIMg[BINg[jm]], CLIMg[BINg[bi]], CLIMg[BINg[jp]]], 1)
        else:
            z0 = torch.zeros((len(bi), 3, H, W), device=dev)
        if a.deq_no_project:
            return z0, (lambda z: z)
        vmask = torch.stack([mm, vis, mp], 1)
        vvals = torch.stack([chl_z(jm) * mm, chl_z(bi) * vis, chl_z(jp) * mp], 1)
        return z0, (lambda z: torch.where(vmask, vvals.to(z.dtype), z))

    def predict_z(model, bi, hid, deq_steps=None):
        """(B, 3, H, W) z-scored ln CHL for days t-1, t, t+1."""
        if recipe:
            x = RXg[bi]
            if a.model == "temporal3d":
                z0 = torch.zeros((len(bi), 8, H, W), device=dev)
                prev_, next_ = z0.clone(), z0.clone()
                prev_[:, 4] = x[:, R3_PREV]; next_[:, 4] = x[:, R3_NEXT]
                return model(torch.stack([prev_, x[:, R3_DAY], next_], 2), x[:, R3_STATIC])
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            return model(x)
        if a.model == "temporal3d":
            if a.stoch:                                        # point prediction of the stochastic model: median of samples (fixed noise)
                return sample_median(predict_samples(model, bi, hid, a.stoch_val_samples, gen=fixed_gen()).float())
            if a.t3d_stem in ("patch", "tokdec"):
                if a.analog:                                   # io7: the batch's store days for the analog draw (chl_cnn/analog.py)
                    model.analog.set_days(bi)
                if a.physmem or a.bsmem:                       # io7 / blk8: past-only physics summaries of the batch's target days (physmem.py / bsmem.py)
                    model.mem_in = PMT.batch(bi)
                x3d, xs = make_inputs_3d(bi, hid)
                if a.ablate:                                   # io5: one input group replaced on every input path (x3d, static, clim3d)
                    x3d, xs, cw = ablate_inputs(bi, x3d, xs, clim_window(bi))
                    z = model(x3d, xs, day_ok(bi), cw)
                else:
                    z = model(x3d, xs, day_ok(bi), clim_window(bi))
            elif a.t3d_stem == "attn":
                x3d, xs = make_inputs_3d(bi, hid)
                z = model(x3d, xs, day_ok(bi), xs[:, -MEM.N_CH:] if a.attn_query_memory else None)
            elif a.ablate:                                     # io5 ablation, conv stem: x3d and the static channels
                x3d, xs = make_inputs_3d(bi, hid)
                x3d, xs, _ = ablate_inputs(bi, x3d, xs, None)
                z = model(x3d, xs)
            else:
                z = model(*make_inputs_3d(bi, hid))
        else:
            x = make_inputs(bi, hid)
            if a.model == "deq":
                z0, proj = deq_state(bi, hid)
                steps = deq_steps if deq_steps is not None else (None if model.training or not a.deq_eval_steps else a.deq_eval_steps)
                return model(x, z0, proj, steps=steps)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            z = model(x)
        return add_base(z, bi)

    def add_base(z, bi):
        """--residual: + DOY climatology of t-1, t, t+1 (+ the shrunk 30-day anomaly with --residual-base anom); z (..., 3, H, W)."""
        if a.residual:
            jm, _, _ = nb_masks(bi, -1); jp, _, _ = nb_masks(bi, 1)
            z = z + torch.stack([CLIMg[BINg[jm]], CLIMg[BINg[bi]], CLIMg[BINg[jp]]], 1)
            if a.residual_base == "anom":                      # set by static_inputs for this batch
                z = z + MEMLAST["anom"].unsqueeze(1)
        return z

    def clim_window(bi):
        """(B, T, H, W) DOY climatology z of each window day (day t's for days missing from the store)."""
        return torch.stack([CLIMg[BINg[bi]] if k == 0 else CLIMg[BINg[torch.where(NBF[k][bi] >= 0, NBF[k][bi], bi)]] for k in win_offs], 1)

    def day_ok(bi):
        """(B, T) bool: window day t+k is in the store (always true for k = 0)."""
        return torch.stack([torch.ones(len(bi), dtype=torch.bool, device=dev) if k == 0 else NBF[k][bi] >= 0 for k in win_offs], 1)

    def sample_median(X):
        """Median over the sample axis 0 (mean of the two middle values for an even count)."""
        v = X.sort(0).values; M = X.shape[0]
        return v[M // 2] if M % 2 else 0.5 * (v[M // 2 - 1] + v[M // 2])

    def fixed_gen():
        g = torch.Generator(device=dev); g.manual_seed(12345)
        return g

    def predict_samples(model, bi, hid, M, gen=None):
        """--stoch: (M, B, 3, H, W) z-scored samples for days t-1, t, t+1 (one stem pass, M noisy backbone passes)."""
        h = model.features(*make_inputs_3d(bi, hid))
        zs = [model.head(h, torch.randn((len(bi), a.stoch_ch, H, W), device=dev, generator=gen)) for _ in range(M)]
        return add_base(torch.stack(zs, 0), bi)

    def pixel_loss(pz, yz):
        e = pz - yz
        if a.loss == "l1":
            return e.abs()
        if a.loss == "mse":
            return e ** 2
        return torch.where(e.abs() < a.huber_delta, 0.5 * e ** 2, a.huber_delta * (e.abs() - 0.5 * a.huber_delta))

    def grad_terms(pz_t, bi, hid):
        """Gradient-difference pieces on day t: |d pred - d truth| along lat and lon (z units per cell) on pixel pairs
        where both have truth (observed) and at least one is hidden. Returns (sum, count)."""
        yz = (Yg[bi] - chl_mu) / chl_sd; ob = OBSg[bi]
        hz = hid if not no_chl_t else ob
        tot, cnt = 0.0, 0.0
        for ax in (-1, -2):
            dp = torch.diff(pz_t, dim=ax); dy_ = torch.diff(yz, dim=ax)
            ok = torch.narrow(ob, ax, 0, ob.shape[ax] - 1) & torch.narrow(ob, ax, 1, ob.shape[ax] - 1)
            hh = torch.narrow(hz, ax, 0, hz.shape[ax] - 1) | torch.narrow(hz, ax, 1, hz.shape[ax] - 1)
            m = (ok & hh).float()
            tot = tot + ((dp - dy_).abs() * m).sum(); cnt = cnt + m.sum()
        return tot, cnt

    LOSS_PRINTED = []

    def data_loss(pz, bi, hid):
        """Weighted hidden-pixel loss over the three output days (z units), plus the gradient-difference term."""
        if recipe:                                              # their loss: MSE in z over every pixel of day t (target NaN -> 0)
            if not LOSS_PRINTED:
                LOSS_PRINTED.append(1)
                print(f"recipe loss check: batch {len(bi)}, loss px per sample {pz[:, 1].numel() // len(bi)} (frame {H} x {W}), "
                      f"target z in this batch: mean {RYg[bi].mean().item():.4f}, finite-target px {int((RYg[bi] != 0).sum())}", flush=True)
            return ((pz[:, 1] - RYg[bi]) ** 2).mean()
        w = OBSg[bi].float() if no_chl_t else hid.float() + a.vis_weight * (OBSg[bi] & ~hid).float()
        if a.rw_loss and model.training:                       # io8: day-t hidden px weighted by their (distance x key) cell (chl_cnn/rwloss.py)
            w = RW_W[rw_cells(bi, hid)] * hid.float() + a.vis_weight * (OBSg[bi] & ~hid).float()
        num = (pixel_loss(pz[:, 1], (Yg[bi] - chl_mu) / chl_sd) * w).sum(); den = w.sum()
        if a.grad_weight > 0:
            gs, gc = grad_terms(pz[:, 1], bi, hid)
            num = num + a.grad_weight * gs * (den / gc.clamp(min=1))      # same normalisation scale as the pixel term
        for c, k in [(0, -1), (2, 1)]:
            if a.model == "temporal3d" and k not in win_offs:     # day outside the input window (t+1 with --context-after 0): head dropped
                continue
            j, ok, m = nb_masks(bi, k)
            hk = (OBSg[j] & ok) if a.no_chl_input else (OBSg[j] & ok & ~m)
            wk = a.nb_weight * hk.float()
            num = num + (pixel_loss(pz[:, c], (Yg[j] - chl_mu) / chl_sd) * wk).sum(); den = den + wk.sum()
        return num / den.clamp(min=1)

    def data_loss_stoch(PZ, bi, hid):
        """--stoch: data_loss with the fair CRPS of the M samples PZ (M, B, 3, H, W) in place of |error|, on the same pixels and weights
        (hidden + --vis-weight x visible on day t, the t+-1 heads), and on the gradient-difference pairs of day t (same --grad-weight and
        normalisation as grad_terms)."""
        yz = (Yg[bi] - chl_mu) / chl_sd; ob = OBSg[bi]
        w = hid.float() + a.vis_weight * (ob & ~hid).float()
        num = (STOCH.fair_crps(PZ[:, :, 1], yz) * w).sum(); den = w.sum()
        if a.grad_weight > 0:
            gs, gc = 0.0, 0.0
            for ax in (-1, -2):
                dP = torch.diff(PZ[:, :, 1], dim=ax); dy_ = torch.diff(yz, dim=ax)
                ok = torch.narrow(ob, ax, 0, ob.shape[ax] - 1) & torch.narrow(ob, ax, 1, ob.shape[ax] - 1)
                hh = torch.narrow(hid, ax, 0, hid.shape[ax] - 1) | torch.narrow(hid, ax, 1, hid.shape[ax] - 1)
                m = (ok & hh).float()
                gs = gs + (STOCH.fair_crps(dP, dy_) * m).sum(); gc = gc + m.sum()
            num = num + a.grad_weight * gs * (den / gc.clamp(min=1))
        for c, k in [(0, -1), (2, 1)]:
            if k not in win_offs:
                continue
            j, ok, m = nb_masks(bi, k)
            wk = a.nb_weight * (OBSg[j] & ok & ~m).float()
            num = num + (STOCH.fair_crps(PZ[:, :, c], (Yg[j] - chl_mu) / chl_sd) * wk).sum(); den = den + wk.sum()
        return num / den.clamp(min=1)

    def source_term(bi):
        """Prior source S (B,H,W) in ln CHL per day (zero tensor if no prior is on)."""
        s = torch.zeros((len(bi), H, W), device=dev)
        if a.src_clim:
            s = s + DCLIMg[BINg[bi]]
        if a.src_ekman:
            s = s + src_ekman(LAGg[bi].float()) * TAPERg
        return s

    def transport_residual(pz, bi, hid, form=None, cur=None):
        """Returns (r, r_base, valid): full residual, transport-only residual (ln CHL/day), validity mask.
        form: 'upwind' (tendency + upwind u.grad C(t) - K lap C(t)) or 'semi-lagrangian' ([C(t+1)(x) - C(t-1)(x - 2 u dt)] / 2dt
        - K lap C(t)); default a.phys. cur: precomputed currents(bi)."""
        form = form or a.phys
        Cc = pz * chl_sd + chl_mu                                    # (B,3,H,W) ln CHL
        jm, okm, mm = nb_masks(bi, -1); jp, okp, mp = nb_masks(bi, 1)
        if not a.no_paste:
            vis = OBSg[bi] & ~hid
            Cc = torch.stack([torch.where(mm, Yg[jm], Cc[:, 0]), torch.where(vis, Yg[bi], Cc[:, 1]),
                              torch.where(mp, Yg[jp], Cc[:, 2])], 1)
        c0, c1, c2 = Cc[:, 0], Cc[:, 1], Cc[:, 2]
        ua, va_, okc = cur if cur is not None else currents(bi)     # cells per day along index axes (-1, -2)
        dxp = c1[:, :, 2:] - c1[:, :, 1:-1]; dxm = c1[:, :, 1:-1] - c1[:, :, :-2]      # along lon (axis -1)
        dyp = c1[:, 2:, :] - c1[:, 1:-1, :]; dym = c1[:, 1:-1, :] - c1[:, :-2, :]      # along lat (axis -2)
        lap = torch.zeros_like(c1)
        lap[:, :, 1:-1] += (dxp - dxm) * KDX2[:, 1:-1]
        lap[:, 1:-1, :] += (dyp - dym) * KDY2[1:-1, :]
        valid = STENg & okc & okm & okp
        if form == "upwind":
            tend = (c2 - c0) / 2.0
            adv = torch.zeros_like(c1)
            ui, vi = ua[:, :, 1:-1], va_[:, 1:-1, :]
            adv[:, :, 1:-1] += torch.where(ui > 0, ui * dxm, ui * dxp)
            adv[:, 1:-1, :] += torch.where(vi > 0, vi * dym, vi * dyp)
            r_base = tend + adv - lap
            if a.co_max > 0:
                valid = valid & ((ua.abs() + va_.abs()) <= a.co_max)
        else:
            # departure point x - 2 u dt in cells (clipped), sampled bilinearly from the day t-1 field; ocean mask sampled alike
            sx = (-2.0 * ua).clamp(-a.lag_max, a.lag_max); sy = (-2.0 * va_).clamp(-a.lag_max, a.lag_max)
            g = (BASEg.unsqueeze(0) + torch.stack([sx * (2.0 / (W - 1)), sy * (2.0 / (H - 1))], -1)).to(c0.dtype)   # grid_sample needs the grid in the input dtype
            ocf = OCEANg.float().view(1, 1, H, W)
            num = F.grid_sample((c0 * OCEANg).unsqueeze(1), g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
            den = F.grid_sample(ocf.expand(len(bi), -1, -1, -1), g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
            c0d = num / den.clamp(min=1e-3)
            r_base = (c2 - c0d) / 2.0 - lap
            valid = valid & (den >= 0.99)
        r = r_base - source_term(bi)
        return r, r_base, valid

    def phys_loss(r, valid):
        e = r.abs() if a.phys_loss == "l1" else r ** 2
        w = valid.float() * PWg
        return (e * w).sum() / w.sum().clamp(min=1e-6)

    dilations = None if a.dilations is None else [int(v) for v in a.dilations.split(",")]   # None -> DilatedCNN default
    if a.model == "deq":
        cell = DQ.DEQCell(in_ch, n_state=3, width=a.width or 64, dilations=tuple(int(v) for v in a.cell_dilations.split(",")),
                          eta=a.deq_eta, learn_eta=not a.deq_fixed_eta, spectral=a.deq_spectral)
        model = DQ.DEQ(cell, solver=a.deq_solver, steps=a.deq_steps, tol=a.deq_tol, grad="phantom", phantom_steps=a.deq_phantom_steps,
                       phantom_tau=a.deq_phantom_tau, anderson_m=a.deq_anderson_m, anderson_beta=a.deq_anderson_beta, checkpoint=a.deq_ckpt).to(dev)
        print(f"deq: solver {a.deq_solver} steps {a.deq_steps} tol {a.deq_tol} phantom {a.deq_phantom_steps} x tau {a.deq_phantom_tau}  cell dilations {a.cell_dilations} eta {a.deq_eta}  project {not a.deq_no_project}  spectral {a.deq_spectral}  ckpt {a.deq_ckpt}", flush=True)
    elif a.model == "temporal3d" and a.t3d_stem == "patch":
        from .ptok import PatchTokenNet
        model = PatchTokenNet(c_day, n_static, n_days, win_offs.index(0), T(ocean), P=a.ptok_patch, dim=a.ptok_dim, heads=a.ptok_heads,
                              layers=a.ptok_layers, arm=a.ptok_arm, value=a.ptok_value, sst_index=a.features.index("sst"), backbone=a.t3d_backbone, width=a.width,
                              dropout=a.dropout, out_channels=3, dilations=dilations).to(dev)
        n_bb = count_params(model.backbone)
        print(f"patch-token stem: {n_days} days, P {a.ptok_patch} -> {model.Np} ocean patches of {model.nPy} x {model.nPx}, dim {a.ptok_dim}, "
              f"heads {a.ptok_heads}, layers {a.ptok_layers}, arm {a.ptok_arm}, value {a.ptok_value} -> {in_ch} ch -> dilated backbone (width {a.width}, FiLM on block 1)  "
              f"stem params {count_params(model) - n_bb}, backbone {n_bb}", flush=True)
    elif a.model == "temporal3d" and a.t3d_stem == "tokdec":
        from .tokdec import TokDecNet
        if a.tokdec_krig:                                      # io6: TokDecNet (blend off) + the learned-kriging stem (chl_cnn/krig.py)
            from . import krig as KRIG
            from . import krig_triton as KRIGT
            KRIGT.CFG["PREC"] = a.krig_prec
            fi = a.features.index
            vel = dict(mu_u=feat_stats["u_curr"]["mean"], sd_u=feat_stats["u_curr"]["std"], mu_v=feat_stats["v_curr"]["mean"],
                       sd_v=feat_stats["v_curr"]["std"], dx=torch.as_tensor(np.broadcast_to(dx, (H, W)).astype("float32")),
                       dy=torch.as_tensor(np.broadcast_to(dy, (H, W)).astype("float32")))
            krig_kw = dict(sst_index=fi("sst"), sss_index=fi("so"), u_index=fi("u_curr"), v_index=fi("v_curr"), vel=vel, dim=a.krig_dim,
                           heads=a.krig_heads, dv=a.krig_dv, chunk=a.krig_chunk, signed=a.krig_signed, advect=a.krig_advect, s0=a.krig_s0,
                           impl=a.krig_impl)
            model = KRIG.build_krig_tokdec(c_day, n_static, n_days, win_offs.index(0), T(ocean), krig_kw, P=a.ptok_patch, dim=a.tokdec_dim,
                                           ch=a.tokdec_ch, heads=a.tokdec_heads, layers=a.tokdec_layers, sst_index=fi("sst"),
                                           spatial=a.tokdec_spatial, backbone=a.t3d_backbone, width=a.width, dropout=a.dropout, out_channels=3,
                                           dilations=dilations).to(dev)
            print(f"kriging stem: {model.krig.K} stencil offsets (|dt| <= 4), D {a.krig_dim}, heads {a.krig_heads}{' + signed pair' if a.krig_signed else ''}, "
                  f"dv {a.krig_dv}, {model.krig.n_out} channels, advect {a.krig_advect}, impl {model.krig.impl} ({a.krig_prec}), "
                  f"{model.krig.N} ocean px; kriging params {count_params(model.krig)}", flush=True)
        elif a.wm_attn:                                        # io7: TokDecNet (blend as passed) + water-mass attention (chl_cnn/wmattn.py)
            from . import wmattn as WMA
            model = WMA.build(a, c_day, n_static, n_days, win_offs.index(0), T(ocean), P=a.ptok_patch, dim=a.tokdec_dim, ch=a.tokdec_ch,
                              heads=a.tokdec_heads, layers=a.tokdec_layers, blend=a.tokdec_blend, sst_index=a.features.index("sst"),
                              spatial=a.tokdec_spatial, backbone=a.t3d_backbone, width=a.width, dropout=a.dropout, out_channels=3,
                              dilations=dilations).to(dev)
        elif a.physmem or a.bsmem:                             # io7: TokDecNet (blend on +-K) over win_offs + physics-memory tokens (chl_cnn/physmem.py)
            from . import physmem as PMEM
            if a.bsmem:                                        # blk8: Black Sea memory table (chl_cnn/bsmem.py), same token net with 12 other channels
                from . import bsmem as BSM
                from . import store_blk4 as SB4
                rnames = json.loads((SB4.store_path() / "river_names.json").read_text())
                PMT = BSM.BSMemTable(lambda f, t=slice(None): rd(f, t), TIMES, tr, ocean, lat, lon, a.ptok_patch, np.asarray(d["river_dist"])[:, ys, xs],
                                     rnames, log=lambda s: print(s, flush=True))
            else:
                PMT = PMEM.MemTable(lambda f, t=slice(None): rd(sname(f), t), TIMES, tr, ocean, lat, lon, a.ptok_patch,   # sname: blk "sst" -> "sst32"
                                    log=lambda s: print(s, flush=True), wind_is_stress=a.store in ("blk", "blk4"))        # blk: u/v_wind are stress (Pa)
            model = PMEM.PhysMemTokDecNet(c_day, n_static, win_offs, T(ocean), blend_k=a.physmem_blend_k, P=a.ptok_patch, dim=a.tokdec_dim,
                                          ch=a.tokdec_ch, heads=a.tokdec_heads, layers=a.tokdec_layers, sst_index=a.features.index("sst"),
                                          spatial=a.tokdec_spatial, backbone=a.t3d_backbone, width=a.width, dropout=a.dropout, out_channels=3,
                                          dilations=dilations, mem_ch=(BSM.N_MEM if a.bsmem else PMEM.N_MEM)).to(dev)
            print(f"{'bsmem' if a.bsmem else 'physmem'}: token days {win_offs[0]:+d}..{win_offs[-1]:+d} ({n_days}), blend days {len(model.bsel)} (|d - t| <= {a.physmem_blend_k}); "
                  f"memory params {count_params(model.mem)}", flush=True)
        else:
            model = TokDecNet(c_day, n_static, n_days, win_offs.index(0), T(ocean), P=a.ptok_patch, dim=a.tokdec_dim, ch=a.tokdec_ch,
                              heads=a.tokdec_heads, layers=a.tokdec_layers, blend=a.tokdec_blend, sst_index=a.features.index("sst"),
                              backbone=a.t3d_backbone, width=a.width, dropout=a.dropout, out_channels=3, dilations=dilations,
                              **(dict(spatial=a.tokdec_spatial) if a.tokdec_spatial else {}),
                              **(dict(spatial_pool=a.tokdec_spatial_pool) if a.tokdec_spatial_pool != 1 else {})).to(dev)
        if a.analog:                                           # io7: analog-token cross-attention on the pooled tokens (chl_cnn/analog.py)
            from . import analog as AN
            t_an = time.time()
            bank = AN.build_bank(model, TIMES, tr, feats, chl_z, lambda i: OBSg[i], lambda i: CLIMg[BINg[i]], val_test=(va, te))
            AN.attach(model, bank, k=a.analog_k, window=a.analog_window, gap=a.analog_gap, seed=a.analog_seed, train_seed=a.seed)
            print(f"analog tokens: bank {len(bank.days)} training days x {model.Np} patches x {bank.data.shape[-1]} "
                  f"({bank.data.numel() * 2 / 2**30:.2f} GiB fp16, {time.time() - t_an:.0f} s), K {a.analog_k}, +-{a.analog_window} DOY, "
                  f"gap {a.analog_gap} d; analog params {count_params(model.analog)}", flush=True)
        n_bb = count_params(model.backbone)
        print(f"token-decoder stem: {n_days} days, P {a.ptok_patch} -> {model.Np} ocean patches of {model.nPy} x {model.nPx}, dim {a.tokdec_dim}, "
              f"heads {a.tokdec_heads}, layers {a.tokdec_layers}, decoded ch {a.tokdec_ch}, blend {a.tokdec_blend}"
              f"{f', spatial blocks {a.tokdec_spatial}' if a.tokdec_spatial else ''}"
              f"{f' (pooled {a.tokdec_spatial_pool} x {a.tokdec_spatial_pool} -> {model.Nc} coarse tokens of {model.nCy} x {model.nCx})' if a.tokdec_spatial_pool != 1 else ''}"
              f"{', kriging' if a.tokdec_krig else ''} -> {in_ch} ch -> dilated backbone "
              f"(width {a.width})  stem params {count_params(model) - n_bb}, backbone {n_bb}", flush=True)
    elif a.model == "temporal3d" and a.t3d_stem == "attn":
        from .tattn import TemporalAttnNet
        model = TemporalAttnNet(c_day, n_static, n_days, win_offs.index(0), backbone=a.t3d_backbone, stem_width=a.attn_dim, heads=a.attn_heads,
                                layers=a.attn_layers, q_extra=(MEM.N_CH if a.attn_query_memory else 0), width=a.width, dropout=a.dropout, out_channels=3,
                                depth=a.unet_depth, spectral=a.spectral, modes=a.spectral_modes, spec_ch=a.spectral_ch, dilations=dilations).to(dev)
        print(f"temporal attention stem: {n_days} days x {c_day} channels/day, width {a.attn_dim}, heads {a.attn_heads}, layers {a.attn_layers}, "
              f"query memory {a.attn_query_memory} -> {a.attn_dim} + {n_static} static -> {a.t3d_backbone} backbone (width {a.width})  "
              f"stem params {count_params(model.stem)}", flush=True)
    elif a.model == "temporal3d":
        T3D = STOCH.StochTemporal3DNet if a.stoch else Temporal3DNet
        model = T3D(c_day, n_static, n_days, backbone=a.t3d_backbone, stem_width=a.t3d_width, stem_depth=a.t3d_depth, kt=a.t3d_kernel_t,
                    width=a.width, dropout=a.dropout, out_channels=3, depth=a.unet_depth,
                    spectral=a.spectral, modes=a.spectral_modes, spec_ch=a.spectral_ch, dilations=dilations,
                    **(dict(noise_ch=a.stoch_ch) if a.stoch else {})).to(dev)
        print(f"temporal3d: stem {a.t3d_depth} x Conv3d({a.t3d_width}, kernel ({a.t3d_kernel_t},3,3)) on {c_day} channels/day x {n_days} days, "
              f"collapse ({n_days},1,1) -> {a.t3d_width} + {n_static} static -> {a.t3d_backbone} backbone (width {a.width})  "
              f"stem params {count_params(model.stem)}", flush=True)
    else:
        model = build_model(a.model, in_ch, width=a.width, dropout=a.dropout, out_channels=3, depth=a.unet_depth,
                            spectral=a.spectral, modes=a.spectral_modes, spec_ch=a.spectral_ch, dilations=dilations).to(dev)
    channels_last = not a.no_channels_last and a.spectral == 0 and a.model == "dilated"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    params = list(model.parameters()) + (list(src_ekman.parameters()) if a.src_ekman else [])
    opt = torch.optim.Adam(params, lr=a.lr, eps=1e-7) if recipe else torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    def make_sched():
        if a.sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        if a.sched == "plateau":                               # stepped with the model-selection score after each val pass
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=a.plateau_factor, patience=a.plateau_patience, min_lr=a.min_lr)
        return None
    sched = make_sched()
    scaler = torch.amp.GradScaler(enabled=not a.no_amp)
    print(f"model {a.model} in_ch {in_ch} params {count_params(model)}  residual pixels/day (stencil) {int(st.sum())}", flush=True)

    RW_INFO = None                                             # io8 --rw-loss: cell weights from training fake / real-cloud px (chl_cnn/rwloss.py)
    if a.rw_loss:
        from . import rwloss as RWL
        RW_BIN = RWL.Binner(dev)

        def rw_cells(bi, hid):
            """(B, H, W) (distance x key) cell of every px under the model-input visibility of the batch (hidden masks hid)."""
            return RW_BIN.cells(OBSg[bi] & ~hid, {k: nb_masks(bi, k)[2] for k in range(-4, 5) if k != 0})

        if not a.eval_only:
            t_rw = time.time(); rw_rng = np.random.default_rng(a.seed + 1000003)        # own generator: training draws / order unchanged
            cnt = {"real": torch.zeros(RWL.N_CELL, dtype=torch.float64, device=dev), "fake": torch.zeros(RWL.N_CELL, dtype=torch.float64, device=dev)}
            with torch.no_grad():
                for _ in range(a.rw_draws):                    # as train_epoch's donor draw
                    if a.donor_exclude > 0:
                        dn = tr[C.seasonal_donor_indices(TIMES[tr], rw_rng, a.donor_window, exclude=a.donor_exclude)] if a.donor_window > 0 \
                            else tr[C.donor_indices(len(tr), rw_rng, times=TIMES[tr], exclude=a.donor_exclude)]
                    else:
                        dn = tr[C.seasonal_donor_indices(TIMES[tr], rw_rng, a.donor_window)] if a.donor_window > 0 else tr[C.donor_indices(len(tr), rw_rng)]
                    HIDg[T(tr)] = T(obs[tr] & ~obs[dn]); DONg[T(tr)] = T(dn)          # overwritten by every training epoch
                    for b in range(0, len(tr), 32):
                        bi = T(tr[b:b + 32]); hid = HIDg[bi]
                        cell = rw_cells(bi, hid)
                        cnt["fake"] += torch.bincount(cell[hid], minlength=RWL.N_CELL).double()
                        cnt["real"] += torch.bincount(cell[OCEANg & ~OBSg[bi]], minlength=RWL.N_CELL).double()
            rw_w, RW_INFO = RWL.cap_weights(cnt["real"].cpu().numpy(), cnt["fake"].cpu().numpy(), a.rw_cap)
            RW_INFO.update(draws=a.rw_draws, n_train_days=len(tr), seconds=time.time() - t_rw)
            (out / "rw_loss.json").write_text(json.dumps(RW_INFO, indent=1))
            RW_W = torch.as_tensor(rw_w, dtype=torch.float32, device=dev)
            print(f"rw-loss: {a.rw_draws} draws x {len(tr)} training days, {RW_INFO['n_fake']} fake / {RW_INFO['n_real']} real-cloud px, "
                  f"cap {a.rw_cap}: max w {RW_INFO['max_weight']:.3f}, {RW_INFO['n_cells_at_cap']} cells at cap, Kish ESS {RW_INFO['kish_ess_frac']:.3f}, "
                  f"TV fake-real {RW_INFO['tv_fake_vs_real']:.3f} -> {RW_INFO['tv_reweighted_vs_real']:.3f}, uncovered real mass "
                  f"{RW_INFO['uncovered_real_mass']:.4f} ({RW_INFO['seconds']:.0f} s)\n{RWL.table(RW_INFO)}", flush=True)

    TRAIN_DEQ = dict(iters=0.0, res=0.0); EVAL_DEQ = dict(iters=0.0, res=0.0, res_max=0.0)

    def lam_at(ep):
        return a.lam * min(1.0, (ep + 1) / max(a.phys_warmup, 1))

    def restore_val_masks():
        """--train-on-val: training donors overwrite the val days' hidden masks each epoch; put the fixed val set back before scoring."""
        if a.train_on_val:
            HIDg[T(va)] = T(hid_va); DONg[T(va)] = T(don_va)

    def train_epoch(ep):
        model.train()
        if a.fake_mask == "shift10":                           # fixed mask (set once in HIDg); no donor draw
            pass
        elif a.donor_exclude > 0:
            donors = tr[C.seasonal_donor_indices(TIMES[tr], rng, a.donor_window, exclude=a.donor_exclude)] if a.donor_window > 0 \
                else tr[C.donor_indices(len(tr), rng, times=TIMES[tr], exclude=a.donor_exclude)]
        else:
            donors = tr[C.seasonal_donor_indices(TIMES[tr], rng, a.donor_window)] if a.donor_window > 0 else tr[C.donor_indices(len(tr), rng)]
        if a.fake_mask != "shift10":
            HIDg[T(tr)] = T(obs[tr] & ~obs[donors]); DONg[T(tr)] = T(donors)
        lam = lam_at(ep)
        td = tp = 0.0; cnt = 0; dq_it = dq_res = 0.0; dq_n = 0
        order = T(buffer_shuffle(tr, 512, rng) if recipe else rng.permutation(tr))
        for nb_, b in enumerate(range(0, len(tr), a.batch)):
            if a.max_batches and nb_ >= a.max_batches:
                break
            bi = order[b:b + a.batch]; hid = HIDg[bi]
            with torch.autocast("cuda", enabled=not a.no_amp):
                if a.stoch:
                    PZ = predict_samples(model, bi, hid, a.stoch_train_samples)
                else:
                    pz = predict_z(model, bi, hid)
            if a.stoch:                                        # fair CRPS over M samples; the residual diagnostic uses sample 0
                PZ = PZ.float(); pz = PZ[0]
                ld = data_loss_stoch(PZ, bi, hid)
            else:
                pz = pz.float()
                ld = data_loss(pz, bi, hid)
            r, _, valid = transport_residual(pz, bi, hid)
            lp = phys_loss(r, valid)
            loss = ld + lam * lp
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            td += ld.item() * len(bi); tp += lp.item() * len(bi); cnt += len(bi)
            if a.model == "deq":
                dq_it += DQ.LAST_STATS["iters"]; dq_res += DQ.LAST_STATS["res"]; dq_n += 1
        if a.model == "deq":
            TRAIN_DEQ.update(iters=dq_it / max(dq_n, 1), res=dq_res / max(dq_n, 1))
        return td / cnt, tp / cnt, lam

    @torch.no_grad()
    def residual_stats(idx, collect_preds=None, collect_samples=None):
        """Hidden-pixel MAE (ln CHL) on day t; unweighted mean|r| and mean|r_base| under real clouds / observed.
        --stoch: the point prediction is the median of K samples (K = --stoch-final-samples when collect_samples is given, else
        --stoch-val-samples; fixed noise seed), and the fair CRPS of the samples on the hidden pixels is returned as "crps" (ln CHL)."""
        model.eval()
        acc = dict(err=0.0, sq=0.0, cnt=0.0, rc=0.0, nc=0.0, ro=0.0, no=0.0, bc=0.0, bo=0.0, gp=0.0, gt=0.0, gpt=0.0, gpp=0.0, gtt=0.0, gn=0.0,
                   up_se=0.0, up_ae=0.0, up_n=0, crps=0.0)
        if a.stoch:
            K = a.stoch_final_samples if collect_samples is not None else a.stoch_val_samples
            gen = fixed_gen()
        other = "semi-lagrangian" if a.phys == "upwind" else "upwind"      # the residual form not used for training, scored too
        acc2 = dict(rc=0.0, nc=0.0, ro=0.0, no=0.0)
        dq = dict(iters=0.0, res=0.0, res_max=0.0, n=0)
        for b in range(0, len(idx), a.eval_batch):
            bi = T(idx[b:b + a.eval_batch]); hid = HIDg[bi]
            with torch.autocast("cuda", enabled=not a.no_amp):
                if a.stoch:
                    PZ = predict_samples(model, bi, hid, K, gen=gen).float()
                    pz = sample_median(PZ)
                else:
                    pz = predict_z(model, bi, hid)
            pz = pz.float()
            if a.stoch:
                acc["crps"] += (STOCH.fair_crps(PZ[:, :, 1], (Yg[bi] - chl_mu) / chl_sd) * hid).sum().item() * chl_sd
                if collect_samples is not None:
                    collect_samples[:, b:b + len(bi)] = (PZ[:, :, 1] * chl_sd + chl_mu).half().cpu().numpy()
            if a.model == "deq":
                dq["iters"] += DQ.LAST_STATS["iters"]; dq["res"] += DQ.LAST_STATS["res"]; dq["res_max"] = max(dq["res_max"], DQ.LAST_STATS["res_max"]); dq["n"] += 1
            if collect_preds is not None:
                collect_preds[b:b + len(bi)] = (pz[:, 1] * chl_sd + chl_mu).cpu().numpy()
            e_ = (pz[:, 1] * chl_sd + chl_mu - Yg[bi]) * hid
            if recipe:                                          # their all-pixel z metric (Keras evaluate), pooled over px
                ez = (pz[:, 1] - RYg[bi]).double()
                acc["up_se"] += (ez ** 2).sum().item(); acc["up_ae"] += ez.abs().sum().item(); acc["up_n"] += ez.numel()
            acc["err"] += e_.abs().sum().item(); acc["sq"] += (e_ ** 2).sum().item(); acc["cnt"] += hid.sum().item()
            # gradient structure on hidden pixels: pairs with truth at both ends and at least one hidden end (ln CHL per cell)
            pt = pz[:, 1] * chl_sd + chl_mu; ob = OBSg[bi]
            for ax in (-1, -2):
                dp = torch.diff(pt, dim=ax); dt_ = torch.diff(Yg[bi], dim=ax)
                m = (torch.narrow(ob, ax, 0, ob.shape[ax] - 1) & torch.narrow(ob, ax, 1, ob.shape[ax] - 1)
                     & (torch.narrow(hid, ax, 0, hid.shape[ax] - 1) | torch.narrow(hid, ax, 1, hid.shape[ax] - 1))).float()
                acc["gp"] += (dp.abs() * m).sum().item(); acc["gt"] += (dt_.abs() * m).sum().item()
                acc["gpt"] += (dp * dt_ * m).sum().item(); acc["gpp"] += (dp * dp * m).sum().item(); acc["gtt"] += (dt_ * dt_ * m).sum().item(); acc["gn"] += m.sum().item()
            cur = currents(bi)
            r, rb, valid = transport_residual(pz, bi, hid, cur=cur)
            cloud = valid & ~OBSg[bi]; clear = valid & OBSg[bi]
            acc["rc"] += (r.abs() * cloud).sum().item(); acc["nc"] += cloud.sum().item()
            acc["ro"] += (r.abs() * clear).sum().item(); acc["no"] += clear.sum().item()
            acc["bc"] += (rb.abs() * cloud).sum().item(); acc["bo"] += (rb.abs() * clear).sum().item()
            _, rb2, valid2 = transport_residual(pz, bi, hid, form=other, cur=cur)
            cloud2 = valid2 & ~OBSg[bi]; clear2 = valid2 & OBSg[bi]
            acc2["rc"] += (rb2.abs() * cloud2).sum().item(); acc2["nc"] += cloud2.sum().item()
            acc2["ro"] += (rb2.abs() * clear2).sum().item(); acc2["no"] += clear2.sum().item()
        if a.model == "deq":
            EVAL_DEQ.update(iters=dq["iters"] / max(dq["n"], 1), res=dq["res"] / max(dq["n"], 1), res_max=dq["res_max"])
        return dict(crps=acc["crps"] / max(acc["cnt"], 1), up_mse=acc["up_se"] / max(acc["up_n"], 1), up_mae=acc["up_ae"] / max(acc["up_n"], 1), up_n=int(acc["up_n"]),
                    mae=acc["err"] / max(acc["cnt"], 1), rmse=(acc["sq"] / max(acc["cnt"], 1)) ** 0.5, n_hidden=int(acc["cnt"]), phys_form=a.phys,
                    deq_eval=dict(EVAL_DEQ) if a.model == "deq" else None,
                    other_form=other, abs_rbase_other_cloud=acc2["rc"] / max(acc2["nc"], 1), abs_rbase_other_obs=acc2["ro"] / max(acc2["no"], 1),
                    n_cloud_other=int(acc2["nc"]), n_obs_other=int(acc2["no"]), abs_r_cloud=acc["rc"] / max(acc["nc"], 1), abs_r_obs=acc["ro"] / max(acc["no"], 1),
                    abs_rbase_cloud=acc["bc"] / max(acc["nc"], 1), abs_rbase_obs=acc["bo"] / max(acc["no"], 1),
                    n_cloud=int(acc["nc"]), n_obs=int(acc["no"]),
                    grad_ratio=acc["gp"] / max(acc["gt"], 1e-9), grad_corr=acc["gpt"] / max((acc["gpp"] * acc["gtt"]) ** 0.5, 1e-9), n_grad=int(acc["gn"]))

    @torch.no_grad()
    def deq_iteration_curve(idx):
        """Hidden-pixel MAE (ln CHL) on the days idx after k plain iterations and after k Anderson iterations, k in KS."""
        model.eval()
        KS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
        out_ = {}
        for solver in ("fixed", "anderson"):
            err = {k: 0.0 for k in KS}; cnt = 0.0
            for b in range(0, len(idx), a.eval_batch):
                bi = T(idx[b:b + a.eval_batch]); hid = HIDg[bi]
                with torch.autocast("cuda", enabled=not a.no_amp):
                    x = make_inputs(bi, hid); z0, proj = deq_state(bi, hid)
                    zk = DQ.iteration_curve(model, x, z0, proj, KS, solver=solver)
                for k in KS:
                    err[k] += (((zk[k][:, 1].float() * chl_sd + chl_mu) - Yg[bi]).abs() * hid).sum().item()
                cnt += hid.sum().item()
            out_[solver] = {str(k): err[k] / max(cnt, 1) for k in KS}
        out_["n_days"] = int(len(idx)); out_["n_hidden"] = int(cnt)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, solver in enumerate(("fixed", "anderson")):
            ax.plot(KS, [out_[solver][str(k)] for k in KS], "o-", color=OKABE_ITO[i], label=f"{solver} iteration")
        ax.set(xlabel="solver iterations at inference", ylabel="hidden-pixel MAE (ln CHL)", xscale="symlog",
               title=f"{a.run}: val hidden-pixel MAE vs solver iterations ({len(idx)} val days)")
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "figures" / "iteration_curve.png", dpi=110); plt.close(fig)
        return out_

    # fixed val day for the diagnostic figure
    hv = hid_va.reshape(len(va), -1).sum(1)
    fig_day = va[np.argsort(hv)[int(0.75 * len(va))]]

    @torch.no_grad()
    def diagnostic_figure(ep):
        model.eval()
        bi = T(np.array([fig_day])); hid = HIDg[bi]
        with torch.autocast("cuda", enabled=not a.no_amp):
            pz = predict_z(model, bi, hid)
        pz = pz.float()
        r, rb, valid = transport_residual(pz, bi, hid)
        src = source_term(bi)[0].cpu().numpy(); src[~ocean] = np.nan
        pred = (pz[0, 1] * chl_sd + chl_mu).cpu().numpy(); pred[~ocean] = np.nan
        truth = y[fig_day].copy()
        vis = (obs[fig_day] & ~hidden[fig_day])
        inp = np.where(vis, truth, np.nan)
        vv = valid[0].cpu().numpy()
        rr = r[0].cpu().numpy(); rr[~vv] = np.nan
        rbb = rb[0].cpu().numpy(); rbb[~vv] = np.nan
        err = np.where(hidden[fig_day], pred - truth, np.nan)
        cloud = vv & ~obs[fig_day]; clear = vv & obs[fig_day]
        date = str(TIMES[fig_day])[:10]
        ext = [lon[0], lon[-1], lat[-1], lat[0]] if lat_desc else [lon[0], lon[-1], lat[0], lat[-1]]
        orig = "upper" if lat_desc else "lower"
        fig, ax = plt.subplots(2, 4, figsize=(20, 8.2))
        vmin, vmax = np.nanpercentile(truth, 1), np.nanpercentile(truth, 99)
        for k, (ttl, img) in enumerate([("Input ln CHL (visible pixels)", inp), ("Predicted ln CHL, day t", pred), ("True ln CHL (observed pixels)", truth)]):
            im = ax[0, k].imshow(img, extent=ext, cmap=MAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto", origin=orig)
            ax[0, k].set(title=ttl, xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[0, k], label="ln CHL", shrink=.85)
        lim = np.nanpercentile(np.abs(err), 99) if np.isfinite(err).any() else 1
        im = ax[0, 3].imshow(err, extent=ext, cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto", origin=orig)
        ax[0, 3].set(title="Error on fake-cloud hidden pixels", xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[0, 3], label="pred − true, ln CHL", shrink=.85)
        lim = np.nanpercentile(np.abs(rbb), 99) if np.isfinite(rbb).any() else 1
        for k, (ttl, img) in enumerate([("Transport-only residual (tendency + adv − diff), day t", rbb), ("Prior source term S, day t", src), ("Full residual r = transport − S, day t", rr)]):
            im = ax[1, k].imshow(img, extent=ext, cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto", origin=orig)
            ax[1, k].set(title=ttl, xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[1, k], label="ln CHL / day", shrink=.85)
        if a.phys_weight == "courant":
            ax[1, 1].contour(np.linspace(lon[0], lon[-1], W), np.linspace(lat[0], lat[-1], H), wco, levels=[0.5, 0.99], colors="k", linewidths=0.6)
        bins = np.linspace(-1.0, 1.0, 61)
        ax[1, 3].hist(rr[clear], bins=bins, density=True, histtype="step", color=OKABE_ITO[0], lw=1.5, label=f"r, observed pixels (n={int(clear.sum())})")
        ax[1, 3].hist(rr[cloud], bins=bins, density=True, histtype="step", color=OKABE_ITO[1], lw=1.5, label=f"r, under real clouds (n={int(cloud.sum())})")
        ax[1, 3].hist(rbb[cloud], bins=bins, density=True, histtype="step", color=OKABE_ITO[2], lw=1.2, ls="--", label="transport-only, under real clouds")
        ax[1, 3].set(title="Distribution of residuals, day t", xlabel="ln CHL / day", ylabel="density"); ax[1, 3].legend(fontsize=8); ax[1, 3].grid(alpha=.3)
        extra = f", alpha {src_ekman.alpha().item():.3g} kappa [{', '.join(f'{k:.2f}' for k in src_ekman.kappa().detach().cpu().tolist())}]" if a.src_ekman else ""
        fig.suptitle(f"{a.run}: epoch {ep + 1}, val day {date}, lambda {lam_at(ep):.3g}{extra}")
        fig.tight_layout(); fig.savefig(out / "figures" / f"epoch_{ep + 1:03d}.png", dpi=100); plt.close(fig)

    hist = dict(train=[], train_phys=[], lam=[], val=[], val_rmse=[], val_grad_ratio=[], val_grad_corr=[], val_phys_cloud=[], val_phys_obs=[], val_physbase_cloud=[], val_physbase_obs=[],
                val_physother_cloud=[], val_physother_obs=[], alpha=[], kappa=[], lr=[], lr_used=[], epoch_s=[],
                deq_train_iters=[], deq_train_res=[], deq_val_iters=[], deq_val_res=[], deq_val_res_max=[], deq_eta=[])
    if recipe:
        hist.update(val_up_mse=[], val_up_mae=[])
    if a.stoch:
        hist.update(val_crps=[])
    va_s, te_s, hid_va_s, hid_te_s = va[::a.val_stride], te[::a.val_stride], hid_va[::a.val_stride], hid_te[::a.val_stride]
    torch.cuda.reset_peak_memory_stats()
    best, best_ep, best_state, best_src = np.inf, -1, None, None
    # ---- --return: resume state (see RESUME_IGNORED and the docstring)
    best_any = np.inf                                          # best val MAE since epoch 1 (resume.pt trigger, independent of --ckpt-from)
    resume_info = dict(events=[], legacy_resume=False, legacy_epoch=None)
    source_info = None                                         # --continue-from: source run, its best epoch and score
    start_ep = 0
    if a.eval_only:
        from .runs import find_run
        src = find_run(a.init_from)
        ck = torch.load(src / "model.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if a.src_ekman and ck.get("src_ekman") is not None:
            src_ekman.load_state_dict(ck["src_ekman"])
        start_ep = a.epochs                                     # skip the training loop
        resume_info["events"].append(dict(epoch=0, kind=f"eval-only from {src}", time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"eval-only: weights of {src / 'model.pt'}", flush=True)
    elif a.continue_from and a.continue_weights_only and not (a.resume_best and (out / "resume.pt").exists()):
        from .runs import find_run
        src = find_run(a.continue_from)
        ck = torch.load(src / "model.pt", map_location=dev, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        bad_missing = [k for k in missing if not (a.stoch and STOCH.is_noise_param(k))]
        if bad_missing or unexpected:
            raise SystemExit(f"--continue-weights-only: state mismatch, missing {bad_missing} unexpected {list(unexpected)}")
        if a.src_ekman and ck.get("src_ekman") is not None:
            src_ekman.load_state_dict(ck["src_ekman"])
        scfg = json.loads((src / "config.json").read_text())
        source_info = dict(run=a.continue_from, path=str(src), best_epoch=int(scfg.get("best_epoch", -1)), best_score=float(scfg.get("best_val_mae", np.nan)),
                           score="val hidden-pixel MAE", weights_only=True, n_new_params=len(missing))
        resume_info["events"].append(dict(epoch=0, kind=f"continue-from {src} model.pt (weights only)", time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"continue-from (weights only): {src / 'model.pt'} (source best epoch {source_info['best_epoch']}, val {source_info['best_score']:.6f}; "
              f"{len(missing)} new zero-init params); fresh AdamW, {a.sched if not a.lr_replay else 'replayed'} schedule from lr {a.lr:.3g} over {a.epochs} epochs", flush=True)
    elif a.continue_from and not (a.resume_best and (out / "resume.pt").exists()):
        from .runs import find_run
        src = find_run(a.continue_from)
        rs = torch.load(src / "resume.pt", map_location=dev, weights_only=False)
        model.load_state_dict(rs["model"]); opt.load_state_dict(rs["opt"]); scaler.load_state_dict(rs["scaler"])
        for g in opt.param_groups:                             # restart the schedule at --lr (the stored groups carry the old run's end lr)
            g["lr"] = g["initial_lr"] = a.lr
        sched = make_sched()                                   # fresh schedule (cosine / plateau) from --lr
        if a.src_ekman and rs.get("src_ekman") is not None:
            src_ekman.load_state_dict(rs["src_ekman"])
        source_info = dict(run=a.continue_from, path=str(src), best_epoch=int(rs["epoch"]) + 1, best_score=float(rs["best_any"]),
                           score=("val all-pixel z MSE" if recipe else "val hidden-pixel MAE"), source_lr_end=float(rs["opt"]["param_groups"][0]["lr"]))
        resume_info["events"].append(dict(epoch=0, kind=f"continue-from {src} epoch {rs['epoch'] + 1} val {rs['best_any']:.6f}", time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"continue-from: {src / 'resume.pt'} (epoch {rs['epoch'] + 1}, val {rs['best_any']:.6f}); fresh {a.sched} schedule from lr {a.lr:.3g} over {a.epochs} epochs", flush=True)
    elif a.resume_best and (out / "resume.pt").exists():
        rs = torch.load(out / "resume.pt", map_location=dev, weights_only=False)
        old, new = rs["args"], vars(a)
        old = {**RESUME_NEW_DEFAULTS, **old}
        diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if k not in RESUME_IGNORED and old.get(k) != new.get(k)}
        if diff:
            raise SystemExit(f"--return: {out / 'resume.pt'} was written with different arguments {diff}; refusing to mix configs "
                             f"(use --no-return to start this run from scratch, or delete the run folder)")
        model.load_state_dict(rs["model"]); opt.load_state_dict(rs["opt"]); scaler.load_state_dict(rs["scaler"])
        if sched is not None:
            sched.load_state_dict(rs["sched"])
        if a.src_ekman:
            src_ekman.load_state_dict(rs["src_ekman"])
        start_ep = rs["epoch"] + 1; best_any = rs["best_any"]; best, best_ep = rs["best"], rs["best_ep"]
        if best_ep >= 0:                                       # the ckpt-from best is this same state (or none yet)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_src = {k: v.detach().clone() for k, v in src_ekman.state_dict().items()} if a.src_ekman else None
        for k in hist:
            hist[k] = rs["hist"].get(k, [])[:start_ep]
        source_info = rs.get("source_info")
        rng.bit_generator.state = rs["np_rng"]; torch.set_rng_state(rs["torch_rng"].cpu()); torch.cuda.set_rng_state(rs["cuda_rng"].cpu())
        resume_info = rs["resume_info"]
        resume_info["events"].append(dict(epoch=start_ep, kind="resume", time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"returning to best checkpoint at epoch {start_ep} (val {best_any:.4f})", flush=True)
    elif a.resume_best and (out / "model.pt").exists() and (out / "history.json").exists():
        ck = torch.load(out / "model.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if a.src_ekman and ck.get("src_ekman") is not None:
            src_ekman.load_state_dict(ck["src_ekman"])
        start_ep = int(ck["epoch"])
        h0 = json.loads((out / "history.json").read_text())
        for k in hist:
            hist[k] = h0.get(k, [])[:start_ep]
        if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
            sched.last_epoch = start_ep; [g.__setitem__("lr", lr_) for g, lr_ in zip(opt.param_groups, sched._get_closed_form_lr())]
        best = best_any = float(ck["val_mae"]); best_ep = start_ep - 1
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_src = {k: v.detach().clone() for k, v in src_ekman.state_dict().items()} if a.src_ekman else None
        resume_info.update(legacy_resume=True, legacy_epoch=start_ep)
        resume_info["events"].append(dict(epoch=start_ep, kind="legacy", time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print("!" * 100 + f"\nLEGACY RESUME: optimizer state reset at epoch {start_ep} (model.pt weights, val {best:.4f}; fresh AdamW moments, "
              f"GradScaler and RNG streams; lr schedule set to epoch {start_ep}: lr {opt.param_groups[0]['lr']:.3g}; arguments not checked)\n" + "!" * 100, flush=True)
    if a.lr_replay:                                            # per-epoch lr taken from another run's history (lr_used)
        from .runs import find_run
        LR_REPLAY = json.loads((find_run(a.lr_replay) / "history.json").read_text())["lr_used"]
        sched = None
        print(f"lr-replay from {a.lr_replay}: {len(LR_REPLAY)} epochs, lr {LR_REPLAY[0]:.3g} .. {LR_REPLAY[-1]:.3g}", flush=True)
    for ep in range(start_ep, a.epochs):
        t0 = time.time()
        if a.lr_replay:
            if ep >= len(LR_REPLAY):
                print(f"lr-replay: {a.lr_replay} has only {len(LR_REPLAY)} epochs; stopping"); break
            for g in opt.param_groups:
                g["lr"] = LR_REPLAY[ep]
        lr_used = opt.param_groups[0]["lr"]                     # lr of this epoch's updates ("lr" below: after the scheduler step)
        ltr, lph, lam = train_epoch(ep)
        restore_val_masks()
        ev_ = residual_stats(va_s)
        sel = ev_["up_mse"] if recipe else (ev_["crps"] if a.stoch else ev_["mae"])   # model-selection score (recipe: their val loss)
        if a.no_select:                                         # every epoch counts as the best: model.pt = last epoch
            sel = -float(ep + 1)
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(sel)
        elif sched is not None:
            sched.step()
        al = float(src_ekman.alpha().detach()) if a.src_ekman else None
        ka = src_ekman.kappa().detach().cpu().numpy().tolist() if a.src_ekman else None
        for k, v in dict(train=ltr, train_phys=lph, lam=lam, val=ev_["mae"], val_rmse=ev_["rmse"], val_grad_ratio=ev_["grad_ratio"], val_grad_corr=ev_["grad_corr"],
                         val_phys_cloud=ev_["abs_r_cloud"], val_phys_obs=ev_["abs_r_obs"],
                         val_physbase_cloud=ev_["abs_rbase_cloud"], val_physbase_obs=ev_["abs_rbase_obs"],
                         val_physother_cloud=ev_["abs_rbase_other_cloud"], val_physother_obs=ev_["abs_rbase_other_obs"], alpha=al, kappa=ka,
                         lr=opt.param_groups[0]["lr"], lr_used=lr_used, epoch_s=time.time() - t0,
                         deq_train_iters=TRAIN_DEQ["iters"], deq_train_res=TRAIN_DEQ["res"], deq_val_iters=EVAL_DEQ["iters"], deq_val_res=EVAL_DEQ["res"],
                         deq_val_res_max=EVAL_DEQ["res_max"], deq_eta=(float(model.cell.eta().detach()) if a.model == "deq" else None)).items():
            hist[k].append(v)
        if recipe:
            hist["val_up_mse"].append(ev_["up_mse"]); hist["val_up_mae"].append(ev_["up_mae"])
        if a.stoch:
            hist["val_crps"].append(ev_["crps"])
        if ep + 1 > a.ckpt_from and sel < best:
            best, best_ep = sel, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_src = {k: v.detach().clone() for k, v in src_ekman.state_dict().items()} if a.src_ekman else None
            torch.save(dict(model=best_state, src_ekman=best_src, epoch=ep + 1, val_mae=best), out / "model.pt")
        print(f"epoch {ep + 1:3d}  data {ltr:.4f}  phys {lph:.4f} (lam {lam:.3g})  val_mae {ev_['mae']:.4f}  "
              f"grad ratio {ev_['grad_ratio']:.3f} corr {ev_['grad_corr']:.3f}  val |r| cloud {ev_['abs_r_cloud']:.4f} obs {ev_['abs_r_obs']:.4f}  base cloud {ev_['abs_rbase_cloud']:.4f}  {ev_['other_form'][:4]} cloud {ev_['abs_rbase_other_cloud']:.4f} obs {ev_['abs_rbase_other_obs']:.4f}  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
              + (f"  alpha {al:.3g} kappa [{', '.join(f'{k:.2f}' for k in ka)}]" if a.src_ekman else "")
              + (f"  deq train it {TRAIN_DEQ['iters']:.1f} res {TRAIN_DEQ['res']:.2e} | val it {EVAL_DEQ['iters']:.1f} res {EVAL_DEQ['res']:.2e} max {EVAL_DEQ['res_max']:.2e} eta {float(model.cell.eta().detach()):.3f}" if a.model == "deq" else "")
              + (f"  val all-px z mse {ev_['up_mse']:.5f} mae {ev_['up_mae']:.4f} (n {ev_['up_n']})" if recipe else "")
              + (f"  val crps {ev_['crps']:.4f}" if a.stoch else "")
              + f"  lr {lr_used:.3g}" + (f" -> {opt.param_groups[0]['lr']:.3g}" if opt.param_groups[0]["lr"] != lr_used else "")
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        (out / "history.json").write_text(json.dumps(dict(hist, source=source_info) if source_info else hist))
        if sel < best_any:                                    # full training state at the best epoch so far (--return)
            best_any = sel
            rs = dict(model=model.state_dict(), opt=opt.state_dict(), sched=(sched.state_dict() if sched is not None else None),
                      scaler=scaler.state_dict(), src_ekman=(src_ekman.state_dict() if a.src_ekman else None), epoch=ep,
                      best_any=best_any, best=best, best_ep=best_ep, hist={k: list(v) for k, v in hist.items()},
                      np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                      args=dict(vars(a)), resume_info=resume_info, source_info=source_info)
            torch.save(rs, out / "resume.pt.tmp"); os.replace(out / "resume.pt.tmp", out / "resume.pt")
        if a.fig_every > 0 and ((ep + 1) % a.fig_every == 0 or ep == 0 or ep + 1 == a.epochs):
            diagnostic_figure(ep)
        if best_ep >= 0 and ep - best_ep >= a.patience:
            print(f"early stop; best epoch {best_ep + 1} val {best:.5f}"); break
        if a.stop_epoch and ep + 1 >= a.stop_epoch:
            print(f"stop-epoch {a.stop_epoch} reached (schedule spans {a.epochs} epochs); best epoch {best_ep + 1} val {best:.4f}"); break
    if a.eval_only:
        best, best_ep = np.nan, -1
    elif best_state is None:                                   # fewer epochs than --ckpt-from: keep the final weights
        best, best_ep = hist["val"][-1], len(hist["val"]) - 1
    else:
        model.load_state_dict(best_state)
        if a.src_ekman:
            src_ekman.load_state_dict(best_src)
    if a.no_select and best_ep >= 0:                           # report the selected (= last) epoch's val MAE, not the -epoch sentinel
        best = hist["val"][best_ep]
    restore_val_masks()

    ev = np.r_[va_s, te_s]
    preds = np.full((len(ev), H, W), np.nan, dtype="float32")
    samples = np.full((a.stoch_final_samples, len(ev), H, W), np.nan, dtype="float16") if a.stoch else None
    rstats = dict(val=residual_stats(va_s, preds[:len(va_s)], None if samples is None else samples[:, :len(va_s)]),
                  test=residual_stats(te_s, preds[len(va_s):], None if samples is None else samples[:, len(va_s):]))
    if a.leak_day is not None and a.memory:                    # leak test: the memory channels (and anomaly) of target t
        mc, an, cn = memory(T(np.array([a.leak_day])), want_anom=True)
        np.savez(out / "leak_memory.npz", channels=mc[0].cpu().numpy(), anom=an[0].cpu().numpy(), cnt30=cn[0].cpu().numpy(),
                 hidden=hidden[a.leak_day], day=a.leak_day)
    if samples is not None:                                    # --stoch: every sample of day t, ln CHL, NaN on land
        samples[:, :, ~ocean] = np.nan
        np.savez_compressed(out / "samples.npz", samples=samples, idx=ev, val_idx=va_s, test_idx=te_s, seed=12345)
        del samples
    preds_raw = preds.copy() if a.store == "io_up" else None     # io_up: raw output on every pixel, float32 (analyses/upstream)
    preds[:, ~ocean] = np.nan
    pv, pt = preds[:len(va_s)], preds[len(va_s):]
    clim_ref = climatology_reference(y, tr)
    scores = {}
    for name, idx, pr, hid in [("val", va_s, pv, hid_va_s), ("test", te_s, pt, hid_te_s)]:
        scores[name] = dict(cnn_hidden=metrics(y[idx], pr, hid), cnn_observed=metrics(y[idx], pr, obs[idx]),
                            climatology_hidden=metrics(y[idx], clim_ref[idx], hid), physics=rstats[name])
        if recipe:
            scores[name]["up_allpx"] = dict(z_mse=rstats[name]["up_mse"], z_mae=rstats[name]["up_mae"], n_px=rstats[name]["up_n"],
                                            zero_pred_z_mse=recipe_info["zero_pred_allpx"][f"{name}_mse"], zero_pred_z_mae=recipe_info["zero_pred_allpx"][f"{name}_mae"],
                                            n_days=len(idx), last_batch_size=len(idx) % 4 or 4)
        if a.clim:   # day-of-year climatology (8-day bins from training days), the reference used in analyses/FINDINGS.md
            scores[name]["doy_climatology_hidden"] = metrics(y[idx], clim_np[doy_bin[idx]], hid)
    if a.model == "deq":
        scores["val"]["iteration_curve"] = deq_iteration_curve(va_s[:a.deq_curve_days] if a.deq_curve_days else va_s)
    m = scores["val"]
    print(f"VAL hidden-pixel MAE {m['cnn_hidden']['mae']:.4f}  |r| cloud {rstats['val']['abs_r_cloud']:.4f} obs {rstats['val']['abs_r_obs']:.4f}"
          f"  base cloud {rstats['val']['abs_rbase_cloud']:.4f}")

    cfg = dict(vars(a), dilations=(dilations if dilations is not None else [1, 2, 4, 8, 16, 1, 2, 4]) if (a.model == "dilated" or (a.model == "temporal3d" and a.t3d_backbone == "dilated")) else a.dilations,
               resume_events=resume_info["events"], legacy_resume=resume_info["legacy_resume"], legacy_resume_epoch=resume_info["legacy_epoch"],
               in_channels=in_ch, n_params=count_params(model), t3d=(dict(c_day=c_day, n_days=n_days, n_static=n_static) if a.model == "temporal3d" else None), best_epoch=best_ep + 1, best_val_mae=best,
               n_days=dict(train=len(tr), val=len(va), test=len(te)), feature_stats=feat_stats, priors=prior_info,
               ekman_alpha=(float(src_ekman.alpha()) if a.src_ekman else None),
               ekman_kappa=(src_ekman.kappa().detach().cpu().numpy().tolist() if a.src_ekman else None),
               chl_norm=dict(mean=chl_mu, std=chl_sd), wall_s=time.time() - t_start, fig_day=str(TIMES[fig_day])[:10],
               split=dict(val_years=list(val_years), test_years=list(test_years), val_stride=a.val_stride, n_val_scored=len(va_s), n_test_scored=len(te_s)),
               recipe_info=(recipe_info if recipe else None), stopped_epoch=len(hist["val"]), continue_from_source=source_info,
               final_lr=opt.param_groups[0]["lr"], n_lr_drops=(int(sum(lr_ < lu for lu, lr_ in zip(hist["lr_used"], hist["lr"]))) if a.sched == "plateau" else None),
               grid=[H, W], n_ocean=int(ocean.sum()), lat_ascending=not lat_desc, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_gpu_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
    if a.ablate:
        cfg["ablate_info"] = ABL_INFO
    if a.rw_loss:                                              # io8: weights and distributions (also in rw_loss.json)
        cfg["rw_loss_info"] = RW_INFO
    if a.bsmem:                                                # blk8: memory inputs report and summary stats
        cfg["bsmem_info"] = dict(report=PMT.report, stats=PMT.stats)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "history.json").write_text(json.dumps(dict(hist, source=source_info) if source_info else hist))
    (out / "metrics.json").write_text(json.dumps(dict(
        scores=scores, per_day_test=per_day_metrics(y[te_s], pt, hid_te_s), per_day_val=per_day_metrics(y[va_s], pv, hid_va_s),
        obs_frac=dict(val=obs_frac[va_s].tolist(), test=obs_frac[te_s].tolist())), indent=1))
    torch.save(dict(model=model.state_dict(), src_ekman=(src_ekman.state_dict() if a.src_ekman else None)), out / "model.pt")
    if preds_raw is not None:
        np.savez_compressed(out / "preds.npz", pred=preds_raw, idx=ev, hidden=hidden[ev], val_idx=va_s, test_idx=te_s, time=TIMES[ev])
    else:
        np.savez_compressed(out / "preds.npz", pred=(preds if a.preds_f32 else preds.astype("float16")), idx=ev, hidden=hidden[ev], val_idx=va_s, test_idx=te_s)
    print("wrote", out, f"({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
