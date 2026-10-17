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
from .evaluate import metrics, per_day_metrics, climatology_reference
from .models import build_model, count_params, Temporal3DNet
from .physics import grid_metrics
from .plotting import OKABE_ITO, MAP_CMAP, ERR_CMAP

from .runs import run_dir
DT = 86400.0
H0 = 30.0   # reference mixed-layer depth (m) for the Ekman source


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--store", choices=["io", "blk", "blk4"], default="io", help="io: Indian Ocean 0.25 deg store (store.py); blk: Black Sea 2 km store (store_blk.py); blk4: Black Sea 4 km store (store_blk4.py)")
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
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--stop-epoch", type=int, default=0, help="stop training after this many epochs while the lr schedule still spans --epochs (0 = off). For quick architecture comparisons against an existing --epochs run at the same epoch, without retraining that run.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--sched", choices=["none", "cosine"], default="cosine")
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-channels-last", action="store_true", help="keep NCHW tensors (channels_last is ~35%% faster for the dilated CNN on A10G)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--fig-every", type=int, default=5)
    p.add_argument("--ckpt-from", type=int, default=0, help="track/checkpoint the best val model only after this epoch (model.pt written on every improvement)")
    p.add_argument("--return", dest="resume_best", action=argparse.BooleanOptionalAction, default=True,
                   help="return to the best checkpointed training state (resume.pt) if the run was interrupted; --no-return starts fresh")
    a = p.parse_args(argv)
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
    if a.lag_max is None:
        a.lag_max = {"blk": 20.0, "blk4": 10.0}.get(a.store, 5.0)
    return a


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
RESUME_NEW_DEFAULTS = {"donor_exclude": 0, "eval_only": False, "init_from": None, "dilations": None, "continue_from": None}


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = run_dir(a.run, "pinn", a.model)
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
    tr, va, te = D.split_by_years(TIMES, val_years, test_years)
    y = D.log_chl(rd("chl_l3"))
    obs = np.isfinite(y) & ocean
    obs_frac = obs.reshape(n, -1).sum(1) / ocean.sum()
    chl_mu, chl_sd = float(np.nanmean(y[tr])), float(np.nanstd(y[tr]))
    print(f"store {a.store}: {n} days  train {len(tr)} val {len(va)} test {len(te)}  grid {H} x {W}  ocean px {int(ocean.sum())}  "
          f"val years {val_years} test years {test_years}", flush=True)

    hid_va, don_va = C.fixed_hidden(obs, va, seed=a.eval_seed, times=TIMES, exclude=a.donor_exclude)
    hid_te, don_te = C.fixed_hidden(obs, te, seed=a.eval_seed + 1, times=TIMES, exclude=a.donor_exclude)
    if a.donor_exclude > 0:
        dd = lambda i, j: np.abs((TIMES[j] - TIMES[i]).astype("timedelta64[D]").astype(int))
        print(f"donor-exclude {a.donor_exclude}: min |donor - day| val {dd(va, don_va).min()} d, test {dd(te, don_te).min()} d", flush=True)
    hidden = np.zeros_like(obs); hidden[va], hidden[te] = hid_va, hid_te
    donor = np.zeros(n, dtype=np.int64); donor[va], donor[te] = don_va, don_te

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
    chl_z = lambda idx: (Yg[idx] - chl_mu) / chl_sd * OBSg[idx]      # z-scored observed ln CHL (0 where unobserved), per batch
    OCEANg = T(ocean)

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
    NB = {k: T(T2.neighbour_index(TIMES, k)) for k in sorted(set(offsets) | set(win_offs) | {-1, 1}) if k != 0}
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
    n_chl = 0 if a.no_chl_input else 1                       # CHL value channels on context days
    n_mask = 1 if (a.cloud_input or not a.no_chl_input) else 0
    n_chl_t = 0 if (a.no_chl_input or a.no_chl_today) else 1   # CHL value channel on day t
    n_mask_t = 1 if (a.cloud_input or n_chl_t) else 0
    foffs = [k for k in range(-a.feat_context, a.feat_context + 1) if k != 0]
    NBF = {k: T(T2.neighbour_index(TIMES, k)) for k in foffs}
    in_ch = len(a.features) * (1 + len(foffs)) + (n_chl_t + n_mask_t) + (n_chl + n_mask) * len(offsets) + (1 if a.clim else 0) + (5 if a.coords else 0) + (1 if a.year_input else 0)
    if a.model == "temporal3d":   # per-day channels [F features, CHL value, visible mask] x T days; static 2D channels after the stem
        c_day, n_days = len(a.features) + 2, len(win_offs)
        n_static = (1 if a.clim else 0) + (5 if a.coords else 0) + (1 if a.year_input else 0)
        in_ch = a.t3d_width + n_static                          # backbone input channels
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

    def nb_masks(bi, k):
        """Neighbour day index, its validity, and the mask of its pixels visible in the input."""
        j = NB[k][bi]; ok = (j >= 0).view(-1, 1, 1); j = j.clamp(min=0)
        m = OBSg[j] & ok
        if a.context_clouds == "persist":
            jd = NB[k][DONg[bi]]; jd = torch.where(jd >= 0, jd, DONg[bi])
            m = m & OBSg[jd]
        return j, ok, m

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
                days.append(torch.cat([feats(jf) * okf, (chl_z(j) * m).unsqueeze(1), m.float().unsqueeze(1)], 1))
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
        if a.model == "temporal3d":
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
        if a.residual:
            jm, _, _ = nb_masks(bi, -1); jp, _, _ = nb_masks(bi, 1)
            z = z + torch.stack([CLIMg[BINg[jm]], CLIMg[BINg[bi]], CLIMg[BINg[jp]]], 1)
        return z

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

    def data_loss(pz, bi, hid):
        """Weighted hidden-pixel loss over the three output days (z units), plus the gradient-difference term."""
        w = OBSg[bi].float() if no_chl_t else hid.float() + a.vis_weight * (OBSg[bi] & ~hid).float()
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
    elif a.model == "temporal3d":
        model = Temporal3DNet(c_day, n_static, n_days, backbone=a.t3d_backbone, stem_width=a.t3d_width, stem_depth=a.t3d_depth, kt=a.t3d_kernel_t,
                              width=a.width, dropout=a.dropout, out_channels=3, depth=a.unet_depth,
                              spectral=a.spectral, modes=a.spectral_modes, spec_ch=a.spectral_ch, dilations=dilations).to(dev)
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
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs) if a.sched == "cosine" else None
    scaler = torch.amp.GradScaler(enabled=not a.no_amp)
    print(f"model {a.model} in_ch {in_ch} params {count_params(model)}  residual pixels/day (stencil) {int(st.sum())}", flush=True)

    TRAIN_DEQ = dict(iters=0.0, res=0.0); EVAL_DEQ = dict(iters=0.0, res=0.0, res_max=0.0)

    def lam_at(ep):
        return a.lam * min(1.0, (ep + 1) / max(a.phys_warmup, 1))

    def train_epoch(ep):
        model.train()
        if a.donor_exclude > 0:
            donors = tr[C.seasonal_donor_indices(TIMES[tr], rng, a.donor_window, exclude=a.donor_exclude)] if a.donor_window > 0 \
                else tr[C.donor_indices(len(tr), rng, times=TIMES[tr], exclude=a.donor_exclude)]
        else:
            donors = tr[C.seasonal_donor_indices(TIMES[tr], rng, a.donor_window)] if a.donor_window > 0 else tr[C.donor_indices(len(tr), rng)]
        HIDg[T(tr)] = T(obs[tr] & ~obs[donors]); DONg[T(tr)] = T(donors)
        lam = lam_at(ep)
        td = tp = 0.0; cnt = 0; dq_it = dq_res = 0.0; dq_n = 0
        order = T(rng.permutation(tr))
        for nb_, b in enumerate(range(0, len(tr), a.batch)):
            if a.max_batches and nb_ >= a.max_batches:
                break
            bi = order[b:b + a.batch]; hid = HIDg[bi]
            with torch.autocast("cuda", enabled=not a.no_amp):
                pz = predict_z(model, bi, hid)
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
    def residual_stats(idx, collect_preds=None):
        """Hidden-pixel MAE (ln CHL) on day t; unweighted mean|r| and mean|r_base| under real clouds / observed."""
        model.eval()
        acc = dict(err=0.0, sq=0.0, cnt=0.0, rc=0.0, nc=0.0, ro=0.0, no=0.0, bc=0.0, bo=0.0, gp=0.0, gt=0.0, gpt=0.0, gpp=0.0, gtt=0.0, gn=0.0)
        other = "semi-lagrangian" if a.phys == "upwind" else "upwind"      # the residual form not used for training, scored too
        acc2 = dict(rc=0.0, nc=0.0, ro=0.0, no=0.0)
        dq = dict(iters=0.0, res=0.0, res_max=0.0, n=0)
        for b in range(0, len(idx), a.eval_batch):
            bi = T(idx[b:b + a.eval_batch]); hid = HIDg[bi]
            with torch.autocast("cuda", enabled=not a.no_amp):
                pz = predict_z(model, bi, hid)
            pz = pz.float()
            if a.model == "deq":
                dq["iters"] += DQ.LAST_STATS["iters"]; dq["res"] += DQ.LAST_STATS["res"]; dq["res_max"] = max(dq["res_max"], DQ.LAST_STATS["res_max"]); dq["n"] += 1
            if collect_preds is not None:
                collect_preds[b:b + len(bi)] = (pz[:, 1] * chl_sd + chl_mu).cpu().numpy()
            e_ = (pz[:, 1] * chl_sd + chl_mu - Yg[bi]) * hid
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
        return dict(mae=acc["err"] / max(acc["cnt"], 1), rmse=(acc["sq"] / max(acc["cnt"], 1)) ** 0.5, n_hidden=int(acc["cnt"]), phys_form=a.phys,
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
                val_physother_cloud=[], val_physother_obs=[], alpha=[], kappa=[], lr=[], epoch_s=[],
                deq_train_iters=[], deq_train_res=[], deq_val_iters=[], deq_val_res=[], deq_val_res_max=[], deq_eta=[])
    va_s, te_s, hid_va_s, hid_te_s = va[::a.val_stride], te[::a.val_stride], hid_va[::a.val_stride], hid_te[::a.val_stride]
    torch.cuda.reset_peak_memory_stats()
    best, best_ep, best_state, best_src = np.inf, -1, None, None
    # ---- --return: resume state (see RESUME_IGNORED and the docstring)
    best_any = np.inf                                          # best val MAE since epoch 1 (resume.pt trigger, independent of --ckpt-from)
    resume_info = dict(events=[], legacy_resume=False, legacy_epoch=None)
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
    elif a.continue_from and not (a.resume_best and (out / "resume.pt").exists()):
        from .runs import find_run
        src = find_run(a.continue_from)
        rs = torch.load(src / "resume.pt", map_location=dev, weights_only=False)
        model.load_state_dict(rs["model"]); opt.load_state_dict(rs["opt"]); scaler.load_state_dict(rs["scaler"])
        for g in opt.param_groups:                             # restart the schedule at --lr (the stored groups carry the old run's end lr)
            g["lr"] = g["initial_lr"] = a.lr
        if sched is not None:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        if a.src_ekman and rs.get("src_ekman") is not None:
            src_ekman.load_state_dict(rs["src_ekman"])
        resume_info["events"].append(dict(epoch=0, kind=f"continue-from {src} epoch {rs['epoch'] + 1} val {rs['best_any']:.4f}", time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"continue-from: {src / 'resume.pt'} (epoch {rs['epoch'] + 1}, val {rs['best_any']:.4f}); fresh schedule from lr {a.lr:.3g} over {a.epochs} epochs", flush=True)
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
            hist[k] = rs["hist"][k][:start_ep]
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
        if sched is not None:
            sched.last_epoch = start_ep; [g.__setitem__("lr", lr_) for g, lr_ in zip(opt.param_groups, sched._get_closed_form_lr())]
        best = best_any = float(ck["val_mae"]); best_ep = start_ep - 1
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_src = {k: v.detach().clone() for k, v in src_ekman.state_dict().items()} if a.src_ekman else None
        resume_info.update(legacy_resume=True, legacy_epoch=start_ep)
        resume_info["events"].append(dict(epoch=start_ep, kind="legacy", time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print("!" * 100 + f"\nLEGACY RESUME: optimizer state reset at epoch {start_ep} (model.pt weights, val {best:.4f}; fresh AdamW moments, "
              f"GradScaler and RNG streams; lr schedule set to epoch {start_ep}: lr {opt.param_groups[0]['lr']:.3g}; arguments not checked)\n" + "!" * 100, flush=True)
    for ep in range(start_ep, a.epochs):
        t0 = time.time()
        ltr, lph, lam = train_epoch(ep)
        ev_ = residual_stats(va_s)
        if sched is not None:
            sched.step()
        al = float(src_ekman.alpha().detach()) if a.src_ekman else None
        ka = src_ekman.kappa().detach().cpu().numpy().tolist() if a.src_ekman else None
        for k, v in dict(train=ltr, train_phys=lph, lam=lam, val=ev_["mae"], val_rmse=ev_["rmse"], val_grad_ratio=ev_["grad_ratio"], val_grad_corr=ev_["grad_corr"],
                         val_phys_cloud=ev_["abs_r_cloud"], val_phys_obs=ev_["abs_r_obs"],
                         val_physbase_cloud=ev_["abs_rbase_cloud"], val_physbase_obs=ev_["abs_rbase_obs"],
                         val_physother_cloud=ev_["abs_rbase_other_cloud"], val_physother_obs=ev_["abs_rbase_other_obs"], alpha=al, kappa=ka,
                         lr=opt.param_groups[0]["lr"], epoch_s=time.time() - t0,
                         deq_train_iters=TRAIN_DEQ["iters"], deq_train_res=TRAIN_DEQ["res"], deq_val_iters=EVAL_DEQ["iters"], deq_val_res=EVAL_DEQ["res"],
                         deq_val_res_max=EVAL_DEQ["res_max"], deq_eta=(float(model.cell.eta().detach()) if a.model == "deq" else None)).items():
            hist[k].append(v)
        if ep + 1 > a.ckpt_from and ev_["mae"] < best:
            best, best_ep = ev_["mae"], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_src = {k: v.detach().clone() for k, v in src_ekman.state_dict().items()} if a.src_ekman else None
            torch.save(dict(model=best_state, src_ekman=best_src, epoch=ep + 1, val_mae=best), out / "model.pt")
        print(f"epoch {ep + 1:3d}  data {ltr:.4f}  phys {lph:.4f} (lam {lam:.3g})  val_mae {ev_['mae']:.4f}  "
              f"grad ratio {ev_['grad_ratio']:.3f} corr {ev_['grad_corr']:.3f}  val |r| cloud {ev_['abs_r_cloud']:.4f} obs {ev_['abs_r_obs']:.4f}  base cloud {ev_['abs_rbase_cloud']:.4f}  {ev_['other_form'][:4]} cloud {ev_['abs_rbase_other_cloud']:.4f} obs {ev_['abs_rbase_other_obs']:.4f}  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
              + (f"  alpha {al:.3g} kappa [{', '.join(f'{k:.2f}' for k in ka)}]" if a.src_ekman else "")
              + (f"  deq train it {TRAIN_DEQ['iters']:.1f} res {TRAIN_DEQ['res']:.2e} | val it {EVAL_DEQ['iters']:.1f} res {EVAL_DEQ['res']:.2e} max {EVAL_DEQ['res_max']:.2e} eta {float(model.cell.eta().detach()):.3f}" if a.model == "deq" else "")
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        (out / "history.json").write_text(json.dumps(hist))
        if ev_["mae"] < best_any:                             # full training state at the best epoch so far (--return)
            best_any = ev_["mae"]
            rs = dict(model=model.state_dict(), opt=opt.state_dict(), sched=(sched.state_dict() if sched is not None else None),
                      scaler=scaler.state_dict(), src_ekman=(src_ekman.state_dict() if a.src_ekman else None), epoch=ep,
                      best_any=best_any, best=best, best_ep=best_ep, hist={k: list(v) for k, v in hist.items()},
                      np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                      args=dict(vars(a)), resume_info=resume_info)
            torch.save(rs, out / "resume.pt.tmp"); os.replace(out / "resume.pt.tmp", out / "resume.pt")
        if a.fig_every > 0 and ((ep + 1) % a.fig_every == 0 or ep == 0 or ep + 1 == a.epochs):
            diagnostic_figure(ep)
        if best_ep >= 0 and ep - best_ep >= a.patience:
            print(f"early stop; best epoch {best_ep + 1} val {best:.4f}"); break
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

    ev = np.r_[va_s, te_s]
    preds = np.full((len(ev), H, W), np.nan, dtype="float32")
    rstats = dict(val=residual_stats(va_s, preds[:len(va_s)]), test=residual_stats(te_s, preds[len(va_s):]))
    preds[:, ~ocean] = np.nan
    pv, pt = preds[:len(va_s)], preds[len(va_s):]
    clim_ref = climatology_reference(y, tr)
    scores = {}
    for name, idx, pr, hid in [("val", va_s, pv, hid_va_s), ("test", te_s, pt, hid_te_s)]:
        scores[name] = dict(cnn_hidden=metrics(y[idx], pr, hid), cnn_observed=metrics(y[idx], pr, obs[idx]),
                            climatology_hidden=metrics(y[idx], clim_ref[idx], hid), physics=rstats[name])
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
               grid=[H, W], n_ocean=int(ocean.sum()), lat_ascending=not lat_desc, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, peak_gpu_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "history.json").write_text(json.dumps(hist))
    (out / "metrics.json").write_text(json.dumps(dict(
        scores=scores, per_day_test=per_day_metrics(y[te_s], pt, hid_te_s), per_day_val=per_day_metrics(y[va_s], pv, hid_va_s),
        obs_frac=dict(val=obs_frac[va_s].tolist(), test=obs_frac[te_s].tolist())), indent=1))
    torch.save(dict(model=model.state_dict(), src_ekman=(src_ekman.state_dict() if a.src_ekman else None)), out / "model.pt")
    np.savez_compressed(out / "preds.npz", pred=preds.astype("float16"), idx=ev, hidden=hidden[ev], val_idx=va_s, test_idx=te_s)
    print("wrote", out, f"({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
