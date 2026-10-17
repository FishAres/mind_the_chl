"""UDE trainer for daily surface ln CHL (chl_cnn/ude.py): a mechanistic transport step plus a learned local source.

    C_{k+1} = T(C_k; u_{t0+k}, v_{t0+k}) + dt [ K lap(C_k) + g_theta(C_k, X_{t0+k}) ],   dt = 1 day, k = 0..L-1

  T   --transport sl: semi-Lagrangian advection along the day's currents (cells/day = u 86400 / dx from physics.grid_metrics,
      signed dy so v points along increasing row index), --n-sub back-trajectory sub-steps, bilinear sampling; pixels whose
      sampled ocean / current validity is < 0.99 keep their value (fraction reported). --transport none: T = identity.
      --adv-scale fixed: the currents as they are (alpha = 1); scalar: one learned alpha = 1.5 sigmoid(a) in (0, 1.5)
      (init --adv-init); field: per-pixel alpha(x) = 1.5 sigmoid(MLP(pixel stack of g)) (1x1 conv, --adv-hidden), initialised
      at --adv-init everywhere, with --adv-reg x mean |alpha - mean alpha| over ocean added to the loss.
  K   --K cells^2/day, 5-point Laplacian on ocean pixels (0 = off).
  g   --g none | relax (g = -lambda_r (C - clim), one learned scalar, exact exponential step) | mlp (1x1-conv MLP, two hidden
      layers) | conv (3x3 conv, three layers) on the pixel stack [C - clim, clim, SST + flag, (so + flag: blk), MLD, dMLD/MLD,
      wind u/v, log depth, (river distance: blk), DOY sin/cos, lat, ocean], all z-scored as train_spatial.py; g = g_max tanh(.),
      last layer zero-initialised.
  C_0 --ic fill: steady-state anisotropic-diffusion fill of day t0's observed pixels (Dirichlet) with the DOY climatology prior
      and soft terms from days t0-1, t0-2 (negative-lag weights of the frozen spatial run --fill-from; SSTTensorD, no gradient).
      --ic persist: the same with isotropic D. --ic clim: climatology with observed pixels pasted. Nothing after t0 is used.
Loss: L1 on the pixels observed at days t0+1..t0+L, equal weight per lead (no fake clouds). Train t0: training days whose
t0+1..t0+L are training days in the store. Val: every --val-stride-th val day as t0; per lead MAE / RMSE on observed pixels
for the model, persistence of C_0, climatology, transport-only (g off, learned alpha), transport_alpha1 (g off, alpha = 1) and
g-only (transport off), plus grad ratio / corr at
lead 1 on observed pixel pairs and (blk) per-region MAE at lead 1. Output: models/ude/<run>/{config,history,metrics}.json,
model.pt, figures/.

Example
  python -m chl_cnn.train_ude --run ude_io_mlp --g mlp --transport sl --epochs 4
  python -m chl_cnn.train_ude --run ude_blk_conv --store blk --crop --chl-max 100 --g conv --K 0.1 --epochs 4
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data as D
from . import store as S
from . import train2 as T2
from . import spatial_node as SN
from . import ude as U
from .physics import grid_metrics
from .runs import CATEGORIES
from .plotting import MAP_CMAP, ERR_CMAP, OKABE_ITO


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--store", choices=["io", "blk"], default="io", help="io: Indian Ocean 0.25 deg store; blk: Black Sea 2 km store")
    p.add_argument("--chl-max", type=float, default=0.0, help="treat CHL above this value (mg/m3) as missing (0 = off)")
    p.add_argument("--crop", action="store_true", help="crop the grid to the bounding box of the ocean mask (+1 pixel margin)")
    p.add_argument("--val-years", type=int, nargs="*", default=None, help="val years (default: data.VAL_YEARS for io; 2017 2021 for blk)")
    p.add_argument("--test-years", type=int, nargs="*", default=None, help="test years (default: data.TEST_YEARS for io; 2019 2023 for blk)")
    p.add_argument("--lead", type=int, default=3, help="L: number of daily steps / forecast leads scored")
    p.add_argument("--transport", choices=["sl", "none"], default="sl", help="sl: semi-Lagrangian advection; none: identity")
    p.add_argument("--n-sub", type=int, default=4, help="back-trajectory sub-steps per day")
    p.add_argument("--lag-max", type=float, default=20.0, help="clip the daily displacement to +-lag-max cells")
    p.add_argument("--adv-scale", choices=["fixed", "scalar", "field"], default="fixed", help="advection factor alpha on the currents: fixed (1), one learned scalar, or a learned per-pixel field")
    p.add_argument("--adv-init", type=float, default=0.25, help="initial alpha (scalar / field)")
    p.add_argument("--adv-hidden", type=int, default=16, help="field: hidden width of the 1x1-conv alpha net")
    p.add_argument("--adv-reg", type=float, default=0.0, help="field: weight of the L1 penalty on alpha's deviation from its ocean mean")
    p.add_argument("--K", type=float, default=0.0, help="diffusivity in cells^2/day (explicit; keep <= 0.25)")
    p.add_argument("--g", choices=["none", "relax", "mlp", "conv"], default="none", help="source term g_theta")
    p.add_argument("--hidden", type=int, default=32, help="mlp / conv width")
    p.add_argument("--g-max", type=float, default=0.5, help="|g| bound (ln CHL per day) for mlp / conv")
    p.add_argument("--lam0", type=float, default=0.1, help="relax: initial lambda_r (1/day)")
    p.add_argument("--ic", choices=["fill", "clim", "persist"], default="fill", help="initial condition at day t0")
    p.add_argument("--fill-from", default=None, help="spatial run (models/spatial/<run>/model.pt, --dmodel sst) whose D, beta and negative-lag w give the fill (default blk_sst_ctx2 / tc_sst_ctx2)")
    p.add_argument("--tol", type=float, default=1e-5, help="fill CG tolerance")
    p.add_argument("--maxiter", type=int, default=400, help="fill CG iterations")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=None, help="samples (t0) per batch (default 8 for blk, 16 for io)")
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--val-stride", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-2, help="lr of the scalar parameters (lambda_r)")
    p.add_argument("--lr-net", type=float, default=3e-3, help="lr of the mlp / conv weights")
    p.add_argument("--wd", type=float, default=0.0, help="weight decay of the net")
    p.add_argument("--sched", choices=["none", "cosine"], default="none")
    p.add_argument("--clip", type=float, default=10.0, help="gradient-norm clip")
    p.add_argument("--feat-device", choices=["cpu", "gpu"], default="cpu", help="cpu: read each batch's covariates from the memory-mapped store; gpu: float16 caches on the GPU (io: ~3 GB; blk: ~12 GB)")
    p.add_argument("--max-batches", type=int, default=0, help="debug: train only the first N batches per epoch (0 = all)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--init-from", default=None, help="ude run name whose model.pt initialises the parameters")
    p.add_argument("--dump-day", type=int, default=None, help="store index t0: save the rollout (C_0, model, transport-only, g-only, truth, g) to <run>/dump_day_<t0>.npz")
    p.add_argument("--fig-day", type=int, default=None, help="store index t0 of the example-day figure (default: a val t0 with many observations at t0+1)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.batch is None:
        a.batch = 8 if a.store == "blk" else 16
    if a.fill_from is None:
        a.fill_from = "blk_sst_ctx2" if a.store == "blk" else "tc_sst_ctx2"
    return a


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = CATEGORIES["ude"] / a.run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    L = a.lead

    # ---- store, split, CHL, climatology (as train_spatial.py)
    if a.store == "blk":
        from . import store_blk as SB
        d = SB.load_store(); store_dir = SB.store_path()
        val_years = tuple(a.val_years) if a.val_years else (2017, 2021); test_years = tuple(a.test_years) if a.test_years else (2019, 2023)
    else:
        d = S.load_store(); store_dir = S.store_path()
        val_years = tuple(a.val_years) if a.val_years else D.VAL_YEARS; test_years = tuple(a.test_years) if a.test_years else D.TEST_YEARS
    TIMES = d["time"]; T2.TIMES = TIMES
    if a.crop:
        oc = d["ocean"]; rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
        ys, xs = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])), slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
        print(f"crop: rows {ys.start}:{ys.stop} cols {xs.start}:{xs.stop} of {oc.shape}", flush=True)
    else:
        ys, xs = slice(None), slice(None)
    rd = lambda name, t=slice(None): np.asarray(d[name][t, ys, xs])
    n = len(TIMES)
    ocean = d["ocean"][ys, xs]; lat, lon = d["lat"][ys].astype("float64"), d["lon"][xs].astype("float64")
    H, W = ocean.shape
    tr, va, te = D.split_by_years(TIMES, val_years, test_years)
    chl = rd("chl_l3")
    if a.chl_max > 0:
        n_bad = int(np.sum(chl > a.chl_max)); chl = np.where(chl > a.chl_max, np.nan, chl); print(f"chl-max {a.chl_max}: {n_bad} values set to missing", flush=True)
    y = D.log_chl(chl); del chl
    obs = np.isfinite(y) & ocean
    y = np.nan_to_num(y, nan=0.0)
    clim_np, doy_bin = T2.doy_climatology(y, obs, tr)
    chl_mu, chl_sd = float(y[tr][obs[tr]].mean()), float(y[tr][obs[tr]].std())
    print(f"store {a.store}: {n} days  train {len(tr)} val {len(va)}  grid {H} x {W}  ocean px {int(ocean.sum())}  val years {val_years} test years {test_years}", flush=True)

    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    OCEAN = T(ocean); OCf = OCEAN.float()
    CLIMg = T(clim_np); BINg = T(doy_bin)
    grid = SN.Grid(ocean, lat, dev)
    doy = (np.asarray(TIMES) - np.asarray(TIMES).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
    ang = 2 * np.pi * doy / 365.25
    DOYSIN, DOYCOS = T(np.sin(ang).astype("float32")), T(np.cos(ang).astype("float32"))
    lat_desc = bool(lat[0] > lat[-1]); sgn = -1.0 if lat_desc else 1.0
    dxm, dym, _, _ = grid_metrics(lat, lon)                       # signed metres along increasing index
    CPDX, CPDY = T((86400.0 / dxm).astype("float32")), T((86400.0 / dym).astype("float32"))   # cells/day per (m/s), signed
    LATg = T(np.broadcast_to(((lat - lat.mean()) / lat.std()).astype("float32")[:, None], (H, W)).copy())

    # ---- static covariates
    depth_src = "log10(1 + distance to coast km)" if a.store == "blk" else "log10(1 + depth m)"
    if (store_dir / "depth.npy").exists():
        dep = np.load(store_dir / "depth.npy")[ys, xs].astype("float32")
        depth = np.log10(np.maximum(np.nan_to_num(dep, nan=5.0), 5.0)); depth_src = "log10(max(depth.npy, 5 m))"
    else:
        depth = np.log10(np.maximum(-d["topo"][ys, xs], 0) + 1)
    DEPTHg = T(((depth - depth[ocean].mean()) / depth[ocean].std()).astype("float32")) * OCf
    river = (store_dir / "river_dist.npy").exists()
    if river:
        rdist = np.load(store_dir / "river_dist.npy")[:, ys, xs].astype("float32")
        riv = np.log10(1.0 + np.nanmin(rdist, 0)); riv = np.where(np.isfinite(riv), riv, np.nanmax(riv[ocean]))
        RIVg = T(((riv - riv[ocean].mean()) / riv[ocean].std()).astype("float32")) * OCf; del rdist
    print(f"depth channel: {depth_src}; river channel: {river}", flush=True)
    regions = None
    if a.store == "blk":
        from pathlib import Path
        rp = Path(__file__).resolve().parents[1] / "analyses" / "blk"
        if (rp / "regions.npy").exists():
            regions = T(np.load(rp / "regions.npy")[ys, xs].astype("int64")); region_names = json.load(open(rp / "region_names.json"))

    # ---- dynamic covariates: z-scoring statistics on training days (every 20th), as train_spatial.py
    sst_name = "sst32" if "sst32" in d else "sst"
    sal = a.store == "blk"
    names = [sst_name, "u_curr", "v_curr", "u_wind", "v_wind", "mlotst"] + (["so"] if sal else [])
    stats = {}
    for nm in [sst_name, "mlotst"] + (["so", "u_wind"] if sal else []):
        sub = rd(nm, tr[::20]).astype("float32"); stats[nm] = (float(np.nanmean(sub)), float(np.nanstd(sub)) + 1e-6); del sub
    wsd = stats["u_wind"][1] if sal else 5.0
    print(f"z-scores: sst {stats[sst_name]}  mld {stats['mlotst']}  wind sd {wsd:.3g}" + (f"  so {stats['so']}" if sal else ""), flush=True)
    cache = {}
    if a.feat_device == "gpu":
        for nm in names:
            g = torch.empty((n, H, W), dtype=torch.float16, device=dev)
            for b0 in range(0, n, 500):
                g[b0:b0 + 500] = torch.as_tensor(rd(nm, slice(b0, b0 + 500)).astype("float32"), device=dev).half()
            cache[nm] = g
        print(f"covariate caches on GPU: {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    def fetch(nm, days):
        """Raw field (NaN kept) for the day-index matrix days (B, nd); days < 0 -> NaN. Returns (B, nd, H, W) float32."""
        flat = days.ravel(); okd = flat >= 0
        if nm in cache:
            x = cache[nm][torch.as_tensor(flat.clip(0), device=dev)].float()
        else:
            uniq, inv = np.unique(flat.clip(0), return_inverse=True)
            x = torch.as_tensor(np.asarray(d[nm][uniq, ys, xs]).astype("float32")[inv], device=dev)
        x[torch.as_tensor(~okd, device=dev)] = float("nan")
        return x.view(days.shape[0], days.shape[1], H, W)

    # ---- sample index matrix: columns k = -2 .. L  (column c = k + 2); t0 at column 2
    OFFS = list(range(-2, L + 1))
    IDX = np.stack([T2.neighbour_index(TIMES, k) for k in OFFS], 1)        # (n, L+3), -1 where missing
    in_split = np.zeros(n, dtype=np.int8); in_split[tr] = 1; in_split[va] = 2; in_split[te] = 3
    def valid_t0(idx_set, code):
        ok = np.ones(len(idx_set), dtype=bool)
        for k in range(1, L + 1):
            j = IDX[idx_set, k + 2]; ok &= (j >= 0) & (in_split[j.clip(0)] == code)
        return idx_set[ok]
    tr_t0 = valid_t0(tr, 1); va_t0 = valid_t0(va, 2)
    va_eval = va_t0[::a.val_stride]
    print(f"t0 candidates: train {len(tr_t0)} (stride {a.train_stride} -> {len(tr_t0[::a.train_stride])})  val {len(va_t0)} (stride {a.val_stride} -> {len(va_eval)})", flush=True)

    # ---- fill model (frozen) for the initial condition
    fill = None
    if a.ic in ("fill", "persist"):
        ck = torch.load(CATEGORIES["spatial"] / a.fill_from / "model.pt", map_location=dev)
        cfg = json.load(open(CATEGORIES["spatial"] / a.fill_from / "config.json"))
        assert cfg.get("dmodel") == "sst" and cfg.get("context", 0) >= 2, "--fill-from must be a --dmodel sst run with context >= 2"
        fillD = SN.SSTTensorD().to(dev); fillD.load_state_dict(ck["dmodel"]); fillD.requires_grad_(False)
        if a.ic == "persist":
            fillD = SN.IsoD(d0=ck["dmodel"]["log_d0"].exp().item()).to(dev); fillD.requires_grad_(False)
        offs = [k for k in range(-cfg["context"], cfg["context"] + 1) if k != 0]
        wk = dict(zip(offs, F.softplus(ck["raw_w"]).tolist()))
        fill = dict(beta=ck["log_beta"].exp().item(), w1=wk[-1], w2=wk[-2], d0=fillD.log_d0.exp().item() if hasattr(fillD, "log_d0") else fillD.log_d.exp().item(),
                    g0=ck["dmodel"]["log_g0"].exp().item(), run=a.fill_from)
        print(f"ic {a.ic} from {a.fill_from}: beta {fill['beta']:.3g} w(-1) {fill['w1']:.3g} w(-2) {fill['w2']:.3g} d0 {fill['d0']:.3g} g0 {fill['g0']:.3g}", flush=True)
        zero_f = torch.zeros((1, H, W), device=dev)

    # ---- model
    transport = U.Transport(OCEAN, n_sub=a.n_sub, lag_max=a.lag_max) if a.transport == "sl" else None
    FEATS = ["C - clim", "clim", "sst", "sst ok"] + (["so", "so ok"] if sal else []) + ["mld", "dmld/mld", "wind u", "wind v", "log depth"] + (["river dist"] if river else []) + ["doy sin", "doy cos", "lat", "ocean"]
    gmodel, scalars, net_params = None, [], []
    if a.g == "relax":
        gmodel = U.Relax(a.lam0).to(dev); scalars = list(gmodel.parameters())
    elif a.g in ("mlp", "conv"):
        gmodel = U.GNet(a.g, len(FEATS), hidden=a.hidden, g_max=a.g_max).to(dev); net_params = list(gmodel.parameters())
    amodel = None
    if a.adv_scale != "fixed" and transport is not None:
        amodel = U.AdvScale(a.adv_scale, len(FEATS), hidden=a.adv_hidden, alpha0=a.adv_init).to(dev)
        if a.adv_scale == "scalar":
            scalars = scalars + list(amodel.parameters())
        else:
            net_params = net_params + list(amodel.parameters())
    if a.init_from:
        ck0 = torch.load(CATEGORIES["ude"] / a.init_from / "model.pt", map_location=dev)
        if gmodel is not None: gmodel.load_state_dict(ck0["gmodel"])
        if amodel is not None and ck0.get("amodel"): amodel.load_state_dict(ck0["amodel"])
    groups = ([{"params": scalars, "lr": a.lr, "weight_decay": 0.0}] if scalars else []) + ([{"params": net_params, "lr": a.lr_net, "weight_decay": a.wd}] if net_params else [])
    opt = torch.optim.AdamW(groups) if groups else None
    n_params = sum(p.numel() for p in scalars + net_params)
    print(f"transport {a.transport} (n_sub {a.n_sub}, lag_max {a.lag_max})  adv-scale {a.adv_scale} (init {a.adv_init}, reg {a.adv_reg})  K {a.K}  g {a.g} ({n_params} params, {len(FEATS)} features)  ic {a.ic}  lead {L}", flush=True)
    alpha_stats = {}                                                   # field: mean / p10 / p90 over ocean pixel-days of the last evaluation

    def params_summary():
        s = dict(K=a.K)
        if a.g == "relax":
            s["lambda_r"] = gmodel.lam().item()
        if amodel is not None and a.adv_scale == "scalar":
            s["alpha"] = amodel().item()
        elif amodel is not None:
            s.update({f"alpha_{k_}": v_ for k_, v_ in alpha_stats.items()})
        return s

    # ---- batches
    def make_batch(t0s):
        """Everything the rollout needs for the samples t0s (np array): dict of GPU tensors indexed [:, c] with c = k + 2."""
        idx = IDX[t0s]; B = len(t0s)
        ic = idx.clip(0)
        b = dict(idx=idx, Y=T(y[ic]), OB=T(obs[ic]) & T(idx >= 0)[:, :, None, None], CLIM=CLIMg[BINg[T(ic)]])
        sst = fetch(sst_name, idx); b["SSTOK"] = torch.isfinite(sst) & OCEAN
        b["SST"] = torch.nan_to_num((sst - stats[sst_name][0]) / stats[sst_name][1]) * b["SSTOK"]
        u, v = fetch("u_curr", idx), fetch("v_curr", idx)
        b["CUROK"] = torch.isfinite(u) & torch.isfinite(v) & OCEAN
        b["U"] = torch.nan_to_num(u) * CPDX; b["V"] = torch.nan_to_num(v) * CPDY
        b["UW"] = torch.nan_to_num(fetch("u_wind", idx) / wsd) * OCf; b["VW"] = torch.nan_to_num(fetch("v_wind", idx) / wsd * sgn) * OCf
        mld = fetch("mlotst", idx)
        b["MLD"] = torch.nan_to_num((mld - stats["mlotst"][0]) / stats["mlotst"][1]) * OCf
        dm = torch.zeros_like(mld); dm[:, 1:] = (mld[:, 1:] - mld[:, :-1]) / mld[:, 1:].clamp(min=1e-3)
        b["DMLD"] = torch.nan_to_num(dm).clamp(-2, 2) * OCf
        if sal:
            so = fetch("so", idx); b["SOOK"] = torch.isfinite(so) & OCEAN
            b["SO"] = torch.nan_to_num((so - stats["so"][0]) / stats["so"][1]) * b["SOOK"]
        b["DSIN"], b["DCOS"] = DOYSIN[T(ic)], DOYCOS[T(ic)]
        return b

    @torch.no_grad()
    def initial_condition(b):
        """C_0 at day t0 (column 2) from day t0's observations and the one-sided context t0-1, t0-2."""
        vis = b["OB"][:, 2]; prior = b["CLIM"][:, 2]; Y0 = b["Y"][:, 2]
        if a.ic == "clim":
            return torch.where(vis, Y0, prior) * OCf
        B = vis.shape[0]; zf = zero_f.expand(B, -1, -1)
        f = SN.corner_features(grid, b["SST"][:, 2], b["SSTOK"][:, 2], zf, zf, torch.ones_like(vis), zf, zf, zf[0])
        Dc = fillD(f)
        wsum = fill["beta"] + fill["w1"] * b["OB"][:, 1].float() + fill["w2"] * b["OB"][:, 0].float()
        tsum = fill["beta"] * prior + fill["w1"] * b["OB"][:, 1].float() * b["Y"][:, 1] + fill["w2"] * b["OB"][:, 0].float() * b["Y"][:, 0]
        free = OCEAN & ~vis
        return SN.steady_state(grid, Dc, wsum, tsum / wsum, Y0 * vis, free, tol=a.tol, maxiter=a.maxiter)

    def gstack(b, k, C):
        """Pixel covariate stack (B, F, H, W) for step k (day t0+k, column c = k + 2) given the state C."""
        c = k + 2; B = C.shape[0]
        ch = [(C - b["CLIM"][:, c]) / chl_sd * OCf, (b["CLIM"][:, c] - chl_mu) / chl_sd * OCf, b["SST"][:, c], b["SSTOK"][:, c].float()]
        if sal:
            ch += [b["SO"][:, c], b["SOOK"][:, c].float()]
        ch += [b["MLD"][:, c], b["DMLD"][:, c], b["UW"][:, c], b["VW"][:, c], DEPTHg.expand(B, -1, -1)]
        if river:
            ch += [RIVg.expand(B, -1, -1)]
        ch += [b["DSIN"][:, c].view(B, 1, 1).expand(B, H, W), b["DCOS"][:, c].view(B, 1, 1).expand(B, H, W), LATg.expand(B, -1, -1), OCf.expand(B, -1, -1)]
        return torch.stack(ch, 1)

    def run(b, C0, use_T=True, use_g=True, collect=None, alpha1=False):
        """Rollout; alpha1 True runs the transport with alpha = 1 (ignores the learned advection factor)."""
        if a.g == "relax":
            gfun = lambda k, C: gmodel(C, b["CLIM"][:, k + 2], 1.0)
        elif a.g in ("mlp", "conv"):
            gfun = lambda k, C: gmodel(gstack(b, k, C)) * OCf
        else:
            gfun = None
        cur = lambda k: (b["U"][:, k + 2], b["V"][:, k + 2], b["CUROK"][:, k + 2])
        afun = None
        if amodel is not None and not alpha1:
            afun = (lambda k, C: amodel()) if a.adv_scale == "scalar" else (lambda k, C: amodel(gstack(b, k, C)))
        return U.rollout(C0, L, OCEAN, transport=transport, currents=cur, K=a.K, gfun=gfun, use_T=use_T, use_g=use_g, dt=1.0, collect=collect, afun=afun)

    def lead_loss(preds, b):
        """Mean over leads of the observed-pixel L1 at day t0+k."""
        tot, cnt = 0.0, 0
        for k in range(1, L + 1):
            m = b["OB"][:, k + 2].float(); s = m.sum()
            if s > 0:
                tot = tot + ((preds[:, k - 1] - b["Y"][:, k + 2]).abs() * m).sum() / s; cnt += 1
        return tot / max(cnt, 1)

    # ---- evaluation
    @torch.no_grad()
    def evaluate(t0s, pd=False):
        if gmodel is not None: gmodel.eval()
        kinds = ["model", "persistence", "climatology", "transport_only", "transport_alpha1", "g_only"]
        acc = {kd: dict(err=np.zeros(L), sq=np.zeros(L)) for kd in kinds}
        cnt = np.zeros(L); gs = dict(gp=0.0, gt=0.0, gpt=0.0, gpp=0.0, gtt=0.0, gn=0.0); frac = []; t_fill = 0.0; t_roll = 0.0
        reg = None
        if regions is not None:
            reg = {kd: np.zeros(8) for kd in kinds}; regn = np.zeros(8)
        gsum = torch.zeros((H, W), device=dev); gcnt = 0; pdx, pdg = [], []; asum = torch.zeros((H, W), device=dev); acnt = 0; avals = []
        for i in range(0, len(t0s), a.batch):
            b = make_batch(t0s[i:i + a.batch])
            t1 = time.time(); C0 = initial_condition(b); torch.cuda.synchronize(); t_fill += time.time() - t1
            t1 = time.time(); col = {}
            Pm, fr = run(b, C0, True, True, collect=col); frac.append(fr.item())
            P = dict(model=Pm, transport_only=run(b, C0, True, False)[0], transport_alpha1=run(b, C0, True, False, alpha1=True)[0], g_only=run(b, C0, False, True)[0])
            P["persistence"] = C0.unsqueeze(1).expand(-1, L, -1, -1); P["climatology"] = b["CLIM"][:, 3:3 + L]
            torch.cuda.synchronize(); t_roll += time.time() - t1
            for k in range(1, L + 1):
                m = b["OB"][:, k + 2].float(); cnt[k - 1] += m.sum().item()
                for kd in kinds:
                    e = (P[kd][:, k - 1] - b["Y"][:, k + 2]) * m
                    acc[kd]["err"][k - 1] += e.abs().sum().item(); acc[kd]["sq"][k - 1] += (e ** 2).sum().item()
            # gradient ratio / correlation at lead 1 on observed pixel pairs
            ob = b["OB"][:, 3]; pr = P["model"][:, 0]; yt = b["Y"][:, 3]
            for ax in (-1, -2):
                dp = torch.diff(pr, dim=ax); dt_ = torch.diff(yt, dim=ax)
                m = (torch.narrow(ob, ax, 0, ob.shape[ax] - 1) & torch.narrow(ob, ax, 1, ob.shape[ax] - 1)).float()
                gs["gp"] += (dp.abs() * m).sum().item(); gs["gt"] += (dt_.abs() * m).sum().item()
                gs["gpt"] += (dp * dt_ * m).sum().item(); gs["gpp"] += (dp * dp * m).sum().item(); gs["gtt"] += (dt_ * dt_ * m).sum().item(); gs["gn"] += m.sum().item()
            if reg is not None:
                m = ob.float()
                regn += torch.bincount(regions.expand_as(m)[ob], minlength=8).cpu().numpy()
                for kd in kinds:
                    e = (P[kd][:, 0] - yt).abs()
                    reg[kd] += torch.bincount(regions.expand_as(m)[ob], weights=e[ob], minlength=8).cpu().numpy()
            if "alpha" in col and col["alpha"][0].dim() == 3:
                a0 = col["alpha"][0]; asum += a0.sum(0); acnt += a0.shape[0]; avals.append(a0[:, OCEAN].flatten().cpu())
            if "g" in col:
                g0 = col["g"][0]; gsum += g0.sum(0); gcnt += g0.shape[0]
                if pd:
                    x = gstack(b, 0, C0)
                    sel = OCEAN.flatten().nonzero()[::53, 0]
                    pdx.append(x.flatten(2)[:, :, sel].permute(0, 2, 1).reshape(-1, x.shape[1]).cpu()); pdg.append(g0.flatten(1)[:, sel].reshape(-1).cpu())
        if gmodel is not None: gmodel.train()
        res = dict(leads=list(range(1, L + 1)), n_obs=cnt.astype(int).tolist(), n_t0=int(len(t0s)),
                   grad_ratio_l1=gs["gp"] / max(gs["gt"], 1e-9), grad_corr_l1=gs["gpt"] / max((gs["gpp"] * gs["gtt"]) ** 0.5, 1e-9), n_grad_l1=int(gs["gn"]),
                   frac_not_advected=float(np.mean(frac)), fill_s_per_batch=t_fill / max(math.ceil(len(t0s) / a.batch), 1), rollout_s_per_batch=t_roll / max(math.ceil(len(t0s) / a.batch), 1))
        for kd in kinds:
            res[kd] = dict(mae=(acc[kd]["err"] / np.maximum(cnt, 1)).tolist(), rmse=np.sqrt(acc[kd]["sq"] / np.maximum(cnt, 1)).tolist())
        if reg is not None:
            res["regions_l1"] = {region_names.get(str(r), f"region {r}"): {**{kd: float(reg[kd][r] / max(regn[r], 1)) for kd in kinds}, "n_obs": int(regn[r])} for r in range(8) if regn[r] > 0}
        if avals:
            av = torch.cat(avals).numpy()
            alpha_stats.clear(); alpha_stats.update(mean=float(av.mean()), p10=float(np.percentile(av, 10)), p90=float(np.percentile(av, 90)))
            res["alpha"] = dict(alpha_stats, n_px_days=int(len(av)))
        elif amodel is not None:
            res["alpha"] = amodel().item()
        extra = dict(gmap=(gsum / max(gcnt, 1)).cpu().numpy() if gcnt else None, pdx=torch.cat(pdx).numpy() if pdx else None, pdg=torch.cat(pdg).numpy() if pdg else None,
                     amap=(asum / max(acnt, 1)).cpu().numpy() if acnt else None)
        return res, extra

    # ---- figures
    ext = [lon[0], lon[-1], lat[-1], lat[0]] if lat_desc else [lon[0], lon[-1], lat[0], lat[-1]]
    origin = "upper" if lat_desc else "lower"
    if a.fig_day is None:
        nobs1 = obs[IDX[va_eval, 3]].reshape(len(va_eval), -1).sum(1)
        fig_day = int(va_eval[np.argsort(nobs1)[int(0.75 * len(va_eval))]]) if len(va_eval) else int(va_t0[0])
    else:
        fig_day = a.fig_day

    def show(ax, img, title, cmap, vmin, vmax):
        img = np.where(ocean, img, np.nan)
        im = ax.imshow(img, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, aspect="auto")
        ax.set_title(title, fontsize=9); ax.set_xlabel("lon"); ax.set_ylabel("lat"); plt.colorbar(im, ax=ax, shrink=0.8)

    @torch.no_grad()
    def example_figure(tag):
        if gmodel is not None: gmodel.eval()
        b = make_batch(np.array([fig_day])); C0 = initial_condition(b)
        Pm, _ = run(b, C0, True, True); Pt, _ = run(b, C0, True, False)
        if gmodel is not None: gmodel.train()
        date = lambda c: str(TIMES[b["idx"][0, c]])[:10]
        fig, ax = plt.subplots(4, L + 1, figsize=(4.2 * (L + 1), 13))
        show(ax[0, 0], C0[0].cpu().numpy(), f"C_0 ({a.ic}) at t0 = {date(2)}", MAP_CMAP, -3, 2)
        show(ax[1, 0], np.where(obs[fig_day], y[fig_day], np.nan), f"observed ln CHL at t0 = {date(2)}", MAP_CMAP, -3, 2)
        ax[2, 0].axis("off"); ax[3, 0].axis("off")
        for k in range(1, L + 1):
            j = b["idx"][0, k + 2]; tr_ = np.where(obs[j], y[j], np.nan)
            show(ax[0, k], tr_, f"observed ln CHL, t0+{k} = {date(k + 2)}", MAP_CMAP, -3, 2)
            show(ax[1, k], Pm[0, k - 1].cpu().numpy(), f"model, lead {k}", MAP_CMAP, -3, 2)
            show(ax[2, k], Pt[0, k - 1].cpu().numpy(), f"transport-only, lead {k}", MAP_CMAP, -3, 2)
            show(ax[3, k], np.where(obs[j], Pm[0, k - 1].cpu().numpy() - y[j], np.nan), f"model minus observed, lead {k} (ln CHL)", ERR_CMAP, -1, 1)
        fig.suptitle(f"{a.run}: {tag}; transport {a.transport}, K {a.K}, g {a.g}; params {json.dumps(params_summary())}", fontsize=10)
        fig.savefig(out / "figures" / f"example_{tag}.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    def alpha_figure(tag, extra):
        if extra.get("amap") is None:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        show(ax, extra["amap"], f"Time-mean learned advection factor alpha at step 0 over {len(va_eval)} val days", "viridis", 0, 1.5)
        fig.savefig(out / "figures" / f"alpha_map_{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    def g_figures(tag, extra):
        if extra["gmap"] is None:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        show(ax, extra["gmap"], f"Time-mean learned tendency g at step 0 over {len(va_eval)} val days (ln CHL per day)", ERR_CMAP, -0.2, 0.2)
        fig.savefig(out / "figures" / f"g_mean_map_{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        if extra["pdx"] is None:
            return
        X, G = extra["pdx"], extra["pdg"]; nf = X.shape[1]; nc = 4; nr = math.ceil(nf / nc)
        fig, ax = plt.subplots(nr, nc, figsize=(4 * nc, 2.8 * nr)); ax = ax.ravel()
        for f in range(nf):
            xv = X[:, f]; u = np.unique(xv)
            if len(u) <= 10:
                centres = u; means = [G[xv == c].mean() for c in u]
            else:
                edges = np.quantile(xv, np.linspace(0, 1, 11)); bins = np.clip(np.searchsorted(edges, xv, side="right") - 1, 0, 9)
                centres = [np.median(xv[bins == q]) for q in range(10)]; means = [G[bins == q].mean() if (bins == q).any() else np.nan for q in range(10)]
            ax[f].plot(centres, means, "o-", color=OKABE_ITO[0]); ax[f].axhline(0, color="k", lw=0.5)
            ax[f].set_title(FEATS[f], fontsize=9); ax[f].set_xlabel("feature value (decile medians)"); ax[f].set_ylabel("mean g (ln CHL/day)")
        for f in range(nf, len(ax)): ax[f].axis("off")
        fig.suptitle(f"{a.run}: {tag}; mean learned tendency g by decile of each input feature (val days, step 0, {len(G)} ocean px)", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "figures" / f"partial_dependence_{tag}.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    def history_figure():
        if not hist["val"]:
            return
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ep = np.arange(len(hist["val"]))
        for k in range(L):
            ax[0].plot(ep, [v["model"]["mae"][k] for v in hist["val"]], "o-", color=OKABE_ITO[k], label=f"model lead {k + 1}")
            ax[0].plot(ep, [v["persistence"]["mae"][k] for v in hist["val"]], "--", color=OKABE_ITO[k], label=f"persistence lead {k + 1}")
        ax[0].set_xlabel("epoch (0 = init)"); ax[0].set_ylabel("val MAE on observed pixels (ln CHL)"); ax[0].set_title("Val MAE by lead per epoch"); ax[0].legend(fontsize=7)
        ax[1].plot(ep[1:], hist["train"][1:], "o-", color=OKABE_ITO[0]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("train L1 (mean over leads)"); ax[1].set_title("Training loss per epoch")
        fig.tight_layout(); fig.savefig(out / "figures" / "history.png", dpi=110); plt.close(fig)

    # ---- bookkeeping
    json.dump(dict(vars(a), sst_source=sst_name, depth_source=depth_src, river_feat=river, features=FEATS, fill=fill, n_train_t0=len(tr_t0), n_val_t0=len(va_t0), n_val_eval=len(va_eval),
                   n_params=n_params, val_years=list(val_years), test_years=list(test_years), grid=[H, W], n_ocean=int(ocean.sum()), chl_mu=chl_mu, chl_sd=chl_sd, fig_day=fig_day),
              open(out / "config.json", "w"), indent=1)
    hist = dict(train=[], val=[], params=[], epoch_time=[], lr=[])

    def log_eval(tag, ev, ltr=None, dt=0.0):
        for k_, v_ in dict(train=ltr, val=ev, params=params_summary(), epoch_time=dt, lr=(opt.param_groups[-1]["lr"] if opt else None)).items():
            hist[k_].append(v_)
        fmt = lambda kd: "/".join(f"{m:.4f}" for m in ev[kd]["mae"])
        print(f"{tag}  train {ltr if ltr is None else round(ltr, 4)}  val MAE by lead  model {fmt('model')}  persist {fmt('persistence')}  clim {fmt('climatology')}  "
              f"T-only {fmt('transport_only')}  T-a1 {fmt('transport_alpha1')}  g-only {fmt('g_only')}  grad ratio {ev['grad_ratio_l1']:.3f} corr {ev['grad_corr_l1']:.3f}  not advected {ev['frac_not_advected']:.3f}  "
              f"fill {ev['fill_s_per_batch']:.2f} s/batch rollout {ev['rollout_s_per_batch']:.2f} s/batch  params {json.dumps(params_summary())}  ({dt:.0f} s)", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)

    def save():
        torch.save(dict(gmodel=gmodel.state_dict() if gmodel is not None else {}, amodel=amodel.state_dict() if amodel is not None else {}, adv_scale=a.adv_scale, g=a.g, K=a.K, features=FEATS), out / "model.pt")

    if a.dump_day is not None:
        with torch.no_grad():
            b = make_batch(np.array([a.dump_day])); C0 = initial_condition(b); col = {}
            Pm, _ = run(b, C0, True, True, collect=col); Pt, _ = run(b, C0, True, False); Pg, _ = run(b, C0, False, True)
            np.savez_compressed(out / f"dump_day_{a.dump_day}.npz", idx=b["idx"][0], C0=C0[0].cpu().numpy(), model=Pm[0].cpu().numpy(), transport_only=Pt[0].cpu().numpy(), g_only=Pg[0].cpu().numpy(),
                                truth=y[b["idx"][0].clip(0)], obs=obs[b["idx"][0].clip(0)], clim=b["CLIM"][0].cpu().numpy(), g=np.stack([g[0].cpu().numpy() for g in col["g"]]) if "g" in col else None,
                                ocean=ocean, lat=lat, lon=lon, times=np.array([str(TIMES[j])[:10] if j >= 0 else "" for j in b["idx"][0]]))
            print(f"dumped rollout from t0 {a.dump_day} {str(TIMES[a.dump_day])[:10]}", flush=True)

    ev, extra = evaluate(va_eval, pd=a.g in ("mlp", "conv")); log_eval("init", ev, dt=time.time() - t_start)
    example_figure("init"); g_figures("init", extra); alpha_figure("init", extra)
    best = float(np.mean(ev["model"]["mae"])); save()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs) if (opt and a.sched == "cosine" and a.epochs > 0) else None
    if not a.eval_only and opt is not None:
        for ep in range(a.epochs):
            te0 = time.time(); order = rng.permutation(tr_t0[::a.train_stride]); tot = cnt = 0.0
            for nb_, i in enumerate(range(0, len(order), a.batch)):
                if a.max_batches and nb_ >= a.max_batches:
                    break
                b = make_batch(order[i:i + a.batch]); C0 = initial_condition(b); col = {}
                preds, _ = run(b, C0, True, True, collect=col)
                loss = lead_loss(preds, b)
                if a.adv_reg > 0 and "alpha" in col:
                    loss = loss + a.adv_reg * torch.stack([U.AdvScale.reg(al, OCEAN) for al in col["alpha"]]).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(scalars + net_params, a.clip); opt.step()
                tot += loss.item() * len(b["idx"]); cnt += len(b["idx"])
                if nb_ % 50 == 0:
                    print(f"  ep {ep} batch {nb_}/{math.ceil(len(order) / a.batch)} loss {loss.item():.4f} params {json.dumps(params_summary())} ({time.time() - te0:.0f} s)", flush=True)
            if sched: sched.step()
            ev, extra = evaluate(va_eval, pd=a.g in ("mlp", "conv")); log_eval(f"ep {ep}", ev, tot / max(cnt, 1), time.time() - te0)
            if float(np.mean(ev["model"]["mae"])) < best:
                best = float(np.mean(ev["model"]["mae"])); save()
            example_figure(f"epoch_{ep:03d}"); g_figures(f"epoch_{ep:03d}", extra); alpha_figure(f"epoch_{ep:03d}", extra)
    history_figure()
    mem = torch.cuda.max_memory_allocated() / 2**30
    json.dump(dict(val=ev, best_val_mae_mean_leads=best, params=params_summary(), alpha=ev.get("alpha"), val_t0=len(va_eval), minutes=(time.time() - t_start) / 60, max_mem_gb=mem), open(out / "metrics.json", "w"), indent=1)
    print(f"VAL MAE by lead: model {ev['model']['mae']}  persistence {ev['persistence']['mae']}  climatology {ev['climatology']['mae']}  n_obs {ev['n_obs']}")
    print(f"wrote {out} ({(time.time() - t_start) / 60:.1f} min, max GPU memory {mem:.2f} GB)")


if __name__ == "__main__":
    main()
