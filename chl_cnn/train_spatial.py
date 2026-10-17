"""Spatial nODE gap filler: steady state of anisotropic diffusion with observed pixels as data and the day-of-year
climatology as a mean-reversion prior (chl_cnn/spatial_node.py).

    pred = argmin_C  1/2 sum_corners a_c grad C^T D(x) grad C
                   + 1/2 sum_pixels a_p [ beta (C - clim)^2 + alpha vis_t (C - y_t)^2 + sum_k w_k m_k (C - y_{t+k})^2 ]

  vis_t   today's visible pixels (fake clouds applied). alpha = 0 (default) means Dirichlet: C = y_t exactly.
  m_k     visible pixels of day t+k (k = +-1..--context) after the donor's cloud sequence is applied to them too
          (as in train_pinn.py, otherwise hidden pixels leak through the neighbouring days); w_k = softplus(.) learned.
  --lagrangian  the neighbour-day field and mask are first advected to day t along the OSCAR currents of day t
          (semi-Lagrangian: C_t(x) ~ y_{t+k}(x + k u dt), bilinear grid_sample, masked so hidden/land pixels do not
          bleed in; the advected mask fraction becomes the per-pixel weight). Off by default.

Same store, split, fake clouds and hidden-pixel scoring as train_pinn.py. D(x) from --dmodel:
  iso   D = I                                                        learned: beta (, w_k)
  sst   1 along SST isotherms, 1/(1+(|grad SST|/g0)^2) across        learned: beta, g0, d0
  mlp   1x1-conv MLP on corner covariates (--extra-feats adds climatology gradient, depth gradient, lat, |grad SST|)
  conv  3x3 conv net on the pixel covariate stack, averaged to corners (--width, --depth, --dilation); with
        --ctx-feat the visible CHL anomaly and mask of days t+-k are extra input channels
Black Sea covariates (--store blk, all off by default): --real-depth (depth.npy), --sal-feat (salinity so + validity; corner
gradient for the mlp), --river-feat (log10(1 + min river-mouth distance)). --norm-d rescales D to mean trace 2 per sample.
Pixel-stack channel order: 11 base, [so, so_ok], [river], then the 2*context ctx channels.
Loss: L1 on fake-cloud-hidden pixels (ln CHL). Scores: hidden-pixel MAE / RMSE, gradient ratio and correlation on
hidden pixel pairs (as train_pinn.py), climatology reference. Output: models/spatial/<run>/.

Example
  python -m chl_cnn.train_spatial --run sp_conv_ctx2 --dmodel conv --context 2 --ctx-feat --epochs 8
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
from . import clouds as C
from . import store as S
from . import train2 as T2
from . import spatial_node as SN
from .runs import CATEGORIES
from .plotting import MAP_CMAP, ERR_CMAP


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--store", choices=["io", "blk"], default="io", help="io: Indian Ocean 0.25 deg store (store.py); blk: Black Sea 2 km store (store_blk.py)")
    p.add_argument("--chl-max", type=float, default=0.0, help="treat CHL above this value (mg/m3) as missing (0 = off); the Black Sea L3 has retrievals up to 2e5 mg/m3")
    p.add_argument("--crop", action="store_true", help="crop the grid to the bounding box of the ocean mask (+1 pixel margin) before anything else")
    p.add_argument("--val-years", type=int, nargs="*", default=None, help="val years (default: data.VAL_YEARS for io; 2017 2021 for blk)")
    p.add_argument("--test-years", type=int, nargs="*", default=None, help="test years (default: data.TEST_YEARS for io; 2019 2023 for blk)")
    p.add_argument("--dmodel", default="iso", help="iso | sst | mlp | conv | any kind registered in spatial_node.build_dmodel")
    p.add_argument("--dkw", default="{}", help="JSON of extra keyword arguments for spatial_node.build_dmodel")
    p.add_argument("--source", action="store_true", help="D model also returns a source field g (e.g. --dmodel convsrc); enters as b += a_p beta g, i.e. g shifts the climatology prior (ln CHL units)")
    p.add_argument("--pixel-input", action="store_true", help="feed the D model the pixel covariate stack (B,C,H,W) like conv does, instead of corner features")
    p.add_argument("--beta0", type=float, default=0.01, help="initial mean-reversion rate beta towards the climatology")
    p.add_argument("--fix-beta", action="store_true")
    p.add_argument("--alpha", type=float, default=0.0, help="soft weight of today's visible pixels (0 = Dirichlet, exact)")
    p.add_argument("--learn-alpha", action="store_true")
    p.add_argument("--context", type=int, default=0, help="neighbour days t+-1..K as soft data terms with learned weights")
    p.add_argument("--w0", type=float, default=0.05, help="initial neighbour-day weight")
    p.add_argument("--ctx-feat", action="store_true", help="conv: neighbour-day visible CHL anomaly + mask as input channels")
    p.add_argument("--lagrangian", action="store_true", help="advect the neighbour-day fields to day t along the day-t OSCAR currents before using them")
    p.add_argument("--lag-max", type=float, default=5.0, help="lagrangian: clip the displacement to +-lag-max cells")
    p.add_argument("--g0", type=float, default=0.05)
    p.add_argument("--extra-feats", action="store_true", help="mlp: add |grad SST|, climatology gradient, depth gradient, lat, clim")
    p.add_argument("--hidden", type=int, default=32, help="mlp hidden units / conv width")
    p.add_argument("--depth", type=int, default=3, help="conv layers")
    p.add_argument("--dilation", type=int, default=1, help="conv dilation growth factor per layer")
    p.add_argument("--d-max", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--val-stride", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-2, help="lr of the scalar parameters (beta, w_k, g0, alpha)")
    p.add_argument("--lr-net", type=float, default=3e-3, help="lr of the mlp / conv weights")
    p.add_argument("--wd", type=float, default=0.0, help="weight decay of the net")
    p.add_argument("--sched", choices=["none", "cosine"], default="cosine")
    p.add_argument("--norm-d", action="store_true", help="normalise D so the mean trace over ocean corners is 2 per sample (removes the D/beta/w scale freedom; non-detached mean)")
    p.add_argument("--real-depth", action="store_true", help="blk: depth channel = log10(max(depth.npy, 5 m)) instead of log10(1 + distance to coast); falls back with a warning if depth.npy is missing")
    p.add_argument("--sal-feat", action="store_true", help="blk: salinity so (z-scored, NaN -> 0) + validity flag as pixel channels (conv); its masked corner gradient + validity for the mlp")
    p.add_argument("--river-feat", action="store_true", help="blk: log10(1 + min river-mouth distance km) from river_dist.npy, z-scored, as a static pixel channel (conv) / corner mean (mlp)")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--donor-window", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--save-preds", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--init-from", default=None, help="run name whose model.pt initialises the parameters")
    p.add_argument("--dump-days", default=None, help="comma-separated day indices: dump the operator inputs (D at corners, data weights/targets, fixed values, free mask, truth, hidden, SST) to <run>/dumps/day_<i>.npz for analyses")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = CATEGORIES["spatial"] / a.run
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.store == "blk":
        from . import store_blk as SB
        d = SB.load_store()
        val_years = tuple(a.val_years) if a.val_years else (2017, 2021); test_years = tuple(a.test_years) if a.test_years else (2019, 2023)
    else:
        d = S.load_store()
        val_years = tuple(a.val_years) if a.val_years else D.VAL_YEARS; test_years = tuple(a.test_years) if a.test_years else D.TEST_YEARS
    TIMES = d["time"]; T2.TIMES = TIMES
    if a.crop:
        oc = d["ocean"]; rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
        ys, xs = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])), slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
        print(f"crop: rows {ys.start}:{ys.stop} cols {xs.start}:{xs.stop} of {oc.shape}", flush=True)
    else:
        ys, xs = slice(None), slice(None)
    rd = lambda name, t=slice(None): np.asarray(d[name][t, ys, xs])          # cropped read of a (time, lat, lon) array
    n = len(TIMES)
    ocean = d["ocean"][ys, xs]; lat, lon = d["lat"][ys], d["lon"][xs]
    H, W = ocean.shape
    tr, va, te = D.split_by_years(TIMES, val_years, test_years)
    print(f"store {a.store}: val years {val_years} test years {test_years}", flush=True)
    chl = rd("chl_l3")
    if a.chl_max > 0:
        n_bad = int(np.sum(chl > a.chl_max)); chl = np.where(chl > a.chl_max, np.nan, chl); print(f"chl-max {a.chl_max}: {n_bad} values set to missing", flush=True)
    y = D.log_chl(chl); del chl
    obs = np.isfinite(y) & ocean
    hid_va, don_va = C.fixed_hidden(obs, va, seed=a.eval_seed)
    hidden = np.zeros_like(obs); hidden[va] = hid_va
    donor = np.zeros(n, dtype=np.int64); donor[va] = don_va
    clim_np, doy_bin = T2.doy_climatology(y, obs, tr)
    chl_mu, chl_sd = float(np.nanmean(y[tr])), float(np.nanstd(y[tr]))
    print(f"{n} days  train {len(tr)} val {len(va)}  ocean px {int(ocean.sum())}", flush=True)

    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    Yg = torch.nan_to_num(T(y), nan=0.0); OBSg = T(obs); HIDg = T(hidden); DONg = T(donor)
    CLIMg = T(clim_np); BINg = T(doy_bin)
    grid = SN.Grid(ocean, lat, dev)
    OCEANf = grid.ocean.float()

    def to_half(name, mu=0.0, sd=1.0, sign=1.0):
        g = torch.empty((n, H, W), dtype=torch.float16, device=dev)
        for b0 in range(0, n, 500):
            x = (rd(name, slice(b0, b0 + 500)).astype("float32") - mu) / sd * sign
            g[b0:b0 + 500] = torch.as_tensor(np.nan_to_num(x, nan=0.0), device=dev).half()
        return g
    def finite_mask(*names):
        g = torch.empty((n, H, W), dtype=torch.bool, device=dev)
        for b0 in range(0, n, 500):
            m = np.ones((min(500, n - b0), H, W), dtype=bool)
            for nm in names:
                m &= np.isfinite(rd(nm, slice(b0, b0 + 500)))
            g[b0:b0 + 500] = torch.as_tensor(m, device=dev)
        return g
    sst_name = "sst32" if "sst32" in d else "sst"
    if sst_name == "sst":
        print("WARNING: sst32.npy missing, using the quantised float16 SST", flush=True)
    sub = rd(sst_name, tr[::20]).astype("float32")
    sst_mu, sst_sd = float(np.nanmean(sub)), float(np.nanstd(sub)); del sub
    SSTg, SSTOKg = to_half(sst_name, sst_mu, sst_sd), finite_mask(sst_name)
    lat_desc = lat[0] > lat[-1]; sgn = -1.0 if lat_desc else 1.0
    UCg, VCg, CUROKg = to_half("u_curr"), to_half("v_curr", sign=sgn), finite_mask("u_curr", "v_curr")
    if a.store == "blk":
        wsd = float(np.nanstd(rd("u_wind", tr[::20]).astype("float32"))) + 1e-6
    else:
        wsd = 5.0
    UWg, VWg = to_half("u_wind", 0.0, wsd), to_half("v_wind", 0.0, wsd, sign=sgn)
    if a.lagrangian:
        from .physics import grid_metrics
        dxm, dym, _, _ = grid_metrics(lat, lon)
        CPDXg = T((86400.0 / np.abs(dxm)).astype("float32")); CPDYg = T((86400.0 / np.abs(dym)).astype("float32"))   # cells per day per (m/s)
        gy_, gx_ = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
        BASEg = torch.stack([gx_, gy_], -1)                                                                         # (H, W, 2) normalised coords
    depth_src = "log10(1 + distance to coast km)" if a.store == "blk" else "log10(1 + depth m)"
    store_dir = SB.store_path() if a.store == "blk" else None
    if a.real_depth and store_dir is not None and (store_dir / "depth.npy").exists():
        dep = np.load(store_dir / "depth.npy")[ys, xs].astype("float32")          # m positive down, NaN on land
        depth = np.log10(np.maximum(np.nan_to_num(dep, nan=5.0), 5.0)); depth_src = "log10(max(depth.npy, 5 m))"
    elif a.real_depth:
        print("WARNING: --real-depth but no depth.npy in the store; using the default depth channel", flush=True)
        depth = np.log10(np.maximum(-d["topo"][ys, xs], 0) + 1)
    else:
        depth = np.log10(np.maximum(-d["topo"][ys, xs], 0) + 1)
    DEPTHg = T(((depth - depth[ocean].mean()) / depth[ocean].std()).astype("float32"))
    print(f"depth channel: {depth_src}", flush=True)
    if a.sal_feat:
        sub = rd("so", tr[::20]).astype("float32"); so_mu, so_sd = float(np.nanmean(sub)), float(np.nanstd(sub)) + 1e-6; del sub
        SOg, SOOKg = to_half("so", so_mu, so_sd), finite_mask("so")
        print(f"salinity channel: mean {so_mu:.2f} sd {so_sd:.3f}  valid frac on ocean (every 200th day) {SOOKg[::200][:, grid.ocean].float().mean().item():.3f}", flush=True)
    if a.river_feat:
        rdist = np.load(store_dir / "river_dist.npy")[:, ys, xs].astype("float32")   # (n_mouths, H, W) km
        riv = np.log10(1.0 + np.nanmin(rdist, 0)); riv = np.where(np.isfinite(riv), riv, np.nanmax(riv[ocean]))
        RIVg = T(((riv - riv[ocean].mean()) / riv[ocean].std()).astype("float32")); del rdist
        print(f"river channel: log10(1 + min distance km), {len(json.load(open(store_dir / 'river_names.json')))} mouths", flush=True)
    LATg = T(np.broadcast_to(((lat - lat.mean()) / lat.std()).astype("float32")[:, None], (H, W)).copy())
    offsets = [k for k in range(-a.context, a.context + 1) if k != 0]
    NB = {k: T(T2.neighbour_index(TIMES, k)) for k in offsets}
    print(f"covariates on GPU ({time.time() - t0:.0f} s); sst {sst_name} mean {sst_mu:.2f} sd {sst_sd:.2f}", flush=True)

    # ---- parameters
    n_pix_ch = 11 + (2 if a.sal_feat else 0) + (1 if a.river_feat else 0) + (2 * len(offsets) if a.ctx_feat else 0)      # conv input channels
    feats = SN.MLP_FEATURES + (SN.EXTRA_FEATURES if a.extra_feats else []) + (["so_gx", "so_gy", "so_ok"] if a.sal_feat else []) + (["river"] if a.river_feat else [])
    dmodel = SN.build_dmodel(a.dmodel, g0=a.g0, hidden=a.hidden, d_max=a.d_max, features=feats, in_ch=n_pix_ch, depth=a.depth, dilation=a.dilation, **json.loads(a.dkw)).to(dev)
    log_beta = nn.Parameter(torch.tensor(math.log(a.beta0), device=dev), requires_grad=not a.fix_beta)
    raw_w = nn.Parameter(torch.full((len(offsets),), math.log(math.expm1(a.w0)) if a.w0 > 0 else -10.0, device=dev))
    log_alpha = nn.Parameter(torch.tensor(math.log(max(a.alpha, 1e-8)), device=dev), requires_grad=a.learn_alpha and a.alpha > 0)
    scalars = [log_beta] + ([raw_w] if offsets else []) + ([log_alpha] if a.learn_alpha and a.alpha > 0 else [])
    if a.init_from:
        ck = torch.load(CATEGORIES["spatial"] / a.init_from / "model.pt", map_location=dev)
        dmodel.load_state_dict(ck["dmodel"]); log_beta.data.copy_(ck["log_beta"])
        if "raw_w" in ck and ck["raw_w"].shape == raw_w.shape: raw_w.data.copy_(ck["raw_w"])
    groups = [{"params": scalars, "lr": a.lr, "weight_decay": 0.0}]
    net_params = list(dmodel.parameters())
    if net_params:
        groups.append({"params": net_params, "lr": a.lr_net if a.dmodel in ("mlp", "conv") else a.lr, "weight_decay": a.wd if a.dmodel in ("mlp", "conv") else 0.0})
    opt = torch.optim.AdamW(groups)
    n_params = sum(p.numel() for p in net_params) + sum(p.numel() for p in scalars)
    print(f"dmodel {a.dmodel} params {n_params}  context {offsets}  alpha {'dirichlet' if a.alpha == 0 else a.alpha}", flush=True)

    def nb_vis(bi, k):
        """Neighbour day index j and its visible mask after the donor's cloud sequence."""
        j = NB[k][bi]; ok = (j >= 0).view(-1, 1, 1); j = j.clamp(min=0)
        jd = NB[k][DONg[bi]]; jd = torch.where(jd >= 0, jd, DONg[bi])
        return j, OBSg[j] & ok & OBSg[jd]

    def nb_field(bi, k):
        """Neighbour-day target field and per-pixel weight (float mask); advected to day t with --lagrangian."""
        j, m = nb_vis(bi, k)
        if not a.lagrangian:
            return Yg[j], m.float()
        ok = CUROKg[bi] & grid.ocean
        u = torch.where(ok, UCg[bi].float(), torch.zeros_like(CPDXg)) * CPDXg * k      # x + k u dt (cells)
        v = torch.where(ok, VCg[bi].float(), torch.zeros_like(CPDYg)) * CPDYg * k
        u = u.clamp(-a.lag_max, a.lag_max); v = v.clamp(-a.lag_max, a.lag_max)
        disp = torch.stack([u * (2.0 / (W - 1)), v * (2.0 / (H - 1))], -1)
        g = BASEg.unsqueeze(0) + disp
        mf = m.float().unsqueeze(1)
        num = F.grid_sample((Yg[j] * mf[:, 0]).unsqueeze(1), g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
        den = F.grid_sample(mf, g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
        return num / den.clamp(min=1e-3), den * grid.ocean

    def data_terms(bi, hid):
        """Per-pixel weight map and weighted target: beta clim + alpha vis y_t + sum_k w_k m_k y_k."""
        vis = OBSg[bi] & ~hid
        prior = CLIMg[BINg[bi]]
        beta = log_beta.exp()
        wsum = beta + torch.zeros_like(prior); tsum = beta * prior
        w = F.softplus(raw_w)
        for i, k in enumerate(offsets):
            yk, m = nb_field(bi, k)
            wsum = wsum + w[i] * m; tsum = tsum + w[i] * m * yk
        if a.alpha > 0:
            al = log_alpha.exp() if a.learn_alpha else torch.tensor(a.alpha, device=dev)
            wsum = wsum + al * vis; tsum = tsum + al * vis * Yg[bi]
            free = grid.ocean.expand_as(vis)
        else:
            free = grid.ocean & ~vis
        return wsum, tsum / wsum, vis, free

    def pixel_stack(bi, vis):
        ch = [SSTg[bi].float(), SSTOKg[bi].float(), UCg[bi].float(), VCg[bi].float(), CUROKg[bi].float(), UWg[bi].float(), VWg[bi].float(),
              DEPTHg.expand(len(bi), -1, -1), LATg.expand(len(bi), -1, -1), OCEANf.expand(len(bi), -1, -1), (CLIMg[BINg[bi]] - chl_mu) / chl_sd * OCEANf]
        if a.sal_feat:
            ch += [SOg[bi].float(), SOOKg[bi].float()]
        if a.river_feat:
            ch += [RIVg.expand(len(bi), -1, -1)]
        if a.ctx_feat:
            for k in offsets:
                yk, m = nb_field(bi, k)
                ch += [(yk - CLIMg[BINg[bi]]) / chl_sd * m, m]
        return torch.stack(ch, 1)

    def dtensor(bi, vis):
        if a.dmodel in ("conv", "convsrc") or a.pixel_input:
            return dmodel(pixel_stack(bi, vis))
        f = SN.corner_features(grid, SSTg[bi].float(), SSTOKg[bi], UCg[bi].float(), VCg[bi].float(), CUROKg[bi], UWg[bi].float(), VWg[bi].float(), DEPTHg,
                               clim=(CLIMg[BINg[bi]] - chl_mu) / chl_sd if a.extra_feats else None, lat=LATg if a.extra_feats else None,
                               extra={"river": RIVg.expand(len(bi), -1, -1)} if a.river_feat else None)
        if a.sal_feat:
            sgx, sgy, sok = SN.masked_grad(grid, SOg[bi].float(), SOOKg[bi] & grid.ocean)
            f["so_gx"], f["so_gy"], f["so_ok"] = sgx, sgy, sok.float()
        return dmodel(f)

    def predict(bi, hid):
        wsum, target, vis, free = data_terms(bi, hid)
        Dc = dtensor(bi, vis)
        if a.source:
            Dc, g = Dc
            target = target + log_beta.exp() * g / wsum    # source enters as beta * g: a learned shift of the climatology prior (ln CHL); b = a_p (wsum * target + beta g)
        if a.norm_d:
            Dc = SN.normalise_trace(Dc, grid.corner_ocean)
        pred = SN.steady_state(grid, Dc, wsum, target, Yg[bi] * vis, free, tol=a.tol, maxiter=a.maxiter)
        return pred, Dc

    def grad_stats(acc, pred, bi, hid):
        ob = OBSg[bi]
        for ax in (-1, -2):
            dp = torch.diff(pred, dim=ax); dt_ = torch.diff(Yg[bi], dim=ax)
            m = (torch.narrow(ob, ax, 0, ob.shape[ax] - 1) & torch.narrow(ob, ax, 1, ob.shape[ax] - 1)
                 & (torch.narrow(hid, ax, 0, hid.shape[ax] - 1) | torch.narrow(hid, ax, 1, hid.shape[ax] - 1))).float()
            acc["gp"] += (dp.abs() * m).sum().item(); acc["gt"] += (dt_.abs() * m).sum().item()
            acc["gpt"] += (dp * dt_ * m).sum().item(); acc["gpp"] += (dp * dp * m).sum().item(); acc["gtt"] += (dt_ * dt_ * m).sum().item(); acc["gn"] += m.sum().item()

    @torch.no_grad()
    def evaluate(idx, collect=None):
        dmodel.eval()
        acc = dict(err=0.0, sq=0.0, cnt=0.0, cerr=0.0, gp=0.0, gt=0.0, gpt=0.0, gpp=0.0, gtt=0.0, gn=0.0, it=0.0, nb=0, l1=0.0, l2=0.0, nc=0.0, aniso=[])
        for b in range(0, len(idx), a.batch):
            bi = T(idx[b:b + a.batch]); hid = HIDg[bi]
            pred, Dc = predict(bi, hid)
            acc["it"] += SN.LAST_STATS["fwd_it"]; acc["nb"] += 1
            acc["relmax"] = max(acc.get("relmax", 0.0), SN.LAST_STATS["fwd_rel"]); acc["cap"] = acc.get("cap", 0) + int(SN.LAST_STATS["fwd_it"] >= a.maxiter)
            e = (pred - Yg[bi]) * hid
            acc["err"] += e.abs().sum().item(); acc["sq"] += (e ** 2).sum().item(); acc["cnt"] += hid.sum().item()
            acc["cerr"] += ((CLIMg[BINg[bi]] - Yg[bi]).abs() * hid).sum().item()
            grad_stats(acc, pred, bi, hid)
            l1, l2, _ = SN.tensor_eig(Dc); m = grid.corner_ocean
            acc["l1"] += l1[:, m].sum().item(); acc["l2"] += l2[:, m].sum().item(); acc["nc"] += l1[:, m].numel()
            acc["aniso"].append((l1 / l2.clamp(min=1e-9))[:, m].flatten()[::97].cpu())
            if collect is not None:
                collect[b:b + len(bi)] = pred.cpu().numpy().astype("float16")
        dmodel.train()
        an = torch.cat(acc["aniso"])
        return dict(mae=acc["err"] / max(acc["cnt"], 1), rmse=(acc["sq"] / max(acc["cnt"], 1)) ** 0.5, clim_mae=acc["cerr"] / max(acc["cnt"], 1),
                    n_hidden=int(acc["cnt"]), grad_ratio=acc["gp"] / max(acc["gt"], 1e-9),
                    grad_corr=acc["gpt"] / max((acc["gpp"] * acc["gtt"]) ** 0.5, 1e-9), n_grad=int(acc["gn"]),
                    mean_l1=acc["l1"] / max(acc["nc"], 1), mean_l2=acc["l2"] / max(acc["nc"], 1),
                    aniso_median=float(an.median()), aniso_p90=float(an.quantile(0.9)), cg_iters=acc["it"] / max(acc["nb"], 1), cg_rel=SN.LAST_STATS["fwd_rel"],
                    cg_rel_max=acc.get("relmax", 0.0), cg_cap_batches=acc.get("cap", 0), cg_batches=acc["nb"])

    def params_summary():
        s = dict(beta=log_beta.detach().exp().item())
        if offsets:
            s["w"] = {str(k): round(v, 4) for k, v in zip(offsets, F.softplus(raw_w).detach().tolist())}
        if a.alpha > 0:
            s["alpha"] = log_alpha.detach().exp().item() if a.learn_alpha else a.alpha
        if a.dmodel == "sst":
            s.update(d0=dmodel.log_d0.detach().exp().item(), g0=dmodel.log_g0.detach().exp().item())
        return s

    hv = hid_va.reshape(len(va), -1).sum(1)
    fig_day = va[np.argsort(hv)[int(0.75 * len(va))]]
    ext = [lon[0], lon[-1], lat[-1], lat[0]] if lat_desc else [lon[0], lon[-1], lat[0], lat[-1]]

    @torch.no_grad()
    def diagnostic_figure(tag):
        dmodel.eval()
        bi = T(np.array([fig_day])); hid = HIDg[bi]
        pred, Dc = predict(bi, hid)
        dmodel.train()
        pred = pred[0].cpu().numpy(); pred[~ocean] = np.nan
        truth = y[fig_day].copy(); vis = obs[fig_day] & ~hidden[fig_day]
        inp = np.where(vis, truth, np.nan)
        err = np.where(hidden[fig_day], pred - truth, np.nan)
        l1, l2, phi = SN.tensor_eig(Dc)
        ratio = (l1 / l2.clamp(min=1e-9))[0].cpu().numpy(); ratio[~grid.corner_ocean.cpu().numpy()] = np.nan
        sst = SSTg[fig_day].float().cpu().numpy() * sst_sd + sst_mu; sst[~SSTOKg[fig_day].cpu().numpy() | ~ocean] = np.nan
        fig, ax = plt.subplots(2, 3, figsize=(17, 9))
        date = str(TIMES[fig_day])[:10]
        for a_, img, t, cm, vr in [(ax[0, 0], truth, f"ln CHL L3 observed, {date}", MAP_CMAP, (-3, 2)),
                                   (ax[0, 1], inp, "Model input: visible pixels (fake clouds applied)", MAP_CMAP, (-3, 2)),
                                   (ax[0, 2], pred, f"Steady-state fill ({a.dmodel} D, context {a.context})", MAP_CMAP, (-3, 2)),
                                   (ax[1, 0], err, "Fill minus truth on hidden pixels (ln CHL)", ERR_CMAP, (-1, 1)),
                                   (ax[1, 1], np.log10(ratio), "log10 anisotropy ratio l1/l2 of D at corners", "viridis", (0, 2)),
                                   (ax[1, 2], sst, "SST (K) with isotherms", "viridis", (None, None))]:
            im = a_.imshow(img, extent=ext, cmap=cm, vmin=vr[0], vmax=vr[1], origin="upper" if lat_desc else "lower")
            a_.set_title(t, fontsize=10); a_.set_xlabel("lon"); a_.set_ylabel("lat"); fig.colorbar(im, ax=a_, shrink=0.8)
        if np.isfinite(sst).any():
            lv = np.nanpercentile(sst, np.linspace(5, 95, 10))
            for a_ in (ax[0, 2], ax[1, 2]):
                a_.contour(lon, lat, sst, levels=lv, colors="white" if a_ is ax[0, 2] else "black", linewidths=0.5, alpha=0.8)
        fig.suptitle(f"{a.run}: {tag}; params {json.dumps(params_summary())}", fontsize=10)
        fig.savefig(out / "figures" / f"{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    json.dump(dict(vars(a), sst_source=sst_name, depth_source=depth_src, n_pix_ch=n_pix_ch, n_train=len(tr), n_val=len(va), n_params=n_params, val_years=list(val_years), test_years=list(test_years), grid=[H, W], n_ocean=int(ocean.sum())), open(out / "config.json", "w"), indent=1)
    va_eval = va[::a.val_stride]
    hist = dict(train=[], val=[], val_rmse=[], grad_ratio=[], grad_corr=[], params=[], epoch_time=[], lr=[], cg=[])
    cgt = dict(n=0, fwd_it=0, bwd_it=0, fwd_cap=0, bwd_cap=0, fwd_rel=0.0, bwd_rel=0.0)   # training CG stats of the current epoch
    def log_eval(tag, ev, ltr=None, dt=0.0):
        cg = dict(eval_it=ev["cg_iters"], eval_rel_max=ev["cg_rel_max"], eval_cap=ev["cg_cap_batches"], eval_batches=ev["cg_batches"], **{f"train_{k}": v for k, v in cgt.items()})
        for k, v in dict(train=ltr, val=ev["mae"], val_rmse=ev["rmse"], grad_ratio=ev["grad_ratio"], grad_corr=ev["grad_corr"], params=params_summary(), epoch_time=dt, lr=opt.param_groups[-1]["lr"], cg=cg).items():
            hist[k].append(v)
        print(f"{tag}  train {ltr if ltr is None else round(ltr, 4)}  val MAE {ev['mae']:.4f} rmse {ev['rmse']:.4f} clim {ev['clim_mae']:.4f}  grad ratio {ev['grad_ratio']:.3f} corr {ev['grad_corr']:.3f}  "
              f"aniso median {ev['aniso_median']:.2f} p90 {ev['aniso_p90']:.2f}  cg {ev['cg_iters']:.0f} (cap {ev['cg_cap_batches']}/{ev['cg_batches']}, rel max {ev['cg_rel_max']:.1e})  "
              f"train cg fwd {cgt['fwd_it'] / max(cgt['n'], 1):.0f} cap {cgt['fwd_cap']}/{cgt['n']} rel max {cgt['fwd_rel']:.1e}, bwd {cgt['bwd_it'] / max(cgt['n'], 1):.0f} cap {cgt['bwd_cap']}/{cgt['n']} rel max {cgt['bwd_rel']:.1e}  "
              f"params {json.dumps(params_summary())}  ({dt:.0f} s)", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
    if a.dump_days:
        (out / "dumps").mkdir(exist_ok=True)
        dmodel.eval()
        with torch.no_grad():
            for i in [int(x) for x in a.dump_days.split(",")]:
                bi = T(np.array([i])); hid = HIDg[bi]
                wsum, target, vis, free = data_terms(bi, hid)
                Dc = dtensor(bi, vis)
                if a.source:
                    Dc, g = Dc; target = target + log_beta.exp() * g / wsum
                if a.norm_d:
                    Dc = SN.normalise_trace(Dc, grid.corner_ocean)
                pred = SN.steady_state(grid, Dc, wsum, target, Yg[bi] * vis, free, tol=a.tol, maxiter=a.maxiter)
                np.savez_compressed(out / "dumps" / f"day_{i}.npz", Dc=Dc[0].cpu().numpy(), wsum=wsum[0].cpu().numpy(), target=target[0].cpu().numpy(),
                                    fixed=(Yg[bi] * vis)[0].cpu().numpy(), free=free[0].cpu().numpy(), vis=vis[0].cpu().numpy(), truth=y[i], obs=obs[i], hidden=hidden[i],
                                    clim=CLIMg[BINg[bi]][0].cpu().numpy(), pred=pred[0].cpu().numpy(), sst=(SSTg[i].float() * sst_sd + sst_mu).cpu().numpy(), sst_ok=SSTOKg[i].cpu().numpy(),
                                    ocean=ocean, lat=lat, lon=lon, time=str(TIMES[i])[:10], area_c=grid.area_c.cpu().numpy(), area_p=grid.area_p.cpu().numpy())
                print(f"dumped day {i} {str(TIMES[i])[:10]}", flush=True)
        dmodel.train()
    ev = evaluate(va_eval); log_eval("init", ev, dt=time.time() - t0); diagnostic_figure("init")
    best = ev["mae"]
    def save():
        torch.save(dict(dmodel=dmodel.state_dict(), log_beta=log_beta.detach().cpu(), raw_w=raw_w.detach().cpu(), log_alpha=log_alpha.detach().cpu()), out / "model.pt")
    save()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs) if (a.sched == "cosine" and a.epochs > 0) else None
    if not a.eval_only:
        for ep in range(a.epochs):
            te0 = time.time()
            days = tr[::a.train_stride]
            donors = tr[C.seasonal_donor_indices(TIMES[tr], rng, a.donor_window)] if a.donor_window > 0 else tr[C.donor_indices(len(tr), rng)]
            HIDg[T(tr)] = T(obs[tr] & ~obs[donors]); DONg[T(tr)] = T(donors)
            order = rng.permutation(days)
            tot = cnt = 0.0
            for k in cgt: cgt[k] = 0
            for b in range(0, len(order), a.batch):
                bi = T(order[b:b + a.batch]); hid = HIDg[bi]
                if hid.sum() == 0:
                    continue
                pred, _ = predict(bi, hid)
                loss = ((pred - Yg[bi]).abs() * hid).sum() / hid.sum()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net_params + scalars, 10.0)
                opt.step()
                tot += loss.item() * len(bi); cnt += len(bi)
                st = SN.LAST_STATS; cgt["n"] += 1; cgt["fwd_it"] += st["fwd_it"]; cgt["bwd_it"] += st.get("bwd_it", 0)
                cgt["fwd_cap"] += int(st["fwd_it"] >= a.maxiter); cgt["bwd_cap"] += int(st.get("bwd_it", 0) >= a.maxiter)
                cgt["fwd_rel"] = max(cgt["fwd_rel"], st["fwd_rel"]); cgt["bwd_rel"] = max(cgt["bwd_rel"], st.get("bwd_rel", 0.0))
                if (b // a.batch) % 100 == 0:
                    print(f"  ep {ep} batch {b // a.batch}/{len(order) // a.batch} loss {loss.item():.4f} cg {SN.LAST_STATS['fwd_it']}/{SN.LAST_STATS.get('bwd_it', 0)} rel {SN.LAST_STATS['fwd_rel']:.1e}/{SN.LAST_STATS.get('bwd_rel', 0.0):.1e} params {json.dumps(params_summary())}", flush=True)
            if sched: sched.step()
            ev = evaluate(va_eval); log_eval(f"ep {ep}", ev, tot / max(cnt, 1), time.time() - te0)
            if ev["mae"] < best:
                best = ev["mae"]; save()
            diagnostic_figure(f"epoch_{ep:03d}")
    preds = np.empty((len(va_eval), H, W), dtype="float16") if a.save_preds else None
    ev = evaluate(va_eval, preds)
    json.dump(dict(val=ev, best_val_mae=best, params=params_summary(), val_days=len(va_eval), minutes=(time.time() - t0) / 60), open(out / "metrics.json", "w"), indent=1)
    if preds is not None:
        np.savez_compressed(out / "preds.npz", val_idx=va_eval, val_pred=preds)
    print(f"VAL hidden-pixel MAE {ev['mae']:.4f} (best {best:.4f})  rmse {ev['rmse']:.4f}  clim {ev['clim_mae']:.4f}  grad ratio {ev['grad_ratio']:.3f} corr {ev['grad_corr']:.3f}  n {ev['n_hidden']}")
    print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
