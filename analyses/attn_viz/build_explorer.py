"""Explorer cache for notebooks/attention_explorer.ipynb (2026-10-24): weights, predictions, context fields and input attributions of one
IO gap-filler on a fixed preset of 12 val + 12 test days. CPU only (CUDA hidden). Data generation only.

Preset days (analyses/attn_viz/explorer_days.json, written once by --choose-days, the same for every run): per split, the days with
>= 500 hidden px are binned by season (DJF, MAM, JJA, SON) x tercile of the day's observed fraction (store-observed / ocean px,
terciles over the split); one day per bin, the one closest to the bin's median observed fraction (4 x 3 = 12 days per split). Val
always includes the 3 dates of the io3 weight-map figures (2001-03-02, 2001-10-29, 2007-06-27); they fill their own bins.
Hidden pixels are the trainer's fixed val / test masks (clouds.fixed_hidden, --eval-seed 0, no donor exclusion: every run here).

How the model is run: chl_cnn.train_pinn.main(--eval-only --init-from RUN) with the run's own arguments (build_cache.argv_from_config)
and --val-stride 100000 (1 val + 1 test day scored, hidden masks unchanged: they are drawn over the full split first). A wrapper
around train_pinn.climatology_reference (called once, after scoring) keeps main()'s local variables: the model with its weights and
the trainer's own input builders (make_inputs_3d, day_ok, clim_window, add_base, predict_z). Each preset day is then run through
those, one day per forward, with the existing recorders (analyses/io3_attn/attn_eval.Recorder, ptok_eval.Rec,
analyses/io4_tokdec/tokdec_eval.Rec). tokdec.py's hook does not give the blend score terms; TokdecRec recomputes them from the
recorded output tokens with the module's own layers (and checks that their softmax reproduces the recorded blend weights).

Attribution: integrated gradients (IG) of f = mean predicted ln CHL of day t over day t's hidden pixels, with respect to the input
groups below; baseline = every attributed input at 0 (the training mean in z-space), the visible-mask channels held at their real
values (not attributed: a baseline of "no pixel visible" would change which days the stems can weight). Path integral by
Gauss-Legendre quadrature on [0, 1] (--ig-steps nodes, default 32). Completeness: sum of all attributions vs f(x) - f(baseline),
stored per day (ig_err, ln CHL). Groups (per-day channels count on every window day; context = all days t != 0):
  context CHL      CHL value channel (z, 0 where not visible) of the days t != 0
  day-t CHL        CHL value channel of day t (visible pixels)
  SST, salinity, MLD, SLA, currents (u, v), winds (u, v), air temperature   the physics channels of every window day
  static           the 7 static channels (DOY climatology of day t, lat, lon, depth, sin / cos DOY, year); for patch / tokdec stems
                   also clim3d, the DOY climatology of every window day (their anomaly input)
Stored: attr_abs (n, G) sum of |attribution| over each group's elements, attr_sum (n, G) signed sums, attr_map (n, G, H, W) |attribution|
summed over each group's channels and days. ptok.py's patch std (sqrt of a clamped variance, x 0 on patches with <= 1 visible px)
has an infinite derivative at 0; during IG ptok's torch.sqrt is replaced by one with the same forward and a derivative capped at
1 / (2 sqrt(1e-12)) (exact elsewhere), so those patches give 0 instead of NaN.

Saved to analyses/attn_viz/explorer_cache/<run>.npz (float16 maps, NaN on land / unobserved; compressed; the old caches in cache/ are
untouched): dates, days (store index), split, ocean, lat, lon, offsets, t_index, features, groups; truth (ln CHL where observed), vis
(n, T, H, W visible in the stem input, donor-masked), hid (hidden px), pred (ln CHL), mae / n_hid (hidden px of day t, from float32),
sst (deg C, store K - 273.15), u_curr / v_curr (m/s); per stem: attn w (n, T, H, W); patch w (n, T + 1, H, W, last = "no day"), s_patch, s_pixel (n, T,
H, W), conf (lavg); tokdec pool_w / visfrac (n, Np, T) with patch_idx, nPy, nPx, P, dec (n, 4, H, W) = the 4 decoded channels with the
highest ocean variance over the 24 days (dec_channels), and with --tokdec-blend w, s_patch, s_pixel; attribution arrays above, f_x,
f_base, ig_err; pred_check = max |CPU pred - the run's saved preds.npz (GPU, AMP)| over ocean px per day.

  python -m analyses.attn_viz.build_explorer --choose-days                       # once: writes explorer_days.json
  python -m analyses.attn_viz.build_explorer --run io4_tokdec                    # one run (~14 GB RAM; run them one at a time)
  python -m analyses.attn_viz.build_explorer --run io4_tokdec --days-limit 1 --ig-steps 8 --out-dir /tmp/x   # quick test
"""
import argparse
import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                    # CPU only: hide the GPU before torch is imported

import json
import math
import shutil
import tempfile
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "explorer_cache"
DAYS_JSON = HERE / "explorer_days.json"
RUNS = ("io2_tattn8", "io3_ptok4", "io3_ptok4_lavg", "io4_tokdec", "io4_tokdec_blend", "t3d_dil_c4_60ep_gdl100")
FORCED_VAL = ("2001-03-02", "2001-10-29", "2007-06-27")
SEASONS = ("DJF", "MAM", "JJA", "SON")
GROUPS = ("context CHL", "day-t CHL", "SST", "salinity", "MLD", "SLA", "currents", "winds", "air temperature", "static")
FEAT_GROUP = dict(sst="SST", so="salinity", mlotst="MLD", sla="SLA", u_curr="currents", v_curr="currents", u_wind="winds",
                  v_wind="winds", air_temp="air temperature")
MIN_HID = 500
N_DEC = 4


# ------------------------------------------------------------------ preset days
def season_of(dates):
    m = np.asarray(dates).astype("datetime64[M]").astype(int) % 12 + 1
    return np.array([SEASONS[(mm % 12) // 3] for mm in m])


def choose_days(out=DAYS_JSON):
    from chl_cnn import viz, clouds as C
    ctx = viz.load_context("io")
    res = dict(method=__doc__.split("\n\n")[1].strip(), min_hidden_px=MIN_HID, splits={})
    for sp, name, seed in (("va", "val", 0), ("te", "test", 1)):
        idx = ctx[sp]
        obs = np.zeros((len(ctx["times"]),) + ctx["ocean"].shape, dtype=bool)
        for i in idx:
            obs[i] = ctx["obs"](int(i))
        hid, _ = C.fixed_hidden(obs, idx, seed=seed)
        nh = hid.reshape(len(idx), -1).sum(1)
        of = ctx["obs_frac"][idx]
        dates = ctx["times"][idx].astype("datetime64[D]").astype(str)
        ok = nh >= MIN_HID
        cuts = np.percentile(of[ok], [100 / 3, 200 / 3])
        terc = np.digitize(of, cuts)                           # 0 low, 1 mid, 2 high
        sea = season_of(dates)
        pick = {}
        if name == "val":
            for d in FORCED_VAL:
                k = int(np.flatnonzero(dates == d)[0])
                cell = (sea[k], int(terc[k]))
                assert cell not in pick, f"forced dates share bin {cell}"
                pick[cell] = k
        for s in SEASONS:
            for t in range(3):
                if (s, t) in pick:
                    continue
                cand = np.flatnonzero(ok & (sea == s) & (terc == t))
                med = np.median(of[cand])
                pick[(s, t)] = int(cand[np.argmin(np.abs(of[cand] - med))])
        ks = sorted(pick.values(), key=lambda k: dates[k])
        cell_of = {k: c for c, k in pick.items()}
        res["splits"][name] = [dict(date=str(dates[k]), store_index=int(idx[k]), obs_frac=round(float(of[k]), 4), n_hidden=int(nh[k]),
                                    season=cell_of[k][0], obs_tercile=["low", "mid", "high"][cell_of[k][1]],
                                    forced=bool(dates[k] in FORCED_VAL and name == "val")) for k in ks]
        res["splits"][name + "_tercile_cuts"] = [round(float(c), 4) for c in cuts]
        res["splits"][name + "_n_days_ok"] = int(ok.sum())
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["splits"], indent=1))
    return res


def preset_days():
    d = json.loads(DAYS_JSON.read_text())["splits"]
    rows = [("val", r) for r in d["val"]] + [("test", r) for r in d["test"]]
    return [r["store_index"] for _, r in rows], [r["date"] for _, r in rows], [s for s, _ in rows]


# ------------------------------------------------------------------ trainer capture
def run_trainer(run, threads=6):
    """chl_cnn.train_pinn.main(--eval-only) on CPU for RUN (1 val + 1 test day scored); returns main()'s local variables."""
    import torch
    from chl_cnn import train_pinn
    from chl_cnn.runs import find_run
    from .build_cache import argv_from_config, cpu_torch, SKIP
    assert not torch.cuda.is_available(), "CPU only: run as python -m analyses.attn_viz.build_explorer (CUDA hidden)"
    torch.set_num_threads(threads)
    src = find_run(run)
    cfg = json.loads((src / "config.json").read_text())
    assert cfg.get("model") == "temporal3d" and cfg.get("store", "io") == "io", f"{run}: not an IO temporal3d run"
    argv = ["--run", f"explorer_{run}"] + argv_from_config(cfg) + ["--eval-only", "--init-from", run, "--val-stride", "100000",
                                                                  "--eval-batch", "1", "--fig-every", "0"]
    ns = train_pinn.parse(argv)
    diff = {k: (cfg[k], v) for k, v in vars(ns).items() if k in cfg and cfg[k] is not None and k not in SKIP and cfg[k] != v
            and not (k == "dilations" and (v is None or v == ",".join(map(str, cfg[k]))))}
    assert not diff, f"rebuilt trainer arguments differ from config.json: {diff}"
    tmp = Path(tempfile.mkdtemp(prefix=f"explorer_{run}_"))
    box = {}
    orig_cr = train_pinn.climatology_reference

    def grab(*args, **kw):
        box.update(sys._getframe(1).f_locals)                  # main()'s locals (cell variables included)
        return orig_cr(*args, **kw)

    saved = dict(run_dir=train_pinn.run_dir, torch=train_pinn.torch)
    train_pinn.run_dir = lambda name, trainer, model: tmp / name
    train_pinn.torch = cpu_torch()
    train_pinn.climatology_reference = grab
    print(f"{run}: trainer args {' '.join(argv)}", flush=True)
    try:
        train_pinn.main(argv)
    finally:
        train_pinn.run_dir, train_pinn.torch = saved["run_dir"], saved["torch"]
        train_pinn.climatology_reference = orig_cr
        shutil.rmtree(tmp, ignore_errors=True)
    assert "model" in box, "climatology_reference was not called from main()"
    box["_cfg"], box["_src"] = cfg, src
    return box


# ------------------------------------------------------------------ recorders
class TokdecRec:
    """tokdec_eval.Rec plus the --tokdec-blend score terms (patch score interpolated, pixel term) recomputed from the recorded output
    tokens with the module's own layers, as in chl_cnn/tokdec.py TokDecNet.forward."""

    def __init__(self):
        from analyses.io4_tokdec.tokdec_eval import Rec
        self.rec = Rec([0], 1)
        self.extra = {}

    def __call__(self, d):
        import torch
        self.rec(d)
        m = d["net"]
        if not m.use_blend or self.extra:
            return
        x3d, tok = d["x3d"], d["tok"]
        B, C, T, H, W = x3d.shape
        vis = x3d[:, -1].float() > 0.5
        s = torch.einsum("bpd,bptd->bpt", m.wq(tok[:, :, m.ti]), m.wk(tok)) / math.sqrt(m.D)
        s_pix = m._upsample(m._to_grid(s.float()))
        offs = torch.arange(T, dtype=torch.float32) - m.ti
        K = max(m.ti, T - 1 - m.ti)
        sst = x3d[:, m.sst].float()
        pf = torch.stack([vis.float(), (offs.abs() / K).view(1, T, 1, 1).expand(B, T, H, W), (offs / K).view(1, T, 1, 1).expand(B, T, H, W),
                          (sst - sst[:, m.ti:m.ti + 1]).abs()], 2)
        p_term = m.pix(pf.view(B * T, 4, H, W)).view(B, T, H, W).float()
        ctx = torch.ones(T, dtype=torch.bool); ctx[m.ti] = False
        logit = (s_pix + p_term).masked_fill(~(vis & ctx.view(1, T, 1, 1)), float("-inf"))
        w = torch.softmax(torch.cat([logit, m.none_logit.float().view(1, 1, 1, 1).expand(B, 1, H, W)], 1), 1)
        self.extra = dict(s_patch=s_pix[0].detach().numpy(), s_pixel=p_term[0].detach().numpy(),
                          w_check=float((w - d["w"].float()).abs().max()))


class _SafeSqrt:
    """Stand-in for the torch module inside chl_cnn.ptok during IG: sqrt with the same forward and a derivative capped at x = 1e-12."""

    def __init__(self, torch):
        self._t = torch

        class F_(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.sqrt(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, g):
                (y,) = ctx.saved_tensors
                return g / (2 * y.clamp(min=1e-6))
        self.sqrt = F_.apply

    def __getattr__(self, k):
        return getattr(self._t, k)


# ------------------------------------------------------------------ build
def build(run, ig_steps=32, ig_batch=4, threads=6, days_limit=None, out=None):
    import torch
    from chl_cnn import memory as MEM, viz, ptok, tokdec, tattn
    L = run_trainer(run, threads)
    a, cfg, src = L["a"], L["_cfg"], L["_src"]
    model = L["model"].eval()
    for p in model.parameters():
        p.requires_grad_(False)
    stem = a.t3d_stem or "conv"
    mk3d, day_ok, clim_window, add_base, predict_z = L["make_inputs_3d"], L["day_ok"], L["clim_window"], L["add_base"], L["predict_z"]
    HIDg, OBSg, Yg, Tt = L["HIDg"], L["OBSg"], L["Yg"], L["T"]
    chl_mu, chl_sd, ocean, obs, rd = L["chl_mu"], L["chl_sd"], L["ocean"].astype(bool), L["obs"], L["rd"]
    TIMES, va, te = L["TIMES"], L["va"], L["te"]
    win_offs = list(L["win_offs"]); ti = win_offs.index(0); T = len(win_offs)
    feats = list(a.features); nF = len(feats)
    days, dates, split = preset_days()
    if days_limit:
        keep = list(range(min(days_limit, 12))) + list(range(12, 12 + min(days_limit, 12)))
        days, dates, split = [days[k] for k in keep], [dates[k] for k in keep], [split[k] for k in keep]
    vset, tset = set(va.tolist()), set(te.tolist())
    for d_, dt_, sp in zip(days, dates, split):
        assert str(TIMES[d_])[:10] == dt_ and d_ in (vset if sp == "val" else tset), (d_, dt_, sp)
    n = len(days)
    print(f"{run}: stem {stem}, T {T}, {n} preset days, IG {ig_steps} Gauss-Legendre nodes, batch {ig_batch}", flush=True)

    def forward(x3d, xs, dok, clim, bi):
        if stem in ("patch", "tokdec"):
            z = model(x3d, xs, dok, clim)
        elif stem == "attn":
            z = model(x3d, xs, dok, xs[:, -MEM.N_CH:] if a.attn_query_memory else None)
        else:
            z = model(x3d, xs)
        return add_base(z, bi)

    # group index of every (channel, day) of x3d; -1 = not attributed (visible mask)
    gi = {g: k for k, g in enumerate(GROUPS)}
    grp3 = np.full((len(feats) + 2, T), -1)
    for c, f in enumerate(feats):
        grp3[c] = gi[FEAT_GROUP[f]]
    grp3[nF] = gi["context CHL"]; grp3[nF, ti] = gi["day-t CHL"]
    attr_ch = torch.ones(nF + 2); attr_ch[-1] = 0
    gl_x, gl_w = np.polynomial.legendre.leggauss(ig_steps)
    alphas, wts = (gl_x + 1) / 2, gl_w / 2

    pr = viz.load_preds(src)
    R = {k: [] for k in ("truth", "vis", "hid", "pred", "mae", "n_hid", "sst", "u_curr", "v_curr", "pred_check", "predz_check", "attr_abs",
                         "attr_sum", "attr_map", "f_x", "f_base", "ig_err", "ig_nonfinite", "w", "s_patch", "s_pixel", "conf", "pool_w", "visfrac",
                         "dec", "w_check", "sec")}
    meta = {}
    for k_, day in enumerate(days):
        t0 = time.time()
        bi = Tt(np.array([day])); hid = HIDg[bi]
        hidm = hid[0].float(); nh = int(hidm.sum())
        with torch.no_grad():
            x3d, xs = mk3d(bi, hid)
            dok = day_ok(bi)
            clim = clim_window(bi) if stem in ("patch", "tokdec") else None
            # recorders on for this forward only
            if stem == "attn":
                from analyses.io3_attn.attn_eval import Recorder
                rec = Recorder(np.array([day]), obs, ocean, 1); tattn.EVAL_HOOKS["recorder"] = rec
            elif stem == "patch":
                from analyses.io3_attn.ptok_eval import Rec
                rec = Rec([0], 1); ptok.HOOKS["recorder"] = rec
            elif stem == "tokdec":
                rec = TokdecRec(); tokdec.HOOKS["recorder"] = rec
            try:
                pz = forward(x3d, xs, dok, clim, bi)
            finally:
                tattn.EVAL_HOOKS["recorder"] = None; ptok.HOOKS["recorder"] = None; tokdec.HOOKS["recorder"] = None
            pz_ref = predict_z(model, bi, hid)
        pred = (pz[0, 1] * chl_sd + chl_mu).numpy()
        R["predz_check"].append(float((pz - pz_ref).abs().max()))
        y = Yg[day].numpy(); ob = OBSg[day].numpy()
        hn = hid[0].numpy()
        R["mae"].append(float(np.abs(pred - y)[hn].mean()) if nh else np.nan); R["n_hid"].append(nh)
        R["truth"].append(np.where(ob, y, np.nan)); R["hid"].append(hn); R["pred"].append(np.where(ocean, pred, np.nan))
        R["vis"].append((x3d[0, -1] > 0.5).numpy())
        R["pred_check"].append(float(np.nanmax(np.abs(pr["pred"][pr["pos"][int(day)]].astype("float32") - pred)[ocean])))
        for nm in ("sst", "u_curr", "v_curr"):                 # store SST is in K
            R[nm].append(np.where(ocean, rd(nm, day).astype("float32") - (273.15 if nm == "sst" else 0.0), np.nan))
        if stem == "attn":
            R["w"].append(rec.maps[0].astype("float32"))
            assert np.array_equal(rec.hid[0], hn) and np.array_equal(rec.vis[0], R["vis"][-1])
        elif stem == "patch":
            o = rec.out[0]
            R["w"].append(o["w"]); R["s_patch"].append(o["s_patch"]); R["s_pixel"].append(o["s_pixel"])
            if "conf" in o:
                R["conf"].append(o["conf"])
        elif stem == "tokdec":
            r0 = rec.rec
            o = r0.out[0]
            R["pool_w"].append(r0.pool[0][0]); R["visfrac"].append(r0.visf[0][0]); R["dec"].append(o["dec"])
            meta.update(r0.meta)
            if "w" in o:
                R["w"].append(o["w"]); R["s_patch"].append(rec.extra["s_patch"]); R["s_pixel"].append(rec.extra["s_pixel"])
                R["w_check"].append(rec.extra["w_check"])

        # ---- integrated gradients
        f_hid = lambda out: (out[:, 1] * chl_sd * hidm).sum((1, 2)) / max(nh, 1)
        G3 = torch.zeros_like(x3d[0]); Gs = torch.zeros_like(xs[0]); Gc = torch.zeros_like(clim[0]) if clim is not None else None
        saved_torch = ptok.torch
        ptok.torch = _SafeSqrt(torch)
        try:
            with torch.no_grad():
                f_x = float(f_hid(forward(x3d, xs, dok, clim, bi))[0])
                f_b = float(f_hid(forward(x3d * attr_ch.view(1, -1, 1, 1, 1) * 0 + x3d * (1 - attr_ch).view(1, -1, 1, 1, 1), xs * 0, dok,
                                          None if clim is None else clim * 0, bi))[0])
            for s0 in range(0, ig_steps, ig_batch):
                al = torch.tensor(alphas[s0:s0 + ig_batch], dtype=torch.float32); ww = torch.tensor(wts[s0:s0 + ig_batch], dtype=torch.float32)
                nb = len(al)
                s3 = al.view(nb, 1, 1, 1, 1) * attr_ch.view(1, -1, 1, 1, 1) + (1 - attr_ch).view(1, -1, 1, 1, 1)
                X3 = (x3d * s3).requires_grad_(True)
                XS = (xs * al.view(nb, 1, 1, 1)).requires_grad_(True)
                CL = (clim * al.view(nb, 1, 1, 1)).requires_grad_(True) if clim is not None else None
                f = f_hid(forward(X3, XS, dok.expand(nb, -1), CL, bi.expand(nb)))
                g = torch.autograd.grad(f.sum(), [X3, XS] + ([CL] if CL is not None else []))
                G3 += (g[0] * ww.view(nb, 1, 1, 1, 1)).sum(0); Gs += (g[1] * ww.view(nb, 1, 1, 1)).sum(0)
                if CL is not None:
                    Gc += (g[2] * ww.view(nb, 1, 1, 1)).sum(0)
        finally:
            ptok.torch = saved_torch
        A3 = (x3d[0] * G3 * attr_ch.view(-1, 1, 1, 1)).numpy().astype("float64")     # (C, T, H, W)
        As = (xs[0] * Gs).numpy().astype("float64")
        Ac = (clim[0] * Gc).numpy().astype("float64") if clim is not None else np.zeros((0,) + As.shape[1:])
        nonfin = int((~np.isfinite(A3)).sum() + (~np.isfinite(As)).sum() + (~np.isfinite(Ac)).sum())
        A3, As, Ac = (np.nan_to_num(v) for v in (A3, As, Ac))
        ab, sg, mp = np.zeros(len(GROUPS)), np.zeros(len(GROUPS)), np.zeros((len(GROUPS),) + As.shape[1:])
        for g_ in range(len(GROUPS)):
            if GROUPS[g_] == "static":
                parts = [As, Ac]
            else:
                sel = grp3 == g_
                parts = [A3[sel]]
            for p_ in parts:
                if len(p_):
                    ab[g_] += np.abs(p_).sum(); sg[g_] += p_.sum(); mp[g_] += np.abs(p_).sum(0)
        R["attr_abs"].append(ab); R["attr_sum"].append(sg); R["attr_map"].append(np.where(ocean, mp, np.nan))
        R["f_x"].append(f_x); R["f_base"].append(f_b); R["ig_err"].append(sg.sum() - (f_x - f_b)); R["ig_nonfinite"].append(nonfin)
        R["sec"].append(time.time() - t0)
        print(f"  {split[k_]} {dates[k_]}: hidden {nh} px, MAE {R['mae'][-1]:.4f}, |pred - saved| {R['pred_check'][-1]:.4f}, "
              f"f(x) {f_x:.4f} f(base) {f_b:.4f} sum attr {sg.sum():.4f} (err {R['ig_err'][-1]:+.2e}), nonfinite {nonfin}, "
              f"{R['sec'][-1]:.0f} s", flush=True)

    f16 = lambda v: np.stack(v).astype("float16")
    res = dict(run=run, stem=stem, value=cfg.get("ptok_value") or "", blend=bool(cfg.get("tokdec_blend")), days=np.array(days),
               dates=np.array(dates), split=np.array(split), ocean=ocean, lat=L["lat"], lon=L["lon"], t_index=ti, offsets=np.array(win_offs),
               features=np.array(feats), groups=np.array(GROUPS), ig_steps=ig_steps, ig_method="integrated gradients, Gauss-Legendre",
               f_units="ln CHL (mean over day t hidden px)", chl_mu=chl_mu, chl_sd=chl_sd,
               truth=f16(R["truth"]), vis=np.stack(R["vis"]), hid=np.stack(R["hid"]), pred=f16(R["pred"]), mae=np.array(R["mae"]),
               n_hid=np.array(R["n_hid"]), sst=f16(R["sst"]), u_curr=f16(R["u_curr"]), v_curr=f16(R["v_curr"]),
               pred_check=np.array(R["pred_check"]), predz_check=np.array(R["predz_check"]), attr_abs=np.stack(R["attr_abs"]),
               attr_sum=np.stack(R["attr_sum"]), attr_map=f16(R["attr_map"]), f_x=np.array(R["f_x"]), f_base=np.array(R["f_base"]),
               ig_err=np.array(R["ig_err"]), ig_nonfinite=np.array(R["ig_nonfinite"]), sec_per_day=np.array(R["sec"]))
    oc_nan = lambda v: np.where(ocean, np.stack(v).astype("float32"), np.nan).astype("float16")
    for k in ("w", "s_patch", "s_pixel", "conf"):
        if R[k]:
            res[k] = oc_nan(R[k])
    if R["w_check"]:
        res["w_check"] = np.array(R["w_check"])
    if stem == "tokdec":
        dec = np.stack(R["dec"])                               # (n, 32, H, W)
        var = np.array([dec[:, c][:, ocean].var(1).mean() for c in range(dec.shape[1])])
        ch = np.sort(np.argsort(var)[::-1][:N_DEC])
        res.update(pool_w=np.stack(R["pool_w"]).astype("float16"), visfrac=np.stack(R["visfrac"]).astype("float16"), dec=oc_nan(list(dec[:, ch])),
                   dec_channels=ch, dec_n_channels=dec.shape[1], **meta)
    out = Path(out or OUT / f"{run}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); max |pred - saved preds| {max(R['pred_check']):.4f} ln CHL; "
          f"IG |err| max {np.abs(res['ig_err']).max():.2e}, median relative {np.median(np.abs(res['ig_err']) / np.maximum(np.abs(res['f_x'] - res['f_base']), 1e-9)):.2e}; "
          f"{np.sum(R['sec']) / 60:.1f} min", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--choose-days", action="store_true", help="write explorer_days.json (preset days) and exit")
    p.add_argument("--run", nargs="+", default=[])
    p.add_argument("--ig-steps", type=int, default=32, help="Gauss-Legendre nodes of the IG path integral")
    p.add_argument("--ig-batch", type=int, default=4, help="IG path points per forward")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--days-limit", type=int, default=None, help="only the first N val and N test preset days (tests)")
    p.add_argument("--out-dir", default=None, help="write <run>.npz here instead of explorer_cache/ (tests)")
    a = p.parse_args()
    if a.choose_days:
        choose_days()
        return
    for r in a.run:
        build(r, a.ig_steps, a.ig_batch, a.threads, a.days_limit, out=(Path(a.out_dir) / f"{r}.npz" if a.out_dir else None))


if __name__ == "__main__":
    main()
