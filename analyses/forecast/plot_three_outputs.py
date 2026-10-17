"""All three output maps (days t-1, t, t+1) of two Temporal3DNet runs on a few validation days.

Runs (Indian Ocean store, dilated backbone, physics loss off):
  gapfill  t3d_dil_c4_60ep_gdl100  window t-4..t+4, --context-clouds persist, donor_exclude 0 (t-1 / t+1 heads had a data loss)
  nowcast  t3d_nc_b8               window t-8..t,   --context-clouds none,    donor_exclude 8 (t-1 / t+1 heads had no loss)

Inputs are rebuilt from the memory-mapped store as train_pinn.make_inputs_3d / static_inputs / nb_masks build them
(features held as float16 like the trainer's GPU copy); the model runs on the CPU in float32. The day-t output is checked
against the run's preds.npz before plotting.

Usage (project root): python analyses/forecast/plot_three_outputs.py
Writes analyses/forecast/figures/three_outputs_{gapfill,nowcast}.png and analyses/forecast/results/three_outputs.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only: the GPU is held by another job
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from chl_cnn import data as D, store as S, train2 as T2, clouds as C       # noqa: E402
from chl_cnn.models import Temporal3DNet                                    # noqa: E402
from chl_cnn.plotting import MAP_CMAP, ERR_CMAP                             # noqa: E402

RUNS = {"gapfill": "t3d_dil_c4_60ep_gdl100", "nowcast": "t3d_nc_b8"}
NAMES = {"gapfill": "Gap-filler", "nowcast": "Nowcast"}
RUN_DIR = ROOT / "models" / "pinn" / "t3d"
FIG, RES = HERE / "figures", HERE / "results"
CLOUD_C, LAND_C = "#DDDDDD", "#8C8C8C"           # light grey: clouds / no data; mid grey: land
# one target date per val year; the eligible day closest to each target is used; 2018-02-20 is fixed.
# No SW-monsoon (Jun-Sep) day of 2001, 2007 or 2013 has real clouds <= 70 % (min 72 %), so 2001 takes a December day.
TARGETS = ["2001-12-15", "2007-04-20", "2013-10-25", "2018-02-20"]


def season(date):
    m = int(date[5:7])
    return {12: "NE monsoon", 1: "NE monsoon", 2: "NE monsoon", 3: "spring inter-monsoon", 4: "spring inter-monsoon",
            5: "spring inter-monsoon", 10: "fall inter-monsoon", 11: "fall inter-monsoon"}.get(m, "SW monsoon")
CLOUD_RANGE = (0.30, 0.70)                       # real-cloud fraction of ocean pixels on day t
MIN_NB_OBS = 0.20                                # observed fraction of ocean pixels required on t-1 and t+1
torch.set_num_threads(8)


# ============================================================================ data
def load_common():
    d = S.load_store(); times = d["time"]; T2.TIMES = times
    ocean = d["ocean"]; lat, lon = d["lat"], d["lon"]
    y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & ocean
    tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    clim, dbin = T2.doy_climatology(y, obs, tr)
    la, lo = np.meshgrid(lat.astype("float64"), lon.astype("float64"), indexing="ij")
    depth = np.log10(np.maximum(-d["topo"], 0) + 1)
    stat = np.stack([(la - la.mean()) / la.std(), (lo - lo.mean()) / lo.std(), (depth - depth.mean()) / depth.std()]).astype("float32")
    NB = {k: T2.neighbour_index(times, k) for k in range(-8, 9) if k}
    return dict(d=d, times=times, ocean=ocean, y=y, obs=obs, tr=tr, va=va, clim=clim, dbin=dbin, stat=stat, NB=NB)


def pick_days(c):
    """Closest eligible val day to each target date (cloud 30-70 %, t-1 and t+1 in the record with >= 20 % observed)."""
    ocean, obs, times, va, NB = c["ocean"], c["obs"], c["times"], c["va"], c["NB"]
    no = ocean.sum()
    frac = obs[va].reshape(len(va), -1).sum(1) / no
    jm, jp = NB[-1][va], NB[1][va]
    fm = np.where(jm >= 0, obs[np.maximum(jm, 0)].reshape(len(va), -1).sum(1) / no, 0)
    fp = np.where(jp >= 0, obs[np.maximum(jp, 0)].reshape(len(va), -1).sum(1) / no, 0)
    ok = (1 - frac >= CLOUD_RANGE[0]) & (1 - frac <= CLOUD_RANGE[1]) & (fm >= MIN_NB_OBS) & (fp >= MIN_NB_OBS)
    dv = times[va].astype("datetime64[D]")
    out = []
    for tgt in TARGETS:
        dist = np.abs((dv - np.datetime64(tgt)).astype(int)).astype(float); dist[~ok] = np.inf
        i = int(np.argmin(dist)); assert np.isfinite(dist[i])
        out.append(dict(t=int(va[i]), date=str(dv[i]), season=season(str(dv[i])), cloud_frac=float(1 - frac[i]),
                        obs_frac_tm1=float(fm[i]), obs_frac_tp1=float(fp[i])))
    assert out[-1]["date"] == "2018-02-20"
    return out


def load_model(run, cfg):
    t3 = cfg["t3d"]
    m = Temporal3DNet(t3["c_day"], t3["n_static"], t3["n_days"], backbone=cfg["t3d_backbone"], stem_width=cfg["t3d_width"],
                      stem_depth=cfg["t3d_depth"], kt=cfg["t3d_kernel_t"], width=cfg["width"], dropout=cfg["dropout"], out_channels=3,
                      depth=cfg["unet_depth"], spectral=cfg["spectral"], modes=cfg["spectral_modes"], spec_ch=cfg["spectral_ch"],
                      dilations=cfg.get("dilations"))
    m.load_state_dict(torch.load(RUN_DIR / run / "model.pt", map_location="cpu")["model"])
    return m.eval()


def build_inputs(c, cfg, t, hid_t, donor_t):
    """numpy copy of train_pinn.make_inputs_3d + static_inputs (+ nb_masks) for one day t. Returns x3d, xs, visible masks."""
    d, y, obs, NB = c["d"], c["y"], c["obs"], c["NB"]
    mu, sd = cfg["chl_norm"]["mean"], cfg["chl_norm"]["std"]
    fs = cfg["feature_stats"]
    B = cfg.get("context_before", cfg["context"]); A = cfg.get("context_after", cfg["context"])
    H, W = obs.shape[1:]

    def feats(j):     # z-scored, NaN -> 0, stored as float16 on the GPU by the trainer
        f = [np.nan_to_num((np.asarray(d[fs[n]["store_var"]][j], "float32") - fs[n]["mean"]) / fs[n]["std"], nan=0.0) for n in cfg["features"]]
        return np.stack(f).astype("float16").astype("float32")

    chl_z = lambda j: (np.nan_to_num(y[j]) - mu) / sd * obs[j]
    days, vis = [], {}
    for k in range(-B, A + 1):
        if k == 0:
            v = obs[t] & ~hid_t; vis[0] = v
            if cfg["no_chl_today"]:
                days.append(np.concatenate([feats(t), np.zeros((2, H, W), "float32")]))
            else:
                days.append(np.concatenate([feats(t), (chl_z(t) * v)[None], v[None].astype("float32")]))
            continue
        j = NB[k][t]
        if j < 0:                                    # day not in the record: all channels zero
            days.append(np.zeros((len(cfg["features"]) + 2, H, W), "float32")); vis[k] = np.zeros((H, W), bool); continue
        m = obs[j].copy()
        if cfg["context_clouds"] == "persist":       # the donor's day-(donor+k) clouds also cover day t+k
            jd = NB[k][donor_t]; jd = jd if jd >= 0 else donor_t
            m &= obs[jd]
        vis[k] = m
        days.append(np.concatenate([feats(j), (chl_z(j) * m)[None], m[None].astype("float32")]))
    x3d = np.stack(days, 1)[None].astype("float32")
    doy = int((c["times"][t] - c["times"][t].astype("datetime64[Y]")).astype("timedelta64[D]").astype(int))
    ang = 2 * np.pi * doy / 365.25
    year = (int(c["times"][t].astype("datetime64[Y]").astype(int)) + 1970 - 2009) / 7.0
    clim_z = (c["clim"][c["dbin"][t]] - mu) / sd
    xs = np.concatenate([clim_z[None], c["stat"], np.stack([np.full((H, W), np.sin(ang)), np.full((H, W), np.cos(ang))]),
                         np.full((1, H, W), year)])[None].astype("float32")
    return x3d, xs, vis


def predict(c, cfg, model, x3d, xs, t):
    """predict_z of the trainer: network output + z-scored climatology of t-1, t, t+1, then un-z-scored to ln CHL."""
    mu, sd = cfg["chl_norm"]["mean"], cfg["chl_norm"]["std"]
    with torch.no_grad():
        z = model(torch.from_numpy(x3d), torch.from_numpy(xs))[0].numpy()
    jm, jp = max(c["NB"][-1][t], 0), max(c["NB"][1][t], 0)       # nb_masks clamps a missing day to index 0
    cl = np.stack([c["clim"][c["dbin"][j]] for j in (jm, t, jp)])
    return (z + (cl - mu) / sd) * sd + mu, cl


def run_model(c, key, days):
    run = RUNS[key]; cfg = json.loads((RUN_DIR / run / "config.json").read_text())
    model = load_model(run, cfg)
    ex = cfg.get("donor_exclude", 0)
    hid_va, don_va = C.fixed_hidden(c["obs"], c["va"], seed=cfg["eval_seed"], times=c["times"], exclude=ex)
    pz = np.load(RUN_DIR / run / "preds.npz")
    assert np.array_equal(pz["val_idx"], c["va"])
    y, obs, ocean = c["y"], c["obs"], c["ocean"]
    res = []
    for dd in days:
        t = dd["t"]; i = int(np.flatnonzero(c["va"] == t)[0])
        hid = hid_va[i]
        assert np.array_equal(hid, pz["hidden"][i]), f"{run}: hidden set differs from preds.npz on {dd['date']}"
        x3d, xs, vis = build_inputs(c, cfg, t, hid, don_va[i])
        out, cl = predict(c, cfg, model, x3d, xs, t)
        saved = pz["pred"][i].astype("float32")
        maxdiff = float(np.nanmax(np.abs(out[1] - saved)[ocean]))
        jm, jp = c["NB"][-1][t], c["NB"][1][t]
        ch = []
        for ci, (k, j) in enumerate([(-1, jm), (0, t), (1, jp)]):
            sc = hid if k == 0 else obs[j]
            inp = np.zeros_like(sc) if k == 0 else (sc & vis.get(k, np.zeros_like(sc)))   # scored pixels that were input pixels
            e = np.abs(out[ci] - y[j])
            ch.append(dict(offset=k, date=str(c["times"][j])[:10], scored="hidden day-t pixels" if k == 0 else "all observed pixels",
                           mae=float(e[sc].mean()), n=int(sc.sum()), n_input=int(inp.sum()),
                           mae_not_input=(float(e[sc & ~inp].mean()) if (sc & ~inp).any() else None), n_not_input=int((sc & ~inp).sum()),
                           mae_climatology=float(np.abs(cl[ci] - y[j])[sc].mean()),
                           out_minus_clim_mean=float((out[ci] - cl[ci])[ocean].mean()),
                           out_minus_clim_sd=float((out[ci] - cl[ci])[ocean].std())))
        res.append(dict(day=dd, out=out, clim=cl, obs={k: (y[j], obs[j]) for k, j in [(-1, jm), (0, t), (1, jp)]},
                        hidden=hid, maxdiff_day_t=maxdiff, channels=ch))
    return dict(run=run, cfg=cfg, days=res, max_abs_diff_day_t=max(r["maxdiff_day_t"] for r in res))


# ============================================================================ figures
def _map(ax, img, ocean, cmap, vmin, vmax):
    ax.imshow(ocean.astype(float), cmap=ListedColormap([LAND_C, CLOUD_C]), vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(np.where(ocean, img, np.nan), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def fig_three(c, R, key, path, dlim):
    ocean = c["ocean"]; rows = R["days"]; nr = len(rows)
    allv = np.concatenate([r["obs"][k][0][r["obs"][k][1]] for r in rows for k in (-1, 0, 1)])
    vmin, vmax = (float(v) for v in np.percentile(allv, [1, 99]))
    FW, L, R_, G, gg = 20.0, 1.75, 0.3, 0.05, 0.35            # inches: left/right margins, panel gap, group gap
    pw = (FW - L - R_ - 6 * G - 2 * gg) / 9; ph = pw * 149 / 181; rh = ph + 0.55
    top = 0.85; FH = top + nr * rh + 0.75
    fig = plt.figure(figsize=(FW, FH))
    at = lambda x, y, w, h: fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])
    xg = [L + g * (3 * pw + 2 * G + gg) for g in range(3)]
    win = "t-4..t+4, donor clouds on context days" if key == "gapfill" else "t-8..t, real clouds on context days"
    fig.text(0.5, 1 - 0.3 / FH, f"{NAMES[key]} {R['run']}: output maps for days t-1, t, t+1 ({nr} val days; input window {win})",
             ha="center", fontsize=15)
    groups = ["Observed ln CHL (grey: clouds / no data)", "Model output, ln CHL (full field)", "Model output minus DOY climatology"]
    for g, s in enumerate(groups):
        fig.text((xg[g] + 1.5 * pw + G) / FW, 1 - 0.68 / FH, s, ha="center", fontsize=12.5, weight="bold")
    for i, r in enumerate(rows):
        y0 = top + i * rh + 0.3
        dd = r["day"]
        fig.text((L - 0.12) / FW, 1 - (y0 + ph / 2) / FH, f"{dd['date']}\n{dd['season']}\nclouds {100 * dd['cloud_frac']:.0f} %",
                 ha="right", va="center", fontsize=10.5)
        for ci, k in enumerate((-1, 0, 1)):
            lab = {-1: "t-1", 0: "t", 1: "t+1"}[k]
            yk, ok = r["obs"][k]; chm = r["channels"][ci]
            ax = at(xg[0] + ci * (pw + G), y0, pw, ph)
            im_c = _map(ax, np.where(ok, yk, np.nan), ocean, MAP_CMAP, vmin, vmax)
            ax.set_title(f"day {lab} ({chm['date'][5:]})  n obs {int(ok.sum()):,}", fontsize=8.5)
            ax = at(xg[1] + ci * (pw + G), y0, pw, ph)
            _map(ax, r["out"][ci], ocean, MAP_CMAP, vmin, vmax)
            sc = "hidden px" if k == 0 else "all obs"
            inp = f", {chm['n_input']:,} input" if k else ""
            ax.set_title(f"day {lab}: MAE {chm['mae']:.3f}, {sc}\n(n {chm['n']:,}{inp})", fontsize=8.5)
            ax = at(xg[2] + ci * (pw + G), y0, pw, ph)
            im_d = _map(ax, r["out"][ci] - r["clim"][ci], ocean, ERR_CMAP, -dlim, dlim)
            ax.set_title(f"day {lab}\nclimatology MAE {chm['mae_climatology']:.3f}", fontsize=8.5)
    yb = top + nr * rh + 0.12
    for g, im, lab in [(0, im_c, "ln CHL (mg m$^{-3}$)"), (1, im_c, "ln CHL (mg m$^{-3}$)"), (2, im_d, "output - climatology (ln CHL)")]:
        w = 3 * pw + 2 * G
        cb = fig.colorbar(im, cax=at(xg[g] + 0.15 * w, yb, 0.7 * w, 0.14), orientation="horizontal", extend="both")
        cb.set_label(lab, fontsize=10)
    fig.savefig(path, dpi=100); plt.close(fig)


# ============================================================================ report
def main():
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    c = load_common()
    days = pick_days(c)
    print("days:", json.dumps(days, indent=1))
    out = dict(days=days, models={})
    RR = {key: run_model(c, key, days) for key in RUNS}
    oc = c["ocean"]   # diverging scale shared by both figures: 99th percentile of |output - climatology| over all maps
    dlim = float(np.ceil(np.percentile(np.concatenate([np.abs(r["out"] - r["clim"])[:, oc].ravel()
                                                       for R in RR.values() for r in R["days"]]), 99) * 10) / 10)
    for key, R in RR.items():
        print(f"{key} {R['run']}: max |day-t output - preds.npz| = {R['max_abs_diff_day_t']:.4g}  "
              f"(per day {[round(r['maxdiff_day_t'], 4) for r in R['days']]})")
        print(f"  {'date':10s} {'ch':>4s} {'MAE':>6s} {'n':>6s} {'n_input':>7s} {'MAE_notinput':>12s} {'MAE_clim':>8s}")
        for r in R["days"]:
            for chm in r["channels"]:
                print(f"  {r['day']['date']:10s} {chm['offset']:>+4d} {chm['mae']:6.3f} {chm['n']:6d} {chm['n_input']:7d} "
                      + (f"{chm['mae_not_input']:12.3f}" if chm['mae_not_input'] is not None else f"{'-':>12s}") + f" {chm['mae_climatology']:8.3f}  out-clim mean {chm['out_minus_clim_mean']:+.3f} sd {chm['out_minus_clim_sd']:.3f}")
        fig_three(c, R, key, FIG / f"three_outputs_{key}.png", dlim)
        out["models"][key] = dict(run=R["run"], donor_exclude=R["cfg"].get("donor_exclude", 0),
                                  context_clouds=R["cfg"]["context_clouds"], max_abs_diff_day_t=R["max_abs_diff_day_t"],
                                  per_day=[dict(date=r["day"]["date"], max_abs_diff_day_t=r["maxdiff_day_t"], channels=r["channels"])
                                           for r in R["days"]])
    (RES / "three_outputs.json").write_text(json.dumps(out, indent=1))
    print("wrote", FIG / "three_outputs_gapfill.png", FIG / "three_outputs_nowcast.png", RES / "three_outputs.json")


if __name__ == "__main__":
    main()
