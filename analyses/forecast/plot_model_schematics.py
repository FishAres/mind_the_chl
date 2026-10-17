"""Input/output schematics of the two Temporal3DNet runs (t3d_fc_b8 forecast, t3d_nc_b8 nowcast) for MODELS.md.

No training and no GPU: the day-t predictions come from each run's preds.npz; the inputs are rebuilt from the
memory-mapped store exactly as train_pinn.make_inputs_3d / static_inputs build them (checked below).

Usage (project root): python analyses/forecast/plot_model_schematics.py
Writes analyses/forecast/figures/{schematic_forecast,schematic_nowcast,architecture}.png
and analyses/forecast/results/model_schematics.json (example-day fractions, checks, reweighted MAEs).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from chl_cnn import data as D, store as S, train2 as T2, clouds as C          # noqa: E402
from chl_cnn.plotting import OKABE_ITO, MAP_CMAP, ERR_CMAP                     # noqa: E402
from analyses.forecast.forecast_eval import persistence, BINS                  # noqa: E402

DAY = "2018-02-20"
RUNS = {"forecast": "t3d_fc_b8", "nowcast": "t3d_nc_b8"}
RUN_DIR = ROOT / "models" / "pinn" / "t3d"
FIG, RES = HERE / "figures", HERE / "results"

CLOUD_C, LAND_C, FAKE_C = "#DDDDDD", "#8C8C8C", OKABE_ITO[3]     # light grey, mid grey, vermillion
NOSCORE_C = "#4D4D4D"                                           # error maps: ocean pixels that are not scored
SST_CMAP = "cividis"


# ============================================================================ data
def build_inputs(cfg, feats_z, chl_z, obs, nbi, t, hid_t, clim_z, stat, doy_sc, year):
    """numpy copy of train_pinn.make_inputs_3d + static_inputs for one day t (batch of 1)."""
    days = []
    for k in range(-cfg["context_before"], cfg["context_after"] + 1):
        if k == 0:
            vis = obs[t] & ~hid_t
            if cfg["no_chl_today"]:
                days.append(np.concatenate([feats_z[t], np.zeros((2,) + vis.shape, "float32")]))
            else:
                days.append(np.concatenate([feats_z[t], (chl_z[t] * vis)[None], vis[None].astype("float32")]))
        else:
            j = nbi[k]
            if j < 0:      # day not in the record: all channels zero
                days.append(np.zeros_like(days[-1] if days else np.zeros((11,) + obs.shape[1:], "float32"))); continue
            m = obs[j]     # --context-clouds none: past days keep their real observation mask
            days.append(np.concatenate([feats_z[j], (chl_z[j] * m)[None], m[None].astype("float32")]))
    x3d = np.stack(days, 1)[None]                                   # (B, c_day, T, H, W)
    H, W = obs.shape[1:]
    xs = np.concatenate([clim_z[None], stat, np.broadcast_to(doy_sc[:, None, None], (2, H, W)),
                         np.full((1, H, W), year, "float32")])[None]   # (B, n_static, H, W)
    return x3d.astype("float32"), xs.astype("float32")


def load_example(day=DAY):
    d = S.load_store(); times = d["time"]; T2.TIMES = times
    ocean, lat, lon = d["ocean"], d["lat"], d["lon"]
    y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & ocean
    tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    t = int(np.flatnonzero(times.astype("datetime64[D]") == np.datetime64(day))[0])
    assert t in va, "example day must be a val day"
    cfgs = {m: json.loads((RUN_DIR / r / "config.json").read_text()) for m, r in RUNS.items()}
    cfg = cfgs["nowcast"]
    checks = {}

    # normalisation exactly as the trainer (chl over training days; features from config feature_stats)
    chl_mu, chl_sd = float(np.nanmean(y[tr])), float(np.nanstd(y[tr]))
    checks["chl_norm_matches_config"] = bool(np.isclose(chl_mu, cfg["chl_norm"]["mean"]) and np.isclose(chl_sd, cfg["chl_norm"]["std"]))
    offs = range(-8, 1)
    nbi = {k: (T2.neighbour_index(times, k)[t] if k else t) for k in offs}
    need = sorted({j for j in nbi.values() if j >= 0})
    feats_z = {}
    for j in need:
        f = [np.nan_to_num((np.asarray(d[cfg["feature_stats"][n]["store_var"]][j], "float32") - cfg["feature_stats"][n]["mean"])
                           / cfg["feature_stats"][n]["std"], nan=0.0) for n in cfg["features"]]
        feats_z[j] = np.stack(f)
    chl_z = {j: np.nan_to_num((y[j] - chl_mu) / chl_sd) * obs[j] for j in need}
    clim, dbin = T2.doy_climatology(y, obs, tr)
    clim_z = (clim[dbin[t]] - chl_mu) / chl_sd
    la, lo = np.meshgrid(lat.astype("float64"), lon.astype("float64"), indexing="ij")
    depth = np.log10(np.maximum(-d["topo"], 0) + 1)
    stat = np.stack([(la - la.mean()) / la.std(), (lo - lo.mean()) / lo.std(), (depth - depth.mean()) / depth.std()])
    doy = int((times[t] - times[t].astype("datetime64[Y]")).astype("timedelta64[D]").astype(int))
    ang = 2 * np.pi * doy / 365.25
    year = (int(times[t].astype("datetime64[Y]").astype(int)) + 1970 - 2009) / 7.0

    ex = dict(day=day, t=t, lat=lat, lon=lon, ocean=ocean, y_t=y[t], obs_t=obs[t], chl_mu=chl_mu, chl_sd=chl_sd,
              past={k: (nbi[k], (y[nbi[k]] if nbi[k] >= 0 else None), (obs[nbi[k]] if nbi[k] >= 0 else None)) for k in offs if k},
              sst={k: (np.asarray(d["sst"][nbi[k]], "float32") - 273.15 if nbi[k] >= 0 else None) for k in offs},
              dates={k: (str(times[nbi[k]])[:10] if nbi[k] >= 0 else None) for k in offs}, models={})

    # fixed val hidden set recomputed from clouds.py must equal the nowcast's preds.npz 'hidden'
    hid_va, _ = C.fixed_hidden(obs, va, seed=cfg["eval_seed"], times=times, exclude=cfg["donor_exclude"])
    for m, r in RUNS.items():
        pz = np.load(RUN_DIR / r / "preds.npz"); c = cfgs[m]
        assert np.array_equal(pz["val_idx"], va) and np.array_equal(pz["idx"][:len(va)], va)
        i = int(np.flatnonzero(pz["idx"] == t)[0])
        pred = pz["pred"][i].astype("float32"); hid = pz["hidden"][i]
        if m == "nowcast":
            checks["nowcast_hidden_equals_clouds_fixed_hidden_all_val_days"] = bool(np.array_equal(pz["hidden"][:len(va)], hid_va))
        scored = obs[t] if c["no_chl_today"] else hid          # forecast: all observed day-t pixels; nowcast: hidden pixels
        x3d, xs = build_inputs(c, feats_z, chl_z, obs, nbi, t, hid, clim_z, stat, np.array([np.sin(ang), np.cos(ang)]), year)
        assert x3d.shape[1:3] == (c["t3d"]["c_day"], c["t3d"]["n_days"]) and xs.shape[1] == c["t3d"]["n_static"]
        vis_t = x3d[0, -1, -1] > 0
        checks[f"{m}_day_t_visible_mask"] = ("zero (no day-t CHL)" if c["no_chl_today"] and not vis_t.any() else
                                            "observed & ~hidden" if np.array_equal(vis_t, obs[t] & ~hid) else "MISMATCH")
        # full-val MAE from preds.npz vs the trainer's metrics.json (float16 preds -> ~1e-5 agreement)
        mj = json.loads((RUN_DIR / r / "metrics.json").read_text())["scores"]["val"]
        key = "cnn_observed" if c["no_chl_today"] else "cnn_hidden"
        yv = y[va]; pv = pz["pred"][:len(va)].astype("float32"); mv = obs[va] if c["no_chl_today"] else pz["hidden"][:len(va)]
        checks[f"{m}_val_mae_preds_vs_metrics_json"] = [float(np.abs(pv - yv)[mv].mean()), mj[key]["mae"]]
        ex["models"][m] = dict(run=r, pred=pred, hidden=hid, scored=scored, x3d_shape=list(x3d.shape), xs_shape=list(xs.shape),
                               mae=float(np.abs(pred - y[t])[scored].mean()), n=int(scored.sum()))
    nh = ex["models"]["nowcast"]["hidden"]
    ex["fractions"] = dict(real_cloud_frac_ocean=float(1 - obs[t].sum() / ocean.sum()),
                           observed_frac_ocean=float(obs[t].sum() / ocean.sum()),
                           fake_cloud_frac_of_observed=float(nh.sum() / obs[t].sum()),
                           fake_cloud_frac_ocean=float(nh.sum() / ocean.sum()),
                           past_observed_frac_ocean={f"t{k}": (float(ex["past"][k][2].sum() / ocean.sum()) if ex["past"][k][0] >= 0 else None)
                                                     for k in range(-8, 0)})
    ex["checks"] = checks

    # age-of-last-observation mix of pixels under real clouds (val) and the reweighted fake-cloud MAEs
    _, age = persistence(y, obs, times, va, 8)
    cl = ocean[None] & ~obs[va]
    mix = {name: float((cl & ((age == 0) if hi == 0 else ((age >= lo_) & (age <= hi)))).sum() / cl.sum()) for name, lo_, hi in BINS}
    nc = json.loads((RES / "nowcast_comparison.json").read_text())["val"]
    fake_mix = {b: nc["climatology"][b]["n"] / nc["n_hidden"] for b in mix}
    rew = {k: sum(mix[b] * nc[k][b]["mae"] for b in mix) for k in ["nowcast", "forecast", "persistence", "climatology"]}
    ex["reweight"] = dict(real_cloud_mix_val=mix, fake_cloud_mix_val=fake_mix, reweighted_val_mae=rew)
    return ex


# ============================================================================ figures
def _map(ax, img, ocean, cmap, vmin, vmax, clouds=True):
    """Land (LAND_C), unobserved ocean (CLOUD_C if clouds else NaN-free), then the field."""
    base = np.where(ocean, 1, 0).astype(float)
    ax.imshow(base, cmap=ListedColormap([LAND_C, CLOUD_C if clouds else "white"]), vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(np.where(ocean, img, np.nan), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _overlay(ax, mask, color):
    ax.imshow(np.where(mask, 1.0, np.nan), cmap=ListedColormap([color]), vmin=0, vmax=1, interpolation="nearest")


def _box(ax, x, y, w, h, text, fc, fs=9, ec="#333333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333333"))


def chl_range(ex):
    v = [ex["y_t"][ex["obs_t"]]] + [p[1][p[2]] for p in ex["past"].values() if p[1] is not None]
    v = np.concatenate(v)
    return float(np.percentile(v, 1)), float(np.percentile(v, 99))


def fig_model(ex, model, path):
    """Top: CHL and SST inputs for days t-8..t; middle: channel list + network chain; bottom: day-t output, target, error."""
    m = ex["models"][model]; fc = model == "forecast"
    oc = ex["ocean"]; vmin, vmax = chl_range(ex)
    svmin, svmax = np.nanpercentile(np.concatenate([s[oc] for s in ex["sst"].values() if s is not None]), [1, 99])
    FW, FH, L, R = 16.0, 12.7, 0.75, 1.0
    fig = plt.figure(figsize=(FW, FH))
    at = lambda x, y, w, h: fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])   # inches from the top-left corner
    g = 0.08; pw = (FW - L - R - 8 * g) / 9; ph = pw * 149 / 181
    name = "Forecast (t3d_fc_b8)" if fc else "Nowcast (t3d_nc_b8)"
    fig.text(0.5, 1 - 0.3 / FH, f"{name}: inputs and output for one validation day, t = {ex['day']}", ha="center", fontsize=16)
    fig.text(0.5, 1 - 0.58 / FH, "Indian Ocean, 45-90°E, 5°S-32°N, 0.25° grid (149 x 181)", ha="center", fontsize=11, color="#444444")
    fig.text(L / FW, 1 - 1.0 / FH, "1  Inputs: days t-8 ... t", fontsize=13, weight="bold")
    leg = [Patch(fc=CLOUD_C, ec="#999999", label="real clouds / no data"), Patch(fc=LAND_C, label="land")]
    if not fc:
        leg.append(Patch(fc=FAKE_C, label="fake clouds on day t (observed, hidden from the input, scored)"))
    fig.legend(handles=leg, loc="upper right", bbox_to_anchor=(1 - R / FW, 1 - 0.78 / FH), ncol=len(leg), fontsize=10, frameon=False)

    y1, y2 = 1.55, 1.55 + ph + 0.35
    for c, k in enumerate(range(-8, 1)):
        x = L + c * (pw + g)
        ax = at(x, y1, pw, ph)
        if k == 0 and fc:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linestyle((0, (4, 3))); sp.set_color("#333333")
            ax.text(0.5, 0.5, "day t CHL\nnot given\n(CHL + mask\nchannels = 0)", ha="center", va="center", fontsize=9.5)
        elif k == 0:
            vis = ex["obs_t"] & ~m["hidden"]
            im_chl = _map(ax, np.where(vis, ex["y_t"], np.nan), oc, MAP_CMAP, vmin, vmax)
            _overlay(ax, m["hidden"], FAKE_C)
        else:
            j, yk, ok = ex["past"][k]
            im_chl = _map(ax, np.where(ok, yk, np.nan), oc, MAP_CMAP, vmin, vmax)
        ax.set_title(f"day t{k}  ({ex['dates'][k][5:]})" if k else f"day t  ({ex['dates'][k][5:]})", fontsize=9.5)
        if c == 0:
            ax.set_ylabel("ln CHL\n(visible pixels)", fontsize=10)
        ax = at(x, y2, pw, ph)
        im_sst = _map(ax, ex["sst"][k], oc, SST_CMAP, svmin, svmax)
        if c == 0:
            ax.set_ylabel("SST (1 of 9\nphysical fields)", fontsize=10)
    xc = FW - R + 0.15
    fig.colorbar(im_chl, cax=at(xc, y1, 0.15, ph)).set_label("ln CHL (mg m$^{-3}$)", fontsize=9)
    fig.colorbar(im_sst, cax=at(xc, y2, 0.15, ph)).set_label("SST (°C)", fontsize=9)

    # channels and network
    y3 = y2 + ph + 0.35
    fig.text(L / FW, 1 - y3 / FH, "2  Network input tensors and network", fontsize=13, weight="bold")
    axt = at(L, y3 + 0.25, 5.6, 2.0); axt.axis("off")
    s3, s2 = m["x3d_shape"], m["xs_shape"]
    day_t = "day t: physical fields only; CHL, mask = 0" if fc else "day t: CHL visible after fake clouds"
    txt = (f"Per-day stack: (B, {s3[1]} channels, {s3[2]} days, {s3[3]}, {s3[4]})\n"
           "  9 physical fields, z-scored, NaN -> 0: sst, so (salinity),\n"
           "    air_temp, u_wind, v_wind, sla (sea level anomaly),\n"
           "    mlotst (mixed-layer depth), u_curr, v_curr\n"
           "  ln CHL, z-scored (0 where not visible)\n"
           "  visible mask (1 = CHL value given)\n"
           f"  {day_t}\n"
           f"Static maps of day t: (B, {s2[1]}, {s2[2]}, {s2[3]})\n"
           "  ln CHL climatology (8-day bin of t), lat, lon,\n"
           "  log depth, sin + cos day of year, year")
    axt.text(0, 1, txt, va="top", ha="left", family="monospace", fontsize=9.2,
             bbox=dict(boxstyle="round,pad=0.5", fc="#F4F4F4", ec="#999999"))
    axn = at(L + 6.3, y3 + 0.35, FW - L - R - 6.3, 1.7); axn.axis("off"); axn.set_xlim(-0.02, 1.02); axn.set_ylim(0, 1)
    blocks = [("3D stem\n2 x Conv3d 3x3x3\nwidth 32,\nGroupNorm, GELU", "#DCEBF5"),
              ("collapse days\nConv3d (9,1,1)\n-> (B, 32, H, W)", "#DCEBF5"),
              ("concat 7\nstatic maps\n-> (B, 39, H, W)", "#F4F4F4"),
              ("dilated CNN\n8 residual blocks\ndil. 1,2,4,8,16,1,2,4\nwidth 64", "#FBE9D0"),
              ("3 maps: t-1, t, t+1\n(+ climatology)\nonly day t used", "#F4F4F4")]
    w, gp = 0.17, 0.0375
    for i, (s, col) in enumerate(blocks):
        x = i * (w + gp)
        _box(axn, x, 0.1, w, 0.8, s, col, fs=9.5)
        if i:
            _arrow(axn, x - gp, 0.5, x, 0.5)

    # outputs
    y4 = y3 + 2.45
    fig.text(L / FW, 1 - y4 / FH, "3  Output and scoring: day t", fontsize=13, weight="bold")
    ow = (FW - L - R - 2 * 0.5) / 3; oh = ow * 149 / 181; y5 = y4 + 0.75
    yt, sc = ex["y_t"], m["scored"]
    ax1 = at(L, y5, ow, oh); im = _map(ax1, m["pred"], oc, MAP_CMAP, vmin, vmax)
    ax1.set_title("Predicted ln CHL, day t\n(all ocean pixels)", fontsize=11)
    ax2 = at(L + ow + 0.5, y5, ow, oh)
    if fc:
        _map(ax2, np.where(ex["obs_t"], yt, np.nan), oc, MAP_CMAP, vmin, vmax)
        ax2.set_title("Target: observed ln CHL, day t\n(all observed pixels are scored)", fontsize=11)
    else:
        _map(ax2, np.where(m["hidden"], yt, np.nan), oc, MAP_CMAP, vmin, vmax)
        _overlay(ax2, ex["obs_t"] & ~m["hidden"], "white")
        ax2.set_title("Target: observed ln CHL under fake clouds, day t\n(scored pixels; white = visible in input, not scored)", fontsize=11)
    ax3 = at(L + 2 * (ow + 0.5), y5, ow, oh)
    err = np.where(sc, m["pred"] - yt, np.nan)
    lim = float(np.ceil(np.nanpercentile(np.abs(err), 99) * 10) / 10)
    ax3.imshow(np.where(oc, 1, 0).astype(float), cmap=ListedColormap([LAND_C, NOSCORE_C]), vmin=0, vmax=1, interpolation="nearest")
    ime = ax3.imshow(err, cmap=ERR_CMAP, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title(f"Prediction - target (dark grey: not scored)\nMAE on scored pixels: {m['mae']:.3f} (n = {m['n']:,})", fontsize=11)
    for a_, im_, lab in [(ax1, im, "ln CHL (mg m$^{-3}$)"), (ax2, im, "ln CHL (mg m$^{-3}$)"), (ax3, ime, "prediction - target (ln CHL)")]:
        p = a_.get_position()
        cax = fig.add_axes([p.x0 + 0.1 * p.width, p.y0 - 0.32 / FH, 0.8 * p.width, 0.14 / FH])
        fig.colorbar(im_, cax=cax, orientation="horizontal").set_label(lab, fontsize=9)
    fig.savefig(path, dpi=110); plt.close(fig)


def fig_architecture(ex, path):
    c = json.loads((RUN_DIR / RUNS["nowcast"] / "config.json").read_text())
    s3, s2 = ex["models"]["nowcast"]["x3d_shape"], ex["models"]["nowcast"]["xs_shape"]
    H, W = s3[3], s3[4]
    fig, ax = plt.subplots(figsize=(15, 5.2)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Temporal3DNet with dilated CNN backbone (t3d_fc_b8 and t3d_nc_b8; {c['n_params']:,} parameters)", fontsize=13)
    top = [("per-day stack\n(B, 11, 9, H, W)", "#F4F4F4"),
           ("Conv3d 11->32, k 3x3x3\nGroupNorm(8), GELU", "#DCEBF5"),
           ("Conv3d 32->32, k 3x3x3\nGroupNorm(8), GELU", "#DCEBF5"),
           ("Conv3d 32->32, k (9,1,1)\ncollapses 9 days\n-> (B, 32, H, W)", "#DCEBF5"),
           ("concat static maps\n(B, 7, H, W)\n-> (B, 39, H, W)", "#F4F4F4")]
    bot = [("Conv2d 39->64, 3x3", "#FBE9D0"),
           ("8 residual blocks, width 64\neach: x + [BN, GELU, Conv 3x3 dil d] x 2\nd = 1, 2, 4, 8, 16, 1, 2, 4", "#FBE9D0"),
           ("BN, GELU, Conv2d 64->3, 1x1\n-> (B, 3, H, W)", "#FBE9D0"),
           ("+ z-scored climatology\nof t-1, t, t+1;\n x sd + mean -> ln CHL", "#F4F4F4"),
           ("3 maps: t-1, t, t+1\nonly day t is used\nand scored", "#F4F4F4")]
    for row, y0, blocks, ws in [(0, 0.58, top, [0.15, 0.18, 0.18, 0.18, 0.17]), (1, 0.12, bot, [0.13, 0.27, 0.18, 0.16, 0.13])]:
        x = 0.01; g = 0.025
        for i, ((s, col), w) in enumerate(zip(blocks, ws)):
            _box(ax, x, y0, w, 0.28, s, col, fs=10)
            if i:
                _arrow(ax, x - g, y0 + 0.14, x, y0 + 0.14)
            x += w + g
    ax.annotate("", xy=(0.08, 0.40), xytext=(0.88, 0.58),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333333", connectionstyle="arc3,rad=-0.05"))
    ax.text(0.5, -0.02, f"H x W = {H} x {W} (0.25° grid); B = batch. Blue: 3D stem over days; orange: 2D backbone at full resolution "
            "(no downsampling).\nOutput heads t-1 and t+1 get no loss in either model (t-1: all its observed pixels are inputs; "
            "t+1: outside the input window). Physics loss off (lambda = 0).", ha="center", va="top", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# ============================================================================ report
def main():
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    ex = load_example()
    print("checks:", json.dumps(ex["checks"], indent=1))
    print("fractions:", json.dumps(ex["fractions"], indent=1))
    print("reweight:", json.dumps(ex["reweight"], indent=1))
    for m in RUNS:
        print(m, "x3d", ex["models"][m]["x3d_shape"], "static", ex["models"][m]["xs_shape"],
              f"MAE {ex['models'][m]['mae']:.4f} n {ex['models'][m]['n']}")
    fig_model(ex, "forecast", FIG / "schematic_forecast.png")
    fig_model(ex, "nowcast", FIG / "schematic_nowcast.png")
    fig_architecture(ex, FIG / "architecture.png")
    out = dict(day=ex["day"], fractions=ex["fractions"], checks=ex["checks"], reweight=ex["reweight"],
               example_mae={m: dict(mae=ex["models"][m]["mae"], n=ex["models"][m]["n"]) for m in RUNS},
               shapes={m: dict(per_day=ex["models"][m]["x3d_shape"], static=ex["models"][m]["xs_shape"]) for m in RUNS})
    (RES / "model_schematics.json").write_text(json.dumps(out, indent=1))
    print("wrote", FIG / "schematic_forecast.png", FIG / "schematic_nowcast.png", FIG / "architecture.png", RES / "model_schematics.json")


if __name__ == "__main__":
    main()
