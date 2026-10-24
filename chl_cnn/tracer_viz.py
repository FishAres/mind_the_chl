"""Figures for the learned-tracer runs (models/tracer/<run>/, chl_cnn/train_tracer.py, chl_cnn/tracer.py).

All draw_* functions take a list of run folders and return a matplotlib Figure (or None after printing a note when the
needed files / history keys are absent). They never call plt.show(); the notebook displays the figure and closes it.

Inputs read: history.json (every epoch), metrics.json (written at the end of a run: baselines on the common pixel set at
init / best epoch), factor_maps.npz (end of run) or model.pt + store static fields (CPU fallback for the factor maps),
source_maps.npz (step-2 --source runs: season-mean lead-1 source per pixel, rewritten at every new best epoch).
Baselines (persistence, advection along reanalysis currents x1.0 / x0.25, DOY climatology) do not change with epochs;
they are drawn as horizontal reference lines from metrics.json["best"]["common"] (falls back to ["init"]).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .plotting import OKABE_ITO, MAP_CMAP, ERR_CMAP
from .runs import CATEGORIES

BASELINES = ["persistence", "adv_x1.0", "adv_x0.25", "doy_clim"]
BASE_LABEL = {"persistence": "persistence", "adv_x1.0": "currents x1.0", "adv_x0.25": "currents x0.25", "doy_clim": "DOY climatology"}
BASE_STYLE = {"persistence": ("#000000", "--"), "adv_x1.0": ("#56B4E9", "-."), "adv_x0.25": ("#D55E00", ":"), "doy_clim": ("#CC79A7", (0, (5, 1)))}
LS = ["-", "--", ":"]


def tracer_runs(train_tracer_only=True):
    """Runs in models/tracer. train_tracer_only: only runs whose history.json is in train_tracer.py format (val_model_common);
    the trb_* / trs_* runs come from other trainers and have none of the curves drawn here."""
    out = []
    for h in CATEGORIES["tracer"].glob("*/history.json"):
        if not train_tracer_only or "val_model_common" in json.loads(h.read_text()):
            out.append(h.parent)
    return sorted(out)


def _hist(p):
    h = json.load(open(Path(p) / "history.json"))
    return h if h.get("epoch") else None


def _style(i):
    return OKABE_ITO[i % len(OKABE_ITO)], LS[(i // len(OKABE_ITO)) % 3]


def _metrics(p):
    f = Path(p) / "metrics.json"
    return json.load(open(f)) if f.exists() else None


def _baselines(p):
    """{pred: [mae lead 1..K]} on the common set from metrics.json (best, else init), plus the epoch tag; or (None, None)."""
    m = _metrics(p)
    if not m:
        return None, None
    for key in ("best", "init"):
        c = m.get(key, {}).get("common")
        if c:
            K = len(c["model"])
            return {b: [c[b][str(k)]["mae"] for k in range(1, K + 1)] for b in BASELINES if b in c}, key
    return None, None


def _loaded(run_dirs, need=None):
    out = []
    for i, p in enumerate(run_dirs):
        h = _hist(p)
        if h is None:
            print(f"skip {Path(p).name}: no epochs in history.json yet"); continue
        if need and not any(h.get(k) for k in need):
            print(f"skip {Path(p).name}: history has no {need}"); continue
        out.append((i, Path(p), h))
    return out


def _legend_run(p, h, done):
    return f"{p.name}{'' if done else ' (running)'}"


def draw_val_by_lead(run_dirs):
    """Val MAE per lead vs epoch (one panel per lead), model on the common pixel set, baselines as horizontal lines."""
    runs = _loaded(run_dirs, ["val_model_common"])
    if not runs:
        return None
    K = len(runs[0][2]["val_model_common"][0])
    fig, axes = plt.subplots(1, K, figsize=(3.4 * K, 4.2), sharey=True, squeeze=False); axes = axes[0]
    base_src = None
    for k in range(K):
        ax = axes[k]
        for i, p, h in runs:
            c, ls = _style(i); v = [e[k] for e in h["val_model_common"]]
            ax.plot(h["epoch"], v, color=c, ls=ls, lw=1.4, label=_legend_run(p, h, (p / "metrics.json").exists()))
        if base_src is None:
            for i, p, h in runs:
                b, tag = _baselines(p)
                if b:
                    base_src = (p.name, b, tag); break
        if base_src:
            for nm, v in base_src[1].items():
                col, ls = BASE_STYLE[nm]
                ax.axhline(v[k], color=col, ls=ls, lw=1.2, label=BASE_LABEL[nm] if k == 0 else None)
        ax.set(title=f"Lead {k + 1} d", xlabel="epoch"); ax.grid(alpha=.3)
    axes[0].set_ylabel("val MAE, ln(CHL), common pixel set")
    axes[0].legend(fontsize=7)
    fig.suptitle("Validation MAE by forecast lead and epoch" + (f" (baselines from {base_src[0]} metrics.json, {base_src[2]})" if base_src else ""), fontsize=11)
    if not base_src:
        print("note: no metrics.json yet (written at the end of a run): baseline lines omitted")
    fig.tight_layout()
    return fig


def draw_best_epoch(run_dirs):
    """MAE vs lead at each run's best epoch (min val_mean_lead): model (common set) and baselines (common set)."""
    runs = _loaded(run_dirs, ["val_model_common"])
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.6)); base_src = None
    for i, p, h in runs:
        c, ls = _style(i); j = int(np.argmin(h["val_mean_lead"])); v = h["val_model_common"][j]
        ax.plot(range(1, len(v) + 1), v, color=c, ls=ls, marker="o", lw=1.6, label=f"{p.name} (epoch {h['epoch'][j]})")
        if base_src is None:
            b, tag = _baselines(p)
            if b:
                base_src = (p.name, b, tag)
    if base_src:
        for nm, v in base_src[1].items():
            col, ls = BASE_STYLE[nm]
            ax.plot(range(1, len(v) + 1), v, color=col, ls=ls, marker="s", ms=4, lw=1.2, label=BASE_LABEL[nm])
    else:
        print("note: no metrics.json yet: baselines omitted")
    ax.set(xlabel="forecast lead (days)", ylabel="val MAE, ln(CHL), common pixel set", title="Validation MAE by lead at best epoch, model and baselines")
    ax.set_xticks(range(1, len(runs[0][2]["val_model_common"][0]) + 1)); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _band(ax, h, key, col, ls, label):
    f = [e.get(key) for e in h["factors"]]
    if not all(f):
        return False
    ep = h["epoch"]
    ax.plot(ep, [x["mean"] for x in f], color=col, ls=ls, lw=1.5, label=label)
    ax.fill_between(ep, [x["p10"] for x in f], [x["p90"] for x in f], color=col, alpha=.15, lw=0)
    return True


def _train_tracer_factors(runs):
    """Keep runs whose history "factors" entries have train_tracer's format ({name: {mean, sd, p10, p90}}); other trainers
    (e.g. analyses/tracer/static_train.py) store different summaries and are skipped with a note."""
    out = []
    for r in runs:
        f = r[2]["factors"][-1] if r[2]["factors"] else None
        if isinstance(f, dict) and f and all(isinstance(v, dict) and "mean" in v for v in f.values()):
            out.append(r)
        else:
            print(f"skip {r[1].name}: history factors not in train_tracer format")
    return out


def draw_factors(run_dirs):
    """Learned factors vs epoch: a, b (mean over ocean per val day: epoch mean line, p10-p90 band over days), wind c, theta."""
    runs = _train_tracer_factors(_loaded(run_dirs, ["factors"]))
    if not runs:
        return None
    panels = []
    for key, lab, ttl in (("a", "a (reanalysis-current factor)", "Factor a"), ("b", "b (geostrophic factor)", "Factor b")):
        if any(h["factors"] and key in h["factors"][-1] for _, _, h in runs):
            panels.append(("fac", key, lab, ttl))
    if any(h.get("wind") and h["wind"][-1] for _, _, h in runs):
        panels += [("wind", "c_m_s_per_Pa", "wind-stress coefficient c (m/s per Pa)", "Wind coefficient c"),
                   ("wind", "theta_deg", "wind turning angle theta (deg)", "Wind turning angle theta")]
    if not panels:
        print("note: no factor statistics in history"); return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False); axes = axes[0]
    for ax, (kind, key, ylab, ttl) in zip(axes, panels):
        for i, p, h in runs:
            c, ls = _style(i)
            if kind == "fac":
                _band(ax, h, key, c, ls, p.name)
            else:
                w = list(h.get("wind") or [])
                ep = [e for e, x in zip(h["epoch"], w) if x]
                if ep:
                    arr = np.array([x[key] for x in w if x])           # (epochs, 12 doys): mean line, min-max band over day of year
                    ax.plot(ep, arr.mean(1), color=c, ls=ls, lw=1.5, label=p.name)
                    ax.fill_between(ep, arr.min(1), arr.max(1), color=c, alpha=.15, lw=0)
        ax.set(xlabel="epoch", ylabel=ylab, title=ttl); ax.grid(alpha=.3)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle("Learned factors by epoch (line: mean; band: p10-p90 over val days; wind: min-max over 12 day-of-year points)", fontsize=10)
    fig.tight_layout()
    return fig


def draw_speed(run_dirs):
    """Mean ocean speed of the model velocity vs the reanalysis current (px/day) and their ratio, by epoch."""
    runs = _train_tracer_factors(_loaded(run_dirs, ["factors"]))
    runs = [r for r in runs if r[2]["factors"] and "speed_model" in r[2]["factors"][-1]]
    if not runs:
        print("note: no speed statistics in history"); return None
    fig, (ax, ar) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for n, (i, p, h) in enumerate(runs):
        c, ls = _style(i); ep = h["epoch"]
        _band(ax, h, "speed_model", c, ls, f"{p.name}: model")
        sr = [e["speed_rea"]["mean"] for e in h["factors"]]
        ax.plot(ep, sr, color=c, ls=":", lw=1, label=f"{p.name}: reanalysis" if n < 4 else None)
        ar.plot(ep, [e["speed_model"]["mean"] / e["speed_rea"]["mean"] for e in h["factors"]], color=c, ls=ls, lw=1.5, label=p.name)
    ax.set(xlabel="epoch", ylabel="mean ocean speed (px/day)", title="Model and reanalysis speed by epoch"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ar.set(xlabel="epoch", ylabel="model speed / reanalysis speed (ratio of means)", title="Speed ratio, model to reanalysis, by epoch"); ar.grid(alpha=.3); ar.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- factor maps
_static_cache = {}


def _static():
    """Static channels exactly as train_tracer.load (z-scored over ocean): [log10 depth, log10(1+dist km), lat, lon], plus ocean/lat/lon.
    Reads only 2-D store arrays (no time series). Mirrors that function; keep in sync if it changes."""
    if "s" not in _static_cache:
        from . import store_blk4 as S4
        sp = S4.store_path()
        oc = np.load(sp / "ocean.npy"); rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
        ys = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])); xs = slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
        ocean = oc[ys, xs]; lat = np.load(sp / "lat.npy")[ys].astype("float64"); lon = np.load(sp / "lon.npy")[xs].astype("float64")
        dep = np.load(sp / "depth.npy")[ys, xs].astype("float32")
        ldep = np.log10(np.maximum(np.nan_to_num(np.abs(dep), nan=5.0), 5.0))
        dist = -np.load(sp / "topo.npy")[ys, xs].astype("float32")
        ldist = np.log10(1.0 + np.maximum(np.nan_to_num(dist, nan=0.0), 0.0))
        la, lo = np.meshgrid(lat, lon, indexing="ij"); st = []
        for x in (ldep, ldist, la, lo):
            x = x.astype("float32"); st.append(np.where(ocean, (x - x[ocean].mean()) / x[ocean].std(), 0.0))
        _static_cache["s"] = (np.stack(st).astype("float32"), ocean, lat, lon)
    return _static_cache["s"]


def factor_maps(p, name, doys=(15.0, 106.0, 197.0, 288.0)):
    """(maps (n_doy, H, W) with NaN on land, doys, lat, lon, source) for factor 'a' or 'b'; None if unavailable.
    Source 1: factor_maps.npz (12 mid-month doys, written at the end of a run). Source 2: model.pt evaluated on CPU."""
    p = Path(p); f = p / "factor_maps.npz"
    if f.exists():
        z = np.load(f)
        if name in z.files:
            idx = [int(np.argmin(np.abs(z["doy"] - d))) for d in doys]
            m = np.where(z["ocean"][None], z[name][idx], np.nan)
            return m, z["doy"][idx], z["lat"], z["lon"], "factor_maps.npz"
        return None
    if not (p / "model.pt").exists() or not (p / "config.json").exists():
        return None
    import torch
    from . import tracer as TR
    cfg = json.load(open(p / "config.json")); ck = torch.load(p / "model.pt", map_location="cpu")
    if name not in {"a": "rea", "b": "geo"}.keys() or {"a": "rea", "b": "geo"}[name] not in ck["terms"]:
        return None
    st, ocean, lat, lon = _static(); H, W = ocean.shape
    model = TR.VelocityModel(cfg["terms"], n_static=st.shape[0], hidden=cfg["hidden"], a0=cfg["a_init"], b0=cfg["b_init"], b_range=(cfg["b_min"], 1.5),
                             cell=cfg["corr_cell"], corr_hidden=cfg["corr_hidden"])
    model.load_state_dict(ck["model"]); model.eval()
    d = np.asarray(doys, dtype="float32"); ang = torch.as_tensor(2 * np.pi * d / 365.25)
    n = len(d)
    x = torch.cat([torch.as_tensor(st)[None].expand(n, -1, -1, -1), torch.sin(ang).view(n, 1, 1, 1).expand(n, 1, H, W),
                   torch.cos(ang).view(n, 1, 1, 1).expand(n, 1, H, W)], 1)
    with torch.no_grad():
        m = getattr(model, name)(x).numpy()
    return np.where(ocean[None], m, np.nan), d, lat, lon, f"model.pt epoch {ck['epoch']} (CPU)"


def draw_factor_maps(run_dirs, name="a", max_runs=3):
    """Maps of factor a (or b) at 4 days of year; one row per run (first max_runs runs that have it)."""
    rows = []
    for p in run_dirs:
        try:
            r = factor_maps(p, name)
        except Exception as e:                                     # missing store / checkpoint mismatch: skip with a note
            print(f"skip factor {name} maps for {Path(p).name}: {type(e).__name__}: {e}"); continue
        if r is None:
            print(f"skip factor {name} maps for {Path(p).name}: term not in run or no model.pt / factor_maps.npz")
        else:
            rows.append((Path(p), r))
        if len(rows) == max_runs:
            break
    if not rows:
        return None
    vals = np.concatenate([r[0][np.isfinite(r[0])] for _, r in rows]); vmin, vmax = float(vals.min()), float(vals.max())
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.1 * len(rows) + .6), squeeze=False, sharex=True, sharey=True)
    for ri, (p, (m, d, lat, lon, src)) in enumerate(rows):
        for ci in range(4):
            ax = axes[ri, ci]
            im = ax.imshow(m[ci], origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap=MAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{name}(x), day of year {int(round(float(d[ci])))}", fontsize=9)
            if ri == len(rows) - 1:
                ax.set_xlabel("longitude (deg E)")
            if ci == 0:
                ax.set_ylabel(f"latitude (deg N)\n{p.name}", fontsize=8)
        fig.colorbar(im, ax=axes[ri].tolist(), label=f"factor {name}", fraction=.02, pad=.01)
        axes[ri, 0].text(.01, .02, src, transform=axes[ri, 0].transAxes, fontsize=6, color="w")
    fig.suptitle(f"Learned factor {name}(x, season) at 4 days of year", fontsize=11)
    return fig


# ---------------------------------------------------------------- source term (step 2)
def draw_source_maps(run_dirs, max_runs=3, min_n=5):
    """Season-mean lead-1 source S (ln CHL / day; prediction minus departure value over one day, val days) from
    source_maps.npz; one row per run that has it, columns DJF / MAM / JJA / SON; pixels with fewer than min_n days blank."""
    rows = []
    for p in run_dirs:
        f = Path(p) / "source_maps.npz"
        if f.exists():
            rows.append((Path(p), np.load(f)))
        if len(rows) == max_runs:
            break
    if not rows:
        print("note: no source_maps.npz (step-2 --source runs only)"); return None
    lim = max(float(np.nanpercentile(np.abs(z["mean"]), 99)) for _, z in rows) or 0.1
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.1 * len(rows) + .6), squeeze=False, sharex=True, sharey=True)
    for ri, (p, z) in enumerate(rows):
        lat, lon = z["lat"], z["lon"]
        for ci in range(4):
            ax = axes[ri, ci]
            m = np.where(z["ocean"] & (z["n"][ci] >= min_n), z["mean"][ci], np.nan)
            im = ax.imshow(m, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto")
            ax.set_title(f"S, {str(z['seasons'][ci])}", fontsize=9)
            if ri == len(rows) - 1:
                ax.set_xlabel("longitude (deg E)")
            if ci == 0:
                ax.set_ylabel(f"latitude (deg N)\n{p.name}", fontsize=8)
        fig.colorbar(im, ax=axes[ri].tolist(), label="S (ln CHL / day)", fraction=.02, pad=.01)
    fig.suptitle("Season-mean source along lead-1 trajectories, val days", fontsize=11)
    return fig


def draw_source_by_epoch(run_dirs):
    """Basin-mean lead-1 source S (val) by epoch, from history.json["source"] (step-2 runs)."""
    runs = [(i, p, h) for i, p, h in _loaded(run_dirs) if h.get("source") and h["source"][0]]
    if not runs:
        print("note: no source statistics in history (step-2 --source runs only)"); return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, p, h in runs:
        c, ls = _style(i)
        ax.plot(h["epoch"], [s["mean"] for s in h["source"]], color=c, ls=ls, marker="o", ms=3, label=p.name)
    ax.axhline(0, color="k", lw=.5)
    ax.set(xlabel="epoch", ylabel="basin-mean S, lead 1 (ln CHL / day)", title="Basin-mean lead-1 source by epoch (val)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
