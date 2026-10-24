"""Figures for the attention gap-fillers (--model temporal3d --t3d-stem attn / patch / tokdec) in notebooks/progress.ipynb.

  draw_curves(runs, refs, x="epoch" | "steps" | "days")   val hidden-pixel MAE vs epoch / optimiser steps / training days seen
  draw_offsets([cache, ...])                              pooling-weight share by day offset on val hidden pixels, split by whether
                                                          the day is visible at the pixel (+ the "no day" share of patch models)
  draw_maps(cache, i, offsets)                            weight maps of one cached val day by offset, next to the visible mask

The weight figures read analyses/attn_viz/cache/<run>.npz (python -m analyses.attn_viz.build_cache --run <run>; see its docstring):
the trainer's own eval-only path with the recorders of chl_cnn/tattn.py, ptok.py and tokdec.py, on 12 val days by default.
Weight sets per stem:
  attn    "pooling weight": per-pixel softmax over the T window days (day t included), mean over heads
  patch   "day weight": per-pixel softmax over the context days available at the pixel + one "no day" slot (day t excluded)
  tokdec  "patch pooling weight": per-patch softmax over the T days (day t included), mean over heads, shown on the pixels of the patch;
          with --tokdec-blend also "blend day weight" (as patch)
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .plotting import OKABE_ITO, MAP_CMAP
from .runs import CATEGORIES, MODELS, run_config, is_running, is_side_run

ROOT = MODELS.parent
CACHE = ROOT / "analyses" / "attn_viz" / "cache"
ATTN_STEMS = ("attn", "patch", "tokdec")
REFS = ("t3d_dil_c4_60ep_gdl100", "io2_seed1", "io2_seed2")
N_TRAIN = 5540                                                 # IO training days per epoch (config n_days.train when present)


def _cfg(p):
    return run_config(p)                                       # config.json, or the live / logged flags of a running run


def _hist(p):
    return json.loads((Path(p) / "history.json").read_text())


def stem_of(p):
    c = _cfg(p)
    return (c.get("t3d_stem") or "conv") if c.get("model") == "temporal3d" else None


def kind_label(p):
    """Short description of a run's temporal stem, e.g. 'patch own ±8', 'tokdec+blend ±8', 'conv ±4'."""
    c = _cfg(p); s = stem_of(p)
    win = f"±{c.get('context_before') or c.get('context')}" if c.get("context_before") == c.get("context_after") else \
        f"-{c.get('context_before')}/+{c.get('context_after')}"
    if s == "patch":
        return f"patch {c.get('ptok_value', 'own')} {win}"
    if s == "tokdec":
        return f"tokdec{'+blend' if c.get('tokdec_blend') else ''} {win}"
    return f"{s} {win}" if s else (c.get("model") or "?")


def _trained(p):
    c = _cfg(p)
    return (Path(p) / "history.json").exists() and not c.get("eval_only")


def store_of(p):
    return _cfg(p).get("store", "io")


def _t3d_runs(pattern, keep, store=None, main_only=False):
    out = []
    for h in CATEGORIES["pinn/t3d"].glob(f"{pattern}/history.json"):
        p = h.parent
        if keep(stem_of(p)) and _trained(p) and (store is None or store_of(p) == store) and not (main_only and is_side_run(p)):
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def attn_runs(pattern="*", store=None, main_only=False):
    """Trained (not eval-only) runs in pinn/t3d with an attention stem, running ones included, sorted by name.
    store: only runs on that store ('io', 'blk4', ...); main_only: drop smoke / leak / eval runs (runs.is_side_run)."""
    return _t3d_runs(pattern, lambda s: s in ATTN_STEMS, store, main_only)


def ref_runs(pattern="*", store=None, main_only=False):
    """Trained runs in pinn/t3d with the Conv3d stem (references); same filters as attn_runs."""
    return _t3d_runs(pattern, lambda s: s == "conv", store, main_only)


def train_info(p):
    """batch, optimiser steps and training days per epoch, median s/epoch (history epoch_s), epochs logged, done."""
    c, h = _cfg(p), _hist(p)
    b = c.get("batch")
    n_tr = (c.get("n_days") or {}).get("train", N_TRAIN) if isinstance(c.get("n_days"), dict) else N_TRAIN
    spe = math.ceil(n_tr / b) if b else None
    if spe and c.get("max_batches"):
        spe = min(spe, c["max_batches"])
    es = [v for v in h.get("epoch_s", []) if v]
    return dict(batch=b, steps_per_epoch=spe, days_per_epoch=(min(n_tr, spe * b) if spe else None), s_per_epoch=(float(np.median(es)) if es else None),
                n_epochs=len(h.get("val", [])), done=(Path(p) / "metrics.json").exists(), running=is_running(p), n_train=n_tr)


XLAB = dict(epoch="epoch", steps="optimiser steps (thousands)", days="training days seen (thousands)")


def draw_curves(runs, refs=(), x="epoch", ax=None):
    """Val hidden-pixel MAE (ln CHL) of the attention runs (solid) and reference runs (dashed) vs epoch / steps / days seen.
    Returns (fig, rows); rows = (name, kind, batch, s/epoch, epochs, best val, status)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 6.4))
    else:
        fig = ax.figure
    rows, lo, hi = [], [], []
    allr = [(Path(p), "-") for p in runs] + [(Path(p), "--") for p in refs if Path(p) not in set(map(Path, runs))]
    for i, (p, ls) in enumerate(allr):
        h = _hist(p)
        v = np.asarray(h.get("val", []), dtype=float)
        if not len(v):
            print(f"skip {p.name}: no validation epochs logged yet")
            continue
        inf = train_info(p)
        ep = np.arange(1, len(v) + 1)
        if x == "steps" and inf["steps_per_epoch"]:
            xx = ep * inf["steps_per_epoch"] / 1e3
        elif x == "days" and inf["days_per_epoch"]:
            xx = ep * inf["days_per_epoch"] / 1e3
        else:
            xx = ep
        sp = f", {inf['s_per_epoch']:.0f} s/ep" if inf["s_per_epoch"] else ""
        st = "done" if inf["done"] else ("running" if inf["running"] else "unfinished")
        lab = f"{p.name} [{kind_label(p)}], batch {inf['batch']}{sp}, {len(v)} ep, best {v.min():.4f}{'' if st == 'done' else f' ({st})'}"
        ax.plot(xx, v, color=OKABE_ITO[i % len(OKABE_ITO)], ls=ls, lw=1.6 if ls == "-" else 1.1, label=lab)
        rows.append((p.name, kind_label(p), inf["batch"], inf["s_per_epoch"], len(v), float(v.min()), st))
        s = np.sort(v); lo.append(s[0]); hi.append(s[int(0.75 * (len(s) - 1))])
    if lo:
        ax.set_ylim(0.97 * min(lo), 1.03 * max(hi))
    ax.set(xlabel=XLAB[x], ylabel="val MAE on hidden pixels, ln(CHL)",
           title=f"Validation MAE by {XLAB[x].split(' (')[0]}: attention runs (solid) and reference runs (dashed)")
    ax.grid(alpha=.3)
    if ax.lines:
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    return fig, rows


# ---------------------------------------------------------------- weight caches
def cache_path(run):
    return CACHE / f"{Path(str(run)).name}.npz"


def cache_cmd(run, device="cpu"):
    name = Path(str(run)).name
    if device == "cpu":
        return f"cd {ROOT} && python -W ignore -m analyses.attn_viz.build_cache --run {name} --days 12"
    return f"cd {ROOT} && flock ~/mind_the_chl_gap/logs/gpu.lock python -W ignore -m analyses.attn_viz.build_cache --run {name} --days 12 --device cuda"


_CACHE = {}


def load_cache(run):
    """The run's weight cache as a dict (None when missing); adds 'sets' = {label: (n, T[+1], H, W) per-pixel weights, float32}."""
    f = cache_path(run)
    if not f.exists():
        return None
    key = (str(f), f.stat().st_mtime)
    if key in _CACHE:
        return _CACHE[key]
    z = dict(np.load(f, allow_pickle=False))
    z = {k: (v.item() if v.ndim == 0 else v) for k, v in z.items()}
    sets = {}
    if z["stem"] == "attn":
        sets["pooling weight"] = z["w"]
    elif z["stem"] == "patch":
        sets["day weight"] = z["w"]
    else:
        sets["patch pooling weight"] = _patch_to_pixels(z)
        if "w" in z:
            sets["blend day weight"] = z["w"]
    z["sets"] = sets
    _CACHE[key] = z
    return z


def cache_stale(run, z):
    """True when the run's model.pt is newer than the cache (e.g. the run finished after the cache was built)."""
    m = Path(_find(run)) / "model.pt"
    return m.exists() and m.stat().st_mtime > cache_path(run).stat().st_mtime


def _find(run):
    from .runs import find_run
    return run if Path(str(run)).is_dir() else find_run(str(run))


def _patch_to_pixels(z):
    """tokdec pool_w (n, Np, T) -> (n, T, H, W): each pixel gets the weights of its P x P patch (0 on land)."""
    P, nPx, nPy = int(z["P"]), int(z["nPx"]), int(z["nPy"])
    H, W = z["ocean"].shape
    g2p = np.full(nPy * nPx, -1); g2p[z["patch_idx"]] = np.arange(len(z["patch_idx"]))
    yy, xx = np.meshgrid(np.arange(H) // P, np.arange(W) // P, indexing="ij")
    pix = g2p[yy * nPx + xx]                                   # (H, W) ocean-patch index, -1 on land-only patches
    w = z["pool_w"].astype("float32")[:, np.maximum(pix, 0), :]  # (n, H, W, T)
    w = np.moveaxis(w, -1, 1) * ((pix >= 0) & z["ocean"])[None, None]
    return w


def offset_shares(z, wset):
    """Over the val hidden pixels of all cached days: mean weight on visible / not-visible days by offset, and the 'no day' share."""
    w = z["sets"][wset].astype("float32")
    T = z["vis"].shape[1]
    hid = z["hid"][:, None]                                     # (n, 1, H, W)
    n_hid = int(z["hid"].sum())
    vis = z["vis"]
    sv = (w[:, :T] * (vis & hid)).sum((0, 2, 3)) / max(n_hid, 1)
    sn = (w[:, :T] * (~vis & hid)).sum((0, 2, 3)) / max(n_hid, 1)
    none = float((w[:, T] * z["hid"]).sum() / max(n_hid, 1)) if w.shape[1] == T + 1 else None
    return dict(offsets=z["offsets"], vis=sv, novis=sn, none=none, n_hid=n_hid, n_days=len(z["days"]),
                frac_vis=(vis & hid).sum((0, 2, 3)) / max(n_hid, 1))


def draw_offsets(caches, ncols=2):
    """One panel per (run, weight set): stacked mean weight per val hidden pixel by day offset, on days visible / not visible at
    the pixel, and the 'no day' slot; the bars of a panel sum to 1."""
    panels = [(z, s) for z in caches for s in z["sets"]]
    if not panels:
        return None
    ncols = min(ncols, len(panels)); nrows = math.ceil(len(panels) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (z, s) in zip(axs.ravel(), panels):
        r = offset_shares(z, s)
        off = np.asarray(r["offsets"])
        ax.bar(off, r["vis"], color=OKABE_ITO[0], label="day visible at the pixel")
        ax.bar(off, r["novis"], bottom=r["vis"], color=OKABE_ITO[1], label="day not visible at the pixel")
        xt, xl = list(off), [f"{k:+d}" if k else "t" for k in off]
        if r["none"] is not None:
            xn = off.max() + 2
            ax.bar([xn], [r["none"]], color=OKABE_ITO[2], label="no day")
            xt.append(xn); xl.append("none")
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8)
        tot_v, tot_n = r["vis"].sum(), r["novis"].sum()
        txt = f"visible {tot_v:.2f}, not visible {tot_n:.2f}" + (f", no day {r['none']:.2f}" if r["none"] is not None else "")
        ax.set(xlabel="day offset from t (days)", ylabel="mean weight per hidden pixel",
               title=f"{z['run']}: {s}, {r['n_days']} val days, {r['n_hid']:,} hidden px\nsums: {txt}")
        ax.title.set_fontsize(10)
        ax.set_ylim(0, max(np.max(r["vis"] + r["novis"]), r["none"] or 0) * 1.3)
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7.5, loc="upper center", ncol=3, frameon=False)
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Weight by day offset on val hidden pixels (mean over heads; bars of a panel sum to 1)")
    fig.tight_layout()
    return fig


def weights_day_t(z):
    """Whether the stem's weights cover day t (per-pixel attention pooling, tokdec patch pooling)."""
    return z["stem"] in ("attn", "tokdec")


def draw_maps(z, i, offsets=(-4, -2, -1, 1, 2, 4), with_t=True, with_none=True):
    """Weight maps of cached val day i by offset (columns), with the day's visible mask; patch models also the patch score and pixel
    term (and the local confidence conv(m) for lavg). Land blank; north up."""
    T = z["vis"].shape[1]; ti = int(z["t_index"])
    offs = sorted({int(k) for k in offsets if -ti <= int(k) <= T - 1 - ti} | ({0} if with_t and weights_day_t(z) else set()))
    rows = [(s, z["sets"][s][i], "w") for s in z["sets"]]
    rows.append(("visible in input", z["vis"][i].astype("float32"), "vis"))
    if "conf" in z:
        rows.append(("local confidence conv(m)", z["conf"][i].astype("float32"), "conf"))
    if "s_pixel" in z:
        rows.append(("patch score (interp.)", z["s_patch"][i].astype("float32"), "score"))
        rows.append(("pixel term", z["s_pixel"][i].astype("float32"), "score"))
    has_none = with_none and any(r[1].shape[0] == T + 1 for r in rows)
    cols = [("off", k) for k in offs] + ([("none", None)] if has_none else [])
    ocean, lat, lon = z["ocean"], z["lat"], z["lon"]
    org = "lower" if lat[0] < lat[-1] else "upper"
    ext = [lon[0], lon[-1], min(lat[0], lat[-1]), max(lat[0], lat[-1])]
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.55 * len(cols) + 1.2, 2.25 * len(rows) + 0.9), squeeze=False,
                            sharex=True, sharey=True, layout="constrained")
    for r, (name, arr, kind) in enumerate(rows):
        sel = [ti + k for k in offs]
        if kind == "w":
            vals = arr[sel + ([T] if arr.shape[0] == T + 1 and has_none else [])][:, ocean]
            vmin, vmax, cm = 0.0, max(float(np.percentile(vals, 99.5)), 1e-3), MAP_CMAP
        elif kind in ("vis", "conf"):
            vmin, vmax, cm = 0.0, 1.0, "cividis"
        else:
            vals = arr[sel][:, ocean]
            vmin, vmax, cm = *np.percentile(vals, [2, 98]), MAP_CMAP
        im = None
        for c, (ck, k) in enumerate(cols):
            ax = axs[r, c]
            if ck == "none":
                if kind == "w" and arr.shape[0] == T + 1:
                    img = arr[T]
                else:
                    ax.axis("off"); continue
            else:
                img = arr[ti + k]
            im = ax.imshow(np.where(ocean, img, np.nan), extent=ext, origin=org, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto",
                           interpolation="nearest")
            ttl = "no day" if ck == "none" else ("day t" if k == 0 else f"day t{k:+d}")
            ax.set_title(f"{ttl}", fontsize=9)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel(f"{name}\nlat (°N)", fontsize=8.5)
            if r == len(rows) - 1:
                ax.set_xlabel("lon (°E)", fontsize=8)
        if im is not None:
            fig.colorbar(im, ax=list(axs[r]), shrink=.9, pad=0.01, aspect=12, label={"w": "weight", "vis": "1 = visible", "conf": "conv(m)",
                                                                                    "score": "logit"}[kind])
    obs_t = (z["vis"][i, ti] | z["hid"][i])[ocean].mean()
    stem = {"patch": f"patch {z.get('value') or ''}".strip(), "tokdec": "tokdec+blend" if z.get("blend") else "tokdec"}.get(z["stem"], z["stem"])
    fig.suptitle(f"{z['run']} ({stem}): weights by day offset, val day {z['dates'][i]} "
                 f"(observed {obs_t:.0%} of ocean, {int(z['hid'][i].sum()):,} hidden px)", fontsize=11)
    return fig
