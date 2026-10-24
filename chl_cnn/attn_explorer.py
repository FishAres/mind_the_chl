"""Figures and widgets of notebooks/attention_explorer.ipynb: one IO gap-filler (optionally two, side by side) on one of 12 preset val or
test days, read from analyses/attn_viz/explorer_cache/<run>.npz (analyses/attn_viz/build_explorer.py; nothing is computed on the model).

  draw_recon(zs, i)                      (a) visible input, prediction, truth and error on day t's hidden pixels, per model
  draw_attention(z, i, offsets, ...)     (b) weight maps by day offset with the day's visibility (attn_viz.draw_maps), per model
  draw_day_offsets(zs, i)                (b) mean weight per hidden pixel by offset for that day (visible / not visible / no day)
  draw_attr(zs, i, group)                (c) share of |integrated gradients| per input group, and the map of one group
  draw_context(z, i)                     (d) SST with surface-current vectors on day t

Cache day index i: 0..11 val, 12..23 test (z["split"]); day(z, split, k) maps (split, k) to i. Maps: north up (IO store rows run
north to south), land blank. Plot code reuses chl_cnn/attn_viz.py (weight sets, per-offset shares, weight maps).
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .plotting import OKABE_ITO, MAP_CMAP, ERR_CMAP
from . import attn_viz as AV

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "analyses" / "attn_viz" / "explorer_cache"
RUNS = ("io2_tattn8", "io3_ptok4", "io3_ptok4_lavg", "io4_tokdec", "io4_tokdec_blend", "t3d_dil_c4_60ep_gdl100")
LABEL = {"attn": "per-pixel attention", "patch": "patch tokens", "tokdec": "token decoder", "conv": "Conv3d stem"}


def cache_path(run):
    return CACHE / f"{run}.npz"


def build_cmd(run):
    return f"cd {ROOT} && python -W ignore -m analyses.attn_viz.build_explorer --run {run}"


_C = {}


def load(run):
    """The run's explorer cache as a dict (None when missing); 'sets' = {label: (n, T[+1], H, W) per-pixel weights} as attn_viz."""
    f = cache_path(run)
    if not f.exists():
        return None
    key = (str(f), f.stat().st_mtime)
    if key in _C:
        return _C[key]
    z = dict(np.load(f, allow_pickle=False))
    z = {k: (v.item() if v.ndim == 0 else v) for k, v in z.items()}
    sets = {}
    w = np.nan_to_num(z["w"]) if "w" in z else None              # land NaN -> 0 (attn_viz sums weights over masks)
    if z["stem"] == "attn":
        sets["pooling weight"] = w
    elif z["stem"] == "patch":
        sets["day weight"] = w
    elif z["stem"] == "tokdec":
        sets["patch pooling weight"] = AV._patch_to_pixels(z)  # each pixel gets its P x P patch's weights (blocks, nearest)
        if w is not None:
            sets["blend day weight"] = w
    z["sets"] = sets
    z["kind"] = {"patch": f"patch tokens, {z.get('value')}", "tokdec": "token decoder" + (" + blend" if z.get("blend") else "")}.get(
        z["stem"], LABEL.get(z["stem"], z["stem"]))
    _C[key] = z
    return z


def missing(runs=RUNS):
    """{run: build command} for runs without an explorer cache."""
    return {r: build_cmd(r) for r in runs if not cache_path(r).exists()}


def day(z, split, k):
    """Cache index of the k-th preset day of split ('val' / 'test')."""
    return int(np.flatnonzero(z["split"] == split)[k])


def _geo(z):
    lat, lon = z["lat"], z["lon"]
    org = "lower" if lat[0] < lat[-1] else "upper"
    return org, [lon[0], lon[-1], min(lat[0], lat[-1]), max(lat[0], lat[-1])]


def _obs_frac(z, i):
    oc = z["ocean"]
    return float(np.isfinite(z["truth"][i].astype("float32"))[oc].mean())


def day_label(z, i):
    return f"{z['split'][i]} day {z['dates'][i]} (observed {_obs_frac(z, i):.0%} of ocean, {int(z['n_hid'][i]):,} hidden px)"


def _map(ax, z, img, cmap, vmin, vmax, title):
    org, ext = _geo(z)
    im = ax.imshow(np.where(z["ocean"], img, np.nan), extent=ext, origin=org, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return im


# ------------------------------------------------------------------ (a) reconstruction
def draw_recon(zs, i):
    """One row per model: visible ln CHL on day t, prediction, truth on hidden px, error on hidden px. Shared colour scales."""
    zs = [z for z in zs if z is not None]
    z0 = zs[0]
    truth = z0["truth"][i].astype("float32"); hid = z0["hid"][i]
    vis_t = z0["vis"][i, int(z0["t_index"])]
    lo, hi = np.nanpercentile(truth, [1, 99])
    errs = [np.where(hid, z["pred"][i].astype("float32") - truth, np.nan) for z in zs]
    lim = max(float(np.nanpercentile(np.abs(e), 99)) if np.isfinite(e).any() else 1.0 for e in errs)
    fig, axs = plt.subplots(len(zs), 4, figsize=(17, 3.9 * len(zs) + 0.6), squeeze=False, layout="constrained", sharex=True, sharey=True)
    for r, (z, err) in enumerate(zip(zs, errs)):
        assert np.array_equal(z["hid"][i], hid) and z["dates"][i] == z0["dates"][i]
        ims = [_map(axs[r, 0], z, np.where(vis_t, truth, np.nan), MAP_CMAP, lo, hi, "input: visible ln CHL, day t"),
               _map(axs[r, 1], z, z["pred"][i].astype("float32"), MAP_CMAP, lo, hi, "prediction, day t"),
               _map(axs[r, 2], z, np.where(hid, truth, np.nan), MAP_CMAP, lo, hi, "truth on hidden px"),
               _map(axs[r, 3], z, err, ERR_CMAP, -lim, lim, "error on hidden px (pred − truth)")]
        fig.colorbar(ims[2], ax=list(axs[r, :3]), shrink=.9, pad=0.01, aspect=25, label="ln CHL")
        fig.colorbar(ims[3], ax=axs[r, 3], shrink=.9, pad=0.01, aspect=15, label="pred − truth, ln CHL")
        axs[r, 0].set_ylabel(f"{z['run']}\n[{z['kind']}]\nlat (°N)", fontsize=8.5)
        axs[r, 3].set_title(f"error on hidden px: MAE {z['mae'][i]:.4f}, n = {int(z['n_hid'][i]):,}", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("lon (°E)", fontsize=8)
    fig.suptitle(f"Reconstruction of day t, {day_label(z0, i)}; MAE in ln CHL over hidden px", fontsize=11)
    return fig


# ------------------------------------------------------------------ (b) attention
def has_weights(z):
    return bool(z is not None and z["sets"])


def draw_attention(z, i, offsets=(-4, -2, -1, 1, 2, 4), with_t=True, with_none=True):
    """attn_viz.draw_maps on the explorer cache (weights, visible mask, score terms of patch / tokdec+blend models)."""
    fig = AV.draw_maps(z, i, offsets, with_t=with_t, with_none=with_none)
    fig.suptitle(f"{z['run']} [{z['kind']}]: weights by day offset, {day_label(z, i)}", fontsize=11)
    return fig


def draw_day_offsets(zs, i):
    """Mean weight per day-t hidden pixel by day offset for one day, split by whether the day is visible at the pixel, plus the
    'no day' slot (patch models, tokdec blend); one panel per (model, weight set); the bars of a panel sum to 1."""
    panels = []
    for z in zs:
        if not has_weights(z):
            continue
        one = dict(sets={s: w[i:i + 1] for s, w in z["sets"].items()}, vis=z["vis"][i:i + 1], hid=z["hid"][i:i + 1], offsets=z["offsets"],
                   days=z["days"][i:i + 1])
        for s in z["sets"]:
            panels.append((z, s, AV.offset_shares(one, s)))
    if not panels:
        return None
    nc = min(2, len(panels)); nr = -(-len(panels) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(7.4 * nc, 3.5 * nr + 0.5), squeeze=False, layout="constrained")
    for ax, (z, s, r) in zip(axs.ravel(), panels):
        off = np.asarray(r["offsets"])
        ax.bar(off, r["vis"], color=OKABE_ITO[0], label="day visible at the pixel")
        ax.bar(off, r["novis"], bottom=r["vis"], color=OKABE_ITO[1], label="day not visible at the pixel")
        xt, xl = list(off), [f"{k:+d}" if k else "t" for k in off]
        if r["none"] is not None:
            xn = off.max() + 2
            ax.bar([xn], [r["none"]], color=OKABE_ITO[2], label="no day")
            xt.append(xn); xl.append("none")
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8)
        ax.set_ylim(0, max(np.max(r["vis"] + r["novis"]), r["none"] or 0) * 1.3)
        ax.set(xlabel="day offset from t (days)", ylabel="mean weight per hidden px")
        ax.set_title(f"{z['run']}: {s}, n = {r['n_hid']:,} hidden px", fontsize=9.5)
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7.5, loc="upper left", ncol=3, frameon=False)
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Weight by day offset on day t's hidden pixels, {zs[0]['split'][i]} day {zs[0]['dates'][i]}", fontsize=11)
    return fig


# ------------------------------------------------------------------ (c) attribution
def attr_shares(z, i):
    a = np.asarray(z["attr_abs"][i], dtype="float64")
    return a / a.sum()


def draw_attr(zs, i, group="context CHL"):
    """Left: share of the day's total |IG attribution| per input group, one bar colour per model. Right: |attribution| map of one group
    per model (summed over the group's channels and window days)."""
    zs = [z for z in zs if z is not None]
    groups = [str(g) for g in zs[0]["groups"]]
    gk = groups.index(group)
    fig = plt.figure(figsize=(6.2 + 5.2 * len(zs), 4.8), layout="constrained")
    gs = fig.add_gridspec(1, 1 + len(zs), width_ratios=[1.25] + [1] * len(zs))
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(groups)); bh = 0.8 / len(zs)
    for m, z in enumerate(zs):
        ax.barh(y + (m - (len(zs) - 1) / 2) * bh, attr_shares(z, i), bh, color=OKABE_ITO[m], label=z["run"])
    ax.set_yticks(y, groups, fontsize=8.5); ax.invert_yaxis()
    ax.get_yticklabels()[gk].set_fontweight("bold")
    ax.set(xlabel="share of total |attribution| for the day")
    ax.set_title("|IG attribution| share by input group", fontsize=9.5)
    ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=7.5, loc="lower right")
    for m, z in enumerate(zs):
        a2 = fig.add_subplot(gs[1 + m])
        mp = z["attr_map"][i, gk].astype("float32")
        v = mp[z["ocean"] & np.isfinite(mp)]
        hi = float(np.percentile(v, 99.5)) if v.size and np.percentile(v, 99.5) > 0 else 1e-9
        org, ext = _geo(z)
        im = a2.imshow(np.where(z["ocean"], np.clip(mp, hi * 1e-3, None), np.nan), extent=ext, origin=org, cmap="magma", aspect="auto",
                       interpolation="nearest", norm=LogNorm(hi * 1e-3, hi))
        a2.set_title(f"{z['run']}: |attribution| of {group}", fontsize=9); a2.tick_params(labelsize=7)
        fig.colorbar(im, ax=a2, shrink=.85, pad=0.01, label="|IG| per px, ln CHL (log scale, 99.5th pct × [0.001, 1])")
        a2.set_xlabel("lon (°E)", fontsize=8); a2.set_ylabel("lat (°N)", fontsize=8)
    txt = "; ".join(f"{z['run']}: f(x) − f(baseline) {z['f_x'][i] - z['f_base'][i]:+.3f}, completeness error {z['ig_err'][i]:+.1e}" for z in zs)
    fig.suptitle(f"Integrated gradients of mean predicted ln CHL over day t's hidden px (n = {int(zs[0]['n_hid'][i]):,}), "
                 f"{zs[0]['split'][i]} day {zs[0]['dates'][i]}\n{txt}", fontsize=10)
    return fig


# ------------------------------------------------------------------ (d) context
def draw_context(z, i, step=6):
    """SST on day t with surface-current vectors (every step-th pixel each way) and the hidden-pixel outline."""
    sst = z["sst"][i].astype("float32"); u = z["u_curr"][i].astype("float32"); v = z["v_curr"][i].astype("float32")
    fig, ax = plt.subplots(figsize=(9.5, 7.2), layout="constrained")
    lo, hi = np.nanpercentile(sst[z["ocean"]], [2, 98]) if np.isfinite(sst[z["ocean"]]).any() else (0, 1)
    im = _map(ax, z, sst, "cividis", lo, hi, "")
    fig.colorbar(im, ax=ax, shrink=.85, pad=0.01, label="SST (°C)")
    s = slice(step // 2, None, step)
    LON, LAT = np.meshgrid(z["lon"], z["lat"])
    ok = np.isfinite(u) & np.isfinite(v) & z["ocean"]
    uu, vv = np.where(ok, u, np.nan)[s, s], np.where(ok, v, np.nan)[s, s]
    q = ax.quiver(LON[s, s], LAT[s, s], uu, vv, color="k", scale=12, width=0.0022, headwidth=3.5)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.01, 0.01), 0.19, 0.055, transform=ax.transAxes, fc="white", ec="0.4", lw=0.6, zorder=4))
    ax.quiverkey(q, 0.065, 0.037, 0.5, "0.5 m/s", labelpos="E", coordinates="axes", fontproperties=dict(size=8),
                 labelcolor="k").set_zorder(5)
    ax.contour(z["lon"], z["lat"], z["hid"][i].astype(float), levels=[0.5], colors=OKABE_ITO[5], linewidths=0.7)
    ax.set(xlabel="lon (°E)", ylabel="lat (°N)")
    ax.set_title(f"SST and surface currents (every {step}th px), {z['split'][i]} day {z['dates'][i]}; blue outline: hidden px", fontsize=10)
    return fig


# ------------------------------------------------------------------ widgets
class Explorer:
    """Shared controls (model, optional second model, split, day) and one output per panel: .controls(), .recon(), .attention(),
    .attribution(), .context()."""

    def __init__(self, runs=RUNS, a="io4_tokdec_blend", b=None):
        import ipywidgets as W
        self.W = W
        for r, cmd in missing(runs).items():
            print(f"{r}: no explorer cache; build it with\n  {cmd}")
        have = [r for r in runs if cache_path(r).exists()]
        self.ok = bool(have)
        if not have:
            return
        a = a if a in have else have[0]
        self.m_a = W.Dropdown(options=have, value=a, description="model")
        self.m_b = W.Dropdown(options=[("none", None)] + [(r, r) for r in have], value=b if b in have else None, description="compare with")
        self.split = W.ToggleButtons(options=["val", "test"], value="val", description="split")
        self.k = W.IntSlider(min=0, max=11, value=0, description="preset day", continuous_update=False)
        self.lab = W.Label()
        self.offs = W.SelectMultiple(options=[(f"t{k:+d}", k) for k in range(-8, 9) if k], value=(-4, -2, -1, 1, 2, 4), rows=6,
                                     description="offsets")
        self.with_t = W.Checkbox(value=True, description="day t (stems that weight it)")
        self.with_none = W.Checkbox(value=True, description='"no day" column')
        self.group = W.Dropdown(options=list(load(a)["groups"]), value="context CHL", description="map group")
        self.outs = {k: W.Output() for k in ("a", "b", "c", "d")}
        for w in (self.m_a, self.m_b, self.split, self.k):
            w.observe(lambda *_: self.redraw(), "value")
        for w in (self.offs, self.with_t, self.with_none):
            w.observe(lambda *_: self.redraw("b"), "value")
        self.group.observe(lambda *_: self.redraw("c"), "value")

    def zs(self):
        return [load(self.m_a.value)] + ([load(self.m_b.value)] if self.m_b.value and self.m_b.value != self.m_a.value else [])

    def i(self):
        return day(load(self.m_a.value), self.split.value, self.k.value)

    def _show(self, key, fn):
        from IPython.display import display, clear_output
        with self.outs[key]:
            clear_output(wait=True)
            fig = fn()
            if fig is not None:
                display(fig)
                plt.close(fig)

    def redraw(self, which="abcd"):
        if not self.ok:
            return
        zs, i = self.zs(), self.i()
        self.lab.value = day_label(zs[0], i)
        if "a" in which:
            self._show("a", lambda: draw_recon(zs, i))
        if "b" in which:
            def _b():
                figs = []
                for z in zs:
                    if not has_weights(z):
                        print(f"{z['run']} [{z['kind']}]: no attention or pooling weights")
                        continue
                    figs.append(draw_attention(z, i, self.offs.value, self.with_t.value, self.with_none.value))
                bar = draw_day_offsets(zs, i)
                from IPython.display import display
                for f in figs + ([bar] if bar is not None else []):
                    display(f); plt.close(f)
            self._show("b", _b)
        if "c" in which:
            self._show("c", lambda: draw_attr(zs, i, self.group.value))
        if "d" in which:
            self._show("d", lambda: draw_context(zs[0], i))

    def _panel(self, key, *extra):
        from IPython.display import display
        if not self.ok:
            print("no explorer cache found (see the build commands above)")
            return
        display(self.W.VBox(list(extra) + [self.outs[key]]))
        self.redraw(key)

    def controls(self):
        from IPython.display import display
        if self.ok:
            display(self.W.HBox([self.m_a, self.m_b]), self.W.HBox([self.split, self.k]), self.lab)

    def recon(self):
        self._panel("a")

    def attention(self):
        self._panel("b", self.W.HBox([self.offs, self.W.VBox([self.with_t, self.with_none])]))

    def attribution(self):
        self._panel("c", self.group)

    def context(self):
        self._panel("d")
