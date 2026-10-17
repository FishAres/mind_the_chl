"""Figures for the results notebooks. Okabe-Ito colours for lines, viridis for maps."""
import numpy as np
import matplotlib.pyplot as plt

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=OKABE_ITO)
MAP_CMAP = "viridis"
ERR_CMAP = "PuOr"


def plot_history(histories, ax=None):
    """histories: {run_name: {'train': [...], 'val': [...]}}. Loss = masked MAE in ln(CHL)."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    for k, (name, h) in enumerate(histories.items()):
        c = OKABE_ITO[k % len(OKABE_ITO)]
        ax.plot(h["train"], color=c, ls="--", label=f"{name} train")
        ax.plot(h["val"], color=c, ls="-", label=f"{name} val")
    ax.set(xlabel="epoch", ylabel="MAE on scored pixels, ln(CHL)", title="Training and validation loss by epoch")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    return ax


def _extent(lat, lon):
    return [lon[0], lon[-1], lat[-1], lat[0]]


def plot_day_maps(lat, lon, panels, date, vmin=None, vmax=None, err=None, ncols=None):
    """panels: {title: 2D ln(CHL) array}; err: optional {title: 2D error array} drawn with diverging cmap."""
    err = err or {}
    n = len(panels) + len(err)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    axes = axes.ravel()
    allv = np.concatenate([v[np.isfinite(v)] for v in panels.values()])
    vmin = np.percentile(allv, 1) if vmin is None else vmin
    vmax = np.percentile(allv, 99) if vmax is None else vmax
    ext = _extent(lat, lon)
    for ax, (t, v) in zip(axes, panels.items()):
        im = ax.imshow(v, extent=ext, cmap=MAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set(title=t, xlabel="lon (°E)", ylabel="lat (°N)")
        fig.colorbar(im, ax=ax, label="ln CHL (mg m$^{-3}$)", shrink=.85)
    for ax, (t, v) in zip(axes[len(panels):], err.items()):
        lim = np.nanpercentile(np.abs(v), 99)
        im = ax.imshow(v, extent=ext, cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto")
        ax.set(title=t, xlabel="lon (°E)", ylabel="lat (°N)")
        fig.colorbar(im, ax=ax, label="pred − true, ln CHL", shrink=.85)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"ln(CHL) maps, {date}", y=1.0)
    fig.tight_layout()
    return fig


def plot_scatter(y_true, y_pred, mask, title, ax=None, lim=(-3.5, 4)):
    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    m = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    ax.hexbin(y_true[m], y_pred[m], gridsize=60, bins="log", cmap=MAP_CMAP, extent=lim + lim)
    ax.plot(lim, lim, color="#D55E00", lw=1)
    ax.set(xlabel="true ln CHL (hidden pixels)", ylabel="predicted ln CHL", title=title, xlim=lim, ylim=lim)
    return ax


def plot_daily_mae(dates, series, title="Daily MAE on hidden test pixels", ax=None):
    """series: {name: per-day MAE array}."""
    ax = ax or plt.subplots(figsize=(9, 3.6))[1]
    for name, s in series.items():
        ax.plot(dates, s, marker=".", label=name)
    ax.set(xlabel="date", ylabel="MAE, ln(CHL)", title=title)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    return ax


def plot_mae_by_bin(bins, title="MAE by true ln(CHL) bin (hidden test pixels)", ax=None):
    """bins: {name: (centres, mae, n)}"""
    ax = ax or plt.subplots(figsize=(7, 3.8))[1]
    for name, (c, mae, n) in bins.items():
        ax.plot(c, mae, marker="o", label=name)
    ax.set(xlabel="true ln CHL (bin centre)", ylabel="MAE, ln(CHL)", title=title)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    return ax
