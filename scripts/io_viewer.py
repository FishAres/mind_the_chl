"""Interactive daily map viewer for IO.zarr (ipywidgets + matplotlib).

Data access pattern: the store is chunked (100 days, full lat, full lon), so a
single day costs one chunk read per variable. The viewer loads the whole
100-day block for the panel variables once (~200 MB in RAM) and slices days
out of it in memory, so stepping day-by-day inside a block is instant and
only crossing a block boundary touches disk.
"""
from functools import lru_cache

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

# Okabe-Ito (lines) + perceptually uniform, colorblind-safe colormaps (maps)
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
SEQ, SEQ2, DIV = "viridis", "cividis", "RdBu_r"

# Each panel: variable, title, units, colour scale, optional quiver components.
PANELS = [
    dict(var="CHL_cmes-level3", title="Chlorophyll-a, L3 (observed; gaps = cloud)", units="mg m$^{-3}$",
         cmap=SEQ, norm=lambda: LogNorm(0.03, 10), anom="logratio"),
    dict(var="CHL_cmes-gapfree", title="Chlorophyll-a, L4 gap-free", units="mg m$^{-3}$",
         cmap=SEQ, norm=lambda: LogNorm(0.03, 10), anom="logratio"),
    dict(var="sst", title="Sea surface temperature (ERA5)", units="°C", offset=-273.15,
         cmap="inferno", norm=lambda: Normalize(20, 32), anom="diff", anom_lim=2),
    dict(var="wind_speed", title="10 m wind speed + direction (ERA5)", units="m s$^{-1}$",
         cmap=SEQ2, norm=lambda: Normalize(0, 15), quiver=("u_wind", "v_wind"), anom="diff", anom_lim=6),
    dict(var="curr_speed", title="Surface current speed + direction (OSCAR)", units="m s$^{-1}$",
         cmap=SEQ2, norm=lambda: Normalize(0, 1), quiver=("u_curr", "v_curr"), anom="diff", anom_lim=0.5),
    dict(var="sla", title="Sea level anomaly (CMEMS SL)", units="m",
         cmap=DIV, norm=lambda: TwoSlopeNorm(0, -0.3, 0.3), anom="diff", anom_lim=0.3),
]
QUIVER_STRIDE = 8  # every 8th grid cell = 2° arrows


class DayViewer:
    def __init__(self, ds, clim=None, daily=None, block=100):
        self.ds, self.clim, self.daily, self.block = ds, clim, daily, block
        self.vars = sorted({p["var"] for p in PANELS} | {c for p in PANELS for c in p.get("quiver", ())})
        self.times = pd.DatetimeIndex(ds.time.values)
        self.land = (ds["CHL_cmes-land"] == 2).load()  # tiny 2-D array, fine to hold
        lat, lon = ds.lat.values, ds.lon.values  # lat is descending (32 -> -12)
        assert lat[0] > lat[-1]
        h = 0.125  # half a 0.25° cell
        self.extent = (lon[0] - h, lon[-1] + h, lat[-1] - h, lat[0] + h)
        self._block = lru_cache(maxsize=2)(self._load_block)

    # ---- data -----------------------------------------------------------
    def _load_block(self, b):
        """Load one chunk-aligned block of days for the panel variables (one dask compute)."""
        sl = slice(b * self.block, (b + 1) * self.block)
        return self.ds[self.vars].isel(time=sl).load()

    def day(self, i):
        return self._block(i // self.block).isel(time=i % self.block)

    # ---- plotting -------------------------------------------------------
    def plot(self, i, anomaly=False):
        d = self.day(i)
        date = self.times[i]
        use_anom = anomaly and self.clim is not None
        fig = plt.figure(figsize=(17, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.5], left=0.04, right=0.97, top=0.92, bottom=0.05,
                              wspace=0.22, hspace=0.3)
        for k, p in enumerate(PANELS):
            ax = fig.add_subplot(gs[k // 3, k % 3])
            self._panel(ax, d, p, date, use_anom)
            if k % 3:
                ax.set_ylabel("")
        ax_ts = fig.add_subplot(gs[2, :])
        self._timeseries(ax_ts, date)
        mode = "anomaly vs. monthly climatology (each variable's available years)" if use_anom else "daily values"
        fig.suptitle(f"Indian Ocean, {date:%Y-%m-%d} ({date:%a}) — {mode}", fontsize=14)
        plt.show()

    def _panel(self, ax, d, p, date, use_anom):
        da = d[p["var"]] + p.get("offset", 0)
        cmap, norm, units = p["cmap"], p["norm"](), p["units"]
        if use_anom:
            ref = self.clim[p["var"]].sel(month=date.month) + p.get("offset", 0)
            if p["anom"] == "logratio":
                da, units = np.log10(da / ref), "log$_{10}$(CHL / clim)"
                norm = TwoSlopeNorm(0, -1, 1)
            else:
                lim = p["anom_lim"]
                da, norm = da - ref, TwoSlopeNorm(0, -lim, lim)
                units = f"Δ {units}"
            cmap = DIV
        valid = int(da.where(~self.land).notnull().sum())
        n_ocean = int((~self.land).sum())
        if valid == 0:
            ax.text(0.5, 0.5, "no data on this date", ha="center", va="center", transform=ax.transAxes)
        else:
            im = ax.imshow(da.values, extent=self.extent, origin="upper", cmap=cmap, norm=norm,
                           interpolation="nearest", aspect="equal")
            plt.colorbar(im, ax=ax, label=units, fraction=0.035, pad=0.02, extend="both")
            if "quiver" in p and not use_anom:
                u, v = (d[c].isel(lat=slice(None, None, QUIVER_STRIDE), lon=slice(None, None, QUIVER_STRIDE))
                        for c in p["quiver"])
                spd = np.hypot(u, v)
                ax.quiver(u.lon, u.lat, u / spd, v / spd, color="white", scale=35, width=0.003,
                          headwidth=4, alpha=0.9)  # unit arrows: direction only, colour = speed
        ax.set_facecolor("0.85")
        ax.contour(self.land.lon, self.land.lat, self.land.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        ax.set_title(f"{p['title']}\nvalid ocean pixels: {valid}/{n_ocean}", fontsize=10)
        ax.set_xlabel("longitude (°E)")
        ax.set_ylabel("latitude (°N)")
        ax.set_xlim(self.extent[:2]); ax.set_ylim(self.extent[2:])

    def _timeseries(self, ax, date):
        if self.daily is None:
            ax.text(0.5, 0.5, "run io_summaries.py to enable domain-mean time series",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        win = slice(date - pd.Timedelta(days=365), date + pd.Timedelta(days=365))
        dd = self.daily.sel(time=win)
        ax.plot(dd.time, dd["CHL_cmes-gapfree__mean"], color=OKABE_ITO[2], label="CHL L4 gap-free (left)")
        ax.set_ylabel("domain-mean CHL (mg m$^{-3}$)", color=OKABE_ITO[2])
        ax2 = ax.twinx()
        ax2.plot(dd.time, dd["sst__mean"] - 273.15, color=OKABE_ITO[5], label="SST (right)")
        ax2.set_ylabel("domain-mean SST (°C)", color=OKABE_ITO[5])
        ax.axvline(date, color="k", lw=1, ls="--")
        ax.set_title("Ocean-pixel mean, ±1 year around selected date (dashed line)", fontsize=10)
        ax.set_xlabel("date")
        h = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        ax.legend(h, [x.get_label() for x in h], loc="upper left", fontsize=8)

    # ---- widgets --------------------------------------------------------
    def show(self, start="2020-06-01"):
        i0 = int(self.times.get_indexer([pd.Timestamp(start)], method="nearest")[0])
        n = len(self.times) - 1
        slider = W.IntSlider(value=i0, min=0, max=n, description="day", layout=W.Layout(width="60%"),
                             continuous_update=False, readout=False)
        label = W.HTML()
        step = W.Dropdown(options=[("1 day", 1), ("1 week", 7), ("1 month", 30), ("1 year", 365)],
                          value=1, description="step", layout=W.Layout(width="160px"))
        play = W.Play(value=i0, min=0, max=n, step=1, interval=700, description="play")
        W.jslink((play, "value"), (slider, "value"))
        step.observe(lambda c: (setattr(play, "step", c["new"]), setattr(slider, "step", c["new"])), "value")
        back = W.Button(description="◀", layout=W.Layout(width="45px"))
        fwd = W.Button(description="▶", layout=W.Layout(width="45px"))
        back.on_click(lambda _: setattr(slider, "value", max(0, slider.value - step.value)))
        fwd.on_click(lambda _: setattr(slider, "value", min(n, slider.value + step.value)))
        picker = W.DatePicker(description="jump to", value=self.times[i0].date())
        picker.observe(lambda c: c["new"] and setattr(
            slider, "value", int(self.times.get_indexer([pd.Timestamp(c["new"])], method="nearest")[0])), "value")
        anom = W.Checkbox(value=False, description="anomaly vs monthly climatology",
                          disabled=self.clim is None, indent=False)

        def _label(c=None):
            label.value = f"<b>{self.times[slider.value]:%Y-%m-%d}</b>"
        slider.observe(_label, "value")
        _label()

        out = W.interactive_output(lambda i, a: self.plot(i, a), {"i": slider, "a": anom})
        controls = W.VBox([W.HBox([play, back, slider, fwd, label]), W.HBox([step, picker, anom])])
        display(controls, out)
