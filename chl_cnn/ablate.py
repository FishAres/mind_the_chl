"""Input ablation for eval-only scoring (io5 track, 2026-10-24; train_pinn.py --ablate GROUP --ablate-mode {mean, doyclim}).

Groups (train_pinn.py --model temporal3d, conv / patch / tokdec stems):
  physics, each on every day of the input window (context days and day t):
    sst, so, mlotst, sla, currents (u_curr + v_curr), winds (u_wind + v_wind), air_temp, physics (all 9 at once)
      mode mean:    the channel is 0 in the z-scored space the model sees (= the feature's training mean, train_pinn feat_stats)
      mode doyclim: the feature's per-pixel day-of-year climatology from the training days (circular +-15-day window = 31 days,
                    NaN-aware mean), for the calendar day of each window day, z-scored with the same feat_stats, NaN -> 0,
                    rounded through float16 as the trainer's feature store; window days missing from the store stay 0 (as the
                    trainer gives them)
  CHL (mode mean only):
    ctx_chl   every window day != t: CHL value 0 and visible mask 0 (the model sees day t's CHL + physics)
    day_chl   day t only: CHL value 0 and visible mask 0
  static (mode mean only; the replacement constant is the channel's mean over ocean pixels of the training days, filled on every px):
    clim         DOY climatology: the static channel AND the per-window-day climatology the token stems subtract from CHL (clim3d)
    clim_static  the static channel only (clim3d unchanged; differs from clim only for the patch / tokdec stems)
    latlon       the lat and lon channels of --coords
    depth        the log-depth channel of --coords
    year         the --year-input channel
The ablation replaces values in copies of the model inputs (x3d, x_static, clim3d) right before the forward pass. Targets, hidden /
observed masks, the --residual output base and the physics-residual diagnostics are not touched.

DOY climatology cache: ~/mind_the_chl_gap/data/io5/doyclim_io.npz (raw units, (366, H, W) float32 per feature; built once by
build_doyclim; python -m chl_cnn.ablate builds it for the IO store and the default split).
"""
from pathlib import Path

import numpy as np

PHYS = ["sst", "so", "air_temp", "u_wind", "v_wind", "sla", "mlotst", "u_curr", "v_curr"]
FEAT_GROUPS = {"sst": ["sst"], "so": ["so"], "mlotst": ["mlotst"], "sla": ["sla"], "currents": ["u_curr", "v_curr"],
               "winds": ["u_wind", "v_wind"], "air_temp": ["air_temp"], "physics": PHYS}
CHL_GROUPS = ("ctx_chl", "day_chl")
STATIC_GROUPS = ("clim", "clim_static", "latlon", "depth", "year")
ALL_GROUPS = list(FEAT_GROUPS) + list(CHL_GROUPS) + list(STATIC_GROUPS)
CACHE = Path.home() / "mind_the_chl_gap" / "data" / "io5" / "doyclim_io.npz"
HALFWIN = 15


def doy_of(times):
    """0-based day of year (0..365), as train_pinn / train2 compute it."""
    t = np.asarray(times)
    return (t - t.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)


def build_doyclim(read, features, tr, times, halfwin=HALFWIN, cache=CACHE):
    """Per-pixel DOY climatology (raw units) of each feature over the training days tr: NaN-aware mean over the training days
    whose day of year lies within +-halfwin days (circular over 366). read(feature, idx) -> (len(idx), H, W) array.
    Returns {feature: (366, H, W) float32, NaN where no finite value}; cached (checked against tr and halfwin)."""
    tr = np.asarray(tr)
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["train_idx"], tr) and int(z["halfwin"]) == halfwin and all(f in z.files for f in features):
            return {f: z[f] for f in features}
    doy = doy_of(times)[tr]
    out = {}
    for f in features:
        x = read(f, tr).astype("float32")
        s = np.zeros((366,) + x.shape[1:]); c = np.zeros((366,) + x.shape[1:])
        for k in range(366):
            sel = x[doy == k]
            if len(sel):
                s[k] = np.nansum(sel, 0); c[k] = np.isfinite(sel).sum(0)
        ss = sum(np.roll(s, j, 0) for j in range(-halfwin, halfwin + 1)); cc = sum(np.roll(c, j, 0) for j in range(-halfwin, halfwin + 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f] = np.where(cc > 0, ss / cc, np.nan).astype("float32")
        print(f"doyclim {f}: {len(tr)} training days, min count per (doy, px) where > 0: {int(cc[cc > 0].min())}, "
              f"px-doys without data {int((cc == 0).sum())}", flush=True)
        del x
    cache.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["train_idx"], tr) and int(z["halfwin"]) == halfwin:
            old = {k: z[k] for k in z.files if k in PHYS and k not in out}
    np.savez(cache, train_idx=tr, halfwin=halfwin, **old, **out)
    return out


if __name__ == "__main__":                                     # build the IO cache once: python -m chl_cnn.ablate
    from . import store as S, data as D
    d = S.load_store()
    tr, _, _ = D.split_by_years(d["time"], D.VAL_YEARS, D.TEST_YEARS)
    build_doyclim(lambda f, idx: np.asarray(d[f][idx]), PHYS, tr, d["time"])
    print("wrote", CACHE)
