"""Shared helpers for the Black Sea analyses (CPU only)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chl_cnn import data as D, clouds as C, train2 as T2, store_blk as S
from chl_cnn.plotting import OKABE_ITO

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEAS = list(SEASONS)


def load_regions():
    reg = np.load(HERE / "regions.npy")
    names = {int(k): v for k, v in json.load(open(HERE / "region_names.json")).items()}
    return reg, names


def months(times):
    return np.asarray(times).astype("datetime64[M]").astype(int) % 12 + 1


def season_idx(times):
    m = months(times); out = np.empty(len(m), dtype=object)
    for s, ms in SEASONS.items():
        out[np.isin(m, ms)] = s
    return out


def load_ln_chl(d, idx, step=1):
    """ln CHL (float32, NaN where non-positive/missing) for days idx."""
    out = np.empty((len(idx),) + d["chl_l3"].shape[1:], dtype="float32")
    for k in range(0, len(idx), 200):
        out[k:k + 200] = D.log_chl(np.asarray(d["chl_l3"][idx[k:k + 200]]))
    return out


def train_climatology(d):
    """DOY climatology from training days (all training days). Returns clim (46,H,W), bin per day, tr, va, y_va, obs_va, hid_va (all val days)."""
    times = d["time"]; T2.TIMES = times
    tr, va, te = D.split_by_years(times, (2017, 2021), (2019, 2023))
    ocean = d["ocean"]
    y = load_ln_chl(d, np.arange(len(times)))
    obs = np.isfinite(y) & ocean
    clim, b = T2.doy_climatology(y, obs, tr)
    hid, _ = C.fixed_hidden(obs, va, seed=0)
    return dict(clim=clim, bin=b, tr=tr, va=va, y=y, obs=obs, hid=hid, times=times)


def score_by_region_season(err_abs, hid, reg, times_va, names):
    """err_abs, hid: (n,H,W). Returns {(region, season): (mae, n)} plus 'all' region 0."""
    seas = season_idx(times_va); res = {}
    for r in list(names) + [-1]:
        rm = (reg > 0) if r == -1 else (reg == r)
        for s in SEAS + ["all"]:
            sel = np.ones(len(seas), bool) if s == "all" else seas == s
            m = hid[sel] & rm[None]
            n = int(m.sum())
            res[(r, s)] = (float(err_abs[sel][m].mean()) if n else np.nan, n)
    return res


def md_table(res, names, title):
    cols = SEAS + ["all"]
    lines = [f"**{title}**", "", "| region | " + " | ".join(cols) + " | n hidden (all) |", "|---|" + "---|" * (len(cols) + 1)]
    for r in list(names) + [-1]:
        nm = "all ocean" if r == -1 else f"{r} {names[r]}"
        cells = []
        for s in cols:
            mae, n = res[(r, s)]
            cells.append("n<5" if n < 5 else f"{mae:.3f} ({n})")
        lines.append(f"| {nm} | " + " | ".join(cells) + f" | {res[(r, 'all')][1]} |")
    return "\n".join(lines)
