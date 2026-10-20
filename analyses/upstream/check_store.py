"""Check the io_up store against their protocol, recomputed independently from IO.zarr.

1. Their day list (U-Net_Tutorial cell 6: box, drop all-NaN L3 days, sortby time) vs store record_time.npy and the time axis
   of their public U-Net_with_CHL_pred.zarr (downloaded by fetch_unet_pred.py).
2. For 3 days: their masked_CHL rule (cell 9: day_shift_flag = CHL_cmes-cloud of record i+10, wrapping; masked_CHL = NaN where
   day_shift_flag == 0) -> hidden = finite ln L3 & NaN masked_CHL, compared pixel by pixel with store hidden_up.npy.
3. Their level3_pred is finite exactly where L3 is finite on those days (2018-2019 only).
Usage: python -m analyses.upstream.check_store    -> results/check_store.json
"""
import json
from pathlib import Path

import numpy as np
import xarray as xr

from chl_cnn import data as D
from chl_cnn import store_io_up as SU

HERE = Path(__file__).resolve().parent
PRED = D.DATA / "upstream_unet_pred"
DAYS = ["2015-03-14", "2018-07-01", "2019-12-27"]


def main():
    ds = xr.open_zarr(D.SRC, consolidated=True).sel(lat=slice(32, -11.75), lon=slice(42, 101.75))
    all_nan = np.isnan(ds["CHL_cmes-level3"]).all(dim=["lon", "lat"]).compute()
    ds = ds.sel(time=~all_nan).sortby("time")
    their_t = ds.time.values.astype("datetime64[D]")
    d = SU.load_store()
    out = dict(n_their_days=len(their_t),
               day_list_equal_store=bool(np.array_equal(their_t, d["record_time"].astype("datetime64[D]"))),
               day_list_equal_pred_zarr=bool(np.array_equal(their_t, np.load(PRED / "time_full.npy").astype("datetime64[D]"))))
    st = d["time"].astype("datetime64[D]")
    pt = np.load(PRED / "time.npy").astype("datetime64[D]"); l3p = np.load(PRED / "level3_pred.npy", mmap_mode="r")
    n = len(their_t); days = []
    for day in DAYS:
        i = int(np.flatnonzero(their_t == np.datetime64(day))[0])
        ln = np.log(ds["CHL_cmes-level3"].isel(time=i).values)
        flag = ds["CHL_cmes-cloud"].isel(time=(i + 10) % n).values
        masked = np.where(flag == 0, np.nan, ln)
        theirs = np.isfinite(ln) & np.isnan(masked)
        k = int(np.flatnonzero(st == np.datetime64(day))[0])
        ours = np.asarray(d["hidden_up"][k])
        row = dict(day=day, record_index=i, shift_day=str(their_t[(i + 10) % n]), their_hidden=int(theirs.sum()),
                   store_hidden=int(ours.sum()), mismatched_px=int((theirs != ours).sum()), observed=int(np.isfinite(ln).sum()))
        j = np.flatnonzero(pt == np.datetime64(day))
        if len(j):
            p = l3p[int(j[0])]
            row["level3_pred_finite_eq_L3_finite"] = bool(np.array_equal(np.isfinite(p), np.isfinite(ln)))
        days.append(row)
    out["days"] = days
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "check_store.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
