"""Pull a lat/lon box of ESA OC-CCI chlorophyll from the PML THREDDS aggregations into a local zarr.

The PML server (https://rsg.pml.ac.uk/thredds/catalog-cci.html) serves whole-record aggregations over OPeNDAP,
so a box can be sliced server-side instead of downloading global daily files:
  CCI_ALL-v6.0-DAILY       4 km (1/24 deg), 1997-09-04 -> 2026-06   (~0.6 s/day for the Indian Ocean box)
  CCI_ALL-v5.0-1km-DAILY   1 km (1/96 deg), 1997-09-04 -> 2021-04   (~1-3 s/day for a basin-sized box)
Variables: chlor_a (mg m-3, linear, L3 = gappy), chlor_a_log10_bias/rmsd, Rrs_*, kd_490, per-sensor *_nobs.
Days are fetched in blocks with retries and appended to <out>/<name>.zarr (time-chunked), so a run can be resumed.

Examples
  python scripts/fetch_occci.py --name io_4km  --dataset CCI_ALL-v6.0-DAILY     --box 32 -5 45 90      --start 1997-09-04 --end 2022-12-31
  python scripts/fetch_occci.py --name blk_1km --dataset CCI_ALL-v5.0-1km-DAILY --box 48 40 26.5 42    --start 2016-01-01 --end 2021-04-30
  python scripts/fetch_occci.py --name med_1km --dataset CCI_ALL-v5.0-1km-DAILY --box 46 30 -6 36.5    --start 2016-01-01 --end 2021-04-30 --vars chlor_a total_nobs
Box is  LAT_MAX LAT_MIN LON_MIN LON_MAX  (lat is stored north -> south on the server).
Output goes to ~/mind_the_chl_gap/data/occci/ (local disk, not ~/shared: zarr = many small files).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT = Path.home() / "mind_the_chl_gap" / "data" / "occci"
BASE = "https://rsg.pml.ac.uk/thredds/dodsC/"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="CCI_ALL-v6.0-DAILY")
    ap.add_argument("--box", nargs=4, type=float, required=True, metavar=("LAT_MAX", "LAT_MIN", "LON_MIN", "LON_MAX"))
    ap.add_argument("--start", default="1997-09-04")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--vars", nargs="*", default=["chlor_a"])
    ap.add_argument("--block", type=int, default=20, help="days per request")
    ap.add_argument("--retries", type=int, default=5)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.name}.zarr"
    ds = xr.open_dataset(BASE + a.dataset, engine="netcdf4")
    la1, la0, lo0, lo1 = a.box
    sel = ds[a.vars].sel(lat=slice(la1, la0), lon=slice(lo0, lo1), time=slice(a.start, a.end))
    times = pd.to_datetime(sel.time.values)
    done = 0
    if out.exists():
        have = pd.to_datetime(xr.open_zarr(out).time.values)
        done = int(np.searchsorted(times, have[-1], side="right")) if len(have) else 0
        print(f"resuming {out}: {done}/{len(times)} days present")
    print(f"{a.dataset}: box lat {la1}..{la0} lon {lo0}..{lo1} -> {sel.sizes['lat']} x {sel.sizes['lon']} px, {len(times)} days, vars {a.vars}", flush=True)
    t_all = time.time()
    for b0 in range(done, len(times), a.block):
        blk = sel.isel(time=slice(b0, b0 + a.block))
        for attempt in range(a.retries):
            try:
                t = time.time(); data = blk.load(); break
            except Exception as e:                                   # OPeNDAP hiccups: wait and retry
                print(f"  retry {attempt + 1}: {type(e).__name__}: {str(e)[:80]}", flush=True); time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"block at {times[b0]} failed {a.retries} times")
        data = data.chunk({"time": a.block, "lat": -1, "lon": -1})
        if out.exists() and b0 > 0:
            data.to_zarr(out, mode="a", append_dim="time")
        else:
            data.to_zarr(out, mode="w")
        if (b0 // a.block) % 10 == 0:
            el = time.time() - t_all; rate = el / max(b0 + a.block - done, 1)
            print(f"  {str(times[b0])[:10]}  {b0 + len(data.time)}/{len(times)}  {time.time() - t:.1f}s/block  eta {rate * (len(times) - b0 - a.block) / 60:.0f} min", flush=True)
    print(f"wrote {out} in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
