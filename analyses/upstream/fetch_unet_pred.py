"""Download their public 2024 U-Net predictions (level3_pred of U-Net_with_CHL_pred.zarr, anonymous GCS) for
2018-01-01..2020-01-01 (their val + test slices), plus the full time axis, lat and lon. Untrusted data: run as
  python -I analyses/upstream/fetch_unet_pred.py      (from another directory; -I drops the user site, re-added below)
Writes ~/mind_the_chl_gap/data/upstream_unet_pred/{time_full,time,lat,lon,level3_pred}.npy (float32)."""
import site
import sys
sys.path.append(site.getusersitepackages())
from pathlib import Path

import numpy as np
import xarray as xr

URL = "gs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/U-Net_with_CHL_pred.zarr"
OUT = Path.home() / "mind_the_chl_gap" / "data" / "upstream_unet_pred"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(URL, storage_options=dict(token="anon"), consolidated=True)
    print(ds)
    np.save(OUT / "time_full.npy", ds.time.values)
    np.save(OUT / "lat.npy", ds.lat.values); np.save(OUT / "lon.npy", ds.lon.values)
    sub = ds["level3_pred"].sel(time=slice("2018-01-01", "2020-01-01"))
    np.save(OUT / "time.npy", sub.time.values)
    np.save(OUT / "level3_pred.npy", sub.values.astype("float32"))
    print("wrote", OUT, sub.shape)


if __name__ == "__main__":
    main()
