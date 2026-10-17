"""Add a float32 copy of SST to the memory-mapped store (the float16 copy is quantised to 0.25 K at 300 K,
which destroys the SST gradient used by the spatial nODE's diffusion tensor). Writes sst32.npy, same layout."""
import numpy as np
import xarray as xr
from . import data as D
from . import store as S


def build(block=300):
    out = S.store_path()
    time = np.load(out / "time.npy")
    if (out / "sst32.npy").exists():
        print("sst32.npy exists"); return
    ds = xr.open_zarr(D.SRC, consolidated=True).sel(**D.REGION).sel(time=time)
    n, ny, nx = len(time), ds.sizes["lat"], ds.sizes["lon"]
    mm = np.lib.format.open_memmap(out / "sst32.npy.part", mode="w+", dtype="float32", shape=(n, ny, nx))
    for b0 in range(0, n, block):
        mm[b0:b0 + block] = ds["sst"].isel(time=slice(b0, b0 + block)).values
        if (b0 // block) % 5 == 0:
            print(str(time[b0])[:10], flush=True)
    mm.flush(); del mm
    (out / "sst32.npy.part").rename(out / "sst32.npy")
    print("wrote sst32.npy")


if __name__ == "__main__":
    build()
