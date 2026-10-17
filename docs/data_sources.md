# Data sources and downloads

Moved from the README on 2026-10-17. Paths updated to `scripts/`.

## IO.zarr (Indian Ocean, 0.25°)

| | |
|---|---|
| Bucket (public, anonymous) | `gs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO.zarr` |
| Size | 24.7 GB on disk, 3,997 files; **66 GB uncompressed** |
| Layout | 27 variables × (time 16071, lat 177, lon 241); chunks of 100 days × full map |

To read it straight from the bucket without downloading (slower, but fine for small slices):

```python
import xarray as xr
ds = xr.open_zarr("gs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO.zarr",
                  storage_options={"token": "anon"}, consolidated=True)
```

The same bucket folder also holds project outputs: `U-Net_with_CHL_pred.zarr`, `U-Net_without_CHL_pred.zarr`, `2015_3_ArabSea*.zarr` and the `cnn_*` / `unet_tutorial` folders. `IO_rechunked.zarr` is a 6-variable subset from 1997 onward with small (1, 40, 56) chunks, so it's not a full copy.

How the dataset was built is documented at https://safs-varanasi-internship.github.io/indian-ocean-zarr/. The build code is not in the project repo.


## Higher-resolution CHL for a region only

Both main 4 km / 1 km sources can be sliced to a lat/lon box server-side, so there is no need to download global files.

**ESA OC-CCI (PML THREDDS, no registration).** `https://rsg.pml.ac.uk/thredds/catalog-cci.html` lists whole-record *aggregations*; open them over OPeNDAP and slice:

| aggregation | resolution | period | Indian Ocean box (32°N-5°S, 45-90°E) | notes |
|---|---|---|---|---|
| `CCI_ALL-v6.0-DAILY` | 4 km (1/24°) | 1997-09 → 2026-06 | 888 × 1080 px, 3.8 MB/day float32, ~0.6 s/day | v6, L3 only (gappy), `chlor_a` in mg m⁻³ linear |
| `CCI_ALL-v5.0-1km-DAILY` | 1 km (1/96°) | 1997-09 → 2021-04 | 3552 × 4320 px, 61 MB/day, ~3.4 s/day | v5; Black Sea box 768 × 1488 px, 5 MB/day, ~1.4 s/day; Med box 1536 × 4080 px, 25 MB/day, ~1.6 s/day |

`scripts/fetch_occci.py` pulls a box in time blocks with retries into `~/mind_the_chl_gap/data/occci/<name>.zarr` (resumable), e.g.
`python scripts/fetch_occci.py --name blk_1km --dataset CCI_ALL-v5.0-1km-DAILY --box 48 40 26.5 42 --start 2016-01-01 --end 2021-04-30 --vars chlor_a total_nobs`.
Compressed on disk ~0.6 MB per day per variable for the Black Sea box. The full 4 km Indian Ocean record (~9,100 days) takes ~1.5 h.
Early days (Sept 1997) are empty; SeaWiFS data start mid-September 1997. OC-CCI has no daily gap-free L4.

**Copernicus GlobColour / regional 1 km (free registration).** `pip install copernicusmarine`, then `copernicusmarine login` once, and subset by box and dates
(output netcdf or zarr; `--dry-run` reports the size first):

```bash
copernicusmarine subset -i cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D -v CHL -v flags \
    -x 45 -X 90 -y -5 -Y 32 -t 1997-09-01 -T 2022-12-31 --file-format zarr -o ~/mind_the_chl_gap/data/globcolour -f io_4km_l3.zarr
copernicusmarine subset -i cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D -v CHL -v CHL_uncertainty -v flags \
    -x 45 -X 90 -y -5 -Y 32 -t 1997-09-01 -T 2022-12-31 --file-format zarr -o ~/mind_the_chl_gap/data/globcolour -f io_4km_l4.zarr
# Black Sea 1 km: cmems_obs-oc_blk_bgc-plankton_my_l3-multi-1km_P1D / ..._l4-gapfree-multi-1km_P1D   (box -x 26.5 -X 42 -y 40 -Y 48)
# Mediterranean 1 km: cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D / ..._l4-gapfree-multi-1km_P1D (box -x -6 -X 36.5 -y 30 -Y 46)
```

`scripts/fetch_cmems.py` wraps this for the Black Sea and Mediterranean: `download` pulls CHL L3 (CHL, QI_CHL) and gap-free L4 at 1 km plus the regional
physics reanalysis surface layer (thetao, so, uo, vo, mlotst, zos, tauuo, tauvo; 2.5 km Black Sea, 4.2 km Med) as one NetCDF per dataset and year;
`arrange` block-averages CHL (geometric mean, valid count kept) by `--coarsen` and interpolates the physics onto that grid into one zarr per basin
(`~/mind_the_chl_gap/data/cmems/<basin>_<N>km.zarr`, same layout idea as IO.zarr). `scripts/fetch_cmems_all.sh blk 1998 2024 2` does both year by year
from tmux and deletes the raw files as it goes. One Black Sea month at 2 km: 135 MB raw, 90 MB arranged, ~1 min; a full year ~1.1 GB arranged.
The reanalysis domains are smaller than the CHL boxes (no Sea of Azov / Marmara in the Black Sea physics), so ~14 % of CHL ocean pixels have no
physics there (NaN → 0 in training, as for the GLORYS gaps in IO.zarr).

The L4 `flags` variable marks interpolated pixels, which gives a hidden-pixel protocol with the same L3 truth as here. Product details, matching
physical forcing (OSTIA 0.05° SST, MUR 0.01°, GlobCurrent, GLORYS 1/12°, regional 2.5-4 km reanalyses) and sizes: `notebooks/high_res_datasets_survey.md`.

