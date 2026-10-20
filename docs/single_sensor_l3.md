# Single-sensor daily L3 CHL for the Black Sea (metadata check, 2026-10-18)

Source: `copernicusmarine describe` (OCEANCOLOUR_BLK*, OCEANCOLOUR_GLO_BGC_L3*), plus a 1-week, 6x4 deg subset (June 2019, 28-34E, 42-46N) of the 1 km multi and 300 m OLCI CHL products. OC-CCI from PML THREDDS `.dds` only.

## Result
No dataset carries a per-sensor CHL field. Only two kinds of "sensor" split exist: OLCI-only (S3A+S3B merged) products, and a per-pixel `SENSORMASK` bitmask in the merged products. CHL in every product is the merged value of whatever sensors saw the pixel.

## CMEMS CHL L3 datasets (Black Sea box 40-48N, 26.5-42E)
| dataset id | sensors in CHL | variables | grid | time coverage | covers 2014-2024 |
|---|---|---|---|---|---|
| `cmems_obs-oc_blk_bgc-plankton_my_l3-multi-1km_P1D` (009_153; v202311, retirement notice 2026-11-24) | SeaWiFS, MODIS-A, MERIS, VIIRS-NPP, OLCI-A, VIIRS-J1, OLCI-B merged | CHL, QI_CHL, MICRO, NANO, PICO, SENSORMASK | 0.0141 x 0.0101 deg (~1 km) | 1997-09-07 to 2026-10 | yes |
| `cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D` | OLCI-A + OLCI-B merged | CHL, QI_CHL, SENSORMASK | 0.00375 x 0.0027 deg (~300 m) | 2016-04-26 to 2026-10 | 2016-04 onward only |
| `cmems_obs-oc_blk_bgc-plankton_nrt_l3-multi-1km_P1D` (009_151) | merged | CHL, QI_CHL, MICRO, NANO, PICO, SENSORMASK | ~1 km | rolling ~2026-10-04 to 2026-10-14 | no (NRT window) |
| `cmems_obs-oc_blk_bgc-plankton_nrt_l3-olci-300m_P1D` | OLCI-A + B | CHL, QI_CHL, SENSORMASK | ~300 m | rolling ~2 weeks | no |
| `cmems_obs_oc_blk_bgc_tur-spm-chl_nrt_l3-hr-mosaic_P1D-m` (HR 009_206) | HR mosaic (NRT, 2021 version) | CHL, SPM, TUR | HR | NRT | not checked beyond listing |
| `cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D` (009_103) | merged, global | CHL, CHL_uncertainty, groups, flags | 4 km | 1997- | yes |
| `cmems_obs-oc_glo_bgc-plankton_my_l3-olci-4km_P1D` | OLCI (A+B) | CHL, CHL_uncertainty, CHL_gradient, CHL_gradient_uncertainty, flags | 4 km | 2016- | 2016 onward |
| `cmems_obs-oc_glo_bgc-plankton_my_l3-olci-300m_P1D` | OLCI (A+B) | CHL, CHL_uncertainty, flags | 300 m global | 2016- | 2016 onward |
| `c3s_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D` | merged (OC-CCI) | CHL, MICRO/NANO/PICO + BIAS/RMSE | 4 km | 1997- | yes |

Global GLO L3 lists only `multi` and `olci` ids. No modis/viirs/meris/seawifs/s3a/s3b-specific dataset id exists in the BLK or GLO L3 listings. Reflectance counterparts (RRS) exist with the same multi/olci split (also merged).

## Per-pixel sensor info in merged product
`SENSORMASK` is in the 1 km multi CHL dataset (also reflectance, optics, transp). Attribute comment: SeaWiFS=1, MODIS-A=2, MERIS=4, VIIRS-NPP=8, OLCI-S3A=16, VIIRS-JPSS1=32, OLCI-S3B=64; pixel value = sum of contributing sensors. The 300 m OLCI product: S3A=1, S3B=2, both=3. It says which sensors contributed, not each sensor's CHL value.

Subset (June 2019, 6x4 deg, 7 days; 168,696 px/day at 1 km):
| quantity | value |
|---|---|
| valid CHL fraction per day (1 km) | 0.08 to 0.60 (mean ~0.44 excl. the 0.08 day: see daily list 0.43, 0.08, 0.60, 0.42, 0.54, 0.53, 0.47) |
| valid px with 1 / 2 / 3 / 4 sensors | 208,072 / 180,607 / 94,974 / 33,391 (week total); 766 px valid CHL with mask 0 |
| valid px with >=2 sensors, per day | 44,377; 2,273; 62,251; 30,639; 50,280; 61,436; 57,716 (~51% of valid px) |
| CHL NaN where mask>0 | 0 px |
| 300 m OLCI: valid CHL fraction per day | 0.10, 0.10, 0.47, 0.27, 0.06, 0.46, 0.45 (of 2.37M px/day) |
| 300 m OLCI: valid px with A only / B only / A+B | 2,028,631 / 2,258,366 / 249,908 (A+B = 5.5% of valid) |

Sensor-pair same-day overlap px per day (1 km merged, week mean, this subbox): MODIS-A/VIIRS-J1 24,495; VIIRS-NPP/VIIRS-J1 17,909; VIIRS-NPP/MODIS-A 14,328; VIIRS-NPP/OLCI-A 8,783; MODIS-A/OLCI-B 7,871; VIIRS-J1/OLCI-B 7,572; MODIS-A/OLCI-A 6,596; VIIRS-J1/OLCI-A 5,744; VIIRS-NPP/OLCI-B 1,826; OLCI-A/OLCI-B ~0. These count pixels where both sensors fed the merged CHL; individual CHL values are not retrievable from this product.

## OC-CCI (PML THREDDS, `.dds` only)
| aggregation | grid | period | per-sensor content |
|---|---|---|---|
| `CCI_ALL-v6.0-DAILY` | 4 km | 1997-09 to 2026-06 | `chlor_a`, `Rrs_*`, bias/rmsd merged; per-sensor `*_nobs` (SeaWiFS, MERIS, MODISA, VIIRS, OLCI) plus `total_nobs` |
| `CCI_ALL-v6.0-1km-DAILY` | 1 km | listed in catalog (variables/period not inspected) | not inspected |
| `CCI_ALL-v5.0-1km-DAILY` | 1 km | 1997-09 to 2021-04 | `chlor_a` merged; `{VIIRS,MODISA,MERIS,SeaWiFS,OLCI}_nobs`, `total_nobs` |
Per-sensor `*_nobs` are observation counts, not per-sensor CHL. CCI CHL is a band-merged, bias-corrected product.

## Download size estimates (compressed netcdf/zarr, CHL+SENSORMASK, from the 1-week subset; June is a high-clear-fraction month, so annual means are likely lower)
| option | period (days) | native | 2 km (after local coarsening) |
|---|---|---|---|
| 1 km multi (CHL+SENSORMASK) | 2014-2024 (4,018 d) | ~1.4 MB/day, ~5.7 GB | ~1.4 GB stored; transfer is native size |
| 300 m OLCI (CHL+SENSORMASK) | 2016-04-26 to 2024-12 (3,172 d) | ~10.7 MB/day, ~34 GB | ~0.5 GB stored; transfer ~34 GB |

Existing local `blk_2km` store: 26 GB (project data, not part of this check).

Notes: the local 7-day 300 m subset write was 133 MB uncompressed (deleted); scratch files in the session scratchpad only.
