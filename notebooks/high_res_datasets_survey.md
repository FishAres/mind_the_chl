# Higher-resolution CHL products, matching forcing, comparability, km-scale ML gap-filling setups (survey 2026-10-02)

Compiled by a delegated search agent from provider pages, the Copernicus Marine STAC catalogue (fetched 2026-10-02) and papers. **[unverified]** / **[estimate]** marks items not confirmed from a primary page.

## 1. Higher-resolution CHL products

| Product (provider) | IDs | Sensors | Spatial | Temporal | Period | Coverage | L3 / L4 (method) | Format | Access | Licence | Gap fraction (documented) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ESA OC-CCI v6.0** (PML/ESA) | `ESACCI-OC-L3S-CHLOR_A-MERGED-*-4km` ([CEDA](https://catalogue.ceda.ac.uk/uuid/474ac06235e54e6cb0ec6eed635e1213/)) | SeaWiFS, MERIS, MODIS-A, VIIRS, OLCI-A/B; reflectances merged then blended CHL algorithm | 4 km | daily, 5-d, 8-d, monthly | 1997-09 → 2024-12 ([ESA CCI](https://climate.esa.int/en/projects/ocean-colour/)); CEDA record to 2022-12-31 | Global | **L3 only**; no operational L4 (Ford et al. 2026 kriged 0.25° gap-free: [ESSD](https://essd.copernicus.org/articles/18/569/2026/)) | NetCDF | CEDA HTTP/OPeNDAP/anon FTP, no registration; oceancolour.org FTP/THREDDS | free and open, attribution | Not stated. Literature: >70 % daily missing (Hong et al. 2023); GlobColour has ~2× OC-CCI daily pixels ([QUID](https://documentation.marine.copernicus.eu/QUID/CMEMS-OC-QUID-009-101to104-111-113-116-118.pdf)). v7 not released |
| **Copernicus GlobColour global 4 km** (ACRI-ST / CMEMS) | L3 `OCEANCOLOUR_GLO_BGC_L3_MY_009_103` → `cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D`; L4 `OCEANCOLOUR_GLO_BGC_L4_MY_009_104` → `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` (+ NRT 009_101/102) | SeaWiFS, MODIS-A/T, MERIS, VIIRS-SNPP/JPSS1, OLCI-A/B; per-sensor CHL (CI/OCx/OC5) merged ([Garnesson 2019](https://os.copernicus.org/articles/15/819/2019/)) | 4 km | daily (L3, L4 gap-free), monthly | 1997-09 → present | Global | **L3 + daily L4 gap-free pair**, space-time interpolation (Saulquin et al. 2019); L4 flags interpolated pixels ([PUM](https://documentation.marine.copernicus.eu/PUM/CMEMS-OC-PUM.pdf)) | NetCDF-4 (ARCO/zarr via toolbox) | Free registration; `copernicusmarine` toolbox, S3 ([register](https://data.marine.copernicus.eu/register)) | Free, commercial OK, attribution | ~2× OC-CCI coverage |
| **GlobColour global coastal 300 m** (ACRI/CMEMS) | `cmems_obs-oc_glo_bgc-plankton_my_l3-olci-300m_P1D`; L4 only monthly | OLCI-A/B | 300 m | daily | 2016-04-25 → present | Global, **200 km coastal strip only** | L3 daily; **no daily L4** | NetCDF-4 | CMEMS | CMEMS | Not stated |
| **CMEMS Mediterranean 1 km** (CNR-GOS) | L3 `OCEANCOLOUR_MED_BGC_L3_MY_009_143` → `cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D` (+ `…l3-olci-300m_P1D`); L4 `OCEANCOLOUR_MED_BGC_L4_MY_009_144` → `…l4-gapfree-multi-1km_P1D` | multi-sensor, band-shifted Rrs merged, MedOC4.2020 + AD4 ([QUID](https://documentation.marine.copernicus.eu/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf)) | 1 km (300 m OLCI) | daily | 1997-09 → present | 6°W-36.5°E, 30-46°N | **L3 + daily L4 pair**; L4 = modified DINEOF (Volpe et al. 2018) | NetCDF-4 | CMEMS | CMEMS | 30-70 % missing, ~45 % Gulf of Lion ([Nguyen et al. 2025](https://arxiv.org/html/2503.11532)) |
| **CMEMS Black Sea 1 km** (CNR) | L3 `OCEANCOLOUR_BLK_BGC_L3_MY_009_153` (+ `…olci-300m_P1D`); L4 `OCEANCOLOUR_BLK_BGC_L4_MY_009_154` | as MED; BSAlg | 1 km / 300 m | daily | 1997-09 → present | 26.5-42°E, 40-48°N | **L3 + daily L4 pair** (DINEOF) | NetCDF-4 | CMEMS | CMEMS | OLCI 300 m ~30 % valid ([Barth et al. 2024](https://os.copernicus.org/articles/20/1567/2024/)) |
| **CMEMS Atlantic 1 km** (ACRI) | L3 `OCEANCOLOUR_ATL_BGC_L3_MY_009_113` (+ `…olci-300m_P1D`); L4 `OCEANCOLOUR_ATL_BGC_L4_MY_009_118` | GlobColour processor | 1 km / 300 m | daily | 1997-09 → present | 46°W-13°E, 20-66°N | **L3 + daily L4 pair** (space-time interpolation) | NetCDF-4 | CMEMS | CMEMS | Not stated |
| **CMEMS Baltic 1 km** | L3 `OCEANCOLOUR_BAL_BGC_L3_MY_009_133`; L4 `009_134` | regional | 1 km / 300 m | daily L3; **L4 monthly only** | 1997 → | Baltic | no daily gap-free | NetCDF-4 | CMEMS | CMEMS | – |
| **CMEMS Arctic** | L3 `OCEANCOLOUR_ARC_BGC_L3_MY_009_123` (multi 4 km, OLCI 300 m); L4 `009_124` monthly | OC-CCI-based | 4 km / 300 m | daily L3 | 1997 → | >52°N | no daily gap-free | NetCDF-4 | CMEMS | CMEMS | – |
| **CMEMS HR coastal 100 m (NRT only)** | `OCEANCOLOUR_{ARC,BAL,NWS,IBI,MED,BLK}_BGC_HR_L3/L4_NRT_009_201-212` | Sentinel-2 MSI | 100 m | daily | NRT, European seas | – | L3 + L4 | NetCDF | CMEMS | CMEMS | – |
| **NASA OB.DAAC MODIS-Aqua L3** | `MODISA_L3m_CHL` v2022.0 ([Earthdata](https://www.earthdata.nasa.gov/data/catalog/ob-cloud-modisa-l3m-chl-2022.0)); L2 `MODISA_L2_OC` 1 km | MODIS-A (Terra, VIIRS analogues) | 4 km & 9 km; L2 ~1 km | daily, 8-d, monthly | 2002-07 → | Global | L3 only | NetCDF-4 | Earthdata Login; HTTPS, OPeNDAP, S3 us-west-2 ([direct access](https://www.earthdata.nasa.gov/data/tools/ob-daac-direct-data-access)) | NASA open | – |
| **NASA PACE OCI L3** | `PACE_OCI_L3M_CHL` v3.2 ([Earthdata](https://www.earthdata.nasa.gov/data/catalog/ob-cloud-pace-oci-l3m-chl-3.2)) | OCI hyperspectral | ~4 km; L2 ~1.2 km | daily | 2024-03 → | Global | L3 only | NetCDF-4 | Earthdata Login | NASA open | – |
| **Sentinel-3 OLCI L2 WFR** (EUMETSAT via CDSE) | `OL_2_WFR`; `CHL_OC4ME`, `CHL_NN` ([handbook](https://earth.esa.int/eogateway/documents/20142/1564943/Sentinel-3-OLCI-Marine-User-Handbook.pdf)) | OLCI-A/B | 300 m FR (1.2 km RR) | per pass; near-daily A+B | 2016-03 → | Global | L2 swath only | SAFE (NetCDF) | Free CDSE registration; S3 `eodata.dataspace.copernicus.eu` ([docs](https://documentation.dataspace.copernicus.eu/APIs/S3.html)) | Copernicus free & open | – |
| **JAXA Himawari-8/9 (P-Tree)** | Himawari L3 Chl-a v2.1 ([guide](https://www.eorc.jaxa.jp/ptree/userguide.html)) | AHI | **0.05°** full disk; 1 km Japan box | hourly, daily, monthly | 2015-07 → | 70°E-150°W, 60°S-60°N; since Sept 2025 also an MTG area 70°W-70°E (Indian Ocean covered by both) | L3 only | NetCDF | **Registration required**; pre-2026-02 data non-commercial, no redistribution, publication needs secretariat approval ([registration](https://www.eorc.jaxa.jp/ptree/registration_top.html)) | restrictive | ~31 % daily missing Bohai/Yellow Sea (Wang et al. 2026) |
| **GOCI-II** (KIOST/NOSC) | `GK2B_GOCI2_L2_*_LA_S0xx_Chl.nc` ([NOSC OPeNDAP](https://nosc.go.kr/opendap/GOCI-II/)) | GOCI-II, 13 bands | 250 m | hourly, 10 slots/day | 2020-10 → ([IOCCG](https://ioccg.org/sensor/goci-2/)) | NE Asia | L2 only | NetCDF | OPeNDAP appears open; policy **[unverified]** | **[unverified]** | – |
| **NOAA CoastWatch MSL12** | L3 VIIRS 4 km daily (ERDDAP `nesdisVHNSQchlaDaily`); **L4 DINEOF 9 km** (`noaacwNPPN20VIIRSSCIDINEOFDaily`; [page](https://coastwatch.noaa.gov/cwn/products/noaa-msl12-multi-sensor-dineof-global-9km-gap-filled-products-chlorophyll-diffuse.html)) | VIIRS SNPP + N20 + OLCI-A | 4 km (L3), 9 km (L4) | daily | L3 2012 →; L4 2018-02 → | Global | L3 and L4 at **different** resolutions | NetCDF | Anonymous ERDDAP/HTTPS/THREDDS | NOAA | – |
| **ISRO/NRSC OCM-3 (EOS-06)** | L3 binned CHL, Bhoonidhi ([ATBD](https://bhoonidhi.nrsc.gov.in/bhoonidhi_resources/help/docs/EOS06-ATBD-OCM3_Ocean_Products-V1.pdf)) | OCM-3 | 1 km L2; L3 2-d/8-d/monthly | **no daily L3 found** | 2023 → | Global / N. Indian Ocean | L3 only | **[unverified]** | Bhoonidhi registration | open | – |
| **INCOIS** | No km-scale gridded CHL product found | – | – | – | – | – | – | – | – | – | – |
| Hong et al. OCNET (known) | Zenodo 10.5281/zenodo.10011908 | OC-CCI v5 + MSL12 input | 0.25° | daily | 2001-2021 | open ocean | ML gap-filled | NetCDF | Zenodo | CC-BY | >70 % missing input |

## 2. Physical forcing at matching resolution

| Variable | Product | Resolution | Temporal | Period | Latency | Access |
|---|---|---|---|---|---|---|
| SST L4 | OSTIA REP `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011` | 0.05° | daily | 1981-10 → 2026-03 | biannual | CMEMS |
| SST L4 | OSTIA NRT `SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001` | 0.05° | daily | 2007 → | ~24 h | CMEMS; PO.DAAC |
| SST L4 | MUR `MUR-JPL-L4-GLOB-v4.1` | 0.01° | daily | 2002-06 → | 1 d / 4 d | PO.DAAC; AWS Open Data zarr `s3://mur-sst/zarr-v1` anonymous ([registry](https://registry.opendata.aws/mur/)) |
| SST L4 | CMC `CMC0.1deg-CMC-L4-GLOB-v3.0` | 0.1° | daily | 2016 → | daily | PO.DAAC |
| Currents | GlobCurrent `MULTIOBS_GLO_PHY_MYNRT_015_003` | 0.25° | hourly, daily | 1993 → | daily | CMEMS |
| Currents | OSCAR v2.0 | 0.25° | daily | 1993 → | ~1 yr / 1 mo / 2 d | PO.DAAC |
| Geostrophic | DUACS `SEALEVEL_GLO_PHY_L4_MY_008_047` (0.125°) | 0.125° | daily | 1993 → | months | CMEMS |
| Currents/MLD/S | GLORYS12 `GLOBAL_MULTIYEAR_PHY_001_030` | 1/12° | daily | 1993 → 2026-06 | months | CMEMS |
| Currents/MLD/S | GLO analysis `GLOBAL_ANALYSISFORECAST_PHY_001_024` | 1/12° | hourly/6-h/daily | 2019 → | daily | CMEMS |
| Regional reanalyses | MED `MEDSEA_MULTIYEAR_PHY_006_004` 1/24°; IBI `IBI_MULTIYEAR_PHY_005_002` 1/36°; BLK `BLKSEA_MULTIYEAR_PHY_007_004` 2.5 km | 2.5-4 km | daily/hourly | 1987-93 → | months | CMEMS |
| HF radar | CMEMS `INSITU_GLO_PHY_UV_DISCRETE_NRT_013_048`; INCOIS Indian coastal network ([INCOIS](https://services.incois.gov.in/portal/datainfo/hfradar.jsp)) | few km | hourly | varies | NRT | CMEMS; INCOIS |
| Winds | ERA5 | 0.25° | hourly | 1940 → | ~5 d | CDS |
| Winds | CCMP v3.1 | 0.25° | 6-h | 1993 → | 2-3 months | PO.DAAC / RSS |
| Winds | CMEMS L4 `WIND_GLO_PHY_L4_MY_012_006` | 0.125° | hourly | 1994 → | – | CMEMS |
| Winds | ASCAT coastal L2 / CMEMS L3 `WIND_GLO_PHY_L3_MY_012_005` | 12.5 km sampling | per pass | 2007 → | NRT | KNMI/EUMETSAT; CMEMS |
| PAR | NASA L3 `MODISA_L3m_PAR`, VIIRS PAR | 4 km / 9 km | daily | 2002 → | days | Earthdata |
| PAR proxy | ERA5 `ssrd` | 0.25° | hourly | 1940 → | ~5 d | CDS |
| MLD | GLORYS12 / GLO analysis / regional reanalyses | 1/12°-1/36° | daily | as above | – | CMEMS |

## 3. Comparability across regions

| Product family | Single global dataset, uniform processing? | What breaks comparability | Matched L3 + L4 pair? |
|---|---|---|---|
| OC-CCI v6 | Yes | Sensor mix over time; v6 ends 2024; lower daily coverage than GlobColour | No L4 |
| GlobColour GLO 4 km | Yes | Per-sensor CHL merged; CI/OCx/OC5 switching; sensor mix over time | **Yes** (daily, same grid, interpolation mask) |
| GlobColour ATL 1 km | Same processor as GLO, N. Atlantic only | Region-limited | **Yes** |
| CMEMS MED / BLK 1 km | No: regional CNR chain | Different algorithms, DINEOF L4 | **Yes** |
| CMEMS BAL / ARC | No | Regional algorithms; no daily L4 | No |
| GlobColour 300 m coastal | Global, coastal strip, OLCI-only, 2016+ | no daily L4 | No |
| NASA MODIS/VIIRS/PACE L3 | Yes per sensor | poorer single-sensor coverage; no L4 | No |
| NOAA MSL12 | Yes (VIIRS) | L3 4 km vs L4 9 km | Not pixel-matched |
| Himawari / GOCI-II | Regional disks | geostationary algorithms, licence (Himawari) | No |

## 4. Km-scale ML gap-filling studies and their datasets

| Study | Dataset (exact ID) | Res / region / period | Setup | Missing % | Code/data |
|---|---|---|---|---|---|
| Barth et al. 2024, Ocean Sci. 20:1567 (DINDiff vs DINCAE) | `cmems_obs-oc_blk_bgc-plankton_my_l3-olci-300m_P1D` | 300 m; W. Black Sea 28.57-28.81°E, 43.64-43.81°N; 2016-04 → 2023-08 | train → 2021-08; dev → 2022-08; test → 2023-08; 64×64 tiles ≥20 % valid; RMSE 0.176 log10 | ~30 % valid | [DINDiff.jl](https://github.com/gher-uliege/DINDiff.jl), Zenodo 10.5281/zenodo.13165363 ([paper](https://os.copernicus.org/articles/20/1567/2024/)) |
| Nguyen, Dorffer, Jourdin, Fablet 2025 (4DVarNet) | `OCEANCOLOUR_MED_BGC_L3_MY_009_143` 1 km | 240×240 px 3-6°E, 41-43.5°N; 2017-2021 | train 2017-18, val Jan 2019, test 2019-20 | ~45 % | [4dvarnet-core](https://github.com/CIA-Oceanix/4dvarnet-core/tree/turbidity-fish-predict) ([arXiv 2503.11532](https://arxiv.org/html/2503.11532)) |
| Nguyen et al. 2025 (generalisation) | CMEMS Med L3 1 km + CMEMS North Sea L3 | Gulf of Lion 240×240 km; Wadden Sea 240×300 km | cross-region transfer; RMSLE | ~45 % Med, ~60 % NS | [arXiv 2503.11588](https://arxiv.org/html/2503.11588) |
| Mehdipour et al. 2026, GMD 19:1619 (DINEOF vs DINCAE, PFT) | `cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D` + OSTIA REP | 4 km; Atlantic 64°W-3°E, 50°S-52°N; 2016-04 → 2019-04 | holdout CV + HPLC | – | [GMD](https://gmd.copernicus.org/articles/19/1619/2026/) |
| Han, He, Liu, Perrie 2020, Remote Sens. 12:480 (DINCAE ± SST) | MODIS-Aqua L3 **[res unverified]** | N. South China Sea 115-125°E, 16-24°N | CV + in situ | – | [doi](https://doi.org/10.3390/rs12030480) |
| Luo et al. 2022, IJRS 43(9) (DINCAE Bohai/Yellow Sea) | MODIS **[unverified]** | Bohai & Yellow Sea | CV + in situ | – | [doi](https://www.tandfonline.com/doi/abs/10.1080/01431161.2022.2090872) |
| Ye et al. 2024 ESSD preprint (OI-SwinUnet) | MODIS-A+T L3 1 km, own processing | South China Sea 0-25°N, 100-125°E; 2013-2017 | missing-rate ≤0.6; artificial masks; vs DINEOF/OI/UNet | – | [preprint](https://essd.copernicus.org/preprints/essd-2024-6/essd-2024-6-manuscript-version3.pdf) |
| Wang, Wang, Zhou 2026, ERC 8:045017 (U-Net dynamic masking) | Himawari-8 L3 daily 0.05° | Bohai/Yellow Sea 31-41°N, 117-127°E; 2015-07 → 2022-12 | self-supervised masks; vs DINCAE, MSL12 DINEOF | 31 % | [IOP](https://iopscience.iop.org/article/10.1088/2515-7620/ae5766) |
| Hong et al. 2023, ESSD 15:5281 (OCNET) | OC-CCI v5 + MSL12, 0.25° | global 2001-2021 | 7:1.5:1.5 split | >70 % | Zenodo 10.5281/zenodo.10011908 |
| Ford et al. 2026, ESSD 18:569 (kriging + BGC-Argo) | OC-CCI v6 at 0.25° | global 1997-2024 | Monte-Carlo uncertainty | – | Zenodo 10.5281/zenodo.15689006 |

## Size and access for a multi-region benchmark

One CHL variable, daily, 2000×2000 km box, float32, uncompressed **[estimate]**; compressed NetCDF with ~70 % missing is 3-5× smaller.

| Candidate | Grid in box | Uncompressed GB / region-year | Access |
|---|---|---|---|
| GlobColour GLO 4 km L3 + L4 (`009_103`/`009_104`) | 500×500 | 0.37 + 0.37 | CMEMS toolbox/S3, free registration |
| OC-CCI v6 4 km daily (L3 only) | 500×500 | 0.37 | CEDA anon HTTP/FTP; global archive ~2 TB |
| CMEMS 1 km L3 + L4 (ATL, MED, BLK) | 2000×2000 | 5.8 + 5.8 | CMEMS; regional domains only |
| GlobColour 300 m coastal L3 | 6667×6667 | 65 (no L4) | CMEMS; coastal strip, 2016+ |

## Comparability (short)

**Easy.** One global pipeline at 4 km is straightforward with GlobColour GLO: single processing chain 1997 to present, pixel-matched daily L3/L4 pair on one grid, interpolated pixels flagged in the L4, one API, one licence. OC-CCI v6 offers the same grid and period with a different (reflectance-merged, blended) algorithm, so the two can cross-check algorithm sensitivity on identical boxes. Physical forcing at 4 km-compatible scales (OSTIA 0.05°, MUR 0.01°, GlobCurrent/DUACS 0.125-0.25°, GLORYS 1/12°, ERA5/CCMP 0.25°) is global and uniform, so colocation code transfers across regions unchanged.

**Hard.** Below 4 km there is no uniform global daily product with an L4 twin. The 1 km L3+L4 pairs exist only for Atlantic, Mediterranean and Black Sea, from two different chains (ACRI space-time interpolation for ATL; CNR regional algorithms plus DINEOF for MED/BLK). Numbers from a MED box and an ATL box therefore differ by algorithm, L4 method and water-type handling, not only by the ML model. The global 300 m OLCI product is coastal-only, single-sensor, starts in 2016 and has no daily L4. For the Indian Ocean no km-scale regional product with an L4 pair was found; the 4 km global products are the only option with an L4 reference, while Himawari/MTG 0.05° hourly data now cover the basin under a non-commercial, no-redistribution licence for pre-2026 data. Sensor-mix changes over 1997-2022 (SeaWiFS-only, MERIS loss 2012, OLCI from 2016) affect gap statistics in every multi-sensor product.
