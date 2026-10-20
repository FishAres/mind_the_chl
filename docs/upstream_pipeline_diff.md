# Upstream gap-fill pipelines vs ours (code audit, 2026-10-20)

Read-only audit of the upstream code (notebooks, `src/`, package modules). Nothing was trained. Repos were cloned to the
session scratchpad. Two small reads of public data were made: the pred-zarr `time` chunks 0 and 91, and one `level3_pred`
chunk (`85.0.0`, 2021-02-07..05-17). Local `IO.zarr` was used to rebuild their day axis and check the cloud-flag codes.

**Ours**: `models/pinn/t3d/t3d_dil_c4_60ep_gdl100`. Hidden-pixel MAE in ln: val 0.1302 (n = 5.00 M px), test 0.1305 (n = 5.52 M px).

## Which repos count as which project

Grouping rule: the event and date of the code commits, not the README.

| label | repos | why |
|---|---|---|
| **2024**: GeoSMART Hackweek 2024 | geo-smart/mind-the-chl-gap @f28a268 (created 2024-08-16, code last changed 2024-08); ocean-satellite-tools/mind-the-chl-gap-2024 @fc6a88d | The archive copy differs only in README.md (`diff -r`). |
| **2025 OHW**: OceanHackWeek 2025 | oceanhackweek/ohw25_proj_gap @236468c (2025-08; 2026-06 edits are package and dtype fixes only) | Listed as project 5 "Mind the CHL Gap" on oceanhackweek.org/ohw25/projects. It is the only CHL project in that list. |
| **2025–26 OST**: continuation book | ocean-satellite-tools/mind-the-chl-gap @2fe06bb | Code from 2025-05 to 2025-10-16; the 2026-04 commits are PDFs only. Its U-Net notebooks are byte-identical (md5) to 2024. |
| **2026 SAFS**: summer 2026 continuation, not a hackweek | SAFS-Varanasi-Internship/mindthegap: `main` @4969cfb (M26, 2026-09-03) and `troy-branch` @b9cedee (T26, 2026-09-20) | troyrusso/noaa-chlorophyll-gap-filling is a README that points here. The 10-02 survey missed this repo. |

**No GeoSMART Hackweek 2025 took place.** The geo-smart org has 65 repos and none from 2025 relates to CHL. geosmart.hackweek.io redirects to 2024.hackweek.io. UW eScience's 2025 hackweek (2025.hackweek.io) was the ICESat-2 Hackweek.

Other repos, not covered further:
- satgapfillr: stubs only.
- SPP-1/Chlorophyll-Gapfiller: kriging/IDW on L2 scenes at Long Island Sound.
- troyrussorepo: a copy of OHW25.
- kaira-repo-summer2026: empty.
- Forks (andypbarrett, scottyhq, LilacHo): no new results.

Book sites: the OHW25 site (TOC: U-Net Modules, Data Prep, Fit, Viz, unet, dincae) and the OST site (Data pages, CNN Parts I–IV, Data Prep I/II, ConvLSTM) list only pages that are in the repos. Neither site lists the U-Net or CNN_2D notebooks.

## Best gap-filling model per project

### 2024: U-Net "with CHL"

Source: `notebooks/U-Net_Tutorial.ipynb`. Saved as `models/2015_3_full_2days/UNet_DoubleConv_mse.keras`; this is the model behind `level3_pred`.

- **Fake clouds (cell 9).** `masked_CHL = where(cloud[i+10] == 0, NaN, lnCHL)`. Index i runs over records, not dates: days with all-NaN L3 are dropped first, and the index wraps circularly at the end.
  - `CHL_cmes-cloud` codes are 0 = observed, 1 = cloud, 2 = land. I checked this against L3 finiteness on 3 days.
  - **Hidden = observed at t ∩ observed at t+10 records.** The 10-02 survey said "cloudy at t+10"; that is wrong.
  - The mask is fixed, one per day, and never resampled. There is no filter on cloud fraction.
- **Context days.** CHL of records t±1 enters unmasked (`prev_day`/`next_day`).
- **Inputs (13 channels).** u, v wind, SST, air temp, sin/cos, masked CHL, CHL t±1, plus flags for land, real cloud, valid CHL and fake cloud.
- **Target and loss.** ln L3, z-scored with mean/std over all 1997–2022 days. NaN→0. MSE over all pixels, land and cloud zeros included (cells 12, 15).
- **Split.** Chronological: train 2015-01-01..2018-01-01, val 2018, test 2019. Slice bounds are inclusive.
- **Box.** 32°N–11.75°S, 42–101.75°E (176×240).
- **Reported number.** Test MSE 0.0082 / MAE 0.0301 (cell 29): z units, all pixels. The hidden-pixel MAE was only plotted, per day for 2020 (`UNet_Evaluation` cells 17-18).

### 2025 OHW: the same U-Net on an Arabian Sea crop

Sources: `book/1-U-Net_Data_Prep`, `book/2-U-Net_Fit`, and `mindthegap/create_zarr.py:120-200`.

- **Fake clouds.** Same rule as 2024, inverted the same way (`create_zarr.py:163-166`). The all-NaN-day filter is commented out (Data_Prep cell 8), so the shift is 10 calendar days over 16071 days (1979–2022).
- **Unchanged from 2024.** Unmasked t±1 CHL, the same 13 channels, MSE over all pixels with NaN→0, train 2015–17 / val 2018 / test 2019.
- **Box.** 31–5°N, 42–80°E (104×152 after crop).
- **Reported number.** Test MAE 0.0354 (z units, all pixels).
- **2026-06 edits.** Package installs, dtypes, batch size 1. No pipeline change.
- **Public zarr.** `gs://…/2015_3_ArabSea.zarr`.

### 2025–26 OST

- **Hidden-pixel gap filler:** the 2024 U-Net notebooks, copied byte-identical.
- **Its own models:** CHL-from-physics CNNs. None use fake clouds.
- **Newest model, `CNN_2D.ipynb` (2025-10-16) with `model_utils.py`:**
  - Target: ln **L4 `CHL_cmes-gapfree`** (`Data_Prep_CNN`: `resp_var`). Box 35→−5°N, 45–90°E, which is our box. 9207 days.
  - Inputs: z-scored SST and salinity, ocean mask, sin/cos, x/y/z geo. **No CHL inputs and no neighbour days.**
  - Days with more than 2 % ocean NaN are dropped.
  - Split: random 70/20/10 days from 2000, 2010 and 2020 (seed 42).
  - Loss: "masked MAE" over all ocean pixels.
  - Reported: test 0.937.
  - **Normalisation bug in the masked MAE (my reading of the code).** It divides the sum over the batch by the ocean-pixel count of one frame, so it equals per-pixel MAE × batch size. The test batch is 4, so the true per-pixel MAE is ≈ 0.23 ln, not 0.937 (inferred, not recomputed). The same function in CNN Part II gives 0.926.
- **ConvLSTM:** forecasts t+1 from a 5-day window of SST, salinity and CHL, over all pixels.

### 2026 SAFS

**M26 (`main`): packaged pipeline.**
- Default clouds: Gaussian noise smoothed (σ_xy 6, σ_t 2) and thresholded to 40 % of **all** pixels, then intersected with observed pixels. Drawn from a precomputed cloud bank with random offsets and flips. Fixed per `cloud_seed`. (`data.py:500-551, 1115-1214`; `options.py:349-353`.)
- `cloud_mode="shift"` reproduces the inverted 2024 rule (`data.py:1203-1207`).
- Neighbour days are t±1 of the fake-clouded field (`data.py:1265-1266`).
- Loss: masked MSE on observed ∪ hidden pixels (`losses.py:27-32`). Validation metric: `fakecloud_mae` on hidden pixels (`losses.py:44-50`).
- Split: random days (`options.py:238-244`).
- No executed run with this loss was found.

**T26 (`troy-branch`): the only clean IO hidden-pixel result, `book/10-IO_Env_Ablation.ipynb`.**
- Data: full IO box, 2021–22 (730 days), all calendar days.
- Clouds: blobs with coverage 0.2, σ_xy 12, σ_t 1.5, seed 0, fixed (`selfsup.py:131-133`).
- Neighbour days: t±1 from the masked CHL (`selfsup.py:148-151`).
- Loss: masked MSE on **hidden pixels only** (cell 7).
- Inputs: the flags (including fake-cloud) plus geo x/y/z; the "env" variant adds SST, wind and air temp.
- Split: random 80/20 days. The val days also drive early stopping.

## Differences vs ours

Columns: 2024 = GeoSMART; OHW = OHW25 (same as 2024 unless stated); OST = CNN_2D; SAFS = M26 / T26-nb10.

The direction column is a hedged judgment of each choice's effect on that project's hidden-pixel MAE relative to ours. "Easier" means lower MAE is expected. None of these directions is measured.

| difference | 2024 | OHW | OST | SAFS | ours | likely direction |
|---|---|---|---|---|---|---|
| Fake-cloud rule | obs_t ∩ obs_(t+10 records); ≈62 % of observed hidden (one-chunk check) | same, but t+10 calendar days | none | blobs: 40 % (M26) or 20 % (nb10) of all px, ∩ observed | obs_t ∩ ¬obs_donor, real cloud shapes; ≈69 % of observed hidden (`clouds.py:35-37`) | 2024/OHW: unclear. SAFS: easier (smaller, rounder gaps) |
| Donor | fixed +10 records (same season; lag 11–17 d on 145 days; last 10 days wrap to 1997) | fixed +10 days | – | no donor | val/test: random day in the split, fixed seed; train: ±15 DOY any year, redrawn each epoch (`clouds.py:40-64`; `train_pinn.py:342-343, 750-751`) | unclear |
| Context-day CHL | t±1 **unmasked** | same | none | t±1 masked by their own clouds, correlated in time | t±4 masked by the donor's own cloud sequence (`train_pinn.py:517-523, 585`) | 2024/OHW: easier. SAFS: similar. OST: harder |
| Physical / static inputs | wind, SST, air temp, sin/cos (day t only) | same | SST, salinity, geo, sin/cos | M26: none by default; nb10 env: SST, wind, air temp | 9 features × 9 days, DOY climatology (residual), lat/lon/depth, year (`train_pinn.py:569-617`) | ours easier |
| Mask channels | fake-cloud and valid flags given as input | same | ocean mask | estimate/observed flags | one visibility mask per day; hidden and real cloud look the same | unclear |
| Target | ln L3, z-scored over all years | same | **ln L4** gap-free | ln L3 | ln L3, z-scored on train years (`train_pinn.py:335-338`) | OST: easier (L4 is smooth) |
| Loss pixels | all px, NaN→0 (land/cloud = 0) | same | all ocean px | M26: observed ∪ hidden; nb10: hidden only | L1 on hidden + 0.1 × visible + t±1 heads + gradient term ×100 (`train_pinn.py:644-654`) | 2024/OHW/M26: harder. nb10: similar |
| Clouds fixed or resampled | fixed | fixed | – | fixed per seed | train: redrawn each epoch | theirs harder |
| Reported metric | all px, z units, zeros included | same | ocean px × batch size (bug) | hidden px, pooled, ln | hidden px, pooled, ln (`train_pinn.py:792`) | 2024/OHW/OST: not comparable |
| Split | chronological by year (2015–17 / 2018 / 2019) | same | random days (2000, 2010, 2020) | random days (M26: 2010–19; nb10: 2021–22) | whole years interleaved: val 2001/07/13/18, test 2004/10/16/20 (`data.py:109-119`) | OST/SAFS: easier (val days have training neighbours) |
| Box | 32–11.75°S, 42–101.75°E | 31–5°N, 42–80°E | 32–5°S, 45–90°E (same as ours) | full IO box | 32°N–5°S, 45–90°E (`data.py:20`) | unclear |
| Days kept | days with ≥1 L3 px; neighbours by record | all calendar days | days with ≤2 % NaN | all calendar days | days with ≥1 L3 px; neighbours by date (`train2.py:95-99`) | unclear |

## Item 4: can we score `level3_pred` head-to-head?

**Yes, on their own hidden set, and on its intersection with ours.**

**Producer.** `notebooks/UNet_Evaluation.ipynb` (identical in OST):
- Cell 4 (`save_predictions`) and cell 7 run the 2024 model above on the standardized zarr `2015_3_full_2days` for 1997–2022.
- The output goes to `../shared/yifei_data/U-Net_with_CHL_pred.zarr`. The GCS copy was uploaded 2025-04-30; the upload code was not found.
- The GCS time axis (9193 days) and grid (176×240) match my rebuild from IO.zarr. The first and last 100 dates are identical.

**What the model saw for each day t.** Its own `masked_CHL` with the hidden pixels at 0, unmasked CHL of records t±1, and the flags. `level3_pred` is NaN wherever L3 is NaN.

**Check (one chunk, 2021-02-07..05-17).** This is consistent with the hidden pixels really being hidden on day t.

| pixel set | share of observed | `level3_pred` MAE (ln) |
|---|---|---|
| code's hidden set | 62.5 % | 0.104 |
| input-visible pixels | 37.5 % | 0.014 |
| hidden set, our box only | — | 0.109 |

**Deterministic recovery.**
- Take the days in their box with any `CHL_cmes-cloud == 0` and sort them (9193 days, index i).
- hidden(i) = obs(i) ∩ (cloud(i+10 mod 9193) == 0).
- The public `unet_tutorial/` zarr (16071 days, 177×241, written 2025-08) is a rebuild on an unfiltered axis. Its `fake_cloud_flag` is **not** this mask.
- Our box is rows 0–148, cols 12–192 of their grid. I checked this.

**Usable years.** Their training years (2015–17 plus 2018-01-01), their val year (2018) and our 2016 are excluded.
- Our val years: 2001, 2007, 2013.
- Our test years: 2004, 2010, 2020.

**Usable pixels:**
- (a) Their hidden set. This scores `level3_pred` under their own protocol. Our model would need an `--eval-only` pass with that mask, because our `preds.npz` exists only under our own masks.
- (b) Our fixed hidden set ∩ theirs. Both models had these pixels hidden on day t. Context still differs: their t±1 are unmasked, ours persist donor clouds.
- (c) Our hidden pixels outside their hidden set are not usable. Their model saw those values in its input.

## Item 5: hidden-pixel numbers in ln CHL

Only the 2026 SAFS repo reports any. The 2024, OHW and OST projects report none.

| source | MAE (ln) | definition | comparability to our 0.130 |
|---|---|---|---|
| T26 nb10, cell 11 (2026-08-06) | **0.1694** CHL-only; **0.1620** +env; nearest-neighbour fill 0.2975 (n = 366,861 px); persistence 0.1353 (n = 63,130, only px with a visible previous day) | IO L3, full box, 2021–22. 146 random val days, also used for early stopping. Fixed blobs (20 %, σ 12). Pooled over hidden px | Low. Years outside our store; random-day split; smaller gaps; fake-cloud flag as input; no separate test set |
| T26 nb11, cell 9 | 0.1295–0.1859 by tile size (whole frame 0.1684) | same data and clouds, but a per-frame mean over **all 730 frames, training frames included** | No |
| T26 nb5, cells 13-14 (2026-07-10) | 0.217–0.266 by strategy (n = 23,420 px). "random" strategy: test 0.201 | Arabian Sea 25–5°N, 50–78°E. 2024 pipeline (inverted shift, unmasked t±1, MSE on all px). Train 2015–17, test 2019 (10 dates) | Low |
| Capstone (`final_notebooks`, md 6 and 10) | 0.168–0.311 vs persistence 0.175–0.223; neighbour-day channels worth ~21 % with independent clouds vs ~4 % with persistent clouds | PACE 0.1° and GlobColour 4 km, not IO 0.25° | No |

**[unverified]**
- The GCS `level3_pred` is the exact output of `UNet_Evaluation` cell 7. Shapes, dates and the one-chunk error split agree.
- No other 2025 CHL gap-fill hackweek project exists. GitHub search hit its rate limit partway through; the geo-smart org listing, the OHW25 project list and hackweek.io were checked.
- The ConvLSTM's `CHL` variable comes from a 2023 intern zarr; which product it is was not checked.
- The CNN_2D per-pixel MAE of ≈0.23 is derived from code reading, not recomputed.
