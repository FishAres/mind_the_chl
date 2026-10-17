# Mind-the-CHL-gap: repos, models and related papers (survey 2026-10-02)

Compiled by a delegated search agent; sources are linked. Anything that could not be opened is marked **[unverified]**.
Assumptions: "ln" = natural log (their notebooks use `np.log`); "z-units" = ln CHL z-scored with the train mean/std.

## (a) Repos, books, presentations

| Item | URL | One line |
|---|---|---|
| geo-smart/mind-the-chl-gap (2024) | https://github.com/geo-smart/mind-the-chl-gap | GeoSMART Hackweek Aug 2024 repo (created 2024-08-16, last push 2024-08-23). U-Net tutorial, U-Net variants, UNet_Evaluation, PINN/DeepONet notebooks, `src/` torch UNet+PDE, AGU abstract, CoastWatch slides. |
| 2024 book | https://geo-smart.github.io/mind-the-chl-gap/ | TOC: dataset pages; "Simple CNN" (Data_Prep, CHL_prediction_CNN); "ConvLSTM". U-Net/PINN notebooks are in the repo but not in the book TOC. |
| ocean-satellite-tools/mind-the-chl-gap (2025-26) | https://github.com/ocean-satellite-tools/mind-the-chl-gap | Continuation (created 2025-05-30, updated 2026-04-27). Adds CNN Parts I-IV, CNN_2D (+BRT), `model_utils.py`, Yifei final slides, 2026 eScience Accelerator + HPC proposals. |
| 2025 book | https://ocean-satellite-tools.github.io/mind-the-chl-gap/ | TOC: data pages; "CNNs for gridded data" (Simple CNN, Data Prep I, CNN Part I, Data Prep II, CNN Part II land+season, Part III xbatcher, Part IV multi-year); ConvLSTM. |
| ocean-satellite-tools/mind-the-chl-gap-2024 | https://github.com/ocean-satellite-tools/mind-the-chl-gap-2024 | Archive copy of the geo-smart 2024 repo (identical tree). |
| oceanhackweek/ohw25_proj_gap (OHW 2025) | https://github.com/oceanhackweek/ohw25_proj_gap ; book https://oceanhackweek.org/ohw25_proj_gap/ | Aug 2025: U-Net tutorial re-run on an Arabian Sea crop (`book/0-3-U-Net_*.ipynb`), `mindthegap` package, `dincae.md` (DINCAE described, "not successful" in getting it to run). Final presentation on gamma.app (403, **unverified**). |
| ocean-satellite-tools/satgapfillr | https://github.com/ocean-satellite-tools/satgapfillr | Jul 2025 package skeleton (`UNetGapFiller`, `simulate_cloud_mask` is a placeholder). No results. |
| Indian Ocean zarr handbook | https://safs-varanasi-internship.github.io/indian-ocean-zarr/ | How IO.zarr was built (Minh Phan 2023): 0.25°, -12…32°N, 42…102°E, CHL from Oct 1997. |
| GCS bucket | gs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/ | `IO.zarr`, `IO_rechunked.zarr`, `cnn_tutorial` (+`cnn_1_day/100_day/200_day` chunking variants), `unet_tutorial`, `2015_3_ArabSea.zarr` (+`_20260710`), `U-Net_with_CHL_pred.zarr`, `U-Net_without_CHL_pred.zarr`. Pred zarrs: vars `level3_pred`, `gapfree_pred`, shape (9193, 176, 240), 1997-2022, ln units. |
| AGU 2024 abstract | ADS 2024AGUFMED13E2471H; PDF in repo `presentations/Abstract - AGU 2024.pdf` | Hang, Sinha, Holmes. "U-Net ... PINN ... compared with random forest and gradient boosted trees". No numbers. |
| CoastWatch seminar 8 Aug 2024 | repo `presentations/Ocean Chlorophyll 2024-08-08.pdf`; figshare doi 10.6084/m9.figshare.27936036.v1 | Train 2015-17, val 2018, test 2020 (slide says 2020; notebooks use 2019). Takeaways only, no numbers. |
| Yifei Hang final slides | repo `presentations/Yifei Final - Chlorophyll-a Gap-Filling Using U-Net.pdf` | "Does environmental variables help? Sadly, not really"; percent-difference maps only. |
| 2024 results Google Slides | docs.google.com/presentation/d/1etA0GIpuRJrahZnKkC36YrCN8A0dmnXPTETdv5v1m0E | Linked from README; not fetchable **[unverified]**. |
| Other GitHub hits | SPP-1/Chlorophyll-Gapfiller (2025-10), troyrusso/noaa-chlorophyll-gap-filling (2026-09) | Found by name search; relation to this project **unverified**. |

No paper or preprint from the group exists beyond the AGU abstract. The 2026 Accelerator proposal states "cross-validation results ... demonstrate that model predictions perform well" with no numbers.

## (b) Every model in the repos/books

Region codes: **FULL** = 32°N-11.75°S, 42-101.75°E (176×240); **OURS** = 35°N-5°S, 45-90°E (149×181); **ARAB** = 5-31°N, 42-80°E (104×152 after crop).

| # | Model / file | Architecture | Inputs | Target | Loss | Period / region / split | Scored pixels | Reported metric |
|---|---|---|---|---|---|---|---|---|
| 1 | **U-Net with CHL**: `U-Net_Tutorial.ipynb` (both repos), `UNet_DoubleConv` | Keras U-Net: 3 encoder blocks (2×Conv3×3 ReLU, MaxPool, BN) filters 64/128/256; decoders Conv2DTranspose+concat+2×Conv+BN; final 3×3 linear; 2,157,825 params | 13 ch: u_wind, v_wind, sst, air_temp, sin/cos time, masked_CHL (ln, z-scored), prev_day_CHL, next_day_CHL, land_flag, real_cloud_flag, valid_CHL_flag, fake_cloud_flag | z-scored ln(CHL_cmes-level3), NaN→0 | MSE over all pixels (land/cloud zeros included) | FULL; train 2015-17, val 2018, test 2019; fake cloud = pixel observed today ∩ cloudy at t+10 d | all pixels | Test MSE 0.00815, MAE 0.0301 (**z-units**, zeros included) |
| 2 | U-Net variants: `PrevDay+NextDay_UNet_MSE.ipynb`: `UNet_noDropout`, `UNet_UpDropout`, `UNet_AllDropout`, `UNet_DoubleConv`, `UNet_Old` | as #1 with single conv / dropout 0.3 / Sequential-upsample variants | as #1 | as #1 | MSE | as #1 | all pixels | Printed "Test MAE 0.0135, MSE 0.0442" (noDropout) and "0.0123 / 0.0416" (UpDropout). `test_mae, test_mse = model.evaluate()` unpacks Keras `[loss=mse, mae]`, so the labels are swapped: MSE≈0.0135, MAE≈0.044 z-units. Last two variants not printed. |
| 3 | **U-Net without CHL**: `UNet_DoubleConv_noCHL`, data label `2011_6_no_CHL`; `UNet_Evaluation.ipynb`; GCS `U-Net_without_CHL_pred.zarr` | as #1 | wind, sst, air_temp, season (+flags), no CHL channels | ln L3 | MSE | train 2011-2016 inferred from label; training notebook **not in repo** | – | No number. AGU abstract: "mean-squared error was orders of magnitude higher"; slides: "captures general patterns". |
| 4 | U-Net cross-validation: `UNet_Evaluation.ipynb` | uses #1 predictions | – | ln L3 on fake-cloud pixels (`cross_val_loss_vs_cloud_perc_plot`); ln L4 on all pixels (`comp_to_cmes_loss_vs_cloud_perc_plot`) | – | 1997-2022 predictions | fake-cloud pixels / all | **Daily MAD plotted only**, no numbers printed. |
| 5 | OHW25 U-Net: `ohw25_proj_gap/book/2-U-Net_Fit.ipynb`; model `models/2015_3_ArabSea/UNet_DoubleConv_mse.keras`; GCS `2015_3_ArabSea.zarr` | as #1 | as #1 | as #1 | MSE | ARAB; train 2015-17, val 2018, test 2019 | all pixels | Test MSE 0.01084, MAE 0.0354 (z-units) |
| 6 | **Simple 3-layer CNN**: `book/notebooks/CHL_prediction_CNN.ipynb` (2024 book "Fit CNN") and `CHL_prediction_CNN_Part_1.ipynb` | Conv2D 64 → 32 → 1 (3×3, BN, dropout 0.2), ~20k params | sst, so (raw, not normalised in the 2024 version) | **ln(CHL_cmes-gapfree) = L4**, NaN→0; days with >5 % ocean NaN dropped | MAE, all pixels incl. land | OURS; 2020 only; **chronological 70/20/10** | all pixels incl. land zeros | Test MAE **0.1907** (2025 run) / **0.2333** (Part I run), ln units |
| 7 | CNN Part II: `CHL_prediction_CNN_Part_2.ipynb` | as #6 | sst, so (z-scored) + sin/cos time + ocean_mask | ln L4 | masked MAE (ocean only) | OURS; 2020; **random** 70/20/10 | ocean pixels | Test masked MAE **0.926** ln |
| 8 | CNN Part III (xbatcher): `..._Part_3.ipynb` | as #6 | as #7 | ln L4 | MAE | OURS; 2020; random split | – | no test number printed |
| 9 | CNN Part IV (multi-year): `..._Part_4.ipynb` | as #6 | as #7 | ln L4 | MAE | train 2015+2020 (50 % sample); val code reuses `train_years` (bug) | monthly R²/RMSE/MAE/bias plots for 2000/05/10/15/20 | plots only |
| 10 | CNN_2D (Oct 2025): `CNN_2D.ipynb` + `model_utils.py` | as #6 | sst, so + sin/cos + x/y/z_geo + ocean_mask | ln L4 | masked MAE | OURS; years 2000, 2010, 2020; random 70/20/10; ≤2 % NaN | ocean pixels | Test masked MAE **0.937** ln; SSIM plots; BRT (`train_brt_from_splits`) run, no number captured |
| 11 | ConvLSTM: `CHL_prediction_ConvLSTM_.ipynb` (+torch twin) | ConvLSTM2D(32) → Conv2D 64/32/1 | 5-day window of sst, so **and ln CHL (L4)** | ln L4 day t+1, NaN→0 | MAE | OURS; 2020; chronological 70/20/10 | all pixels | Test MAE **0.1045** ln; one day (2020-12-02): model 0.161 vs previous-day persistence 0.132 |
| 12 | PINN / DeepONet: `notebooks/PINN_Tutorial.ipynb`, `Data_Drive_PINN.ipynb`, `PINN_plots_.ipynb`, `src/pdes.py`, `src/models.py` | DeepXDE DeepONet branch [8,64,64,64], trunk [3,64,64,64], tanh; PDE: ∂²U/∂lat²+∂²U/∂lon²+∂²U/∂t² = ρ(env vars); Robin BC | air_temp, sst, curr_dir, ug_curr, u_wind, v_wind, v_curr | z-scored ln CHL (L3 in `Data_Drive`) | PDE residual + BC | 2018-04…2022-04 (OURS) / 2019-2022 (FULL); 10k iterations | – | none; slides: "deviates significantly in overall value" |
| 13 | Torch U-Net: `src/models.py` + `src/training.py` | 4-ch Conv 64/64, 128/128, 2 pools, no skips | u_wind, v_wind, sst, air_temp | CHL_cmes-level3 (raw) | MSE | 2019-2022 FULL | – | never run in repo |
| 14 | Feature importance: `notebooks/Feature Importance Analysis.ipynb` | PCA + lstsq, XGBoost, decision tree | 14 env vars | ln CHL | – | Sep-Dec 2022, OURS | – | no numbers captured |
| 15 | RF / gradient-boosted trees (AGU abstract, slide 7) | – | 11 env vars | – | – | – | – | used for variable importance; no numbers |
| 16 | DINCAE / DINEOF | described only (`ohw25 book/dincae.md`); DINEOF only as a "TO DO" dataset variable | – | – | – | – | – | not run |
| 17 | satgapfillr `UNetGapFiller` | skeleton | – | – | MSE | – | – | none |

Climatology, persistence, kriging, nearest-neighbour baselines: **none** in any repo (only the one-day persistence check in #11).

## (c) Recent papers (2024-2026) on ML CHL gap filling

| Paper | Data / region | Protocol | Headline numbers |
|---|---|---|---|
| Nguyen, Dorffer, Jourdin, Fablet 2025, IEEE JSTARS 18 (arXiv 2503.11588): 4DVarNet ocean colour | CMEMS L3 Chl & BBP, Med (Gulf of Lion 240 km) & North Sea; train 2017-18, test 2019-20 | Scored on pixels present in target but absent in observation (hidden); RMSLE (log base not stated) | Med Chl RMSLE: DInEOF 0.112, UNet 0.095, 4DVarNet 0.058 |
| Hong et al. 2023 ESSD 15:5281 (OCNET); Zenodo 17377256 (v2, 2001-2023, Oct 2025) | Global open ocean 45°N-45°S, 0.25° daily; target NOAA MSL12; inputs monthly SST/SAL/PAR, daily SST/SSP, OCCCI climatology | 405 regional U-Nets on 64×64 patches; 70/15/15, 2021 held out | median r 0.4-0.6 per region; RMSE "<0.2, most <0.1" (log10 implied, **unverified**) |
| Xu & Li, NeurIPS 2025 (PPIDM) | Secondary sources disagree: BSOSE 1/6°, 3-day, 2008-12, log Chl vs "MODIS". PDF blocked **[unverified]** | infilling and forecasting of 64×64 patches | RMSE infilling 0.270 (PPIDM) vs 0.410 vanilla diffusion vs 0.538 PIDM; forecast 0.268 vs 0.306 |
| Barth et al. 2024, Ocean Sci. 20:1567: diffusion ensemble, Black Sea | OLCI 300 m, 2016-2023, log10 | hidden pixels using cloud masks from other images (same idea as ours) | RMSE 0.180 (diffusion) vs 0.183 (DINCAE) log10 mg m⁻³ |
| Mehdipour et al. 2025/26 GMD (egusphere-2025-112): DINEOF vs DINCAE, Atlantic | OLCI PFTs, 3 years | hidden-pixel CV + in situ | DINCAE RMSLE 66 % lower for total Chl; DINEOF 68 % higher R² vs in situ |
| Wang, Wang, Zhou 2026, Env. Res. Comm. 8(4): dynamic masking, Bohai/Yellow Sea | Himawari-8 L3, 0.05°, 2015-2022 | cloud-pattern / scatter / block masks; scored on masked pixels | U-Net R² 0.942 (MSE 1.962, linear) vs DINCAE 2.0 max R² 0.905 |
| Remote Sensing 2026, 18(11):1736: Chesapeake Bay comparison | coastal, OLCI 2023-25 | IDW, DINEOF, KNN, RF, ET, XGBoost, LSTM, Temporal DINCAE; linear mg m⁻³ | XGBoost R² 0.86, RMSE 9.61; LSTM RMSE 5.87; T-DINCAE R² 0.84, RMSE 11.15 |
| Modi, Roxy, Ghosh 2022 Sci. Rep.: Monte-Carlo gap filling, tropical Indian Ocean | OC-CCI 4 km → 1°, 8-day, 1998-2019 | not ML; validation r = 0.80 vs BGC-Argo | regional precedent only |
| Ecol. Informatics 2025 (S1574954125004911): N. Indian Ocean ML | monthly 1998-2022; XGBoost/LSTM/RF/SVR/MLP | **forecasting**, not gap filling; paywalled **[unverified]** | – |
| Zenodo 18190723 (Zhang 2026) 25-yr global 9 km DCT-PLS; EGU25-18806 Lakra et al. diffusion Chl prediction | – | no metrics available | – |

Nothing found that gap-fills daily 0.25° CHL specifically for the Arabian Sea / N. Indian Ocean with ML other than this project.

## (d) Comparability to our numbers

Our protocol: ln(L3), hidden pixels (observed, masked by another day's cloud), OURS region, 2020 chronological 70/20/10 or 1997-2020 year-interleaved.

| Their number | Comparable? | Why / what differs |
|---|---|---|
| U-Net with CHL: test MAE 0.0301 z-units (#1); OHW25 0.0354 (#5); variants ≈0.044 (#2) | **No** | z-scored, averaged over all pixels including land and cloud zeros (target and most inputs are 0 there), FULL/ARAB region, 2015-17/18/19 split. The z-score std is not public (`.npy` not in repo). |
| U-Net hidden-pixel MAD (#4) | **Protocol matches ours** (ln L3 on fake-cloud pixels) but **no number was published**; only daily plots. | Their fake cloud = observed today ∩ cloud at t+10 d; ours = cloud pattern of another day. Predictions `level3_pred` (1997-2022, FULL) are public in GCS, so their hidden-pixel MAE can be computed by us. |
| Simple CNN sst+so: MAE 0.191 / 0.233 ln (#6) | **No**, but closest in setup | Same region, year, chronological split and architecture as our baseline port. Differences: target is L4 gap-free (smoother), scored on all pixels incl. land zeros (prediction≈0 vs target 0 deflates MAE), and days with >5 % NaN removed. Our SST+salinity hidden-pixel MAE on ln L3 is 0.278; our "as published" variant (NaN→0, loss everywhere, L3 target) gives 1.58. The difference from their 0.19 is consistent with the L4-vs-L3 target choice (observation, not verified). |
| CNN Part II 0.926, CNN_2D 0.937 masked ln MAE on L4 (#7, #10) | **No** | random day split, ocean-only masked MAE, L4 target; larger than #6 for reasons not discussed in the notebooks. |
| ConvLSTM 0.1045 ln (#11) | **No** | ln L4 of day t itself is in the input window; all pixels; effectively persistence-like (persistence 0.132 vs model 0.161 on the one printed day). |
| U-Net without CHL (#3) | **No number** | Only qualitative. Our physics-only runs would be the first quantitative result for this configuration on this dataset. |

Our full-period val MAE 0.138 / RMSE 0.231 / r 0.96 (ln L3, hidden) has no counterpart in their repos. Among external papers, only Barth 2024 (0.180 log10 RMSE ≈ 0.41 ln) and Nguyen 2025 (RMSLE 0.058-0.112) use a hidden-pixel protocol, but on 300 m / 1 km coastal products, so the numbers are not transferable.

## (e) What would enable a direct comparison

1. **Score their public U-Net predictions with our metric**: `gs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/U-Net_with_CHL_pred.zarr/level3_pred` vs ln(`CHL_cmes-level3`) on their fake-cloud pixels (observed at t, cloudy at t+10 d), restricted to our box and to 2019 (their test year) and 2020. `level3_pred` is NaN where L3 is NaN and was produced with masked_CHL as input, so it is their cross-validation prediction.
2. **Adopt their fake-cloud rule (t+10 d)** as an alternative mask in our pipeline; report both.
3. **Their split**: train 2015-17, val 2018, test 2019 on the FULL region (176×240) with inputs u_wind, v_wind, sst, air_temp, season, CHL t±1.
4. **L4 (`CHL_cmes-gapfree`) as target on all ocean pixels, 2020 chronological**, to put a number next to their 0.19/0.23.
5. **Arabian Sea crop** 5-31°N, 42-80°E using public `2015_3_ArabSea.zarr`. The z-score mean/std are not public; recover by regressing against ln L3 from IO.zarr.
6. **MSE in z-score units with NaN→0** on all pixels, to reproduce 0.030/0.035 literally (low value).
7. **No-CHL U-Net**: `U-Net_without_CHL_pred.zarr/gapfree_pred` is public and can be scored against ln L3 observed pixels for any year, the first number for that model.
