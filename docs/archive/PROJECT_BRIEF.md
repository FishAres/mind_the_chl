# Project brief: filling cloud gaps in satellite chlorophyll with CNNs and physics-informed losses

Written 2026-10-03 as a self-contained hand-off; section 11 added 2026-10-04 with everything since. Everything below was measured in this workspace unless marked as literature.

## 1. Goal

Reconstruct surface chlorophyll-a (CHL) under clouds in daily satellite ocean-colour imagery of the western Indian Ocean, and find out whether adding physics (a transport equation, wind-driven upwelling, climatology) to a convolutional network improves the reconstruction over a purely data-driven network. The work builds on the open "mind the chl gap" project (GeoSMART / NOAA NWFSC, 2024-2026), which released the dataset and tutorial models but no quantitative benchmark.

## 2. Data

`IO.zarr`, public at `gs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO.zarr` (24.7 GB, 66 GB uncompressed). Daily, 0.25°, 1979-2022, 32°N-12°S, 42-102°E. Variables used:

| variable | source | note |
|---|---|---|
| `CHL_cmes-level3` | Copernicus GlobColour L3 | the gappy observation; our **target** as ln(CHL) |
| `CHL_cmes-gapfree` | GlobColour L4 | operational gap-filled product; interpolation, not observation; not used as truth |
| sst, so (salinity), mlotst (mixed-layer depth) | GLORYS reanalysis | NaN ~38 % of ocean pixels (coastal/shelf gaps); `so` ends 2020 |
| air_temp, u_wind, v_wind | ERA5 | complete |
| sla (sea level anomaly) | CMEMS altimetry | |
| u_curr, v_curr | OSCAR total surface currents | NaN ~39 % |
| topo, land mask | | 0 = ocean, 2 = land; 43 lake pixels unflagged |

Study box: 32°N-5°S, 45-90°E, 149 × 181 pixels, 16,695 ocean pixels. Period for all main runs: 1997-10-01 to 2020-12-31 (8,461 days with at least one CHL observation). Real-cloud fraction (no L3 observation) averages 70 % of ocean pixel-days; CHL coverage rises from ~13 % (1997) to ~53 % (2020) as sensors were added. Known data issues: ERA5 missing whole months in 1979-92 and 2021-22; `mlotst` for 1979-92 is a copy of `u_wind` (build bug); so we never use pre-1997 data.

Local copies live on the SageMaker instance under `~/mind_the_chl_gap/data/` (zarr, plus a float16 memory-mapped store `cnn_store_1997_2020/`, ~5 GB). Rebuild with `python fetch_data.py && python -m chl_cnn.store` (3 min + 2 min).

## 3. Evaluation protocol (identical for every run)

* **Target**: ln(CHL L3). All errors below are in ln units (0.14 ≈ 15 % relative error).
* **Fake clouds**: for each day, hide the observed pixels that are cloudy on a randomly drawn donor day (the donor's real cloud pattern). Hidden pixels have a real observation as truth. For context days t±k the donor's cloud sequence (donor±k) is applied too, so a hidden pixel stays hidden across the input window the way real clouds persist (without this, the model copies the pixel from a neighbouring day and scores an unrealistic 0.106).
* **Score**: MAE on hidden pixels. Val and test days use fixed donors (seeded), so every run is scored on the same ~5.0 M val pixels. Pixels under real clouds are never scored (no truth) and never enter the data loss.
* **Split**: year-interleaved. Val years 2001, 2007, 2013, 2018 (1,458 days); test years 2004, 2010, 2016, 2020 (1,463 days); train the remaining 5,540 days. A chronological split fails because clear days only exist late in the record. **All model selection so far is on val; test is untouched except for the early baseline runs.**
* **References**: day-of-year climatology from training days scores 0.358 on the same hidden pixels.

## 4. Models

Code: `chl_cnn/` (PyTorch). `train2.py` = data-only runs; `train_pinn.py` = runs with the 3-day head and optional physics loss; both share the input pipeline.

**Inputs (full set, 25 channels)**: 9 physical fields z-scored (NaN → 0); CHL at day t with fake clouds applied + a 0/1 visible flag; CHL at t±1, t±2 with flags; a day-of-year ln CHL climatology channel (8-day bins from training days); lat, lon, log depth, sin/cos day-of-year. The network predicts the anomaly from the climatology channel ("residual"). Loss on hidden pixels of day t (L1) plus 0.1 × the visible pixels.

**Dilated residual CNN** (default): 3×3 conv → 8 residual blocks with dilations 1,2,4,8,16,1,2,4 (64 channels, BatchNorm, GELU) → 1×1 head. 0.61 M parameters, receptive field ~150 px, no downsampling. AdamW 1e-3, weight decay 1e-4, cosine schedule, batch 8, mixed precision. ~40 s/epoch on an A10G.

**U-Net**: double-conv encoder/decoder with skips; 2-level 64/128/256 (1.88 M params, the shape used in the upstream repo) and 3-level 32/64/128/256 (1.93 M).

**PINN head and physics loss** (`train_pinn.py`): the network outputs ln CHL for days t−1, t, t+1 (3 channels). Data loss as above, plus the same hidden-pixel loss on t±1. Physics loss = mean |r| over every ocean pixel with a full 5-point ocean stencil, finite currents and both neighbour days present, **including pixels under real clouds**, after pasting observed pixels back into the predicted fields (hard constraint):

r = (Ĉ(t+1) − Ĉ(t−1)) / 2Δt + u·∇Ĉ(t) − K∇²Ĉ(t) − S,  units ln CHL per day

with first-order upwind advection using the OSCAR currents in index space (cells/day), K = 500 m² s⁻¹, Δt = 1 day. Total loss = data + λ·mean|r|, λ ramped from 0 over the first 3 epochs. Options: L1 or L2 on r; a Courant cutoff; three prior terms (below).

**Prior-knowledge terms tried** (each off by default):
* `--src-clim`: S includes ∂t of the day-of-year climatology (the seasonal cycle is not charged to transport). Mean magnitude 0.0075 /day.
* `--src-ekman`: S includes α·Σκ_k·w_E⁺_k / 30 m, where w_E⁺ is upward Ekman pumping from the ERA5 winds (Large & Pond drag, curl of τ/f plus the β term), averaged over lag windows 0-3, 4-10, 11-20, 21-30 days; α ≥ 0 (softplus) and κ (softmax over windows) learned; multiplied by a taper that is 0 for |lat| < 3° and 1 beyond 5° because f → 0 at the equator.
* `--phys-weight courant`: residual weight 1/max(1, Co_clim), Co_clim = training-mean Courant number of the currents (median 1.18; 63 % of ocean pixels > 1).

**Other input configurations**: `--no-chl-input` (no CHL at any day; data loss on all observed pixels of t−1, t, t+1), `--cloud-input` (observation masks as channels), `--no-chl-today` (no CHL at t, CHL context at t±k kept), `--feat-context K` (physical fields at t±1..K).

## 5. Results (val hidden-pixel MAE, ln CHL; single seed 0 unless stated)

### 5.1 Data-only runs (train2.py), 1997-2020

| configuration | val MAE | val RMSE |
|---|---|---|
| climatology | 0.358 | 0.499 |
| dilated CNN, sst + so, CHL t only | 0.168 | 0.272 |
| + climatology + coordinate channels + residual | 0.167 | 0.271 |
| + CHL context t±2 (donor clouds persisted) | 0.141 | 0.237 |
| + all 9 physical features | 0.140 | 0.233 |
| + 0.1 × visible-pixel loss (`perf_dilated_vis01`, 30 ep) | **0.138** | 0.231 |
| loss L1 vs Huber vs MSE (same config) | 0.140 / 0.141 / 0.143 | 0.233 / 0.233 / 0.235 |

Test MAE for `perf_dilated_vis01`: 0.140 (r = 0.96, R² = 0.92). An earlier 2020-only replication of the upstream tutorial CNN gave test MAE 0.278 with SST+salinity inputs and 0.195 with the CHL image, vs 0.357 climatology; the tutorial's own "NaN → 0, loss everywhere" recipe gives 1.58 (broken on L3).

### 5.2 PINN runs (train_pinn.py), dilated CNN with 3-day head

| run family | λ = 0 | λ = 0.1 | λ = 0.3 | λ = 1 |
|---|---|---|---|---|
| 12 epochs | 0.1444 | 0.1442 | 0.1440 | 0.1450 |
| 60 epochs, best-val checkpoint from epoch 10 | 0.1396 | – | 0.1388 | 0.1393 |
| 12 ep, λ 0.3 + seasonal source | | | 0.1441 | |
| 12 ep, λ 0.3 + Ekman source | | | 0.1442 | |
| 12 ep, λ 0.3 + Courant weighting | | | 0.1441 | |
| 12 ep, λ 0.3 + seasonal + Ekman | | | 0.1443 | |
| 12 ep, λ 0.3 + all three | | | 0.1443 | |

Transport-only residual under real clouds (ln CHL/day): 0.066 (λ = 0) → 0.057 (0.1) → 0.032 (0.3) → 0.020 (λ = 1) at 12 epochs; 0.069 / 0.026 / 0.016 at 60 epochs. On observed pixels (pasted back, only stencil neighbours can move): 0.082 → 0.055.

Learned Ekman coefficient α decayed monotonically from its 0.036 initialisation to 0.005 (source ≈ 4·10⁻⁵ /day, i.e. switched off); κ ended at [0.40, 0.18, 0.18, 0.23].

### 5.3 Input ablations with the PINN trainer, 12 epochs

| inputs | λ = 0 | λ = 0.3 |
|---|---|---|
| no CHL at any day (9 features, climatology, coords) | 0.2228 | 0.2239 |
| + observation masks as channels | 0.2248 | 0.2255 |
| no CHL at t; CHL context t±1; physical fields at t−1, t, t+1 | 0.1621 | 0.1612 |

### 5.4 U-Nets, 30 epochs

| U-Net | inputs / loss | params | λ = 0 | λ = 0.3 |
|---|---|---|---|---|
| 2-level 64/128/256 | upstream repo's: u/v wind, sst, air_temp, season, CHL t±1; MSE | 1.87 M | 0.1508 | 0.1535 |
| 2-level 64/128/256 | full set, L1 | 1.88 M | 0.1426 | 0.1426 |
| 3-level 32/64/128/256 | full set, L1 | 1.93 M | 0.1442 | 0.1446 |
| dilated CNN at 30 epochs for comparison | full set, L1 | 0.61 M | 0.1381-0.1405 | |

### 5.5 Qualitative checks on the λ = 0 / 0.3 / 1 predictions (val days, 12-epoch models)

| | λ = 0 | λ = 0.3 | λ = 1 |
|---|---|---|---|
| mean \|pred − pred(λ=0)\| under real clouds | – | 0.051 | 0.060 |
| mean \|∇ ln CHL\| under real clouds (per 100 km); truth on fully observed stencils = 0.57 | 0.32 | 0.26 | 0.24 |
| \|pred(t+1) − pred(t)\| under persistent clouds; truth on observed pairs = 0.143 | 0.094 | 0.088 | 0.085 |
| median corr(predicted day-to-day change under clouds, advective change from back-tracking yesterday's field along the currents) | 0.084 | 0.082 | 0.080 |
| median corr(predicted change, climatological change) | 0.22 | 0.17 | 0.18 |

Same on clear val days (observed fraction ≥ 0.6, 53 days, fake-cloud pixels with truth): MAE 0.1388 / 0.1383 / 0.1387; gradient amplitude ratio pred/truth 0.58 / 0.53 / 0.51; gradient correlation 0.61 at every λ. With the 60-epoch models: 0.1339 / 0.1330 / 0.1329, gradient cosine 0.45 / 0.46 / 0.47.

Data-only checks: within a day, cloudy pixels sit in stronger and more westerly wind (standardised difference +0.39, +0.29; cloud fraction 0.60 → 0.82 from the lowest to the highest wind-speed decile), lower salinity (−0.14) and lower climatological CHL (−0.12); SST is non-monotone; air temperature, sea level and currents are flat. On observed consecutive days the correlation between the ln CHL tendency and resolved advection is r ≈ 0.11 (design notebook), and the Courant number of the currents exceeds 1 on about half of pixel-days.

## 6. What we have learned

1. **The physics loss does not change the hidden-pixel error.** Across λ ∈ {0, 0.1, 0.3, 1}, three priors, 12 vs 60 epochs, and two architectures, the spread in val MAE is 0.001 on a single seed. We have not yet measured seed-to-seed noise, so none of these differences can be called real.
2. **It does change the field under real clouds**, by 0.05-0.06 ln CHL (≈40 % of the MAE), where nothing is scored, and it does so by smoothing: gradients drop 25 %, small-scale spectral power drops, and the residual falls fourfold without the predicted evolution following the currents any more closely (correlation with advective change stays at 0.08). Case days show filaments under large clouds replaced by smooth fields.
3. **Prior source terms were an order of magnitude smaller than the residual** (seasonal 0.0075 /day vs 0.03-0.07), and the optimiser switched the Ekman source off. The transport residual carries a persistent negative bias over the Arabian Sea in spring that no prior removed.
4. **Information, not architecture, is the limit.** Without the CHL image the model reaches 0.223 from physics + climatology; with the neighbouring days' CHL but no image at t, 0.16; with the image, 0.14. A U-Net of 3× the size with the same inputs matches the dilated CNN to within 0.002; the upstream repo's smaller input set costs 0.008. The loss function (L1/Huber/MSE) matters at the 0.003 level.
5. At 0.25°/daily the transport equation is a weak constraint: resolved advection explains ~1 % of the day-to-day variance of ln CHL on observed pixels, and the grid is Courant-unstable in the boundary currents. The literature's gains from physics-aware gap filling (4DVarNet, diffusion models, DINCAE) come on 300 m-1 km products where fronts are advected coherently within a day.

## 7. Comparison with the upstream project

Surveyed in `notebooks/mind_the_chl_gap_survey.md`. Four iterations (GeoSMART 2024, ocean-satellite-tools 2025-26, OceanHackWeek 2025, a package skeleton); no paper, only an AGU 2024 abstract. Seventeen model artefacts: a Keras U-Net with the CHL image (2.2 M params, MSE on z-scored ln CHL averaged over all pixels incl. land/cloud zeros: test "MAE 0.030" in z-units, not comparable), a U-Net without CHL (no number published), the 3-layer CNN (MAE 0.19-0.23 but on the L4 gap-free product and all pixels incl. land), a ConvLSTM that has the target day in its input window, a DeepONet PINN with a Laplacian PDE ("deviates significantly", no number), tree models for feature importance. No climatology, persistence, DINEOF or DINCAE baseline was run. Their hidden-pixel cross-validation notebook, the one protocol matching ours, plots daily MAD and prints no number. Their 1997-2022 U-Net predictions are public in the bucket (`U-Net_with_CHL_pred.zarr`, `U-Net_without_CHL_pred.zarr`) and could be scored with our protocol without retraining; not done yet.

## 8. Open questions and candidate next steps (none started)

* Seed-to-seed spread of the val MAE (3 seeds of λ = 0 and 0.3) to know what a real difference is.
* Is the Arabian Sea negative residual bias in the data? Compute the transport residual on fully observed consecutive-day pairs, no model, by month.
* Does the hidden-pixel score hide a trade-off? Bin the error by gap size and by how many context days are also hidden; the physics term can only matter on large persistent gaps.
* Score the upstream U-Net predictions and the GlobColour L4 product on our hidden pixels.
* FiLM-style conditioning of the CNN on season/region (a hypernetwork in the weak sense) as a one-run check.
* Higher resolution: surveyed in `notebooks/high_res_datasets_survey.md`. At 4 km, Copernicus GlobColour (global, L3 + daily gap-free L4 with interpolation flags, 1997-present) allows one pipeline for any region; OC-CCI v6 (4 km, L3 only) is sliceable by box over OPeNDAP from PML (`fetch_occci.py`, ~0.6 s/day for the Indian Ocean box). Below 4 km only the Atlantic, Mediterranean and Black Sea have 1 km L3/L4 pairs (two different processing chains). Physical forcing at matching scale is global (OSTIA 0.05°, MUR 0.01°, GLORYS 1/12°, regional 2.5-4 km reanalyses). The L4 products are interpolations of limited accuracy and are used only as an observation mask and a baseline, never as truth. The user is acquiring the Copernicus Black Sea and Mediterranean 1 km data.

## 9. Infrastructure and files

SageMaker instance, A10G 23 GB, 30 GB RAM, 8 CPUs; data on local NVMe. Python 3.12, torch 2.8, numpy pinned to 1.26.4 (newer numpy breaks SageMaker Studio). Training queue for tmux: `train_queue.sh` (idempotent: skips runs with `metrics.json` or already running; logs to `models/queue.log`). Progress figure: `python -m chl_cnn.watch --cat pinn/cnn --every 60 --out models/progress_pinn.png`, or `notebooks/progress.ipynb`.

| path | content |
|---|---|
| `README.md` | setup, data, all CLI examples, regional data access, gotchas |
| `chl_cnn/data.py, store.py, clouds.py` | subset/split/normalisation, memory-mapped store, fake clouds |
| `chl_cnn/models.py` | SimpleCNN, DilatedCNN, UNet (`out_channels`, `depth`) |
| `chl_cnn/train2.py`, `chl_cnn/train_pinn.py` | data-only trainer; PINN trainer (3-day head, residual, priors, input ablations, checkpointing, diagnostic figures every N epochs) |
| `chl_cnn/physics.py`, `chl_cnn/priors.py` | wind stress, Ekman pumping, Courant, tracer terms; precomputed prior fields |
| `models/{cnn,unet,pinn/cnn,pinn/unet}/<run>/` | grouped by trainer and architecture (`chl_cnn/runs.py`, `find_run(name)`): `config.json`, `history.json` (per-epoch val MAE, physics residuals, α, κ), `metrics.json` (val/test scores, per-day), `model.pt`, `preds.npz` (full val/test fields incl. under real clouds), `figures/epoch_NNN.png` |
| `analyses/pinn/` | tables and figures across runs (`pinn_sweep.py`, `pinn_priors.py`, `pinn_runs_summary.py`) and the qualitative checks in 5.5 (`pinn_qualitative.py`, `pinn_clear_days.py`); figures in `analyses/pinn/figures/` |
| `analyses/clouds/`, `analyses/trends/` | cloud cover vs features; yearly drift of CHL and the forcing |
| `notebooks/progress.ipynb` | interactive training-progress figure by run category |
| `notebooks/pinn_results.ipynb` | executed notebook: loss definitions, all tables and figures, observations |
| `notebooks/pinn_design.ipynb`, `pinn_literature_notes.md` | physics derivation, data sanity checks (Ekman magnitudes, Courant, tendency vs advection), PINN pitfalls, literature |
| `notebooks/cnn_results.ipynb`, `cloud_effects.ipynb` | data-only runs; error vs cloud-spell length, gap size, month |
| `notebooks/mind_the_chl_gap_survey.md`, `high_res_datasets_survey.md` | upstream project survey; high-resolution datasets survey |
| `fetch_data.py`, `fetch_occci.py` | download IO.zarr; regional OC-CCI pulls |

Run names: `perf_*` data-only (train2), `pinn_lam*` λ sweep, `pinn_v1_*` priors, `pinn_long_*` 60 epochs, `nochl_*` no-CHL ablations, `nochl_today_*` no CHL at t, `unet_mtcg_*` upstream U-Net setup, `unet_d2w64_*`/`unet_d3w32_*` U-Net PINNs.

## 10. Conventions used in reporting

Observations first, interpretation separate and hedged; sample sizes with every aggregate; val only for model selection; no run compared across different λ definitions without the transport-only residual as a common yardstick; plots colour-blind safe (Okabe-Ito, viridis, PuOr); plot titles describe what is shown, never a conclusion.


## 11. Update 2026-10-04: training recipe v2/v3, Black Sea data, code layout

### 11.1 New trainer options (`chl_cnn/train_pinn.py`), all off by default
| flag | what it does |
|---|---|
| `--year-input` | constant channel (year − 2009)/7, for calendar-year drift (yearly box-mean anomalies 1998-2020: SST +0.09 K/decade, air T +0.13, SLA +0.046 m, MLD +0.7 m; ln CHL −0.03/decade with ±0.2 interannual swings) |
| `--donor-window W` | training fake-cloud donors drawn from days within ±W days of year in other years (94 % other-year at W = 15); val/test donors unchanged so scores stay comparable |
| `--no-chl-today`, `--feat-context K` | no CHL image at t but CHL context t±k kept; physical fields of t±1..K added as channels |
| `--spectral N --spectral-modes M --spectral-ch C` | FNO-style spectral path in the first N residual blocks of the dilated CNN: 1×1 down to C channels, learned complex weights on the lowest M×M Fourier modes (stored as real pairs for AMP), 1×1 up, summed with the local conv path. N = 4, M = C = 16 adds 1.06 M real params (0.61 → 1.67 M) and ~14 s/epoch |
| `--grad-weight G` | gradient-difference loss: L1 on neighbour differences of z-scored ln CHL along lat and lon, on pixel pairs with truth at both ends and at least one hidden end, weighted G relative to the pixel L1. Chosen because the diagnosed failure is blur (gradient ratio ≈ 0.55-0.6 of truth on hidden pixels); needs only L3 truth, unlike SSIM/spectral losses |
| `--ckpt-from E` | best-val tracking and `model.pt` checkpoint only after epoch E |
| `--unet-depth D` | U-Net pooling levels |
Metrics added to every run: `grad_ratio` and `grad_corr` on hidden pixels (val, per epoch in history.json and in metrics.json).

### 11.2 Results since section 5 (val hidden-pixel MAE, ln CHL, seed 0)
| run | recipe | epochs | val MAE | RMSE | grad ratio | grad corr |
|---|---|---|---|---|---|---|
| `pinn_v2_lam0` | full inputs + year channel + seasonal donors, λ 0 | 60 | **0.1365** | 0.231 | – | – |
| `pinn_v2_lam0p3` | same, λ 0.3 | 60 | 0.1370 | 0.233 | – | – |
| `pinn_v2_lam1p0` | same, λ 1 | 60 | 0.1378 | 0.235 | – | – |
| `pinn_v3_spec` | v2 recipe + 4 spectral blocks, λ 0 | 20 | 0.1502 | 0.249 | 0.549 | 0.570 |
| `pinn_v3_gdl` | v2 recipe + gradient loss G = 1, λ 0 | 20 | 0.1401 | 0.234 | 0.536 | 0.591 |
| `pinn_v3_spec_gdl` | both | 20 | **not run** (session died) | | | |
| `pinn_v3_ctrl` | v2 recipe, same 20-epoch schedule | 20 | **not run** (session died) | | | |
Reading so far: the year channel + seasonal donors gave the best model yet (0.1381 → 0.1365, i.e. the earlier "λ makes no difference" picture holds and the data recipe matters more). The spectral blocks look worse at 20 epochs (0.150) but have no same-schedule control yet; the gradient loss at G = 1 did not raise the gradient ratio (0.536 vs ~0.55-0.58 in earlier runs), so G needs to be larger or the term rethought before concluding. **Next concrete step: `./train_queue.sh v3` from tmux runs the two missing runs (≈ 35 min), then compare the four at equal schedule.**

### 11.3 Black Sea data (done)
`~/mind_the_chl_gap/data/cmems/blk_2km/<year>.zarr`, 1998-2024, 27 years, 26 GB; `fetch_cmems.open_arranged("blk", 2)` opens all years. 9,862 consecutive days, 395 × 550 grid at 0.02° (2 km blocks of the 1 km product), 93,594 ocean pixels. Variables: `chl_l3` (geometric block mean of valid 1 km CHL, mg m⁻³, linear), `chl_l3_n` (valid 1 km px per block, 0-4), `chl_l4` (gap-free product, same blocking), `qi_chl`, and the 2.5 km physics reanalysis surface layer interpolated to the grid: `thetao`, `so`, `uo`, `vo`, `mlotst`, `zos`, `tauuo`, `tauvo`; `ocean` mask. Facts: L3 observed on 23-49 % of ocean pixel-days by year (strong seasonal cycle, 0.17 in Dec to 0.62 in Jul-Aug), L4 complete from 2001 (92 % in 1998), physics covers 85.5 % of CHL ocean pixels (no Azov, Marmara, some coast). L4 − L3 on observed pixels: MAE 0.020; L4 under real clouds averages −0.057 ln CHL below the clear-sky monthly climatology. Day-to-day |Δ ln CHL| on pixels observed both days: median 0.11-0.16, mean 0.17-0.28. **Courant number of the reanalysis currents on the 2 km grid: median ≈ 6 cells/day, 98 % > 1, ~60 % > 5**, so the daily finite-difference transport residual used so far is not usable here; a semi-Lagrangian (back-trajectory, `grid_sample`) formulation is required. Cloudy pixels differ little from clear ones in SST/salinity/MLD/SSH here (|z| ≤ 0.04) and sit in stronger wind stress (+0.16). Latitude ascends in this store (IO.zarr descends): plot with `origin="lower"`.
EDA notebook `notebooks/blk_eda.ipynb` is written (sections: availability, seasonal climatology, box time series, L3 vs L4, day-to-day change + Courant, cloud vs physics, QI/pixel count, modelling notes) but **its last execution was cut off; re-run with `jupyter nbconvert --execute --inplace`** (uses every 10th day; needs ~8 GB RAM while a trainer holds 12 GB).
Fetch tooling: `fetch_cmems.py download|arrange` (one zarr per year, resumable, `--delete-raw`), `fetch_cmems_all.sh <blk|med> <y0> <y1> [coarsen]` for tmux; `fetch_occci.py` for OC-CCI 4 km / 1 km boxes over OPeNDAP (PML aggregations `CCI_ALL-v6.0-DAILY`, `CCI_ALL-v5.0-1km-DAILY`). Mediterranean not pulled (1 km L3/L4 ≈ 7 GB/yr each uncompressed; disk 42 GB free).

### 11.4 Code layout (since 2026-10-03)
`models/{cnn,unet,pinn/cnn,pinn/unet}/<run>/`; `chl_cnn/runs.py` (`run_dir`, `find_run`, `list_runs`) is the only place that knows the layout. `analyses/{pinn,clouds,trends}/` each with `figures/`. `train_queue.sh` is the resumable queue (skips runs with metrics.json or already running; filters by substring: `v2`, `v3`, `unet`, `season`, `year`); run it from tmux, not from a tool shell (tool-shell processes die with the session). Progress: `python -m chl_cnn.watch --cat pinn/cnn,pinn/unet --every 60 --out models/progress_pinn.png` or `notebooks/progress.ipynb` (ipywidgets; needs a live kernel). Notebooks `pinn_results.ipynb`, `cnn_results.ipynb`, `cloud_effects.ipynb` use `find_run`.

### 11.5 Open items carried forward
1. Finish v3 (`./train_queue.sh v3`), then decide on gradient-loss weight (try G = 3-10) and whether spectral blocks stay.
2. Seed noise (3 seeds × λ ∈ {0, 0.3}) is still unmeasured; every difference < 0.002 above is unproven.
3. Black Sea modelling: build a `store.py`-style memory-mapped store from `blk_2km`, year-interleaved split, same hidden-pixel protocol; transport term must be semi-Lagrangian. L4 is a reference row, never truth.
4. Items from `claudes_summary.md` §11 (noise floor, OSSE/MNAR test, clear-sky bias, benchmark of upstream U-Net preds and L4 on our hidden pixels) are not started.
