# Ideas: where the remaining signal is, and models that could reach it (2026-10-09)

Written for: the mind_the_chl_gap team. Proposals, not results. Each item says what it would test, what it needs, and a cost class
(S = hours with existing code, M = a day with a new trainer option or dataset, L = a week or a new data pull). Numbers quoted are from
analyses/FINDINGS.md and analyses/spatial_node/NOTES.md.

## 1. What the Indian Ocean results say about where the signal is

| evidence | number | implication for new models |
|---|---|---|
| temporal context is the biggest lever for the spatial model | 0.208 -> 0.170 (t+-2) -> 0.166 (t+-4) | persistence over 1-4 days carries most of the recoverable information; models should be built around a short memory, not a long one |
| learned context weights fall geometrically | w(+-2)/w(+-1) ~ 0.25, w(+-3)/w(+-1) ~ 0.04 | an AR(1)-like decay with e-folding ~1 day at 0.25 deg; a state-space model with a 1-2 day memory is the natural form |
| Lagrangian advection of neighbour days adds nothing | +0.000 | at 0.25 deg the resolved currents do not move the CHL field coherently in a day; transport terms need finer currents (Black Sea 2.5 km) |
| CNN beats the spatial model by 0.02-0.03 at every distance, including distance 1 | 0.085 vs 0.105 at 1 cell | the gap is a local, nonlinear combination of covariates and context, not information propagation; a linear-Gaussian model cannot close it, a learned local source term could |
| bathymetry, not SST, steers the MLP tensor; SST alone still gains 0.018 | ablation: -0.0002 (SST) vs -0.020 (depth) | along-isobath structure dominates at the coast and shelf; SST fronts matter offshore; the Black Sea has a sharp shelf break and strong plumes, so both should be stronger there |
| physics loss on the CNN changes nothing at 0.25 deg | <= 0.001 | physics must enter as structure (operators, state) rather than as a soft penalty on a field the network already fits |

## 2. Model families worth building

### 2.1 State-space nODE (latent dynamics + the elliptic solver as the observation step)  [M]
The steady-state model treats each day as a separate elliptic problem with neighbour days as soft data. The state-space version keeps a latent
field x_t and runs a daily recursion: x_t = F_theta(x_{t-1}, covariates_t) (prediction), then x_t <- argmin E(C | x_t as prior, today's visible
pixels, D(x)) (the existing solver as the update). F_theta can be as small as "persistence + learned decay toward climatology + semi-Lagrangian
advection": x_t = clim_t + a(x) * [x_{t-1} - clim_{t-1}](X_back) with a(x) in (0,1) learned from covariates. This is the Kalman-filter shape the
context weights already imply (w(+-k) geometric), with one difference: it is causal, so it gives a forecast at the same time as a fill. Train
through 5-7 day windows with the implicit gradient at each update; score the fill (hidden pixels of day t) and the 1-3 day forecast (observed
pixels of t+1..t+3 that were not in the window). The CNN cannot produce a forecast without retraining, so the comparison is new ground.
What it needs: a windowed data loader (train_pinn.py already builds t+-K indices), the existing solver, ~1 day of work.

### 2.2 UDE with a mechanistic transport term and a learned biological source  [M-L, Black Sea first]
dC/dt = -u.grad C (semi-Lagrangian, reanalysis currents) + div(K grad C) + g_theta(C, covariates, lags). Fit g_theta as a small network on
local state (C, SST, MLD, its change, wind stress, salinity, day of year, distance to river mouths, lagged discharge). Only the source is learned,
so it can be inspected and, as a second step, fitted symbolically (SINDy on g_theta samples). At 0.25 deg the transport term is noise (see
Lagrangian null); on the 2 km Black Sea grid the reanalysis currents are resolved enough that the semi-Lagrangian step is at least a
plausible operator (Courant ~6 means the back-trajectory needs sub-stepping or the 2.5 km hourly/daily fields; daily is what we have).
Score: hidden pixels at day t when integrating from t-1 and t-2 (fill) and observed pixels at t+1..t+3 (forecast). Separately report the
variance explained by transport alone vs transport + source, by region (NW shelf, open sea, Anatolian coast) and season.

### 2.3 Seasonality as structure, not a channel  [S-M]
The climatology prior is a per-pixel mean by 8-day bin. Three cheap upgrades:
- Condition D, beta and w_k on day-of-year (phase features sin/cos into the D-net; beta and w_k as small functions of phase). Tests whether the
  persistence time scale and the anisotropy change with season (winter mixing vs summer stratification). One run per store.  [S]
- Replace the climatology prior by a harmonic regression per pixel (annual + semi-annual + trend) fitted on training days; gives a smoother prior,
  a trend term (eutrophication, see 4.2) and an anomaly definition with no bin edges.  [S]
- Phenology-aware scoring: bloom onset/peak timing per pixel from the filled fields vs the observed ones on clear years; a model that fills the
  bloom too smoothly fails here even with a good MAE (the MAE-rewards-smoothing problem in docs/archive/claudes_summary.md section 2.1).  [M]

### 2.4 Probabilistic fills from the GMRF  [S]
The energy already defines a Gaussian field. Exact samples cost one CG solve each (perturb the data terms and the corner gradients with
white noise scaled by their weights, then solve). 30-50 samples per day give a variance map and an ensemble for CRPS; one temperature fitted on
val makes the variance globally calibrated, then calibration by distance bin and region is a real test. The Black Sea with its larger gaps is
where this matters most. Half a day; see the GP discussion in the session log.

### 2.5 Learned local source on top of the elliptic solver, done properly  [S]
The previous source term entered as a shift of the climatology prior and was almost switched off wherever neighbour days were visible. The honest
version: normalise trace(D) (removes the D/beta/w scale freedom), then add a bounded per-pixel source g_theta(x) directly to the right-hand side
with its own weight, and let g_theta see the same context channels as the conv D-net. This is the smallest step from "linear-Gaussian" toward
the CNN's local nonlinearity, and the distance-1 gap (0.02) is the number it has to move.

### 2.6 Hybrid ordering: CNN first, elliptic solver as a structured post-processor  [S]
The fill-as-input-channel hybrid gave nothing. The reverse ordering has not been tried: CNN output as the prior mean (replacing the climatology)
with a learned beta, observed pixels pinned, anisotropic smoothing along the learned D. It can only sharpen the CNN along structure and repair
inconsistencies with the pinned observations. Cheap: one eval-only run with the CNN preds as prior, then one training run of beta and D.

## 3. Black Sea specifics

### 3.1 Rivers and plumes
Freshwater input is the dominant nutrient and turbidity driver of the NW shelf (Danube ~ 200 km3/yr, then Dnieper, Dniester, Southern Bug;
the Don via the Azov Sea through the Kerch Strait; smaller Anatolian rivers such as the Sakarya and Kizilirmak; the Rioni in the east).
- The salinity field `so` in the store is the plume tracer and is not used by any model yet. Add it (and its gradient) to the D-net and conv
  inputs; expect anisotropy along the plume front and a lower salinity where CHL is high on the shelf.  [S]
- Distance-to-mouth features for the main mouths (Danube delta ~45.2N 29.7E, Dniester liman ~46.1N 30.4E, Dnieper-Bug ~46.6N 31.6E,
  Kerch Strait ~45.3N 36.6E, Rioni ~42.2N 41.6E, Sakarya ~41.1N 30.6E, Kizilirmak ~41.7N 36.0E, Bosporus ~41.2N 29.1E). Static channel, one
  line of code each.  [S]
- Discharge time series as lagged covariates, and as the forcing of the UDE source term in 2.2 (g = f(discharge lagged 5-30 days) near a mouth).
  Sources: GloFAS-ERA5 river discharge reanalysis (daily, 0.05 deg, open through the Copernicus Climate Data Store), GRDC stations
  (Ceatal Izmail for the Danube; needs a data request), or the Danube Commission monthly series. GloFAS is the practical one.  [M]
- Check first, before any model: does CHL on the NW shelf lag Danube discharge, and by how much (cross-correlation of shelf-box mean CHL anomaly
  with discharge, per season)? If the lag correlation is weak, the discharge covariate will not help a daily gap filler either.  [S once data is in]

### 3.2 Eutrophication and trends
The NW shelf went through eutrophication (1970s-80s), partial recovery (1990s-2000s) and recent re-intensification reports. Within 2014-2024 the
trend is small but the year channel (used in the IO CNN) should be tested on the Black Sea; a harmonic-plus-trend prior (2.3) makes the trend
explicit and inspectable. Report trend per region from the observed pixels first (no model), then check whether the models' residuals carry a
trend.  [S]

### 3.3 Ports, shipping and optical artefacts
Ports (Constanta, Odesa, Novorossiysk, Samsun, Istanbul approach) are point sources of turbidity and nutrients but small next to rivers; on
2 km pixels their footprint is a few pixels. Two things matter more than the ports themselves:
- Case-2 water: CHL retrievals near river mouths and ports are biased by sediment and CDOM. The store has `qi_chl` (quality index) and `chl_l3_n`
  (valid 1 km pixels per block); use them as a per-pixel observation weight (a soft data term with weight from qi) instead of pinning every observed
  pixel equally. This is the first real use of the soft alpha term.  [S]
- Coccolithophore blooms (June, open sea) raise reflectance and bias CHL high; the bloom is visible as a bright patch with low qi. A season-and-region
  flag, or the qi weight, keeps the model from learning the artefact as structure.  [S]

### 3.4 Physical covariates that are resolved here but not at 0.25 deg
- Mixed-layer depth change (dilution/entrainment): CHL scaled by h_old/h_new on deepening is a mechanistic source term with no parameters; test as
  a fixed source in the UDE and as a feature.  [S]
- Wind-stress curl and alongshore stress on the Anatolian coast (upwelling index) with 3-20 day lags, as in priors.py for the IO.  [S]
- Rim current position (from SSH `sla`): the cyclonic boundary current separates shelf and open-sea regimes; an SSH-gradient feature in the D-net
  would let D align with the rim current.  [S]

## 4. Evaluation ideas that change what "better" means
- Region-stratified scores (NW shelf, western/eastern gyres, Anatolian coast, Azov approach), because the models will disagree most on the shelf.
- Gap-size and distance-binned scores are in place; add season bins.
- Sharpness metrics (gradient ratio/correlation) are in place; add a front-position error (distance between detected fronts in fill vs truth).
- Forecast skill at 1-3 days for the state-space and UDE models (no CNN counterpart).
- CRPS and reliability by distance bin for the probabilistic fills.

## 5. Data worth pulling
| dataset | why | cost |
|---|---|---|
| GEBCO or EMODnet bathymetry for the Black Sea box | depth was the strongest single covariate on the IO; the current proxy is distance to coast | S (one file) |
| GloFAS-ERA5 daily discharge at the main mouths | river forcing for 3.1 and the UDE source | M (CDS request, 1 file per mouth) |
| CMEMS Black Sea biogeochemical reanalysis (nutrients, model CHL, 3 km, daily) | a model-based prior that already contains river and mixing effects; as a prior mean it competes with the climatology; as covariates it is the closest thing to a mechanistic source | M (large pull) |
| Sentinel-3 OLCI 300 m CHL for a few clear months | tests whether the anisotropy learned at 2 km holds at 300 m (fronts, plume edges) | L |

## 6. Suggested order
1. Cheap structure tests on the Black Sea once the ladder is done: salinity and distance-to-mouth features, qi-weighted observations, phase-conditioned
   D/beta, GEBCO depth (all S).
2. Probabilistic fills with CRPS (S), because it is the one deliverable the CNN cannot match.
3. State-space nODE with forecast scoring (M).
4. UDE source term with discharge forcing on the NW shelf (M-L), after the lag-correlation check.
