# Chlorophyll gap-filling: deliberation notes and candidate directions

Written 2026-10-03 from a deliberation chat. Companion to the hand-off brief
*"Filling cloud gaps in satellite chlorophyll with CNNs and physics-informed losses"*
(the "brief" below). Nothing here has been run; the brief holds all measured numbers.

Purpose: decide what to do next. Sections 2–3 critique the current setup, sections 4–9 list
candidate projects, section 10 lists related work, section 11 proposes two "spines" to choose between.

---

## 1. Where the project stands (from the brief)

- Best data-only dilated CNN: val hidden-pixel MAE 0.138 ln CHL (test 0.140, R² 0.92) vs climatology 0.358.
- Physics (transport) loss: val MAE spread across λ ∈ {0, 0.1, 0.3, 1}, three priors, 12 vs 60 epochs, two
  architectures is 0.001 on a single seed. No measurable effect on hidden-pixel error.
- The physics loss *does* change the field under real clouds by 0.05–0.06 ln CHL (≈40 % of MAE), by smoothing:
  gradients fall ~25 %, residual falls fourfold, but the predicted evolution follows the currents no better
  (corr with advective change stays ≈0.08).
- Information is the limit: no CHL image → 0.223; neighbour days but no image at t → 0.16; with image → 0.14.
  A 3× bigger U-Net matches the dilated CNN within 0.002.
- Resolved advection explains ~1 % of day-to-day ln CHL variance on observed pixels (r ≈ 0.11); Courant > 1 on
  about half of pixel-days at 0.25°/daily.
- Cloudy pixels sit in stronger, more westerly wind (+0.39, +0.29 standardised), lower salinity (−0.14) and
  lower climatological CHL (−0.12).
- Open items already listed in the brief: seed-to-seed spread; Arabian Sea spring residual bias; error binned
  by gap size / context-day hiding; score upstream U-Net preds and GlobColour L4 on our protocol; FiLM check.

---

## 2. Critique of the current evaluation and loss

These are the points everything else hangs on.

### 2.1 MAE rewards smoothing
- A model that predicts the conditional mean is rewarded for blurring. The brief's own diagnostics show it:
  gradient amplitude ratio pred/truth ≈ 0.58; day-to-day change under persistent clouds 0.094 vs 0.143 in truth;
  small-scale spectral power drops with λ.
- For most downstream uses (front detection for fisheries, bloom onset timing, HAB alerts, eddy tracking,
  extremes) the smoothing is the main failure, and MAE is blind to it.
- Consequence: a gap filler with MAE 0.14 that halves gradients can be worse for users than one with MAE 0.15
  that keeps them.
- Fixes: add sharpness/spectral metrics, gradient-field correlation and amplitude ratio, front-detection
  metrics (e.g. Belkin–O'Reilly fronts), phenology metrics (§4.3), object-level checks (§4.4); or move to
  probabilistic/generative filling scored with CRPS (§5.2).

### 2.2 Clouds are missing-not-at-random (MNAR) and the protocol cannot see it
- The fake-cloud protocol hides *clear-sky* pixels behind borrowed cloud masks. Scores are therefore
  clear-sky-condition scores.
- The data show clouds co-occur with stronger wind (more mixing) and lower salinity. CHL under *real* clouds is
  plausibly systematically different from CHL on clear days.
- Consequences:
  1. Any clear-sky climatology (incl. our climatology channel and the "residual" target) is biased.
  2. "Physics does not change MAE" and "physics changes the field under real clouds by 0.05" can both be true
     without telling us which field is closer to reality. The protocol has no truth exactly where physical
     inputs could earn their keep.
- Fixes: truth under real clouds (BGC-Argo matchups, §4.1), OSSE on model output (§4.2), weather-based
  clear-sky-bias correction (§4.5).

### 2.3 The transport equation is a weak constraint at 0.25°/daily — and this is now supported by the literature
- Our 1 % result matches a Biogeosciences 2022 study (§10) that found Lagrangian ≈ Eulerian tendencies in
  0.25° GlobColour, attributing day-to-day change to pixel-scale processes and sources/sinks, with unresolved
  advection dominating decorrelation.
- A fine-scale study with HF-radar currents (§10) finds advection comparable to the biological term.
  → Resolution decides whether transport matters. This is the main reason to move to the Black Sea 0.025° data.
- The Ekman prior used wind-stress *curl* (open-ocean pumping) and misses *coastal* upwelling (alongshore wind
  stress at a boundary). GLORYS (38 % NaN) and OSCAR (39 % NaN) have their gaps exactly on the coasts and
  shelves where upwelling/downwelling live. Inputs are weakest where the physics is strongest.
- SST and SLA already integrate upwelling history (cold anomaly = recent upwelling; positive SLA = deep
  thermocline/downwelling), which is probably why the static no-CHL model reaches 0.223. The *transport*
  term is the weak part, not "physics" in general.

### 2.4 What a day-to-day CHL change is made of (why physics cannot explain most of it)
1. Retrieval noise: daily L3 merges several sensors and overpass times; cloud-edge contamination. Possibly a
   large share of the 0.143 ln change. Physics cannot explain noise → sets a floor (measure it, §4.6).
2. Horizontal advection: ~1 % of variance; OSCAR effective resolution is closer to 100 km than 25 km.
3. Vertical processes: mixed-layer deepening dilutes surface CHL; entrainment supplies nutrients; the satellite
   sees only the first optical depth.
4. Biology with lags: upwelled nutrients become a bloom days to weeks later; grazing, sinking.
5. Sub-grid structure: filaments narrower than a pixel moving through the pixel average.

### 2.5 Is daily temporal resolution the limit? Mostly no.
- Courant is a numerical problem, fixable with semi-Lagrangian advection or sub-stepping; it does not need
  hourly inputs.
- Daily-mean currents are adequate for 0.25° features (tides, inertial oscillations largely average out).
- Spatial resolution (1–4 km) is what lets fronts/filaments be advected coherently within a day.
- Counter-intuitive point: **longer** horizons (3–10 days) may reveal more physics than 1 day, because the real
  signal grows above noise, persistence loses skill, and lagged upwelling responses appear.
- At 1 km the question returns in a different form: with ~0.3 m s⁻¹ currents a parcel moves ~26 km/day, so
  trajectory accuracy over a day is limited by *current* accuracy, not by CHL sampling frequency.

### 2.6 Smaller methodological notes
- Seed-to-seed noise still unmeasured; no λ comparison is meaningful until it is (3 seeds × λ ∈ {0, 0.3}).
- Hidden-pixel MAE may hide a trade-off by gap size and by how many context days are also hidden; the physics
  term can only matter on large persistent gaps.
- 43 unflagged lake pixels; `so` ends 2020; pre-1997 data never usable.

---

## 3. What "useful" means for this problem

Who uses gap-filled CHL: fisheries (upwelling, fronts, tuna/sardine habitat), HAB/bloom monitoring, primary
production and carbon budgets, biogeochemical data assimilation, bloom phenology, eddy/front studies,
resilience/trend analyses. Useful contributions therefore look like:

1. Evaluation tied to use (phenology, fronts, sharpness, extremes), not only MAE.
2. Calibrated per-pixel uncertainty (DINCAE-style). L4 gives only coarse flags.
3. Truth or at least a bias estimate under *real* clouds.
4. An honest public benchmark: nobody has published one for this dataset (score upstream U-Net preds and
   GlobColour L4 on our protocol — cheap, no retraining).

---

## 4. Evaluation and bias projects

### 4.1 Truth under real clouds from BGC-Argo
- BGC-Argo floats carry CHL fluorescence; some have operated in the Arabian Sea; also in the Black Sea.
- Matchups will be sparse; fluorescence calibration, quenching, and depth vs first optical depth are issues.
  Check float coverage before committing.
- Only route to a direct answer to §2.2.

### 4.2 OSSE: observing-system simulation experiment on model output
- Take a biogeochemical reanalysis (Copernicus Black Sea BGC reanalysis; GLORYS-coupled products elsewhere),
  overlay *real* cloud masks from the same dates, train/score on model output where truth under clouds exists.
- Tests MNAR directly: is error larger under real cloud positions than under donor masks?
- Caveat: a model reproduces the cloud–wind–CHL coupling only to the extent its forcing and biology do.
  Treat as a plausibility test, not ground truth.
- Also useful as the "ceiling" case for the transport-budget diagnostic (§7.1): currents and tracer are
  mutually consistent in a model.

### 4.3 Phenology as a downstream metric
- Bloom onset/peak/duration are known to be sensitive to gaps (Cole et al. 2012, "Mind the gap", ICES JMS —
  the upstream project's name reads as a nod to it; verify citation).
- Score gap fillers on recovery of phenology metrics on held-out years. Cheap addition that converts the
  MAE table into something ecologists care about.

### 4.4 Object-level evaluation with the mesoscale eddy atlas
- AVISO/Chelton mesoscale eddy trajectory atlas tracks eddies from altimetry; Chelton et al. 2011 (Science)
  showed eddies carry CHL anomalies.
- Ask: does gap-filled CHL inside an eddy stay coherent as the eddy moves? Does the physics loss help there?
- Independent coherent structures with known motion from altimetry → sidesteps OSCAR accuracy.
- Pixel MAE cannot see this.

### 4.5 Clear-sky-bias correction with the weather we do have
- ERA5 gives total cloud cover and wind on every pixel-day: we know the weather on cloudy days even without CHL.
- Fit CHL | weather on clear days; reweight to the full weather distribution (inverse-probability weighting)
  to estimate the bias of clear-sky climatologies. No in-situ data needed.
- Clear-sky bias is well known for SST climatologies and suspected for CHL; I have not seen it done this way for
  ocean colour — verify.

### 4.6 Noise floor
- Preferred: GlobColour per-sensor L3 products — same-day overlaps between sensors give retrieval noise directly.
  Check availability in the bucket / Copernicus Marine store.
- Fallback: structure function mean|Δ ln CHL| at lags 1, 2, 3… on pixels observed at both ends; extrapolate to
  lag 0 (conflates noise with sub-pixel/sub-daily variability → an upper bound on noise).
- If the 1-day change ≈ noise floor, no model can explain day-to-day change and forecast targets should move to
  longer lags.

### 4.7 Public benchmark
- Score `U-Net_with_CHL_pred.zarr`, `U-Net_without_CHL_pred.zarr` and `CHL_cmes-gapfree` on our hidden pixels.
- Add climatology, persistence, Lagrangian persistence, DINEOF if feasible.

---

## 5. Gap-filling model ideas

### 5.1 FiLM conditioning (cheap one-run check)
- Per-channel scale/shift of CNN feature maps from a small MLP of slowly varying context (DOY climatology,
  monthly-mean wind stress, MLD, lat/lon). Hypernetwork in the weak sense.
- Expectation at 0.25°: little change (information, not architecture, is the limit). Worth one run because the
  upstream claim that environmental variables do not help has not been tested with a conditioning architecture.

### 5.2 Generative / probabilistic gap filling
- Conditional diffusion or flow model on the existing 25-channel input stack; sample plausible fields.
- Score with CRPS, spread–skill, spectral and gradient metrics. Keeps sharpness; gives calibrated uncertainty.
- Directly addresses §2.1 and §3.2. One-month-feasible as a small conditional diffusion model.

### 5.3 Fill reflectance / multi-variable state, not CHL alone
- CHL is derived; filling Rrs bands (or CHL jointly with backscatter and SST) lets the model use bio-optical
  consistency. Makes the Black Sea coccolithophore/CDOM retrieval problems tractable rather than a trap.

### 5.4 Hypernetworks — when they are worth it
A hypernet adds value over plain conditioning only when (a) generalisation to *unseen* conditions is tested,
or (b) the learned condition embedding is itself an output of interest.
- **Physics/weather-conditioned hypernet**: same information as the input channels → not worth it beyond FiLM.
- **Location-conditioned hypernet** (bigger canvas): train on many regions at 4 km (global GlobColour), generate
  a region-specific small net per region, evaluate on *held-out regions*. Useful for data-poor regions.
  Complex-systems angle: do learned region embeddings rediscover Longhurst biogeochemical provinces?
  (interpretability testable against an independent classification; plays to the RNP/hypernetwork background).
- **Resolution-conditioned hypernet / downscaling**: weak unless tied to super-resolution with real fine-scale
  truth (1 km Med/Black Sea). Keep last; domain shift to the Indian Ocean is hard to validate.
- DeepONet: for gap filling on a fixed grid it is a CNN with a worse inductive bias (low-rank trunk); only pays
  off for continuous queries or autodiff physics terms. Not recommended here.

---

## 6. Forecast models: "where does the CHL go?"

### 6.1 Framing
- Current runs interpolate (use t+1). A *forecast* uses only t−1, t−2 → t at lead times 1, 3, 5, 10 days.
- Baselines: Eulerian persistence; Lagrangian persistence (yesterday's field carried along the currents,
  semi-Lagrangian); climatology; the no-CHL static physics model.
- Expectation: day-to-day change noise-dominated; physics adds real skill at 5–10 day horizons and in monsoon
  upwelling zones. Test with §4.6 first.

### 6.2 Data-driven baseline: ConvLSTM, not per-pixel LSTM
- A per-pixel LSTM cannot represent anything moving between pixels, so it cannot learn transport.

### 6.3 UDE / grey-box hybrid
- Fixed mechanistic semi-Lagrangian transport (observed/reanalysis currents) + learned source term S(local
  physics with lags: wind, ΔMLD, light, upwelling index, SLA).
- Implementation: PyTorch — backward trajectory step ≈ one `grid_sample` call (fits existing pipeline);
  Julia — SciML (DiffEqFlux / Lux) is the natural UDE home.
- Initial state is cloudy → the gap-filler supplies the initial condition; the whole thing becomes learned data
  assimilation (4DVarNet territory). Loss only on observed future pixels.
- Payoff: direct split of change into transport vs source; inspect the learned S against known mechanisms;
  optionally SINDy on S (as the Baltic UDE paper did).
- Add processes one at a time and measure variance explained by region/season:
  - mixed-layer dilution: surface CHL scaled by h_old/h_new on deepening
  - coastal upwelling index from alongshore wind stress, several lags
- Novelty: I found no published UDE with mechanistic transport + learned biological source on gridded
  satellite CHL (see §10 for the nearest neighbours). Do a proper literature pass before claiming this.

---

## 7. Angles that need high spatial resolution (Black Sea 0.025° data arriving)

### 7.1 Budget-by-scale diagnostic (do first)
- Decompose the CHL budget (Eulerian change, advection, Lagrangian remainder) at 0.025°, then coarsen to 4, 9,
  25 km and repeat → curve of fraction explained by advection vs grid size.
- **Critique**: this measures the *currents* as much as the ocean. At 1–3 km the reanalysis currents are weakest,
  so the curve is a lower bound confounded with product quality. Pair it with (a) the same decomposition on
  model output (ceiling; §4.2) and/or (b) flow inferred from the imagery (§7.3). A flat curve alone is not
  informative.

### 7.2 Re-run the PINN gap-filling test at 1–3 km
- The brief predicts the physics loss helps where fronts are advected coherently. Helps at 1 km and not at
  0.25° → clean result with a clear explanation; strong portfolio piece.

### 7.3 CHL as a tracer to estimate currents
- Infer surface flow from motion between consecutive CHL images (de Bézenac et al. 2018 learned-flow approach,
  or classic maximum cross-correlation). Compare with reanalysis currents and altimetry.
- Useful where altimetry is poor (small scales, coasts); inferred flow can drive the UDE transport step.

### 7.4 UDE forecast (§6.3) at the scale where §7.1 shows advection carries signal.

### 7.5 Learned downscaling (last; see §5.4).

### 7.6 Black Sea specifics
- Strong permanent halocline → surface dynamics plausibly closer to 2-D; transport may explain more than in the
  monsoon-mixed Arabian Sea (hypothesis to test).
- Features to track: Rim Current, coastal anticyclonic eddies, Danube plume.
- Documented regime shift (Daskalov et al. 2007, PNAS: overfishing, eutrophication, *Mnemiopsis*); satellite
  record starts 1997, after the main shift — relevant to §8 but limited.
- Retrieval traps: CDOM-rich (Case-2) waters; late-spring coccolithophore blooms can bias standard CHL
  algorithms. Check product quality flags before trusting May–June tendencies.
- Matching currents: Copernicus Black Sea physics reanalysis ≈ 3 km (confirm grid). Sentinel-3 OLCI 300 m exists
  if needed.

---

## 8. Tipping points / early-warning signals (EWS)

### 8.1 Why classic critical slowing down on basin CHL is hard here
- Short record (~23 yr; Henson et al. 2010: decades needed to detect CHL climate trends).
- Sampling changes: coverage 13 % → 53 % as sensors were added → noise and sampling change over time.
- Strong seasonality and forcing (monsoon) swamp slowing-down signals; daily lag-1 AC is dominated by
  persistence.
- No clean bistable system in total CHL in this region. Best-known regime shift (Arabian Sea diatoms →
  *Noctiluca*, Gomes et al. 2014) is a composition change; needs phytoplankton-type products, not CHL.

### 8.2 Better angles
**(a) Gap filling biases resilience indicators — the strongest idea.**
- Filled values inherit persistence from context days and climatology → inflated lag-1 AC, deflated variance
  (the two standard indicators). Brief's own numbers: day-to-day change 0.094 vs 0.143; gradient ratio 0.58.
- Fill fraction *declines* over the record (13 % → 53 % coverage) → spurious trend in the indicators that can
  mimic or mask critical slowing down; a climatology-leaning fill shrinks anomalies, so declining fill alone
  produces upward variance trends and downward AC trends.
- **Critique/refinement**: almost nobody computes EWS from daily L3; they use L4, 8-day or monthly composites.
  Frame the project around how changing gap fraction propagates into *composites and trends*, then show the
  daily-pixel mechanism underneath. More useful and more publishable.
- Implementation: `preds.npz` holds full val/test fields; val/test years are complete, so indicators can be
  computed within-year on truth vs filled for the same fake-cloud pixels. First pass ≈ 1–2 days.
- Audience: anyone doing satellite resilience work, ocean or land. Also bears on whether reported Indian
  Ocean CHL trends are sampling-affected.

**(b) Recovery rate after perturbations (cyclones).**
- IBTrACS storm tracks → CHL bloom response → recovery rate as resilience; test whether lag-1 AC predicts it
  (Smith, Traxl & Boers 2022, Nature Climate Change, did this for vegetation). Cyclones are also where gap
  filling is hardest → ties to (a).

**(c) Spatial EWS** (spatial variance, correlation, skewness; space-for-time) — suits a short record with a
large domain. Needs resolution → Black Sea.

---

## 9. Design notes and gotchas collected along the way
- Donor-cloud persistence across the context window is essential (without it the model copies neighbours
  and scores an unrealistic 0.106).
- Loss on visible pixels (0.1×) helps slightly; L1 ≈ Huber < MSE at the 0.003 level.
- Report conventions (brief §10): observations first, interpretation hedged, sample sizes with every aggregate,
  val only for selection, Okabe-Ito/viridis/PuOr, descriptive plot titles.
- Claude Code's prior assessments (hypernet/DeepONet notes) agreed: at 0.25°/daily a physics/climatology
  hypernet is premature; FiLM is the one cheap check; HyperDeepONet-style transport operators become
  reasonable only on km-scale data.

---

## 10. Related work (as found in this deliberation; verify before citing)

### UDE / neural-ODE / hybrid forecasting
| Work | What it is | Relevance |
|---|---|---|
| *Machine learning to identify environmental drivers of phytoplankton blooms in the Southern Baltic Sea*, Sci. Rep. 2025 — https://www.nature.com/articles/s41598-025-85605-y | ANN inside a UDE on monitoring-station time series (CHL, DIN, DIP, totN, totP; drivers T, S, Kd), then SINDy | Closest published UDE for CHL; **no spatial/transport term** |
| *Multi-Scale Spatiotemporal Graph ODE Networks for Marine Chlorophyll-a Prediction*, Remote Sensing 18:2828 — https://doi.org/10.3390/rs18162828 | Neural ODE in learned latent space on a graph | Lists advection–diffusion + growth inside the ODE as **future work** (per search excerpt; confirm in paper) |
| *SSTODE*, arXiv 2511.05629 — https://arxiv.org/pdf/2511.05629 | Advection–diffusion inside a neural ODE with learned velocity evolution, for SST | Closest technical template for the UDE |
| *NeuroDDAF*, arXiv 2604.01175 — https://arxiv.org/pdf/2604.01175 (describes **AirPhyNet**) | Diffusion–advection in neural ODE for air quality | Analogue in another domain; AirPhyNet cited second-hand |
| de Bézenac, Pajot & Gallinari, *Deep Learning for Physical Processes: Incorporating Prior Scientific Knowledge*, ICLR 2018 (from memory) | Learned flow field warping SST via advection–diffusion | Ancestor of §7.3 |
| Ramadhan et al. (CliMA / Oceananigans), neural differential equations for ocean mixing parameterisations (from memory; title to verify) | UDE for vertical mixing in Julia | Julia/SciML precedent for ocean UDEs |

### Transport vs biology in satellite CHL
| Work | Finding | Relevance |
|---|---|---|
| *Lagrangian and Eulerian time and length scales of mesoscale ocean chlorophyll from Bio-Argo floats and satellites*, Biogeosciences 19, 5927 (2022) — https://bg.copernicus.org/articles/19/5927/2022/ | At 0.25° GlobColour, Lagrangian ≈ Eulerian tendencies → pixel-scale processes/sources dominate; unresolved advection drives decorrelation | Explains our 1 % result; motivates high resolution |
| *Disentangling Advection and Lagrangian Evolution of Surface Chlorophyll in a Nearshore Submarine Canyon…* — https://par.nsf.gov/biblio/10611343 | With HF-radar currents, Lagrangian (biological) term comparable to advection | Fine-scale counterpart |
| *Dominant timescales of variability in global satellite chlorophyll and SST (MOSS)*, RSE 2022 — https://www.sciencedirect.com/science/article/pii/S0034425722005107 | Shortest CHL timescales in subtropical gyres and upwelling regions; CHL timescales decoupled from SST except in coastal upwelling | Supports lag/horizon choices |

### Gap filling, physics-aware
- 4DVarNet (Fablet et al.), DINCAE (Barth et al.), diffusion-based gap filling, DINEOF — gains reported on
  300 m–4 km products (brief §6.5).
- *Neural Networks Technique for Filling Gaps in Satellite Measurements: Application to Ocean Color*,
  PMC4706868 — NN gap filling using physical observations (VIIRS).
- Upstream project: GeoSMART / NOAA NWFSC "mind the chl gap" (2024–26); AGU 2024 abstract; no benchmark.

### Phenology, EWS, regime shifts, eddies (from memory; verify)
- Cole, Henson, Martin & Yool 2012, *Mind the gap: the impact of missing data on the calculation of
  phytoplankton phenology metrics*, ICES J. Mar. Sci.
- Henson et al. 2010, time needed to detect climate-driven CHL trends (~decades).
- Smith, Traxl & Boers 2022, Nature Climate Change — empirical validation of lag-1 AC as vegetation
  resilience via recovery rates.
- Dakos et al. 2012 (EWS methods); Boettiger & Hastings 2012 (caveats).
- Gomes et al. 2014, Nature Communications — Arabian Sea *Noctiluca* shift. Roxy et al. 2016, GRL — Indian
  Ocean CHL decline (sampling caveat applies).
- Daskalov et al. 2007, PNAS — Black Sea regime shift.
- Chelton et al. 2011, Science — eddies carry CHL anomalies; AVISO/Chelton eddy trajectory atlas.
- Hypernet/DeepONet references from Claude Code's notes: HyperPINN (2021), HyperDeepONet (ICLR 2023),
  physics-informed DeepONets (Sci. Adv. 2021).

### Other ML forecasting of CHL (for the ConvLSTM baseline context)
- GNN-Transformer CHL prediction in semi-enclosed gulfs, Sci. Rep. 2026 — https://www.nature.com/articles/s41598-026-42388-0
- XGBoost/LSTM monthly CHL forecasting, northern Indian Ocean — https://www.sciencedirect.com/science/article/pii/S1574954125004911
- BiLSTM for cyclone-induced CHL response, WNP — https://www.sciencedirect.com/science/article/abs/pii/S1463500324000325
- 4DMED-bionet / MuSt3Net, physics-informed 3-D CHL in the Mediterranean — https://www.sciencedirect.com/science/article/pii/S1364815225003445

---

## 11. Two candidate spines (pick one, finish it, then open the other)

### Spine A — "Measure what the metric can't see" (finishable in ~1 month, reuses pipeline)
1. Seed noise (3 seeds × λ ∈ {0, 0.3}).
2. Noise floor (§4.6).
3. OSSE on BGC reanalysis with real cloud masks (§4.2) → MNAR test.
4. Clear-sky-bias correction via ERA5 weather reweighting (§4.5).
5. Eddy-atlas object-level check (§4.4) as the qualitative showcase.
6. Indicator/trend bias from changing gap fraction (§8.2a), framed on composites.
7. Public benchmark table (§4.7). Optional: phenology metrics (§4.3), generative filler (§5.2).
One-sentence story: *standard gap-filling metrics cannot see where physics matters; here is how to measure
what they miss, and what it does to downstream resilience and trend analyses.*

### Spine B — "Where does the CHL go?" (richer research programme; continues after the subscription)
1. Budget-by-scale diagnostic on Black Sea 0.025° data, with the model-output ceiling and/or imagery-inferred
   flow so currents don't masquerade as physics (§7.1, §7.3).
2. Forecast setup with ConvLSTM, Eulerian and Lagrangian persistence baselines (§6.1–6.2).
3. UDE with semi-Lagrangian transport + learned source at the scale where advection carries signal (§6.3).
4. PINN gap-filling repeat at 1–3 km (§7.2).
5. Optional: location-conditioned hypernet with Longhurst comparison (§5.4).

### Cross-cutting first step for either spine
Measure seed-to-seed spread and the noise floor; both are cheap and both decide what a "real" difference is.