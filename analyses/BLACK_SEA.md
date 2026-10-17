# Black Sea: results so far (2026-10-17)

A starting point for deciding what to try next. Numbers come from [FINDINGS.md](FINDINGS.md) and
[models/RUNS.md](../models/RUNS.md). The error breakdown is from `python -m analyses.blk4.error_breakdown`.

**Set-up.** Copernicus 1 km L3 chlorophyll and the regional physics reanalysis cover 2014–2024, 3,997 days. Two grids:

- **2 km:** 395 × 550 pixels, 93,594 of them ocean.
- **4 km:** a 2 × 2 block average of the 2 km grid, 197 × 275 pixels, 23,606 of them ocean.

**Split.** Val years are 2017 and 2021, test years 2019 and 2023, and the rest train (2,542 days). Scoring uses every
fourth val and test day: 182 val days and 183 test days.

**Score.** MAE in ln CHL on observed pixels hidden under a real cloud mask taken from another day (the "donor" day).
**2 km and 4 km scores are not comparable**: the target and the hidden pixels differ.

![summary](blk4/figures/blk_summary.png)

## 1. Best models

| grid | model | val MAE | test MAE | DOY climatology, val / test | n hidden px, val / test |
|---|---|---|---|---|---|
| 4 km | 3D stem, λ 0.3, 150 epochs (50 + two warm restarts) | **0.1498** | **0.1430** | 0.3583 / 0.3112 | 1.06 M / 1.10 M |
| 4 km | 3D stem, λ 0, 50 epochs | 0.1561 | 0.1509 | same | same |
| 2 km | 3D stem, λ 0.3, 20 epochs | 0.1606 | 0.1525 | 0.3604 / – | 4.20 M / 4.38 M |

"3D stem" means `--model temporal3d`. It sees CHL with donor clouds on days t−4..t+4, plus 8 physics features
(SST, salinity, mixed-layer depth, sea-level anomaly, currents, wind), the day-of-year climatology, coordinates and the
year. It has 8 dilated residual blocks (661 k parameters) and is trained with L1 loss plus a gradient loss (weight 100).
λ is the weight of the semi-Lagrangian transport residual.

**Other metrics for the best 4 km model on val.** Values are for val days; test is in metrics.json.

| metric | value |
|---|---|
| RMSE | 0.243 |
| bias | −0.006 |
| r | 0.966 |
| gradient ratio (std of the predicted gradient / std of the true gradient) | 0.54 |
| gradient correlation | 0.66 |
| MAE on visible (not hidden) pixels | 0.092 |

## 2. Model families at 2 km (same val days)

| model | val MAE | epochs | note |
|---|---|---|---|
| DOY climatology | 0.3604 | – | |
| UDE: daily transport + learned source, forecast | 0.2007 | 4 | 1-day lead; persistence 0.2142 on that set (different pixel set) |
| dilated CNN, ±2 days | 0.1975 | 12 | batch 8 |
| spatial nODE (steady anisotropic diffusion) | 0.1772 | 6 | |
| 3D stem, λ 0 | 0.1627 | 20 | resumed after an interruption with a fresh optimizer state |
| 3D stem, λ 0.3 | 0.1606 | 20 | |

## 3. What changed the score (each a single seed)

| change | grid | Δ val MAE | comparison |
|---|---|---|---|
| physics loss λ 0 → 0.3 | 2 km, 20 ep | −0.0021 | the λ 0 run was resumed after an interruption |
| physics loss λ 0 → 0.3 | 4 km, 20 ep, batch 8 | −0.0001 | |
| physics loss λ 0 → 0.3 | 4 km, 50 ep, batch 16 | −0.0015 | |
| physics loss λ 0.3 → 1.0 | 4 km, epoch 12 of a 50-ep schedule | +0.026 | periodic val 0.1968 → 0.2230 |
| physics loss, residual on cloud pixels | 4 km, 20 ep | \|r\| 0.127 → 0.098 | MAE unchanged |
| 8 → 7 / 6 / 5 / 4 residual blocks | 2 km, epoch 12 | +0.0045 / +0.0011 / +0.0061 / +0.0102 | |
| window ±4 → ±3 days | 2 km, epoch 12 | −0.0014 | |
| 20 → 50 epochs, batch 8 → 16 | 4 km, λ 0.3 | −0.0091 | two things changed at once |
| +50 epochs, warm restart at lr 1e-3 | 4 km | −0.0047 | val jumped to 0.178 at the restart, then recovered |
| +50 epochs, warm restart at lr 1e-4 | 4 km | −0.0001 | periodic val 0.150–0.153 throughout |

Seed-to-seed noise has not been measured on the Black Sea. On the Indian Ocean it is about 0.0001.

## 4. Where the best 4 km model errs (val)

![error breakdown](blk4/figures/error_breakdown.png)

**By region:**

| region | MAE | bias | DOY clim MAE | n px | n days |
|---|---|---|---|---|---|
| W open sea | 0.114 | +0.002 | 0.331 | 283,566 | 166 |
| E open sea | 0.115 | +0.002 | 0.334 | 458,430 | 177 |
| Crimea/Kerch/Caucasus shelf | 0.177 | −0.070 | 0.354 | 21,784 | 161 |
| NW shelf | 0.205 | −0.030 | 0.415 | 154,519 | 168 |
| Anatolian shelf | 0.206 | −0.053 | 0.388 | 32,553 | 173 |
| Azov + Kerch | 0.274 | +0.017 | 0.403 | 86,163 | 140 |
| Marmara/Bosporus | 0.324 | −0.065 | 0.568 | 25,307 | 144 |

**By true ln CHL decile.** MAE is 0.10–0.12 from D1 to D7. It rises to 0.148 in D8, 0.254 in D9 and 0.334 in D10.
Bias runs from +0.084 in D1 to −0.128 in D10: low values come out too high and high values too low.

**By the day's observed fraction.** MAE falls from 0.213 on the cloudiest fifth of days to 0.133 in the fourth
fifth, then reads 0.143 on the clearest fifth.

**By month.** MAE ranges from 0.129 in July to 0.178 in February. The climatology's error ranges from 0.240 in
December to 0.531 in July.

**Data notes.**
- In Azov and Marmara the salinity input is NaN almost everywhere, set to 0. The currents there are also missing,
  since the reanalysis domain is smaller than the CHL box.
- `chl_l3` reaches 234,663 mg/m³ on ocean pixels. 5,257 of 3.89 M sampled values exceed 100 mg/m³. The 3D-stem runs
  do not clip them.

## 5. Related baselines (2 km, forecasting rather than gap filling)

| baseline, 1-day lead | val MAE | n px |
|---|---|---|
| persistence (copy yesterday's observed pixels) | 0.188 | 9.20 M |
| persistence moved along the currents | 0.198 | 9.20 M |
| persistence moved along currents scaled by 0.25 | 0.186 | 9.20 M |
| DOY climatology | 0.375 | 9.20 M |

The median current displacement is 5.4 cells per day at 2 km. The physics loss uses these same currents.

## 6. Open questions and untried options (for brainstorming)

**Evaluation**
- How large is seed noise on the Black Sea? Every Δ in section 3 is a single seed.
- Does the 4 km model hold up when scored against the 2 km truth, after upsampling its output? This would make the two
  grids comparable.
- How does the Copernicus gap-free L4 product score on the same hidden pixels? The Black Sea has no external
  benchmark yet.
- Black Sea runs use `donor_exclude` 0, so the donor day can sit inside the ±4-day window. The Indian Ocean nowcast
  used 8.

**Error structure (section 4)**
- Error is concentrated in the top two CHL deciles, on the shelves, in Azov and in Marmara. The bias there is negative
  (predictions too low).
- The gradient ratio is 0.54, so predicted fields are smoother than the truth.
- Visible-pixel MAE is 0.092, against 0.150 on hidden pixels.

**Physics**
- The transport residual falls with λ, but MAE barely moves.
- λ = 1.0 slowed early training.
- Advecting with the full-strength currents does worse than plain persistence.
- Untried: λ between 0.3 and 1.0, and a currents scale factor in the loss.

**Inputs**
- No outlier clipping of CHL (`--chl-max`) in the 3D-stem runs; the spatial nODE runs used 100 mg/m³.
- The 3D stem's only static geography input is log distance to the coast (one of the 5 coordinate channels).
  Bathymetry (`depth.npy`) and distance to river mouths (`river_dist.npy`) exist in the 2 km store but are not inputs.
- Azov and Marmara have no salinity or currents.

**Training**
- Restarting at lr 1e-4 gave nothing.
- Untried on the 4 km grid: a wider model, more blocks, the ±3-day window (which helped slightly at 2 km), a
  different loss weighting for high-CHL pixels, and the forecast and nowcast set-ups (only run on the Indian Ocean).
