# io7_physmem: physics memory + longer token window (bridge #25, Phase 1 run 1)

Base: io6_tokdec_blend_spatial (token decoder, own-pixel blend, 2 spatial blocks). Code: `chl_cnn/physmem.py`
(`PhysMemTokDecNet`, `MemTable`); train_pinn.py flags `--physmem [--physmem-blend-k 8] [--physmem-stride2]`, default off.

Run: `train io7_physmem $SPATIAL --context 15 --feat-context 15 --physmem` (or + `--physmem-stride2` if the smoke OOMs;
the flags used are in `~/mind_the_chl_gap/logs/io7/physmem_flags.txt`).

## Model
- Token window t-15..t+15 (31 days; fallback: every day to ±8, even offsets beyond, 23 days). CHL of every window day goes
  through the same donor (persist) masking as the ±8 window (`nb_masks` in train_pinn.py, unchanged).
- Blend: days |d - t| ≤ 8 only (17 slots + "no day"); pixel-term offset features scaled by 8, as in io6.
- Memory: 12 channels per P x P patch (38 x 46 grid) -> Conv2d(12, 128, 3) -> GELU -> Conv2d(128, 128, 1), zero-init,
  added to every day token of the patch before the temporal self-attention (30,464 params). At init the memory is a no-op.

## Physics summaries (past-only, target day t)
Daily fields at 0.25°, then mean over ocean px with finite values per patch (cache `~/mind_the_chl_gap/data/io7/`):
- tau = rho_a C_d |U| U from 10 m winds (Large & Pond C_d, `chl_cnn/physics.py`).
- curl = d(tau_y)/dx - d(tau_x)/dy.
- Ekman pumping w_E = curl / (rho_w f) + beta tau_x / (rho_w f^2); NaN for |lat| < 5° (f -> 0).
- Alongshore stress tau_s = sign(lat) tau · s_hat, s_hat = z_hat x n_hat. n_hat is the offshore unit normal, -grad L / |grad L|,
  where L = land fraction (1 - ocean mask) smoothed by a Gaussian (sigma 2 px). Only ocean px within 4 px (~100 km) of land;
  + = upwelling-favourable (offshore Ekman transport; e.g. southward wind on the Indian west coast, SW monsoon off Somalia/Oman).
- SSTA = SST - DOY climatology of the training days (per patch, 31-day circular mean over DOY).
- MLD = mlotst.
Summaries: windows A = [t-30, t-1], B = [t-60, t-31] (calendar days). Means of curl, w_E, tau_s, SSTA, and
dMLD = mean(second half) - mean(first half) of each window -> 10 channels, plus the fraction of each window's days missing
from the store (2) -> 12. Two 30-day means give 60 days of memory in 2 numbers per quantity and cover the 1-2 month lag of
upwelling / mixing responses. A half-window difference is a smoothed MLD tendency (endpoint differences of daily MLD are noisy).
The 10 means are z-scored (train days, ocean patches) and clipped to ±6; a mean needs ≥ 15 finite days (half ≥ 8), else 0.
Start of record (1997-10-01) and store gaps: windows just hold fewer days; the missing-fraction channels say how many
(day 0: 1.0). Val starts 2001, so every val / test day has a full 60-day history. Valid fraction on train days: w_E 0.65
(equatorial band), tau_s 0.25 (coastal patches), the rest ≥ 0.985. Train days near a val/test year use that year's physics
(not CHL).

## Checks (CPU)
`python -m analyses.io7.physmem.test_physmem` (17 checks): PhysMemTokDecNet on ±8 equals TokDecNet exactly; forward /
backward shapes; memory gradients; CHL under invisible px changes no token; summaries of day t do not change when physics on
days ≥ t changes and do change for day t-1; start of record; stride-2 offsets. CPU dry runs of train_pinn (±15, stride-2,
default io6 config) ran to scoring.
Leak test (queue): `analyses/io7/physmem/leak_physmem.py`: hidden, masked t+2 / t+12 / t-12 must change nothing; obs t+2 /
t+12 must change the prediction (and the tokens for t+12).

## Memory (A10G 23 GB)
Saved activations on CPU (fp32, B 2, full grid): io6 ±8 1.67 GB, ±15 physmem 2.22 GB (x1.33), stride-2 1.92 GB; the backbone
part is unchanged (0.72 GB). io6 peak at B 8: 11.2 GB allocated / 16.0 reserved; about 6.6 GB of it is the GPU feature
store. Estimate for ±15: ~12.5-13.5 GB allocated, ~17-19 reserved. Expect ~160-180 s/epoch (io6 128).
