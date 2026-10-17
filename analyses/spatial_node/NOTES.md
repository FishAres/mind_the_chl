# Spatial nODE gap filler: notes (2026-10-06; all queued runs finished)

Model: steady state of dC/ds = div(D(x) grad C) + data/prior terms, i.e. the minimiser of
E(C) = 1/2 sum_corners a_c grad C^T D grad C + 1/2 sum_pixels a_p [beta (C - clim)^2 + sum_k w_k m_k (C - y_{t+k})^2],
with C = y_t on today's visible pixels (Dirichlet; --alpha makes it soft). One batched CG solve per day, implicit-function
gradient (one more CG solve). Code: chl_cnn/spatial_node.py, chl_cnn/train_spatial.py. Checks: analyses/spatial_node/test_operator.py (16/16).
Scoring identical to train_pinn.py (fake-cloud hidden pixels, val years 2001/2007/2013/2018; "val days 365" = every 4th val day).

## Val hidden-pixel MAE (ln CHL), every 4th val day, 1.23 M hidden pixels, 3 epochs on every 4th training day unless noted
| model | context | MAE | grad ratio | grad corr | learned |
|---|---|---|---|---|---|
| DOY climatology (prior) | | 0.2523 | | | |
| iso D | 0 | 0.2254 | 0.41 | 0.22 | beta 0.026 |
| SST tensor | 0 | 0.2079 | 0.56 | 0.28 | g0 0.02, anisotropy median 7, p90 34 |
| MLP (9 corner covariates) | 0 | 0.1976 | 0.58 | 0.42 | |
| MLP, 8 epochs (dn_mlp_lr3e3_d4) | 0 | 0.1958 | 0.59 | 0.41 | beta 0.0106 -> 0.0045 over 8 epochs |
| SST tensor | t+-1 soft terms | 0.1782 | 0.64 | 0.38 | w(+-1) 0.34, beta 0.0013 |
| SST tensor | t+-2 soft terms | **0.1701** | 0.65 | 0.41 | w(+-1) 0.38, w(+-2) 0.11, beta 0.0014 |
| SST tensor, Lagrangian advection of neighbour days | t+-2 | 0.1704 | 0.59 | 0.40 | |
| SST tensor | t+-4 | 0.1660 | 0.66 | 0.42 | w(+-1) 0.46, w(+-2) 0.13, w(+-3) 0.04, w(+-4) 0.025 |
| MLP + extra features | t+-2 | 0.1634 | 0.65 | 0.48 | |
| conv D-net (3x3, 3 layers) | 0 | 0.1944 | 0.57 | 0.42 | |
| conv D-net + t+-2 CHL channels | t+-2 soft terms | **0.1613** | 0.63 | 0.48 | w(+-1) 0.57, w(+-2) 0.17, beta 0.0010 |
| MLP, 8 epochs, lr 1e-2 (dn_mlp_lr1e2_d4) | 0 | 0.1919 | 0.61 | 0.42 | |
| MLP + extra features | t+-4 | 0.1603 | 0.65 | 0.48 | w(+-1) 0.60, w(+-2) 0.17, w(+-3) 0.05, w(+-4) 0.03 |
| SST tensor, soft today's term (alpha learned) | t+-2 | 0.1703 | 0.65 | 0.41 | alpha -> 104 (i.e. Dirichlet) |
| conv D-net + t+-2 channels, 6 epochs on ALL training days (mt_final_conv_ctx2) | t+-2 | 0.1552 | 0.65 | 0.50 | w(+-1) 1.05, w(+-2) 0.21, beta 0.0004 |
| **conv D-net + t+-4 channels, 6 epochs on ALL training days (mt_final_conv_ctx4)** | t+-4 | **0.1520** (final 0.1524) | 0.63 | 0.50 | beta 0.00023 |
| conv D-net + t+-4 channels, 12 epochs on all training days (mt_long_conv_ctx4); curve 0.1565 0.1546 0.1533 0.1534 0.1519 0.1509 0.1509 0.1499 0.1500 0.1498 0.1498 0.1498 | t+-4 | **0.1498** | 0.62 | 0.51 | flat from epoch 8 |
| same model scored on ALL 1,458 val days, 5.0 M hidden px (comparable with the CNN table) | t+-4 | **0.1516**, RMSE 0.252 | 0.66 | 0.51 | CNN pinn_v2_lam0: 0.1365, RMSE 0.231, 0.55 / 0.59; climatology 0.254 |
| conv D-net + learned source g (prior shift, |g| <= 1), t+-2 (mt_convsrc_ctx2) | t+-2 | 0.1618 | 0.65 | 0.49 | vs 0.1613 without g; anisotropy median 1.8 vs 2.1 |

Seed noise (seed 1 vs seed 0, 3 epochs stride 4): sst ctx0 0.2079 vs 0.2079; sst ctx2 0.1701 vs 0.1701; conv ctx2 0.1614 vs 0.1613. Differences >= 0.001 between configurations are above seed noise.
| dilated CNN pinn_v2_lam0 (reference, 60 epochs, CHL t+-2 inputs) | | 0.1365 | 0.55 | 0.59 | |

## Hidden-pixel MAE by distance to the nearest visible pixel (analyses/spatial_node/error_by_distance.py, figures/mae_by_distance.png)
| distance (cells) | 1 | 2 | 3-4 | 5-7 | 8-11 | 12-19 | 20-39 | >=40 |
|---|---|---|---|---|---|---|---|---|
| MLP D, no context | 0.127 | 0.156 | 0.181 | 0.207 | 0.226 | 0.252 | 0.284 | 0.343 |
| SST tensor, t+-2 soft terms (tc_sst_ctx2) | 0.114 | 0.131 | 0.143 | 0.158 | 0.176 | 0.213 | 0.271 | 0.376 |
| conv D + t+-2, all training days (mt_final_conv_ctx2) | 0.105 | 0.122 | 0.133 | 0.145 | 0.161 | 0.192 | 0.249 | 0.332 |
| dilated CNN | 0.085 | 0.106 | 0.118 | 0.130 | 0.143 | 0.171 | 0.223 | 0.307 |
| DOY climatology | 0.244 | 0.241 | 0.241 | 0.242 | 0.243 | 0.254 | 0.282 | 0.350 |
| share of hidden pixels | 0.24 | 0.13 | 0.13 | 0.14 | 0.10 | 0.11 | 0.10 | 0.05 |
The spatial fill without context is 0.04 worse than the CNN already at distance 1 and reaches the climatology level by 12-19 cells. With t+-2 terms the error falls at every distance up to 20 cells; beyond 40 cells it is above the climatology (0.376 vs 0.350; 5 % of hidden pixels), where the learned prior weight beta = 0.0014 leaves the field weakly tied to the climatology.

## MLP corner-feature ablation (dn_abl_*, 3 epochs stride 4, no context)
| features | all 9 | minus SST gradient | minus currents | minus wind | minus log depth |
|---|---|---|---|---|---|
| val MAE | 0.1985 | 0.1987 | 0.1987 | 0.1978 | 0.2182 |
Seed noise is 0.0001. The SST-tensor model alone (SST gradient only, 3 params) scores 0.2079 vs isotropic 0.2254.

## Other D-net findings (dn_* runs, 8 epochs stride 4, no context)
- MLP lr 1e-2 vs 3e-3: 0.1919 vs 0.1958 (base features); 0.1899 vs 0.1942 (extra features). d_max 16 vs 4: no change (0.1954 vs 0.1958; 0.1942 vs 0.1942).
- conv D-net, 8 epochs, no context: depth 3 dilation 1 0.1901; dilation 2 0.1881; depth 5 0.1890; 5 input channels (SST, flag, depth, ocean, clim) 0.1929. MLP at the same schedule: 0.1958 (lr 3e-3) / 0.1919 (lr 1e-2).
- SST-residual MLP (MLP correction on the SST tensor): 0.1931 (base feats) / 0.1919 (extra feats) vs plain MLP 0.1958 / 0.1942 at lr 3e-3.
- MLP val MAE curves flatten after epoch 1-2 (gain after epoch 0 is 0.003-0.005 over 7 epochs); beta drifts down while D grows (scale freedom of D vs beta).

## Data issues found
- The float16 store quantises SST to 0.25 K; 78 % of neighbouring SST pairs are identical. sst32.npy (float32, chl_cnn/store_sst32.py) is used here; the zarr itself still has 27 % identical pairs (steps near 1/16 K), visible as speckle in the anisotropy maps. The CNN runs so far used the quantised SST as an input channel.
- The brief's climatology reference 0.358 is the per-pixel all-year mean; the day-of-year climatology scores 0.254 on the same pixels.

## Run queues (all finished 04:52 UTC; tmux sessions dn_queue, tc_queue1, hy_queue, mt*_queue)
- dn_*: MLP lr/d_max grid, feature ablation, conv D-net (depth 3/5, dilation 1/2, 5-channel input), SST-residual MLP. Curves: analyses/spatial_node/dn_curves.py.
- tc_*: MLP + extra features with/without t+-2, conv with t+-2 channels.
- DONE hy_fill_sstctx2 / hy_ctrl (models/pinn/cnn): the dilated CNN v2 recipe with the SST-tensor t+-2 steady-state fill added as a 27th input channel (frozen parameters of tc_sst_ctx2, computed on the fly per batch, ~170 s/epoch vs 45 s), 20 epochs, and the same recipe without it.

## Hybrid: steady-state fill as a CNN input channel (20-epoch cosine, all val days, 5.0 M hidden px)
| epoch | 1 | 3 | 5 | 8 | 10 | 15 | 20 | test (20) | grad ratio | grad corr |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN + fill channel | 0.1579 | 0.1527 | 0.1505 | 0.1464 | 0.1446 | 0.1426 | **0.1412** | 0.1434 | 0.552 | 0.589 |
| CNN control (same schedule) | 0.1822 | 0.1598 | 0.1611 | 0.1473 | 0.1457 | 0.1431 | **0.1410** | 0.1439 | 0.540 | 0.578 |
The fill channel shortens the early training (0.024 lower at epoch 1, 0.009 at epoch 8) and the two runs are equal from epoch 16 on. (pinn_v2_lam0 at 60 epochs: 0.1365.)
- mt_*: context t+-4, soft today's data term (alpha 20 learned), saved fills of tc_sst_ctx2 for the distance diagnostic.
