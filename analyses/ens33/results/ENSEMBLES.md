# ens33 ensembles: members, weights, scores (eval only, not frozen)

Source: results/ens33.json (analyses/ens33/report.py). Weights per bin are ≥ 0 and sum to 1, fit on one val half (A or B) to minimise hidden-px MAE. oos val = half A scored with the B fit + half B with the A fit. Inputs: ~/mind_the_chl_gap/data/ens33/{io,blk4}_hidden.npz (per-hidden-px member predictions, from analyses/ens33/prep.py); member preds: IO via score.load_preds(run), Black Sea via <run>/preds.npz. Full weight table: results/weights.csv.

## io

Members: io6_tokdec_blend_spatial, io4_tokdec_blend, io3_ptok4, io2_base_eval, io2_seed1, io2_seed2, io7_physmem, io6_tokdec_blend_spatial_s1, io6_tokdec_blend_spatial_s2, io7_wmattn, io7_analog

| ensemble | val | test (fit A / fit B) | val A | val B |
|---|---|---|---|---|
| ens11_mean | 0.12418 | 0.12444 | 0.12985 | 0.12033 |
| w_keybins (oos) | 0.12366 | 0.12375 / 0.12386 | 0.12905 | 0.12000 |
| w_distbins (oos) | 0.12384 | 0.12397 / 0.12405 | 0.12926 | 0.12016 |
| w_global (oos) | 0.12411 | 0.12430 / 0.12428 | 0.12959 | 0.12039 |

### io w_keybins weights (fit A / fit B)

| member | 0 | 1-8 | 9+ |
|---|---|---|---|
| io6_tokdec_blend_spatial | 0.11 / 0.00 | 0.04 / 0.05 | 0.07 / 0.06 |
| io4_tokdec_blend | 0.00 / 0.07 | 0.09 / 0.09 | 0.00 / 0.01 |
| io3_ptok4 | 0.00 / 0.00 | 0.05 / 0.07 | 0.11 / 0.09 |
| io2_base_eval | 0.00 / 0.02 | 0.00 / 0.08 | 0.10 / 0.16 |
| io2_seed1 | 0.00 / 0.00 | 0.02 / 0.03 | 0.09 / 0.11 |
| io2_seed2 | 0.00 / 0.00 | 0.00 / 0.03 | 0.17 / 0.16 |
| io7_physmem | 0.48 / 0.34 | 0.28 / 0.23 | 0.14 / 0.12 |
| io6_tokdec_blend_spatial_s1 | 0.14 / 0.13 | 0.13 / 0.15 | 0.13 / 0.11 |
| io6_tokdec_blend_spatial_s2 | 0.18 / 0.24 | 0.14 / 0.13 | 0.08 / 0.08 |
| io7_wmattn | 0.06 / 0.10 | 0.10 / 0.08 | 0.00 / 0.01 |
| io7_analog | 0.04 / 0.09 | 0.14 / 0.06 | 0.11 / 0.09 |

### io w_distbins weights (fit A / fit B)

| member | 1 | 2-3 | 4-7 | 8-15 | 16+ |
|---|---|---|---|---|---|
| io6_tokdec_blend_spatial | 0.07 / 0.07 | 0.11 / 0.06 | 0.06 / 0.02 | 0.01 / 0.03 | 0.11 / 0.00 |
| io4_tokdec_blend | 0.01 / 0.03 | 0.04 / 0.02 | 0.01 / 0.00 | 0.03 / 0.01 | 0.00 / 0.09 |
| io3_ptok4 | 0.15 / 0.11 | 0.12 / 0.11 | 0.09 / 0.10 | 0.07 / 0.05 | 0.00 / 0.00 |
| io2_base_eval | 0.13 / 0.14 | 0.08 / 0.14 | 0.06 / 0.13 | 0.04 / 0.13 | 0.00 / 0.11 |
| io2_seed1 | 0.12 / 0.14 | 0.10 / 0.12 | 0.05 / 0.09 | 0.05 / 0.02 | 0.00 / 0.00 |
| io2_seed2 | 0.15 / 0.14 | 0.15 / 0.14 | 0.15 / 0.13 | 0.10 / 0.11 | 0.00 / 0.00 |
| io7_physmem | 0.10 / 0.08 | 0.16 / 0.10 | 0.22 / 0.15 | 0.30 / 0.22 | 0.42 / 0.33 |
| io6_tokdec_blend_spatial_s1 | 0.12 / 0.12 | 0.10 / 0.11 | 0.11 / 0.14 | 0.11 / 0.16 | 0.17 / 0.11 |
| io6_tokdec_blend_spatial_s2 | 0.05 / 0.05 | 0.03 / 0.09 | 0.08 / 0.10 | 0.13 / 0.10 | 0.20 / 0.21 |
| io7_wmattn | 0.00 / 0.02 | 0.01 / 0.00 | 0.01 / 0.05 | 0.00 / 0.08 | 0.06 / 0.07 |
| io7_analog | 0.10 / 0.10 | 0.10 / 0.09 | 0.15 / 0.09 | 0.17 / 0.09 | 0.03 / 0.08 |

### io w_global weights (fit A / fit B)

| member | all |
|---|---|
| io6_tokdec_blend_spatial | 0.10 / 0.02 |
| io4_tokdec_blend | 0.01 / 0.04 |
| io3_ptok4 | 0.04 / 0.06 |
| io2_base_eval | 0.00 / 0.13 |
| io2_seed1 | 0.06 / 0.05 |
| io2_seed2 | 0.05 / 0.07 |
| io7_physmem | 0.31 / 0.23 |
| io6_tokdec_blend_spatial_s1 | 0.15 / 0.13 |
| io6_tokdec_blend_spatial_s2 | 0.14 / 0.13 |
| io7_wmattn | 0.04 / 0.05 |
| io7_analog | 0.11 / 0.09 |

## blk4

Members: blk4_tokdec_blend_spatial, blk4_physmem, blk4_tokdec_blend, blk4_tokdec_krig, blk4_tokdec_krig_signed, blk8_spatial_bsmem, blk8_spatial_p12, blk4_t3d_c4_gdl100_lam0_b16_50ep, blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50

| ensemble | val | test (fit A / fit B) | val A | val B |
|---|---|---|---|---|
| mean_all9 | 0.14109 | 0.13415 | 0.13754 | 0.14410 |
| mean_core7 | 0.14009 | 0.13290 | 0.13676 | 0.14291 |
| mean_bestsub_val | 0.13931 | 0.13260 | 0.13636 | 0.14180 |
| mean_bestsub_A | 0.13931 | 0.13260 | 0.13636 | 0.14180 |
| mean_bestsub_B | 0.13946 | 0.13322 | 0.13679 | 0.14171 |
| w_keybins (oos) | 0.13941 | 0.13249 / 0.13287 | 0.13627 | 0.14207 |
| w_distbins (oos) | 0.13951 | 0.13250 / 0.13297 | 0.13655 | 0.14202 |
| w_global (oos) | 0.13954 | 0.13247 / 0.13297 | 0.13659 | 0.14203 |

Plain-mean members: mean_core7: blk4_tokdec_blend_spatial, blk4_physmem, blk4_tokdec_blend, blk4_tokdec_krig, blk4_tokdec_krig_signed, blk8_spatial_bsmem, blk8_spatial_p12; mean_bestsub_val: blk4_tokdec_blend_spatial, blk4_physmem, blk4_tokdec_krig_signed, blk8_spatial_bsmem; mean_bestsub_A: blk4_tokdec_blend_spatial, blk4_physmem, blk4_tokdec_krig_signed, blk8_spatial_bsmem; mean_bestsub_B: blk4_tokdec_blend_spatial, blk4_physmem, blk8_spatial_bsmem

### blk4 w_keybins weights (fit A / fit B)

| member | 0 | 1-8 | 9+ |
|---|---|---|---|
| blk4_tokdec_blend_spatial | 0.00 / 0.00 | 0.52 / 0.32 | 0.35 / 0.24 |
| blk4_physmem | 0.28 / 0.36 | 0.31 / 0.47 | 0.12 / 0.30 |
| blk4_tokdec_blend | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| blk4_tokdec_krig | 0.15 / 0.00 | 0.00 / 0.00 | 0.04 / 0.00 |
| blk4_tokdec_krig_signed | 0.00 / 0.00 | 0.09 / 0.04 | 0.27 / 0.13 |
| blk8_spatial_bsmem | 0.00 / 0.36 | 0.00 / 0.17 | 0.22 / 0.33 |
| blk8_spatial_p12 | 0.49 / 0.28 | 0.07 / 0.00 | 0.00 / 0.00 |
| blk4_t3d_c4_gdl100_lam0_b16_50ep | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 | 0.08 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 |

### blk4 w_distbins weights (fit A / fit B)

| member | 1 | 2-3 | 4-7 | 8-15 | 16+ |
|---|---|---|---|---|---|
| blk4_tokdec_blend_spatial | 0.20 / 0.19 | 0.30 / 0.30 | 0.28 / 0.29 | 0.11 / 0.12 | 0.45 / 0.22 |
| blk4_physmem | 0.20 / 0.22 | 0.13 / 0.15 | 0.07 / 0.18 | 0.16 / 0.27 | 0.19 / 0.38 |
| blk4_tokdec_blend | 0.12 / 0.08 | 0.09 / 0.00 | 0.10 / 0.00 | 0.02 / 0.14 | 0.00 / 0.00 |
| blk4_tokdec_krig | 0.19 / 0.22 | 0.11 / 0.20 | 0.00 / 0.21 | 0.00 / 0.04 | 0.07 / 0.00 |
| blk4_tokdec_krig_signed | 0.13 / 0.16 | 0.14 / 0.13 | 0.31 / 0.15 | 0.46 / 0.21 | 0.09 / 0.00 |
| blk8_spatial_bsmem | 0.13 / 0.14 | 0.20 / 0.18 | 0.19 / 0.16 | 0.25 / 0.22 | 0.16 / 0.40 |
| blk8_spatial_p12 | 0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.04 / 0.00 |
| blk4_t3d_c4_gdl100_lam0_b16_50ep | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 | 0.02 / 0.00 | 0.04 / 0.05 | 0.06 / 0.02 | 0.00 / 0.00 | 0.00 / 0.00 |

### blk4 w_global weights (fit A / fit B)

| member | all |
|---|---|
| blk4_tokdec_blend_spatial | 0.35 / 0.24 |
| blk4_physmem | 0.18 / 0.33 |
| blk4_tokdec_blend | 0.00 / 0.00 |
| blk4_tokdec_krig | 0.06 / 0.00 |
| blk4_tokdec_krig_signed | 0.22 / 0.10 |
| blk8_spatial_bsmem | 0.19 / 0.33 |
| blk8_spatial_p12 | 0.02 / 0.00 |
| blk4_t3d_c4_gdl100_lam0_b16_50ep | 0.00 / 0.00 |
| blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 | 0.00 / 0.00 |
