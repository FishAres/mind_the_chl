# Shared findings log (agents: read before planning, append after each finished run or batch of runs)

Format: one bullet per finding, dated, with the run names and numbers. Observations only; no interpretation beyond "consistent with".
Val hidden-pixel MAE (ln CHL) is the score unless stated. Indian Ocean (IO) = 0.25 deg store; Black Sea (BLK) = 2 km store (--store blk).
Every run with its settings and scores, generated: [models/RUNS.md](../models/RUNS.md) (`python -m chl_cnn.runs_index`).

## Protocol and references
- 2026-10-06 IO: dilated CNN (train_pinn.py, pinn_v2_lam0, 60 ep) val 0.1365, test 0.1434; DOY climatology 0.254 on the same hidden pixels (the per-pixel all-year mean scores 0.358). Seed noise of the spatial model: 0.0001.
- 2026-10-06 IO: the physics (transport) loss of train_pinn.py changes val MAE by <= 0.001 for lambda in {0, 0.1, 0.3, 1}; it smooths the field under real clouds. OSCAR currents at 0.25 deg explain ~1 % of day-to-day ln CHL variance; Courant > 1 on half the pixel-days.
- 2026-10-09 BLK store: 395 x 550, 93,594 ocean px, 2014-2024, 3,997 days; val years 2017+2021, test 2019+2023 (untouched). Climatology on val hidden pixels 0.371 (every 40th day). Courant number of the 2.5 km reanalysis currents on the 2 km grid: median ~6 cells/day, so the finite-difference transport residual is unusable; a semi-Lagrangian (back-trajectory) formulation is required.

## Spatial nODE (train_spatial.py), IO
- Context (soft terms from days t+-k) is worth ~0.03 for every D model: sst 0.208 -> 0.170 (t+-2) -> 0.166 (t+-4); mlp+extra 0.196 -> 0.163; conv 0.194 -> 0.161 (t+-2, ctx channels). Learned w(+-2)/w(+-1) ~ 0.2-0.3, w(+-3)/w(+-1) ~ 0.04.
- Best IO spatial model: conv D + t+-4 channels, all training days, 12 ep: 0.1498 (every 4th val day), 0.1516 on all val days; grad ratio 0.66, grad corr 0.51 (CNN 0.55 / 0.59).
- No gain: Lagrangian advection of neighbour days (+0.000), soft today's term (alpha -> 100), learned source g as a prior shift (+0.0005), d_max 16, feeding the steady-state fill to the CNN as a channel (equal at 20 ep, faster early epochs).
- MLP feature ablation (no context): dropping SST gradient / currents / wind costs <= 0.0002; dropping log depth costs 0.020. The 3-parameter SST tensor alone gains 0.018 over isotropic.
- Error by distance to the nearest visible pixel: the spatial model is 0.02-0.03 above the CNN at every distance, including distance 1 (0.105 vs 0.085).
- Training: the spatial models converge within 1-2 epochs at stride 4; the 12-epoch stride-1 curve is flat from epoch 8 (0.1499 -> 0.1498). lr schedule comparison (cosine vs constant) in progress (lr_* runs).

## Open threads being run
- BLK spatial ladder (blk_* runs, agent), lr schedule (lr_* runs, agent), BLK CNN +- physics loss (cnn_blk_* runs, agent).

## Black Sea static covariates
- 2026-10-09: Added to data/blk_store_2014_2024/: depth.npy (float32, 395x550, m positive down, NaN on land) and river_dist.npy (float32, 8x395x550, km haversine from mouths snapped to nearest ocean pixel) + river_names.json (Danube, Dniester, Dnieper-Bug, Kerch, Rioni, Sakarya, Kizilirmak, Bosporus). Depth source: NOAA ERDDAP etopo180 (ETOPO1, 1 arc-min, bilinear to the 2 km grid; GEBCO/ETOPO 2022 not used). Depth over 93,594 ocean pixels: min -268 m, median 1307 m, max 2581 m; 1.6% of ocean pixels have depth <= 0 (coastal mask/resolution mismatch, not clipped). Script: analyses/spatial_node/static_covariates.py.
- 2026-10-09 BLK CNN (agent): train_pinn.py ported to --store blk (--crop, --feat-device cpu streams features/currents from the memmap store, --phys semi-lagrangian back-trajectory residual with --lag-max 20, --max-batches, --val-stride, --eval-batch; IO defaults unchanged). No run completed yet: the GPU was held at 20-22.5 GB by the blk_* spatial queue and the lr_* queue throughout; first smoke attempt (batch 8, 8 features, crop) OOMed at 8.75 GB when a second job started simultaneously. Queue ready: ~/mind_the_chl_gap/logs/cnn_blk_queue.sh (waits for NEED_MB free before each of cnn_blk_lam0, cnn_blk_sl_lam0p3, cnn_blk_up_lam0p3, cnn_blk_sl_lam1); smoke watcher in tmux session cnn_blk_smoke. Summary/plots: analyses/pinn/blk_cnn_runs.py.

## Black Sea regions, climatology error and salinity (CPU analysis)
- 2026-10-09. Scripts and figures: analyses/blk/ (regions.py, clim_by_region.py, salinity_chl.py, model_by_region.py; figures/). Regions (px): NW shelf 14,972; W open sea 26,392; E open sea 36,259; Anatolian shelf 3,096; Crimea/Kerch/Caucasus shelf 2,168; Azov+Kerch 8,325; Marmara/Bosporus 2,382 (all 93,594 ocean px assigned). Overlaps resolved by rule order (later rule wins).
- DOY climatology (train days, 2,542) hidden-pixel MAE on val days every 4th (182 days, 4.20M hidden px): 0.360 overall. By region: NW shelf 0.421, W open 0.332, E open 0.337, Anatolian 0.392, Crimea/Caucasus 0.359, Azov 0.407, Marmara 0.566. By season (all ocean): DJF 0.271, MAM 0.318, JJA 0.463, SON 0.327.
- Open-sea MAE is 0.22-0.29 in DJF/MAM and 0.45-0.48 in JJA; shelves are 0.30-0.53 in all seasons.
- Observed-pixel fraction peaks in Jul-Aug (0.51-0.70) and is lowest in Dec-Jan (0.14-0.26; Azov 0.05-0.09 in Dec/Jan).
- Salinity (so) is NaN everywhere in Azov (100 %) and 98 % of Marmara; 24 % in Anatolian and Crimea shelves, 11 % NW shelf, ~0 % open sea. Azov has no salinity r.
- ln CHL anomaly vs salinity anomaly, Pearson r: NW shelf -0.31 (n 6.26M px-days; DJF -0.24, MAM -0.40, JJA -0.34, SON -0.16); Anatolian shelf -0.19; open seas -0.04 (W) and +0.07 (E); all ocean -0.12. NW shelf mean ln CHL anomaly runs +0.40 (lowest salinity-anomaly decile) to -0.32 (highest). NW shelf raw salinity: mean ln CHL 0.68 in the lowest decile (S 15.8) vs -0.85 in the top two (S 18.3-18.5).
- NW shelf: r(ln CHL anomaly, min river-mouth distance) is -0.01 to 0.01 in every season (the anomaly has zero mean per pixel; pixel-mean ln CHL vs distance on the same shelf, every 40th day: r = -0.50, n px = NW shelf pixels with observations).
- Data quality: chl_l3 max 234,663 mg/m3 on ocean px; of 3.89M finite ocean values (every 40th day) 5,257 exceed 100 and 270 exceed 1000 mg/m3. No all-NaN or all-zero CHL/so days. ln CHL NaN fraction on ocean px 0.585.
- 2026-10-09 (main): train_spatial.py gained --chl-max (default off): CHL above the threshold is treated as missing before the log. To be used (e.g. --chl-max 100) consistently in the next Black Sea round; runs so far include the outliers (0.14 % of observed pixels > 100 mg/m3).

## Learning-rate schedule (IO, conv t+-4, train stride 2)
- 2026-10-09 (agent): lr_* runs, 6 epochs, val every 4th val day (1.23 M hidden px), scalars lr / D-net lr. Cosine 3e-2/3e-3 (lr_cos) 0.1555; constant 3e-2/3e-3 (lr_const) 0.1541; constant 1e-2/1e-3 0.1548; constant 3e-2/1e-2 0.1551. Per-epoch curves identical for cosine and constant through epoch 3 (0.1587, 0.1576, 0.1559/0.1560); they separate once the cosine lr drops below ~1e-3 (epochs 5-6: 0.1563/0.1555 cosine vs 0.1549/0.1541 constant).
- 12 epochs: constant 3e-2/3e-3 continued 6 more epochs from lr_const (lr_const_cont) 0.1537, 0.1533, 0.1528, 0.1525, 0.1537, 0.1519 (final/best 0.1519); 12-epoch cosine at the same strides (lr_cos12) 0.1523 (best 0.1523). Both are 0.002 above the 12-epoch stride-1 cosine reference (0.1498). Constant lr is as good as or 0.0004-0.0014 better than cosine at both 6 and 12 epochs, differences at or above the 0.0001 seed noise but within the epoch-to-epoch jitter of the constant curve (+0.0012 at epoch 11).
- Constant-lr curves are not monotone: lr_const +0.0003 at epoch 4, lr_const_cont +0.0012 at epoch 11; cosine 12 ep +0.0017 at epoch 7. No NaN, no sustained rise. Train L1 keeps falling under constant lr (0.1413 at epoch 6, 0.1388 at epoch 12) while val flattens (gain 0.0022 over the last 6 epochs).
- Lower constant lr (1e-2/1e-3) starts worse (0.1628 vs 0.1587 after epoch 1) and is 0.0007 behind at epoch 6; higher net lr (1e-2) is 0.0006 ahead at epoch 1 and 0.0010 behind at epoch 6. Final beta 0.00028-0.00037 and grad ratio 0.64-0.66 / corr 0.50-0.51 are the same across schedules.
- Figure analyses/spatial_node/figures/lr_curves.png (script lr_curves.py); queue analyses/spatial_node/lr_queue.sh (lr_const_cont and lr_cos12 were relaunched after OOM collisions with the blk_ queue; logs ~/mind_the_chl_gap/logs/lr_queue{,2,3}.log).

## Black Sea spatial nODE round 2 (r2_* runs, agent)
- 2026-10-09: train_spatial.py gained four flags, all default off (IO runs unchanged; test_operator.py 16/16): --norm-d (D divided by its per-sample mean trace over ocean corners / 2, mean not detached; spatial_node.normalise_trace), --real-depth (blk: depth channel log10(max(depth.npy, 5 m)) instead of log10(1 + coast distance)), --sal-feat (blk: z-scored so + validity flag as conv channels; masked corner gradient + validity for the mlp), --river-feat (blk: z-scored log10(1 + min river-mouth distance km) as a conv channel / corner mean for the mlp). Pixel-stack order: 11 base, [so, so_ok], [river], ctx channels. Logs now carry CG iterations, cap hits and max relative residual for eval and training (fwd/bwd); history.json has a "cg" entry per epoch; config.json records depth_source and n_pix_ch. Smoke tests (stride 60/40, 1 epoch) of every flag combination and of --dmodel mlp passed (models/spatial/smk_*). Queue: ~/mind_the_chl_gap/logs/r2_queue.sh; summary: analyses/spatial_node/r2_summary.py.
- 2026-10-10 r2 results (r2_*, 3 epochs, stride 4, chl-max 100, sched none, maxiter 800, val every 4th day; 4,193,683 hidden px, DOY climatology 0.3575 on this set vs 0.3604 without chl-max): val MAE r2_ctrl 0.1813, r2_normd 0.1821, r2_depth 0.1814, r2_sal 0.1801; RMSE 0.2744 / 0.2750 / 0.2759 / 0.2730. Round 1 blk_conv_ctx4_cf: 0.1843 (cosine, maxiter 400, no chl-max); blk_final_conv_ctx4_cf (6 ep, stride 1): 0.1772. r2_ctrl is 0.0030 below blk_conv_ctx4_cf (schedule, maxiter and chl-max all changed together).
- Versus r2_ctrl: --sal-feat -0.0012, --real-depth +0.0001, --norm-d +0.0008. Single seeds; IO seed noise is 0.0001.
- Final beta / w(+-1): r2_ctrl 0.0041 / 0.111, r2_depth 0.0034 / 0.105, r2_sal 0.0036 / 0.103, r2_normd 0.0022 / 0.056 (blk_conv_ctx4_cf 0.0051 / 0.125). beta fell every epoch in all runs including r2_normd (0.0059, 0.0033, 0.0022).
- CG at maxiter 800: eval iterations 271-330 (mean per batch), 0/23 eval batches and 0/80 train batches capped in every run, rel residual max 1e-5. Round 1 used maxiter 400 with mean 289 (ctx4_cf) and 395 (final).
- Grad ratio / corr: r2_ctrl 0.467/0.326, r2_normd 0.453/0.334, r2_depth 0.470/0.325, r2_sal 0.444/0.337 (blk_conv_ctx4_cf 0.432/0.337). Aniso p90: 3.6 (ctrl), 2.1, 1.9, 1.9.
- r2_river: no metrics.json; relaunched by r2_queue2.sh at 01:17 on 2026-10-10 (history.json in the folder is from the interrupted run). r2_all, r2_all_lr1e2 and r2_final are queued behind it. Summary: analyses/spatial_node/r2_summary.py (figure figures/r2_curves.png).

## Black Sea CNN +- physics loss (cnn_blk_* runs)
- 2026-10-10: no results. cnn_blk_lam0, cnn_blk_sl_lam0p3, cnn_blk_up_lam0p3, cnn_blk_sl_lam1 each have only an empty figures/ folder (no config/history/metrics/model). All four crashed on 2026-10-09 in chl_cnn/train_pinn.py transport_residual, F.grid_sample line ~497: "input FloatTensor vs grid DoubleTensor" (semi-Lagrangian runs in train_epoch, line 539; lam0 and up_lam0p3 in residual_stats, line 577, which evaluates the other form, i.e. also semi-Lagrangian).
- Logs ~/mind_the_chl_gap/logs/cnn_blk_*.log stop after the "model dilated in_ch 25 params 607683" line (crop rows 9:361, cols 0:544, 352 x 544 grid, 93,594 ocean px; train 2542 / val 726 / test 729 days); the queue log reports "exit 0" for each because of the flock wrapper, so the exit codes are not informative. No epoch completed.
- Reference: IO physics loss changes val MAE by <= 0.001 (pinn_v2 lam 0 / 0.3 / 1: 0.1365 / 0.1370 / 0.1378). Black Sea CNN val MAE: not available. analyses/pinn/blk_cnn_runs.py runs but prints empty CNN table; blk_cnn_curves.png not written.
- 2026-10-10 (after the reboot; r2_river had been cut at epoch 0 and was rerun from scratch). Stride 4, 3 epochs, --chl-max 100 --maxiter 800 --sched none, conv D + t+-4 ctx channels; climatology on the same hidden pixels 0.3575: r2_river (--river-feat) 0.1803; r2_all (--norm-d --real-depth --sal-feat --river-feat) 0.1830; r2_all_lr1e2 (same, --lr-net 1e-2) 0.1832. Ranking of the seven stride-4 runs: r2_sal 0.1801 < r2_river 0.1803 < r2_ctrl 0.1813 < r2_depth 0.1814 < r2_normd 0.1821 < r2_all 0.1830 < r2_all_lr1e2 0.1832. Every run with --norm-d is 0.0008-0.0017 behind the control; beta and w(+-1) still fall over the epochs with --norm-d (r2_normd: beta 0.010 -> 0.0022, w(+-1) 0.05 -> 0.056 after a rise to 0.08 at epoch 0; r2_ctrl: 0.010 -> 0.0041, 0.05 -> 0.111). CG at maxiter 800 never hit the cap in any run (training fwd/bwd 0/80 batches per epoch, eval 0/23; max relative residual 1.0e-05 = tol); mean eval iterations 271-330, training 296-393 fwd. Table + curves: analyses/spatial_node/r2_summary.py -> figures/r2_curves.png.
- 2026-10-10 r2_final (best config = r2_ctrl + --sal-feat, --train-stride 1 --epochs 8 --save-preds) was launched by the queue at 01:44 and killed 25 s later by me while stopping the queue for the UDE runs (tmux server kill; nothing else was running on it). Not yet rerun; r2_all_mlp also pending. Queue script for the rest: ~/mind_the_chl_gap/logs/r2_queue2.sh (skips finished runs).

## Forecast baselines (persistence vs advected persistence)
- 2026-10-10 (agent). Script analyses/ude/forecast_baselines.py; results analyses/ude/results/*.json; figures analyses/ude/figures/mae_vs_lead_{io,blk}.png, blk_current_scale_sweep.png. Target days: every 2nd val day (IO 728, BLK 361-363 per lead); DOY climatology from training years; blend weights a by least squares on every 4th training day. Scores on the common pixel set (observed at t and t-k). Advection: semi-Lagrangian back-trajectory, 4 sub-steps/day, daily currents of days t-1..t-k, bilinear grid_sample; fallback to persistence where the trajectory crosses missing currents/land or the departure point is cloudy at t-k. v is converted to cells/s with the signed dy of grid_metrics, so positive (northward) v moves the field towards larger row index on BLK (lat ascends) and smaller row index on IO (lat descends). BLK CHL > 100 mg/m3 set to NaN (225,131 values).
- Displacement, cells/day (ocean px with valid currents, val days): IO median 0.76, p90 1.88; BLK median 5.36, p90 11.45.
- IO MAE (ln CHL), k = 1/2/3: persistence 0.1422/0.1874/0.2213; advected 0.1396/0.1838/0.2180; clim 0.2748/0.2728/0.2722; blend 0.1407/0.1786/0.2044 (a 0.81/0.70/0.60); adv blend 0.1385/0.1760/0.2020 (a 0.84/0.73/0.63); anom 0.1435/0.1898/0.2226; adv anom 0.1407/0.1855/0.2183. Common px 1.79M/1.69M/1.62M. Not advected 36/41/44 % (trajectory 6-9 %, cloudy departure point 30-36 %). On accepted pixels only (k=1, 1.15M): persistence 0.1421 vs advected 0.1381. Reversed currents (every 20th val day): advected 0.1546 vs persistence 0.1486 at k=1.
- BLK MAE, k = 1/2/3: persistence 0.1882/0.2248/0.2470; advected 0.1982/0.2408/0.2665; clim 0.3751/0.3844/0.3810; blend 0.1928/0.2273/0.2508 (a 0.73/0.63/0.55); adv blend 0.2029/0.2417/0.2664 (a 0.69/0.56/0.48); anom 0.1897/0.2270/0.2492; adv anom 0.1973/0.2386/0.2639. Common px 9.20M/8.13M/7.89M. Not advected 21/25/28 % (trajectory 12-15 %, cloudy departure 9-13 %). Accepted pixels only (k=1, 7.28M): persistence 0.1701 vs advected 0.1828.
- BLK by region, k=1 (persistence / advected / clim / blend, n px): NW shelf 0.239/0.269/0.438/0.239 (1.49M); W open 0.145/0.151/0.347/0.156 (2.83M); E open 0.171/0.177/0.367/0.179 (3.61M); Anatolian 0.256/0.276/0.392/0.245 (0.24M); Crimea/Caucasus 0.203/0.214/0.367/0.197 (0.16M); Azov 0.270/0.270/0.350/0.255 (0.66M, 100 % fallback: no currents); Marmara 0.363/0.363/0.521/0.344 (0.20M, 99 % fallback).
- BLK current scale alpha (advected MAE, k=1 / k=2): 0.25 -> 0.1861/0.2234; 0.5 -> 0.1884/0.2275; 0.75 -> 0.1928/0.2338; 1.0 -> 0.1982/0.2408; 1.5 -> 0.2093/0.2537 (persistence 0.1882/0.2248). Orientation at alpha 1, k=1: flip u 0.2107, flip v 0.2082, flip both 0.2078, swap u/v 0.2091, both reversed (every 20th day subset) 0.2020 vs forward 0.1947 on that subset. Same ordering at k=2.

## UDE stage A (Black Sea, transport off)
- 2026-10-10 (agent): udeA_* runs, chl_cnn/train_ude.py, --store blk --crop --chl-max 100 --transport none --ic fill --lead 3 --batch 8, 4 epochs at train stride 4 (78 batches/epoch), val = every 4th val t0 (178 t0; 6.6-7.5 M observed px per lead), constant lr 3e-2 / 3e-3. 6.7-7.8 min and 3.15 GB per mlp/conv run (fill 0.5 s/batch dominates); 2.8 min for --ic clim (no CG fill). Queue ~/mind_the_chl_gap/logs/udeA_queue.sh, logs udeA_*.log; summary analyses/ude/udeA_summary.py (figures analyses/ude/figures/udeA_curves.png, udeA_gain_by_lead.png).
- References on this val set (IC fill): persistence MAE lead 1/2/3 0.2142/0.2433/0.2566, climatology 0.3510/0.3735/0.3677; persistence RMSE l1 0.3087, grad ratio/corr 0.859/0.261. With --ic clim persistence is 0.2503/0.2836/0.2982.
- Final val MAE lead 1/2/3 (g-only = model, transport off): relax 0.2065/0.2423/0.2591 (lambda_r 0.10 -> 0.19 1/day; per epoch 0.200, 0.200, 0.202, 0.191); mlp 0.2015/0.2356/0.2542; conv 0.2007/0.2291/0.2450; mlp g-max 1.0 0.2015/0.2357/0.2545; mlp lr-net 1e-2 0.2014/0.2345/0.2513; mlp ic clim 0.2418/0.2779/0.2931. RMSE l1: 0.2962 / 0.2898 / 0.2891 / 0.2896 / 0.2898 / 0.3505. Grad ratio/corr l1: 0.745/0.287, 0.725/0.299, 0.763/0.267, 0.718/0.303, 0.724/0.298, 0.876/0.232.
- Gain over persistence: lead 1 -0.013 to -0.014 for every mlp/conv run (-0.008 relax); lead 2 -0.008 (mlp) to -0.014 (conv); lead 3 -0.002 (mlp), -0.005 (lr 1e-2), -0.012 (conv); relax +0.0025 at lead 3. Relax at init (lambda 0.1) scored 0.2075/0.2359/0.2467, below every trained relax epoch at leads 2-3; its mean-lead val MAE rose from 0.2300 (init) to 0.2359-0.2371 while train L1 fell 0.2205 -> 0.2197.
- Epoch curves are not monotone: mean-lead val MAE mlp 0.2297, 0.2360, 0.2316, 0.2305 (best epoch 1; model.pt is epoch 1), conv 0.2333, 0.2359, 0.2280, 0.2249 (best = last), lr 1e-2 0.2332, 0.2384, 0.2333, 0.2291. Epoch 2 is worse than epoch 1 at leads 2-3 in all mlp runs (lead 3 up to +0.017). Train L1 falls every epoch (mlp 0.229 -> 0.2157). No NaN.
- Regions at lead 1 (conv / persistence / climatology): NW shelf 0.2514 / 0.2621 / 0.4321; W open sea 0.1647 / 0.1791 / 0.3247; E open sea 0.1760 / 0.1889 / 0.3300; Anatolian shelf 0.2839 / 0.3032 / 0.3738; Crimea/Kerch/Caucasus 0.2408 / 0.2561 / 0.3517; Azov+Kerch 0.2856 / 0.3033 / 0.3482; Marmara/Bosporus 0.3793 / 0.3920 / 0.5442. Model minus persistence -0.011 to -0.019 in every region (mlp: -0.008 to -0.024, largest Anatolian shelf).
- g-max is not binding: on val day 2021-07-11 (dump, step 0) |g| median 0.106, p90 0.146, p99 0.231, max 0.453 with g-max 0.5; with g-max 1.0 median 0.108, p90 0.151, p99 0.252, max 0.624, 0.09 % of ocean px above 0.45. Val MAE identical to 0.0003. |g| shrinks with step (median 0.106, 0.097, 0.086).
- Partial dependence, mlp epoch 3 (mean g by feature decile, 314 k val px): C - clim moves g most, monotone from +0.15 (lowest decile, z-scored anomaly -0.55) to -0.15 ln CHL/day (+0.75); clim next, -0.045 at the lowest decile to +0.03 for clim above its mean; mld +0.055 at the lowest decile, otherwise within +-0.03; sst +0.025 to -0.035 (peaks near z -0.2, falls at high sst); doy sin +0.045 to -0.05, doy cos -0.04 to +0.02; lat within -0.012 to +0.026; log depth +0.025 (shallow) to -0.017 (deep); so, so ok, sst ok, wind u/v, river dist, dmld/mld within +-0.02. Time-mean g map (178 val days, step 0): -0.02 to -0.05 ln CHL/day over the open basin, +0.05 to +0.15 along the NW shelf coast, Danube mouth and the Azov Sea.
- --ic clim: persistence of the clim-with-observations IC is 0.036-0.042 worse than the diffusion-fill IC at every lead; the trained mlp on top of it is 0.040-0.042 worse than mlp on the fill IC (0.2418 vs 0.2015 at lead 1), gain over its own persistence -0.005 to -0.009.
- Notes: train_ude.py logs gained a "T-a1" column during the session (udeA_mlp_ic_clim log) and the earlier runs lack it; metrics.json "val" is the last epoch, model.pt the best mean-lead epoch. Dump runs: models/ude/udeA_mlp_dump, udeA_mlp_gmax1_dump (eval-only, val stride 40, not part of the table).
- 2026-10-12 (main): cnn_blk_lam0 (dilated CNN v2 recipe, Black Sea, 8 features, crop, batch 8, 12 ep cosine, val every 4th day) finished: val hidden-pixel MAE 0.1975, test 0.1831 (test touched by the trainer's final scoring only), grad ratio 0.529, corr 0.418, peak 15.0 GB. The spatial nODE on the same val days: 0.1772 (blk_final_conv_ctx4_cf). cnn_blk_sl_lam0p3 (semi-Lagrangian physics loss, lambda 0.3) reached 0.2047 after 8 of 12 epochs before an instance reboot; cnn_blk_up_lam0p3 and cnn_blk_sl_lam1 not run yet.

## Deep equilibrium (DEQ) gap filler, IO (deq_* runs, train_pinn.py --model deq, chl_cnn/deq.py; 2026-10-11 to 10-14)
- Cell: GroupNorm dilated CNN (1,2,4,8; 314 k params) with learned step, visible pixels overwritten after each application, z0 = climatology; literature notes in notebooks/deq_literature_notes.md.
- 12 epochs, val hidden MAE: cnn_12ep (dilated, 608 k) 0.1439 / test 0.1472; deq_unroll_k1 (single pass of the cell) 0.1449 / 0.1484; deq_unroll_k6 0.1469; deq_and24 (Anderson, 24 it) 0.1581 / 0.1632 at 386 s/epoch (CNN 31 s). deq_fixed24_30ep (plain iteration): best 0.1477 at epoch 14, run lost to an instance reboot at epoch 15; CNN 60-ep curve is 0.145 at epoch 12 and 0.144 at epoch 14.
- Iteration curve (64 val days of 2001, deq_and24): plain iteration still improving at 64 iterations (0.205 at 24, 0.188 at 64); Anderson best at 24 (0.188) then worse. Solver residual stays 1e-2 to 1e-3 after 24 iterations: the trained map is slowly convergent, not at equilibrium. The k1/k6 unrolled cells diverge when iterated past their training depth.
- Bugs found on the way: (1) autocast caches fp16 weights made in the no-grad solve, detaching the phantom-gradient step (fix: torch.clear_autocast_cache()); (2) Anderson with absolute Tikhonov 1e-4 extrapolated geometrically to fp16 overflow (fix: relative regularisation + per-element safeguard); (3) spectral norm on the zero-init output conv gives NaN (excluded).
- Net: no iterative variant beat a single pass of the same cell; every DEQ run is worse than the explicit CNN at matched epochs and 10x the cost. Consistent with FNO-DEQ (weight-tied unroll = DEQ) and 4DVarNet (no gain past 5 steps) in the notes.
- Instance rebooted 5 times in 3 days (uptime checks); every run longer than ~2 h was lost. train_pinn.py has no resume.

## U-Net campaign (unet_c* runs, 2026-10-14)
- 2026-10-14 (agent): unet_c1_d3w64 (U-Net depth 3 / width 64, 7.72 M params, v2 recipe: full inputs + clim + coords + residual + context 2 + year channel + donor window 15, L1, 20 ep cosine): best val 0.1431 at ep 19, test 0.1463, train loss 0.150 at the end (above val throughout), 28 s/epoch, peak 8.0 GB. Val at ep 20 of the 30-ep d2w64 run was 0.1439; CNN pinn_v2_lam0 at ep 20 of 60: 0.1411. Notes: analyses/unet/NOTES.md.
- 2026-10-14 (agent): unet_c2_ctx4 (c1 + --context 4, 34 input channels): best val 0.1417 at ep 20, test 0.1447 (c1: 0.1431 / 0.1463); train loss 0.148 at the end, still above val; 29 s/epoch, 8.0 GB.
- 2026-10-14 (agent): unet_c3_lr2e3 (c2 + lr 2e-3): best val 0.1431 at ep 19, test 0.1462, end train loss 0.1504 (c2: 0.1417 / 0.1447 / 0.1483); above c2 at every epoch from 3 on, bump at ep 5 (0.1765).
- 2026-10-14 (agent): unet_c4_b4 (c2 + batch 4): best val 0.1429 at ep 20, test 0.1460, end train loss 0.1506; below c2 for epochs 1-6, above it from epoch 7 on; 35.5 s/epoch. Campaign best: c2 0.1417 val / 0.1447 test at 20 ep (CNN pinn_v2_lam0 at ep 20 of 60: 0.1411; final 0.1365). Train loss stayed above val MAE in all four runs. Curves: analyses/unet/learning_curves.png.

## 3D temporal stem (t3d_* runs, train_pinn.py --model temporal3d; 2026-10-14)
- Model: 2 x Conv3d (3,3,3) + GroupNorm/GELU over the per-day stack [9 features, CHL, visible mask] of days t-K..t+K, learned collapse over days, static channels (clim, coords, doy, year) concatenated, then the dilated CNN backbone (width 64). 658-662 k params (CNN 608 k).
- 20 epochs, val hidden MAE / test: t3d_dil_c2 (K=2) 0.1406 / 0.1434, 61 s/epoch; t3d_dil_c4 (K=4) 0.1378 / 0.1398, 77 s/epoch. References at 20 epochs: dilated CNN 0.1411 (epoch 20 of 60; 0.1365 / 0.1434 at 60), best U-Net unet_c2_ctx4 0.1417 / 0.1447.
- K=4 vs K=2 with the 3D stem: -0.0028 val, -0.0036 test. The K=4 run has the lowest test MAE of any model in the workspace (0.1398 vs 0.1434 for the 60-epoch CNN); its val curve is still falling at epoch 20 (0.1384, 0.1381, 0.1380, 0.1378 over epochs 17-20).

## 3D temporal stem (t3d_* runs, train_pinn.py --model temporal3d; 2026-10-14)
- Conv3d stem over the t+-K day window (per-day channels: 9 features, CHL, visible mask; 2 layers width 32, kernel 3x3x3, learned collapse over days) in front of the dilated CNN; +50 k params, 1.5-1.9x the CNN step time.
- 20 epochs, val hidden MAE / test: t3d_dil_c2 (K=2) 0.1406 / 0.1434; t3d_dil_c4 (K=4) 0.1378 / 0.1398. References at 20 epochs: dilated CNN pinn_v2_lam0 curve 0.1411; best U-Net unet_c2_ctx4 0.1417 / 0.1447. CNN 60 epochs: 0.1365 / 0.1390.
- t3d_dil_c4 val still falling at the end of the 20-epoch cosine schedule (0.1384, 0.1381, 0.1380, 0.1378). Val-test gap 0.002 (CNN 0.0025).
- Correction (2026-10-14): the best CNN so far is s3_gdl100_w96 (dilated, width 96, gradient loss G=100, 60 epochs) at 0.1342 val, with a seed replicate at 0.1342; it is in models/history/pinn/cnn/ but was missing from this log. CNN gradient-loss sweep (12 ep, w64/w96): G=3 0.1416/0.1404, G=10 0.1385/0.1381, G=30 0.1382/0.1363, G=100 0.1374/0.1357, G=300 -/0.1369. Gradient loss raised grad corr (0.59 -> 0.62) but left grad ratio at 0.51-0.54.
- 2026-10-14 t3d_dil_c4_60ep (3D stem K=4, dilated w64, 60 epochs, no gradient loss): val 0.1331 (epoch 57) / test 0.1344; grad ratio 0.560, grad corr 0.599 at the best epoch. References: CNN s3_gdl100_w96 0.1342 val; CNN pinn_v2_lam0 0.1365 / 0.1390.
- 2026-10-15 t3d_dil_c4_60ep_gdl100 (same + gradient loss G=100): val 0.1302 (epoch 54) / test 0.1305; RMSE val 0.2263 / test 0.2330; grad ratio 0.543, grad corr 0.633. Versus no gradient loss: val -0.0029, test -0.0039, grad corr +0.034, grad ratio -0.017. Best model in the project so far.
- 2026-10-15 t3d_dil_c4_60ep_gdl100 (same + gradient loss G=100): val 0.1302 (epoch 54) / test 0.1305; grad ratio 0.543, grad corr 0.633. vs no gradient loss: val -0.0029, test -0.0039, grad corr +0.034, grad ratio -0.017. Best model in the project so far.

## Black Sea 3D stem + gradient loss, +- semi-Lagrangian physics loss (blk_t3d_* runs; 2026-10-15/16)
- Config: --store blk --crop --feat-device cpu --val-stride 4 --eval-batch 8, features sst so mlotst sla u_curr v_curr u_wind v_wind, --model temporal3d --t3d-backbone dilated --context 4 --feat-context 4 --grad-weight 100 --clim --coords --residual --vis-weight 0.1 --year-input --donor-window 15, 20 ep cosine, --ckpt-from 5. --batch 4 (batch 8 OOMed at 21.3 GB in the smoke test; cnn_blk_lam0 used batch 8), t3d width 32. Both runs --phys semi-lagrangian (lag-max 20), so val |r| is the same residual form; lambda 0 vs 0.3 (warm-up 3 ep). Queue ~/mind_the_chl_gap/logs/blk_t3d_queue.sh; table/figure analyses/pinn/blk_t3d_runs.py -> analyses/pinn/figures/blk_t3d_val_by_epoch.png.
- blk_t3d_c4_gdl100_lam0: val 0.1627 (epoch 20) / test 0.1552; grad ratio 0.387, corr 0.480; val |r| last epoch cloud 0.1678 / obs 0.4603; 488 s/epoch; peak 14.6 GB. LEGACY-RESUMED: the instance stopped after epoch 15 (val 0.1672); epochs 16-20 continued from model.pt (epoch 15) with the cosine lr set to epoch 15 but a fresh AdamW state, GradScaler and RNG streams (config.json legacy_resume / resume_events). Its comparison with the sl run carries that caveat. Val after the restart: 0.1681, 0.1641, 0.1642, 0.1634, 0.1627.
- blk_t3d_c4_gdl100_sl_lam0p3 (uninterrupted): val 0.1606 (epoch 20) / test 0.1525; grad ratio 0.380, corr 0.482; val |r| last epoch cloud 0.1135 / obs 0.4321; 448 s/epoch; peak 14.6 GB.
- sl minus lam0: val -0.0021, test -0.0027, |r| cloud -0.054, |r| obs -0.028. Single seeds, and lam0 was legacy-resumed (above). The sl curve is below the lam0 curve at 19 of 20 epochs (above it at epoch 8: 0.1936 vs 0.1890). Both curves still fall at epoch 20 (best epoch = last in both).
- References on the same val days (every 4th, 4.20 M hidden px): dilated CNN cnn_blk_lam0 0.1975 / test 0.1831 (12 ep, context 2, batch 8); spatial nODE blk_final_conv_ctx4_cf 0.1772; DOY climatology 0.3604. No NaN in either history.

## Causal 3D-stem forecast, IO (t3d_fc_b8; 2026-10-16)
- train_pinn.py: --context-before B / --context-after A (default --context; window t-B..t+A, --feat-context must equal max(B, A)); --no-chl-today allowed with temporal3d (day t keeps its features, CHL value and mask channels zero); neighbour-day heads get a data loss only when their day is inside the input window (t+1 skipped when A = 0; t-1 empty with --context-clouds none). Docstring section "Causal (forecast) window".
- t3d_fc_b8 = t3d_dil_c4_60ep_gdl100 recipe with --context-before 8 --context-after 0 --feat-context 8 --no-chl-today --context-clouds none, 20 epochs, --ckpt-from 5: T = 9 days, peak 10.05 GB (K=4 symmetric run: 10.05 GB), 77 s/epoch, best epoch 20.
- MAE on all observed day-t pixels (val 7.26 M px, 1458 days / test 8.79 M px, 1463 days): model 0.1285 / 0.1353; persistence (last obs. within t-8..t-1, else DOY clim) 0.1708 / 0.1786; DOY climatology 0.2605 / 0.2958. Hidden-pixel MAE (fixed fake-cloud set, continuity): val 0.1251 / test 0.1321.
- By age of the last observation (val; model / persistence / clim; n): 1 d 0.1242 / 0.1413 / 0.2740 (3.61 M); 2-3 d 0.1282 / 0.1741 / 0.2471 (1.98 M); 4-8 d 0.1311 / 0.2158 / 0.2407 (1.13 M); none 0.1535 / 0.2619 / 0.2619 (0.53 M). Test: 1 d 0.1308 / 0.1503 / 0.3056; 2-3 d 0.1361 / 0.1828 / 0.2827; 4-8 d 0.1401 / 0.2306 / 0.2785; none 0.1659 / 0.3110 / 0.3110.
- The model gets the physical features of day t (sst, so, air_temp, winds, sla, mlotst, currents), so it is a forecast of CHL given same-day physics. Leak test (logs/leak_test_t3d_fc_b8.log): ln CHL of val day 2001-07-21 shifted by +1.0, model re-scored; max |prediction change| 0.0000 for targets t0-1, t0 and t0+9; 0.20-1.10 (mean 0.012-0.22) for t0+1..t0+8.
- The hidden-pixel MAE (0.1251) is below the gap filler t3d_dil_c4_60ep_gdl100 (0.1302, 60 ep) on the same pixels. The two protocols differ there: the gap filler's context days carry the donor's clouds (--context-clouds persist), the forecast's context days carry their real observations.
- Script analyses/forecast/forecast_eval.py -> analyses/forecast/results/t3d_fc_b8.json, figures/t3d_fc_b8_by_age.png.

## train_pinn.py --return (2026-10-15)
- --return / --no-return (default on): resume.pt (full training state) written atomically at every new best val since epoch 1; an interrupted run continues from its best epoch, refuses on changed arguments (ignored: --return, --fig-every, --eval-batch), exits 0 if metrics.json exists; legacy fallback from model.pt + history.json with fresh optimizer. Tested on tiny IO runs (logs/return_test.log): the returned run reproduced the uninterrupted run's epochs 2-3 exactly (0.2455, 0.2429). Queue scripts no longer delete run folders and start the keepalive tmux session.

## Black Sea block-count and window sweep (20-epoch schedule stopped at epoch 12, vs the control's epochs 1-12; 2026-10-16)
- Config: recipe of blk_t3d_c4_gdl100_lam0 (Black Sea 3D stem, dilated w64, G=100, --lam 0 with --phys semi-lagrangian for the residual diagnostic), batch 4, --epochs 20 --stop-epoch 12 --ckpt-from 5 --fig-every 4. Control = the existing blk_t3d_c4_gdl100_lam0 (8 blocks, +-4 days), not retrained: its history epochs 1-12 (before its legacy resume at epoch 15). Test MAE exists only for the sweep runs (the control's test is from epoch 20). New flag train_pinn.py --dilations (comma-separated; default None = DilatedCNN default 1,2,4,8,16,1,2,4; config.json records the effective list). Block counts truncate the default: 4 = 1,2,4,8 ... 8 = default. Receptive field = 1 + 2 (input conv) + 4 (2 x 3x3 spatial in the Conv3d stem) + sum 4d px; 2 km per px. Queue ~/mind_the_chl_gap/logs/blk_sweep2_queue.sh; table/figure analyses/pinn/blk_block_sweep.py. blk_t3d_c4_b8_12ep (12-epoch schedule, killed at ~epoch 8) is a partial duplicate and is excluded.
- blk_t3d_c4_gdl100_lam0 (8 blocks, dilations 1,2,4,8,16,1,2,4, receptive field 159 px = 318 km, window +-4 days): best val within epochs 1-12 0.1768 (epoch 12), val at epoch 12 0.1768 (control, from its 20-epoch history; no epoch-12 test score); grad ratio 0.388, corr 0.470 at the best epoch; 487 s/epoch; peak 14.6 GB; 661,475 params.

## Nowcast t-8..t (t3d_nc_b8) and donor exclusion (2026-10-16)
- train_pinn.py / clouds.py: --donor-exclude D (default 0 = old draws, verified bit-identical for uniform, seasonal and fixed val/test donors): no fake-cloud donor within [t-D, t+D] calendar days of its target day, for training donors (uniform and --donor-window) and the fixed val/test donors. Recorded in config.json (donor_exclude). Also --eval-only --init-from RUN (score a trained model.pt under new arguments, no training). With D = 8, 16 of 1458 val and 18 of 1463 test donors change (old set had 13 / 18 donors within 8 days); new val hidden set 5.01 M px (old 5.00 M); min donor gap 9 d (val), 11 d (test). Eval-only check: t3d_fc_b8 re-scored reproduces its old preds exactly (max |diff| 0).
- t3d_nc_b8 = t3d_dil_c4_60ep_gdl100 recipe with --context-before 8 --context-after 0 --feat-context 8 --context-clouds none --donor-exclude 8, 20 epochs, --ckpt-from 5: T = 9 days, 77 s/epoch, peak 10.05 GB, best epoch 20 (curve 0.1410 ep 1, 0.1263 ep 5, 0.1177 ep 10, 0.1112 ep 15, 0.1090 ep 20; still falling). Heads: t-1 data term empty (context-clouds none), t+1 head dropped (A = 0); day-t head gets the usual --vis-weight 0.1 on its visible pixels.
- Hidden-pixel MAE on the donor-exclude-8 set, val / test (n = 5.01 M / 5.55 M px, all 1458 / 1463 days): nowcast t3d_nc_b8 0.1090 / 0.1126; gap filler t3d_dil_c4_60ep_gdl100 (eval-only, context-clouds persist) 0.1303 / 0.1313; forecast t3d_fc_b8 (eval-only) 0.1251 / 0.1323; persistence 0.1671 / 0.1757; DOY climatology 0.2541 / 0.2920.
- By age of the last observation in t-8..t-1, val (nowcast / gap filler / forecast / persistence / clim; n): 1 d 0.1077 / 0.1397 / 0.1201 / 0.1366 / 0.2664 (2.46 M); 2-3 d 0.1066 / 0.1200 / 0.1244 / 0.1695 / 0.2403 (1.37 M); 4-8 d 0.1092 / 0.1183 / 0.1293 / 0.2129 / 0.2370 (0.80 M); none 0.1257 / 0.1323 / 0.1510 / 0.2595 / 0.2595 (0.38 M).
- Test: 1 d 0.1107 / 0.1368 / 0.1273 / 0.1467 / 0.3006 (2.96 M); 2-3 d 0.1112 / 0.1231 / 0.1328 / 0.1792 / 0.2792 (1.51 M); 4-8 d 0.1143 / 0.1222 / 0.1383 / 0.2279 / 0.2775 (0.78 M); none 0.1341 / 0.1420 / 0.1632 / 0.3090 / 0.3090 (0.30 M).
- Apples-to-apples gap filler vs forecast on the same new set: val 0.1303 vs 0.1251 (forecast lower), test 0.1313 vs 0.1323 (gap filler lower). On val the gap filler is above the forecast only in the 1 d bin (0.1397 vs 0.1201) and below it in 2-3 d, 4-8 d and none; same pattern on test. The age bins are defined from real observations; the gap filler's context days carry the donor's clouds, so its input does not follow these bins.
- Script analyses/forecast/nowcast_eval.py -> results/nowcast_comparison.json, figures/nowcast_comparison.png. Queue ~/mind_the_chl_gap/logs/t3d_nc_queue.sh (one flock over smoke, training, eval-only runs t3d_dil_c4_60ep_gdl100_dx8 / t3d_fc_b8_dx8, analysis).

## Black Sea 4 km store (blk4; 2026-10-16)
- Store ~/mind_the_chl_gap/data/blk4_store_2014_2024/ (local disk, 4.5 GB), built by chl_cnn/store_blk4.py (2x2 block average of the 2 km store; 317 s CPU). Grid 395 x 550 -> 197 x 275 (last 2 km row dropped), ~4.5 km; ocean = >= 2 of 4 fine px ocean: 93,594 -> 23,606 px. CHL observed if >= 2 of 4 fine px observed, value exp(mean ln CHL); features/currents/wind stress nanmean (sst32 float32, others float16); topo, depth, river_dist nanmean; time unchanged (3,997 days).
- Checks vs 2 km (analyses/blk4/store_checks.py, figure analyses/blk4/store_coverage.png): mean daily CHL coverage of ocean px 0.4425 (2 km) vs 0.4522 (4 km); Pearson r of daily coverage 0.9998; 2 of 3,997 days have no 4 km CHL observation (all 2 km days have >= 1). Daily mean ln CHL over observed px, 4 km - 2 km, 5 random days: -0.019, +0.008, +0.015, -0.006, -0.002; all 3,995 days: mean |diff| 0.0083, max 0.175, 12 days > 0.05. All-NaN-on-ocean feature days: 0 new at 4 km (v_wind 5 days at both resolutions). Ocean px never observed on training days: 70 / 23,606 (4 km), 184 / 93,594 (2 km).
- train_pinn.py --store blk4: same as blk (val 2017 2021 / test 2019 2023, sst -> sst32, latitude handling) but --lag-max default 10 cells (blk 20; same distance). store_blk.load_store got an optional path argument. io / blk unchanged.
- Smoke (recipe of blk_t3d_c4_gdl100_*, batch 8, --feat-device gpu, --lam 0.3): crop 177 x 273, peak GPU 10.26 GB allocated (10.83 reserved), 0.136 s/batch (2,542 train days / 8 = 318 batches per epoch).
- Queued (~/mind_the_chl_gap/logs/blk4_t3d_queue.sh, tmux blk4; starts after blk_sweep2_post "post done"): blk4_t3d_c4_gdl100_lam0 and blk4_t3d_c4_gdl100_sl_lam0p3, 20 epochs, batch 8, --feat-device gpu. No results yet.
- 4 km scores are not comparable with 2 km scores (different target: 4 km block-mean ln CHL, different hidden pixels). 4 km baselines reported by the trainer on the same hidden pixels: climatology_hidden (per-pixel mean ln CHL over training days) and, with --clim, doy_climatology_hidden (8-day DOY bins from training days).
- blk_t3d_c4_b4_s12 (4 blocks, dilations 1,2,4,8, receptive field 67 px = 134 km, window +-4 days): best val within epochs 1-12 0.1870 (epoch 12), val at epoch 12 0.1870; test 0.1793 (checkpoint epoch 12); grad ratio 0.376, corr 0.464 at the best epoch; 461 s/epoch; peak 12.2 GB; 365,027 params; n hidden px val 4,200,874 / test 4,377,100.
- blk_t3d_c4_b5_s12 (5 blocks, dilations 1,2,4,8,16, receptive field 131 px = 262 km, window +-4 days): best val within epochs 1-12 0.1829 (epoch 12), val at epoch 12 0.1829; test 0.1754 (checkpoint epoch 12); grad ratio 0.387, corr 0.465 at the best epoch; 443 s/epoch; peak 12.8 GB; 439,139 params; n hidden px val 4,200,874 / test 4,377,100.
- blk_t3d_c4_b6_s12 (6 blocks, dilations 1,2,4,8,16,1, receptive field 135 px = 270 km, window +-4 days): best val within epochs 1-12 0.1779 (epoch 12), val at epoch 12 0.1779; test 0.1682 (checkpoint epoch 12); grad ratio 0.395, corr 0.470 at the best epoch; 450 s/epoch; peak 13.4 GB; 513,251 params; n hidden px val 4,200,874 / test 4,377,100.
- blk_t3d_c4_b7_s12 (7 blocks, dilations 1,2,4,8,16,1,2, receptive field 143 px = 286 km, window +-4 days): best val within epochs 1-12 0.1813 (epoch 12), val at epoch 12 0.1813; test 0.1749 (checkpoint epoch 12); grad ratio 0.388, corr 0.467 at the best epoch; 464 s/epoch; peak 14.0 GB; 587,363 params; n hidden px val 4,200,874 / test 4,377,100.
- blk_t3d_c3_b8_s12 (8 blocks, dilations 1,2,4,8,16,1,2,4, receptive field 159 px = 318 km, window +-3 days): best val within epochs 1-12 0.1754 (epoch 12), val at epoch 12 0.1754; test 0.1702 (checkpoint epoch 12); grad ratio 0.391, corr 0.472 at the best epoch; 396 s/epoch; peak 13.5 GB; 659,427 params; n hidden px val 4,200,874 / test 4,377,100.
- 2026-10-16 blk4 runs finished (both uninterrupted, 20 epochs, batch 8, best epoch 20 in both, 61 s/epoch, peak GPU 10.26 GB, no NaN). Val n = 1,062,322 hidden px (every 4th day, 182 days), test n = 1,101,920.
  - blk4_t3d_c4_gdl100_lam0: val 0.1638 / test 0.1585; RMSE 0.2578 / 0.2615; bias -0.016 / -0.031; grad ratio 0.509 / 0.545, corr 0.625 / 0.648; |r| cloud 0.127 / 0.125, obs 0.210 / 0.209.
  - blk4_t3d_c4_gdl100_sl_lam0p3: val 0.1637 / test 0.1585; RMSE 0.2593 / 0.2623; bias -0.022 / -0.037; grad ratio 0.503 / 0.539, corr 0.625 / 0.647; |r| cloud 0.098 / 0.099, obs 0.194 / 0.196.
  - sl minus lam0: val MAE -0.0001, test 0.0000; |r| cloud -0.029 (val); bias 0.006 more negative. Val curves within 0.01 of each other from epoch 7 on; both still falling at epoch 20. References (same hidden px): per-pixel climatology 0.397 / 0.387, DOY climatology 0.358 / 0.311. Single seed.
  - Table/figure: analyses/pinn/blk4_t3d_runs.py -> analyses/pinn/figures/blk4_t3d_by_epoch.png.
- 2026-10-17 results (20 epochs, batch 8, features on GPU, ~61 s/epoch vs ~470 s at 2 km): blk4_t3d_c4_gdl100_lam0 val 0.1638 / test 0.1585; blk4_t3d_c4_gdl100_sl_lam0p3 (semi-Lagrangian, lambda 0.3) val 0.1637 / test 0.1585. DOY climatology on the same 4 km hidden pixels: val 0.3583 / test 0.3112. Grad ratio/corr 0.509/0.625 (lam 0) and 0.503/0.625 (lam 0.3). Val physics residual |r| under clouds, last epoch: 0.127 (lam 0) vs 0.098 (lam 0.3). Single seed each; best epoch 20 for both; val still at its minimum at the last epoch. Not comparable with 2 km scores.

## Black Sea 4 km, longer training and warm restarts (blk4_*_b16_*; 2026-10-17)
- Batch 16 (batch 24 reserves 22.0 of 23 GB), lr 1e-3 cosine over 50 epochs, all inputs on the GPU, 62 s/epoch. Val / test hidden-pixel MAE (val on every 4th day, n 182 / 183):

| run | val | test | epochs |
|---|---|---|---|
| blk4_t3d_c4_gdl100_lam0_b16_50ep | 0.1561 | 0.1509 | 50 |
| blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep | 0.1546 | 0.1490 | 50 |
| blk4_t3d_c4_gdl100_sl_lam1_b16_stop12 (lambda 1.0, 50-ep schedule stopped at 12) | 0.1978 | 0.1893 | 12 |
| ..._lam0p3_b16_50ep_cont50 (warm restart from lam0p3 best epoch 47, lr 1e-3 cosine, 50 ep) | 0.1499 | 0.1437 | 50 |
| ..._cont50_lr1e-4_cont50 (warm restart from cont50 best epoch 46, lr 1e-4 cosine, 50 ep) | 0.1498 | 0.1430 | 50 |

- lambda 1.0 vs 0.3 at epoch 12 (periodic val): 0.2230 vs 0.1968. |r| cloud 0.084 (lambda 1.0).
- lr 1e-4 restart: periodic val 0.1524 (ep 1), 0.1527 (10), 0.1503 (20), 0.1506 (30), 0.1506 (40), 0.1501 (50); best epoch 42. Change in best val vs its start: -0.0001.
- Day-of-year climatology on the same val days: 0.3583.
- `--continue-from RUN` (train_pinn.py): loads RUN's resume.pt (model, AdamW, GradScaler), fresh cosine schedule from --lr. Single seed throughout.

## Gradient diagnosis, Black Sea 4 km best run (2026-10-17; analyses/blk4/gradient_diagnosis.py)
- Val, blk4_..._cont50_lr1e-4_cont50 (lam0_b16_50ep in brackets). Trainer definition reproduced exactly (ratio 0.5407, corr 0.6577, 2,067,873 lag-1 pairs).
- By pair visibility: both visible 0.952 (0.876), exactly one hidden 0.784 (0.727), both hidden 0.518 (0.509); 92 % of scored pairs are both-hidden.
- By lag 5 / 9 / 18 / 36 / 72 km: ratio 0.541 / 0.602 / 0.676 / 0.748 / 0.815; corr 0.658 / 0.737 / 0.811 / 0.873 / 0.922.
- By distance of hidden pixel to nearest visible: 1 px 0.752, 2-3 0.601, 4-7 0.516, 8-15 0.493, 16+ 0.480.
- By region: shelves 0.66-0.71; Azov 0.52, Marmara 0.54; open sea 0.44-0.46 (lowest ratio, lowest MAE).
- |grad ln CHL| vs |grad SST| on observed px: Pearson 0.14 (1x1), 0.12 (2x2), 0.08 (4x4); vs current speed -0.04 to -0.06. Ratio by |grad SST| decile 0.476 (D1) to 0.558 (D10).
- Same pairs: DOY climatology ratio 0.533 / corr 0.532; persistence (truth t-1, 1.23 M pairs) 0.988 / 0.538.

## Black Sea 4 km + --geo-static (bathymetry, 8 river distances, physics-ok mask; 2026-10-18)
- Same recipe as the b16_50ep runs (single seed each), + 10 static channels. Val / test MAE, grad ratio / corr (val):
  - lam0: 0.1561 / 0.1509, 0.527 / 0.642 -> geo 0.1556 / 0.1478, 0.533 / 0.643 (val -0.0005, test -0.0031)
  - lam0.3: 0.1546 / 0.1490, 0.522 / 0.643 -> geo 0.1536 / 0.1504, 0.521 / 0.642 (val -0.0010, test +0.0014)
- By region (val, geo - base, lam0 / lam0.3): NW shelf -0.0016 / +0.0011; W open +0.0014 / +0.0002; E open -0.0005 / -0.0019; Anatolian +0.0003 / -0.0035; Crimea/Caucasus -0.0067 / +0.0010; Azov -0.0059 / -0.0027; Marmara +0.0069 / -0.0022. Sign agrees between the two lambdas only for E open sea and Azov. Script analyses/blk4/geo_static_by_region.py.
- Best epoch 47 in all four; peak GPU 16.4 GB reserved with and without.

## Black Sea 4 km, wider backbone (blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_w96; 2026-10-18)
- lam0p3_b16_50ep recipe with --width 96 (stem 32; 1.41 M params vs 0.66 M; 19.7 GB reserved; 74 min vs 52 min). Single seed.
- Val / test MAE 0.1527 / 0.1480 vs 0.1546 / 0.1490 (w64): -0.0019 / -0.0010. RMSE 0.2456 vs 0.2492; bias -0.014 both; visible-pixel MAE 0.0971 vs 0.0997; grad ratio / corr 0.530 / 0.649 vs 0.522 / 0.643. Best epoch 47 in both.
- Periodic val at epochs 10 / 20 / 30 / 40: 0.1960 / 0.1729 / 0.1627 / 0.1566 vs 0.2045 / 0.1744 / 0.1641 / 0.1572.
- Smoke memory (batch 16 unless noted): w96 + stem 48 22.3 GB reserved (not run); w128 stem 32 batch 12 17.9 GB.

## UDE stage B (Black Sea 2 km, learned advection factor; runs 2026-10-12, logged 2026-10-18)
- udeB_* = udeA recipe (--store blk --crop --chl-max 100 --ic fill --lead 3, 4 epochs, train stride 4, val 178 t0) with --transport sl --adv-scale scalar --adv-init 0.25. Table: analyses/ude/stageB_summary.py (figures analyses/ude/figures/stageB_*.png).
- Learned current factor alpha ended at 0.167 (transport only), 0.131 (+ mlp source), 0.127 (+ relax).
- Val MAE lead 1/2/3: transport only 0.2125/0.2411/0.2543; transport alpha=1 0.2207/0.2566/0.2745; persistence 0.2142/0.2433/0.2566; mlp + transport 0.2003/0.2332/0.2513 vs mlp without transport (udeA_mlp) 0.2015/0.2356/0.2542; conv without transport (udeA_conv) 0.2007/0.2291/0.2450. n obs 7.47/7.40/7.34 M.
- Grad ratio/corr at lead 1: persistence 0.859/0.261; transport only 0.767/0.273; mlp + transport 0.674/0.307.

## Gradient calibration and input screening, Black Sea 4 km (2026-10-18; analyses/blk4/gradient_calibration.py)
- Best run, val, trainer's lag-1 pairs: OLS slope of Δtruth on Δpred 1.078 [1.059, 1.095] (r 0.658, n 2.07 M); by lag 2/4/8/16 px 1.096/1.101/1.093/1.069; by distance to visible 1.05 (1 px) to 1.11 (16+ px); by region 1.04-1.11.
- Post-hoc high-pass gain pred' = blur + g (pred - blur): best g 1.2 on val (sigma 1 and 2 px); test MAE 0.14300 -> 0.14293 (sigma 1) / 0.14284 (sigma 2); val MAE rises above g = 1.4-1.5.
- |∇field| vs |∇ln CHL| on observed px (val + every 8th train day), Pearson, no coastal band / >8 px from land: SST 0.137 / 0.056; SLA -0.022 / -0.033; current speed 0.107 / 0.017; vorticity 0.158 / 0.055; strain 0.146 / 0.050; ln CHL t-1 (control) 0.708 / 0.580. With ±1 px tolerance: vorticity 0.193, SST 0.154 (no band).
- Gradient direction mean |cos| (2/π = 0.637 for random directions, null not validated on this grid): SST 0.709, SLA 0.716, speed 0.691, vorticity 0.681, strain 0.673, ln CHL t-1 0.792.

## Observed CHL feature motion vs currents, Black Sea 4 km (2026-10-18; analyses/blk4/chl_motion.py)
- Block matching of ln CHL between consecutive days, 16x16 px tiles (≈72 km), search ±6 px, 2014-2024; accepted tiles (peak r ≥ 0.6, not on window edge, non-zero shift significant vs zero shift) 18,110 of 57,428 candidates (16,861 with currents) on 2,296 days. Accepted tiles only in NW shelf (3,067), W open (6,884), E open (6,910); 0 in the other four regions; DJF 538.
- 48 % of accepted tiles have a zero best shift; only 5.7 % of those have |d_cur| < 0.5 px/day. Median |d_obs| 1.43, |d_cur| 1.88, |d_geo| (geostrophic from zos) 1.40 px/day.
- OLS slope of observed shift on current, east / north: 0.57 / 0.51 (R² 0.38 / 0.24); geostrophic 0.74 / 0.59. Shuffled-day current 0.36 / 0.23. Ambiguous tiles counted as zero: 0.21 / 0.19; no significance test: 0.46 / 0.41; T8 tiles 0.43 / 0.34.
- Direction: mean resultant length 0.65 (current, mean offset -2.6°), 0.67 (geostrophic, +10.9°), 0.42 shuffled.
- By region (current slope e / n): NW shelf 0.30 / 0.16, W open 0.56 / 0.55, E open 0.64 / 0.48. By season: DJF 0.36 / 0.33, MAM 0.60 / 0.53, JJA 0.61 / 0.56, SON 0.51 / 0.45.
- Lag 2 vs lag 1 (same tiles, n 5,577): median |d_obs| ratio 1.83; slope 1.28 / 1.12.
- Reference: UDE stage B learned factor 0.13-0.17 (2 km, semi-Lagrangian loss).

## blk4 store: rsntds and friver added (2026-10-18; chl_cnn/store_blk4_extra.py, scripts/fetch_cmems.py --what extra)
- Source CMEMS BS reanalysis (E4R1, v202411): hflux rsntds (net downward shortwave, W m-2) and wflux friver (river freshwater flux, kg m-2 s-1). Files rsntds.npy (float16) and friver.npy (float32), (3997, 197, 275), store day axis. +1.24 GB.
- rsntds: same chain as the other physics (interp to 2 km, 2x2 mean); basin mean ~45 (Dec-Jan) to ~310 W m-2 (day ~180); all-NaN in the source on 2024-10-16..20 (5 days).
- friver: conservative snap of each source mouth cell to the nearest 4 km ocean pixel (78 cells -> 76 px; daily basin total preserved to 3e-8; linear + 2x2 mean would lose 14-26 %). Mean inflow Danube 5,859 m3/s (daily-varying), Dnieper-Bug 1,476, Kizilirmak 539, Dniester 353, Sakarya 196, unnamed sources 1,593; non-Danube rivers follow a smooth repeating annual cycle. No source cell near the river_names.json Rioni mouth (nearest 54 km) or Kerch (84 km).
- Check figure analyses/blk4/figures/extra_vars_check.png.

## Learned Lagrangian tracer, step 1: velocity terms only (Black Sea 4 km; 2026-10-18; chl_cnn/train_tracer.py, tracer.py)
- Model: ln C(x, t+k) = ln C_obs at the backward-RK2 departure point on day t (4 sub-steps/day, daily fields linear in time, bilinear sampling, |u| components clipped at 10 px/day); u_eff = a(x,s) u_rea + b(x,s) u_geo + w(s) tau (complex doy-harmonic coefficient: c = |w|, theta = arg w). a in (0, 1.5), init 0.25; b in (-0.5, 1.5), init 0; w init 0. No source, no CHL input to velocity, no filling. Valid: trajectory never samples (ocean & finite reanalysis current) < 0.99, departure-point observed fraction >= 0.99. Loss L1, leads 1-5 equal weights. Start days need t..t+5 consecutive calendar days (store keeps only days with CHL; 18 gaps). Train 2,449 starts, val 351 (every 2nd val start, 2017+2021). 20 epochs, batch 16, 3,060 Adam steps, 13-15 min/run, peak 7.1 GB. Test years not scored.
- Synthetic checks (chl_cnn/test_tracer.py): uniform velocity displacement error <= 6e-5 px at k = 5; linear-in-time velocity exact; solid-body rotation error max 5e-4 px (n_sub 4, k = 5, 18 px displacement); learned factor recovers 0.997 (true 1.0, init 0.25); learned uniform velocity (3.001, 2.001) vs (3, 2) px/day.
- At init the model equals the x0.25 baseline to 4 decimals (consistency check).
- Val MAE, common set (target observed, model + persistence + x1.0 + x0.25 + clim all valid), leads 1-5; n 1.76 / 1.48 / 1.37 / 1.34 / 1.29 M (rea+geo+wind run):

| predictor | l1 | l2 | l3 | l4 | l5 |
|---|---|---|---|---|---|
| rea (tr_rea) | 0.1601 | 0.1924 | 0.2182 | 0.2356 | 0.2436 |
| rea+geo (tr_rea_geo) | 0.1599 | 0.1921 | 0.2179 | 0.2350 | 0.2433 |
| rea+geo+wind (tr_rea_geo_wind) | 0.1596 | 0.1916 | 0.2174 | 0.2346 | 0.2428 |
| adv x0.25 | 0.1594 | 0.1923 | 0.2186 | 0.2360 | 0.2446 |
| persistence | 0.1640 | 0.1967 | 0.2221 | 0.2395 | 0.2470 |
| adv x1.0 | 0.1703 | 0.2126 | 0.2439 | 0.2664 | 0.2765 |
| DOY clim | 0.3651 | 0.3855 | 0.3722 | 0.3811 | 0.3673 |

(baseline rows: tr_rea_geo_wind's common set; the other runs' sets differ by < 0.5 % of pixels and baselines by <= 0.0002.)
- Gating, best model (rea+geo+wind) minus baseline, leads 1-4: vs persistence -0.0044 / -0.0051 / -0.0047 / -0.0049; vs x0.25 +0.0002 / -0.0007 / -0.0012 / -0.0014. rea only vs x0.25: +0.0005 / -0.0001 / -0.0005 / -0.0005. Single seed; no repeat to size run-to-run noise.
- Own sets (each predictor where valid), lead 1 MAE (n): model 0.1633 (1.90 M), x0.25 0.1626 (1.89 M), x1.0 0.1720 (1.87 M), persistence 0.1838 (2.32 M), clim 0.3580 (3.80 M). Persistence's own set includes Azov and coastal pixels without valid trajectories.
- Learned factors (ocean mean over val days, mean / sd): a 0.160 / 0.023 (rea), 0.171 / 0.026 (rea+geo), 0.169 / 0.015 (rea+geo+wind); b 0.003 / 0.022 and 0.019 / 0.017. a stops moving after epoch 1 (0.15-0.17 every epoch). Spatial a maps differ between the three runs (e.g. July maximum along 43-44 N in rea and rea+geo+wind, on the NW shelf in rea+geo; rea+geo+wind a ~0 in the SE basin in July / October). Seasonal ocean-mean a 0.14-0.21, highest May-July in the rea and rea+geo runs; b 0 to 0.045, highest Aug.
- Wind: c 0.03 (Oct) to 0.17 (Apr) m/s per Pa, theta -131 to -144 deg (clockwise ~140 deg from tau). Median |tau| over ocean 0.037 Pa (p90 0.136) -> median wind term ~0.006 m/s (~0.1 px/day) vs ~0.46 px/day mean |u_eff|.
- Gradient ratio / corr at lead 1, common-set pairs, lag 1 / 4 / 16 px (rea+geo+wind): model 0.896/0.514, 0.968/0.720, 0.984/0.885; x0.25 0.893/0.515, 0.969/0.721, 0.986/0.886; persistence 1.004/0.463, 1.003/0.688, 0.992/0.873; x1.0 0.982/0.401, 1.014/0.641, 1.006/0.859; clim 0.467/0.428, 0.544/0.515, 0.716/0.666 (3.25 / 2.83 / 2.10 M pairs).
- Lead-1 MAE by region, common set (rea+geo+wind / x0.25 / persistence, n): NW shelf 0.2028 / 0.2032 / 0.2104 (308 k); W open 0.1387 / 0.1388 / 0.1425 (600 k); E open 0.1557 / 0.1553 / 0.1593 (797 k); Anatolian 0.2206 / 0.2176 / 0.2254 (30 k); Crimea/Caucasus 0.1812 / 0.1790 / 0.1843 (24 k); Marmara 0.2355 / 0.2244 / 0.2390 (272 px); Azov 0 px (no reanalysis currents).
- Direction vs the 2 km forecast baselines (2026-10-10): same ordering x0.25 < persistence < x1.0 at every lead; learned a 0.16-0.17 vs stage B 0.13-0.17 at 2 km.
- Tables analyses/tracer/results/step1_tables.md; figures analyses/tracer/figures/step1_{mae_vs_lead,gain_vs_lead,factor_map_a,factor_map_b,seasonal_factors,training_curves}.png (analyses/tracer/tracer_report.py).

## Spectral CHL motion and scale-dependent coherence, Black Sea 4 km (2026-10-18; analyses/blk4/spectral_motion.py)
- Method: 32 x 32 px patches (~145 km), stride 16, aligned with the chl_motion 16 px tile grid (91 of 176 positions >= 50 % ocean); day pairs t, t+k (k = 1, 2, 3 consecutive calendar days; 11,877 pairs + 3,941 random-day null pairs); Hann window; masked weighted NCC (Padfield 2012) on patches >= 50 % observed on both days, search +-6/9/12 px, parabolic sub-pixel, accept peak r >= 0.6 and not on edge; phase correlation on >= 90 % clear patches as cross-check. Full run 183 s CPU (pairs 67 s on 6 processes). Assumptions in results/spectral_motion.json.
- Synthetic (known shift U(-5,5) px, n 300 per cloud fraction, independent blob clouds each day): masked NCC median |error| 0.05 / 0.06 / 0.07 / 0.08 / 0.11 / 1.18 px at cloud 0 / 10 / 20 / 30 / 40 / 50 % (Gaussian random field k^-3); real ln CHL field 0.09 / 0.08 / 0.09 / 0.11 / 0.14 / 0.50 px; share > 1 px 0-5 % up to 30 % cloud, 14-15 % at 40 %, 30-56 % at 50 %. Phase correlation 0.12-0.17 px at 0-10 % cloud. Aligned coherence of the shifted GRF 0.998 for 10-144 km, 0.98 at 9.6 km, 0.82 at 9.0 km (Nyquist ring).
- u_eff rebuilt from tr_rea_geo_wind (tracer.VelocityModel imported read-only, CPU, wind term included); rebuilt a matches factor_maps.npz to 2.7e-7.
- Accepted patch-pairs, lead 1 / 2 / 3: 77,039 / 60,979 / 56,334 (of 95,173 / 85,482 / 82,160 candidates; 71 % of lead-1 acceptances are 50-90 % cloudy patches). Median |d_obs| 1.19 / 1.07 / 1.20 px/day; |d_cur| 1.80, |d_geo| 1.38, |u_eff| 0.37 px/day (lead 1).
- OLS slope of observed shift on reference, east / north, lead 1 (n 70,990; geo 66,466); R = mean resultant length of direction difference:

| reference | lead 1 slope e / n | R | lead 2 slope e / n | R | lead 3 slope e / n | R |
|---|---|---|---|---|---|---|
| reanalysis | 0.46 / 0.42 | 0.53 | 0.31 / 0.29 | 0.46 | 0.20 / 0.19 | 0.33 |
| geostrophic | 0.59 / 0.50 | 0.56 | 0.39 / 0.32 | 0.47 | 0.25 / 0.19 | 0.33 |
| tracer u_eff | 1.92 / 1.81 | 0.69 | 1.36 / 1.33 | 0.59 | 0.92 / 0.87 | 0.43 |
| shuffled-day reanalysis | 0.28 / 0.16 | 0.31 | 0.19 / 0.12 | 0.27 | 0.13 / 0.09 | 0.20 |

  (R for u_eff uses only pairs with |u_eff| >= 0.5 px/day: n 21,198 at lead 1 vs 54,830 for reanalysis.)
- Lead 1 by region (reanalysis e / n): NW shelf 0.32 / 0.25, W open 0.47 / 0.48, E open 0.49 / 0.37; other regions n = 0. By season: DJF 0.35 / 0.32 (n 3,126), MAM 0.49 / 0.42, JJA 0.49 / 0.46, SON 0.39 / 0.37. By |d_cur| tercile (edges 1.33, 2.34 px): slopes 0.51 / 0.50 / 0.44 (east); R 0.32 / 0.52 / 0.69. Clear-only patches 0.50 / 0.44 (n 21,952), 50-90 % cloudy 0.45 / 0.41 (n 49,038).
- Lead consistency (same patch and start day accepted at both leads): lead-2 shift on lead-1 shift slope 1.04 / 0.93 (n 44,481; medians 1.14 -> 2.04 px); lead-3 on lead-1 0.94 / 0.82 (n 37,989; R² 0.11 / 0.07).
- Phase correlation vs masked NCC on the same clear patches: lead 1 median |diff| 0.62 px, 67 % within 1 px, slope PC on NCC 0.62 / 0.59 (n 21,896); >= 99 % observed subset 0.53 px, 73 % (n 8,367). Lead 2: 1.57 px, 36 %; lead 3: 3.41 px, 20 %. PC slope on current lead 1 0.37 / 0.33 vs NCC 0.49 / 0.43 on the same patches.
- vs block matching (patch shift vs mean of accepted 16 px tiles inside it, same day): lag 1 n 41,426, median |diff| 0.92 px, 54 % within 1 px, slope 0.58 / 0.49; lag 2 n 31,530, median |diff| 2.47 px, slope 0.26 / 0.20, median |d| 2.34 (patch) vs 3.58 px (tiles).
- Coherence (ring-averaged MSC, >= 90 % clear patches), median, lead 1 aligned / unaligned (n 24,320) / null aligned (n 1,044): 144 km 0.91 / 0.87 / 0.48; 48 km 0.72 / 0.56 / 0.17; 24 km 0.31 / 0.14 / 0.03; 18 km 0.15 / 0.07 / 0.02; 12 km 0.03 / 0.02 / 0.01; 9 km 0.016 / 0.015 / 0.011. 9-18 km band MSC, lead 1: aligned 0.056, unaligned 0.023, null 0.004.
- Null-corrected rho, per-wavelength e-folding lead (aligned): 144 km 21.6 d, 72 km 10.1, 48 km 4.8, 36 km 3.1, 29 km 2.3, 24 km 1.9, 14-21 km 1.6-1.7 d, 9-12 km 2.1-4.2 d (rho there 0.04-0.15, near the null). Implied K = rate / q²: 282 m²/s (144 km) falling monotonically to 46 (16 km) and 6 (9 km). Single K for all wavelengths: free per-wavelength intercepts K 16.5 m²/s, R² 0.935 (residual RMS 0.15-0.38 in ln rho at 18-48 km); strict exp(-K q² t) without intercept K 41 m²/s, R² 0.48. Intercepts (ln rho extrapolated to lead 0) fall from -0.06 (>= 48 km) to -2.4 (9 km). Same pattern in each season and in NW shelf / W open / E open.
- By region at lead 1 (aligned, 36 km ring): NW shelf 0.24 (n 2,513), W open 0.14, E open 0.14, Azov 0.34 (n 259). By season DJF 0.08 (n 804) to JJA 0.17.
- Figures: analyses/blk4/figures/spectral_{synthetic,components,slopes_by_group,slopes_by_lead,angles,vs_blockmatch_phasecorr,coherence_curves,coherence_by_group,decay_fits,maps}.png. Cache results/spectral_motion_patches.npz.

## Resolution check: 4 km gap-filler on 2 km truth, and 2 km lead-1 coherence (2026-10-18; analyses/blk/res_compare.py, analyses/blk/coherence_2km.py)
- Runs: 4 km blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (50+50+50 epochs, batch 16) vs 2 km blk_t3d_c4_gdl100_sl_lam0p3 (20 epochs, batch 4). Same 365 val+test days (every 4th day). Training budgets differ; not controlled.
- Fair set: 2 km hidden pixels whose parent 4 km block was not in the 4 km input (4 km input = 4 km observed [>= 2 of 4 children] & not 4 km hidden; 4 km hidden = observed today & not observed on donor day, same donor days at both resolutions). 8,577,974 hidden 2 km px; excluded 258,480 (3.0 %) parent visible to 4 km model, 6,643 parent not 4 km ocean; kept 8,312,851 (8,210,959 parent 4 km hidden, 101,892 parent 4 km not observed). Stored hidden masks reproduce exactly from store + donors at both resolutions (0 mismatches); own-mask val MAE reproduces metrics.json (0.1606 at 2 km, 0.1498 at 4 km). No NaN / all-zero predictions on scored pixels.
- MAE (ln CHL), fair set, 95 % day-bootstrap CI of difference:

| subset | 2 km model | 4 km nearest | 4 km bilinear | bilinear - 2 km [CI] | n px |
|---|---|---|---|---|---|
| all (365 d) | 0.1596 | 0.1519 | 0.1510 | -0.0086 [-0.0103, -0.0067] | 8,312,851 |
| val | 0.1636 | 0.1552 | 0.1544 | -0.0092 | 4,077,489 |
| test | 0.1557 | 0.1487 | 0.1477 | -0.0080 | 4,235,362 |
| NW shelf | 0.2361 | 0.2279 | 0.2258 | -0.0104 | 1,333,385 |
| W open | 0.1188 | 0.1102 | 0.1099 | -0.0089 | 2,325,023 |
| E open | 0.1191 | 0.1127 | 0.1122 | -0.0069 | 3,389,163 |
| Anatolian | 0.2422 | 0.2228 | 0.2208 | -0.0213 | 239,835 |
| Crimea/Caucasus | 0.1911 | 0.1837 | 0.1810 | -0.0101 | 174,395 |
| Azov | 0.2787 | 0.2732 | 0.2722 | -0.0065 | 667,419 |
| Marmara | 0.2951 | 0.2860 | 0.2846 | -0.0105 | 183,631 |
| dist 1 px (2 km) | 0.0622 | 0.0648 | 0.0621 | -0.0001 [-0.0006, 0.0004] | 547,356 |
| dist 2-3 | 0.0847 | 0.0795 | 0.0780 | -0.0066 | 823,762 |
| dist 4-7 | 0.1106 | 0.1038 | 0.1029 | -0.0078 | 976,102 |
| dist 8-15 | 0.1368 | 0.1287 | 0.1280 | -0.0087 | 1,143,700 |
| dist 16-31 | 0.1615 | 0.1537 | 0.1531 | -0.0084 | 1,332,013 |
| dist >= 32 | 0.2129 | 0.2030 | 0.2023 | -0.0106 | 3,489,918 |

  Nearest at dist 1 px: +0.0027 [0.0022, 0.0031] vs 2 km. All other region / distance CIs exclude 0 except nearest in Azov (upper 0.0000) and Marmara (upper +0.0002).
- Pair differences vs 2 km truth, both ends fair (ratio / corr; 2 km / 4 km nearest / 4 km bilinear): lag 2 km 0.378/0.473, 0.380/0.328, 0.372/0.457 (15.6 M pairs); 4 km 0.438/0.562, 0.464/0.559, 0.441/0.567; 8 km 0.511/0.655, 0.542/0.666, 0.525/0.673; 16 km 0.595/0.744, 0.625/0.761, 0.615/0.766 (12.6 M). ">= 1 end fair" rule: same ordering, values +0.01-0.04.
- Coherence at 2 km: 64 x 64 px patches (~144 km, = 4 km patch grid x 2; 91 of 176 kept), start days every 5th store day (798 lead-1 pairs, 798 null pairs, one per pair), spectral_motion code imported with constants rebound (search +-12 px, overlap min 1024 px). Median MSC lead 1, aligned / unaligned / null aligned (n 4,540 / 4,779 / 665):

| wavelength km | 2 km aligned | 2 km unaligned | 2 km null | 4 km aligned (all days, n 24,320) | 4 km unaligned | matched pairs 2 km / 4 km aligned (n 4,505) |
|---|---|---|---|---|---|---|
| 144 | 0.910 | 0.872 | 0.449 | 0.906 | 0.863 | 0.910 / 0.912 |
| 48 | 0.734 | 0.558 | 0.171 | 0.719 | 0.544 | 0.736 / 0.734 |
| 24 | 0.330 | 0.148 | 0.036 | 0.310 | 0.139 | 0.332 / 0.318 |
| 18 | 0.177 | 0.070 | 0.022 | 0.152 | 0.066 | 0.178 / 0.163 |
| 14.4 | 0.087 | 0.040 | 0.016 | 0.066 | 0.038 | 0.088 / 0.071 |
| 12 | 0.044 | 0.026 | 0.011 | 0.034 | 0.024 | 0.044 / 0.035 |
| 9 | 0.018 | 0.017 | 0.009 | 0.016 | 0.015 | 0.018 / 0.015 |
| 8.0 | 0.014 | 0.011 | 0.008 | - | - | - |
| 6.0 | 0.007 | 0.007 | 0.005 | - | - | - |
| 4.5 | 0.006 | 0.005 | 0.005 | - | - | - |

  2 km and 4 km agree within 0.02 at >= 24 km; at 9-21 km 2 km aligned is 0.003-0.025 higher on the same patch-pairs. At 4.5-8.5 km aligned MSC 0.005-0.015 vs null 0.004-0.008. Matched-pair NCC shifts: median |2 km shift / 2 - 4 km shift| 0.15 (4 km px), 92 % within 0.5 px. Data quality: 2 km chl NaN on 56 % of ocean px-days of the start days (41.8 M of 74.7 M), 48,377 values > 100 set missing.
- Outputs: analyses/blk/results/{res_compare,coherence_2km}.json, coherence cache results/coherence_2km_patches.npz; figures analyses/blk/figures/res_compare_{mae,gradients}.png, coherence_2km.png.

## Learned Lagrangian tracer: seed check and step 2, pointwise source (Black Sea 4 km; 2026-10-18/19; chl_cnn/train_tracer.py --source, tracer.py SourceModel)
- Assumptions: velocity terms initialised from scratch (same seed-0 init as tr_rea_geo_wind; no reason found to warm-start). Source S = 0.5 tanh(MLP), 1x1 convs, hidden 32, zero at init; integrated forward along the backward trajectory, S evaluated at the RK2 sub-step midpoints with the carried value from the sub-step start (explicit). River influence uses a fixed L = 50 km. Discharge per mouth = friver x cell area summed over friver pixels whose nearest river_dist mouth is within 40 km (grouping of analyses/blk4/extra_vars_check.py): Danube 5 px 5,859 m3/s, Dnieper-Bug 3 px 1,476, Kizilirmak 3 px 539, Dniester 3 px 353, Sakarya 2 px 196, Bosporus 1 px 11, Kerch and Rioni 0 px; 59 unassigned px (1,593 m3/s) not used. rsntds 2024-10-16..20 (all NaN) filled with the per-pixel mean of the same doy over 10 other years. Inputs z-scored over ocean pixel-days of every 5th train day (509 days); physics NaN only where reanalysis currents are NaN (so all inputs are finite on valid trajectories; the so_ok flag is < 0.999 at ~0.3 % of samples). Best epoch = min val mean-lead MAE (own set), as step 1. Step-1 comparison ("ref") = tr_rea_geo_wind seed 0 rescored inside each source run's final evaluation, so source and ref share pixels.
- Code checks: --eval-only rescoring of tr_rea_geo_wind with the new code (tr_rea_geo_wind_reeval) reproduces all 1,867 numbers of its metrics["best"] exactly. test_tracer: path recording leaves trajectories bit-identical; constant S = 0.2 over k = 3 days adds 0.6000 (err 6e-8); S = 0 at init (smoke run init equals x0.25). 209 s/epoch, peak 11.7 GB (gradient checkpointing per sub-step).
- Seed check (step 1, rea+geo+wind, seeds 0/1/2, own common sets, n 1.76 / 1.48 / 1.37 / 1.34 / 1.29 M): model MAE spread (max-min) 0.0003 / 0.0002 / 0.0002 / 0.0001 / 0.0003 at leads 1-5; model - x0.25 spread <= 0.0002 (seed values +0.0001..+0.0002 at lead 1, -0.0018..-0.0019 at lead 5). Factor a 0.158-0.169, b 0.019-0.056, wind c 0.10-0.16, theta circular mean -140 / -146 / -160 deg; a-map pattern correlation between seeds 0.92-0.98.
- Val MAE, common set incl. ref (n 1.76 / 1.47 / 1.36 / 1.34 / 1.28 M), leads 1-5:

| predictor | l1 | l2 | l3 | l4 | l5 |
|---|---|---|---|---|---|
| source seed 0 (tr_src, best epoch 1) | 0.1526 | 0.1822 | 0.1995 | 0.2173 | 0.2219 |
| source seed 1 (tr_src_s1, best epoch 3) | 0.1516 | 0.1841 | 0.2009 | 0.2225 | 0.2287 |
| source without ln C (tr_src_noC, best epoch 14) | 0.1574 | 0.1880 | 0.2112 | 0.2280 | 0.2338 |
| step-1 rea+geo+wind (ref) | 0.1595 | 0.1915 | 0.2174 | 0.2346 | 0.2428 |
| x0.25 | 0.1594 | 0.1922 | 0.2186 | 0.2360 | 0.2446 |
| persistence | 0.1639 | 0.1966 | 0.2220 | 0.2395 | 0.2470 |
| DOY clim | 0.3651 | 0.3855 | 0.3722 | 0.3811 | 0.3674 |

- Source minus step-1 (same pixels), leads 1-5: seed 0 -0.0070 / -0.0093 / -0.0179 / -0.0173 / -0.0209; seed 1 -0.0080 / -0.0075 / -0.0165 / -0.0121 / -0.0141; no ln C -0.0021 / -0.0035 / -0.0062 / -0.0066 / -0.0090. Step-1 seed spread 0.0001-0.0003 per lead.
- Training curves: train L1 falls every epoch (0.197 -> 0.185, both seeds); val mean-lead MAE does not: seed 0 is best at epoch 1 (0.2006) and ranges 0.2009-0.2142 over epochs 2-20 (0.2142 at epoch 20); seed 1 0.2031-0.2102. Basin-mean val S per epoch -0.012 to -0.039 ln CHL/day (with ln C); Pearson r over 20 epochs between val mean S and val lead-5 MAE -0.86 (seed 0), -0.67 (seed 1), +0.80 (no ln C, mean S -0.012..+0.001). Val curves are not falling; they fluctuate by up to 0.03 at lead 5 between consecutive epochs (constant lr 3e-3). Reported numbers are single best-epoch checkpoints selected on val.
- Velocity factors: a at best epoch 0.186 (seed 0, epoch 1), 0.215 (seed 1), 0.184 (no ln C) vs 0.158-0.169 in step 1; epochs 11-20 mean a 0.228 / 0.224 / 0.188 vs 0.157-0.166. b -0.021..0.009 (epochs 11-20) vs 0.024-0.052. Wind c 0.11-0.13, theta -127 / 178 / -160 deg. a-map pattern correlation with step-1 seed 0: 0.74 (seed 0), 0.86 (seed 1), 0.97 (no ln C), vs 0.93-0.98 among step-1 seeds. Model speed 0.51-0.57 vs 0.47-0.50 px/day.
- Lead-1 source (prediction - departure value, ln CHL/day), val, mean (n 3.02 M pixel-days): seed 0 DJF -0.003, MAM -0.025, JJA -0.030, SON +0.014, all -0.015; seed 1 -0.002 / -0.032 / -0.034 / +0.031, all -0.014; no ln C -0.000 / -0.011 / -0.011 / -0.000, all -0.007. Regions (seed 0, all seasons) -0.014 (W, E open sea, Anatolian) to -0.020 (NW shelf, Crimea/Caucasus); Azov 0 px. |S| at sub-step samples (n 241 k): median 0.027 / 0.034 / 0.011, p99 0.11 / 0.19 / 0.04, max 0.26 / 0.45 / 0.14; g_max 0.5 not reached (0 samples > 0.45).
- S by input decile (range of decile means, seed 0 / seed 1 / no ln C, ln CHL/day): carried ln C 0.062 / 0.100 / -, monotone decreasing (higher C -> more negative S); rsntds 0.056 / 0.077 / 0.021, decreasing with light; doy sin/cos 0.05-0.08 / 0.01-0.02; log10 MLD 0.050 / 0.068 / 0.016, S lowest at shallow MLD; |tau| 0.033 / 0.036 / 0.025, increasing; SST 0.029 / 0.046 / 0.006, peak near 15 °C; river 0.013 / 0.031 / 0.004; dilution 0.017 / 0.020 / 0.014; static depth / coast distance <= 0.02. Decile means are conditional means over correlated inputs, not isolated effects.
- Gradient ratio / corr at lead 1, lag 1 / 4 / 16 px (3.24 / 2.83 / 2.10 M pairs): seed 0 0.840/0.519, 0.912/0.724, 0.935/0.889; seed 1 0.811/0.523, 0.883/0.728, 0.910/0.890; no ln C 0.896/0.514, 0.968/0.721, 0.985/0.886; ref 0.896/0.513, 0.968/0.719, 0.984/0.885; x0.25 0.893/0.515; persistence 1.004/0.462, 1.003/0.687, 0.992/0.873.
- Lead-1 MAE by region, source seed 0 / ref (n): NW shelf 0.1980 / 0.2027 (308 k); W open 0.1310 / 0.1387 (599 k); E open 0.1484 / 0.1557 (796 k); Anatolian 0.2140 / 0.2203 (29 k); Crimea/Caucasus 0.1714 / 0.1810 (24 k); Marmara 0.2278 / 0.2347 (226 px; x0.25 0.2245); Azov 0 px. Seed 1 -0.005 to -0.010 vs ref in every region; no ln C -0.003 to +0.002.
- Outputs: models/tracer/{tr_rea_geo_wind_s1,_s2,tr_rea_geo_wind_reeval,tr_src,tr_src_s1,tr_src_noC,tr_src_smoke}/ (source runs add source_maps.npz, source_pd.npz; model.pt has a "source" state dict; metrics["final"] holds the ref comparison). Tables analyses/tracer/results/step2_tables.md; figures analyses/tracer/figures/step2_{seed_check,mae_vs_lead,velocity_factors,source_maps,source_pd,training_curves}.png (analyses/tracer/step2_report.py). Notebook panels: tracer_viz.draw_source_by_epoch, draw_source_maps. Logs ~/mind_the_chl_gap/logs/tracer_{seed,s2}_queue*, tr_src*.log. Test years not scored.

## Learned Lagrangian tracer: seed check and step 2, pointwise source (Black Sea 4 km; runs 2026-10-18/19; chl_cnn/train_tracer.py --source, tracer.SourceModel)
- Assumptions: val = 2017 + 2021, every 2nd start day (351 starts); test years not scored. Best epoch = min mean-lead val MAE (own set), selected and reported on the same val days (as step 1). Source-vs-step-1 comparisons use a final re-evaluation of each source run's best model with tr_rea_geo_wind (seed 0) scored as predictor "ref" on identical pixels (metrics.json "final"). Velocity terms of the source runs initialised from scratch (same init as step 1, so at epoch 0 the model equals x0.25); reason: the identifiability check needs velocity learned jointly with the source. 20 epochs as step 1.
- Code: --source off reproduces step 1 bit-identically: tr_rea_geo_wind re-scored with --eval-only --init-from (run tr_rea_geo_wind_reeval) matches its metrics["best"] in all 1,867 numeric fields. test_tracer: path recording leaves trajectories identical; SourceModel outputs 0 at init; constant S = 0.2 over k = 3 days accumulates 0.6000 (err 6e-8).
- Source: S = 0.5 tanh(MLP), 1x1 convs (hidden 32), integrated forward from the departure point along the backward trajectory, S at each RK2 sub-step midpoint (position, time), carried value c at the sub-step start (explicit). Inputs (13): sst32, rsntds, log10 mlotst, dilution max(ΔMLD,0)/MLD per day, |τ|, river influence log10(1 + Σ_r Q_r exp(-d_r / 50 km)) (L fixed), so, so_ok, log10 depth, log10(1 + dist to coast), doy sin/cos, c (z-scored). z-stats: ocean px of every 5th train day (509 days). rsntds all-NaN 2024-10-16..20 filled per pixel with the same-doy mean of the other 10 years. Physics NaN only where reanalysis currents are NaN (14.1 % of ocean px-days; trajectories there are invalid anyway). River grouping: friver pixels assigned to the nearest river_dist mouth if <= 40 km (as extra_vars_check): Danube 5 px 5,859 m3/s, Dnieper-Bug 3 px 1,476, Kizilirmak 3 px 539, Dniester 3 px 353, Sakarya 2 px 196, Bosporus 1 px 11, Kerch and Rioni 0 px; 59 px (1,593 m3/s) unassigned and not used. Gradient checkpointing per sub-step; 209 s/epoch, 71 min/run, peak 11.7 GB.
- Task A, step-1 seeds 0/1/2 (rea+geo+wind), own common sets, MAE l1-l5: seed 0 0.1596/0.1916/0.2174/0.2346/0.2428; seed 1 0.1593/0.1914/0.2172/0.2346/0.2427; seed 2 0.1596/0.1916/0.2173/0.2345/0.2425. Spread (max-min) 0.0001-0.0003 per lead; model - x0.25 spread 0.0001-0.0002 (seed 0 +0.0002/-0.0007/-0.0012/-0.0014/-0.0018). n 1.76/1.48/1.37/1.34/1.29 M. Ocean-mean a 0.158-0.169; b 0.019-0.056; wind c 0.10-0.16 m/s per Pa, θ circular mean -140/-146/-160 deg. a-map pattern correlation between seeds 0.92-0.98.
- Step 2, val MAE on the final common set (incl. ref), leads 1-5 (n 1.76/1.47/1.36/1.34/1.28 M):

| predictor | l1 | l2 | l3 | l4 | l5 |
|---|---|---|---|---|---|
| tr_src (seed 0, best epoch 1) | 0.1526 | 0.1822 | 0.1995 | 0.2173 | 0.2219 |
| tr_src_s1 (seed 1, best epoch 3) | 0.1516 | 0.1841 | 0.2009 | 0.2225 | 0.2287 |
| tr_src_noC (no ln C input, best epoch 14) | 0.1574 | 0.1880 | 0.2112 | 0.2280 | 0.2338 |
| step-1 tr_rea_geo_wind (ref) | 0.1595 | 0.1915 | 0.2174 | 0.2346 | 0.2428 |
| x0.25 | 0.1594 | 0.1922 | 0.2186 | 0.2360 | 0.2446 |
| persistence | 0.1639 | 0.1966 | 0.2220 | 0.2395 | 0.2470 |
| DOY clim | 0.3651 | 0.3855 | 0.3722 | 0.3811 | 0.3674 |

- Source minus step-1 (same pixels), l1-l5: tr_src -0.0070/-0.0093/-0.0179/-0.0173/-0.0209; tr_src_s1 -0.0080/-0.0075/-0.0165/-0.0121/-0.0141; tr_src_noC -0.0021/-0.0035/-0.0062/-0.0066/-0.0090. Step-1 seed spread 0.0001-0.0003.
- Training curves: train L1 (mean lead) falls every few epochs to the end (tr_src 0.1970 -> 0.1848; s1 0.1971 -> 0.1853; noC 0.2056 -> 0.2031, flatter); val does not follow. Val common-set MAE range over the 20 epochs: tr_src l1 0.1509-0.1526, l3 0.1983-0.2100, l5 0.2218-0.2503 (epoch 20 l5 0.2503 > step-1 0.2428); s1 l5 0.2285-0.2418; noC l5 0.2333-0.2415; step-1 l5 0.2427-0.2430. Mean of the last 5 epochs l1-l5: tr_src 0.1517/0.1878/0.2047/0.2320/0.2410; s1 0.1512/0.1858/0.2022/0.2281/0.2367; noC 0.1579/0.1892/0.2128/0.2299/0.2359. The best-epoch gains at leads 4-5 are therefore partly epoch selection on val; leads 1-3 are below step 1 at every epoch of both ln-C runs. Across epochs, val basin-mean S vs val l5 MAE: r -0.86 (tr_src), -0.67 (s1), +0.80 (noC), n 20 epochs each.
- Velocity factors with the source (identifiability check). Ocean-mean a at best epoch: step 1 0.158-0.169; tr_src 0.186, s1 0.215, noC 0.184; mean over epochs 11-20: step 1 0.157-0.166, tr_src 0.228, s1 0.224, noC 0.188. b: step 1 0.019-0.056; source runs -0.008 to 0.022 (epochs 11-20 -0.021 to 0.009). Wind c 0.11-0.13, θ -127 / 178 / -160 deg. Model speed 0.47-0.50 (step 1) vs 0.51-0.57 px/day. a-map pattern correlation with step-1 seeds: tr_src 0.61-0.74, s1 0.75-0.87, noC 0.89-0.97 (step-1 inter-seed 0.92-0.98). Context: the spectral motion analysis (above) gives observed 1-day feature motion slope 0.46/0.42 on reanalysis and 1.92/1.81 on step-1 u_eff; with the ln C source a rises from ~0.16 to ~0.22, still below 0.42-0.46.
- Lead-1 source S (prediction minus departure value, 1 day), val, basin mean (n 3.02 M px-days): tr_src -0.015 (DJF -0.003, MAM -0.025, JJA -0.030, SON +0.014); s1 -0.014 (SON +0.031); noC -0.007 (SON -0.000). By region (tr_src, all seasons) -0.014 to -0.020; Azov n = 0. Maps: negative along the NW-shelf west coast and SE coast in MAM/JJA (to ~-0.10), positive open basin in SON; noC maps within ~±0.02. |S| at sub-step samples (n 241 k): median 0.027 / 0.034 / 0.011, p99 0.11 / 0.19 / 0.04, max 0.26 / 0.45 / 0.14; none above 0.9 g_max.
- Mean S by input decile (tr_src / s1): carried ln C decreasing, +0.008 / +0.016 at ln C -1.6 to -0.054 / -0.084 at ln C +0.05 (range 0.062 / 0.100); rsntds decreasing, +0.015 / +0.035 at 64 W m-2 to -0.041 / -0.043 at 330; doy sin/cos range 0.05-0.08; MLD increasing, -0.05 / -0.07 at the shallowest decile to ~0 at log10 MLD 1.5; |τ| increasing, -0.025 to +0.009 (all three runs); SST peaked near 14-17 °C; river influence, salinity, depth, coast distance ranges 0.008-0.034. noC ranges 0.001-0.025 (largest |τ| 0.025, rsntds 0.021). so_ok < 0.999 in 780-973 samples.
- Gradient ratio / corr at lead 1, lag 1 / 4 / 16 px (3.24 / 2.83 / 2.10 M pairs): tr_src 0.840/0.519, 0.912/0.724, 0.935/0.889; s1 0.811/0.523, 0.883/0.728, 0.910/0.890; noC 0.896/0.514, 0.968/0.721, 0.985/0.886; ref 0.896/0.513, 0.968/0.719, 0.984/0.885; x0.25 0.893/0.515; persistence 1.004/0.462; x1.0 0.982/0.401; clim 0.466/0.427.
- Lead-1 MAE by region, source minus ref (tr_src / s1 / noC): NW shelf -0.0047/-0.0053/-0.0010 (308 k); W open -0.0077/-0.0088/-0.0025 (599 k); E open -0.0073/-0.0083/-0.0023 (796 k); Anatolian -0.0063/-0.0086/-0.0003 (30 k); Crimea/Caucasus -0.0096/-0.0101/-0.0029 (24 k); Marmara -0.0069/-0.0103/+0.0021 (226-269 px); Azov 0 px.
- Outputs: models/tracer/tr_rea_geo_wind_s1, _s2, tr_src, tr_src_s1, tr_src_noC (+ source_maps.npz, source_pd.npz), tr_rea_geo_wind_reeval, tr_src_smoke (20-batch smoke). Tables analyses/tracer/results/step2_tables.md; figures analyses/tracer/figures/step2_{seed_check,mae_vs_lead,velocity_factors,source_maps,source_pd,training_curves}.png (analyses/tracer/step2_report.py). Notebook: chl_cnn/tracer_viz.py gained draw_source_by_epoch / draw_source_maps (progress.ipynb tracer cell).
- Open questions: why val MAE at leads 4-5 swings by up to 0.03 between epochs while train L1 falls smoothly (val mean S correlates with l5 MAE across epochs); whether the a increase with the source reflects a trade-off between transport and the ln C relaxation term.

## Learned Lagrangian tracer, step 1b: decomposed velocity, band-passed loss (Black Sea 4 km; 2026-10-19; train_tracer.py --vel-param decomp --loss-band, tracer.py, analyses/tracer/step1b_report.py)
- Model: u_eff = Σ_j α_j(region, season) u_j, u_j = geostrophic (zos) and ageostrophic (u_rea − u_geo; = u_rea where u_geo invalid, i.e. within 2 px of land / NaN zos) velocities, each split into train-years doy climatology (366 bins, circular Gaussian σ 15 d) + daily anomaly; α free scalars per component × 7 regions × 4 seasons (112), init 0.25 (= x0.25 baseline at init, checked to 4 decimals); no wind. Runs: A = tr1b_band_s0 (loss L1 on 24-150 km band-passed ln CHL), A_s1 (seed 1), B = tr1b_pix_s0 (pixel L1). 20 epochs, 39 s/epoch, ~14 min/run, peak 5.3 GB.
- Band-pass: normalised-convolution DoG (σ = 0.1874 λ/4.5 km for 0.5 response at λ; filter mask = target observed ∩ valid prediction; valid where both normalised weights ≥ 0.5). Response 24-150 km band: peak 0.88 at ~65 km, 0.50 at 24 and ~150 km; 20-70 km band: peak 0.72 at ~35 km. Measured vs analytic on sinusoids max |dev| 0.014 (clear) / 0.038 (30 % clouds + land). Constant field under clouds/land: max |output| 1.4e-6 (naive zero-filled DoG 0.85). Synthetic α recovery (true 0.6 / 1.3, init 0.25): band loss 0.599 / 1.299, pixel loss 0.597 / 1.297. Decomposition sums to u_rea to 9e-8. (results/step1b_checks.json)
- Step-1 behaviour with new flags off: --eval-only re-score of tr_rea_geo_wind identical to the earlier re-score and to its best-epoch metrics (all keys).
- Selection: all three runs' best val epoch = 1 (val criterion flat to ±0.0002 over epochs 1-20 while α kept moving); evaluated both best (epoch 1) and epoch-20 α (from history.json).
- Lead-1 observed patch shift (masked NCC, spectral_motion cache, val-year starts, common set 12,044 pairs / 520 days), OLS slope e / n [95 % CI], R:

| reference | slope east | slope north | R (n_R) |
|---|---|---|---|
| reanalysis | 0.47 [0.45, 0.49] | 0.40 [0.38, 0.42] | 0.51 (9,387) |
| reanalysis x0.25 | 1.89 [1.81, 1.97] | 1.61 [1.50, 1.70] | 0.65 (4,399) |
| step-1 u_eff | 1.99 [1.90, 2.08] | 1.73 [1.62, 1.83] | 0.65 (3,250) |
| A / A_s1 / B (epoch 1) | 1.42 / 1.43 / 1.46 | 1.47 / 1.50 / 1.54 | 0.64 (6.2-6.5 k) |
| A / A_s1 / B (epoch 20) | 1.23 / 1.24 / 1.25 | 1.31 / 1.32 / 1.38 | 0.63 (6.6-6.9 k) |

- Val MAE, common set (n 1.74 M at lead 1 to 1.25 M at lead 5), leads 1 / 3 / 5: A 0.1584 / 0.2171 / 0.2428; B 0.1585 / 0.2171 / 0.2427; x0.25 0.1591 / 0.2188 / 0.2450; persistence 0.1637 / 0.2222 / 0.2474; x1.0 0.1700 / 0.2444 / 0.2774. A − B within ±0.0001 (epoch 1) and ±0.0003 (epoch 20) at every lead. 20-70 km band MAE (n 1.56 M → 1.08 M): A 0.0450 / 0.0618 / 0.0661, B 0.0451 / 0.0618 / 0.0661, x0.25 0.0457 / 0.0631 / 0.0676, persistence 0.0487 / 0.0647 / 0.0677; A − B within ±0.0001.
- Lead-1 gradient calibration in the 20-70 km band (Δtruth on Δpred, lag-1 pairs, n 2.96 M): A 0.579 [0.559, 0.595], B 0.577, epoch-20 0.578-0.580, x0.25 0.569, persistence 0.495, x1.0 0.410.
- α (epoch 20, mean over seasons; A / B): geo_mean W open 0.56 / 0.60, E open 0.54 / 0.45, NW shelf 0.09 / 0.14; geo_anom 0.08-0.16 in all three open/shelf regions; ageo_mean −0.09 to 0.19 (A), −0.02 to 0.13 (B); ageo_anom 0.13-0.20. A vs A_s1 epoch-20 α: max |diff| 0.045 over NW shelf / W open / E open cells, 0.16 including Anatolian / Crimea-Caucasus. Unconstrained cells: Azov (no currents, n = 0, α stays 0.25), Marmara (≤ 2.1 k px; geo components stay 0.25: no valid u_geo), Anatolian / Crimea-Caucasus DJF (< 10 k px). Train-px counts per cell in results/step1b_tables.md.
- Tables analyses/tracer/results/step1b_tables.md; figures analyses/tracer/figures/step1b_{bandpass_response,patch_slopes,shift_scatter,mae_vs_lead,grad_calibration,alpha,training_curves}.png.

## Static base + advected anomaly hypothesis, ln C = B + A (Black Sea 4 km; 2026-10-19; chl_cnn/tracer_static.py, analyses/tracer/static_{synth,zero_tiles,scan,train,anom_motion}.py)
- Assumptions: bases fitted on train days only (doy = 366-bin doy climatology, circular Gaussian sigma 15 d; harm = per-pixel mean + annual + semi-annual harmonics, >= 30 train obs); run = causal mean of observed ln C over calendar days t-20..t-1 (>= 3 obs, else doy); lp = 150 km normalised-convolution low-pass of day t (sigma 6.25 px, weight >= 0.05, else doy). run / lp are held fixed over the lead (B(t+k) = B(t)). Forecast ln C(x, t+k) = B(x, t+k) + A(X_k(x), t) with the unchanged tracer.Integrator. Bases cached in ~/mind_the_chl_gap/data/static_bases (doy fallback on the scan common set: run 0.6-0.9 % of px, lp < 0.001 %). Val = 351 starts (2017 + 2021); test years not scored.
- Test 1, synthetic (static GRF with variance fraction f + AR(1)-per-mode moving GRF at gamma x real patch current, per-wavelength decay rates and lead-0 intercepts from spectral_motion.json, real cloud masks, unchanged NCC / block-matching code; 2,000 patch sequences): frozen-field control recovers slope 1.00 at lead 1, lead-2/lead-1 OLS slope 1.89 (true 1.93). Real ln C on the same sequences: NCC zero share 0.205 / 0.103 / 0.064 (leads 1-3), block-matching zero share 0.43, NCC slope on current 0.41 (l1) / 0.18 (l3), median |shift| ratio l2/l1 1.91, l3/l1 3.36, OLS l2-on-l1 1.08, l3-on-l1 0.74. Synthetic f = 0: zero share 0.06-0.18 / 0.01-0.04 / 0.01-0.03, median ratios 2.0-2.1 / 3.2-3.7, OLS l2-on-l1 1.45-1.60 (true 1.93), BM zero share 0.16 / 0.31 / 0.65 at gamma 1 / 0.75 / 0.5. f >= 0.4: lead-3 zero share >= 0.13, median ratios <= 1.69 / 2.9; f >= 0.6: median ratios 1.15-1.25 / 1.4-1.7, BM zero share >= 0.99. No single f matches all statistics: zero shares by lead and median ratios fall in f 0-0.2; OLS lead-k-on-lead-1 slopes fall in f 0.6-0.8; slope on current at l1 / l3 matches f 0-0.4 depending on gamma. Synthetic NCC acceptance 88-99 % of candidates vs 82 % real. Table results/static_synth.md, figure figures/static_synth_stats.png.
- Test 2, zero-shift tiles (chl_motion lag 1, T16, S6: 8,723 zero / 9,387 moving / 29,027 ambiguous): zero share NW shelf 0.79 (n 3,067), W open 0.39 (6,884), E open 0.35 (6,910), Azov 0.98 (1,249; no reanalysis currents, 1,228 of the zero tiles); without Azov 0.44. Zero share falls with |current| 0.70 (< 0.5 px/day, n 961) -> 0.15 (>= 5, n 551), rises with cloud fraction 0.44 (< 5 %) -> 0.72 (10-20 %) and is 0.91 at depth < 50 m. Median r(ln C day t, long-term-mean pattern) zero 0.45 vs moving 0.36 (val years 0.43 vs 0.26); within region zero minus moving: NW shelf +0.05, W open -0.13, E open -0.06. Masked NCC lead 1 (18,379 zero / 58,660 moving): r_ltm median 0.74 vs 0.60; within region +0.03 / -0.03 / +0.08 (NW / W / E). Results results/static_zero_tiles.json, figures figures/static_zero_tiles_{map,covariates}.png.
- Test 3, scalar-a scan (a 0..1.2, common set fixed across 14 a x 5 bases + step 1 + persistence + bases alone; n 1.72 / 1.40 / 1.26 / 1.21 / 1.13 M px): script reproduces tr_rea_geo_wind metrics.json common-set MAE exactly (step 1 0.1596, x0.25 0.1594 at lead 1). MAE-optimal a: none 0.3/0.2/0.2/0.2/0.1; doy and harm 0.3/0.25/0.2/0.2/0.1; run 0.2/0.1/0.1/0.1/0.1; lp 0.3/0.25/0.25/0.2/0.25. Band (20-70 km) and anomaly-correlation optima 0.1-0.4. Best MAE leads 1-5: doy 0.1584/0.1917/0.2173/0.2355/0.2438, lp 0.1586/0.1912/0.2171/0.2348/0.2432, none 0.1584/0.1910/0.2171/0.2344/0.2432, step 1 0.1587/0.1904/0.2163/0.2335/0.2424. doy@0.3 - step 1 (day bootstrap 95 % CI): -0.0003 [-0.0007, +0.0002] l1, +0.0014 to +0.0031 at l2-l5 (CIs exclude 0); lp@0.3 - step 1 -0.0001 [-0.0003, +0.0001] l1, +0.0008 to +0.0014 l2-l5; lp@0.3 - x0.25 within +-0.0005, all CIs include 0. Base alone: B_lp 0.1848 l1 / 0.2360 l5 (l5 below step 1 0.2424); B_doy 0.36-0.38. At a = 1.0 MAE l1: none 0.1690, doy 0.1678, lp 0.1661. Tables results/static_scan.md, figures figures/static_scan_{landscape,best_a}.png.
- Test 3, short training with the lp base (5 epochs, init 0.25, models/tracer/trs_lp_factor, trs_lp_decomp): factor run (rea+geo+wind) best epoch 0 (init; val own mean-lead flat within 0.0007), ocean-mean a 0.17-0.21 over epochs 1-5 (step 1 0.16-0.17). Decomp run best epoch 1; common set (n 1.84 M l1) model - step 1 -0.0007 / -0.0001 / -0.0002 / +0.0002 / 0.0000. geo_mean alpha epoch 5 W open 0.72 / E open 0.65 vs 0.58 / 0.45 for step-1b tr1b_pix_s0 at epoch 5 (no base); NW shelf 0.24 vs 0.15. Single seed. results/static_train.md.
- Test 3, motion of A (NCC on the spectral-cache pairs, block matching lags 1-2; bases in-sample on train days): raw cache numbers reproduced (77,039 accepted, slope 0.46 / 0.42, BM zero 0.482). NCC zero share l1/l2/l3: ln C 0.239/0.132/0.087; doy 0.185/0.077/0.039; harm 0.188/0.080/0.041; lp 0.252/0.153/0.108; run 0.380/0.253/0.181. Slope on current l1 (e, n): ln C 0.46, 0.42; doy 0.51, 0.47; harm 0.51, 0.47; lp 0.48, 0.46; run 0.38, 0.35; at l3 0.16-0.22 for every field. Lead-2-on-lead-1 OLS 1.04-1.16 (e) for every field; median |shift| l1 -> l3: ln C 1.16 -> 3.31, doy 1.32 -> 3.91, lp 1.11 -> 3.25. BM zero share lag 1: ln C 0.48, doy 0.41, harm 0.42, lp 0.45, run 0.72. Accepted counts fall with base removal (NCC l1 62.6-71.0 k vs 77.0 k). results/static_anom_motion.{json,md}, figure figures/static_anom_motion.png.
- Open questions: why the OLS lead-k-on-lead-1 slopes (real 1.08 / 0.74) sit below the synthetic f = 0 values (1.45-1.60 / 1.66-1.92) while median ratios match f = 0; whether real measurement noise (lower NCC acceptance) is larger than in the synthetic fields.

## Learned tracer diagnostics: loss landscape, synthetic recovery, training dynamics (Black Sea 4 km; 2026-10-19; chl_cnn/tracer_diag.py, analyses/tracer/diag_*.py)
- Assumptions: val years only (2017, 2021), trainer val starts (every 2nd) restricted to those with store days t-1..t+6 consecutive (351 -> 348; needed for the ±1 day current shift); test years not scored. Scalar model u_eff = a × u_rea (no geo / wind / source). Fixed set = target observed and prediction valid for every a in 0..1.2 (step 0.05, + 0.17) and, for the sampling scan, every sampling mode; own set = each a's valid set. Landscape minimum = parabolic vertex through the grid argmin. Band = tracer.BandPass 24-150 km, filter mask = scored set. CIs: 300 day-bootstrap resamples. Moving / zero-shift subsets: lead-1 accepted spectral-motion patches of the start day (|shift| ≥ 1 px / integer peak 0), pixels in both dropped (612 k px-days), applied to all leads. tracer.py / train_tracer.py unchanged (source-run options by in-process patches in diag_src_runs.py).
- Code checks: scan at a = 0.25 / 1.0 on the unfiltered val starts reproduces tr_rea_geo_wind's metrics["best"] adv_x0.25 / adv_x1.0 own-set MAE to 1e-11 with identical n. No non-finite predictions inside valid masks (0 in every scan). Source-run per-day records reproduce history val_model_common to 2e-11.
- Test 1, minimum a by lead, fixed-set pixel MAE [95 % CI] (n fixed px 1.72 M at lead 1, 1.12 M at lead 5):

| variant | l1 | l3 | l5 | mean lead |
|---|---|---|---|---|
| base (u_rea, n_sub 4, bilinear) | 0.31 [0.29, 0.33] | 0.16 [0.14, 0.19] | 0.15 [0.12, 0.17] | 0.19 [0.17, 0.21] |
| currents of day j-1 / j+1 | 0.24 / 0.30 | 0.13 / 0.18 | 0.12 / 0.15 | 0.15 / 0.19 |
| n_sub 1 / 16 | 0.31 / 0.31 | 0.17 / 0.16 | 0.15 / 0.15 | 0.19 / 0.19 |
| nearest / bilinear / bicubic (strict set, 1.43 M px) | 0.28 / 0.31 / 0.30 | 0.17 / 0.18 / 0.18 | 0.17 / 0.16 / 0.16 | 0.20 / 0.20 / 0.20 |
| geo_mean component instead of u_rea | 0.61 | 0.36 | 0.26 | 0.37 |
| base, train starts (every 3rd, n 817) | 0.31 | 0.21 | 0.18 | 0.21 |
| base, moving patches (n 833 k / 445 k px) | 0.41 | 0.21 | 0.16 | - |
| base, zero-shift patches (n 165 k / 85 k px) | 0.18 | 0.12 | 0.12 | - |

  Band-passed MAE minima within 0.05 of the pixel minima (base 0.32 / 0.22 / 0.15); Pearson-r maxima base 0.33 / 0.20 / 0.18 (pixel), 0.36 / 0.26 / 0.20 (band). Flatness, MAE(0.5) - MAE(0.17), fixed set: lead 1 +0.0004 [0.0000, 0.0008], lead 3 +0.0051, lead 5 +0.0072 (band +0.0001 / +0.0031 / +0.0056); moving patches lead 1 -0.0015; zero-shift patches lead 1 +0.0053. MAE at the minimum vs a = 0: lead 1 0.1584 vs 0.1630, lead 5 0.2431 vs 0.2468.
- Test 2, synthetic (truth a = 0.5; anomaly GRF calibrated to the observed patch ring spectrum minus climatology, advected with 0.5 u_rea, n_sub 8, bicubic; real cloud masks; same 348 val starts; fixed-set pixel minimum; trained = scalar a, init 0.25, lr 3e-2, 306 steps on train starts):

| case | l1 | l3 | l5 | mean lead | trained a |
|---|---|---|---|---|---|
| anomaly only | 0.50 | 0.50 | 0.50 | 0.50 | 0.500 |
| + doy climatology base (static) = "none" | 0.46 | 0.48 | 0.49 | 0.48 | 0.486 |
| none + iid noise σ 0.094 | 0.43 | 0.47 | 0.48 | 0.47 | 0.479 |
| none + day-specific correlated noise (1 - rho0) P | 0.43 | 0.47 | 0.48 | 0.48 | 0.479 |
| none + static 30 % of anomaly variance | 0.29 | 0.37 | 0.43 | 0.37 | 0.341 |
| none + static 50 % | 0.18 | 0.10 | 0.05 | 0.09 | 0.073 |
| none + Lagrangian decay (per-wavelength e-folding) | 0.40 | 0.42 | 0.42 | 0.42 | 0.413 |
| none + correlated noise + static 30 % + decay | 0.24 | 0.13 | 0.07 | 0.12 | 0.104 |

  Day-bootstrap 95 % CI width at lead 1 ≤ 0.02 in every case. Inputs (results/diag_prep.json): σ_iid = sqrt(semivariance at lag 1 px) = 0.094, an upper bound (OLS nugget from lags 1-3 px is negative, -0.005); rho0 = exp(lead-0 coherence intercept), capped at 1; synthetic signal spectrum floored at 2 % of the observed power in rings 14-16 (9-10 km). Calibrated ring power within ±8 % of target except ring 2 (+15 %). Synthetic anomaly variance at observed px 0.04 vs real 0.21 (real includes > 145 km anomalies not represented).
- Test 3a, scalar a (= 1.5 sigmoid(θ), Adam, batch 16, all 2,449 train starts, 459 steps): lr 3e-2 converges to 0.165 / 0.170 / 0.172 from a0 0.1 / 0.5 / 1.0; lr 3e-3 reaches 0.169 / 0.249 / 0.526; lr 3e-4 moves ≤ 0.05 (0.112 / 0.457 / 0.953). Final val MAE (mean lead, own) 0.21543-0.21545 for the converged runs. The converged a (0.17) is below the pooled landscape minimum on train starts (0.21-0.22). Check over all train batches: the batch-mean training loss is lowest at a ≈ 0.21-0.225 while its autograd gradient crosses zero at a ≈ 0.17 (gradient +0.0049 at 0.20, loss still falling). Finite differences at a = 0.2 (60 batches): with the valid mask held fixed, autograd matches FD within 3 % at every lead (l5 +0.0177 vs +0.0172); with the mask free to vary with a (as the scored loss), FD is +0.0042 vs autograd +0.0180 at l5 (l3 -0.0022 vs +0.0056); 0.3-2.5 % of pixels change validity between a = 0.19 and 0.21. On the synthetic data, trained a and landscape minima agree within 0.03.
- Test 3b, source runs (rea+geo+wind --source, seed 0, 12 evals × 50 steps of batch 16, val common set, 351 starts; single seed each):

| run | val l5 range | mean abs change between evals l3 / l5 | r(change l5, control) | final a |
|---|---|---|---|---|
| control (lr 3e-3, lr-wind 1e-2) | 0.0223 | 0.0035 / 0.0085 | 1.00 | 0.197 |
| wind lr ×0.1 | 0.0225 | 0.0035 / 0.0086 | 1.00 | 0.186 |
| dropout 0.2 in the source MLP | 0.0192 | 0.0029 / 0.0073 | 0.99 | 0.198 |
| AdamW weight decay 1.0 on the source net | 0.0143 | 0.0022 / 0.0056 | 0.83 | 0.200 |
| g_max 0.1 | 0.0124 | 0.0024 / 0.0043 | 0.84 | 0.190 |
| all lr ×0.1 | 0.0192 (drift) | 0.0017 / 0.0030 | 0.54 | 0.191 |
| tr_src (earlier, 20 × 153 steps) | 0.0285 | 0.0033 / 0.0077 | - | 0.219 |

  Last-eval val l1 / l5: control 0.1517 / 0.2334, wind lr ×0.1 0.1517 / 0.2332, dropout 0.1517 / 0.2316, g_max 0.1 0.1522 / 0.2288, wd 0.1529 / 0.2276, lr ×0.1 0.1531 / 0.2249 (l1 still falling). The eval-to-eval l5 changes coincide across runs that share the batch order (r 0.83-0.99, 11 transitions). Per-day decomposition (control, lead 5): the 10 days with the largest |change| carry 21 % of the summed |change| (9 % of pixels); 62 % of pixel-days change in the direction of the total; leaving out the 10 / 30 most variable days keeps the zig-zag (range 0.0223 -> 0.0202 / similar shape, figure). r(val mean S, val l5) over evals: control -0.84, wd -0.75, dropout -0.78, g_max 0.1 +0.14, lr ×0.1 +0.78.
- Outputs: results/diag_tables.md, diag_summary.json, diag_landscape_*.npz, diag_synth_*.npz / diag_synth_train_*.json, diag_scalar_lr*.json, diag_gradcheck*.json, diag_prep.json; models/tracer/diag_src_*/ (+ perday_val.npz); figures analyses/tracer/figures/diag_{landscape_base,landscape_variants,synth,scalar_train,gradcheck,src_val_curves,src_perday}.png. Logs ~/mind_the_chl_gap/logs/diag_*.log.
- Interpretation (hedged): the trivial settings tested (n_sub, sampling kernel, ±1 day current timing, learning rate given enough steps) move the minimum by ≤ 0.07 and do not reach the 0.4-0.5 range; the landscape itself has its minimum at 0.15-0.31 and falls with lead, a pattern the synthetic reproduces only when a pinned (static) component and decay are present, consistent with the learned factor reflecting the data's mixture of moving and stationary structure rather than an optimisation failure. The lower trained a (0.17 vs loss minimum 0.21) is consistent with the valid-set selection effect (pixels leaving the valid set as a grows), which gradients do not see. The step-2 val swings appear tied to the training batch sequence and shrink with smaller steps / a tighter source bound; they are not carried by a few val days.
- Open questions: why the own-set selection effect lowers the gradient optimum on real data but not on synthetic; whether the val swings follow the seasonal composition of recent batches (a second seed would separate batch order from initialisation).

## Flow gate: oracle motion ceiling and motion-only predictors vs the gap-filler (Black Sea 4 km; 2026-10-20; analyses/flow_gate/check_{a,b}.py)
- Assumptions: val years 2017 / 2021 only (no train years added); crop 177 x 273; CPU only. Full list in analyses/flow_gate/results/flow_gate.md.
- Check A, lead 1, accepted masked-NCC patches (spectral_motion cache; 13,026 val accepted of 77,039 all years; 12,044 patches with scored px on 520 days). Common set: day t+1 observed in patch, day t observed at x, x0.25 / oracle / 0.46-current-shift valid; px counted once per overlapping patch (8,419,723 patch-px). MAE: persistence 0.1426, x0.25 0.1380, oracle NCC shift (bilinear, from day t) 0.1381, 0.46 x patch-mean current shift 0.1392. Oracle gain vs persistence 0.0045 [0.0039, 0.0051], vs x0.25 -0.0001 [-0.0006, 0.0003] (day bootstrap).
  - Inner 16 x 16 px of each patch (2.34 M patch-px): persistence 0.1453, x0.25 0.1404, oracle 0.1324, 0.46 cur 0.1403; oracle gain vs x0.25 0.0080 [0.0072, 0.0087], vs persistence 0.0128 [0.0119, 0.0137].
  - By region, oracle gain vs x0.25: NW shelf -0.0010 [-0.0029, 0.0007] (4,356 patches), W open +0.0002, E open +0.0003, Anatolian -0.0082 [-0.0114, -0.0050] (85 k px), Crimea/Caucasus -0.0075 (54 k px), Marmara +0.0048 (685 px), Azov 0. By season: DJF -0.0021, MAM +0.0003, JJA +0.0002, SON -0.0013 [-0.0026, -0.0000].
  - Coverage: accepted-patch footprints hold 56.5 % of day-t+1 observed px of all 723 val lead-1 pairs; the scored common set 41.3 % (unique px); 525 of 723 pairs have >= 1 accepted patch.
  - Checks: zero-shift oracle = persistence exactly; reversed-sign oracle 0.1572 and row/col-swapped oracle 0.1518 vs persistence 0.1418 on the same px.
- Check B, best gap-filler (pinn/t3d blk4_..._cont50_lr1e-4_cont50), 1,062,322 hidden px on 182 val days; gap-filler MAE 0.1498 (= metrics.json); hidden masks reproduce from store + donors with 0 mismatches. Paired MAE (predictor / gap-filler on the same px, coverage):

| predictor | pred - gap-filler [CI] | pred | gap-filler | coverage |
|---|---|---|---|---|
| (a) persistence, nearest visible day t-1..t-4 | +0.0892 [+0.0789, +0.0998] | 0.2165 | 0.1274 | 52.9 % |
| (b) x0.25 advected, nearest valid past day | +0.0843 [+0.0741, +0.0950] | 0.1964 | 0.1121 | 41.3 % |
| (c) x0.25 advected, nearest valid day t-4..t+4 | +0.0822 [+0.0731, +0.0926] | 0.1994 | 0.1172 | 60.7 % |
| (d) DOY climatology | +0.2086 [+0.1825, +0.2376] | 0.3583 | 0.1498 | 100 % |

  - By distance into the gap 1 / 2-3 / 4-7 / 8-15 / 16+ px, (c) - gap-filler: +0.147 / +0.120 / +0.098 / +0.069 / +0.045 (CIs exclude 0 in every bin); (c) MAE 0.19-0.20 in every bin, gap-filler on the same px 0.047 -> 0.158; (c) coverage 66 % -> 55 %. Same pattern for (a), (b).
  - By source lag, (b) - gap-filler: lag 1 +0.068, 2 +0.082, 3 +0.106, 4 +0.108; (c) future lag 1 +0.071. By region: every predictor x region CI above 0; (b) / (c) cover 0 % of Azov and < 1 % of Marmara (no valid currents).
  - (a) and (b) both defined (427 k px): persistence 0.1966, x0.25 advected 0.1951, gap-filler 0.1116.
  - Gradient ratio / corr, lag 1 px, >= 1 hidden end, predictor defined at both ends: (b) 1.056 / 0.344 vs gap-filler 0.547 / 0.646 (824 k pairs); (c) 1.171 / 0.301 vs 0.539 / 0.644 (1.22 M pairs). Both-hidden pairs: same values within 0.03.
  - Checks: lag -1 advected 0.1745 vs reversed velocity 0.1807 vs persistence 0.1784 (158 k px); lag +1 0.1628 / 0.1689 / 0.1667 (129 k px).
  - Data quality: 0 non-finite gap-filler values; 2,241 hidden px with CHL > 100 kept (train_pinn has no cap); 2 days with no visible px; 1-2 days per lag with a missing calendar context day or advection path; viz.draw_day's store-index rule differs from train_pinn's calendar rule on 18 days (56,262 px); 241 region ties.
- Outputs: analyses/flow_gate/results/flow_gate.{md,json}, check_{a,b}.{md,json}; figures analyses/flow_gate/figures/check_a_oracle_ceiling.png, check_b_distance.png.
- Open questions: why the oracle ties x0.25 on whole patches but leads by 0.008 on the inner 16 x 16 px; why (c)'s gradient ratio exceeds 1 (pairs can mix source days).

## Band-split mean + anomaly tracer, tests 1 and 2 (Black Sea 4 km; 2026-10-19/20; chl_cnn/tracer_band.py, analyses/tracer/band_test1.py, band_train.py, band_report.py)
- Model: ln C(x, t+k) = B(x, t+k) + Σ_b D_b(k) A_b(X^b_k(x), t). B = 150 km low-pass of day t (tracer_static lp), stepped forward by a learned Eulerian source S_B (test 2). A_b = anomaly in bands < 20 / 20-50 / 50-150 km. Band b moves with a_b u_rea and fades as D_b = exp(-k / tau_b): constant tau_b, or tau_b x MLP(strain, |u|, MLD, doy) ("phys"). Val 351 starts (2017 + 2021), test-1 common set (n 1,713,353 px lead 1 to 1,121,757 lead 5); test years not scored.
- Test 1 (no training; x0.25 currents, measured tau 1.67 / 2.38 / 13.77 d): MAE leads 1-5 0.1549 / 0.1837 / 0.2073 / 0.2231 / 0.2305 vs step 1 0.1586 / 0.1903 / 0.2162 / 0.2334 / 0.2423 and persistence 0.1630 / 0.1954 / 0.2209 / 0.2385 / 0.2467. Bands with no motion (a 0, measured tau) 0.1572 / 0.1855 / 0.2088 / 0.2246 / 0.2318. Velocity x0.25 / x0.45 / step-1 u_eff within 0.0005 of each other. Full grid in results/band_test1.md.
- Test 2 (20 epochs, lr 3e-3, batch 16; seeds 0 / 1), best epoch, MAE leads 1-5: const s0 (ep 8) 0.1487 / 0.1763 / 0.1907 / 0.2093 / 0.2129; const s1 (ep 2) 0.1491 / 0.1759 / 0.1926 / 0.2099 / 0.2143; phys s0 (ep 5) 0.1486 / 0.1755 / 0.1900 / 0.2073 / 0.2116; phys s1 (ep 2) 0.1489 / 0.1755 / 0.1916 / 0.2087 / 0.2131. Last-5-epoch means 0.1482-0.1486 / 0.1776-0.1790 / 0.1913-0.1933 / 0.2137-0.2167 / 0.2205-0.2238. Step 2 tr_src on the same px 0.1516 / 0.1809 / 0.1982 / 0.2158 / 0.2206.
  - Model - step 1, best epoch: -0.0095 to -0.0100 (l1) to -0.0281 to -0.0308 (l5), all CIs exclude 0; last 5: -0.0100 to -0.0104 (l1), -0.0185 to -0.0219 (l5), all CIs exclude 0. Model - test 1, last 5: CIs include 0 at leads 2, 4, 5 for const s0 / s1 and phys s0, at leads 4, 5 for phys s1.
  - Phys - const, matched seed: best epoch -0.0020 to +0.0000; last 5 s0 -0.0001 to +0.0016 (l3, l5 CIs above 0), s1 -0.0004 to -0.0018 (CIs below 0).
  - Learned a (< 20 / 20-50 / 50-150 km): 0.67-0.73 / 0.02-0.03 / 0.34-0.37 in all four runs. tau const 0.97-1.14 / 2.30-2.60 / 9.3-10.9 d; phys base 1.35-1.39 / 2.25-2.29 / 12.7-13.3 d. Phys tau multiplier decreases with log strain, |u| and log MLD and towards January in both seeds (figures/band_test2_tau_pd.png); seasonal mean multiplier 20-50 km 1.4-1.9 DJF/SON, 2.0-2.4 MAM/JJA.
  - Lead-1 basin-mean dB -0.012 to -0.046 ln CHL/day, most negative MAM / JJA. r(dB, val l5 MAE) over epochs -0.58 to -0.68 (n 20). Val l5 range over epochs 0.2116-0.2326.
  - Gradient ratio / corr, lead 1, lag 1 px: 0.604-0.620 / 0.563-0.568 (test 1 0.669 / 0.547, step 1 0.896 / 0.511, step 2 0.840 / 0.516, persistence 1.005 / 0.459).
  - Lead-1 MAE by region: every band run below test 1, step 1 and tr_src in every scored region (W open 0.1282-0.1286 vs tr_src 0.1304; NW shelf 0.1938-0.1945 vs 0.1964); Azov 0 px.
- Outputs: results/band_test{1,2}.{md,json}; figures/band_test1_*.png, band_test2_*.png; models/tracer/trb_{const,phys}_s{0,1}. trb_phys_s0 was interrupted by an instance stop after epoch 3 and resumed from state.pt (epoch 4 recomputed).
- 2026-10-20: tracer project shelved; summary in analyses/tracer/TRACER.md (figure figures/tracer_summary.png, analyses/tracer/summary_fig.py). Tracer checkpoints (*.pt) deleted except trb_phys_s0, trb_const_s0 and tr_rea_geo_wind/model.pt (step-1 reference loaded by band_train / band_test1 / static_*); JSON and npz outputs kept for every run.

## 2024 hackweek ("upstream") protocol: our 3D stem vs their public U-Net on identical pixels (IO; 2026-10-20; chl_cnn/store_io_up.py, train_pinn.py --store io_up --fake-mask shift10, analyses/upstream/compare.py)
- Protocol, checked against U-Net_Tutorial cells 6, 9, 12, 28-29: box 32N-11.75S, 42-101.75E (176 x 240); 9193 record days (>= 1 L3 px in box, 1997-10-01..2022-12-31), identical to the time axis of their U-Net_with_CHL_pred.zarr; hidden = finite L3(t) & cloud code 0 at record t+10, one fixed mask per day; t+-1 CHL unmasked; slices inclusive (train 2015-01-01..2018-01-01, val 2018-01-01..2019-01-01, test 2019-01-01..2020-01-01, 366 days each); label z-score over all records (mean -1.7432, std 0.8358, n 75.2 M px). Store hidden mask = their rule recomputed from IO.zarr on 3 days (2015-03-14, 2018-07-01, 2019-12-27): 0 mismatched px. No calendar gaps in 2014-12-20..2020-01-15 (records = calendar days there). Cloud code 0 <=> finite L3 on every store px.
- Deviations: train drops 2018-01-01 (a val day in their slices); hidden px outside our ocean mask dropped (val 2,171 of 1,946,508; test 2,636 of 2,854,295); the year channel of A is outside its training range on val / test (train 0.86-1.14, val 1.29, test 1.43); B's sin/cos uses doy / 365.25 (theirs: days since 1900 / 365); B gets no real-cloud / fake-cloud flags (only the visible mask), no clim / coords / year, so no --residual; physics loss (upwind, lam 0.3) on both, currents enter B only through the loss.
- Runs (60 ep, batch 8, single seed): up_t3d_lam0p3_ours (A, 9 features, +-4 d, clim/coords/residual/year; best ep 52, 23.9 min) and up_t3d_lam0p3_theirin (B, u/v wind, SST, air temp on day t only, CHL t+-1, --doy-input --land-input; best ep 28, 16.5 min). Folder models/pinn/upstream/.

| model | val MAE | test MAE | val / test RMSE | test bias | test grad ratio | test grad corr | test z all-px MAE raw | test z composite | n px val / test |
|---|---|---|---|---|---|---|---|---|---|
| A ours, our inputs | 0.0966 | 0.0998 | 0.1714 / 0.1663 | -0.0001 | 0.722 | 0.719 | 0.0292 | 0.0225 | 1,944,337 / 2,851,659 |
| B ours, their inputs | 0.0961 | 0.0976 | 0.1699 / 0.1646 | -0.0105 | 0.704 | 0.730 | 0.0277 | 0.0220 | same |
| 2024 U-Net (level3_pred) | 0.1067 | 0.1075 | 0.1782 / 0.1743 | -0.0102 | 0.716 | 0.700 | 0.0266 | 0.0238 | same |
| DOY climatology 2015-17 | 0.2605 | 0.2653 | 0.3712 / 0.3874 | -0.0126 | 0.505 | 0.483 | 0.1063 | 0.0589 | same |

  Val z raw / composite: A 0.0208 / 0.0149, B 0.0203 / 0.0149, U-Net 0.0187 / 0.0161, clim 0.0810 / 0.0394. Gradient pairs: val 3.71 M, test 5.48 M. 366 days per split, none with < 5 hidden px; 0 non-finite predictions in the scored px; 0 all-zero prediction days.
- Their published test MAE 0.0301 (MSE 0.00815) recomputed from level3_pred (raw output at finite-target px, 0 error at the 67 % of test px with NaN target): MAE 0.0266, MSE 0.00819. level3_pred carries no output at land / cloud px; UNet_Evaluation (the producer) ran 2024-08-06, the tutorial's evaluated training 2024-08-15, so the two may be different fits of the same model file.
- Default protocol unchanged: --eval-only --init-from t3d_dil_c4_60ep_gdl100 (models/pinn/t3d/up_check_default_eval) gives val 0.130165 / test 0.130548, same n, as the original metrics.json.
- Outputs: analyses/upstream/results/{summary.json,summary.csv,per_day_val.csv,per_day_test.csv,check_store.json}; figures analyses/upstream/figures/{summary_metrics,per_day_mae_test}.png; logs ~/mind_the_chl_gap/logs/upstream/.
- Open questions: why the gap between 0.0266 and 0.0301 in MAE is not matched in MSE; whether B's earlier best epoch (28 vs 52) and A ≈ B reflect the protocol (unmasked t+-1) or seed noise (single seed each).
- 2026-10-20, follow-up: --vis-weight 1.0 (every observed day-t px weighted equally in the pixel loss), everything else as the vis 0.1 runs (L1, grad loss 100, upwind lam 0.3, 60 ep, seed 0): up_t3d_lam0p3_ours_vis1, up_t3d_lam0p3_theirin_vis1. Loss check (train_pinn.data_loss / grad_terms, not changed): the pixel term is then a plain mean over observed px; the gradient term still uses only pairs with >= 1 hidden end (visible-visible pairs excluded); the physics residual pastes visible px before the residual, so the output there gets no physics gradient; neighbour heads are empty under --context-clouds none. Same pixels as the table above; visible = observed ocean px not in their hidden mask (val 2,222,937, test 2,242,444).

| model | vis w | best ep | val / test hidden MAE | val / test RMSE | val / test bias | test grad ratio / corr | val / test z raw | val / test z comp | val / test visible MAE |
|---|---|---|---|---|---|---|---|---|---|
| A ours | 0.1 | 52 | 0.0966 / 0.0998 | 0.1714 / 0.1663 | -0.0085 / -0.0001 | 0.722 / 0.719 | 0.0208 / 0.0292 | 0.0149 / 0.0225 | 0.0328 / 0.0374 |
| A ours | 1.0 | 46 | 0.0971 / 0.1002 | 0.1715 / 0.1669 | -0.0043 / +0.0013 | 0.715 / 0.718 | 0.0209 / 0.0290 | 0.0149 / 0.0225 | 0.0330 / 0.0361 |
| B their inputs | 0.1 | 28 | 0.0961 / 0.0976 | 0.1699 / 0.1646 | -0.0063 / -0.0105 | 0.704 / 0.730 | 0.0203 / 0.0277 | 0.0149 / 0.0220 | 0.0303 / 0.0318 |
| B their inputs | 1.0 | 33 | 0.0966 / 0.0983 | 0.1728 / 0.1670 | -0.0102 / -0.0105 | 0.700 / 0.722 | 0.0195 / 0.0271 | 0.0149 / 0.0220 | 0.0259 / 0.0283 |
| 2024 U-Net | - | - | 0.1067 / 0.1075 | 0.1782 / 0.1743 | -0.0149 / -0.0102 | 0.716 / 0.700 | 0.0187 / 0.0266 | 0.0161 / 0.0238 | 0.0152 / 0.0163 |
| DOY climatology | - | - | 0.2605 / 0.2653 | 0.3712 / 0.3874 | -0.0758 / -0.0126 | 0.505 / 0.483 | 0.0810 / 0.1063 | 0.0394 / 0.0589 | 0.2404 / 0.2718 |

  Hidden px n val 1,944,337 / test 2,851,659, 366 days each; 0 non-finite predictions. Epoch-60 val: A 0.0971 (vis 0.1) / 0.0977 (vis 1.0); B 0.0977 / 0.0978. Figures analyses/upstream/figures/{summary_metrics,per_day_mae_test}.png now include the vis 1.0 rows (panel added: visible-pixel MAE).
- Interpretation (hedged): with equal pixel weights, visible-pixel MAE stays 1.6-2.2x the U-Net's, consistent with the parts of the loss still focused on hidden px (gradient pairs, physics paste), or with capacity spent on the hidden task; hidden-MAE changes are +0.0004 to +0.0007 on single seeds, with no seed-noise estimate on this protocol.

## 2024 hackweek training recipe: our two best IO architectures trained and scored their way (IO; 2026-10-21; train_pinn.py --recipe up2024, store_io_up.recipe_up2024, analyses/upstream/compare_recipe.py)
- Protocol, checked against U-Net_Tutorial cells 6, 9, 10, 12, 15, 18, 28-29: io_up store days and split, with train 2015-01-01..2018-01-01 inclusive (1097 days; 2018-01-01 is also a val day). Their hidden mask with no ocean cut (finite L3 & cloud code 0 at record t+10). Their 13 inputs:
  - u/v wind, SST and air temp, float32 values re-read from IO.zarr (the store copies are float16);
  - sin/cos of float32((days since 1900-01-01) / 365 · 2π);
  - masked CHL, plus CHL of records t±1 (unmasked);
  - land flag (cloud code 2 on 1997-10-01), real-cloud flag (code 1 at t), valid-CHL flag and fake-cloud flag (land + real + valid == 0).
  - Numeric channels are z-scored over the train records [2015-01-01, 2018-01-01), and NaN is set to 0 after scaling.
  - Target: ln L3 z-scored over all 9193 records (mean -1.7432, std 0.8358), NaN → 0.
  - Loss: plain MSE in z over all 42,240 px of each frame, day-t head only, no gradient term, no neighbour-day loss.
  - Optimiser: Adam (eps 1e-7, no weight decay), lr 1e-3 constant, batch 4, max 50 epochs, fp32 (--no-amp). Early stopping on val all-pixel z MSE with patience 10; best-val-loss weights restored; no --ckpt-from delay.
  - Train order: a tf.data-style shuffle buffer of 512 days (their `shuffle(buffer_size=512)`), redrawn each epoch. Seed 0, single seed per config.
- Data checks: the fake-cloud flag equals the store hidden mask on every px (0 mismatched of 9,023,264). The land flag has 15,138 px with 0 finite-L3 px. Real-cloud px with finite L3: 0. Cloud codes {0, 1, 2}.
- Architectures:
  - t3d: temporal3d with a dilated backbone, width 64, 8 blocks, stem 2 × Conv3d(32); 651,299 params. Its 3-day window has 8 slots per day: [4 features, masked CHL, real cloud, valid, fake cloud] filled on day t. CHL t±1 sits in the CHL slot of days t±1, and every other slot is 0. sin, cos and the land flag enter as static channels after the stem.
  - dil96: 2D dilated CNN, width 96, default dilations; 1,343,523 params; the 13 channels as input.
  - Physics runs: upwind residual, λ 0.3 (ramped 0.1 / 0.2 / 0.3 over the first 3 epochs, existing --phys-warmup). It acts on ocean px with a full 5-point ocean stencil (25,788 of 42,240 px/day), finite currents and both neighbour days. Visible day-t px (and observed t±1 px) are pasted in before the residual, so the physics term reaches the output only at hidden and real-cloud ocean px of day t and at unobserved px of the t±1 heads.
- Deviations from their recipe:
  - architecture (by design);
  - 3-day output head (the t±1 heads get no data loss and feed only the physics residual);
  - PyTorch default init instead of Glorot;
  - our RNG for the shuffle buffer;
  - normalisation statistics in float64;
  - physics currents taken from the float16 store as before;
  - the best-val weights are always restored. Older Keras restored them only when early stopping fired, and 3 of 4 runs reached epoch 50 without firing.
  - Train has 1097 days = 274 × 4 + 1, and val / test 366 = 91 × 4 + 2, so the last batch is partial. The pooled pixel mean is reported; tf.keras evaluate weights batch means by sample count, so the two agree.
- Runs: models/pinn/upstream/upr_{t3d,dil96}_lam{0,0p3}, wall time 23.7-34.1 min each. Logs and queue: ~/mind_the_chl_gap/logs/upstream_recipe/.

| model | test z MAE | test z MSE | test hidden MAE full / ocean | test vis MAE | test grad ratio / corr | best / stopped ep |
|---|---|---|---|---|---|---|
| upr_t3d_lam0 | 0.0302 | 0.00795 | 0.1051 / 0.1049 | 0.0197 | 0.706 / 0.725 | 46 / 50 |
| upr_t3d_lam0p3 | 0.0434 | 0.01036 | 0.1154 / 0.1153 | 0.0227 | 0.662 / 0.708 | 48 / 50 |
| upr_dil96_lam0 | 0.0293 | 0.00805 | 0.1049 / 0.1048 | 0.0165 | 0.676 / 0.723 | 40 / 50 (early stop) |
| upr_dil96_lam0p3 | 0.0417 | 0.01023 | 0.1155 / 0.1154 | 0.0212 | 0.675 / 0.708 | 48 / 50 |
| 10-20 B up_t3d_lam0p3_theirin | 0.8097 (lb 0.0277) | 1.61636 (lb 0.01009) | 0.0994 / 0.0976 | 0.0318 | 0.704 / 0.730 | 28 / 60 |
| 10-20 A up_t3d_lam0p3_ours | 0.3664 (lb 0.0292) | 0.35780 (lb 0.01088) | 0.1017 / 0.0998 | 0.0374 | 0.722 / 0.719 | 52 / 60 |
| 2024 U-Net level3_pred | lb 0.0266 | lb 0.00819 | 0.1077 / 0.1075 | 0.0163 | 0.716 / 0.700 | - |
| published 2024 U-Net | 0.0301 | 0.00815 | - | - | - | - |

  Val, same columns: upr_t3d_lam0 0.0224 / 0.00557 / 0.1021 / 0.1020 / 0.0196 / 0.698 / 0.685; upr_t3d_lam0p3 0.0339 / 0.00771 / 0.1152 / 0.1152 / 0.0216 / 0.652 / 0.664; upr_dil96_lam0 0.0216 / 0.00564 / 0.1024 / 0.1022 / 0.0151 / 0.668 / 0.682; upr_dil96_lam0p3 0.0324 / 0.00743 / 0.1150 / 0.1149 / 0.0210 / 0.666 / 0.666; B lb 0.0203 / lb 0.00824 / 0.0986 / 0.0961 / 0.0303 / 0.707 / 0.680; A lb 0.0208 / lb 0.00837 / 0.0991 / 0.0966 / 0.0328 / 0.736 / 0.657; U-Net lb 0.0187 / lb 0.00583 / 0.1068 / 0.1067 / 0.0152 / 0.711 / 0.655.
  - z metric: raw output at every px, 366 × 176 × 240 = 15,459,840 px per split. lb = 0 error at NaN-target px (lower bound), the only convention available for level3_pred (no output at 10.4 M test px). Lower bounds of the recipe runs: test z MAE 0.0266 / 0.0294 / 0.0260 / 0.0292.
  - The 10-20 runs were trained on ocean hidden / visible px only, so their raw output on land and cloud px is unconstrained (hence 0.81 / 0.37).
  - n px, test / val: hidden full 2,854,295 / 1,946,508; hidden ocean 2,851,659 / 1,944,337; visible (ocean) 2,242,444 / 2,222,937; gradient pairs (ocean) 5.48 M / 3.71 M. 0 non-finite predictions from our runs.
  - All-zero prediction: test z MAE 0.2328 / MSE 0.2971, val 0.1761 / 0.2277. This equals mean |target z| over all px, matching the trainer's value.
- Checks:
  - --eval-only --init-from up_t3d_lam0p3_theirin (original flags) reproduces val 0.096096 / test 0.097627 with the same n (models/pinn/upstream/upr_regcheck_theirin_eval).
  - The default IO check --init-from t3d_dil_c4_60ep_gdl100 gives val 0.130165 / test 0.130548 with the same n (models/pinn/t3d/upr_regcheck_default_eval).
  - Smoke runs (1 epoch × 4 configs): 13 input channels, 42,240 loss px per sample, target mean / std as up_norm.json.
  - compare_recipe's all-pixel values match the trainer's metrics.json up_allpx to ~1e-11.
- Outputs: analyses/upstream/results/recipe_{summary.json,summary.csv,per_day_val.csv,per_day_test.csv}; figure analyses/upstream/figures/recipe_summary.png.
- Interpretation (hedged), single seed per config:
  - Without physics, both architectures trained the 2024 way land at the published U-Net's test numbers (z MAE 0.0293-0.0302 vs 0.0301; MSE 0.00795-0.00805 vs 0.00815).
  - Their hidden MAE (0.1048-0.1049 ocean) sits between the U-Net's level3_pred (0.1075) and the 10-20 runs trained with our hidden-focused loss (0.0976-0.0998). This is consistent with the loss and weighting, more than the architecture, setting most of the hidden-pixel gap.
  - Under this all-pixel MSE recipe the upwind physics term at λ 0.3 raised every error metric on both architectures. 3 of 4 runs were still improving at epoch 50.

## 2024 hackweek recipe, stage 1: training length, lr schedule, physics weight (IO; 2026-10-21; train_pinn.py --sched plateau, analyses/upstream/compare_recipe.py --set recipe2)
- Protocol unchanged from the upr_* section above (their data, 13 inputs, all-pixel MSE in z, split, Adam, batch 4). Only length, lr schedule and one physics weight change. Model selection and early stopping use val (2018) all-pixel z MSE only. Test is scored for every run and used for no choice. Single seed (0) per config.
- Code (opt-in, defaults unchanged):
  - `--sched plateau`: ReduceLROnPlateau on the selection score, `--plateau-factor 0.5 --plateau-patience 5 --min-lr 1e-6`, torch's default relative threshold 1e-4. Its state is saved in resume.pt.
  - `--recipe up2024` now keeps explicit `--sched / --epochs / --patience` (defaults stay none / 50 / 10) and accepts `--continue-from`.
  - history.json gains `lr_used` (lr of the epoch's updates; `lr` stays the post-step value) and `source` (warm restarts: source run, best epoch, best val z MSE). config.json gains continue_from_source, final_lr and n_lr_drops.
- Runs (models/pinn/upstream/; flags beyond the upr_* common flags):
  - upr_dil96_lam0_plat, upr_t3d_lam0_plat: `--continue-from <source> --lr 1e-3 --sched plateau --plateau-patience 5 --patience 15 --epochs 200`.
  - upr_t3d_lam0p03: `--phys upwind --lam 0.03`, otherwise as upr_t3d_lam0.
  - upr_dil96_lam0_cos150: `--sched cosine --epochs 150 --patience 30`.
  - Wall time 16 / 13 / 24 / 58 min.

| model | test z MAE full / lb | test z MSE full / lb | test hidden MAE (full) | test vis MAE | test grad ratio / corr | best / stopped ep, final lr |
|---|---|---|---|---|---|---|
| upr_t3d_lam0 | 0.0302 / 0.0266 | 0.00795 / 0.00791 | 0.1051 | 0.0197 | 0.706 / 0.725 | 46 / 50, 1e-3 |
| upr_t3d_lam0_plat | 0.0276 / 0.0250 | 0.00783 / 0.00780 | 0.1031 | 0.0129 | 0.729 / 0.725 | 10 / 25 (+46), 1.25e-4 |
| upr_t3d_lam0p03 | 0.0308 / 0.0271 | 0.00800 / 0.00794 | 0.1053 | 0.0220 | 0.706 / 0.726 | 49 / 50, 1e-3 |
| upr_t3d_lam0p3 | 0.0434 / 0.0294 | 0.01036 / 0.00870 | 0.1154 | 0.0227 | 0.662 / 0.708 | 48 / 50, 1e-3 |
| upr_dil96_lam0 | 0.0293 / 0.0260 | 0.00805 / 0.00801 | 0.1049 | 0.0165 | 0.676 / 0.723 | 40 / 50, 1e-3 |
| upr_dil96_lam0_plat | 0.0289 / 0.0261 | 0.00796 / 0.00792 | 0.1051 | 0.0165 | 0.709 / 0.723 | 8 / 23 (+40), 2.5e-4 |
| upr_dil96_lam0_cos150 | 0.0287 / 0.0252 | 0.00791 / 0.00787 | 0.1034 | 0.0137 | 0.703 / 0.720 | 56 / 86, 3.9e-4 |
| upr_dil96_lam0p3 | 0.0417 / 0.0292 | 0.01023 / 0.00871 | 0.1155 | 0.0212 | 0.675 / 0.708 | 48 / 50, 1e-3 |
| 2024 U-Net level3_pred | - / 0.0266 | - / 0.00819 | 0.1077 | 0.0163 | 0.716 / 0.700 | - |
| published 2024 U-Net | 0.0301 | 0.00815 | - | - | - | - |

  Val, same columns:
  - upr_t3d_lam0 0.0224 / 0.0188, 0.00557 / 0.00553, 0.1021, 0.0196, 0.698 / 0.685.
  - _plat 0.0197 / 0.0172, 0.00549 / 0.00547, 0.1004, 0.0121, 0.716 / 0.685.
  - lam0p03 0.0224 / 0.0189, 0.00561 / 0.00555, 0.1024, 0.0199, 0.690 / 0.686.
  - lam0p3 0.0339 / 0.0211, 0.00771 / 0.00629, 0.1152, 0.0216, 0.652 / 0.664.
  - upr_dil96_lam0 0.0216 / 0.0180, 0.00564 / 0.00561, 0.1024, 0.0151, 0.668 / 0.682.
  - _plat 0.0210 / 0.0177, 0.00558 / 0.00552, 0.1019, 0.0137, 0.697 / 0.683.
  - cos150 0.0212 / 0.0174, 0.00561 / 0.00557, 0.1013, 0.0127, 0.689 / 0.677.
  - lam0p3 0.0324 / 0.0209, 0.00743 / 0.00621, 0.1150, 0.0210, 0.666 / 0.666.
  - U-Net - / 0.0187, - / 0.00583, 0.1068, 0.0152, 0.711 / 0.655.
  - n px per split: z metric 15,459,840; hidden (full) test 2,854,295 / val 1,946,508; visible 2,242,444 / 2,222,937; gradient pairs 5.48 M / 3.71 M. Non-finite predictions from our runs: 0. lb = 0 error at NaN-target px.
- Warm restarts (val z MSE):
  - Epoch-1 val 0.00597 (dil96) / 0.00607 (t3d) vs source best 0.00564 / 0.00557. A fresh model is at 0.022 / 0.018 after epoch 1, and the sources' last 10 constant-lr epochs ranged 0.00564-0.00674 / 0.00557-0.00663.
  - Loaded weights equal the source model.pt (max abs diff 0).
  - Best 0.005579 at restart epoch 8 (dil96; −0.000061 vs source) and 0.005491 at epoch 10 (t3d; −0.000078). Each was the first epoch below the source's best.
  - lr drops: 2 for dil96 (after epochs 14, 20) and 3 for t3d (after 9, 16, 22). dil96: the best epoch came before the first drop. t3d: the best is epoch 10, the first epoch at 5e-4; no epoch after the second or third drop beat it.
  - Early stop at 23 / 25. dil96_plat has one spike at restart epoch 10 (0.0238).
- λ 0.03 vs λ 0 (t3d, val z MSE at epochs 10 / 20 / 30 / 40 / 50): 0.00883 / 0.00709 / 0.00646 / 0.00576 / 0.00564 vs 0.00793 / 0.00717 / 0.00645 / 0.00566 / 0.00562. Best 0.005607 (ep 49) vs 0.005569 (ep 46). Train at epoch 50: data 0.0039, λ·phys 0.0027.
- Cosine 150 vs constant (dil96, same epochs): 0.00898 / 0.00713 / 0.00609 / 0.00563 / 0.00565 vs 0.00823 / 0.00759 / 0.00598 / 0.00564 / 0.00571. Best 0.005611 at epoch 56 (lr 7e-4). Val rose to 0.0060-0.0064 over epochs 60-86 while lr fell to 3.9e-4, then early stop at 86.
- Checks:
  - `--eval-only --init-from upr_t3d_lam0` (original flags, models/pinn/upstream/upr_regcheck_t3d_lam0_eval) reproduces metrics.json val / test z MSE, z MAE and hidden MAE exactly (0.005569385 / 0.007954644).
  - --return with a plateau run (smoke: metrics.json removed, rerun) resumed at the best epoch with the scheduler state.
  - The continuation was bit-identical to the smoke run for epochs 1-3.
  - compare_recipe all-pixel values match the trainers' metrics.json to ~3e-12.
- Outputs: analyses/upstream/results/recipe2_{summary.json,summary.csv,per_day_val.csv,per_day_test.csv,stage1.json}; figures analyses/upstream/figures/recipe2_curves.png (val z MSE and lr vs epoch; y axis clipped at 1.6 × the minimum) and recipe2_summary.png. Queue ~/mind_the_chl_gap/logs/upstream_recipe/queue2.sh, logs there.
- Interpretation (hedged; single seeds, no seed-noise estimate on this protocol):
  - The val gains of the warm restarts (−0.00006 to −0.00008) and of cosine 150 (−0.00003 vs upr_dil96_lam0) are smaller than the epoch-to-epoch jitter of every val curve here (≥ 0.0003).
  - The best epochs came at lr 5e-4-1e-3. Lower lr did not lower val z MSE further.
  - At λ 0.03 the physics run trails λ 0 early (epoch 10) and matches it from epoch 20 on, within jitter.

## Efficient-attention track, Part A: eval-only checks of the ±8-day attention stem (IO; 2026-10-23; chl_cnn/tattn.py EVAL_HOOKS, analyses/io3_attn/{attn_eval,checks}.py)
- Model: io2_tattn8 (attention stem ±8 d, batch 4, 60 epochs; val 0.128594 / test 0.127181). Val n = 1,458 days, 4,998,642 hidden px; no non-finite predictions.
- Hooks are opt-in. With them off, preds are bit-identical to an eval made before the edit and to io2_tattn8's own preds.npz (max |Δ| 0). With the recorder on, preds are also identical.
- A1a, pooling weights (mean over the 4 heads) on val hidden px:
  - share on |offset| > 4: 0.155 (0.032 on days visible at the px, 0.122 on days not visible);
  - share on visible days, all offsets: 0.238; day t's own token: 0.157;
  - each context day is visible (donor-masked) at 10.5-11.1 % of hidden px;
  - mean weight of a day given visible vs not visible: ±1 d 0.449-0.453 vs 0.111-0.122; ±4 d 0.086-0.089 vs 0.030-0.033; ±8 d 0.023 vs 0.014.
- A1b, pooling restricted to |offset| ≤ 4 (renormalised, eval-only): val 0.128594 → 0.130435 (+0.00184).
- A2, optimiser steps (ceil(5540 / batch) per epoch):
  - at about 41.6k steps, the attention run (epoch 30) has best val over epochs 11-30 of 0.130606; the conv ±4 seeds (epoch 60, best over 11-60) have 0.130165 / 0.130113 / 0.130352;
  - every run uses a cosine over 60 epochs, so per step the batch-4 schedule is 2× longer.
- A3, self-attention block bypassed (eval-only): val 1.1714 (bias +1.158). Indicative only (distribution shift).
- A4a, pooling-weight map autocorrelation (98 val days, every 15th; variance-weighted over the 17 offsets):
  - lag where it drops below 0.5: median 3.9 px (IQR 3.4-4.7);
  - visible-pair variant: 4.8 px (4.0-6.3);
  - rho at lags 1 / 2 / 4 / 8 px: 0.85 / 0.69 / 0.49 / 0.32.
  - Score-arm rule (median < 2 px → pixel-query): interpolated arm.
- A4b, val MAE gain of io2_tattn8 over conv ±4, seed 0 [range over seeds 0/1/2]:
  - all: 0.0016 [0.0015-0.0018];
  - past-only age (t−8..t−1): 1 d 0.0012, 2-3 d 0.0013, 4-8 d 0.0027, none 0.0024;
  - two-sided age (nearest real obs in t±8, excluding t): 1 d 0.0012 (n 3.51 M), 2-3 d 0.0020 (1.05 M), 4-8 d 0.0035 (0.35 M), none 0.0018 (0.09 M);
  - distance to nearest visible px: 1 px 0.0008, 2-3 px 0.0015, 4-7 px 0.0024, 8-15 px 0.0025, 16+ px 0.0009 [0.0003-0.0020].
- Outputs: analyses/io3_attn/results/checks.json; figures a1_pool_weights, a2_steps, a4_autocorr, a4_bins, a4_weight_maps (.png). Runs models/pinn/t3d/io3_eval_tattn8_{pre,post,rec,pool4,noself}.

## Efficient-attention track, Part B: gridded patch-token stem, two value arms (IO; 2026-10-23; chl_cnn/ptok.py, train_pinn.py --t3d-stem patch, analyses/io3_attn/{report_b,leak_ptok,ptok_eval}.py)
- Set-up:
  - P = 4 (1,164 ocean patches of a 38 × 46 grid), ±8 days, token dim 64, 4 heads, 2 day-attention layers, interpolated score arm (A4a median lag 3.9 px ≥ 2);
  - recipe otherwise as the conv baseline: batch 8, 60-epoch cosine, single seed 0.
  - The two arms differ in each context day's value at a pixel:
    - arm A (io3_ptok4): the pixel's own visible anomaly;
    - arm B (io3_ptok4_lavg; bridge UI #8 / user #9): Gaussian normalised-convolution local average (σ 1.5 px, r ≤ 5 px), with conf(m) in the pixel term and the softmax over conf(m) > 0.
  - Arm B ran first.
- Checks:
  - `--eval-only --init-from t3d_dil_c4_60ep_gdl100` gives val 0.130165 / test 0.130548, after both edits;
  - leak tests on both smoke models: +1 on day t's hidden px, and +1 on the 7,955 px of t+2 that are observed but hidden by the donor mask, change tokens, token stats, day weights and hidden-px predictions by exactly 0;
  - the positive control (all observed px of t+2) changes predictions by up to 0.64 (B) / 0.74 (A).
- Val / test hidden-px MAE (val n = 1,458 days, 4,998,642 px; test 1,463 days; 0 non-finite predictions):
  - arm A 0.128197 / 0.128637 (best epoch 53);
  - arm B 0.130949 / 0.131993 (53);
  - conv ±4 seeds 0 / 1 / 2: 0.130165 / 0.130113 / 0.130352 val;
  - io2_tattn8 0.128594 / 0.127181;
  - restarted 3-seed conv ensemble 0.127881 / 0.127895.
- RMSE val / test: A 0.2219 / 0.2291; B 0.2257 / 0.2340 (conv seed 0 0.2263 val). Bias val / test: A −0.0172 / −0.0201; B −0.0205 / −0.0234.
- Gradient ratio / corr, val, lags 1 / 2 / 4 px (8.40 M lag-1 pairs):
  - A 0.546/0.633, 0.647/0.734, 0.741/0.815;
  - B 0.527/0.608, 0.632/0.718, 0.731/0.806;
  - conv seed 0 0.543/0.633, 0.643/0.734, 0.737/0.813;
  - io2_tattn8 0.549/0.636, 0.652/0.738, 0.747/0.817.
- Best val over epochs 11-30:
  - A 0.12934 (epoch 29); B 0.13193 (26);
  - conv 0.13283 / 0.13434 / 0.13327 (S30 0.0015);
  - io2_tattn8_s30 0.13061.
- Cost: A 80 s/epoch, B 81 s/epoch at batch 8 (conv 77 at batch 8, attention ±8 286 at batch 4); peak 9.7 / 9.8 GB allocated, 12.8 / 12.9 GB reserved; 807,141 / 807,157 parameters (backbone 605,379; position embedding 111,872).
- Val MAE gain over conv seed 0, by bin:
  - distance 1 / 2-3 / 4-7 / 8-15 / 16+ px: A +0.0004 / +0.0005 / +0.0010 / +0.0020 / +0.0064; B −0.0027 / −0.0029 / −0.0023 / −0.0010 / +0.0055;
  - two-sided age 1 d / 2-3 d / 4-8 d / none: A +0.0021 / +0.0016 / +0.0026 / +0.0003; B −0.0011 / −0.0002 / +0.0008 / −0.0024.
- Score terms (6 val days, softmax-domain pixel-days):
  - A, 137,939 pixel-days: std patch score 2.15, pixel term 0.45, corr 0.70; share of logit variance 0.75 / 0.03; mean "no day" weight over ocean px 0.77, entropy 0.34;
  - B, 740,618 pixel-days: std 1.70 / 1.24, corr 0.07; share 0.61 / 0.33; "no day" 0.42, entropy 1.11.
- Outputs:
  - analyses/io3_attn/results/{io3_ptok4,io3_ptok4_lavg,partb,leak_io3_smoke_ptok4_lavg,leak_io3_smoke_ptok4_own}.json;
  - figures b_curves, b_bins, b_weight_maps_<run>_<date> (.png);
  - queue ~/mind_the_chl_gap/logs/io3/queue.sh.

## Efficient-attention track, io4: token-decoder stem and arm A with the blend off (IO; 2026-10-23; chl_cnn/tokdec.py, train_pinn.py --t3d-stem tokdec [--tokdec-blend], ptok.py HOOKS["no_blend"], analyses/io4_tokdec/)
- Set-up:
  - io4_tokdec: per-day 12 channels (CHL z-anomaly x visible, visible mask, 9 physics x ocean mask, ocean mask) -> shared Conv2d(12, 128, kernel 8, stride 4, padding 2) on ptok's 38 x 46 grid (1,164 ocean patches), LayerNorm + day / position embeddings, 2 temporal self-attention blocks (4 heads), 4-head pooling with day t's token as query over all 17 tokens, [pooled, day-t token] -> Linear 256 -> 4 x 4 x 32, pixel shuffle, Conv 3x3 + GELU; backbone input 32 decoded + 11 day-t + 7 static = 50 ch; no blend, no FiLM.
  - io4_tokdec_blend: + io3 arm-A blend / mass / entropy from the same output tokens (53 ch).
  - Recipe as io3_ptok4: batch 8, 60-epoch cosine, lr 1e-3, wd 1e-4, --ckpt-from 10, single seed 0. D 128 / C 32 fit at batch 8.
  - Assumption: the physics channels of the token input are multiplied by the ocean mask (wind / air temperature are finite over land); day t's backbone channels are unchanged.
- Checks:
  - `--eval-only --init-from t3d_dil_c4_60ep_gdl100` gives val 0.130165 / test 0.130548 after the edits;
  - io3_ptok4 eval-only with hooks off reproduces its preds.npz (max |Δ| 0, same NaN mask);
  - leak tests on both smoke models (val day 2018-02-17): +1 on day t's 7,516 hidden px and +1 on the 7,955 donor-hidden observed px of t+2 change embeddings, tokens, pooling weights, decoded channels, blend / mass / entropy (Job 3) and hidden-px predictions by exactly 0;
  - the positive control (all 10,499 observed px of t+2) changes hidden-px predictions by up to 0.48 (tokdec) / 0.79 (tokdec_blend).
- Val / test hidden-px MAE (val n = 1,458 days, 4,998,642 px; test 1,463 days; 0 non-finite predictions):
  - Job 1, io3_ptok4 eval-only with blend 0, mass 0, entropy 0 (FiLM and day t unchanged): 0.145147 / 0.147829 (+0.01498 / +0.01728 vs conv seed 0). Indicative only (distribution shift);
  - io4_tokdec 0.128208 / 0.128698 (best epoch 56);
  - io4_tokdec_blend 0.127585 / 0.127977 (best epoch 54);
  - io3_ptok4 0.128197 / 0.128637; io2_tattn8 0.128594 / 0.127181; conv ±4 seed 0 0.130165 / 0.130548; 3-seed conv ensemble 0.127881 / 0.127895.
- RMSE val / test: tokdec 0.2214 / 0.2292; tokdec_blend 0.2204 / 0.2274; Job 1 0.2452 / 0.2571. Bias val / test: −0.0160 / −0.0190; −0.0179 / −0.0196; −0.0288 / −0.0317.
- Gradient ratio / corr, val, lags 1 / 2 / 4 px (8.40 M lag-1 pairs):
  - tokdec 0.545/0.631, 0.646/0.735, 0.741/0.817;
  - tokdec_blend 0.548/0.631, 0.649/0.733, 0.745/0.815;
  - Job 1 0.476/0.571, 0.571/0.674, 0.671/0.766;
  - io3_ptok4 0.546/0.633, 0.647/0.734, 0.741/0.815; conv seed 0 0.543/0.633, 0.643/0.734, 0.737/0.813.
- Best val over epochs 11-30 / 11-60: tokdec 0.13108 (ep 25) / 0.12821; tokdec_blend 0.13013 (26) / 0.12759; io3_ptok4 0.12934 / 0.12820; io2_tattn8 0.13061 / 0.12859; conv seeds 0.1328-0.1343 / 0.1301-0.1304 (S30 0.0015, S60 0.00024).
- Cost at batch 8: tokdec 97 s/epoch, 11.0 GB allocated / 14.4 GB reserved, 1,418,787 params (stem 796,704, of which position embedding 223,744); tokdec_blend 108 s/epoch, 11.0 / 15.0 GB, 1,453,637; io3_ptok4 80 s/epoch, 9.7 / 12.8 GB, 807,141.
- Val MAE gain over conv seed 0, by bin (tokdec / tokdec_blend / io3_ptok4):
  - two-sided age 1 d (n 3.51 M): +0.0021 / +0.0026 / +0.0021; 2-3 d (1.05 M): +0.0017 / +0.0021 / +0.0016; 4-8 d (0.35 M): +0.0023 / +0.0038 / +0.0026; none (0.09 M): −0.0001 / +0.0021 / +0.0003;
  - past-only age 1 d / 2-3 d / 4-8 d / none: +0.0021 / +0.0016 / +0.0026 / +0.0007; +0.0027 / +0.0019 / +0.0033 / +0.0026; +0.0022 / +0.0016 / +0.0024 / +0.0011;
  - distance 1 / 2-3 / 4-7 / 8-15 / 16+ px: +0.0005 / +0.0011 / +0.0013 / +0.0014 / +0.0059; +0.0004 / +0.0005 / +0.0009 / +0.0022 / +0.0094; +0.0005 / +0.0005 / +0.0010 / +0.0020 / +0.0063.
- Pooling weights (mean over heads; 1,458 val days x 1,164 ocean patches):
  - tokdec: day t 0.110, t−1 0.185, t+1 0.155, |offset| > 4 0.164; patches with any visible px on a day vs none: t−1 0.422 vs 0.105, t+1 0.370 vs 0.082, ±8 0.014-0.015 vs 0.013-0.021;
  - tokdec_blend: day t 0.082, t∓1 0.122 / 0.119, |offset| > 4 0.263;
  - weight-map autocorrelation on the patch grid (98 val days, every 15th) drops below 0.5 at 1.85 patches ≈ 7.4 px (IQR 1.65-2.06 patches) for tokdec, 1.69 patches for tokdec_blend; rho at 1 / 2 / 4 / 8 patches (tokdec) 0.70 / 0.46 / 0.25 / 0.06.
- Decoded context channels (6 val days x 32 channels): mean |Δ| between adjacent ocean px across patch edges / within patches, median 1.12 (q10-q90 1.01-1.32) for tokdec, 1.06 (0.96-1.17) for tokdec_blend.
- Outputs:
  - analyses/io4_tokdec/results/{io4_summary,io4_tokdec,io4_tokdec_blend,io4_eval_ptok4_noblend,io4_eval_ptok4_repro,leak_io4_smoke_tokdec,leak_io4_smoke_tokdec_blend}.json;
  - figures io4_curves, io4_bins, io4_pool_weights, io4_decoded_2001-03-02, tokdec_schematic (.png);
  - recorder data ~/mind_the_chl_gap/data/io4/<run>_maps_tokdec.npz; queue ~/mind_the_chl_gap/logs/io4/queue.sh (checks.sh before it).
- Interpretation (hedged; single seeds):
  - Without the blend, io4_tokdec matches io3_ptok4 on val (+0.00001) and test (+0.00006), within S60. The 1-px gradient and the 1-day and 2-3-day age bins are within 0.0002 of arm A. In these numbers, fine detail from neighbouring days does not appear to be lost through the tokens.
  - With the blend, Job 3 is 0.00061 / 0.00066 below arm A on val / test, about 2.5 x S60. It is below the 3-seed conv ensemble on val (−0.0003) and slightly worse on test (+0.0001). Its largest bin gains are at 16+ px and at "no obs within ±8 d".
  - Job 1's +0.017 rise shows that arm A's backbone depends on the blend inputs. Because of the distribution shift, it does not measure what the blend contributes.

## Global input ablation of three IO gap-fillers (io5 Job 1; eval-only, val; chl_cnn/ablate.py, train_pinn.py --ablate, analyses/io5_ablation/)
- Logged 2026-10-24 13:18 by the io5 queue (logs ~/mind_the_chl_gap/logs/io5/).
- Models: conv ±4 `t3d_dil_c4_60ep_gdl100`, `io3_ptok4` (patch tokens, arm A), `io4_tokdec_blend`. Val, all 1,458 days, fixed hidden px; each model's Δ is against its own eval-only reproduction (io5_eval_<model>_base).
- Groups, one at a time, on every window day (context days and day t): SST, salinity, MLD, SLA, currents (u + v), winds (u + v), air temp, all 9 physics; context-day CHL (value and visible mask 0 on every day ≠ t); day-t CHL (value and mask 0 on day t); static channels: DOY climatology, lat/lon, depth, year.
- Replacements: physics `mean` = 0 in the z-scored input (the training mean); `doyclim` = the field's per-pixel day-of-year climatology over the training days (circular ±15-day window, NaN-aware), z-scored with the trainer's feature stats, NaN → 0 (cache ~/mind_the_chl_gap/data/io5/doyclim_io.npz). Static channels → their mean over ocean px of the training days.
- Assumptions:
  - `clim` replaces the static climatology channel and the per-day climatology the token stems subtract from CHL (clim3d). `clim_static` replaces the static channel only (token models).
  - The --residual output base (climatology added to the output) is not an input and is unchanged. The sin/cos DOY channels were not ablated.
  - Window days missing from the store stay 0 under doyclim, as the trainer gives them.
  - Targets, hidden / observed masks and the residual diagnostics are untouched.
- Checks: `--eval-only --init-from t3d_dil_c4_60ep_gdl100` (default flags, after the edit) val / test 0.130165 / 0.130548. Each ablation run checks its first val batch: ablated slots hold the replacement (doyclim against an independent numpy reference) and every other input value is unchanged (exact). 68 ablation evals; first-batch input check passed on 68; rows with non-finite predictions on hidden px: 0

| model | val MAE | RMSE | bias | grad ratio / corr lag 1 | n days | n hidden px |
|---|---|---|---|---|---|---|
| conv ±4 (t3d_dil_c4_60ep_gdl100) | 0.130165 | 0.2263 | -0.0214 | 0.543 / 0.633 | 1458 | 4,998,642 |
| patch tokens, arm A (io3_ptok4) | 0.128197 | 0.2219 | -0.0172 | 0.546 / 0.633 | 1458 | 4,998,642 |
| token decoder + blend (io4_tokdec_blend) | 0.127585 | 0.2204 | -0.0179 | 0.548 / 0.631 | 1458 | 4,998,642 |

ΔMAE (val hidden-px MAE, ln CHL; ablated − unablated), training-mean replacement:

| group | conv | ptok4 | tdb |
|---|---|---|---|
| SST | +0.0180 | +0.0100 | +0.0087 |
| salinity | +0.0007 | +0.0010 | +0.0007 |
| MLD | +0.0000 | +0.0010 | +0.0008 |
| SLA | +0.0007 | +0.0007 | +0.0008 |
| currents (u, v) | +0.0041 | +0.0034 | +0.0042 |
| winds (u, v) | +0.0015 | +0.0043 | +0.0030 |
| air temp | +0.0123 | +0.0023 | +0.0018 |
| all 9 physics | +0.0327 | +0.0229 | +0.0252 |
| context-day CHL | +0.0348 | +0.0351 | +0.0352 |
| day-t CHL | +0.0144 | +0.0150 | +0.0149 |
| DOY clim (all paths) | +0.4151 | +0.4451 | +0.4382 |
| DOY clim (static ch only) | - | +0.1504 | +0.1645 |
| lat/lon | +0.0060 | +0.0021 | +0.0024 |
| depth | +0.0104 | +0.0042 | +0.0063 |
| year | +0.0001 | -0.0001 | -0.0000 |

ΔMAE, DOY-climatology replacement (physics groups):

| group | conv | ptok4 | tdb |
|---|---|---|---|
| SST | +0.0024 | +0.0021 | +0.0019 |
| salinity | +0.0002 | +0.0001 | +0.0001 |
| MLD | +0.0002 | +0.0007 | +0.0006 |
| SLA | +0.0004 | +0.0004 | +0.0004 |
| currents (u, v) | +0.0029 | +0.0027 | +0.0028 |
| winds (u, v) | +0.0002 | +0.0010 | +0.0012 |
| air temp | +0.0005 | +0.0003 | +0.0003 |
| all 9 physics | +0.0083 | +0.0082 | +0.0093 |

- Groups with the largest mean ΔMAE over the three models (mean mode): clim, clim_static, ctx_chl, physics.
- Full per-model tables (RMSE, bias, gradient ratio / corr at lag 1, ΔMAE by distance and two-sided age bin, n): analyses/io5_ablation/results/job1_tables.md; rows in job1_table.csv.
- Figures: analyses/io5_ablation/figures/job1_heatmap_mean.png, job1_bars.png, job1_dist_bins.png.
- Caveat: an ablation shows what a trained model relies on at inference. It includes distribution shift (the model never saw the replaced values in training), and correlated inputs can stand in for each other. It is not a causal contribution of the input.
- Interpretation: not written by the queue (pending review).

## Physics loss λ 0.3 on the token-decoder + blend recipe (io5 Job 2, io5_tokdec_blend_up0p3; analyses/io5_ablation/report_job2.py)
- Logged 2026-10-24 15:12 by the io5 queue.
- Recipe: io4_tokdec_blend flags (batch 8, 60-epoch cosine, lr 1e-3, wd 1e-4, --ckpt-from 10, seed 0) + `--phys upwind --lam 0.3` (phys warm-up 3 epochs, L1 residual, K 500 m²/s, as earlier IO λ 0.3 runs). Single seed. Test scored once, at the checkpoint selected on val over epochs 11-60.
- At 30 epochs only the per-epoch history exists (val MAE, RMSE, gradient lag 1, residual); lags 2 / 4, bins and test are for the selected checkpoint.

Val n: 1458 days, 4,998,642 hidden px; test n: 1463 days, 5,522,954 hidden px. Residual |r| (upwind form, ln CHL / day): val px-days under real clouds 16,186,210, observed 6,822,242.

### Per-epoch history at the best val epoch (val only)

| run | window | best epoch | val MAE | val RMSE | grad ratio / corr lag 1 | val mean abs r cloud / obs | train data / phys loss | n epochs |
|---|---|---|---|---|---|---|---|---|
| token decoder + blend, λ 0.3 (io5) | 11-30 | 26 | 0.129816 | 0.2212 | 0.533 / 0.631 | 0.0389 / 0.0655 | 3.8321 / 0.0486 | 60 |
| token decoder + blend, λ 0.3 (io5) | 11-60 | 52 | 0.127555 | 0.2200 | 0.545 / 0.630 | 0.0419 / 0.0673 | 3.7120 / 0.0514 | 60 |
| token decoder + blend, λ 0 (io4) | 11-30 | 26 | 0.130135 | 0.2215 | 0.533 / 0.630 | 0.0494 / 0.0732 | 3.8351 / 0.0591 | 60 |
| token decoder + blend, λ 0 (io4) | 11-60 | 54 | 0.127585 | 0.2204 | 0.548 / 0.631 | 0.0594 / 0.0792 | 3.7310 / 0.0669 | 60 |

### Selected checkpoint (best val over epochs 11-60), scored once

| run | split | MAE | Δ vs λ 0 | RMSE | bias | grad ratio / corr lag 1 | lag 2 | lag 4 | mean abs r cloud / obs | n hidden px |
|---|---|---|---|---|---|---|---|---|---|---|
| token decoder + blend, λ 0.3 (io5) | val | 0.127555 | -0.000031 | 0.2200 | -0.0175 | 0.545 / 0.630 | 0.646 / 0.732 | 0.742 / 0.815 | 0.0419 / 0.0673 | 4,998,642 |
| token decoder + blend, λ 0 (io4) | val | 0.127585 | +0.000000 | 0.2204 | -0.0179 | 0.548 / 0.631 | 0.649 / 0.733 | 0.745 / 0.815 | 0.0594 / 0.0792 | 4,998,642 |
| token decoder + blend, λ 0.3 (io5) | test | 0.128578 | +0.000601 | 0.2290 | -0.0194 | 0.562 / 0.631 | 0.665 / 0.737 | 0.759 / 0.820 | 0.0476 / 0.0783 | 5,522,954 |
| token decoder + blend, λ 0 (io4) | test | 0.127977 | +0.000000 | 0.2274 | -0.0196 | 0.565 / 0.632 | 0.668 / 0.738 | 0.762 / 0.821 | 0.0642 / 0.0900 | 5,522,954 |

### Val MAE by bin, selected checkpoints (Δ = λ 0.3 − λ 0)

| bin | λ 0.3 MAE | λ 0 MAE | Δ | n hidden px |
|---|---|---|---|---|
| distance 1 px | 0.0770 | 0.0771 | -0.0001 | 1,213,097 |
| distance 2-3 px | 0.1005 | 0.1005 | +0.0000 | 898,560 |
| distance 4-7 px | 0.1188 | 0.1189 | -0.0001 | 1,048,246 |
| distance 8-15 px | 0.1401 | 0.1400 | +0.0001 | 880,930 |
| distance 16+ px | 0.2150 | 0.2151 | -0.0001 | 957,809 |
| two-sided age 1 d | 0.1316 | 0.1316 | -0.0001 | 3,506,780 |
| two-sided age 2-3 d | 0.1143 | 0.1144 | -0.0001 | 1,052,756 |
| two-sided age 4-8 d | 0.1218 | 0.1211 | +0.0007 | 352,033 |
| two-sided age none | 0.1495 | 0.1498 | -0.0003 | 87,073 |
| past-only age 1 d | 0.1365 | 0.1366 | -0.0001 | 2,451,259 |
| past-only age 2-3 d | 0.1179 | 0.1182 | -0.0002 | 1,368,813 |
| past-only age 4-8 d | 0.1153 | 0.1151 | +0.0002 | 795,388 |
| past-only age none | 0.1301 | 0.1297 | +0.0005 | 383,182 |

Cost: λ 0.3 median 107 s/epoch, peak 11.0 GB allocated / 15.0 GB reserved; λ 0 108 s/epoch, 11.0 / 15.0 GB. Non-finite predictions on hidden px (val / test): λ 0.3 0 / 0.

- Figures: analyses/io5_ablation/figures/job2_val_curves.png, job2_residual.png, job2_bins.png. Smoke test (1 epoch): io5_smoke_up0p3, log ~/mind_the_chl_gap/logs/io5/io5_smoke_up0p3.log.
- Interpretation: not written by the queue (pending review). Single seed; earlier S60 between conv seeds was 0.00024.

## Learned kriging and spatial token attention, io6 (IO; 2026-10-24; chl_cnn/krig.py, chl_cnn/krig_triton.py, chl_cnn/tokdec.py --tokdec-spatial, train_pinn.py --tokdec-krig [--krig-signed] / --tokdec-spatial, analyses/io6_kriging/)
- Set-up (design: analyses/io6_kriging/DESIGN.md, revision 2026-10-24):
  - K1 io6_tokdec_krig: io4 tokens + decoder; the own-pixel blend replaced by the learned-kriging stem. Stencil |dt| <= 4, 484 offsets (dt 0: disc r3 excluding the px, 28; |dt| 1-2: disc r5, 81 each; |dt| 3-4: 5 x 5 block + 8 points at r6, 33 each). Masked softmax over the visible keys (after donor masking) + a no-key slot per head; query / key / value features of DESIGN.md section 2 (advected offset with learned s, init 0.25; local visible density); relative-position bias per (offset, head); D 32, 4 heads, dv 16. 20 channels (attention output 16, estimate, spread, mass, entropy) -> backbone input 32 + 20 + 11 + 7 = 70 ch. No back-advected centring (--krig-advect implemented, not run).
  - K1-signed io6_tokdec_krig_signed: K1 + a signed head pair, w = softmax(s+) − lambda softmax(s−), lambda = sigmoid (init 0.1); 3 more channels (signed estimate, sum softmax(s+), lambda sum softmax(s−)) -> 73 ch.
  - spatial io6_tokdec_blend_spatial: io4_tokdec_blend + 2 pre-norm spatial self-attention blocks over the 1,164 pooled ocean-patch tokens (global, fused SDPA, 4 heads, D 128, learned relative-position bias per (patch offset, head); output projections zero-initialised).
  - Recipe as io4_tokdec_blend: batch 8, 60-epoch cosine, lr 1e-3, wd 1e-4, --ckpt-from 10, single seed 0.
  - Kriging implementation: fused Triton kernels (online softmax over all offsets per 16-px block, flash-style backward; tl.dot in tf32); the PyTorch streamed path (chunks, checkpointing) is the reference.
- Checks:
  - CPU unit tests (analyses/io6_kriging/test_krig.py): 19 of 19 passed; streamed vs dense masked softmax max |Δ| outputs 1.4e-06, gradients 4.8e-05 (relative); signed weights incl. no-key slots sum to 1 − lambda within 5.4e-07.
  - GPU check (gpu_check.py, random inputs, batch 2): Triton vs PyTorch reference, max |Δ| outputs / max relative |Δ| gradients: K1_ieee 9.5e-07 / 3.6e-05; K1_tf32 6.1e-05 / 1.7e-03; K1_signed_ieee 1.4e-06 / 2.7e-05; K1_signed_tf32 1.3e-04 / 2.1e-03.
  - leak test io6_init_krig (val day 2018-02-17, 7516 hidden px): +1 on day t's hidden px and on t+2's donor-hidden observed px: max change 0.0e+00 over every recorded tensor and the hidden-px predictions; positive control (all observed px of t+2): hidden-px predictions change by up to 0.015; no_leak True.
  - leak test io6_init_krig_signed (val day 2018-02-17, 7516 hidden px): +1 on day t's hidden px and on t+2's donor-hidden observed px: max change 0.0e+00 over every recorded tensor and the hidden-px predictions; positive control (all observed px of t+2): hidden-px predictions change by up to 0.018; no_leak True.
  - leak test io6_init_spatial (val day 2018-02-17, 7516 hidden px): +1 on day t's hidden px and on t+2's donor-hidden observed px: max change 0.0e+00 over every recorded tensor and the hidden-px predictions; positive control (all observed px of t+2): hidden-px predictions change by up to 0.012; no_leak True.
- Val / test hidden-px MAE (val n = 1,458 days, 4,998,642 px; test 1,463 days, 5,522,954 px):
  - K1 (io6_tokdec_krig) 0.128261 / 0.129415 (best epoch 58); vs io4_tokdec_blend +0.00068 / +0.00144; vs conv seed 0 -0.00190 / -0.00113; non-finite predictions 0 / 0;
  - spatial (io6_tokdec_blend_spatial) 0.127060 / 0.127516 (best epoch 58); vs io4_tokdec_blend -0.00053 / -0.00046; vs conv seed 0 -0.00311 / -0.00303; non-finite predictions 0 / 0;
  - K1-signed (io6_tokdec_krig_signed) 0.129143 / 0.129664 (best epoch 58); vs io4_tokdec_blend +0.00156 / +0.00169; vs conv seed 0 -0.00102 / -0.00088; non-finite predictions 0 / 0;
  - references: io4_tokdec_blend 0.127585 / 0.127977; io3_ptok4 0.128197 / 0.128637; io2_tattn8 0.128594 / 0.127181; conv ±4 seed 0 0.130165 / 0.130548; 3-seed conv ensemble 0.127881 / 0.127895.
- RMSE val / test, bias val / test: K1 (io6_tokdec_krig) 0.2218 / 0.2309, -0.0171 / -0.0200; spatial (io6_tokdec_blend_spatial) 0.2189 / 0.2262, -0.0151 / -0.0172; K1-signed (io6_tokdec_krig_signed) 0.2237 / 0.2313, -0.0180 / -0.0202; io4_tokdec_blend 0.2204 / 0.2274, -0.0179 / -0.0196.
- Gradient ratio / corr, val, lags 1 / 2 / 4 px (8.40 M lag-1 pairs):
  - K1 (io6_tokdec_krig) 0.546/0.632, 0.646/0.734, 0.741/0.815;
  - spatial (io6_tokdec_blend_spatial) 0.546/0.629, 0.647/0.732, 0.743/0.815;
  - K1-signed (io6_tokdec_krig_signed) 0.544/0.631, 0.644/0.733, 0.738/0.814;
  - io4_tokdec_blend 0.548/0.631, 0.649/0.733, 0.745/0.815;
  - conv ±4 seed 0 0.543/0.633, 0.643/0.734, 0.737/0.813;
- Best val over epochs 11-30 / 11-60: K1 (io6_tokdec_krig) 0.12966 (ep 26) / 0.12826 (ep 58); spatial (io6_tokdec_blend_spatial) 0.12968 (ep 29) / 0.12706 (ep 58); K1-signed (io6_tokdec_krig_signed) 0.13017 (ep 25) / 0.12914 (ep 58); io4_tokdec_blend 0.13013 (ep 26) / 0.12759 (ep 54) (S30 0.0015, S60 0.00024 from the conv seeds).
- Cost at batch 8 (median s / epoch incl. the val pass, peak GB allocated / reserved, params): K1 (io6_tokdec_krig) 145 s, 11.1 / 14.9 GB, 1,437,424; spatial (io6_tokdec_blend_spatial) 128 s, 11.2 / 16.0 GB, 1,773,197; K1-signed (io6_tokdec_krig_signed) 159 s, 11.1 / 15.0 GB, 1,441,235; io4_tokdec_blend 108 s, 11.0 / 15.0 GB, 1,453,637.
- Smokes (2 epochs, batch 8): io6_smoke_krig 146, 145 s/epoch, 11.1 / 14.9 GB, 1,437,424 params; io6_smoke_spatial 129, 128 s/epoch, 11.2 / 15.9 GB, 1,773,197 params; io6_smoke_krig_signed 160, 159 s/epoch, 11.1 / 15.0 GB, 1,441,235 params.
- Val MAE gain over conv seed 0, by bin (K1 (io6_tokdec_krig) / spatial (io6_tokdec_blend_spatial) / K1-signed (io6_tokdec_krig_signed) / io4_tokdec_blend):
  - distance 1 / 2-3 / 4-7 / 8-15 / 16+ px (n 1.21 M, 0.90 M, 1.05 M, 0.88 M, 0.96 M): +0.0006 / +0.0005 / +0.0007 / +0.0008 / +0.0073; +0.0005 / +0.0007 / +0.0013 / +0.0027 / +0.0110; +0.0005 / +0.0003 / +0.0001 / -0.0003 / +0.0045; +0.0004 / +0.0005 / +0.0009 / +0.0022 / +0.0094.
  - two-sided age 1 d / 2-3 d / 4-8 d / none (n 3.51 M, 1.05 M, 0.35 M, 0.09 M): +0.0023 / +0.0013 / +0.0010 / -0.0013; +0.0032 / +0.0027 / +0.0036 / +0.0012; +0.0013 / +0.0006 / +0.0003 / -0.0010; +0.0026 / +0.0021 / +0.0038 / +0.0021.
  - past-only age 1 d / 2-3 d / 4-8 d / none (n 2.45 M, 1.37 M, 0.80 M, 0.38 M): +0.0025 / +0.0016 / +0.0016 / -0.0000; +0.0034 / +0.0026 / +0.0034 / +0.0021; +0.0014 / +0.0007 / +0.0010 / -0.0002; +0.0027 / +0.0019 / +0.0033 / +0.0026.
- Val hidden px by visible kriging-stencil keys: >= 1 key n 4,359,604, no key n 639,038; keys per hidden px mean 48.9, median 29. MAE >= 1 key / no key:
  - K1 (io6_tokdec_krig) 0.11085 / 0.24706;
  - spatial (io6_tokdec_blend_spatial) 0.11014 / 0.24249;
  - K1-signed (io6_tokdec_krig_signed) 0.11130 / 0.25087;
  - io4_tokdec_blend 0.11041 / 0.24476; conv seed 0 0.11147 / 0.25768.
- Kriging weights, K1 (io6_tokdec_krig) (146 val days, 487,572 hidden px, 417,788 with >= 1 key; mean over the value heads):
  - weight share by dt −4..+4: 0.017 / 0.029 / 0.068 / 0.141 / 0.124 / 0.148 / 0.072 / 0.030 / 0.016; total on real keys 0.645;
  - mean weight per visible key by dt −4..+4: 0.0050 / 0.0086 / 0.0079 / 0.0164 / 0.0558 / 0.0174 / 0.0087 / 0.0092 / 0.0047;
  - mean weight per visible key by |dSST| 0 / 0.25 / 0.5 / 0.75 / 1.0 / 1.25-1.5 / 1.75-2 / 2.25-3 / >3 K (store SST quantised to 0.25 K): 0.0205 / 0.0111 / 0.0052 / 0.0032 / 0.0022 / 0.0018 / 0.0017 / 0.0020 / 0.0026 (n 8.66 M, 10.27 M, 3.46 M, 1.07 M, 0.38 M, 0.23 M, 0.07 M, 0.04 M, 0.04 M);
  - learned s 0.944 (init 0.25); no-key logits +0.09, +0.22, +0.17, +0.17.
- Kriging weights, K1-signed (io6_tokdec_krig_signed) (146 val days, 487,572 hidden px, 417,788 with >= 1 key; mean over the value heads):
  - weight share by dt −4..+4: 0.017 / 0.029 / 0.070 / 0.141 / 0.150 / 0.162 / 0.081 / 0.032 / 0.017; total on real keys 0.699;
  - mean weight per visible key by dt −4..+4: 0.0049 / 0.0084 / 0.0081 / 0.0165 / 0.0675 / 0.0191 / 0.0097 / 0.0099 / 0.0052;
  - mean weight per visible key by |dSST| 0 / 0.25 / 0.5 / 0.75 / 1.0 / 1.25-1.5 / 1.75-2 / 2.25-3 / >3 K (store SST quantised to 0.25 K): 0.0225 / 0.0119 / 0.0055 / 0.0034 / 0.0024 / 0.0020 / 0.0020 / 0.0024 / 0.0033 (n 8.66 M, 10.27 M, 3.46 M, 1.07 M, 0.38 M, 0.23 M, 0.07 M, 0.04 M, 0.04 M);
  - learned s 0.977 (init 0.25); lambda 0.297 (init 0.1); no-key logits +0.32, +0.14, +0.24, +0.17, +0.13, -0.35.
  - signed pair, weight share by dt (softmax(s+) / softmax(s−)): 0.000:0.039 / 0.001:0.050 / 0.015:0.109 / 0.067:0.140 / 0.088:0.118 / 0.083:0.153 / 0.021:0.110 / 0.003:0.048 / 0.000:0.028.
- Kriging spread vs |error|, K1 (io6_tokdec_krig) (val hidden px with >= 1 key, n 4,359,604): Spearman 0.267; MAE by spread decile 0.1132 (lowest) ... 0.2536 (highest); Spearman by distance bin 0.387 / 0.332 / 0.270 / 0.232 / 0.163.
- Kriging spread vs |error|, K1-signed (io6_tokdec_krig_signed) (val hidden px with >= 1 key, n 4,359,604): Spearman 0.282; MAE by spread decile 0.1010 (lowest) ... 0.2569 (highest); Spearman by distance bin 0.379 / 0.333 / 0.283 / 0.254 / 0.203.
- Outputs:
  - analyses/io6_kriging/results/{io6_summary, io6_tokdec_krig, io6_tokdec_blend_spatial, io6_tokdec_krig_signed, gpu_check, unit_tests, leak_*}.json;
  - figures io6_curves, io6_bins, io6_variogram_lines_<run>, io6_variogram_maps_<run>, io6_weight_vs_sst, io6_learned_params, io6_spread_vs_error (.png);
  - recorder data ~/mind_the_chl_gap/data/io6/<run>_{krigall,diag}.npz; queue ~/mind_the_chl_gap/logs/io6/queue.sh.

## Phase 0 evaluations, bridge #25 (IO; 2026-10-25 01:00 UTC; eval only; analyses/io7/phase0/, analyses/io7/postrun.py, analyses/io7/keys.py)
- Assumptions:
  - Key counts (≥ 1 visible kriging-stencil key, |dt| ≤ 4, 484 offsets) computed on CPU from the visibility masks (analyses/io7/keys.py); identical to the io6_tokdec_krig recorder pass on all 48.8 M val+test ocean px-days (0 differences).
  - Regions (no IO region mask in the repo): Arabian Sea: lat >= 5, lon < 78 (incl. Gulf of Aden, Gulf of Oman, Persian Gulf, Laccadive Sea); Bay of Bengal: lat >= 5, lon >= 78 (incl. Andaman Sea west of 90 E); Equatorial and south: lat < 5 (5 S-5 N).
  - "io3 attention" = io3_ptok4 (patch tokens, arm A). Conv ±4 seeds = io2_base_eval (seed 0), io2_seed1, io2_seed2.
  - Real-cloud reweighting: hidden-px errors weighted so the joint (distance to nearest day-t visible px bin × key-count bin) distribution matches the val days' real-cloud px (ocean, not observed; n 17,084,105); distance bins 1, 2-3, 4-7, 8-15, 16-31, 32+ px; key bins 0, 1-4, 5-19, 20-59, 60+; cells with < 5 hidden px dropped. Assumes errors are exchangeable between fake-cloud and real-cloud px within a cell. The same val target is applied to test.
- 1. io6_tokdec_blend_spatial val MAE on no-key hidden px (639,038 px, 1458 days) by region: Arabian Sea 0.2699 (n 486,511); Bay of Bengal 0.1790 (n 88,691); Equatorial and south 0.1219 (n 63,836).
  - By month (all regions): Jan 0.275 (n 71,003), Feb 0.275 (n 69,618), Mar 0.246 (n 84,072), Apr 0.165 (n 65,698), May 0.174 (n 49,182), Jun 0.194 (n 14,833), Jul 0.202 (n 14,434), Aug 0.290 (n 19,845), Sep 0.275 (n 23,410), Oct 0.282 (n 58,883), Nov 0.250 (n 92,234), Dec 0.245 (n 75,826).
  - Month × region: min 0.096, max 0.366; cells with < 5 px: 0 of 36. Table: results/phase0.json item1; figure figures/p0_nokey_month_region.png.
- 2. Spatial attention weights of io6_tokdec_blend_spatial (146 val days, every 10th val position; all 1164 ocean-patch queries; uniform weight 1/Np = 8.59e-04). Mean weight per (query, key) pair, head mean:
  - block 1 by distance (patches): 0 1.05e-03, 1 2.85e-03, 2 1.98e-03, 3-4 1.51e-03, 5-8 1.11e-03, 9-16 8.59e-04, 17-32 7.32e-04, 33+ 7.73e-04; self weight share 0.001, share within ≤ 2 patches 0.043.
  - block 1 by |ΔSST| (K): 0 9.95e-04, 0.25 9.51e-04, 0.5 8.96e-04, 0.75 8.62e-04, 1.0 8.39e-04, 1.25-1.5 8.19e-04, 1.75-2.0 7.99e-04, 2.25-3.0 7.83e-04, >3.0 7.76e-04.
  - block 2 by distance (patches): 0 1.39e-03, 1 2.95e-03, 2 2.07e-03, 3-4 1.69e-03, 5-8 1.11e-03, 9-16 8.46e-04, 17-32 7.20e-04, 33+ 7.90e-04; self weight share 0.001, share within ≤ 2 patches 0.045.
  - block 2 by |ΔSST| (K): 0 1.01e-03, 0.25 9.57e-04, 0.5 8.96e-04, 0.75 8.59e-04, 1.0 8.34e-04, 1.25-1.5 8.13e-04, 1.75-2.0 7.95e-04, 2.25-3.0 7.82e-04, >3.0 7.76e-04.
  - pairs per distance bin: 0 169,944, 1 1,268,740, 2 1,804,268, 3-4 6,614,676, 5-8 18,499,076, 9-16 55,480,000, 17-32 92,129,504, 33+ 21,848,608. Figures figures/p0_spatial_attn_bins.png (incl. |ΔSST| within distance bins), p0_spatial_attn_offsets.png.
- 3-4. Val / test MAE (val n 4,998,642 hidden px, 1458 days; test n 5,522,954, 1463 days); key split n val ≥ 1 key 4,359,604, no key 639,038:

| model | val | test | val ≥ 1 key | val no key | test ≥ 1 key | test no key | val rw | test rw | preds |
|---|---|---|---|---|---|---|---|---|---|
| io6_tokdec_blend_spatial | 0.12706 | 0.12752 | 0.11014 | 0.24249 | 0.11502 | 0.26879 | 0.15182 | 0.16962 | float32 |
| io4_tokdec_blend | 0.12759 | 0.12798 | 0.11041 | 0.24476 | 0.11516 | 0.27281 | 0.15258 | 0.17071 | float32 |
| io3_ptok4 | 0.12820 | 0.12864 | 0.11053 | 0.24870 | 0.11539 | 0.27835 | 0.15377 | 0.17264 | float32 |
| io2_base_eval | 0.13016 | 0.13055 | 0.11147 | 0.25768 | 0.11636 | 0.29094 | 0.15700 | 0.17716 | float32 |
| io2_seed1 | 0.13011 | 0.13019 | 0.11156 | 0.25669 | 0.11635 | 0.28664 | 0.15686 | 0.17599 | float32 |
| io2_seed2 | 0.13035 | 0.12996 | 0.11145 | 0.25932 | 0.11620 | 0.28544 | 0.15741 | 0.17564 | float32 |
| ens6_mean | 0.12520 | 0.12539 | 0.10790 | 0.24323 | 0.11250 | 0.27109 | 0.15027 | 0.16826 | mean |

- Ensemble (ens6_mean, per-px mean of 6) − io6_tokdec_blend_spatial: val -0.00186, test -0.00213; rw val -0.00155, test -0.00136. Gradient ratio/corr lag 1 val: ensemble 0.531/0.644, io6_tokdec_blend_spatial 0.546/0.629.
- Real-cloud reweighting: uncovered real-cloud mass 0.0000; cells used 29 of 30. Test with the test days' own real-cloud target instead: io6_tokdec_blend_spatial 0.15154, io4_tokdec_blend 0.15230, io3_ptok4 0.15357, io2_base_eval 0.15678, io2_seed1 0.15609, io2_seed2 0.15576, ens6_mean 0.14971. Figures figures/p0_models_mae.png, p0_rw_cells.png.

## io7 postrun: io6_tokdec_blend_spatial_s1 (IO; 2026-10-25 01:00 UTC; analyses/io7/postrun.py)
- Run: t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, seed 1, epochs 60, batch 8, best_epoch 48. Val n: 1458 days, 4,998,642 hidden px; test n: 1463 days, 5,522,954.
- Val / test MAE 0.126741 / 0.127239; RMSE 0.2180 / 0.2249; bias -0.0150 / -0.0170; non-finite 0 / 0.
- Δ vs io6_tokdec_blend_spatial (0.127060 / 0.127516): val -0.00032, test -0.00028; ≥ 1 key -0.00020 / -0.00014; no key -0.00114 / -0.00183; real-cloud rw -0.00049 / -0.00071.
- Δ vs io4_tokdec_blend (0.127585 / 0.127977): val -0.00084, test -0.00074; ≥ 1 key -0.00047 / -0.00029; no key -0.00341 / -0.00584; real-cloud rw -0.00125 / -0.00179.
- Key split val / test: ≥ 1 key 0.10994 (n 4,359,604) / 0.11488 (n 5,074,031); no key 0.24135 (n 639,038) / 0.26697 (n 448,923).
- Real-cloud reweighted MAE val / test: 0.15133 / 0.16892 (target: val real-cloud px, n 17,084,105; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.15104); io6_tokdec_blend_spatial 0.15182 / 0.16962, io4_tokdec_blend 0.15258 / 0.17071.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.549/0.631, 0.652/0.733, 0.748/0.816; test 0.568/0.632, 0.671/0.738, 0.766/0.821 (val lag-1 pairs 8,403,189).
- Val MAE gain over conv ±4 seed 0 by distance to visible px (conv − run): 1 px +0.00049 (n 1,213,097), 2-3 px +0.00074 (n 898,560), 4-7 px +0.00168 (n 1,048,246), 8-15 px +0.00337 (n 880,930), 16+ px +0.01161 (n 957,809).

## io7 postrun: io6_tokdec_blend_spatial_s2 (IO; 2026-10-25 03:12 UTC; analyses/io7/postrun.py)
- Run: t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, seed 2, epochs 60, batch 8, best_epoch 58. Val n: 1458 days, 4,998,642 hidden px; test n: 1463 days, 5,522,954.
- Val / test MAE 0.126605 / 0.127352; RMSE 0.2184 / 0.2257; bias -0.0170 / -0.0171; non-finite 0 / 0.
- Δ vs io6_tokdec_blend_spatial (0.127060 / 0.127516): val -0.00045, test -0.00016; ≥ 1 key -0.00018 / -0.00014; no key -0.00230 / -0.00043; real-cloud rw -0.00080 / -0.00029.
- Δ vs io4_tokdec_blend (0.127585 / 0.127977): val -0.00098, test -0.00062; ≥ 1 key -0.00045 / -0.00029; no key -0.00457 / -0.00445; real-cloud rw -0.00156 / -0.00137.
- Key split val / test: ≥ 1 key 0.10996 (n 4,359,604) / 0.11488 (n 5,074,031); no key 0.24019 (n 639,038) / 0.26836 (n 448,923).
- Real-cloud reweighted MAE val / test: 0.15102 / 0.16933 (target: val real-cloud px, n 17,084,105; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.15130); io6_tokdec_blend_spatial 0.15182 / 0.16962, io4_tokdec_blend 0.15258 / 0.17071.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.542/0.631, 0.643/0.734, 0.739/0.816; test 0.561/0.632, 0.663/0.738, 0.758/0.821 (val lag-1 pairs 8,403,189).
- Val MAE gain over conv ±4 seed 0 by distance to visible px (conv − run): 1 px +0.00038 (n 1,213,097), 2-3 px +0.00066 (n 898,560), 4-7 px +0.00160 (n 1,048,246), 8-15 px +0.00335 (n 880,930), 16+ px +0.01264 (n 957,809).

## io7 postrun: io7_physmem (IO; 2026-10-25 06:25 UTC; analyses/io7/postrun.py)
- Run: t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, seed 0, epochs 60, batch 8, best_epoch 58. Val n: 1458 days, 4,998,642 hidden px; test n: 1463 days, 5,522,954.
- Val / test MAE 0.126563 / 0.126886; RMSE 0.2179 / 0.2246; bias -0.0138 / -0.0161; non-finite 0 / 0.
- Δ vs io6_tokdec_blend_spatial (0.127060 / 0.127516): val -0.00050, test -0.00063; ≥ 1 key -0.00005 / +0.00001; no key -0.00353 / -0.00782; real-cloud rw -0.00097 / -0.00218.
- Δ vs io4_tokdec_blend (0.127585 / 0.127977): val -0.00102, test -0.00109; ≥ 1 key -0.00032 / -0.00014; no key -0.00580 / -0.01184; real-cloud rw -0.00173 / -0.00326.
- Key split val / test: ≥ 1 key 0.11009 (n 4,359,604) / 0.11502 (n 5,074,031); no key 0.23896 (n 639,038) / 0.26097 (n 448,923).
- Real-cloud reweighted MAE val / test: 0.15085 / 0.16745 (target: val real-cloud px, n 17,084,105; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.15018); io6_tokdec_blend_spatial 0.15182 / 0.16962, io4_tokdec_blend 0.15258 / 0.17071.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.547/0.629, 0.649/0.732, 0.745/0.815; test 0.564/0.629, 0.667/0.735, 0.762/0.820 (val lag-1 pairs 8,403,189).
- Val MAE gain over conv ±4 seed 0 by distance to visible px (conv − run): 1 px +0.00044 (n 1,213,097), 2-3 px +0.00054 (n 898,560), 4-7 px +0.00134 (n 1,048,246), 8-15 px +0.00331 (n 880,930), 16+ px +0.01322 (n 957,809).

## io7 postrun: io7_wmattn (IO; 2026-10-25 10:27 UTC; analyses/io7/postrun.py)
- Run: t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, seed 0, epochs 60, batch 8, best_epoch 44. Val n: 1458 days, 4,998,642 hidden px; test n: 1463 days, 5,522,954.
- Val / test MAE 0.127964 / 0.128345; RMSE 0.2194 / 0.2261; bias -0.0144 / -0.0140; non-finite 0 / 0.
- Δ vs io6_tokdec_blend_spatial (0.127060 / 0.127516): val +0.00090, test +0.00083; ≥ 1 key +0.00080 / +0.00054; no key +0.00163 / +0.00411; real-cloud rw +0.00099 / +0.00132.
- Δ vs io4_tokdec_blend (0.127585 / 0.127977): val +0.00038, test +0.00037; ≥ 1 key +0.00053 / +0.00039; no key -0.00065 / +0.00009; real-cloud rw +0.00023 / +0.00023.
- Key split val / test: ≥ 1 key 0.11094 (n 4,359,604) / 0.11556 (n 5,074,031); no key 0.24411 (n 639,038) / 0.27290 (n 448,923).
- Real-cloud reweighted MAE val / test: 0.15281 / 0.17094 (target: val real-cloud px, n 17,084,105; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.15259); io6_tokdec_blend_spatial 0.15182 / 0.16962, io4_tokdec_blend 0.15258 / 0.17071.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.540/0.632, 0.642/0.734, 0.738/0.815; test 0.559/0.633, 0.663/0.739, 0.758/0.821 (val lag-1 pairs 8,403,189).
- Val MAE gain over conv ±4 seed 0 by distance to visible px (conv − run): 1 px -0.00035 (n 1,213,097), 2-3 px -0.00005 (n 898,560), 4-7 px +0.00075 (n 1,048,246), 8-15 px +0.00207 (n 880,930), 16+ px +0.00925 (n 957,809).

## io7 postrun: io7_analog (IO; 2026-10-25 13:01 UTC; analyses/io7/postrun.py)
- Run: t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, seed 0, epochs 60, batch 8, best_epoch 58. Val n: 1458 days, 4,998,642 hidden px; test n: 1463 days, 5,522,954.
- Val / test MAE 0.127096 / 0.127728; RMSE 0.2192 / 0.2268; bias -0.0144 / -0.0164; non-finite 0 / 0.
- Δ vs io6_tokdec_blend_spatial (0.127060 / 0.127516): val +0.00004, test +0.00021; ≥ 1 key -0.00007 / -0.00008; no key +0.00074 / +0.00348; real-cloud rw +0.00018 / +0.00099.
- Δ vs io4_tokdec_blend (0.127585 / 0.127977): val -0.00049, test -0.00025; ≥ 1 key -0.00034 / -0.00022; no key -0.00154 / -0.00053; real-cloud rw -0.00059 / -0.00009.
- Key split val / test: ≥ 1 key 0.11007 (n 4,359,604) / 0.11494 (n 5,074,031); no key 0.24322 (n 639,038) / 0.27228 (n 448,923).
- Real-cloud reweighted MAE val / test: 0.15200 / 0.17062 (target: val real-cloud px, n 17,084,105; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.15213); io6_tokdec_blend_spatial 0.15182 / 0.16962, io4_tokdec_blend 0.15258 / 0.17071.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.548/0.630, 0.649/0.733, 0.745/0.815; test 0.566/0.632, 0.669/0.738, 0.763/0.821 (val lag-1 pairs 8,403,189).
- Val MAE gain over conv ±4 seed 0 by distance to visible px (conv − run): 1 px +0.00061 (n 1,213,097), 2-3 px +0.00070 (n 898,560), 4-7 px +0.00149 (n 1,048,246), 8-15 px +0.00304 (n 880,930), 16+ px +0.01016 (n 957,809).

## blk7 postrun: blk4_tokdec_blend_spatial (Black Sea 4 km; 2026-10-26 04:10 UTC; analyses/io7/postrun.py)
- Run: store blk4, crop True, val_stride 4, t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, tokdec_krig False, krig_signed False, physmem False, physmem_stride2 False, context 8, seed 0, epochs 60, batch 8, lr 0.001, lam 0.0, best_epoch 59. Val n: 182 days, 1,062,322 hidden px; test n: 183 days, 1,101,920.
- Val / test MAE 0.141189 / 0.133981; RMSE 0.2313 / 0.2278; bias -0.0066 / -0.0235; non-finite 0 / 0.
- Δ vs blk4_t3d_c4_gdl100_lam0_b16_50ep (0.156113 / 0.150893): val -0.01492, test -0.01691; ≥ 1 key -0.01450 / -0.01602; no key -0.02171 / -0.03261; real-cloud rw -0.01658 / -0.01930.
- Δ vs blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (0.149787 / 0.142998): val -0.00860, test -0.00902; ≥ 1 key -0.00826 / -0.00818; no key -0.01396 / -0.02380; real-cloud rw -0.01069 / -0.01183.
- Key split val / test: ≥ 1 key 0.13418 (n 1,000,176) / 0.12762 (n 1,042,846); no key 0.25408 (n 62,146) / 0.24627 (n 59,074).
- Real-cloud reweighted MAE val / test: 0.16460 / 0.16054 (target: val real-cloud px, n 2,340,558; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.15921); blk4_t3d_c4_gdl100_lam0_b16_50ep 0.18118 / 0.17985, blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 0.17529 / 0.17237.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.556/0.665, 0.617/0.745, 0.690/0.819; test 0.593/0.687, 0.654/0.763, 0.726/0.835 (val lag-1 pairs 2,067,873).
- Note: blk4_t3d_* references use another recipe (conv ±4, batch 16, 50 epochs; the _sl_lam0p3_..._cont50 run: λ 0.3, semi-Lagrangian, 50 epochs then two 50-epoch continuations, the last at lr 1e-4); blk7 runs: batch 8, 60-epoch cosine, lr 1e-3, λ 0. Scored days: every 4th val / test day (--val-stride 4).
- Val MAE gain over conv ±4 b16 50 ep (blk4_t3d_c4_gdl100_lam0_b16_50ep) by distance to visible px (conv − run): 1 px +0.01675 (n 138,579), 2-3 px +0.01368 (n 117,729), 4-7 px +0.01235 (n 157,936), 8-15 px +0.01142 (n 179,608), 16+ px +0.01691 (n 468,470).

## blk7 postrun: blk4_physmem (Black Sea 4 km; 2026-10-26 06:11 UTC; analyses/io7/postrun.py)
- Run: store blk4, crop True, val_stride 4, t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, tokdec_krig False, krig_signed False, physmem True, physmem_stride2 False, context 15, seed 0, epochs 60, batch 8, lr 0.001, lam 0.0, best_epoch 55. Val n: 182 days, 1,062,322 hidden px; test n: 183 days, 1,101,920.
- Val / test MAE 0.142260 / 0.136556; RMSE 0.2320 / 0.2327; bias -0.0106 / -0.0311; non-finite 0 / 0.
- Δ vs blk4_tokdec_blend_spatial (0.141189 / 0.133981): val +0.00107, test +0.00258; ≥ 1 key +0.00133 / +0.00255; no key -0.00310 / +0.00305; real-cloud rw +0.00074 / +0.00316.
- Δ vs blk4_t3d_c4_gdl100_lam0_b16_50ep (0.156113 / 0.150893): val -0.01385, test -0.01434; ≥ 1 key -0.01317 / -0.01347; no key -0.02481 / -0.02956; real-cloud rw -0.01585 / -0.01614.
- Δ vs blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (0.149787 / 0.142998): val -0.00753, test -0.00644; ≥ 1 key -0.00693 / -0.00563; no key -0.01706 / -0.02076; real-cloud rw -0.00996 / -0.00867.
- Key split val / test: ≥ 1 key 0.13551 (n 1,000,176) / 0.13017 (n 1,042,846); no key 0.25097 (n 62,146) / 0.24932 (n 59,074).
- Real-cloud reweighted MAE val / test: 0.16533 / 0.16370 (target: val real-cloud px, n 2,340,558; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.16247); blk4_tokdec_blend_spatial 0.16460 / 0.16054, blk4_t3d_c4_gdl100_lam0_b16_50ep 0.18118 / 0.17985, blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 0.17529 / 0.17237.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.558/0.660, 0.618/0.739, 0.690/0.813; test 0.590/0.682, 0.650/0.757, 0.721/0.829 (val lag-1 pairs 2,067,873).
- Note: blk4_t3d_* references use another recipe (conv ±4, batch 16, 50 epochs; the _sl_lam0p3_..._cont50 run: λ 0.3, semi-Lagrangian, 50 epochs then two 50-epoch continuations, the last at lr 1e-4); blk7 runs: batch 8, 60-epoch cosine, lr 1e-3, λ 0. Scored days: every 4th val / test day (--val-stride 4).
- Val MAE gain over conv ±4 b16 50 ep (blk4_t3d_c4_gdl100_lam0_b16_50ep) by distance to visible px (conv − run): 1 px +0.01644 (n 138,579), 2-3 px +0.01221 (n 117,729), 4-7 px +0.01030 (n 157,936), 8-15 px +0.01102 (n 179,608), 16+ px +0.01579 (n 468,470).

## blk7 postrun: blk4_tokdec_blend (Black Sea 4 km; 2026-10-26 07:28 UTC; analyses/io7/postrun.py)
- Run: store blk4, crop True, val_stride 4, t3d_stem tokdec, tokdec_blend True, tokdec_spatial 0, tokdec_krig False, krig_signed False, physmem False, physmem_stride2 False, context 8, seed 0, epochs 60, batch 8, lr 0.001, lam 0.0, best_epoch 59. Val n: 182 days, 1,062,322 hidden px; test n: 183 days, 1,101,920.
- Val / test MAE 0.143159 / 0.135924; RMSE 0.2354 / 0.2320; bias -0.0086 / -0.0256; non-finite 0 / 0.
- Δ vs blk4_tokdec_blend_spatial (0.141189 / 0.133981): val +0.00197, test +0.00194; ≥ 1 key +0.00194 / +0.00181; no key +0.00244 / +0.00436; real-cloud rw +0.00270 / +0.00245.
- Δ vs blk4_physmem (0.142260 / 0.136556): val +0.00090, test -0.00063; ≥ 1 key +0.00061 / -0.00074; no key +0.00554 / +0.00132; real-cloud rw +0.00196 / -0.00072.
- Δ vs blk4_t3d_c4_gdl100_lam0_b16_50ep (0.156113 / 0.150893): val -0.01295, test -0.01497; ≥ 1 key -0.01256 / -0.01422; no key -0.01927 / -0.02825; real-cloud rw -0.01388 / -0.01686.
- Δ vs blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (0.149787 / 0.142998): val -0.00663, test -0.00707; ≥ 1 key -0.00632 / -0.00637; no key -0.01152 / -0.01944; real-cloud rw -0.00799 / -0.00938.
- Key split val / test: ≥ 1 key 0.13612 (n 1,000,176) / 0.12943 (n 1,042,846); no key 0.25651 (n 62,146) / 0.25063 (n 59,074).
- Real-cloud reweighted MAE val / test: 0.16730 / 0.16299 (target: val real-cloud px, n 2,340,558; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.16177); blk4_tokdec_blend_spatial 0.16460 / 0.16054, blk4_physmem 0.16533 / 0.16370, blk4_t3d_c4_gdl100_lam0_b16_50ep 0.18118 / 0.17985, blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 0.17529 / 0.17237.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.554/0.662, 0.615/0.742, 0.688/0.816; test 0.590/0.685, 0.650/0.761, 0.721/0.833 (val lag-1 pairs 2,067,873).
- Note: blk4_t3d_* references use another recipe (conv ±4, batch 16, 50 epochs; the _sl_lam0p3_..._cont50 run: λ 0.3, semi-Lagrangian, 50 epochs then two 50-epoch continuations, the last at lr 1e-4); blk7 runs: batch 8, 60-epoch cosine, lr 1e-3, λ 0. Scored days: every 4th val / test day (--val-stride 4).
- Val MAE gain over conv ±4 b16 50 ep (blk4_t3d_c4_gdl100_lam0_b16_50ep) by distance to visible px (conv − run): 1 px +0.01661 (n 138,579), 2-3 px +0.01290 (n 117,729), 4-7 px +0.01134 (n 157,936), 8-15 px +0.01119 (n 179,608), 16+ px +0.01311 (n 468,470).

## blk7 postrun: blk4_tokdec_krig (Black Sea 4 km; 2026-10-26 09:11 UTC; analyses/io7/postrun.py)
- Run: store blk4, crop True, val_stride 4, t3d_stem tokdec, tokdec_blend False, tokdec_spatial 0, tokdec_krig True, krig_signed False, physmem False, physmem_stride2 False, context 8, seed 0, epochs 60, batch 8, lr 0.001, lam 0.0, best_epoch 53. Val n: 182 days, 1,062,322 hidden px; test n: 183 days, 1,101,920.
- Val / test MAE 0.145181 / 0.136485; RMSE 0.2385 / 0.2327; bias -0.0093 / -0.0285; non-finite 0 / 0.
- Δ vs blk4_tokdec_blend_spatial (0.141189 / 0.133981): val +0.00399, test +0.00250; ≥ 1 key +0.00415 / +0.00212; no key +0.00148 / +0.00922; real-cloud rw +0.00497 / +0.00382.
- Δ vs blk4_physmem (0.142260 / 0.136556): val +0.00292, test -0.00007; ≥ 1 key +0.00282 / -0.00043; no key +0.00458 / +0.00618; real-cloud rw +0.00423 / +0.00065.
- Δ vs blk4_tokdec_blend (0.143159 / 0.135924): val +0.00202, test +0.00056; ≥ 1 key +0.00221 / +0.00032; no key -0.00095 / +0.00486; real-cloud rw +0.00227 / +0.00137.
- Δ vs blk4_t3d_c4_gdl100_lam0_b16_50ep (0.156113 / 0.150893): val -0.01093, test -0.01441; ≥ 1 key -0.01035 / -0.01390; no key -0.02022 / -0.02338; real-cloud rw -0.01162 / -0.01549.
- Δ vs blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (0.149787 / 0.142998): val -0.00461, test -0.00651; ≥ 1 key -0.00412 / -0.00606; no key -0.01248 / -0.01458; real-cloud rw -0.00573 / -0.00801.
- Key split val / test: ≥ 1 key 0.13832 (n 1,000,176) / 0.12974 (n 1,042,846); no key 0.25556 (n 62,146) / 0.25549 (n 59,074).
- Real-cloud reweighted MAE val / test: 0.16956 / 0.16436 (target: val real-cloud px, n 2,340,558; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.16294); blk4_tokdec_blend_spatial 0.16460 / 0.16054, blk4_physmem 0.16533 / 0.16370, blk4_tokdec_blend 0.16730 / 0.16299, blk4_t3d_c4_gdl100_lam0_b16_50ep 0.18118 / 0.17985, blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 0.17529 / 0.17237.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.559/0.658, 0.619/0.739, 0.690/0.813; test 0.598/0.682, 0.657/0.759, 0.728/0.832 (val lag-1 pairs 2,067,873).
- Note: blk4_t3d_* references use another recipe (conv ±4, batch 16, 50 epochs; the _sl_lam0p3_..._cont50 run: λ 0.3, semi-Lagrangian, 50 epochs then two 50-epoch continuations, the last at lr 1e-4); blk7 runs: batch 8, 60-epoch cosine, lr 1e-3, λ 0. Scored days: every 4th val / test day (--val-stride 4).
- Val MAE gain over conv ±4 b16 50 ep (blk4_t3d_c4_gdl100_lam0_b16_50ep) by distance to visible px (conv − run): 1 px +0.01634 (n 138,579), 2-3 px +0.01212 (n 117,729), 4-7 px +0.01039 (n 157,936), 8-15 px +0.00937 (n 179,608), 16+ px +0.00981 (n 468,470).

## blk7 postrun: blk4_tokdec_krig_signed (Black Sea 4 km; 2026-10-26 11:04 UTC; analyses/io7/postrun.py)
- Run: store blk4, crop True, val_stride 4, t3d_stem tokdec, tokdec_blend False, tokdec_spatial 0, tokdec_krig True, krig_signed True, physmem False, physmem_stride2 False, context 8, seed 0, epochs 60, batch 8, lr 0.001, lam 0.0, best_epoch 53. Val n: 182 days, 1,062,322 hidden px; test n: 183 days, 1,101,920.
- Val / test MAE 0.144113 / 0.135502; RMSE 0.2369 / 0.2302; bias -0.0113 / -0.0243; non-finite 0 / 0.
- Δ vs blk4_tokdec_blend_spatial (0.141189 / 0.133981): val +0.00292, test +0.00152; ≥ 1 key +0.00276 / +0.00137; no key +0.00555 / +0.00425; real-cloud rw +0.00424 / +0.00192.
- Δ vs blk4_physmem (0.142260 / 0.136556): val +0.00185, test -0.00105; ≥ 1 key +0.00143 / -0.00118; no key +0.00866 / +0.00120; real-cloud rw +0.00351 / -0.00124.
- Δ vs blk4_tokdec_blend (0.143159 / 0.135924): val +0.00095, test -0.00042; ≥ 1 key +0.00082 / -0.00044; no key +0.00312 / -0.00012; real-cloud rw +0.00154 / -0.00052.
- Δ vs blk4_tokdec_krig (0.145181 / 0.136485): val -0.00107, test -0.00098; ≥ 1 key -0.00139 / -0.00076; no key +0.00407 / -0.00498; real-cloud rw -0.00072 / -0.00190.
- Δ vs blk4_t3d_c4_gdl100_lam0_b16_50ep (0.156113 / 0.150893): val -0.01200, test -0.01539; ≥ 1 key -0.01174 / -0.01466; no key -0.01615 / -0.02836; real-cloud rw -0.01234 / -0.01738.
- Δ vs blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (0.149787 / 0.142998): val -0.00567, test -0.00750; ≥ 1 key -0.00550 / -0.00681; no key -0.00841 / -0.01955; real-cloud rw -0.00645 / -0.00991.
- Key split val / test: ≥ 1 key 0.13694 (n 1,000,176) / 0.12899 (n 1,042,846); no key 0.25963 (n 62,146) / 0.25052 (n 59,074).
- Real-cloud reweighted MAE val / test: 0.16884 / 0.16246 (target: val real-cloud px, n 2,340,558; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.16112); blk4_tokdec_blend_spatial 0.16460 / 0.16054, blk4_physmem 0.16533 / 0.16370, blk4_tokdec_blend 0.16730 / 0.16299, blk4_tokdec_krig 0.16956 / 0.16436, blk4_t3d_c4_gdl100_lam0_b16_50ep 0.18118 / 0.17985, blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 0.17529 / 0.17237.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.564/0.659, 0.625/0.740, 0.696/0.814; test 0.600/0.683, 0.660/0.760, 0.731/0.833 (val lag-1 pairs 2,067,873).
- Note: blk4_t3d_* references use another recipe (conv ±4, batch 16, 50 epochs; the _sl_lam0p3_..._cont50 run: λ 0.3, semi-Lagrangian, 50 epochs then two 50-epoch continuations, the last at lr 1e-4); blk7 runs: batch 8, 60-epoch cosine, lr 1e-3, λ 0. Scored days: every 4th val / test day (--val-stride 4).
- Val MAE gain over conv ±4 b16 50 ep (blk4_t3d_c4_gdl100_lam0_b16_50ep) by distance to visible px (conv − run): 1 px +0.01608 (n 138,579), 2-3 px +0.01208 (n 117,729), 4-7 px +0.01097 (n 157,936), 8-15 px +0.01178 (n 179,608), 16+ px +0.01121 (n 468,470).

## io7 ensembles, bridge #29 step 1 (IO; 2026-10-26 19:08 UTC; eval only; analyses/io7/ensemble/report.py)
- Assumptions:
  - ens7_mean = per-px mean of io6_tokdec_blend_spatial, io4_tokdec_blend, io3_ptok4, io2_base_eval, io2_seed1, io2_seed2, io7_physmem. ens11_mean adds io6_tokdec_blend_spatial_s1, io6_tokdec_blend_spatial_s2, io7_wmattn, io7_analog: every IO run with a postrun result in analyses/io7/results/ (all 60-epoch IO recipe runs). Other full-length IO runs with preds.npz (e.g. io6_tokdec_krig, io6_tokdec_krig_signed, io4_tokdec, io2_tattn8, io5_tokdec_blend_up0p3) are not included.
  - Val halves by year: A = (2001, 2013) (728 days, 2,022,116 hidden px), B = (2007, 2018) (730 days, 2,976,526 hidden px). Year split chosen over alternate days because adjacent days share clouds and fields.
  - Fitted weights: over all 11 members, minimising hidden-px MAE on one half (SLSQP on sqrt(r² + ε²), ε 1e-3 then 1e-4; starts: uniform and each member-dominant vertex; lowest true MAE kept; weights < 1e-4 set to 0, renormalised). fitA = fit on half A.
  - Scoring as analyses/io7/postrun.py (rw = real-cloud reweighted, val real-cloud target applied to test too). Non-finite member preds on hidden px: 0.

| model set | val | test | val ≥ 1 key | val no key | test no key | val rw | test rw | grad ratio/corr lag 1 val |
|---|---|---|---|---|---|---|---|---|
| io6_tokdec_blend_spatial | 0.12706 | 0.12752 | 0.11014 | 0.24249 | 0.26879 | 0.15182 | 0.16962 | 0.546/0.629 |
| ens6_mean | 0.12520 | 0.12539 | 0.10790 | 0.24323 | 0.27109 | 0.15027 | 0.16826 | 0.531/0.644 |
| ens7_mean | 0.12464 | 0.12483 | 0.10766 | 0.24054 | 0.26706 | 0.14935 | 0.16690 | 0.531/0.644 |
| ens11_mean | 0.12418 | 0.12444 | 0.10750 | 0.23796 | 0.26441 | 0.14852 | 0.16592 | 0.531/0.644 |
| fitA_ens11 | 0.12402 | 0.12430 | 0.10773 | 0.23518 | 0.25965 | 0.14796 | 0.16477 | 0.533/0.642 |
| fitB_ens11 | 0.12399 | 0.12428 | 0.10749 | 0.23658 | 0.26216 | 0.14814 | 0.16531 | 0.531/0.644 |

- n: val 4,998,642 hidden px (1458 days; ≥ 1 key 4,359,604, no key 639,038); test 5,522,954 (1463 days). fitA/fitB val rows include their own fitting half (in-sample); see the half table for out-of-sample.
- Gradient ratio/corr lags 1, 2, 4 px (val; test): ens6_mean 0.531/0.644, 0.633/0.745, 0.730/0.823; 0.549/0.648, 0.653/0.751, 0.749/0.830; ens7_mean 0.531/0.644, 0.633/0.745, 0.730/0.823; 0.548/0.648, 0.653/0.751, 0.749/0.830; ens11_mean 0.531/0.644, 0.633/0.745, 0.731/0.824; 0.548/0.647, 0.653/0.751, 0.750/0.830; fitA_ens11 0.533/0.642, 0.636/0.743, 0.734/0.823; 0.551/0.644, 0.655/0.748, 0.752/0.829; fitB_ens11 0.531/0.644, 0.634/0.745, 0.732/0.824; 0.549/0.646, 0.654/0.750, 0.750/0.830.
- Fitted weights (zeros omitted): fitA_ens11: io6_tokdec_blend_spatial 0.097, io4_tokdec_blend 0.006, io3_ptok4 0.036, io2_base_eval 0.004, io2_seed1 0.060, io2_seed2 0.050, io7_physmem 0.311, io6_tokdec_blend_spatial_s1 0.145, io6_tokdec_blend_spatial_s2 0.144, io7_wmattn 0.041, io7_analog 0.106; fitB_ens11: io6_tokdec_blend_spatial 0.018, io4_tokdec_blend 0.035, io3_ptok4 0.062, io2_base_eval 0.129, io2_seed1 0.053, io2_seed2 0.070, io7_physmem 0.232, io6_tokdec_blend_spatial_s1 0.130, io6_tokdec_blend_spatial_s2 0.129, io7_wmattn 0.053, io7_analog 0.090.

| combination | val half A | val half B | test |
|---|---|---|---|
| ens7_mean | 0.13046 | 0.12069 | 0.12483 |
| ens11_mean | 0.12985 | 0.12033 | 0.12444 |
| fitA_ens11 | 0.12937 (fit) | 0.12039 | 0.12430 |
| fitB_ens11 | 0.12959 | 0.12019 (fit) | 0.12428 |

- Half table n: A 2,022,116 px, B 2,976,526 px, test 5,522,954 px. Out-of-sample fitted − ens11_mean: half B (fitA) +0.00006, half A (fitB) -0.00026; test fitA -0.00014, fitB -0.00016.
- Members' half MAE (A / B): io6_tokdec_blend_spatial 0.1323/0.1235, io4_tokdec_blend 0.1331/0.1238, io3_ptok4 0.1341/0.1242, io2_base_eval 0.1371/0.1254, io2_seed1 0.1365/0.1258, io2_seed2 0.1368/0.1260, io7_physmem 0.1316/0.1232, io6_tokdec_blend_spatial_s1 0.1321/0.1231, io6_tokdec_blend_spatial_s2 0.1321/0.1229, io7_wmattn 0.1336/0.1242, io7_analog 0.1326/0.1234.
- Results results/ensemble.json; figure figures/ens_models.png.

## io7 postrun: io7_physmem_s1 (IO; 2026-10-26 22:41 UTC; analyses/io7/postrun.py)
- Run: t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, seed 1, epochs 60, batch 8, best_epoch 55. Val n: 1458 days, 4,998,642 hidden px; test n: 1463 days, 5,522,954.
- Val / test MAE 0.126929 / 0.127905; RMSE 0.2194 / 0.2269; bias -0.0189 / -0.0197; non-finite 0 / 0.
- Δ vs io6_tokdec_blend_spatial (0.127060 / 0.127516): val -0.00013, test +0.00039; ≥ 1 key +0.00041 / +0.00067; no key -0.00382 / -0.00278; real-cloud rw -0.00059 / -0.00011.
- Δ vs io4_tokdec_blend (0.127585 / 0.127977): val -0.00066, test -0.00007; ≥ 1 key +0.00014 / +0.00052; no key -0.00610 / -0.00679; real-cloud rw -0.00135 / -0.00119.
- Key split val / test: ≥ 1 key 0.11055 (n 4,359,604) / 0.11569 (n 5,074,031); no key 0.23867 (n 639,038) / 0.26602 (n 448,923).
- Real-cloud reweighted MAE val / test: 0.15123 / 0.16951 (target: val real-cloud px, n 17,084,105; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.15177); io6_tokdec_blend_spatial 0.15182 / 0.16962, io4_tokdec_blend 0.15258 / 0.17071.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.546/0.627, 0.648/0.730, 0.745/0.813; test 0.564/0.628, 0.667/0.734, 0.762/0.819 (val lag-1 pairs 8,403,189).
- Seed context io7_physmem (n 2: io7_physmem, io7_physmem_s1): val MAE mean 0.126746 [0.126563, 0.126929], run − mean +0.00018 (inside range); test MAE mean 0.127395 [0.126886, 0.127905], run − mean +0.00051 (inside range); real-cloud rw val mean 0.151042 [0.150853, 0.151231], run − mean +0.00019 (inside range); test mean 0.168481 [0.167446, 0.169515], run − mean +0.00103 (inside range).
- Seed context io6_spatial (n 3: io6_tokdec_blend_spatial, io6_tokdec_blend_spatial_s1, io6_tokdec_blend_spatial_s2): val MAE mean 0.126802 [0.126605, 0.127060], run − mean +0.00013 (inside range); test MAE mean 0.127369 [0.127239, 0.127516], run − mean +0.00054 (outside range); real-cloud rw val mean 0.151392 [0.151021, 0.151820], run − mean -0.00016 (inside range); test mean 0.169290 [0.168915, 0.169622], run − mean +0.00022 (inside range).
- Val MAE gain over conv ±4 seed 0 by distance to visible px (conv − run): 1 px +0.00032 (n 1,213,097), 2-3 px +0.00014 (n 898,560), 4-7 px +0.00091 (n 1,048,246), 8-15 px +0.00249 (n 880,930), 16+ px +0.01306 (n 957,809).

## io7 postrun: io8_physmem_rwloss (IO; 2026-10-27 01:42 UTC; analyses/io7/postrun.py)
- Run: t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, seed 0, epochs 60, batch 8, best_epoch 47, physmem True, rw_loss True, rw_cap 10.0, bsmem False, ptok_patch 4. Val n: 1458 days, 4,998,642 hidden px; test n: 1463 days, 5,522,954.
- Val / test MAE 0.127378 / 0.127966; RMSE 0.2191 / 0.2258; bias -0.0186 / -0.0191; non-finite 0 / 0.
- Δ vs io6_tokdec_blend_spatial (0.127060 / 0.127516): val +0.00032, test +0.00045; ≥ 1 key +0.00065 / +0.00082; no key -0.00194 / -0.00379; real-cloud rw -0.00002 / -0.00042.
- Δ vs io4_tokdec_blend (0.127585 / 0.127977): val -0.00021, test -0.00001; ≥ 1 key +0.00038 / +0.00068; no key -0.00422 / -0.00780; real-cloud rw -0.00079 / -0.00151.
- Key split val / test: ≥ 1 key 0.11079 (n 4,359,604) / 0.11584 (n 5,074,031); no key 0.24054 (n 639,038) / 0.26501 (n 448,923).
- Real-cloud reweighted MAE val / test: 0.15180 / 0.16920 (target: val real-cloud px, n 17,084,105; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.15159); io6_tokdec_blend_spatial 0.15182 / 0.16962, io4_tokdec_blend 0.15258 / 0.17071.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.547/0.629, 0.650/0.732, 0.748/0.815; test 0.562/0.631, 0.667/0.737, 0.763/0.820 (val lag-1 pairs 8,403,189).
- Seed context io7_physmem (n 2: io7_physmem, io7_physmem_s1): val MAE mean 0.126746 [0.126563, 0.126929], run − mean +0.00063 (outside range); test MAE mean 0.127395 [0.126886, 0.127905], run − mean +0.00057 (outside range); real-cloud rw val mean 0.151042 [0.150853, 0.151231], run − mean +0.00075 (outside range); test mean 0.168481 [0.167446, 0.169515], run − mean +0.00072 (inside range).
- Seed context io6_spatial (n 3: io6_tokdec_blend_spatial, io6_tokdec_blend_spatial_s1, io6_tokdec_blend_spatial_s2): val MAE mean 0.126802 [0.126605, 0.127060], run − mean +0.00058 (outside range); test MAE mean 0.127369 [0.127239, 0.127516], run − mean +0.00060 (outside range); real-cloud rw val mean 0.151392 [0.151021, 0.151820], run − mean +0.00041 (inside range); test mean 0.169290 [0.168915, 0.169622], run − mean -0.00009 (inside range).
- Val MAE gain over conv ±4 seed 0 by distance to visible px (conv − run): 1 px -0.00022 (n 1,213,097), 2-3 px -0.00012 (n 898,560), 4-7 px +0.00066 (n 1,048,246), 8-15 px +0.00254 (n 880,930), 16+ px +0.01188 (n 957,809).

## blk7 postrun: blk8_spatial_p12 (Black Sea 4 km; 2026-10-27 02:38 UTC; analyses/io7/postrun.py)
- Run: store blk4, crop True, val_stride 4, t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, tokdec_krig False, krig_signed False, physmem False, physmem_stride2 False, context 8, seed 0, epochs 60, batch 8, lr 0.001, lam 0.0, best_epoch 46, rw_loss False, rw_cap 10.0, bsmem False, ptok_patch 12. Val n: 182 days, 1,062,322 hidden px; test n: 183 days, 1,101,920.
- Val / test MAE 0.149134 / 0.141411; RMSE 0.2409 / 0.2388; bias -0.0205 / -0.0345; non-finite 0 / 0.
- Δ vs blk4_tokdec_blend_spatial (0.141189 / 0.133981): val +0.00794, test +0.00743; ≥ 1 key +0.00860 / +0.00766; no key -0.00264 / +0.00342; real-cloud rw +0.00819 / +0.00819.
- Δ vs blk4_physmem (0.142260 / 0.136556): val +0.00687, test +0.00485; ≥ 1 key +0.00727 / +0.00511; no key +0.00046 / +0.00038; real-cloud rw +0.00746 / +0.00503.
- Δ vs blk4_tokdec_blend (0.143159 / 0.135924): val +0.00597, test +0.00549; ≥ 1 key +0.00666 / +0.00585; no key -0.00507 / -0.00094; real-cloud rw +0.00549 / +0.00574.
- Δ vs blk4_tokdec_krig (0.145181 / 0.136485): val +0.00395, test +0.00493; ≥ 1 key +0.00445 / +0.00553; no key -0.00412 / -0.00580; real-cloud rw +0.00323 / +0.00437.
- Δ vs blk4_tokdec_krig_signed (0.144113 / 0.135502): val +0.00502, test +0.00591; ≥ 1 key +0.00584 / +0.00629; no key -0.00819 / -0.00082; real-cloud rw +0.00395 / +0.00627.
- Δ vs blk4_t3d_c4_gdl100_lam0_b16_50ep (0.156113 / 0.150893): val -0.00698, test -0.00948; ≥ 1 key -0.00590 / -0.00837; no key -0.02434 / -0.02918; real-cloud rw -0.00839 / -0.01112.
- Δ vs blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (0.149787 / 0.142998): val -0.00065, test -0.00159; ≥ 1 key +0.00034 / -0.00052; no key -0.01660 / -0.02038; real-cloud rw -0.00250 / -0.00364.
- Key split val / test: ≥ 1 key 0.14278 (n 1,000,176) / 0.13528 (n 1,042,846); no key 0.25144 (n 62,146) / 0.24969 (n 59,074).
- Real-cloud reweighted MAE val / test: 0.17279 / 0.16873 (target: val real-cloud px, n 2,340,558; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.16737); blk4_tokdec_blend_spatial 0.16460 / 0.16054, blk4_physmem 0.16533 / 0.16370, blk4_tokdec_blend 0.16730 / 0.16299, blk4_tokdec_krig 0.16956 / 0.16436, blk4_tokdec_krig_signed 0.16884 / 0.16246, blk4_t3d_c4_gdl100_lam0_b16_50ep 0.18118 / 0.17985, blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 0.17529 / 0.17237.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.546/0.649, 0.604/0.729, 0.677/0.803; test 0.582/0.669, 0.639/0.745, 0.710/0.819 (val lag-1 pairs 2,067,873).
- Note: blk4_t3d_* references use another recipe (conv ±4, batch 16, 50 epochs; the _sl_lam0p3_..._cont50 run: λ 0.3, semi-Lagrangian, 50 epochs then two 50-epoch continuations, the last at lr 1e-4); blk7 runs: batch 8, 60-epoch cosine, lr 1e-3, λ 0. Scored days: every 4th val / test day (--val-stride 4).
- Val MAE gain over conv ±4 b16 50 ep (blk4_t3d_c4_gdl100_lam0_b16_50ep) by distance to visible px (conv − run): 1 px +0.01303 (n 138,579), 2-3 px +0.00835 (n 117,729), 4-7 px +0.00541 (n 157,936), 8-15 px +0.00456 (n 179,608), 16+ px +0.00630 (n 468,470).

## blk7 postrun: blk8_spatial_bsmem (Black Sea 4 km; 2026-10-27 04:14 UTC; analyses/io7/postrun.py)
- Run: store blk4, crop True, val_stride 4, t3d_stem tokdec, tokdec_blend True, tokdec_spatial 2, tokdec_krig False, krig_signed False, physmem False, physmem_stride2 False, context 8, seed 0, epochs 60, batch 8, lr 0.001, lam 0.0, best_epoch 59, rw_loss False, rw_cap 10.0, bsmem True, ptok_patch 4. Val n: 182 days, 1,062,322 hidden px; test n: 183 days, 1,101,920.
- Val / test MAE 0.141390 / 0.134847; RMSE 0.2325 / 0.2302; bias -0.0053 / -0.0232; non-finite 0 / 0.
- Δ vs blk4_tokdec_blend_spatial (0.141189 / 0.133981): val +0.00020, test +0.00087; ≥ 1 key +0.00032 / +0.00064; no key -0.00177 / +0.00484; real-cloud rw +0.00017 / +0.00119.
- Δ vs blk4_physmem (0.142260 / 0.136556): val -0.00087, test -0.00171; ≥ 1 key -0.00101 / -0.00191; no key +0.00133 / +0.00180; real-cloud rw -0.00057 / -0.00197.
- Δ vs blk4_tokdec_blend (0.143159 / 0.135924): val -0.00177, test -0.00108; ≥ 1 key -0.00162 / -0.00117; no key -0.00421 / +0.00048; real-cloud rw -0.00253 / -0.00126.
- Δ vs blk4_tokdec_krig (0.145181 / 0.136485): val -0.00379, test -0.00164; ≥ 1 key -0.00382 / -0.00148; no key -0.00325 / -0.00438; real-cloud rw -0.00480 / -0.00263.
- Δ vs blk4_tokdec_krig_signed (0.144113 / 0.135502): val -0.00272, test -0.00065; ≥ 1 key -0.00244 / -0.00073; no key -0.00732 / +0.00060; real-cloud rw -0.00408 / -0.00073.
- Δ vs blk4_t3d_c4_gdl100_lam0_b16_50ep (0.156113 / 0.150893): val -0.01472, test -0.01605; ≥ 1 key -0.01418 / -0.01538; no key -0.02348 / -0.02776; real-cloud rw -0.01642 / -0.01811.
- Δ vs blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (0.149787 / 0.142998): val -0.00840, test -0.00815; ≥ 1 key -0.00794 / -0.00754; no key -0.01573 / -0.01896; real-cloud rw -0.01053 / -0.01064.
- Key split val / test: ≥ 1 key 0.13450 (n 1,000,176) / 0.12826 (n 1,042,846); no key 0.25231 (n 62,146) / 0.25111 (n 59,074).
- Real-cloud reweighted MAE val / test: 0.16477 / 0.16173 (target: val real-cloud px, n 2,340,558; uncovered mass 0.0000 / 0.0000; test with the test days' real-cloud target 0.16045); blk4_tokdec_blend_spatial 0.16460 / 0.16054, blk4_physmem 0.16533 / 0.16370, blk4_tokdec_blend 0.16730 / 0.16299, blk4_tokdec_krig 0.16956 / 0.16436, blk4_tokdec_krig_signed 0.16884 / 0.16246, blk4_t3d_c4_gdl100_lam0_b16_50ep 0.18118 / 0.17985, blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 0.17529 / 0.17237.
- Gradient ratio/corr lags 1, 2, 4 px: val 0.556/0.662, 0.617/0.742, 0.690/0.816; test 0.592/0.685, 0.653/0.761, 0.725/0.833 (val lag-1 pairs 2,067,873).
- Note: blk4_t3d_* references use another recipe (conv ±4, batch 16, 50 epochs; the _sl_lam0p3_..._cont50 run: λ 0.3, semi-Lagrangian, 50 epochs then two 50-epoch continuations, the last at lr 1e-4); blk7 runs: batch 8, 60-epoch cosine, lr 1e-3, λ 0. Scored days: every 4th val / test day (--val-stride 4).
- Val MAE gain over conv ±4 b16 50 ep (blk4_t3d_c4_gdl100_lam0_b16_50ep) by distance to visible px (conv − run): 1 px +0.01653 (n 138,579), 2-3 px +0.01323 (n 117,729), 4-7 px +0.01195 (n 157,936), 8-15 px +0.01191 (n 179,608), 16+ px +0.01658 (n 468,470).
