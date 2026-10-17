# Training runs: index

Generated 2026-10-17 22:40 by `python -m chl_cnn.runs_index` (196 runs). What each round found, and why it was run: [analyses/FINDINGS.md](../analyses/FINDINGS.md).

- **val / test MAE**: hidden-pixel MAE in ln CHL (fake clouds from a donor day). Best val epoch; test at that epoch. Only comparable within one region and one `val_stride` / `donor_exclude` setting.
- **settings**: non-default flags only (see the trainer docstrings). `stop_epoch=N`: the schedule spans `epochs` but training stopped at N.
- **status**: `weights` = model.pt still on disk (run folder under models/<category>/); otherwise only config/history/metrics are kept, in models/history/.

## cnn (14 runs)

| run | region | settings | val MAE | test MAE | epochs | finished | status |
|---|---|---|---|---|---|---|---|
| baseline_asis | Indian Ocean | ? feat=2 ctx=None epochs=50 batch=8 | 1.5838 | 1.6152 | 26 | 2026-09-30 | done |
| baseline_sst_so | Indian Ocean | ? feat=2 ctx=None epochs=50 batch=8 | 0.3685 | 0.2780 | 43 | 2026-09-30 | done |
| chl_in_fakeclouds | Indian Ocean | ? feat=2 ctx=None epochs=50 batch=8 | 0.2586 | 0.1951 | 50 | 2026-09-30 | done |
| full_alldays_fakeclouds | Indian Ocean | ? feat=2 ctx=None epochs=50 batch=8 | 0.2338 | 0.2386 | 50 | 2026-09-30 | done |
| full_clear_fakeclouds | Indian Ocean | ? feat=2 ctx=None epochs=50 batch=8 | 0.3689 | 0.3586 | 32 | 2026-09-30 | done |
| perf_dilated_allfeat | Indian Ocean | dilated feat=9 ctx=2 width=64 epochs=30 batch=8 | 0.1397 | 0.1418 | 30 | 2026-09-30 | done |
| perf_dilated_base | Indian Ocean | dilated feat=2 ctx=0 width=64 epochs=30 batch=8 | 0.1676 | 0.1714 | 30 | 2026-09-30 | done |
| perf_dilated_climcoord | Indian Ocean | dilated feat=2 ctx=0 width=64 epochs=30 batch=8 | 0.1670 | 0.1720 | 29 | 2026-09-30 | done |
| perf_dilated_ctx2 | Indian Ocean | dilated feat=2 ctx=2 width=64 epochs=30 batch=8 | 0.1415 | 0.1437 | 30 | 2026-09-30 | done |
| perf_dilated_ctx2_leaky | Indian Ocean | dilated feat=2 ctx=2 width=64 epochs=15 context_clouds=none batch=8 | 0.1058 | 0.1126 | 15 | 2026-09-30 | done |
| perf_dilated_huber | Indian Ocean | dilated feat=9 ctx=2 loss=huber width=64 epochs=30 batch=8 | 0.1408 | 0.1429 | 30 | 2026-09-30 | done |
| perf_dilated_mse | Indian Ocean | dilated feat=9 ctx=2 loss=mse width=64 epochs=30 batch=8 | 0.1433 | 0.1454 | 30 | 2026-09-30 | done |
| perf_dilated_vis01 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 epochs=30 batch=8 | 0.1381 | 0.1400 | 30 | 2026-09-30 | done, weights |
| smoke_ctx | Indian Ocean | dilated feat=1 ctx=1 width=64 epochs=1 batch=8 | 0.1899 | 0.1998 | 1 | 2026-09-30 | done |

## pinn/cnn (47 runs)

| run | region | settings | val MAE | test MAE | epochs | finished | status |
|---|---|---|---|---|---|---|---|
| cnn_blk_lam0 | Black Sea 2 km | store=blk dilated feat=8 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input phys=upwind batch=8 | 0.1975 | 0.1831 | 12 | 2026-10-10 | done, weights |
| pinn_lam0 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 epochs=12 batch=8 | 0.1444 | 0.1484 | 12 | 2026-10-01 | done |
| pinn_lam0p1 | Indian Ocean | dilated feat=9 ctx=2 lam=0.1 vis_weight=0.1 width=64 epochs=12 batch=8 | 0.1442 | 0.1484 | 12 | 2026-10-01 | done |
| pinn_lam0p3 | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 epochs=12 batch=8 | 0.1440 | 0.1480 | 12 | 2026-10-01 | done |
| nochl_cloud_lam0 | Indian Ocean | dilated feat=9 ctx=2 width=64 epochs=12 no_chl_input cloud_input batch=8 | 0.2248 | 0.2410 | 12 | 2026-10-02 | done |
| nochl_cloud_lam0p3 | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 width=64 epochs=12 no_chl_input cloud_input batch=8 | 0.2255 | 0.2403 | 12 | 2026-10-02 | done |
| nochl_lam0 | Indian Ocean | dilated feat=9 ctx=2 width=64 epochs=12 no_chl_input batch=8 | 0.2228 | 0.2400 | 12 | 2026-10-02 | done |
| nochl_lam0p3 | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 width=64 epochs=12 no_chl_input batch=8 | 0.2239 | 0.2403 | 12 | 2026-10-02 | done |
| nochl_today_lam0 | Indian Ocean | dilated feat=9 ctx=1 width=64 epochs=12 no_chl_today batch=8 | 0.1621 | 0.1672 | 12 | 2026-10-02 | done |
| nochl_today_lam0p3 | Indian Ocean | dilated feat=9 ctx=1 lam=0.3 width=64 epochs=12 no_chl_today batch=8 | 0.1612 | 0.1666 | 12 | 2026-10-02 | done |
| pinn_lam1p0 | Indian Ocean | dilated feat=9 ctx=2 lam=1.0 vis_weight=0.1 width=64 epochs=12 batch=8 | 0.1450 | 0.1490 | 12 | 2026-10-02 | done |
| pinn_long_lam0 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 epochs=60 batch=8 | 0.1396 | 0.1420 | 60 | 2026-10-02 | done |
| pinn_long_lam0p3 | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 epochs=60 batch=8 | 0.1388 | 0.1410 | 60 | 2026-10-02 | done |
| pinn_long_lam1p0 | Indian Ocean | dilated feat=9 ctx=2 lam=1.0 vis_weight=0.1 width=64 epochs=60 batch=8 | 0.1393 | 0.1403 | 60 | 2026-10-02 | done |
| pinn_v1_all | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 epochs=12 src_clim src_ekman phys_weight=courant batch=8 | 0.1443 | 0.1483 | 12 | 2026-10-02 | done |
| pinn_v1_clim | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 epochs=12 src_clim batch=8 | 0.1441 | 0.1483 | 12 | 2026-10-02 | done |
| pinn_v1_clim_ekman | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 epochs=12 src_clim src_ekman batch=8 | 0.1443 | 0.1484 | 12 | 2026-10-02 | done |
| pinn_v1_courant | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 epochs=12 phys_weight=courant batch=8 | 0.1441 | 0.1482 | 12 | 2026-10-02 | done |
| pinn_v1_ekman | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 epochs=12 src_ekman batch=8 | 0.1442 | 0.1483 | 12 | 2026-10-02 | done |
| pinn_v2_lam0 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=60 year_input batch=8 | 0.1365 | 0.1390 | 60 | 2026-10-03 | done, weights |
| pinn_v2_lam0p3 | Indian Ocean | dilated feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 donor_window=15 epochs=60 year_input batch=8 | 0.1370 | 0.1383 | 60 | 2026-10-03 | done |
| pinn_v2_lam1p0 | Indian Ocean | dilated feat=9 ctx=2 lam=1.0 vis_weight=0.1 width=64 donor_window=15 epochs=60 year_input batch=8 | 0.1378 | 0.1391 | 60 | 2026-10-03 | done |
| pinn_v3_gdl | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=1.0 width=64 donor_window=15 epochs=20 year_input batch=8 | 0.1401 | 0.1429 | 20 | 2026-10-03 | done |
| pinn_v3_spec | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 spectral=4 donor_window=15 epochs=20 year_input batch=8 | 0.1502 | 0.1554 | 20 | 2026-10-03 | done |
| pinn_v3_ctrl | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=20 year_input batch=8 | 0.1410 | 0.1439 | 20 | 2026-10-04 | done |
| pinn_v3_spec_gdl | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=1.0 width=64 spectral=4 donor_window=15 epochs=20 year_input batch=8 | 0.1502 | 0.1542 | 20 | 2026-10-04 | done |
| s1_base | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1441 | 0.1475 | 12 | 2026-10-05 | done |
| s1_ctx3 | Indian Ocean | dilated feat=9 ctx=3 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1433 | 0.1462 | 12 | 2026-10-05 | done |
| s1_drop01 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1451 | 0.1489 | 12 | 2026-10-05 | done |
| s1_gdl10 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=10.0 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1385 | 0.1405 | 12 | 2026-10-05 | done |
| s1_gdl3 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=3.0 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1416 | 0.1444 | 12 | 2026-10-05 | done |
| s1_huber | Indian Ocean | dilated feat=9 ctx=2 loss=huber vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1448 | 0.1473 | 12 | 2026-10-05 | done |
| s1_lr2e3 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input batch=8 lr=0.002 | 0.1444 | 0.1479 | 12 | 2026-10-05 | done |
| s1_seed1 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1441 | 0.1476 | 12 | 2026-10-05 | done |
| s1_w96 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=96 donor_window=15 epochs=12 year_input batch=8 | 0.1431 | 0.1467 | 12 | 2026-10-05 | done |
| s2_gdl100 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1374 | 0.1395 | 12 | 2026-10-05 | done |
| s2_gdl100_w96 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=100.0 width=96 donor_window=15 epochs=12 year_input batch=8 | 0.1357 | 0.1376 | 12 | 2026-10-05 | done |
| s2_gdl10_w96 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=10.0 width=96 donor_window=15 epochs=12 year_input batch=8 | 0.1381 | 0.1399 | 12 | 2026-10-05 | done |
| s2_gdl30 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=30.0 width=64 donor_window=15 epochs=12 year_input batch=8 | 0.1382 | 0.1402 | 12 | 2026-10-05 | done |
| s2_gdl300_w96 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=300.0 width=96 donor_window=15 epochs=12 year_input batch=8 | 0.1369 | 0.1394 | 12 | 2026-10-05 | done |
| s2_gdl30_w96 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=30.0 width=96 donor_window=15 epochs=12 year_input batch=8 | 0.1363 | 0.1381 | 12 | 2026-10-05 | done |
| s2_gdl3_w96 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=3.0 width=96 donor_window=15 epochs=12 year_input batch=8 | 0.1404 | 0.1430 | 12 | 2026-10-05 | done |
| hy_ctrl | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=20 year_input batch=8 | 0.1410 | 0.1439 | 20 | 2026-10-06 | done |
| hy_fill_sstctx2 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=20 year_input diff_fill=tc_sst_ctx2 batch=8 | 0.1412 | 0.1434 | 20 | 2026-10-06 | done |
| s3_gdl100_w96 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=100.0 width=96 donor_window=15 epochs=60 year_input batch=8 | 0.1342 | 0.1359 | 60 | 2026-10-06 | done |
| s3_gdl100_w96_seed1 | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 grad_weight=100.0 width=96 donor_window=15 epochs=60 year_input batch=8 | 0.1342 | 0.1358 | 60 | 2026-10-06 | done |
| cnn_12ep | Indian Ocean | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input phys=upwind batch=8 | 0.1439 | 0.1472 | 12 | 2026-10-11 | done, weights |

## pinn/unet (10 runs)

| run | region | settings | val MAE | test MAE | epochs | finished | status |
|---|---|---|---|---|---|---|---|
| unet_d2w64_lam0 | Indian Ocean | unet feat=9 ctx=2 vis_weight=0.1 width=64 unet_depth=2 epochs=30 batch=8 | 0.1426 | 0.1457 | 30 | 2026-10-02 | done |
| unet_d2w64_lam0p3 | Indian Ocean | unet feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=64 unet_depth=2 epochs=30 batch=8 | 0.1426 | 0.1455 | 30 | 2026-10-02 | done |
| unet_mtcg_lam0 | Indian Ocean | unet feat=4 ctx=1 loss=mse width=64 unet_depth=2 epochs=30 batch=8 | 0.1508 | 0.1543 | 30 | 2026-10-02 | done |
| unet_mtcg_lam0p3 | Indian Ocean | unet feat=4 ctx=1 lam=0.3 loss=mse width=64 unet_depth=2 epochs=30 batch=8 | 0.1535 | 0.1572 | 30 | 2026-10-02 | done |
| unet_d3w32_lam0 | Indian Ocean | unet feat=9 ctx=2 vis_weight=0.1 width=32 unet_depth=3 epochs=30 batch=8 | 0.1442 | 0.1478 | 30 | 2026-10-03 | done |
| unet_d3w32_lam0p3 | Indian Ocean | unet feat=9 ctx=2 lam=0.3 vis_weight=0.1 width=32 unet_depth=3 epochs=30 batch=8 | 0.1446 | 0.1483 | 30 | 2026-10-03 | done |
| unet_c1_d3w64 | Indian Ocean | unet feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 unet_depth=3 epochs=20 year_input phys=upwind batch=8 | 0.1431 | 0.1463 | 20 | 2026-10-14 | done, weights |
| unet_c2_ctx4 | Indian Ocean | unet feat=9 ctx=4 vis_weight=0.1 width=64 donor_window=15 unet_depth=3 epochs=20 year_input phys=upwind batch=8 | 0.1417 | 0.1447 | 20 | 2026-10-14 | done, weights |
| unet_c3_lr2e3 | Indian Ocean | unet feat=9 ctx=4 vis_weight=0.1 width=64 donor_window=15 unet_depth=3 epochs=20 year_input phys=upwind batch=8 lr=0.002 | 0.1431 | 0.1462 | 20 | 2026-10-14 | done, weights |
| unet_c4_b4 | Indian Ocean | unet feat=9 ctx=4 vis_weight=0.1 width=64 donor_window=15 unet_depth=3 epochs=20 year_input phys=upwind batch=4 | 0.1429 | 0.1460 | 20 | 2026-10-14 | done, weights |

## pinn/deq (2 runs)

| run | region | settings | val MAE | test MAE | epochs | finished | status |
|---|---|---|---|---|---|---|---|
| deq_unroll_k1 | Indian Ocean | deq feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input phys=upwind batch=8 | 0.1449 | 0.1484 | 12 | 2026-10-11 | done, weights |
| deq_and24 | Indian Ocean | deq feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=12 year_input phys=upwind batch=8 | 0.1581 | 0.1632 | 12 | 2026-10-13 | done, weights |

## pinn/t3d (22 runs)

| run | region | settings | val MAE | test MAE | epochs | finished | status |
|---|---|---|---|---|---|---|---|
| blk_t3d_c4_gdl100_lam0 | Black Sea 2 km | store=blk temporal3d feat=8 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input phys=semi-lagrangian batch=4 | 0.1627 | 0.1552 | 20 | 2026-10-15 | done, weights |
| blk_t3d_c3_b8_s12 | Black Sea 2 km | store=blk temporal3d feat=8 ctx=3 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input blocks=8 phys=semi-lagrangian batch=4 stop_epoch=12 | 0.1754 | 0.1702 | 12 | 2026-10-16 | done, weights |
| blk_t3d_c4_b4_s12 | Black Sea 2 km | store=blk temporal3d feat=8 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input blocks=4 phys=semi-lagrangian batch=4 stop_epoch=12 | 0.1870 | 0.1793 | 12 | 2026-10-16 | done, weights |
| blk_t3d_c4_b5_s12 | Black Sea 2 km | store=blk temporal3d feat=8 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input blocks=5 phys=semi-lagrangian batch=4 stop_epoch=12 | 0.1829 | 0.1754 | 12 | 2026-10-16 | done, weights |
| blk_t3d_c4_b6_s12 | Black Sea 2 km | store=blk temporal3d feat=8 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input blocks=6 phys=semi-lagrangian batch=4 stop_epoch=12 | 0.1779 | 0.1682 | 12 | 2026-10-16 | done, weights |
| blk_t3d_c4_b7_s12 | Black Sea 2 km | store=blk temporal3d feat=8 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input blocks=7 phys=semi-lagrangian batch=4 stop_epoch=12 | 0.1813 | 0.1749 | 12 | 2026-10-16 | done, weights |
| blk_t3d_c4_gdl100_sl_lam0p3 | Black Sea 2 km | store=blk temporal3d feat=8 ctx=4 lam=0.3 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input phys=semi-lagrangian batch=4 | 0.1606 | 0.1525 | 20 | 2026-10-16 | done, weights |
| blk4_t3d_c4_gdl100_lam0 | Black Sea 4 km | store=blk4 temporal3d feat=8 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input blocks=8 phys=semi-lagrangian batch=8 | 0.1638 | 0.1585 | 20 | 2026-10-16 | done, weights |
| blk4_t3d_c4_gdl100_sl_lam0p3 | Black Sea 4 km | store=blk4 temporal3d feat=8 ctx=4 lam=0.3 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input blocks=8 phys=semi-lagrangian batch=8 | 0.1637 | 0.1585 | 20 | 2026-10-16 | done, weights |
| blk4_t3d_c4_gdl100_lam0_b16_50ep | Black Sea 4 km | store=blk4 temporal3d feat=8 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=50 year_input blocks=8 phys=semi-lagrangian batch=16 | 0.1561 | 0.1509 | 50 | 2026-10-17 | done, weights |
| blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep | Black Sea 4 km | store=blk4 temporal3d feat=8 ctx=4 lam=0.3 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=50 year_input blocks=8 phys=semi-lagrangian batch=16 | 0.1546 | 0.1490 | 50 | 2026-10-17 | done, weights |
| blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50 | Black Sea 4 km | store=blk4 temporal3d feat=8 ctx=4 lam=0.3 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=50 year_input blocks=8 phys=semi-lagrangian batch=16 continue_from=blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep | 0.1499 | 0.1437 | 50 | 2026-10-17 | done, weights |
| blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 | Black Sea 4 km | store=blk4 temporal3d feat=8 ctx=4 lam=0.3 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=50 year_input blocks=8 phys=semi-lagrangian batch=16 lr=0.0001 continue_from=blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50 | 0.1498 | 0.1430 | 50 | 2026-10-17 | done, weights |
| blk4_t3d_c4_gdl100_sl_lam1_b16_stop12 | Black Sea 4 km | store=blk4 temporal3d feat=8 ctx=4 lam=1.0 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=50 year_input blocks=8 phys=semi-lagrangian batch=16 stop_epoch=12 | 0.1978 | 0.1893 | 12 | 2026-10-17 | done, weights |
| t3d_dil_c2 | Indian Ocean | temporal3d feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 epochs=20 year_input phys=upwind batch=8 | 0.1406 | 0.1434 | 20 | 2026-10-14 | done, weights |
| t3d_dil_c4 | Indian Ocean | temporal3d feat=9 ctx=4 vis_weight=0.1 width=64 donor_window=15 epochs=20 year_input phys=upwind batch=8 | 0.1378 | 0.1398 | 20 | 2026-10-14 | done, weights |
| t3d_dil_c4_60ep | Indian Ocean | temporal3d feat=9 ctx=4 vis_weight=0.1 width=64 donor_window=15 epochs=60 year_input phys=upwind batch=8 | 0.1331 | 0.1344 | 60 | 2026-10-14 | done, weights |
| t3d_dil_c4_60ep_gdl100 | Indian Ocean | temporal3d feat=9 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=60 year_input phys=upwind batch=8 | 0.1302 | 0.1305 | 60 | 2026-10-15 | done, weights |
| t3d_dil_c4_60ep_gdl100_dx8 | Indian Ocean | temporal3d feat=9 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=12 year_input blocks=8 phys=upwind donor_exclude=8 batch=8 init_from=t3d_dil_c4_60ep_gdl100 | nan | 0.1313 |  | 2026-10-16 | done, weights |
| t3d_fc_b8 | Indian Ocean | temporal3d feat=9 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input no_chl_today window=t-8..t+0 phys=upwind context_clouds=none batch=8 | 0.1251 | 0.1321 | 20 | 2026-10-16 | done, weights |
| t3d_fc_b8_dx8 | Indian Ocean | temporal3d feat=9 ctx=4 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=12 year_input no_chl_today blocks=8 window=t-8..t+0 phys=upwind context_clouds=none donor_exclude=8 batch=8 init_from=t3d_fc_b8 | nan | 0.1323 |  | 2026-10-16 | done, weights |
| t3d_nc_b8 | Indian Ocean | temporal3d feat=9 ctx=2 vis_weight=0.1 grad_weight=100.0 width=64 donor_window=15 epochs=20 year_input blocks=8 window=t-8..t+0 phys=upwind context_clouds=none donor_exclude=8 batch=8 | 0.1090 | 0.1126 | 20 | 2026-10-16 | done, weights |

## spatial (85 runs)

| run | region | settings | val MAE | test MAE | epochs | finished | status |
|---|---|---|---|---|---|---|---|
| blk_conv_ctx2_cf | Black Sea 2 km | conv context=2 ctx_feat train_stride=4 ep=3 | 0.1888 |  | 4 | 2026-10-09 | done, weights |
| blk_conv_ctx2_cf_lag | Black Sea 2 km | conv context=2 ctx_feat lagrangian train_stride=4 ep=3 | 0.1909 |  | 4 | 2026-10-09 | done, weights |
| blk_conv_ctx4_cf | Black Sea 2 km | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1843 |  | 4 | 2026-10-09 | done, weights |
| blk_final_conv_ctx4_cf | Black Sea 2 km | conv context=4 ctx_feat ep=6 | 0.1772 |  | 7 | 2026-10-09 | done, weights |
| blk_iso_ctx0 | Black Sea 2 km | iso train_stride=4 ep=3 | 0.2808 |  | 4 | 2026-10-09 | done, weights |
| blk_mlpx_ctx2 | Black Sea 2 km | mlp context=2 extra_feats train_stride=4 ep=3 | 0.1911 |  | 4 | 2026-10-09 | done, weights |
| blk_sst_ctx0 | Black Sea 2 km | sst train_stride=4 ep=3 | 0.2674 |  | 4 | 2026-10-09 | done, weights |
| blk_sst_ctx2 | Black Sea 2 km | sst context=2 train_stride=4 ep=3 | 0.2042 |  | 4 | 2026-10-09 | done, weights |
| blk_sst_ctx2_lag | Black Sea 2 km | sst context=2 lagrangian train_stride=4 ep=3 | 0.2039 |  | 4 | 2026-10-09 | done, weights |
| blk_sst_ctx2_lag15 | Black Sea 2 km | sst context=2 lagrangian train_stride=4 ep=3 | 0.2056 |  | 4 | 2026-10-09 | done, weights |
| blk_sst_ctx4 | Black Sea 2 km | sst context=4 train_stride=4 ep=3 | 0.1970 |  | 4 | 2026-10-09 | done, weights |
| r2_ctrl | Black Sea 2 km | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1813 |  | 4 | 2026-10-09 | done, weights |
| r2_depth | Black Sea 2 km | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1814 |  | 4 | 2026-10-09 | done, weights |
| r2_normd | Black Sea 2 km | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1821 |  | 4 | 2026-10-09 | done, weights |
| r2_sal | Black Sea 2 km | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1801 |  | 4 | 2026-10-09 | done, weights |
| smk_all | Black Sea 2 km | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2002 |  | 2 | 2026-10-09 | done, weights |
| smk_all_mlp | Black Sea 2 km | mlp context=4 ctx_feat extra_feats train_stride=60 ep=1 | 0.2050 |  | 2 | 2026-10-09 | done, weights |
| smk_all_mlp_plain | Black Sea 2 km | mlp context=4 ctx_feat train_stride=60 ep=1 | 0.2050 |  | 2 | 2026-10-09 | done, weights |
| smk_ctrl | Black Sea 2 km | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2024 |  | 2 | 2026-10-09 | done, weights |
| smk_depth | Black Sea 2 km | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2022 |  | 2 | 2026-10-09 | done, weights |
| smk_normd | Black Sea 2 km | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2011 |  | 2 | 2026-10-09 | done, weights |
| smk_river | Black Sea 2 km | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2025 |  | 2 | 2026-10-09 | done, weights |
| smk_sal | Black Sea 2 km | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2037 |  | 2 | 2026-10-09 | done, weights |
| r2_all | Black Sea 2 km | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1830 |  | 4 | 2026-10-10 | done, weights |
| r2_all_lr1e2 | Black Sea 2 km | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1832 |  | 4 | 2026-10-10 | done, weights |
| r2_river | Black Sea 2 km | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1803 |  | 4 | 2026-10-10 | done, weights |
| dn_abl_nocur | Indian Ocean | mlp train_stride=4 dkw={"drop": ["u_cur","v_cur","cur_ok"]} ep=3 | 0.1987 |  | 4 | 2026-10-06 | done |
| dn_abl_nodepth | Indian Ocean | mlp train_stride=4 dkw={"drop": ["log_depth"]} ep=3 | 0.2182 |  | 4 | 2026-10-06 | done |
| dn_abl_none | Indian Ocean | mlp train_stride=4 ep=3 | 0.1985 |  | 4 | 2026-10-06 | done |
| dn_abl_nosst | Indian Ocean | mlp train_stride=4 dkw={"drop": ["sst_gx","sst_gy","sst_ok"]} ep=3 | 0.1987 |  | 4 | 2026-10-06 | done |
| dn_abl_nowind | Indian Ocean | mlp train_stride=4 dkw={"drop": ["u_wind","v_wind"]} ep=3 | 0.1978 |  | 4 | 2026-10-06 | done |
| dn_conv_d3_dil1 | Indian Ocean | conv train_stride=4 ep=8 | 0.1901 |  | 9 | 2026-10-06 | done |
| dn_conv_d3_dil2 | Indian Ocean | conv train_stride=4 ep=8 | 0.1881 |  | 9 | 2026-10-06 | done |
| dn_conv_d3_small | Indian Ocean | conv train_stride=4 dkw={"channels": [0,1,7,9,10]} ep=8 | 0.1929 |  | 9 | 2026-10-06 | done |
| dn_conv_d5_dil1 | Indian Ocean | conv train_stride=4 ep=8 | 0.1890 |  | 9 | 2026-10-06 | done |
| dn_mlp_lr1e2_d16 | Indian Ocean | mlp train_stride=4 ep=8 | 0.1919 |  | 9 | 2026-10-06 | done |
| dn_mlp_lr1e2_d4 | Indian Ocean | mlp train_stride=4 ep=8 | 0.1919 |  | 9 | 2026-10-06 | done |
| dn_mlp_lr3e3_d16 | Indian Ocean | mlp train_stride=4 ep=8 | 0.1954 |  | 9 | 2026-10-06 | done |
| dn_mlp_lr3e3_d4 | Indian Ocean | mlp train_stride=4 ep=8 | 0.1958 |  | 9 | 2026-10-06 | done |
| dn_mlpx_lr1e2_d16 | Indian Ocean | mlp extra_feats train_stride=4 ep=8 | 0.1891 |  | 9 | 2026-10-06 | done |
| dn_mlpx_lr1e2_d4 | Indian Ocean | mlp extra_feats train_stride=4 ep=8 | 0.1899 |  | 9 | 2026-10-06 | done |
| dn_mlpx_lr3e3_d16 | Indian Ocean | mlp extra_feats train_stride=4 ep=8 | 0.1942 |  | 9 | 2026-10-06 | done |
| dn_mlpx_lr3e3_d4 | Indian Ocean | mlp extra_feats train_stride=4 ep=8 | 0.1942 |  | 9 | 2026-10-06 | done |
| dn_resid_lr3e3_d4 | Indian Ocean | mlp train_stride=4 dkw={"sst_residual": true} ep=8 | 0.1931 |  | 9 | 2026-10-06 | done |
| dn_residx_lr3e3_d4 | Indian Ocean | mlp extra_feats train_stride=4 dkw={"sst_residual": true} ep=8 | 0.1919 |  | 9 | 2026-10-06 | done |
| dn_smoke_convch | Indian Ocean | conv train_stride=60 dkw={"channels": [0,1,7,9,10]} ep=1 | 0.2223 |  | 2 | 2026-10-06 | done |
| dn_smoke_drop | Indian Ocean | mlp train_stride=60 dkw={"drop": ["u_cur","v_cur","cur_ok"]} ep=1 | 0.2354 |  | 2 | 2026-10-06 | done |
| dn_smoke_resid | Indian Ocean | mlp train_stride=60 dkw={"sst_residual": true} ep=1 | 0.2204 |  | 2 | 2026-10-06 | done |
| mt_conv_ctx2_cf_seed1 | Indian Ocean | conv context=2 ctx_feat train_stride=4 seed=1 ep=3 | 0.1614 |  | 4 | 2026-10-06 | done |
| mt_convsrc_ctx2 | Indian Ocean | convsrc context=2 ctx_feat source train_stride=4 ep=3 | 0.1618 |  | 4 | 2026-10-06 | done |
| mt_final_conv_ctx2 | Indian Ocean | conv context=2 ctx_feat ep=6 | 0.1552 |  | 7 | 2026-10-06 | done |
| mt_final_conv_ctx4 | Indian Ocean | conv context=4 ctx_feat ep=6 | 0.1520 |  | 7 | 2026-10-06 | done, weights |
| mt_final_conv_ctx4_fullval | Indian Ocean | conv context=4 ctx_feat init_from=mt_final_conv_ctx4 eval_only ep=2 | 0.1516 |  | 1 | 2026-10-06 | done |
| mt_long_conv_ctx4 | Indian Ocean | conv context=4 ctx_feat ep=12 | 0.1498 |  | 13 | 2026-10-06 | done, weights |
| mt_mlpx_ctx4 | Indian Ocean | mlp context=4 extra_feats train_stride=4 ep=3 | 0.1603 |  | 4 | 2026-10-06 | done |
| mt_sst_ctx0_seed1 | Indian Ocean | sst train_stride=4 seed=1 ep=3 | 0.2079 |  | 4 | 2026-10-06 | done |
| mt_sst_ctx2_a20 | Indian Ocean | sst context=2 alpha=20.0 train_stride=4 ep=3 | 0.1703 |  | 4 | 2026-10-06 | done |
| mt_sst_ctx2_seed1 | Indian Ocean | sst context=2 train_stride=4 seed=1 ep=3 | 0.1700 |  | 4 | 2026-10-06 | done |
| mt_sst_ctx4 | Indian Ocean | sst context=4 train_stride=4 ep=3 | 0.1659 |  | 4 | 2026-10-06 | done |
| smoke_ctx | Indian Ocean | iso context=2 eval_only ep=2 | 0.2202 |  | 1 | 2026-10-06 | done |
| smoke_mlpx | Indian Ocean | mlp extra_feats alpha=20.0 train_stride=60 ep=1 | 0.2364 |  | 2 | 2026-10-06 | done |
| sp_iso | Indian Ocean | iso train_stride=4 ep=2 | 0.2254 |  | 3 | 2026-10-06 | done |
| sp_iso_init | Indian Ocean | iso eval_only ep=2 | 0.2267 |  | 1 | 2026-10-06 | done |
| sp_mlp | Indian Ocean | mlp train_stride=4 ep=3 | 0.1976 |  | 4 | 2026-10-06 | done |
| sp_mlp_preds | Indian Ocean | mlp init_from=sp_mlp eval_only ep=2 | 0.1976 |  | 1 | 2026-10-06 | done |
| sp_sst | Indian Ocean | sst train_stride=4 ep=2 | 0.2079 |  | 3 | 2026-10-06 | done |
| sp_sst_preds | Indian Ocean | sst init_from=sp_sst eval_only ep=2 | 0.2079 |  | 1 | 2026-10-06 | done |
| tc_conv_ctx0 | Indian Ocean | conv train_stride=4 ep=3 | 0.1944 |  | 4 | 2026-10-06 | done |
| tc_conv_ctx2_cf | Indian Ocean | conv context=2 ctx_feat train_stride=4 ep=3 | 0.1613 |  | 4 | 2026-10-06 | done |
| tc_mlpx_ctx0 | Indian Ocean | mlp extra_feats train_stride=4 ep=3 | 0.1964 |  | 4 | 2026-10-06 | done |
| tc_mlpx_ctx2 | Indian Ocean | mlp context=2 extra_feats train_stride=4 ep=3 | 0.1634 |  | 4 | 2026-10-06 | done |
| tc_smoke_lag | Indian Ocean | sst context=1 lagrangian train_stride=60 ep=1 | 0.2038 |  | 2 | 2026-10-06 | done |
| tc_sst_ctx0 | Indian Ocean | sst train_stride=4 ep=3 | 0.2079 |  | 4 | 2026-10-06 | done |
| tc_sst_ctx1 | Indian Ocean | sst context=1 train_stride=4 ep=3 | 0.1781 |  | 4 | 2026-10-06 | done |
| tc_sst_ctx1_lag | Indian Ocean | sst context=1 lagrangian train_stride=4 ep=3 | 0.1766 |  | 4 | 2026-10-06 | done |
| tc_sst_ctx2 | Indian Ocean | sst context=2 train_stride=4 ep=3 | 0.1701 |  | 4 | 2026-10-06 | done, weights |
| tc_sst_ctx2_lag | Indian Ocean | sst context=2 lagrangian train_stride=4 ep=3 | 0.1703 |  | 4 | 2026-10-06 | done |
| tc_sst_ctx2_preds | Indian Ocean | sst context=2 init_from=tc_sst_ctx2 eval_only ep=2 | 0.1701 |  | 1 | 2026-10-06 | done |
| viz_conv_ctx4 | Indian Ocean | conv context=4 ctx_feat init_from=mt_long_conv_ctx4 eval_only ep=2 | 0.1278 |  | 1 | 2026-10-06 | done, weights |
| lr_const | Indian Ocean | conv context=4 ctx_feat train_stride=2 ep=6 | 0.1541 |  | 7 | 2026-10-09 | done, weights |
| lr_const_cont | Indian Ocean | conv context=4 ctx_feat train_stride=2 init_from=lr_const ep=6 | 0.1519 |  | 7 | 2026-10-09 | done, weights |
| lr_const_high | Indian Ocean | conv context=4 ctx_feat train_stride=2 ep=6 | 0.1551 |  | 7 | 2026-10-09 | done, weights |
| lr_const_low | Indian Ocean | conv context=4 ctx_feat train_stride=2 ep=6 | 0.1548 |  | 7 | 2026-10-09 | done, weights |
| lr_cos | Indian Ocean | conv context=4 ctx_feat train_stride=2 ep=6 | 0.1555 |  | 7 | 2026-10-09 | done, weights |
| lr_cos12 | Indian Ocean | conv context=4 ctx_feat train_stride=2 ep=12 | 0.1523 |  | 13 | 2026-10-09 | done, weights |

## ude (16 runs)

| run | region | settings | val MAE | test MAE | epochs | finished | status |
|---|---|---|---|---|---|---|---|
| smk_blk_mlp | Black Sea 2 km |  | 0.2468 |  | 2 | 2026-10-10 | done, weights |
| udeA_conv | Black Sea 2 km |  | 0.2249 |  | 5 | 2026-10-10 | done, weights |
| udeA_mlp | Black Sea 2 km |  | 0.2297 |  | 5 | 2026-10-10 | done, weights |
| udeA_mlp_dump | Black Sea 2 km |  | 0.2232 |  | 1 | 2026-10-10 | done, weights |
| udeA_mlp_gmax1 | Black Sea 2 km |  | 0.2297 |  | 5 | 2026-10-10 | done, weights |
| udeA_mlp_gmax1_dump | Black Sea 2 km |  | 0.2231 |  | 1 | 2026-10-10 | done, weights |
| udeA_mlp_ic_clim | Black Sea 2 km |  | 0.2679 |  | 5 | 2026-10-10 | done, weights |
| udeA_mlp_lr1e2 | Black Sea 2 km |  | 0.2291 |  | 5 | 2026-10-10 | done, weights |
| udeA_none | Black Sea 2 km |  | 0.2381 |  | 1 | 2026-10-10 | done, weights |
| udeA_relax | Black Sea 2 km |  | 0.2300 |  | 5 | 2026-10-10 | done, weights |
| udeA_smoke | Black Sea 2 km |  | 0.2365 |  | 2 | 2026-10-10 | done, weights |
| udeB_T_scalar | Black Sea 2 km |  | 0.2359 |  | 5 | 2026-10-12 | done, weights |
| udeB_mlp_scalar | Black Sea 2 km |  | 0.2277 |  | 5 | 2026-10-12 | done, weights |
| udeB_relax_scalar | Black Sea 2 km |  | 0.2286 |  | 5 | 2026-10-12 | done, weights |
| smk_io_mlp | Indian Ocean |  | 0.1784 |  | 2 | 2026-10-10 | done, weights |
| smk_io_persist | Indian Ocean |  | 0.1857 |  | 1 | 2026-10-10 | done, weights |
