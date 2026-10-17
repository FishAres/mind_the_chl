# Training runs: history index

Generated 2026-10-10 01:19 by chl_cnn/archive_runs.py. Each run folder here holds config.json, history.json (per-epoch curves), metrics.json and figures/. Weights and predictions were deleted except for the runs marked 'kept', which stay under models/<category>/. Val MAE = hidden-pixel MAE (ln CHL) on the val years; spatial runs with train_stride > 1 or val_stride > 1 were scored on every k-th day (see config.json). Settings list only the non-default flags; see the trainer docstrings for meanings.

## cnn (1 runs)

| run | finished | settings | val MAE | test MAE | grad ratio | grad corr | note |
|---|---|---|---|---|---|---|---|
| perf_dilated_vis01 | 2026-09-30 23:53 | dilated feat=9 ctx=2 vis_weight=0.1 width=64 epochs=30 | 0.1381 | 0.1400 |  |  | kept |

## pinn/cnn (1 runs)

| run | finished | settings | val MAE | test MAE | grad ratio | grad corr | note |
|---|---|---|---|---|---|---|---|
| pinn_v2_lam0 | 2026-10-03 02:56 | dilated feat=9 ctx=2 vis_weight=0.1 width=64 donor_window=15 unet_depth=4 epochs=60 year_input | 0.1365 | 0.1390 |  |  | kept |

## spatial (34 runs)

| run | finished | settings | val MAE | test MAE | grad ratio | grad corr | note |
|---|---|---|---|---|---|---|---|
| tc_sst_ctx2 | 2026-10-06 02:07 | sst context=2 train_stride=4 ep=3 | 0.1701 |  | 0.6531 | 0.4058 | kept |
| mt_final_conv_ctx4 | 2026-10-06 03:37 | conv context=4 ctx_feat ep=6 | 0.1520 |  | 0.6341 | 0.5030 | kept |
| mt_long_conv_ctx4 | 2026-10-06 04:27 | conv context=4 ctx_feat ep=12 | 0.1498 |  | 0.6242 | 0.5105 | kept |
| viz_conv_ctx4 | 2026-10-06 05:33 | conv context=4 ctx_feat init_from=mt_long_conv_ctx4 eval_only ep=2 | 0.1278 |  | 0.6493 | 0.5903 | kept |
| blk_iso_ctx0 | 2026-10-09 00:53 | iso train_stride=4 ep=3 | 0.2808 |  | 0.2815 | 0.2153 |  |
| blk_sst_ctx0 | 2026-10-09 01:00 | sst train_stride=4 ep=3 | 0.2674 |  | 0.4796 | 0.2233 |  |
| blk_sst_ctx2 | 2026-10-09 01:07 | sst context=2 train_stride=4 ep=3 | 0.2042 |  | 0.5861 | 0.2398 |  |
| blk_sst_ctx4 | 2026-10-09 01:14 | sst context=4 train_stride=4 ep=3 | 0.1970 |  | 0.5917 | 0.2467 |  |
| lr_cos | 2026-10-09 01:19 | conv context=4 ctx_feat train_stride=2 ep=6 | 0.1555 |  | 0.6566 | 0.4951 |  |
| blk_mlpx_ctx2 | 2026-10-09 01:30 | mlp context=2 extra_feats train_stride=4 ep=3 | 0.1911 |  | 0.3911 | 0.3320 |  |
| lr_const | 2026-10-09 01:32 | conv context=4 ctx_feat train_stride=2 ep=6 | 0.1541 |  | 0.6435 | 0.4996 |  |
| lr_const_low | 2026-10-09 01:44 | conv context=4 ctx_feat train_stride=2 ep=6 | 0.1548 |  | 0.6364 | 0.4948 |  |
| blk_conv_ctx2_cf | 2026-10-09 01:46 | conv context=2 ctx_feat train_stride=4 ep=3 | 0.1888 |  | 0.4390 | 0.3304 |  |
| lr_const_high | 2026-10-09 01:55 | conv context=4 ctx_feat train_stride=2 ep=6 | 0.1551 |  | 0.6581 | 0.4952 |  |
| blk_conv_ctx4_cf | 2026-10-09 01:58 | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1843 |  | 0.4323 | 0.3372 |  |
| blk_sst_ctx2_lag | 2026-10-09 02:11 | sst context=2 lagrangian train_stride=4 ep=3 | 0.2039 |  | 0.5883 | 0.2286 |  |
| lr_const_cont | 2026-10-09 02:12 | conv context=4 ctx_feat train_stride=2 init_from=lr_const ep=6 | 0.1519 |  | 0.6553 | 0.5078 |  |
| blk_conv_ctx2_cf_lag | 2026-10-09 02:22 | conv context=2 ctx_feat lagrangian train_stride=4 ep=3 | 0.1909 |  | 0.4282 | 0.3078 |  |
| blk_sst_ctx2_lag15 | 2026-10-09 02:31 | sst context=2 lagrangian train_stride=4 ep=3 | 0.2056 |  | 0.5832 | 0.2267 |  |
| blk_final_conv_ctx4_cf | 2026-10-09 03:33 | conv context=4 ctx_feat ep=6 | 0.1772 |  | 0.4259 | 0.3420 |  |
| lr_cos12 | 2026-10-09 03:48 | conv context=4 ctx_feat train_stride=2 ep=12 | 0.1523 |  | 0.6397 | 0.5035 |  |
| smk_ctrl | 2026-10-09 03:50 | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2024 |  | 0.3457 | 0.2923 |  |
| smk_normd | 2026-10-09 03:51 | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2011 |  | 0.3726 | 0.3108 |  |
| smk_depth | 2026-10-09 03:53 | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2022 |  | 0.3446 | 0.2914 |  |
| smk_sal | 2026-10-09 03:58 | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2037 |  | 0.3566 | 0.2934 |  |
| smk_river | 2026-10-09 04:00 | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2025 |  | 0.3481 | 0.2936 |  |
| smk_all | 2026-10-09 04:02 | conv context=4 ctx_feat train_stride=60 ep=1 | 0.2002 |  | 0.3739 | 0.3119 |  |
| smk_all_mlp | 2026-10-09 04:05 | mlp context=4 ctx_feat extra_feats train_stride=60 ep=1 | 0.2050 |  | 0.3603 | 0.2902 |  |
| smk_all_mlp_plain | 2026-10-09 04:10 | mlp context=4 ctx_feat train_stride=60 ep=1 | 0.2050 |  | 0.3603 | 0.2903 |  |
| r2_ctrl | 2026-10-09 04:25 | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1813 |  | 0.4666 | 0.3260 |  |
| r2_normd | 2026-10-09 04:35 | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1821 |  | 0.4531 | 0.3342 |  |
| r2_depth | 2026-10-09 04:52 | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1814 |  | 0.4700 | 0.3247 |  |
| r2_sal | 2026-10-09 05:08 | conv context=4 ctx_feat train_stride=4 ep=3 | 0.1801 |  | 0.4440 | 0.3367 |  |
| r2_river | 2026-10-09 05:13 | conv context=4 ctx_feat train_stride=4 ep=3 |  |  |  |  | unfinished |
