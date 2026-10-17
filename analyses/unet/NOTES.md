# U-Net campaign notes (unet_c* runs, IO store, val hidden-pixel MAE in ln CHL)

Marks: dilated CNN pinn_v2_lam0 0.1365 val / 0.1434 test (60 ep); upstream-style U-Net unet_mtcg_lam0 0.1508 val (30 ep).
Prior U-Nets (30 ep, full inputs, L1, no year/donor recipe): d2w64 0.1426 (val at ep 20: 0.1439), d3w32 0.1442.
Common flags: ALL="sst so air_temp u_wind v_wind sla mlotst u_curr v_curr"; FULL="--features $ALL --clim --coords --residual --context 2 --vis-weight 0.1 --year-input --donor-window 15".
Smoke test (d3w64, 1 ep, 5 batches): peak 7.4 GB, folder deleted.

## 2026-10-14 c1: unet_c1_d3w64 (depth 3, width 64 = 64/128/256/512, L1)
Command: python -m chl_cnn.train_pinn --run unet_c1_d3w64 $FULL --model unet --width 64 --unet-depth 3 --epochs 20 --ckpt-from 5 --fig-every 5
- val MAE by epoch: 0.1861, 0.1771, 0.1681, 0.1596, 0.1635, 0.1579, 0.1503, 0.1502, 0.1518, 0.1490, 0.1473, 0.1459, 0.1461, 0.1449, 0.1447, 0.1439, 0.1437, 0.1432, 0.1431, 0.1431
- best val 0.1431 at epoch 19 (epoch 20 equal); test MAE 0.1463; train loss at end 0.1499 (epoch 1: 0.2217)
- 28 s/epoch, 10.2 min wall; peak GPU 7.96 GB; n_params 7,716,355; no NaN
Observations:
- Val 0.1431 at 20 epochs vs d2w64's 0.1439 at its epoch 20 (of 30) and 0.1426 at 30; CNN pinn_v2_lam0 at epoch 20 (of 60): 0.1411.
- Train loss (0.150, includes t+-1 and visible terms) stays above val MAE throughout; the gap narrows from 0.036 (ep 1) to 0.007 (ep 20).
- Val falls 0.0008 over epochs 16-20 while the cosine lr goes to 0; train falls 0.0013 over the same epochs.
- 4x the params of d2w64 cost 6 s/epoch more (28 vs 22) and 8 GB peak.
Next (c2): c1 + --context 4. Reason: no train/val gap to regularise; the brief notes inputs matter and t+-4 context gave the spatial model 0.03 over t+-2.

## 2026-10-14 c2: unet_c2_ctx4 (c1 + --context 4; in_ch 26 -> 34)
Command: python -m chl_cnn.train_pinn --run unet_c2_ctx4 $FULL --model unet --width 64 --unet-depth 3 --context 4 --epochs 20 --ckpt-from 5 --fig-every 5  (the later --context 4 overrides FULL's --context 2; config.json context = 4)
- val MAE by epoch: 0.1883, 0.1718, 0.1687, 0.1641, 0.1635, 0.1561, 0.1517, 0.1500, 0.1506, 0.1491, 0.1468, 0.1453, 0.1450, 0.1434, 0.1434, 0.1423, 0.1422, 0.1422, 0.1417, 0.1417
- best val 0.1417 at epoch 20 (19 equal); test MAE 0.1447; train loss at end 0.1483
- 29 s/epoch, 10.5 min wall; peak GPU 7.99 GB; n_params 7,720,963; no NaN
Observations:
- 0.1417 vs c1 0.1431 (-0.0014 val, -0.0016 test); c2 is below c1 from epoch 11 on by 0.0005-0.0015.
- Train loss above val MAE throughout (gap 0.0066 at the end, as in c1).
- Val falls 0.0006 over epochs 16-20 (lr -> 0 under cosine); train falls 0.0014.
- Cost of context 4: +1 s/epoch, +0.03 GB.
Next (c3): c2 + --lr 2e-3. Reason: no train/val gap in either run and val still falling into the cosine tail, consistent with convergence within 20 epochs being the limit; a higher peak lr is the cheapest convergence lever.

## 2026-10-14 c3: unet_c3_lr2e3 (c2 + --lr 2e-3)
Command: python -m chl_cnn.train_pinn --run unet_c3_lr2e3 $FULL --model unet --width 64 --unet-depth 3 --context 4 --lr 2e-3 --epochs 20 --ckpt-from 5 --fig-every 5
- val MAE by epoch: 0.1878, 0.1724, 0.1719, 0.1685, 0.1765, 0.1595, 0.1553, 0.1529, 0.1534, 0.1507, 0.1499, 0.1481, 0.1473, 0.1453, 0.1452, 0.1437, 0.1439, 0.1440, 0.1431, 0.1433
- best val 0.1431 at epoch 19; test MAE 0.1462; train loss at end 0.1504
- 29 s/epoch, 10.5 min wall; peak GPU 7.99 GB; n_params 7,720,963; no NaN
Observations:
- 0.1431 vs c2 0.1417 (+0.0014 val, +0.0015 test): lr 2e-3 is above c2 at every epoch from 3 on (by 0.002-0.013), with a bump at epoch 5 (0.1765 vs 0.1635).
- End train loss 0.1504 vs c2 0.1483: the higher lr ends with a higher train loss too.
- Across c1-c3 the ordering of end train loss (c2 < c1 < c3) matches the ordering of best val; no run shows train below val.
Next (c4): c2 + --batch 4 (lr 1e-3). Reason: lower train loss at 20 epochs tracked lower val and no run overfits; halving the batch doubles the optimizer steps per epoch without raising the lr, which hurt in c3.

## 2026-10-14 c4: unet_c4_b4 (c2 + --batch 4, lr 1e-3)
Command: python -m chl_cnn.train_pinn --run unet_c4_b4 $FULL --model unet --width 64 --unet-depth 3 --context 4 --batch 4 --epochs 20 --ckpt-from 5 --fig-every 5
(queued ~25 min behind another agent's run t3d_dil_c2 on the shared GPU lock)
- val MAE by epoch: 0.1811, 0.1709, 0.1669, 0.1618, 0.1599, 0.1542, 0.1546, 0.1517, 0.1501, 0.1483, 0.1474, 0.1457, 0.1472, 0.1448, 0.1442, 0.1431, 0.1437, 0.1432, 0.1434, 0.1429
- best val 0.1429 at epoch 20; test MAE 0.1460; train loss at end 0.1506
- 35.5 s/epoch, 12.6 min wall; peak GPU 7.99 GB (same as batch 8: the peak is set by the 32-day eval chunks, not the training batch); n_params 7,720,963; no NaN
Observations:
- 0.1429 vs c2 0.1417 (+0.0012 val, +0.0013 test); batch 4 is below c2 for epochs 1-6 (by 0.002-0.007) and above it from epoch 7 on.
- End train loss 0.1506 vs c2 0.1483: twice the steps per epoch did not lower the train loss at the end of the 20-epoch cosine schedule.
- Train loss above val MAE throughout, as in c1-c3.
Campaign summary (20 ep each): c2 (d3w64, ctx 4, L1, lr 1e-3, batch 8) best at 0.1417 val / 0.1447 test; c1 0.1431 / 0.1463; c3 0.1431 / 0.1462; c4 0.1429 / 0.1460. None reaches the CNN's 0.1365 (60 ep) or its 0.1411 at epoch 20; all are below the upstream-style U-Net's 0.1508 and the prior d2w64's 0.1439 at epoch 20.
Figure: analyses/unet/learning_curves.png (analyses/unet/plot_unet.py).
