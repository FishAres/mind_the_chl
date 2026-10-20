# Short training runs with the lp base (150 km low-pass of day t, held over the lead)

5 epochs, batch 16, Adam 3e-3, init a = 0.25 / alpha = 0.25; best epoch by val mean-lead own-set MAE (as step 1). Final common set: target observed & model, step-1 tr_rea_geo_wind, x0.25, lp@0.3 (scan best scalar at lead 1), persistence valid.

| run | best epoch | l1 MAE model / step1 / x0.25 / lp@0.3 / pers. | l2 MAE model / step1 / x0.25 / lp@0.3 / pers. | l3 MAE model / step1 / x0.25 / lp@0.3 / pers. | l4 MAE model / step1 / x0.25 / lp@0.3 / pers. | l5 MAE model / step1 / x0.25 / lp@0.3 / pers. | n l1 |
|---|---|---|---|---|---|---|---|
| trs_lp_factor | 0 | 0.1621 / 0.1620 / 0.1619 / 0.1620 / 0.1663 | 0.1964 / 0.1953 / 0.1963 / 0.1965 / 0.2004 | 0.2220 / 0.2209 / 0.2226 / 0.2224 / 0.2255 | 0.2405 / 0.2386 / 0.2408 / 0.2411 / 0.2434 | 0.2482 / 0.2467 / 0.2494 / 0.2488 / 0.2508 | 1,858,942 |
| trs_lp_decomp | 1 | 0.1611 / 0.1618 / 0.1617 / 0.1617 / 0.1661 | 0.1951 / 0.1952 / 0.1962 / 0.1964 / 0.2003 | 0.2209 / 0.2211 / 0.2228 / 0.2226 / 0.2257 | 0.2392 / 0.2390 / 0.2412 / 0.2414 / 0.2437 | 0.2472 / 0.2472 / 0.2499 / 0.2494 / 0.2514 | 1,835,947 |

trs_lp_factor ocean-mean a / b by epoch 1-5: 0.173 / 0.066, 0.183 / 0.051, 0.209 / 0.059, 0.208 / 0.052, 0.209 / 0.082 (step 1 tr_rea_geo_wind: a 0.169, b 0.019)

Decomp alpha (season mean) at epochs 1 and 5, with lp base vs step 1b tr1b_pix_s0 (no base, same settings):

| component | region | tr1b_pix_s0 ep1 | trs_lp_decomp ep1 | tr1b_pix_s0 ep5 | trs_lp_decomp ep5 |
|---|---|---|---|---|---|
| geo_mean | NW shelf | 0.189 | 0.256 | 0.146 | 0.242 |
| geo_mean | W open | 0.452 | 0.492 | 0.584 | 0.715 |
| geo_mean | E open | 0.414 | 0.479 | 0.447 | 0.650 |
| geo_anom | NW shelf | 0.140 | 0.166 | 0.122 | 0.147 |
| geo_anom | W open | 0.067 | 0.108 | 0.084 | 0.127 |
| geo_anom | E open | 0.098 | 0.145 | 0.079 | 0.135 |
| ageo_mean | NW shelf | 0.166 | 0.203 | 0.136 | 0.200 |
| ageo_mean | W open | 0.115 | 0.127 | 0.004 | 0.011 |
| ageo_mean | E open | 0.168 | 0.175 | 0.093 | 0.099 |
| ageo_anom | NW shelf | 0.200 | 0.204 | 0.177 | 0.186 |
| ageo_anom | W open | 0.217 | 0.198 | 0.198 | 0.166 |
| ageo_anom | E open | 0.139 | 0.130 | 0.116 | 0.103 |
