### Check A: oracle motion ceiling, lead 1, val accepted NCC patches

| predictor | MAE ln CHL | n patches | n px (patch-px) |
|---|---|---|---|
| persistence | 0.1426 | 12,044 | 8,419,723 |
| x0.25 | 0.1380 | 12,044 | 8,419,723 |
| oracle NCC shift | 0.1381 | 12,044 | 8,419,723 |
| 0.46 x patch current shift | 0.1392 | 12,044 | 8,419,723 |

Oracle gain over persistence 0.0045 [0.0039, 0.0051]; over x0.25 -0.0001 [-0.0006, 0.0003] (day bootstrap, 520 days, 2000 resamples).

By region (pixel label):

| group | oracle gain vs pers [CI] | oracle gain vs x0.25 [CI] | pers | x0.25 | oracle | 0.46 cur | n patches | n px | n days |
|---|---|---|---|---|---|---|---|---|---|
| NW shelf | 0.0062 [0.0040, 0.0082] | -0.0010 [-0.0029, 0.0007] | 0.1910 | 0.1839 | 0.1849 | 0.1873 | 4,356 | 1,505,328 | 371 |
| W open | 0.0041 [0.0034, 0.0048] | 0.0002 [-0.0004, 0.0007] | 0.1268 | 0.1229 | 0.1227 | 0.1237 | 6,784 | 3,297,741 | 423 |
| E open | 0.0045 [0.0038, 0.0052] | 0.0003 [-0.0003, 0.0008] | 0.1353 | 0.1311 | 0.1308 | 0.1316 | 5,836 | 3,476,346 | 441 |
| Anatolian | -0.0019 [-0.0052, 0.0015] | -0.0082 [-0.0114, -0.0050] | 0.1911 | 0.1848 | 0.1930 | 0.1904 | 2,762 | 85,536 | 404 |
| Crimea/Caucasus | -0.0017 [-0.0055, 0.0019] | -0.0075 [-0.0112, -0.0041] | 0.1616 | 0.1558 | 0.1632 | 0.1644 | 1,852 | 54,087 | 346 |
| Azov | n < 5 | | | | | | 0 | 0 | 0 |
| Marmara | 0.0072 [-0.0058, 0.0201] | 0.0048 [-0.0085, 0.0184] | 0.1718 | 0.1694 | 0.1646 | 0.1682 | 196 | 685 | 126 |

By season (month of day t):

| group | oracle gain vs pers [CI] | oracle gain vs x0.25 [CI] | pers | x0.25 | oracle | 0.46 cur | n patches | n px | n days |
|---|---|---|---|---|---|---|---|---|---|
| DJF | 0.0039 [0.0011, 0.0066] | -0.0021 [-0.0048, 0.0006] | 0.2137 | 0.2077 | 0.2098 | 0.2085 | 594 | 363,213 | 77 |
| MAM | 0.0048 [0.0036, 0.0061] | 0.0003 [-0.0007, 0.0012] | 0.1419 | 0.1375 | 0.1372 | 0.1387 | 3,098 | 2,201,019 | 139 |
| JJA | 0.0052 [0.0044, 0.0060] | 0.0002 [-0.0003, 0.0008] | 0.1323 | 0.1273 | 0.1271 | 0.1286 | 6,239 | 4,462,345 | 176 |
| SON | 0.0020 [0.0005, 0.0035] | -0.0013 [-0.0026, -0.0000] | 0.1583 | 0.1550 | 0.1563 | 0.1561 | 2,113 | 1,393,146 | 128 |

Coverage: accepted-patch footprints hold 56.5% of the day-t+1 observed px of all 723 val lead-1 pairs (4,333,034 of 7,667,719); the scored common set holds 41.3% (3,167,158 unique px); 525 of 723 pairs have >= 1 accepted patch.

Checks: zero-shift oracle minus persistence 0.00e+00; reversed-sign oracle MAE 0.1572 vs persistence 0.1418 on the same 8,184,977 px; row/col-swapped oracle 0.1518 vs persistence 0.1418 (8,224,604 px).
Inner 16 x 16 px of each patch (Hann weight >= 0.5 per axis; 2,338,112 patch-px): persistence 0.1453, x0.25 0.1404, oracle 0.1324, 0.46 cur 0.1403; oracle gain vs x0.25 0.0080 [0.0072, 0.0087], vs persistence 0.0128 [0.0119, 0.0137].
