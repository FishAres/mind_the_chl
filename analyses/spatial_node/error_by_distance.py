"""Hidden-pixel MAE vs distance to the nearest visible pixel, for the spatial nODE fills (models/spatial/*_preds),
the dilated CNN (pinn_v2_lam0) and the day-of-year climatology, on the same val days and hidden pixels.
Run: python analyses/spatial_node/error_by_distance.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn import store as S, data as D, clouds as C, train2 as T2
from chl_cnn.runs import find_run, CATEGORIES
from chl_cnn.plotting import OKABE_ITO

FIG = Path(__file__).resolve().parent / "figures"
d = S.load_store(); TIMES = d["time"]; T2.TIMES = TIMES
ocean = d["ocean"]; y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & ocean
tr, va, te = D.split_by_years(TIMES)
hid_va, _ = C.fixed_hidden(obs, va, seed=0)
clim, dbin = T2.doy_climatology(y, obs, tr)

models = {}
for p in sorted(CATEGORIES["spatial"].glob("*/preds.npz")):
    z = np.load(p); models[p.parent.name.replace("_preds", "")] = (z["val_idx"], z["val_pred"].astype("float32"))
    assert np.array_equal(z["val_idx"], next(iter(models.values()))[0]), p
cnn = np.load(find_run("pinn_v2_lam0") / "preds.npz")
cnn_pos = {int(i): k for k, i in enumerate(cnn["idx"])}
days = models[next(iter(models))][0]
va_pos = {int(i): k for k, i in enumerate(va)}
edges = np.array([0, 1, 2, 3, 5, 8, 12, 20, 40, 1000]); labels = [f"{edges[i]}-{edges[i+1]-1}" if edges[i+1] - edges[i] > 1 else f"{edges[i]}" for i in range(len(edges) - 1)]; labels[-1] = f">={edges[-2]}"
acc = {k: np.zeros((len(edges) - 1, 2)) for k in list(models) + ["dilated CNN (pinn_v2_lam0)", "DOY climatology"]}
for k, day in enumerate(days):
    h = hid_va[va_pos[int(day)]]; vis = obs[day] & ~h
    dist = distance_transform_edt(~vis)                      # cells to the nearest visible pixel
    b = np.digitize(dist[h], edges) - 1; t = y[day][h]
    preds = {m: models[m][1][k][h] for m in models}
    preds["dilated CNN (pinn_v2_lam0)"] = cnn["pred"][cnn_pos[int(day)]][h]
    preds["DOY climatology"] = clim[dbin[day]][h]
    assert np.array_equal(cnn["hidden"][cnn_pos[int(day)]], h), "hidden masks differ"
    for m, p in preds.items():
        e = np.abs(p - t)
        for j in range(len(edges) - 1):
            s = b == j; acc[m][j, 0] += e[s].sum(); acc[m][j, 1] += s.sum()
n = acc["DOY climatology"][:, 1]
print(f"{'distance (cells)':16s} " + " ".join(f"{m[:22]:>22s}" for m in acc) + f" {'hidden px':>10s}")
for j, lab in enumerate(labels):
    print(f"{lab:16s} " + " ".join(f"{acc[m][j, 0] / max(acc[m][j, 1], 1):22.4f}" for m in acc) + f" {int(n[j]):10d}")
print("overall          " + " ".join(f"{acc[m][:, 0].sum() / acc[m][:, 1].sum():22.4f}" for m in acc) + f" {int(n.sum()):10d}  ({len(days)} val days)")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw=dict(width_ratios=[2, 1]))
for (m, a_), col in zip(acc.items(), OKABE_ITO):
    ax[0].plot(range(len(labels)), a_[:, 0] / np.maximum(a_[:, 1], 1), "o-", color=col, label=m)
ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels); ax[0].set_xlabel("distance to nearest visible pixel (cells, 0.25 deg)"); ax[0].set_ylabel("MAE on hidden pixels (ln CHL)")
ax[0].set_title("Hidden-pixel MAE by distance to the nearest visible pixel"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].bar(range(len(labels)), n / n.sum(), color=OKABE_ITO[7]); ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels, rotation=45)
ax[1].set_xlabel("distance to nearest visible pixel (cells)"); ax[1].set_ylabel("fraction of hidden pixels"); ax[1].set_title("Distribution of hidden val pixels by distance")
fig.suptitle(f"Val days (every 4th, n={len(days)}), fake-cloud hidden pixels (n={int(n.sum())})")
fig.savefig(FIG / "mae_by_distance.png", dpi=120, bbox_inches="tight"); print("wrote", FIG / "mae_by_distance.png")
