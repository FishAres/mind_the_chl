"""Clear val days: truth, fake-cloud input, best spatial nODE fill (mt_long_conv_ctx4 saved preds), dilated CNN
(pinn_v2_lam0) and both errors on the hidden pixels. Run: python analyses/spatial_node/clear_day_examples.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn import store as S, data as D, clouds as C
from chl_cnn.runs import find_run, CATEGORIES
from chl_cnn.plotting import MAP_CMAP, ERR_CMAP

FIG = Path(__file__).resolve().parent / "figures"
RUN = "mt_long_conv_ctx4"; N_DAYS = 4
d = S.load_store(); ocean = d["ocean"]; lat, lon = d["lat"], d["lon"]
y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & ocean
tr, va, te = D.split_by_years(d["time"]); hid_va, _ = C.fixed_hidden(obs, va, seed=0); va_pos = {int(i): k for k, i in enumerate(va)}
z = np.load(CATEGORIES["spatial"] / RUN / "preds.npz"); idx, node = z["val_idx"], z["val_pred"].astype("float32")
cnn = np.load(find_run("pinn_v2_lam0") / "preds.npz"); cnn_pos = {int(i): k for k, i in enumerate(cnn["idx"])}
frac = np.isfinite(y[idx][:, ocean]).mean(1)
pick = np.argsort(-frac)[:N_DAYS]
ext = [lon[0], lon[-1], lat[-1], lat[0]]
fig, ax = plt.subplots(N_DAYS, 6, figsize=(26, 4.3 * N_DAYS))
print(f"{'day':12s} {'obs frac':>8s} {'hidden px':>9s} {'nODE MAE':>9s} {'CNN MAE':>8s} {'clim MAE':>8s}")
for r, k in enumerate(pick):
    day = int(idx[k]); h = hid_va[va_pos[day]]; vis = obs[day] & ~h
    t = y[day]; p_node = node[k].copy(); p_cnn = cnn["pred"][cnn_pos[day]].copy()
    for p in (p_node, p_cnn): p[~ocean] = np.nan
    e_node, e_cnn = np.where(h, p_node - t, np.nan), np.where(h, p_cnn - t, np.nan)
    mae_n, mae_c = np.nanmean(np.abs(e_node)), np.nanmean(np.abs(e_cnn))
    print(f"{str(d['time'][day])[:10]:12s} {frac[k]:8.2f} {int(h.sum()):9d} {mae_n:9.3f} {mae_c:8.3f}")
    panels = [(t, f"ln CHL L3 observed, {str(d['time'][day])[:10]} ({frac[k]:.0%} of ocean)", MAP_CMAP, (-3, 2)),
              (np.where(vis, t, np.nan), "Input: visible pixels after fake clouds", MAP_CMAP, (-3, 2)),
              (p_node, f"Spatial nODE fill (conv D, t+-4)", MAP_CMAP, (-3, 2)),
              (p_cnn, "Dilated CNN (pinn_v2_lam0)", MAP_CMAP, (-3, 2)),
              (e_node, f"nODE minus truth, hidden px (MAE {mae_n:.3f})", ERR_CMAP, (-1, 1)),
              (e_cnn, f"CNN minus truth, hidden px (MAE {mae_c:.3f})", ERR_CMAP, (-1, 1))]
    for a_, (img, title, cm, vr) in zip(ax[r], panels):
        im = a_.imshow(img, extent=ext, cmap=cm, vmin=vr[0], vmax=vr[1]); a_.set_title(title, fontsize=10); a_.set_xlabel("lon"); a_.set_ylabel("lat")
        fig.colorbar(im, ax=a_, shrink=0.75)
fig.suptitle("Clearest val days (every-4th-day val set): truth, fake-cloud input, spatial nODE and CNN fills, errors on hidden pixels (ln CHL)", fontsize=12)
fig.savefig(FIG / "clear_day_examples.png", dpi=90, bbox_inches="tight"); print("wrote", FIG / "clear_day_examples.png")
