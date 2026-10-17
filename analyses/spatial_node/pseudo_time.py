"""Pseudo-time view of the gap filling: integrate dC/ds = -(A C - b) explicitly (forward Euler, stable step from the
largest eigenvalue of A) from the climatology, with observed pixels held fixed, and snapshot the field; compare with
the CG steady state and track the hidden-pixel MAE vs pseudo-time. Operator inputs come from a trainer dump
(models/spatial/<run>/dumps/day_<i>.npz, written with --dump-days).
Run: python analyses/spatial_node/pseudo_time.py [day_index ...]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn import spatial_node as SN
from chl_cnn.plotting import MAP_CMAP, ERR_CMAP, OKABE_ITO

FIG = Path(__file__).resolve().parent / "figures"
RUN = "viz_conv_ctx4"
days = [int(x) for x in sys.argv[1:]] or [5618, 7684]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAPS = [0, 0.3, 1, 3, 10, 30, 100, 300]

for day in days:
    z = np.load(Path(__file__).resolve().parents[2] / "models" / "spatial" / RUN / "dumps" / f"day_{day}.npz")
    ocean, lat, lon = z["ocean"], z["lat"], z["lon"]
    grid = SN.Grid(ocean, lat, dev)
    T = lambda k, dt=torch.float32: torch.as_tensor(z[k], device=dev, dtype=dt).unsqueeze(0)
    Dc, wsum, target, fixed, free = T("Dc"), T("wsum"), T("target"), T("fixed"), T("free", torch.bool)
    freef = free.float()
    b = grid.area_p * wsum * target
    # largest eigenvalue of P A P by power iteration -> stable explicit step
    v = torch.randn_like(b) * freef
    for _ in range(60):
        v = SN.apply_operator(grid, Dc, wsum, v) * freef; lam = v.norm(); v = v / lam
    ds = 1.8 / lam.item()
    C = torch.where(free, T("clim"), fixed)                     # start from the climatology, observed pixels pinned
    truth, hidden = z["truth"], z["hidden"]
    snaps, curve = {}, []
    s, n_steps, k = 0.0, int(np.ceil(SNAPS[-1] / ds)), 0
    snaps[0] = C[0].cpu().numpy().copy()
    for step in range(1, n_steps + 1):
        C = C - ds * (SN.apply_operator(grid, Dc, wsum, C) - b) * freef
        s = step * ds
        if step % max(1, n_steps // 400) == 0 or step == n_steps:
            e = np.abs(C[0].cpu().numpy() - truth)[hidden]; curve.append((s, float(np.nanmean(e))))
        while k + 1 < len(SNAPS) and s >= SNAPS[k + 1] - 1e-9:
            k += 1; snaps[SNAPS[k]] = C[0].cpu().numpy().copy()
    steady = z["pred"]
    mae = lambda f: float(np.nanmean(np.abs(f - truth)[hidden]))
    print(f"day {z['time']}: lambda_max {lam.item():.1f}, ds {ds:.4f}, {n_steps} explicit steps; MAE hidden: climatology {mae(z['clim']):.4f}, "
          + ", ".join(f"s={sv:g}: {mae(f):.4f}" for sv, f in snaps.items()) + f", CG steady state {mae(steady):.4f}, |C(s=300) - steady| mean {np.abs(snaps[SNAPS[-1]] - steady)[ocean].mean():.4f}")
    ext = [lon[0], lon[-1], lat[-1], lat[0]]
    fig = plt.figure(figsize=(26, 13)); gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.9])
    tiles = [("Truth (ln CHL L3)", truth, MAP_CMAP), ("Input: visible pixels, pinned", np.where(z["vis"], truth, np.nan), MAP_CMAP)]
    tiles += [(f"pseudo-time s = {sv:g}  (MAE {mae(f):.3f})", np.where(ocean, f, np.nan), MAP_CMAP) for sv, f in snaps.items()]
    tiles += [("CG steady state (MAE %.3f)" % mae(steady), np.where(ocean, steady, np.nan), MAP_CMAP),
              ("Steady state minus truth, hidden px", np.where(hidden, steady - truth, np.nan), ERR_CMAP)]
    for i, (title, img, cm) in enumerate(tiles[:12]):
        a_ = fig.add_subplot(gs[i // 6, i % 6])
        vr = (-3, 2) if cm == MAP_CMAP else (-1, 1)
        im = a_.imshow(img, extent=ext, cmap=cm, vmin=vr[0], vmax=vr[1]); a_.set_title(title, fontsize=10); a_.set_xlabel("lon"); a_.set_ylabel("lat")
        fig.colorbar(im, ax=a_, shrink=0.7)
    a_ = fig.add_subplot(gs[2, :3]); cs = np.array(curve)
    a_.semilogx(cs[:, 0], cs[:, 1], color=OKABE_ITO[0], label="explicit pseudo-time integration")
    a_.axhline(mae(steady), color=OKABE_ITO[1], ls="--", label="CG steady state"); a_.axhline(mae(z["clim"]), color=OKABE_ITO[7], ls=":", label="climatology (start)")
    a_.set_xlabel("pseudo-time s"); a_.set_ylabel("MAE on hidden pixels (ln CHL)"); a_.set_title("Hidden-pixel MAE vs pseudo-time"); a_.legend(fontsize=9); a_.grid(alpha=0.3)
    a_ = fig.add_subplot(gs[2, 3:])
    l1, l2, _ = SN.tensor_eig(Dc); ratio = (l1 / l2.clamp(min=1e-9))[0].cpu().numpy(); ratio[~grid.corner_ocean.cpu().numpy()] = np.nan
    im = a_.imshow(np.log10(ratio), extent=ext, cmap="viridis", vmin=0, vmax=2); a_.set_title("log10 anisotropy ratio of D (corners)"); a_.set_xlabel("lon"); a_.set_ylabel("lat"); fig.colorbar(im, ax=a_, shrink=0.8)
    sst = z["sst"].copy(); sst[~z["sst_ok"] | ~ocean] = np.nan
    if np.isfinite(sst).any():
        a_.contour(lon, lat, sst, levels=np.nanpercentile(sst, np.linspace(5, 95, 10)), colors="white", linewidths=0.5, alpha=0.8)
    fig.suptitle(f"Gap filling as pseudo-time integration, val day {z['time']} (conv D, t+-4 soft terms; start = DOY climatology, observed pixels pinned; white: SST isotherms)", fontsize=12)
    fig.savefig(FIG / f"pseudo_time_{z['time']}.png", dpi=85, bbox_inches="tight"); plt.close(fig)
    print("wrote", FIG / f"pseudo_time_{z['time']}.png")
