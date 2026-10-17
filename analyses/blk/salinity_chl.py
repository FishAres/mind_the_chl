"""ln CHL anomaly vs salinity anomaly / raw salinity / river distance, on observed pixels (every 4th day, all years).
Assumptions: anomalies = value minus DOY climatology (T2.doy_climatology, 8-day bins, +-2 bin smoothing) computed from the same every-4th-day set
(all years, incl. val years); salinity 'observed' = finite & ocean; min river distance = min over all 8 mouths in river_names.json (incl. Bosporus)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def decile_means(x, y, nb=10):
    q = np.quantile(x, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9
    k = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nb - 1)
    xm = np.array([x[k == i].mean() for i in range(nb)]); ym = np.array([y[k == i].mean() for i in range(nb)])
    se = np.array([y[k == i].std() / np.sqrt((k == i).sum()) for i in range(nb)]); n = np.array([(k == i).sum() for i in range(nb)])
    return xm, ym, se, n


if __name__ == "__main__":
    d = S.load_store(); reg, names = load_regions(); ocean = d["ocean"].astype(bool)
    idx = np.arange(0, len(d["time"]), 4); times = d["time"][idx]; T2.TIMES = times
    y = load_ln_chl(d, idx); so = np.asarray(d["so"][idx]).astype("float32")
    print(f"days {len(idx)}; so on ocean: NaN {np.isnan(so[:, ocean]).mean():.4f}, zeros {(so[:, ocean] == 0).mean():.5f}, min {np.nanmin(so[:, ocean]):.2f}, max {np.nanmax(so[:, ocean]):.2f}; all-zero so days {int(((so[:, ocean] == 0).all(1)).sum())}")
    so[:, ~ocean] = np.nan
    obs_c = np.isfinite(y) & ocean; obs_s = np.isfinite(so) & ocean
    ar = np.arange(len(idx))
    cy, b = T2.doy_climatology(y, obs_c, ar); cs, _ = T2.doy_climatology(np.nan_to_num(so), obs_s, ar)
    ay = y - cy[b]; asl = so - cs[b]
    m = obs_c & obs_s
    print(f"observed CHL px-days {int(obs_c.sum())}; with finite salinity {int(m.sum())}")
    seas = season_idx(times); rows = []
    print("\n**Pearson r, ln CHL anomaly vs salinity anomaly; cell = r (n px-days)**\n")
    print("| region | DJF | MAM | JJA | SON | all |\n|---|---|---|---|---|---|")
    R = {}
    for r in list(names) + [-1]:
        rm = (reg > 0) if r == -1 else (reg == r); cells = []
        for s in SEAS + ["all"]:
            sel = np.ones(len(seas), bool) if s == "all" else seas == s
            mm = m[sel] & rm[None]; n = int(mm.sum())
            rv = pearson(ay[sel][mm], asl[sel][mm]) if n >= 5 else np.nan; R[(r, s)] = (rv, n)
            cells.append(f"{rv:.3f} ({n})" if n >= 5 else "n<5")
        print(f"| {'all ocean' if r == -1 else f'{r} {names[r]}'} | " + " | ".join(cells) + " |")
    # NW shelf
    nw = (reg == 1)[None] & m
    xa, ya, sea, na = decile_means(asl[nw], ay[nw]); xr_, yr, ser, nr = decile_means(so[nw], y[nw])
    print("\n**NW shelf: mean ln CHL anomaly by salinity-anomaly decile**\n\n| decile | mean sal anomaly | mean ln CHL anomaly | SE | n |\n|---|---|---|---|---|")
    for i in range(10): print(f"| {i + 1} | {xa[i]:.3f} | {ya[i]:.3f} | {sea[i]:.4f} | {na[i]} |")
    print("\n**NW shelf: mean ln CHL by raw salinity decile**\n\n| decile | mean salinity | mean ln CHL | SE | n |\n|---|---|---|---|---|")
    for i in range(10): print(f"| {i + 1} | {xr_[i]:.2f} | {yr[i]:.3f} | {ser[i]:.4f} | {nr[i]} |")
    # river distance
    rd = np.asarray(d["river_dist"]).min(0).astype("float32")
    print("\n**NW shelf: Pearson r, ln CHL anomaly vs min river-mouth distance (static, km); cell = r (n)**\n\n| DJF | MAM | JJA | SON | all |\n|---|---|---|---|---|")
    cells = []; RD = {}
    for s in SEAS + ["all"]:
        sel = np.ones(len(seas), bool) if s == "all" else seas == s
        mm = obs_c[sel] & (reg == 1)[None]; n = int(mm.sum())
        dist = np.broadcast_to(rd[None], mm.shape)[mm]
        RD[s] = (pearson(ay[sel][mm], dist), n); cells.append(f"{RD[s][0]:.3f} ({n})")
    print("| " + " | ".join(cells) + " |")
    print(f"NW shelf min river distance km: median {np.median(rd[reg == 1]):.0f}, range {rd[reg == 1].min():.0f}-{rd[reg == 1].max():.0f}")
    # figure
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    rr = list(names); mat = np.array([[R[(r, s)][0] if R[(r, s)][1] >= 5 else np.nan for s in SEAS + ["all"]] for r in rr])
    im = ax[0, 0].imshow(mat, cmap="PuOr", vmin=-0.5, vmax=0.5, aspect="auto")
    ax[0, 0].set_xticks(range(5)); ax[0, 0].set_xticklabels(SEAS + ["all"]); ax[0, 0].set_yticks(range(len(rr))); ax[0, 0].set_yticklabels([f"{r} {names[r]}" for r in rr], fontsize=8)
    for i in range(len(rr)):
        for j in range(5): ax[0, 0].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax[0, 0], label="Pearson r"); ax[0, 0].set_title("Pearson r: ln CHL anomaly vs salinity anomaly, by region and season"); ax[0, 0].set_xlabel("Season"); ax[0, 0].set_ylabel("Region")
    ax[0, 1].errorbar(xa, ya, 1.96 * sea, marker="o", color=OKABE_ITO[0]); ax[0, 1].set_xlabel("Salinity anomaly, decile mean (PSU)"); ax[0, 1].set_ylabel("Mean ln CHL anomaly (observed px)"); ax[0, 1].set_title("NW shelf: ln CHL anomaly by salinity-anomaly decile (+-95% CI)")
    ax[1, 0].errorbar(xr_, yr, 1.96 * ser, marker="o", color=OKABE_ITO[1]); ax[1, 0].set_xlabel("Salinity, decile mean (PSU)"); ax[1, 0].set_ylabel("Mean ln CHL (observed px)"); ax[1, 0].set_title("NW shelf: ln CHL by raw salinity decile (+-95% CI)")
    ax[1, 1].bar(range(5), [RD[s][0] for s in SEAS + ["all"]], color=OKABE_ITO[2]); ax[1, 1].set_xticks(range(5)); ax[1, 1].set_xticklabels(SEAS + ["all"]); ax[1, 1].set_xlabel("Season"); ax[1, 1].set_ylabel("Pearson r"); ax[1, 1].set_title("NW shelf: ln CHL anomaly vs min distance to nearest river mouth")
    for a in ax.ravel()[1:]: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "salinity_chl.png", dpi=130)
