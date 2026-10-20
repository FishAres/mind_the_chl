"""Flow gate, Check A: oracle motion ceiling at lead 1 on the accepted masked-NCC patches (val years 2017, 2021).

On every accepted lead-1 patch (analyses/blk4/results/spectral_motion_patches.npz: k = 1, not null, peak exists, not on
the search edge, peak r >= 0.6) whose day t is a val day, the day-t+1 observed pixels inside the 32 x 32 patch are
predicted from day t by
  (i)   persistence: ln C(x, t)
  (ii)  x0.25: semi-Lagrangian backward trajectory with 0.25 x reanalysis currents (tracer.Integrator, RK2, 4 sub-steps,
        vmax 10 px/day, trajectory validity >= 0.99, departure observed fraction >= 0.99; = tracer x0.25 reference)
  (iii) oracle: ln C(x - s, t), s = measured NCC shift (row fy, col fx; a feature at x on t is at x + s on t+1),
        bilinear, from the full day-t field, valid if the observed bilinear weight >= 0.99 (same rule as depart_sample)
  (iv)  current shift: as (iii) with s = 0.46 x patch-mean reanalysis displacement over t..t+1 (trapezoid of the cached
        Hann-weighted patch means vel_cu / vel_cv, px/day, east = +col, north = +row)
All four scored on the same pixels: target observed & day t observed at x & (ii), (iii), (iv) valid. Pixels in
overlapping patches count once per patch. ln CHL as spectral_motion (CHL <= 0 or > 100 -> missing), crop as train_pinn.

Sections: compute() data generation (cached to results/check_a_rows.npz), analyse() tables / bootstrap, report() json,
md fragment and figure.  Run from the repo root: CUDA_VISIBLE_DEVICES="" python -m analyses.flow_gate.check_a
"""
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
from analyses.flow_gate.common import *  # noqa: F401,F403
import time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chl_cnn.plotting import OKABE_ITO as C

CACHE = ROOT / "analyses/blk4/results/spectral_motion_patches.npz"
ROWS = RES / "check_a_rows.npz"
P = 32
R_MIN = 0.6
CUR_FACTOR = 0.46
PRED = ["persistence", "x0.25", "oracle NCC shift", "0.46 x patch current shift"]
KEYS = ["pers", "x025", "orc", "cur"]


def ln_chl(c):
    c = np.asarray(c, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(c) & (c > 0) & (c <= 100), np.log(c), np.nan)


def bilinear(A, obs, r, c):
    """Sample A at float (row r, col c) like grid_sample(align_corners=True, zeros padding) with observed weights.
    Returns value (obs-weighted) and validity (observed weight >= 0.99)."""
    H, W = A.shape
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int)
    fr, fc = r - r0, c - c0
    num = np.zeros(r.shape); den = np.zeros(r.shape)
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (1, 0, fr * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 1, fr * fc)):
        rr, cc = r0 + dr, c0 + dc
        ins = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)
        o = obs[rr, cc] & ins
        num += np.where(o, w * np.nan_to_num(A[rr, cc]), 0.0); den += np.where(o, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.maximum(den, 1e-3), den >= 0.99


# ============================== data generation ==============================
def compute():
    t0 = time.time()
    S = load_store(); d, ys, xs, ocean = S["d"], S["ys"], S["xs"], S["ocean"]
    H, W = ocean.shape
    reg, ties = regions(ocean)
    z = np.load(CACHE)
    va = set(S["va"].tolist())
    acc = (z["k"] == 1) & (z["null"] == 0) & z["okpk"] & ~z["edge"] & (z["rpk"] >= R_MIN)
    acc_all = int(acc.sum())
    sel = acc & np.isin(z["i"], S["va"])
    idx = np.flatnonzero(sel)
    pos = {int(p): z["pos"][q] for q, p in enumerate(z["pidx"])}
    pcol = np.searchsorted(z["pidx"], z["p"][idx])
    dcx = 0.5 * (z["vel_cu"][z["i"][idx], pcol] + z["vel_cu"][z["j"][idx], pcol])
    dcy = 0.5 * (z["vel_cv"][z["i"][idx], pcol] + z["vel_cv"][z["j"][idx], pcol])
    nb1 = neighbour_index(S["tday"], 1)
    assert (nb1[z["i"][idx]] == z["j"][idx]).all(), "cache j is not the calendar day after i"
    pairs = [(int(i), int(nb1[i])) for i in S["va"] if nb1[i] >= 0]
    by_day = {}
    for q, r in enumerate(idx):
        by_day.setdefault(int(z["i"][r]), []).append(q)
    it = integrator(S)
    rows = {k: [] for k in ["rec", "day", "patch", "region", "n", "pers", "x025", "orc", "cur", "n_neg", "orc_neg", "pers_neg", "orc_zero",
                                     "n_core", "pers_core", "x025_core", "orc_core", "cur_core", "n_swap", "orc_swap", "pers_swap"]}
    cov = dict(obs_t1=0, foot_obs_t1=0, scored_unique=0, n_pairs=len(pairs), n_pairs_with_patch=0)
    dq = dict(chl_gt100_or_le0_px=0, foot_px_target_obs=0, excl_day_t_unobs=0, excl_x025=0, excl_oracle=0, excl_cur=0,
              recs_cur_nan=int((~np.isfinite(dcx) | ~np.isfinite(dcy)).sum()), recs_no_scored_px=0, nonfinite_pred_in_scored=0)
    for (i, j) in pairs:
        cB = np.asarray(d["chl_l3"][j, ys, xs]); B = ln_chl(cB)
        dq["chl_gt100_or_le0_px"] += int((np.isfinite(cB) & ~np.isfinite(B) & ocean).sum())
        obsB = np.isfinite(B) & ocean
        cov["obs_t1"] += int(obsB.sum())
        if i not in by_day:
            continue
        cov["n_pairs_with_patch"] += 1
        A = ln_chl(d["chl_l3"][i, ys, xs]); obsA = np.isfinite(A) & ocean
        px, pv = advect(S, it, [i, j], A, obsA)
        foot = np.zeros((H, W), bool); scored = np.zeros((H, W), bool)
        for q in by_day[i]:
            r = idx[q]
            pr, pc = pos[int(z["p"][r])]
            rr = (pr - ys.start + np.arange(P))[:, None] * np.ones((1, P), int)
            cc = (pc - xs.start + np.arange(P))[None, :] * np.ones((P, 1), int)
            ins = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rr, cc = np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)
            tgt = ins & obsB[rr, cc]
            foot[rr[tgt], cc[tgt]] = True
            fy, fx = float(z["fy"][r]), float(z["fx"][r])
            o_v, o_ok = bilinear(A, obsA, rr - fy, cc - fx)
            n_v, n_ok = bilinear(A, obsA, rr + fy, cc + fx)            # sign check: shift in the wrong direction
            s_v, s_ok = bilinear(A, obsA, rr - fx, cc - fy)            # axis check: row / col components swapped
            if np.isfinite(dcx[q]) and np.isfinite(dcy[q]):
                c_v, c_ok = bilinear(A, obsA, rr - CUR_FACTOR * dcy[q], cc - CUR_FACTOR * dcx[q])
            else:
                c_v, c_ok = np.zeros(rr.shape), np.zeros(rr.shape, bool)
            p_ok = obsA[rr, cc]; x_ok = pv[rr, cc]
            m = tgt & p_ok & x_ok & o_ok & c_ok
            dq["foot_px_target_obs"] += int(tgt.sum())
            dq["excl_day_t_unobs"] += int((tgt & ~p_ok).sum()); dq["excl_x025"] += int((tgt & ~x_ok).sum())
            dq["excl_oracle"] += int((tgt & ~o_ok).sum()); dq["excl_cur"] += int((tgt & ~c_ok).sum())
            if not m.any():
                dq["recs_no_scored_px"] += 1
                continue
            scored[rr[m], cc[m]] = True
            truth = B[rr, cc]
            e = {"pers": np.abs(A[rr, cc] - truth), "x025": np.abs(px[rr, cc] - truth), "orc": np.abs(o_v - truth), "cur": np.abs(c_v - truth)}
            dq["nonfinite_pred_in_scored"] += int(sum((~np.isfinite(v[m])).sum() for v in e.values()))
            mn = m & n_ok
            eo0 = np.abs(bilinear(A, obsA, rr.astype(float), cc.astype(float))[0] - truth)   # zero-shift oracle = persistence
            rg = reg[rr, cc]
            core = np.zeros((P, P), bool); core[8:24, 8:24] = True     # inner 16 x 16 (Hann weight >= 0.5 on both axes)
            ms = m & s_ok
            for g in np.unique(rg[m]):
                mg = m & (rg == g); mng = mn & (rg == g)
                rows["rec"].append(int(r)); rows["day"].append(i); rows["patch"].append(int(z["p"][r])); rows["region"].append(int(g))
                rows["n"].append(int(mg.sum()))
                for k in KEYS:
                    rows[k].append(float(e[k][mg].sum()))
                rows["n_neg"].append(int(mng.sum())); rows["orc_neg"].append(float(np.abs(n_v - truth)[mng].sum()))
                rows["pers_neg"].append(float(e["pers"][mng].sum())); rows["orc_zero"].append(float(eo0[mg].sum()))
                mc = mg & core; rows["n_core"].append(int(mc.sum()))
                for k in KEYS:
                    rows[f"{k}_core"].append(float(e[k][mc].sum()))
                msg = ms & (rg == g); rows["n_swap"].append(int(msg.sum()))
                rows["orc_swap"].append(float(np.abs(s_v - truth)[msg].sum())); rows["pers_swap"].append(float(e["pers"][msg].sum()))
        cov["foot_obs_t1"] += int(foot.sum()); cov["scored_unique"] += int(scored.sum())
    out = {k: np.asarray(v) for k, v in rows.items()}
    meta = dict(cov=cov, dq=dq, accepted_lead1_all_years=acc_all, accepted_lead1_val=int(len(idx)),
                accepted_lead1_val_days=len(by_day), region_ties=int(ties), seconds=round(time.time() - t0, 1))
    np.savez_compressed(ROWS, meta=json.dumps(meta), times=S["times"].astype(str), **out)
    print(json.dumps(meta, indent=1), flush=True)


# ============================== analysis ==============================
def table(r, m, times):
    n = r["n"][m].sum()
    recs = np.unique(r["rec"][m]); days = np.unique(r["day"][m])
    row = dict(n_patches=int(len(recs)), n_px=int(n), n_days=int(len(days)), flag_lt5=bool(len(recs) < MIN_N or len(days) < MIN_N))
    if row["flag_lt5"] or n == 0:
        row.update({k: None for k in KEYS}); return row
    for k in KEYS:
        row[k] = float(r[k][m].sum() / n)
    row["gain_vs_pers"] = row["pers"] - row["orc"]; row["gain_vs_x025"] = row["x025"] - row["orc"]
    row["gain_vs_pers_ci"] = boot_ratio_diff(r["day"][m], r["pers"][m], r["orc"][m], r["n"][m])
    row["gain_vs_x025_ci"] = boot_ratio_diff(r["day"][m], r["x025"][m], r["orc"][m], r["n"][m])
    return row


def analyse():
    z = np.load(ROWS, allow_pickle=False)
    meta = json.loads(str(z["meta"])); times = z["times"].astype("datetime64[D]")
    r = {k: z[k] for k in z.files if k not in ("meta", "times")}
    all_m = np.ones(len(r["n"]), bool)
    res = dict(overall=table(r, all_m, times))
    res["by_region"] = {REG_NAMES[g]: table(r, r["region"] == g, times) for g in range(1, 8)}
    seas = season_of(times[r["day"]])
    res["by_season"] = {s: table(r, seas == s, times) for s in SEAS}
    n = r["n"].sum(); nn = r["n_neg"].sum()
    res["checks"] = dict(zero_shift_oracle_minus_persistence=float((r["orc_zero"].sum() - r["pers"].sum()) / n),
                         reversed_shift_oracle_mae=float(r["orc_neg"].sum() / nn), persistence_same_px=float(r["pers_neg"].sum() / nn), n_px_reversed=int(nn),
                         swapped_axes_oracle_mae=float(r["orc_swap"].sum() / r["n_swap"].sum()), persistence_swap_px=float(r["pers_swap"].sum() / r["n_swap"].sum()),
                         n_px_swap=int(r["n_swap"].sum()),
                         core16={k: float(r[f"{k}_core"].sum() / r["n_core"].sum()) for k in KEYS}, n_px_core=int(r["n_core"].sum()))
    cm = r["n_core"] > 0
    res["checks"]["core16_gain_vs_x025_ci"] = boot_ratio_diff(r["day"][cm], r["x025_core"][cm], r["orc_core"][cm], r["n_core"][cm])
    res["checks"]["core16_gain_vs_pers_ci"] = boot_ratio_diff(r["day"][cm], r["pers_core"][cm], r["orc_core"][cm], r["n_core"][cm])
    cov = meta["cov"]
    res["coverage"] = dict(val_lead1_pairs=cov["n_pairs"], pairs_with_accepted_patch=cov["n_pairs_with_patch"],
                           obs_t1_px_all_val_pairs=cov["obs_t1"], patch_footprint_obs_t1_unique_px=cov["foot_obs_t1"],
                           scored_unique_px=cov["scored_unique"], frac_footprint=cov["foot_obs_t1"] / cov["obs_t1"],
                           frac_scored=cov["scored_unique"] / cov["obs_t1"])
    res["meta"] = meta
    return res


# ============================== reporting ==============================
def report(res):
    json.dump(res, open(RES / "check_a.json", "w"), indent=1)
    lab = dict(zip(KEYS, PRED))
    o = res["overall"]
    L = ["### Check A: oracle motion ceiling, lead 1, val accepted NCC patches", "",
         "| predictor | MAE ln CHL | n patches | n px (patch-px) |", "|---|---|---|---|"]
    for k in KEYS:
        L.append(f"| {lab[k]} | {fmt(o[k])} | {o['n_patches']:,} | {o['n_px']:,} |")
    L += ["", f"Oracle gain over persistence {fmt(o['gain_vs_pers'])} [{fmt(o['gain_vs_pers_ci'][0])}, {fmt(o['gain_vs_pers_ci'][1])}]; "
          f"over x0.25 {fmt(o['gain_vs_x025'])} [{fmt(o['gain_vs_x025_ci'][0])}, {fmt(o['gain_vs_x025_ci'][1])}] (day bootstrap, {o['n_days']} days, 2000 resamples).", ""]
    for key, ttl in (("by_region", "region (pixel label)"), ("by_season", "season (month of day t)")):
        L += [f"By {ttl}:", "", "| group | oracle gain vs pers [CI] | oracle gain vs x0.25 [CI] | pers | x0.25 | oracle | 0.46 cur | n patches | n px | n days |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for g, t in res[key].items():
            if t["flag_lt5"] or t["pers"] is None:
                L.append(f"| {g} | n < 5 | | | | | | {t['n_patches']} | {t['n_px']:,} | {t['n_days']} |"); continue
            ci1, ci2 = t["gain_vs_pers_ci"], t["gain_vs_x025_ci"]
            L.append(f"| {g} | {fmt(t['gain_vs_pers'])} [{fmt(ci1[0])}, {fmt(ci1[1])}] | {fmt(t['gain_vs_x025'])} [{fmt(ci2[0])}, {fmt(ci2[1])}] | "
                     f"{fmt(t['pers'])} | {fmt(t['x025'])} | {fmt(t['orc'])} | {fmt(t['cur'])} | {t['n_patches']:,} | {t['n_px']:,} | {t['n_days']} |")
        L.append("")
    c = res["coverage"]
    L += [f"Coverage: accepted-patch footprints hold {c['frac_footprint']:.1%} of the day-t+1 observed px of all {c['val_lead1_pairs']} val lead-1 pairs "
          f"({c['patch_footprint_obs_t1_unique_px']:,} of {c['obs_t1_px_all_val_pairs']:,}); the scored common set holds {c['frac_scored']:.1%} "
          f"({c['scored_unique_px']:,} unique px); {c['pairs_with_accepted_patch']} of {c['val_lead1_pairs']} pairs have >= 1 accepted patch.", ""]
    ch = res["checks"]
    L += [f"Checks: zero-shift oracle minus persistence {ch['zero_shift_oracle_minus_persistence']:.2e}; reversed-sign oracle MAE "
          f"{ch['reversed_shift_oracle_mae']:.4f} vs persistence {ch['persistence_same_px']:.4f} on the same {ch['n_px_reversed']:,} px; "
          f"row/col-swapped oracle {ch['swapped_axes_oracle_mae']:.4f} vs persistence {ch['persistence_swap_px']:.4f} ({ch['n_px_swap']:,} px).",
          f"Inner 16 x 16 px of each patch (Hann weight >= 0.5 per axis; {ch['n_px_core']:,} patch-px): persistence {ch['core16']['pers']:.4f}, x0.25 "
          f"{ch['core16']['x025']:.4f}, oracle {ch['core16']['orc']:.4f}, 0.46 cur {ch['core16']['cur']:.4f}; oracle gain vs x0.25 "
          f"{ch['core16']['x025'] - ch['core16']['orc']:.4f} [{ch['core16_gain_vs_x025_ci'][0]:.4f}, {ch['core16_gain_vs_x025_ci'][1]:.4f}], "
          f"vs persistence {ch['core16']['pers'] - ch['core16']['orc']:.4f} [{ch['core16_gain_vs_pers_ci'][0]:.4f}, {ch['core16_gain_vs_pers_ci'][1]:.4f}].", ""]
    (RES / "check_a.md").write_text("\n".join(L))
    print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1, 2.6]))
    vals = [o[k] for k in KEYS]
    ax[0].bar(range(4), vals, color=[C[7], C[3], C[0], C[2]])
    for q, v in enumerate(vals):
        ax[0].text(q, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["persistence", "x0.25\ncurrents", "oracle\nNCC shift", "0.46 x patch\ncurrent shift"], fontsize=8)
    ax[0].set(ylabel="MAE, ln CHL (day t+1 observed px)", title=f"Lead-1 MAE on accepted NCC patches, val\n(n {o['n_patches']:,} patches, {o['n_px']:,} patch-px)")
    ax[0].grid(alpha=.3, axis="y")
    gs = [g for g, t in res["by_region"].items() if not t["flag_lt5"] and t["pers"] is not None]
    w = 0.2
    for q, (k, col) in enumerate(zip(KEYS, [C[7], C[3], C[0], C[2]])):
        ax[1].bar(np.arange(len(gs)) + (q - 1.5) * w, [res["by_region"][g][k] for g in gs], w, color=col, label=lab[k])
    ax[1].set_xticks(range(len(gs)))
    ax[1].set_xticklabels([f"{g}\n(n {res['by_region'][g]['n_patches']:,} patches)" for g in gs], fontsize=8)
    ax[1].set(ylabel="MAE, ln CHL (day t+1 observed px)", title="Lead-1 MAE on accepted NCC patches by region (pixel region label), val")
    ax[1].legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16)); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "check_a_oracle_ceiling.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    if "--recompute" in sys.argv or not ROWS.exists():
        compute()
    report(analyse())
