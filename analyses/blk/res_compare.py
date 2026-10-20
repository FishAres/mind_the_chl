"""Best 4 km gap-filler scored against 2 km truth, compared with the 2 km model on the same hidden pixels.

Method (also written to results/res_compare.json "assumptions")
- Runs: 4 km models/pinn/t3d/blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50 (150 epochs in total, 50 per
  stage, batch 16); 2 km models/pinn/t3d/blk_t3d_c4_gdl100_sl_lam0p3 (20 epochs, batch 4). preds.npz of each: ln CHL
  predictions on the val (2017, 2021) and test (2019, 2023) days, every 4th day; both use the same day indices.
- Grids: each run's grid is its store cropped to the ocean bounding box + 1 px (train_pinn --crop rule, as
  chl_cnn.viz.load_context). 2 km full row / col r -> 4 km full row / col r // 2 (store_blk4: 2 x 2 blocks, last 2 km
  row dropped); every 2 km crop pixel falls inside the 4 km crop.
- Mapping 4 km -> 2 km. nearest: value of the parent 4 km block (piecewise constant). bilinear: 4 km pixel centres
  at 2 km coordinate 2 i + 0.5; bilinear weights over the 4 surrounding 4 km pixels, renormalised over those that are
  4 km ocean with a finite prediction (none -> nearest).
- Truth: 2 km ln CHL (store chl_l3, D.log_chl). Scored pixels = the 2 km run's hidden mask (preds.npz 'hidden':
  observed today and cloudy on the donor day at 2 km).
- Fair set rule: a 2 km hidden pixel is kept only if its parent 4 km block was not in the 4 km model's input, i.e.
  NOT (4 km observed & not 4 km hidden). 4 km observed = >= 2 of the 4 children observed (store_blk4.coarse_chl);
  4 km hidden = 4 km observed today & 4 km not observed on the donor day (clouds.hidden_mask on the 4 km grid). So a
  kept pixel's parent is either 4 km hidden or 4 km not observed (< 2 children observed). Pixels whose parent was
  visible to the 4 km model, or whose parent is not 4 km ocean, are excluded and counted.
- Distance = Euclidean distance (2 km px) from the hidden pixel to the nearest pixel visible to the 2 km model
  (2 km observed & not 2 km hidden), scipy distance_transform_edt. Bins on distance d: [1, 2), [2, 4), [4, 8), [8, 16), [16, 32), >= 32 px (labels 1, 2-3, ...).
- Regions: analyses/blk/regions.npy (2 km), cropped as the 2 km run.
- MAE = pooled mean |pred - truth| over pixels (ln CHL). Difference CI: 95 % percentile bootstrap over days (1000
  resamples, seed 0) of the pooled MAE difference.
- Gradient statistic (as analyses/blk4/gradient_diagnosis / train_pinn): pixel pairs at lag L = 1, 2, 4, 8 2 km px
  along x and y; truth observed at both ends; ratio = sum|dpred| / sum|dtruth|, corr = sum(dp dt) / sqrt(sum dp^2 sum
  dt^2) (uncentred). Pair rules: "both ends fair" (both ends in the fair set) and ">= 1 end fair" (other end observed,
  any status; predictions are raw model outputs at visible pixels, not pasted). Same pairs for all predictors.
- preds.npz predictions are float16.

Run from the repo root: python -m analyses.blk.res_compare   (CPU only)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from chl_cnn import data as D, clouds as C
from chl_cnn.plotting import OKABE_ITO as PAL
from chl_cnn.viz import BLK_SPLIT

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
DATA = Path.home() / "mind_the_chl_gap/data"
S2, S4 = DATA / "blk_store_2014_2024", DATA / "blk4_store_2014_2024"
RUN4 = ROOT / "models/pinn/t3d/blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50"
RUN2 = ROOT / "models/pinn/t3d/blk_t3d_c4_gdl100_sl_lam0p3"
PRED = ["2 km model", "4 km nearest", "4 km bilinear"]
LAGS = [1, 2, 4, 8]
DBINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, np.inf)]
RULES = ["both ends fair", ">=1 end fair"]
MIN_N = 5
NB = 1000


# ============================== data ==============================
def crop_of(ocean):
    rows, cols = np.flatnonzero(ocean.any(1)), np.flatnonzero(ocean.any(0))
    return (slice(max(rows[0] - 1, 0), min(rows[-1] + 2, ocean.shape[0])),
            slice(max(cols[0] - 1, 0), min(cols[-1] + 2, ocean.shape[1])))


def donors(times, va, te):
    """Donor store index per val / test day (C.fixed_hidden draw, seed 0 / 1, no exclusion; both runs)."""
    out = {}
    for idx, seed in ((va, 0), (te, 1)):
        local = C.donor_indices(len(idx), np.random.default_rng(seed))
        out.update(dict(zip(idx.tolist(), idx[local].tolist())))
    return out


def bilinear_setup(ys2, xs2, ys4, xs4, H4, W4):
    """Per 2 km crop row / col: lower 4 km crop index and weight of the upper one."""
    def axis(sl2, sl4, n2, n4):
        f = (np.arange(n2) + sl2.start - 0.5) / 2.0 - sl4.start
        lo = np.floor(f).astype(int); w = f - lo
        lo_c = np.clip(lo, 0, n4 - 1); hi_c = np.clip(lo + 1, 0, n4 - 1)
        near = (np.arange(n2) + sl2.start) // 2 - sl4.start
        return lo_c, hi_c, w, near
    return axis(ys2, ys4, ys2.stop - ys2.start, H4), axis(xs2, xs4, xs2.stop - xs2.start, W4)


def map_bilinear(p4, valid4, ax_y, ax_x):
    (ylo, yhi, wy, ny), (xlo, xhi, wx, nx) = ax_y, ax_x
    num = np.zeros((len(ylo), len(xlo))); den = np.zeros_like(num)
    for yi, wyy in ((ylo, 1 - wy), (yhi, wy)):
        for xi, wxx in ((xlo, 1 - wx), (xhi, wx)):
            v = valid4[yi[:, None], xi[None, :]]
            w = wyy[:, None] * wxx[None, :] * v
            num += w * np.where(v, p4[yi[:, None], xi[None, :]], 0.0); den += w
    near = p4[ny[:, None], nx[None, :]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, near)


def compute():
    t0 = time.time()
    oc2f = np.load(S2 / "ocean.npy").astype(bool); oc4f = np.load(S4 / "ocean.npy").astype(bool)
    ys2, xs2 = crop_of(oc2f); ys4, xs4 = crop_of(oc4f)
    oc2, oc4 = oc2f[ys2, xs2], oc4f[ys4, xs4]
    times = np.load(S2 / "time.npy"); assert (times == np.load(S4 / "time.npy")).all()
    chl2, chl4 = np.load(S2 / "chl_l3.npy", mmap_mode="r"), np.load(S4 / "chl_l3.npy", mmap_mode="r")
    z2, z4 = np.load(RUN2 / "preds.npz"), np.load(RUN4 / "preds.npz")
    idx2, idx4 = z2["idx"], z4["idx"]
    common = np.array(sorted(set(idx2.tolist()) & set(idx4.tolist())))
    pos2, pos4 = {int(i): k for k, i in enumerate(idx2)}, {int(i): k for k, i in enumerate(idx4)}
    P2, H2m, P4, H4m = z2["pred"], z2["hidden"], z4["pred"], z4["hidden"]
    assert P2.shape[1:] == oc2.shape and P4.shape[1:] == oc4.shape, (P2.shape, oc2.shape, P4.shape, oc4.shape)
    tr, va, te = D.split_by_years(times, *BLK_SPLIT)
    don = donors(times, va, te)
    vaset = set(va.tolist())
    reg_full = np.load(HERE / "regions.npy"); reg = reg_full[ys2, xs2]
    names = {int(k): v for k, v in json.load(open(HERE / "region_names.json")).items()}
    ax_y, ax_x = bilinear_setup(ys2, xs2, ys4, xs4, *oc4.shape)
    ny, nx = ax_y[3], ax_x[3]
    nd, nreg, ndb = len(common), 8, len(DBINS)
    # accumulators: per day x region x dist bin: sum |err| per predictor, count
    S = np.zeros((nd, 3, nreg, ndb)); N = np.zeros((nd, nreg, ndb))
    G = {(r, L): np.zeros((3, 5)) for r in RULES for L in LAGS}          # ap, at, pt, pp, tt per predictor
    GN = {(r, L): 0 for r in RULES for L in LAGS}
    dq = dict(n_days_run2=len(idx2), n_days_run4=len(idx4), n_days_common=nd, hid2_total=0, hid2_ocean2_not_obs2=0,
              hid4_not_obs4=0, hid2_recompute_mismatch=0, hid4_recompute_mismatch=0,
              excl_parent_visible4=0, excl_parent_not_ocean4=0, kept_parent_hidden4=0, kept_parent_not_obs4=0,
              pred2_nonfinite_on_fair=0, pred4n_nonfinite_on_fair=0, pred4b_nonfinite_on_fair=0,
              pred2_allzero_days=0, pred4_allzero_days=0, truth_nonfinite_on_hidden2=0, days_no_visible2=0)
    own = dict(mae2_own_hidden_val=[0.0, 0], mae4_own_hidden_val_4kmtruth=[0.0, 0])
    split = np.array(["val" if int(i) in vaset else "test" for i in common])
    for q, i in enumerate(common):
        i = int(i); dn = don[i]
        y2 = D.log_chl(np.asarray(chl2[i, ys2, xs2])); y2d = D.log_chl(np.asarray(chl2[dn, ys2, xs2]))
        y4 = D.log_chl(np.asarray(chl4[i, ys4, xs4])); y4d = D.log_chl(np.asarray(chl4[dn, ys4, xs4]))
        obs2, obs2d = np.isfinite(y2) & oc2, np.isfinite(y2d) & oc2
        obs4, obs4d = np.isfinite(y4) & oc4, np.isfinite(y4d) & oc4
        h2, h4 = H2m[pos2[i]], H4m[pos4[i]]
        p2 = P2[pos2[i]].astype("float32"); p4 = P4[pos4[i]].astype("float32")
        dq["hid2_recompute_mismatch"] += int((h2 != (obs2 & ~obs2d)).sum())
        dq["hid4_recompute_mismatch"] += int((h4 != (obs4 & ~obs4d)).sum())
        dq["hid2_total"] += int(h2.sum()); dq["hid2_ocean2_not_obs2"] += int((h2 & ~obs2).sum()); dq["hid4_not_obs4"] += int((h4 & ~obs4).sum())
        dq["truth_nonfinite_on_hidden2"] += int((h2 & ~np.isfinite(y2)).sum())
        if not np.any(p2[oc2]): dq["pred2_allzero_days"] += 1
        if not np.any(p4[oc4]): dq["pred4_allzero_days"] += 1
        if split[q] == "val":
            own["mae2_own_hidden_val"][0] += float(np.abs(p2 - y2)[h2].sum()); own["mae2_own_hidden_val"][1] += int(h2.sum())
            own["mae4_own_hidden_val_4kmtruth"][0] += float(np.abs(p4 - y4)[h4].sum()); own["mae4_own_hidden_val_4kmtruth"][1] += int(h4.sum())
        vis4 = obs4 & ~h4
        par_vis, par_oc, par_h4, par_obs4 = vis4[ny[:, None], nx[None, :]], oc4[ny[:, None], nx[None, :]], \
            h4[ny[:, None], nx[None, :]], obs4[ny[:, None], nx[None, :]]
        dq["excl_parent_visible4"] += int((h2 & par_oc & par_vis).sum()); dq["excl_parent_not_ocean4"] += int((h2 & ~par_oc).sum())
        fair = h2 & par_oc & ~par_vis
        dq["kept_parent_hidden4"] += int((fair & par_h4).sum()); dq["kept_parent_not_obs4"] += int((fair & ~par_obs4).sum())
        p4n = p4[ny[:, None], nx[None, :]]
        p4b = map_bilinear(p4, oc4 & np.isfinite(p4), ax_y, ax_x).astype("float32")
        preds = [p2, p4n, p4b]
        fin = np.isfinite(y2) & np.isfinite(p2) & np.isfinite(p4n) & np.isfinite(p4b)
        dq["pred2_nonfinite_on_fair"] += int((fair & ~np.isfinite(p2)).sum())
        dq["pred4n_nonfinite_on_fair"] += int((fair & ~np.isfinite(p4n)).sum())
        dq["pred4b_nonfinite_on_fair"] += int((fair & ~np.isfinite(p4b)).sum())
        fair &= fin
        vis2 = obs2 & ~h2
        if vis2.any():
            dist = distance_transform_edt(~vis2)
        else:
            dist = np.full(vis2.shape, np.inf); dq["days_no_visible2"] += 1
        db = np.full(vis2.shape, -1)
        for b, (lo, hi) in enumerate(DBINS):
            db[(dist >= lo - 1e-6) & (dist < hi + 1 - 1e-6) if np.isfinite(hi) else (dist >= lo - 1e-6)] = b
        m = fair & (db >= 0)
        rr, bb = reg[m], db[m]
        lin = rr * ndb + bb
        N[q] = np.bincount(lin, minlength=nreg * ndb).reshape(nreg, ndb)
        for k, p in enumerate(preds):
            S[q, k] = np.bincount(lin, weights=np.abs(p - y2)[m], minlength=nreg * ndb).reshape(nreg, ndb)
        # gradients
        obsf = np.isfinite(y2) & fin
        for L in LAGS:
            for axis in (0, 1):
                sl = (lambda a: (a[:-L], a[L:])) if axis == 0 else (lambda a: (a[:, :-L], a[:, L:]))
                oa, ob_ = sl(obsf); fa, fb = sl(fair)
                for rule, mm in (("both ends fair", oa & ob_ & fa & fb), (">=1 end fair", oa & ob_ & (fa | fb))):
                    ta, tb = sl(y2); dt = (tb - ta)[mm].astype("float64")
                    GN[(rule, L)] += dt.size
                    for k, p in enumerate(preds):
                        pa, pb = sl(p); dp = (pb - pa)[mm].astype("float64")
                        G[(rule, L)][k] += [np.abs(dp).sum(), np.abs(dt).sum(), (dp * dt).sum(), (dp * dp).sum(), (dt * dt).sum()]
        if q % 50 == 0:
            print(f"{q}/{nd} {time.time() - t0:.0f}s", flush=True)
    return dict(S=S, N=N, G=G, GN=GN, dq=dq, own=own, split=split, names=names, common=common,
                crops=dict(ys2=[ys2.start, ys2.stop], xs2=[xs2.start, xs2.stop], ys4=[ys4.start, ys4.stop], xs4=[xs4.start, xs4.stop]),
                seconds=round(time.time() - t0))


# ============================== analysis ==============================
def mae_from(S, N, sel_days=slice(None), reg=None, db=None):
    s, n = S[sel_days], N[sel_days]
    if reg is not None: s, n = s[:, :, reg], n[:, reg]
    if db is not None: s, n = s[..., db], n[..., db]
    ss = s.reshape(s.shape[0], 3, -1).sum(-1); nn = n.reshape(n.shape[0], -1).sum(-1)
    return ss, nn                                    # per day: (nd, 3), (nd,)


def summarise(ss, nn, rng):
    n = int(nn.sum())
    if n < MIN_N:
        return dict(n=n, flag="n<5")
    mae = ss.sum(0) / n
    out = dict(n=n, mae={PRED[k]: round(float(mae[k]), 4) for k in range(3)})
    dd = np.flatnonzero(nn > 0)
    if len(dd) >= MIN_N:
        bi = rng.integers(0, len(dd), (NB, len(dd)))
        bs, bn = ss[dd][bi].sum(1), nn[dd][bi].sum(1)
        bm = bs / bn[:, None]
        for k in (1, 2):
            dif = bm[:, k] - bm[:, 0]
            out[f"diff {PRED[k]} - 2 km"] = dict(value=round(float(mae[k] - mae[0]), 4),
                                                ci95=[round(float(np.percentile(dif, 2.5)), 4), round(float(np.percentile(dif, 97.5)), 4)])
        out["n_days"] = int(len(dd))
    return out


def analyse(r):
    rng = np.random.default_rng(0)
    S, N, names = r["S"], r["N"], r["names"]
    res = {}
    res["overall"] = summarise(*mae_from(S, N), rng)
    for sp in ("val", "test"):
        res[f"overall_{sp}"] = summarise(*mae_from(S, N, r["split"] == sp), rng)
    res["by_region"] = {f"{k} {names[k]}": summarise(*mae_from(S, N, reg=k), rng) for k in names}
    res["region_0_unlabelled"] = summarise(*mae_from(S, N, reg=0), rng)
    res["by_distance_px"] = {(f"{lo}" if lo == hi else f"{lo}-{hi}" if np.isfinite(hi) else f">={lo}"): summarise(*mae_from(S, N, db=b), rng)
                             for b, (lo, hi) in enumerate(DBINS)}
    gr = {}
    for rule in RULES:
        gr[rule] = {}
        for L in LAGS:
            g = r["G"][(rule, L)]; n = r["GN"][(rule, L)]
            gr[rule][f"lag {L} px ({2 * L} km)"] = dict(n_pairs=int(n), **{PRED[k]: dict(
                ratio=round(float(g[k, 0] / g[k, 1]), 3), corr=round(float(g[k, 2] / np.sqrt(g[k, 3] * g[k, 4])), 3)) for k in range(3)})
    res["gradients"] = gr
    o = r["own"]
    res["sanity_own_val_mae"] = {k: dict(mae=round(v[0] / v[1], 4), n=v[1]) for k, v in o.items()}
    return res


# ============================== plots ==============================
def plots(res, r):
    cols = [PAL[0], PAL[1], PAL[2]]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    # by region
    keys = [k for k, v in res["by_region"].items() if "mae" in v]
    x = np.arange(len(keys)); w = 0.27
    for k in range(3):
        ax[0].bar(x + (k - 1) * w, [res["by_region"][kk]["mae"][PRED[k]] for kk in keys], w, color=cols[k], label=PRED[k])
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{kk}\n(n {res['by_region'][kk]['n']:,})" for kk in keys], rotation=40, ha="right", fontsize=8)
    ax[0].set_ylabel("MAE, ln CHL (fair hidden 2 km px)"); ax[0].set_title("MAE by region on fair hidden 2 km pixels"); ax[0].legend()
    # by distance
    dk = [k for k, v in res["by_distance_px"].items() if "mae" in v]
    for k in range(3):
        ax[1].plot(range(len(dk)), [res["by_distance_px"][kk]["mae"][PRED[k]] for kk in dk], "o-", color=cols[k], label=PRED[k])
    ax[1].set_xticks(range(len(dk))); ax[1].set_xticklabels([f"{kk}\n(n {res['by_distance_px'][kk]['n']:,})" for kk in dk], fontsize=8)
    ax[1].set_xlabel("distance to nearest 2 km-visible pixel (2 km px)"); ax[1].set_ylabel("MAE, ln CHL")
    ax[1].set_title("MAE by distance to nearest visible 2 km pixel"); ax[1].legend()
    # difference vs distance with CI
    for k in (1, 2):
        v = [res["by_distance_px"][kk][f"diff {PRED[k]} - 2 km"] for kk in dk]
        y = np.array([d["value"] for d in v]); lo = np.array([d["ci95"][0] for d in v]); hi = np.array([d["ci95"][1] for d in v])
        ax[2].errorbar(np.arange(len(dk)) + (k - 1.5) * 0.1, y, yerr=[y - lo, hi - y], fmt="o-", color=cols[k], capsize=3, label=f"{PRED[k]} minus 2 km model")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(range(len(dk))); ax[2].set_xticklabels(dk); ax[2].set_xlabel("distance to nearest 2 km-visible pixel (2 km px)")
    ax[2].set_ylabel("MAE difference, ln CHL (95 % day bootstrap CI)"); ax[2].set_title("MAE difference (4 km mapped minus 2 km) by distance"); ax[2].legend()
    o = res["overall"]
    fig.suptitle(f"4 km model on 2 km truth vs 2 km model, fair hidden 2 km pixels ({o['n']:,} px, {o['n_days']} val+test days)")
    fig.tight_layout(); fig.savefig(FIG / "res_compare_mae.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for j, rule in enumerate(RULES):
        g = res["gradients"][rule]; ks = list(g)
        lagkm = [2 * L for L in LAGS]
        for k in range(3):
            ax[0, j].plot(lagkm, [g[kk][PRED[k]]["ratio"] for kk in ks], "o-", color=cols[k], label=PRED[k])
            ax[1, j].plot(lagkm, [g[kk][PRED[k]]["corr"] for kk in ks], "o-", color=cols[k], label=PRED[k])
        ax[0, j].axhline(1, color="k", lw=0.8)
        nn = ", ".join(f"{g[kk]['n_pairs'] / 1e6:.2f}" for kk in ks)
        ax[0, j].set_title(f"Gradient ratio sum|dpred|/sum|dtruth|, pairs: {rule}\n(n pairs M by lag: {nn})", fontsize=10)
        ax[1, j].set_title(f"Gradient corr (uncentred), pairs: {rule}", fontsize=10)
        for a in ax[:, j]:
            a.set_xscale("log", base=2); a.set_xticks(lagkm); a.set_xticklabels(lagkm); a.set_xlabel("pair lag (km; 2 km px x 2)"); a.legend()
        ax[0, j].set_ylabel("gradient ratio"); ax[1, j].set_ylabel("gradient correlation")
    fig.suptitle("Pair-difference ratio and correlation vs 2 km truth, fair hidden 2 km pixels")
    fig.tight_layout(); fig.savefig(FIG / "res_compare_gradients.png", dpi=120); plt.close(fig)


def main():
    r = compute()
    res = analyse(r)
    doc = __doc__.split('Method (also written to results/res_compare.json "assumptions")')[1].split("Run from")[0].strip()
    out = dict(assumptions=doc.split("\n- "), runs=dict(run2=str(RUN2.relative_to(ROOT)), run4=str(RUN4.relative_to(ROOT)),
               epochs_note="2 km: 20 epochs, batch 4, lr 1e-3; 4 km: 50 + 50 + 50 epochs (continued), batch 16, last stage lr 1e-4"),
               crops=r["crops"], data_quality=r["dq"], seconds=r["seconds"], **res)
    json.dump(out, open(RES / "res_compare.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    plots(res, r)
    print(json.dumps({k: out[k] for k in ["data_quality", "overall", "overall_val", "overall_test", "by_region", "by_distance_px", "gradients", "sanity_own_val_mae"]}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
