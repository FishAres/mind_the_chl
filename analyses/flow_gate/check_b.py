"""Flow gate, Check B: where could the gap-filler gain from motion? Motion-only predictors vs the best 4 km gap-filler
(models/pinn/t3d/blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50, preds.npz, 182 val days) on its hidden
pixels (hidden = preds 'hidden' & ocean & observed today; = observed today & not observed on the donor day).

Visibility of a context day t+k (k = -4..4, k != 0) to the model, as train_pinn.nb_masks (--context-clouds persist):
  j = calendar day t+k (missing in the store -> not available); jd = calendar day donor+k (missing -> the donor day itself);
  visible = observed(j) & observed(jd) & ocean. (viz.draw_day uses store-index offsets instead; differences counted.)
Predictors (each scored only where it has a value):
  (a) persistence: ln C(x, t-m) from the smallest m in 1..4 with x visible on t-m
  (b) x0.25 advection from the past: for m = 1..4, semi-Lagrangian backward trajectory from (x, t) to t-m with 0.25 x
      reanalysis currents (tracer.Integrator, RK2, 4 sub-steps/day, vmax 10 px/day, trajectory validity >= 0.99), value =
      bilinear sample of day t-m's VISIBLE pixels with visible weight >= 0.99; the smallest m with a valid value is used
      (source day chosen by visibility at the departure point, not at x)
  (c) as (b) over k in -4..4: order -1, +1, -2, +2, ... (ties in |k| -> the earlier day); future source days use the
      forward trajectory (negated velocity on the reversed day sequence)
  (d) DOY climatology (error_breakdown.doy_clim = train2.doy_climatology on the train days)
  Advection needs every calendar day between source and target in the store (else that k is skipped).
ln CHL as train_pinn (D.log_chl, no upper cap). Distance into the gap = Euclidean distance (EDT) of the hidden pixel to
the nearest visible pixel of day t (gradient_diagnosis), bins 1 / 2-3 / 4-7 / 8-15 / 16+ px.
Gradients (gradient_diagnosis definitions): adjacent-pixel pairs in x and y (lag 1 px), truth observed at both ends;
ratio = sum|dpred| / sum|dtruth|, corr = sum(dp dt) / sqrt(sum dp^2 sum dt^2) (uncentred); pairs where the predictor
is defined at both ends; (1) >= 1 hidden end (trainer's set), (2) both ends hidden.

Sections: compute() (cache results/check_b_px.npz), analyse(), report() (json, md fragment, figure).
Run from the repo root: CUDA_VISIBLE_DEVICES="" python -m analyses.flow_gate.check_b [--recompute]
"""
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
from analyses.flow_gate.common import *  # noqa: F401,F403
import time
from scipy.ndimage import distance_transform_edt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chl_cnn.plotting import OKABE_ITO as C
from chl_cnn import clouds as CL
from chl_cnn.runs import find_run, category_of

RUN = "blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50"
PX = RES / "check_b_px.npz"
KMAX = 4
DIST_EDGES = np.array([0.5, 1.5, 3.5, 7.5, 15.5, np.inf]); DIST_LAB = ["1", "2-3", "4-7", "8-15", "16+"]
PREDS = {"a": "(a) persistence, nearest past visible day", "b": "(b) x0.25 advected, nearest past day",
         "c": "(c) x0.25 advected, nearest past/future day", "d": "(d) DOY climatology"}
GPRED = ["b", "c"]


class Acc:
    def __init__(self):
        self.s = np.zeros(6)          # ap, at, pt, pp, tt, n

    def add(self, dp, dt):
        dp = dp.astype("float64"); dt = dt.astype("float64")
        self.s += [np.abs(dp).sum(), np.abs(dt).sum(), (dp * dt).sum(), (dp * dp).sum(), (dt * dt).sum(), dp.size]

    def out(self):
        ap, at, pt, pp, tt, n = self.s
        return dict(ratio=ap / at if n else None, corr=pt / np.sqrt(pp * tt) if n else None, n_pairs=int(n))


def pairs(a, ax):
    return (a[:, :-1], a[:, 1:]) if ax == 1 else (a[:-1], a[1:])


# ============================== data generation ==============================
def compute():
    t0 = time.time()
    S = load_store(); d, ys, xs, ocean = S["d"], S["ys"], S["xs"], S["ocean"]
    rd = find_run(RUN); cat = category_of(rd)
    cfg = json.load(open(rd / "config.json"))
    assert cfg["context_before"] == KMAX and cfg["context_after"] == KMAX and cfg["context_clouds"] == "persist" and cfg["donor_exclude"] == 0
    z = np.load(rd / "preds.npz")
    pos = {int(i): k for k, i in enumerate(z["idx"])}
    va = S["va"]; tday = S["tday"]; n = len(tday)
    y = lambda i: D.log_chl(np.asarray(d["chl_l3"][i, ys, xs]))
    obs_cache = {}

    def obs(i):
        if i not in obs_cache:
            obs_cache[i] = np.isfinite(y(i)) & ocean
        return obs_cache[i]
    # fixed val donors, as C.fixed_hidden(obs, va, seed=eval_seed, exclude=0)
    local = CL.donor_indices(len(va), np.random.default_rng(cfg["eval_seed"]))
    donor = dict(zip(va.tolist(), va[local].tolist()))
    NB = {k: neighbour_index(tday, k) for k in range(-KMAX, KMAX + 1) if k != 0}
    from analyses.blk4.error_breakdown import doy_clim
    clim, cbin = doy_clim(dict(times=S["times"], ocean=ocean, tr=S["tr"], y=y))
    reg, ties = regions(ocean)
    it = integrator(S); ucache = {}
    cols = {c: [] for c in ["day", "region", "dist", "truth", "gf", "a", "lag_a", "b", "lag_b", "c", "lag_c", "d"]}
    gacc = {(set_, p, who): Acc() for set_ in ("ge1", "both") for p in GPRED for who in ("pred", "gf")}
    dq = dict(n_days=0, hidden_mask_mismatch_px=0, hidden_off_obs_px=0, gf_nonfinite_ocean_px=0, gf_nonfinite_hidden_px=0,
              chl_gt100_hidden_px=0, days_no_hidden=0, days_no_visible=0, ctx_day_missing={k: 0 for k in NB},
              adv_path_missing={k: 0 for k in NB}, viz_rule_vs_trainer_visible_diff_px=0, viz_rule_diff_days=0,
              clim_nonfinite_hidden_px=0, adv_nonfinite_valid_px=0)
    sign_chk = {k: np.zeros(4) for k in (-1, 1)}          # sum|err| correct, sum|err| reversed velocity, n, sum|err| persistence(lag) same px
    for i in (int(v) for v in z["val_idx"]):
        dq["n_days"] += 1
        k0 = pos[i]; dn = donor[i]
        yt = y(i); ot = np.isfinite(yt) & ocean
        hid_store = ot & ~obs(dn)
        hz = z["hidden"][k0] & ocean
        dq["hidden_mask_mismatch_px"] += int((hz != hid_store).sum())
        dq["hidden_off_obs_px"] += int((hz & ~ot).sum())
        hid = hz & ot; vis = ot & ~hid
        if not hid.any():
            dq["days_no_hidden"] += 1; continue
        cchl = np.asarray(d["chl_l3"][i, ys, xs]); dq["chl_gt100_hidden_px"] += int((hid & (cchl > 100)).sum())
        gf = z["pred"][k0].astype("float32")
        dq["gf_nonfinite_ocean_px"] += int((~np.isfinite(gf) & ocean).sum()); dq["gf_nonfinite_hidden_px"] += int((~np.isfinite(gf) & hid).sum())
        if vis.any():
            dist = distance_transform_edt(~vis).astype("float32")
        else:
            dq["days_no_visible"] += 1; dist = np.full(yt.shape, np.inf, "float32")
        pers, adv = {}, {}
        for k in NB:
            j = int(NB[k][i])
            if j < 0:
                dq["ctx_day_missing"][k] += 1; continue
            jd = int(NB[k][dn]); jd = jd if jd >= 0 else dn
            yj = y(j); vk = obs(j) & obs(jd)
            # viz.draw_day rule (store-index offsets) for comparison
            if 0 <= i + k < n and 0 <= dn + k < n:
                vz = obs(i + k) & obs(dn + k)
                df = int((vz != vk).sum()); dq["viz_rule_vs_trainer_visible_diff_px"] += df; dq["viz_rule_diff_days"] += int(df > 0)
            if k < 0:
                pers[k] = (np.where(vk, yj, np.nan))
            path = [int(NB[s][i]) if s != 0 else i for s in (range(k, 1) if k < 0 else range(k, -1, -1))]
            if any(p < 0 for p in path):
                dq["adv_path_missing"][k] += 1; continue
            pv, ok = advect(S, it, path, yj, vk, ucache)
            dq["adv_nonfinite_valid_px"] += int((ok & ~np.isfinite(pv)).sum())
            adv[k] = np.where(ok, pv, np.nan)
            if abs(k) == 1:                                   # time-direction / sign check: reversed velocity
                pr_, okr = advect_reversed(S, it, path, yj, vk)
                mm = hid & ok & okr & vk
                sign_chk[k] += [np.abs(pv - yt)[mm].sum(), np.abs(pr_ - yt)[mm].sum(), mm.sum(), np.abs(yj - yt)[mm].sum()]

        def pick(order, src):
            v = np.full(yt.shape, np.nan, "float32"); lag = np.zeros(yt.shape, np.int8)
            for k in order:
                if k in src:
                    f = np.isnan(v) & np.isfinite(src[k])
                    v[f] = src[k][f]; lag[f] = k
            return v, lag
        fa, la = pick(range(-1, -KMAX - 1, -1), pers)
        fb, lb = pick(range(-1, -KMAX - 1, -1), adv)
        fc, lc = pick([s * m for m in range(1, KMAX + 1) for s in (-1, 1)], adv)
        fd = clim[cbin[i]]
        dq["clim_nonfinite_hidden_px"] += int((~np.isfinite(fd) & hid).sum())
        h = hid
        for c, v in (("day", np.full(int(h.sum()), i, np.int32)), ("region", reg[h]), ("dist", dist[h]), ("truth", yt[h]), ("gf", gf[h]),
                     ("a", fa[h]), ("lag_a", la[h]), ("b", fb[h]), ("lag_b", lb[h]), ("c", fc[h]), ("lag_c", lc[h]), ("d", fd[h])):
            cols[c].append(v)
        for ax in (0, 1):                                     # lag-1 px gradient pairs
            o1, o2 = pairs(ot, ax); h1, h2 = pairs(hid, ax); t1, t2 = pairs(yt, ax); g1, g2 = pairs(gf, ax)
            sets = dict(ge1=o1 & o2 & (h1 | h2), both=o1 & o2 & h1 & h2)
            for p, F in (("b", fb), ("c", fc)):
                p1, p2 = pairs(F, ax); ok = np.isfinite(p1) & np.isfinite(p2)
                for nm, m in sets.items():
                    mm = m & ok
                    gacc[(nm, p, "pred")].add((p2 - p1)[mm], (t2 - t1)[mm]); gacc[(nm, p, "gf")].add((g2 - g1)[mm], (t2 - t1)[mm])
        if dq["n_days"] % 20 == 0:
            print(f"  day {dq['n_days']}/{len(z['val_idx'])} {time.time() - t0:.0f}s", flush=True)
        if len(obs_cache) > 64:
            obs_cache.clear()
        if len(ucache) > 64:
            ucache.clear()
    out = {c: np.concatenate(v) for c, v in cols.items()}
    grad = {f"{a}|{b}|{c}": v.out() for (a, b, c), v in gacc.items()}
    dq["ctx_day_missing"] = {str(k): v for k, v in dq["ctx_day_missing"].items()}
    dq["adv_path_missing"] = {str(k): v for k, v in dq["adv_path_missing"].items()}
    meta = dict(run=RUN, category=cat, dq=dq, grad=grad, region_ties=int(ties), seconds=round(time.time() - t0, 1),
                sign_check={str(k): dict(mae_adv=v[0] / v[2], mae_adv_reversed_velocity=v[1] / v[2], mae_persistence=v[3] / v[2], n_px=int(v[2]))
                            for k, v in sign_chk.items()},
                metrics_best_val_mae=cfg.get("best_val_mae"))
    np.savez_compressed(PX, meta=json.dumps(meta, default=float), **out)
    print(json.dumps(meta, indent=1, default=float), flush=True)


def advect_reversed(S, it, path, y_src, obs_src):
    """Check only: the same advection with the velocity sign flipped."""
    sign = 1.0 if path[-1] > path[0] else -1.0
    Us, oks = [], []
    for i in path:
        U, ok = currents_px(S, i); Us.append(-sign * U); oks.append(ok)
    U = torch.as_tensor(np.stack(Us)[None]); ok = torch.as_tensor(np.stack(oks)[None].astype("float32"))
    with torch.no_grad():
        pos, val = it.departure(U, ok, len(path) - 1)
        pred, v2 = it.depart_sample(torch.as_tensor(np.nan_to_num(y_src)[None].astype("float32")), torch.as_tensor(obs_src[None]), pos)
    return pred[0].numpy(), (val & v2)[0].numpy()


# ============================== analysis ==============================
def row(t, m, p, n_hidden_group):
    mm = m & np.isfinite(t[p])
    n = int(mm.sum()); days = np.unique(t["day"][mm])
    r = dict(n_px=n, n_days=int(len(days)), coverage=n / n_hidden_group if n_hidden_group else None, flag_lt5=bool(n < MIN_N or len(days) < MIN_N))
    if r["flag_lt5"]:
        r.update(mae_pred=None, mae_gf=None, diff=None, diff_ci=None); return r
    ep = np.abs(t[p][mm] - t["truth"][mm]).astype("float64"); eg = np.abs(t["gf"][mm] - t["truth"][mm]).astype("float64")
    r.update(mae_pred=float(ep.mean()), mae_gf=float(eg.mean()), diff=float(ep.mean() - eg.mean()),
             diff_ci=boot_ratio_diff(t["day"][mm], ep, eg, np.ones(n)))
    return r


def analyse():
    zz = np.load(PX)
    meta = json.loads(str(zz["meta"]))
    t = {k: zz[k] for k in zz.files if k != "meta"}
    N = len(t["truth"])
    gid = np.searchsorted(DIST_EDGES, t["dist"], side="right") - 1
    gid[t["dist"] < 0.5] = -1
    res = dict(n_hidden=N, gf_mae_all_hidden=float(np.abs(t["gf"] - t["truth"]).mean()),
               gf_mae_day_mean=float(np.mean([np.abs(t["gf"][t["day"] == q] - t["truth"][t["day"] == q]).mean() for q in np.unique(t["day"])])),
               dist_unbinned=int((gid < 0).sum()), nan_counts={k: int(np.isnan(t[k]).sum()) for k in ("truth", "gf", "a", "b", "c", "d")})
    allm = np.ones(N, bool)
    res["overall"] = {p: row(t, allm, p, N) for p in PREDS}
    res["by_distance"] = {lab: {p: row(t, gid == g, p, int((gid == g).sum())) for p in PREDS} for g, lab in enumerate(DIST_LAB)}
    res["by_region"] = {REG_NAMES[g]: {p: row(t, t["region"] == g, p, int((t["region"] == g).sum())) for p in PREDS} for g in range(1, 8)}
    res["by_lag"] = {}
    for p, lc in (("a", "lag_a"), ("b", "lag_b"), ("c", "lag_c")):
        ks = range(-KMAX, 0) if p != "c" else [k for k in range(-KMAX, KMAX + 1) if k != 0]
        res["by_lag"][p] = {str(k): row(t, t[lc] == k, p, N) for k in ks}
    # paired: (b) vs (a) on px where both exist (motion vs no motion from the same nearest-past search)
    m = np.isfinite(t["a"]) & np.isfinite(t["b"])
    res["b_vs_a_same_px"] = dict(n_px=int(m.sum()), mae_a=float(np.abs(t["a"][m] - t["truth"][m]).mean()),
                                 mae_b=float(np.abs(t["b"][m] - t["truth"][m]).mean()), mae_gf=float(np.abs(t["gf"][m] - t["truth"][m]).mean()))
    res["grad"] = meta["grad"]
    res["meta"] = meta
    return res


# ============================== reporting ==============================
def report(res):
    json.dump(res, open(RES / "check_b.json", "w"), indent=1, default=float)
    L = [f"### Check B: motion-only predictors vs the gap-filler on its hidden val pixels", "",
         f"Hidden px {res['n_hidden']:,} on {res['meta']['dq']['n_days']} val days; gap-filler MAE all hidden px {res['gf_mae_all_hidden']:.4f} "
         f"(pixel mean), {res['gf_mae_day_mean']:.4f} (mean of daily MAE); metrics.json best_val_mae {res['meta']['metrics_best_val_mae']:.4f}.", "",
         "Overall (each predictor on its own px; gap-filler on the same px):", "",
         "| predictor | pred - gap-filler [95 % CI] | MAE pred | MAE gap-filler | coverage | n px | n days |", "|---|---|---|---|---|---|---|"]

    def line(lab, r):
        if r["flag_lt5"]:
            return f"| {lab} | n < 5 | | | {r['coverage'] or 0:.1%} | {r['n_px']:,} | {r['n_days']} |"
        ci = r["diff_ci"]
        return (f"| {lab} | {r['diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {r['mae_pred']:.4f} | {r['mae_gf']:.4f} | "
                f"{r['coverage']:.1%} | {r['n_px']:,} | {r['n_days']} |")
    for p, lab in PREDS.items():
        L.append(line(lab, res["overall"][p]))
    for key, ttl in (("by_distance", "distance into the gap (px, EDT to nearest visible px)"), ("by_region", "region")):
        L += ["", f"By {ttl}; coverage = share of the group's hidden px:", "",
              "| group | predictor | pred - gap-filler [CI] | MAE pred | MAE gap-filler | coverage | n px | n days |", "|---|---|---|---|---|---|---|---|"]
        for g, rr in res[key].items():
            for p in PREDS:
                L.append(f"| {g} " + line(p, rr[p]))
    L += ["", "By lag of the source day (coverage = share of all hidden px):", "",
          "| predictor | lag (days) | pred - gap-filler [CI] | MAE pred | MAE gap-filler | coverage | n px | n days |", "|---|---|---|---|---|---|---|---|"]
    for p, rr in res["by_lag"].items():
        for k, r in rr.items():
            L.append(f"| {p} " + line(k, r))
    b = res["b_vs_a_same_px"]
    L += ["", f"(a) and (b) both defined ({b['n_px']:,} px): persistence {b['mae_a']:.4f}, x0.25 advected {b['mae_b']:.4f}, gap-filler {b['mae_gf']:.4f}.", "",
          "Gradient ratio / corr, lag 1 px, pairs with the predictor defined at both ends:", "",
          "| pair set | predictor | ratio pred | ratio gap-filler | corr pred | corr gap-filler | n pairs |", "|---|---|---|---|---|---|---|"]
    for s_, slab in (("ge1", ">= 1 hidden end (trainer)"), ("both", "both ends hidden")):
        for p in GPRED:
            a, g = res["grad"][f"{s_}|{p}|pred"], res["grad"][f"{s_}|{p}|gf"]
            L.append(f"| {slab} | {p} | {fmt(a['ratio'], 3)} | {fmt(g['ratio'], 3)} | {fmt(a['corr'], 3)} | {fmt(g['corr'], 3)} | {a['n_pairs']:,} |")
    sc = res["meta"]["sign_check"]
    L += ["", "Time-direction / sign check, lag +-1, hidden px visible at x on the source day: " + "; ".join(
        f"k {k}: advected {v['mae_adv']:.4f}, reversed velocity {v['mae_adv_reversed_velocity']:.4f}, persistence {v['mae_persistence']:.4f} (n {v['n_px']:,})" for k, v in sc.items()), ""]
    (RES / "check_b.md").write_text("\n".join(L))
    print("\n".join(L))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
    x = np.arange(len(DIST_LAB))
    cols = dict(a=C[1], b=C[0], c=C[2], d=C[4])
    for p, lab in PREDS.items():
        rr = [res["by_distance"][g][p] for g in DIST_LAB]
        mp = [r["mae_pred"] if not r["flag_lt5"] else np.nan for r in rr]; mg = [r["mae_gf"] if not r["flag_lt5"] else np.nan for r in rr]
        ax[0].plot(x, mp, "-o", color=cols[p], label=lab)
        ax[0].plot(x, mg, "--", marker="x", color=cols[p], label=f"gap-filler on {p}'s px")
        ax[1].plot(x, [r["coverage"] for r in rr], "-o", color=cols[p], label=lab)
    ax[0].set(xticks=x, xticklabels=DIST_LAB, xlabel="distance of hidden px to nearest visible px (px, 1 px ~ 4.5 km)",
              ylabel="MAE, ln CHL (hidden val px)", title="Hidden-pixel MAE by distance into the gap:\nmotion-only predictors (solid) vs gap-filler on the same px (dashed)")
    ax[0].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14)); ax[0].grid(alpha=.3)
    ax[1].set(xticks=x, xticklabels=DIST_LAB, xlabel="distance of hidden px to nearest visible px (px)", ylabel="share of hidden px with a value",
              title="Predictor coverage of hidden val px by distance into the gap", ylim=(0, 1.05))
    ax[1].legend(fontsize=7, loc="lower left"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "check_b_distance.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    if "--recompute" in sys.argv or not PX.exists():
        compute()
    report(analyse())
