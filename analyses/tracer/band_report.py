"""Band-split tracer, TEST 2 report (CPU): runs from analyses/tracer/band_train.py (models/tracer/trb_*), compared with
test 1 (analyses/tracer/band_test1.py), step 1 and step 2 on the test-1 common set.

Per run: best epoch (min mean-lead val own-set MAE, as step 1 / 2) and the mean over the last 5 epochs of each metric.
Model and references (step1, x0.25, persistence, lp_alone, test1 = x0.25 bands with measured tau) are scored every epoch
on (test-1 common mask & model valid); step 2 (tr_src, tr_src_s1, saved best epoch only) comes from band_test1 on the
full test-1 common mask: identical px when the model is valid on every mask px (checked per start and lead; stated).
Day-bootstrap (2000 resamples of the 351 start days) 95 % CIs: best epoch: (sum|e_model| - sum|e_ref|) / n; last 5:
mean over the 5 epochs of that difference, same resampled days in every epoch.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
MODELS = ROOT / "models/tracer"
RUNS = ["trb_const_s0", "trb_const_s1", "trb_phys_s0", "trb_phys_s1"]
K = 5
LAGS = ("1", "4", "16")
REG = {1: "NW shelf", 2: "W open sea", 3: "E open sea", 4: "Anatolian shelf", 5: "Crimea/Kerch/Caucasus", 6: "Azov+Kerch", 7: "Marmara/Bosporus"}
BANDS = ["< 20 km", "20-50 km", "50-150 km"]
SEAS = ["DJF", "MAM", "JJA", "SON"]


def load(run):
    p = MODELS / run
    if not (p / "metrics.json").exists():
        return None
    h = json.load(open(p / "history.json")); m = json.load(open(p / "metrics.json"))
    z = np.load(p / "perday.npz")
    return dict(h=h, m=m, acc=z["acc"], preds=list(z["preds"]), cfg=json.load(open(p / "config.json")), p=p)


def boot(acc_i, acc_j, epochs, nb=2000, seed=0):
    """acc_*: (E+1, ns, K, 4) per-day sums; epochs list -> per lead diff and CI of mean over epochs of MAE_i - MAE_j (each MAE
    = its own sum|e| / its own n; identical px except for the step-2 rows, see report text)."""
    rng = np.random.default_rng(seed); ns = acc_i.shape[1]
    Q = rng.integers(0, ns, (nb, ns))
    out = []
    for k in range(K):
        d0 = np.mean([acc_i[e, :, k, 0].sum() / acc_i[e, :, k, 1].sum() - acc_j[e, :, k, 0].sum() / acc_j[e, :, k, 1].sum() for e in epochs])
        bs = np.mean([acc_i[e][Q, k, 0].sum(1) / acc_i[e][Q, k, 1].sum(1) - acc_j[e][Q, k, 0].sum(1) / acc_j[e][Q, k, 1].sum(1) for e in epochs], 0)
        out.append(dict(diff=float(d0), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
    return out


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from chl_cnn.plotting import OKABE_ITO as C, ERR_CMAP
    t1 = json.load(open(RES / "band_test1.json")); z1 = np.load(RES / "band_test1_days.npz"); p1 = list(z1["preds"]); acc1 = z1["acc"]
    R = {r: load(r) for r in RUNS}
    R = {r: v for r, v in R.items() if v is not None}
    out = dict(assumptions=__doc__.strip(), runs={})
    L = ["# Band-split tracer, test 2 (learned): val 2017 + 2021, 351 starts, test-1 common set", ""]
    tab = []
    for r, v in R.items():
        h, m, acc, preds = v["h"], v["m"], v["acc"], v["preds"]
        ev = h["eval"]; E = len(ev) - 1
        be = m["best_epoch"]; last = list(range(max(1, E - 4), E + 1))
        qi = {p: i for i, p in enumerate(preds)}
        mean5 = lambda f: np.mean([f(ev[e]) for e in last], 0)
        rr = dict(best_epoch=be, last5_epochs=last, n_epochs=E)
        for tag, eps in (("best", [be]), ("last5", last)):
            g = lambda f: np.mean([f(ev[e]) for e in eps], 0)
            rr[tag] = dict(mae={p: g(lambda e_: np.array(e_["mae"][p])).tolist() for p in preds}, band={p: g(lambda e_: np.array(e_["band_mae"][p])).tolist() for p in preds},
                           n=g(lambda e_: np.array(e_["n"])).tolist(), n_mask=ev[eps[0]]["n_mask"], own_mae=g(lambda e_: np.array(e_["own_mae"])).tolist(),
                           grad_l1={p: {L_: [float(g(lambda e_: e_["grad"][p]["1"][L_]["ratio"])), float(g(lambda e_: e_["grad"][p]["1"][L_]["corr"]))] for L_ in LAGS} for p in preds},
                           region_l1={p: {REG[int(q)]: dict(mae=float(g(lambda e_: e_["region_lead1"][p][q]["mae"])), n=int(ev[eps[0]]["region_lead1"][p][q]["n"])) for q in ev[eps[0]]["region_lead1"][p]} for p in preds},
                           a=g(lambda e_: np.array(e_["a"])).tolist(), tau=g(lambda e_: np.array(e_["tau"])).tolist(),
                           srcB=g(lambda e_: np.array([e_["srcB_lead1"][s] for s in SEAS] + [e_["srcB_lead1_all"]])).tolist())
            if "tau_mult_mean" in ev[eps[0]]:
                rr[tag]["tau_mult"] = g(lambda e_: np.array([e_["tau_mult_mean"][s] for s in SEAS])).tolist()
            # same px as the test-1 mask? (model valid on every mask px)
            miss = [float((acc1[:, :, p1.index("step1"), 1] - acc[e, :, :, 0, 1]).sum(0).max()) for e in eps]
            same = max((acc1[:, :, p1.index("step1"), 1] - acc[e, :, :, 0, 1]).sum(0).max() / acc1[:, :, p1.index("step1"), 1].sum(0).min() for e in eps) < 1e-3
            rr[tag]["mask_px_model_invalid_max_per_lead"] = max(miss)
            rr[tag]["same_px_as_test1_mask"] = bool(same)
            cis = {f"model - {p}": boot(acc[:, :, :, qi["model"]], acc[:, :, :, qi[p]], eps) for p in ["step1", "test1", "x0.25"]}
            rs = RES / "band_step2_rescore.npz"
            if rs.exists():
                zr = np.load(rs); jobs = list(zr["jobs"])
                if all(f"{r}@{e}" in jobs for e in eps):
                    for i2, s2 in enumerate(["tr_src", "tr_src_s1"], 1):
                        a2 = np.zeros_like(acc[:, :, :, 0]); am = np.zeros_like(a2)
                        for e in eps:
                            a2[e] = zr["acc"][jobs.index(f"{r}@{e}"), :, :, i2]; am[e] = zr["acc"][jobs.index(f"{r}@{e}"), :, :, 0]
                        rr[tag]["mae_" + s2 + "_same_px"] = [float(np.mean([a2[e, :, k, 0].sum() / a2[e, :, k, 1].sum() for e in eps])) for k in range(K)]
                        rr[tag]["band_" + s2 + "_same_px"] = [float(np.mean([a2[e, :, k, 2].sum() / a2[e, :, k, 3].sum() for e in eps])) for k in range(K)]
                        rr[tag]["rescore_model_sums_match"] = bool(all(np.allclose(am[e], acc[e, :, :, qi["model"]]) for e in eps))
                        cis[f"model - {s2} (best ep; same px, re-scored)"] = boot(acc[:, :, :, qi["model"]], a2, eps)
            elif same:
                for s2 in ["tr_src", "tr_src_s1"]:
                    a2 = np.repeat(acc1[None, :, :, p1.index(s2)], acc.shape[0], 0)
                    cis[f"model - {s2} (best ep; test-1 sums, mask px)"] = boot(acc[:, :, :, qi["model"]], a2, eps)
            rr[tag]["ci"] = cis
        rr["n_mask"] = ev[0]["n_mask"]
        sb = np.array([e_["srcB_lead1_all"] for e_ in ev[1:]]); l5 = np.array([e_["mae"]["model"][4] for e_ in ev[1:]])
        rr["r_srcB_vs_l5_over_epochs"] = float(np.corrcoef(sb, l5)[0, 1]) if len(sb) > 2 else None
        rr["val_l5_range_epochs"] = [float(l5.min()), float(l5.max())]
        out["runs"][r] = rr
    # phys vs const, matched seed (each MAE on its own px: test-1 mask & that run's valid px)
    out["phys_minus_const"] = {}
    for sd in ("s0", "s1"):
        a_, b_ = f"trb_phys_{sd}", f"trb_const_{sd}"
        if a_ in R and b_ in R:
            for tag in ("best", "last5"):
                ea, eb = out["runs"][a_][tag], out["runs"][b_][tag]
                qa, qb = R[a_]["preds"].index("model"), R[b_]["preds"].index("model")
                Aa, Ab = R[a_]["acc"][:, :, :, qa], R[b_]["acc"][:, :, :, qb]
                epa = [out["runs"][a_]["best_epoch"]] if tag == "best" else out["runs"][a_]["last5_epochs"]
                epb = [out["runs"][b_]["best_epoch"]] if tag == "best" else out["runs"][b_]["last5_epochs"]
                rng = np.random.default_rng(0); Q = rng.integers(0, Aa.shape[1], (2000, Aa.shape[1])); res_ = []
                for k in range(K):
                    fa = lambda Q_: np.mean([Aa[e][Q_, k, 0].sum(-1) / Aa[e][Q_, k, 1].sum(-1) for e in epa], 0)
                    fb = lambda Q_: np.mean([Ab[e][Q_, k, 0].sum(-1) / Ab[e][Q_, k, 1].sum(-1) for e in epb], 0)
                    allq = np.arange(Aa.shape[1])[None]
                    bs = fa(Q) - fb(Q)
                    res_.append(dict(diff=float(fa(allq)[0] - fb(allq)[0]), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
                out["phys_minus_const"][f"{sd} {tag}"] = res_
    (RES / "band_test2.json").write_text(json.dumps(out, indent=1))
    f4 = lambda x: f"{x:.4f}"
    # ---- main table
    L += ["Pixel MAE (ln CHL) on the test-1 common set & model valid; best = epoch with min mean-lead own-set val MAE; last5 = mean over epochs 16-20.", "",
          "| predictor | l1 | l2 | l3 | l4 | l5 | band 20-70 l1 | l3 | l5 | n l1 |", "|---" * 10 + "|"]
    for r, rr in out["runs"].items():
        for tag in ("best", "last5"):
            x = rr[tag]
            nm = f"{r} {tag}" + (f" (ep {rr['best_epoch']})" if tag == "best" else "")
            L.append(f"| {nm} | " + " | ".join(f4(v) for v in x["mae"]["model"]) + " | " + " | ".join(f4(x["band"]["model"][k]) for k in (0, 2, 4)) + f" | {int(x['n'][0]):,} |")
    rr0 = next(iter(out["runs"].values()))["best"] if out["runs"] else None
    for p, lb in [("test1", "test 1 (x0.25 bands, measured tau)"), ("step1", "step 1"), ("x0.25", "currents x0.25"), ("persistence", "persistence"), ("lp_alone", "lp base alone")]:
        if rr0:
            L.append(f"| {lb} | " + " | ".join(f4(v) for v in rr0["mae"][p]) + " | " + " | ".join(f4(rr0["band"][p][k]) for k in (0, 2, 4)) + f" | {int(rr0['n'][0]):,} |")
    for r, rr in out["runs"].items():
        for tag in ("best", "last5"):
            for s2 in ["tr_src", "tr_src_s1"]:
                if "mae_" + s2 + "_same_px" in rr[tag]:
                    L.append(f"| step 2 {s2} (best ep) on {r} {tag} px | " + " | ".join(f4(v) for v in rr[tag]["mae_" + s2 + "_same_px"]) + " | "
                             + " | ".join(f4(rr[tag]["band_" + s2 + "_same_px"][k]) for k in (0, 2, 4)) + f" | {int(rr[tag]['n'][0]):,} |")
    for s2 in ["tr_src", "tr_src_s1"]:
        L.append(f"| step 2 {s2} (best ep, test-1 set) | " + " | ".join(f4(v) for v in t1["mae"][s2]) + " | " + " | ".join(f4(t1["band_mae"][s2][k]) for k in (0, 2, 4)) + f" | {int(t1['n_common'][0]):,} |")
    L += ["", "Step 2 last-5-epoch means (history.json, train_tracer's own per-epoch common set, NOT this set; FINDINGS): tr_src 0.1517/0.1878/0.2047/0.2320/0.2410, "
          "tr_src_s1 0.1512/0.1858/0.2022/0.2281/0.2367; on that set at the best epoch tr_src 0.1526/0.1822/0.1995/0.2173/0.2219 (final common set incl. ref).", "",
          "Day-bootstrap 95 % CI of pixel-MAE differences, leads 1-5:", ""]
    for r, rr in out["runs"].items():
        for tag in ("best", "last5"):
            for nm, v in rr[tag]["ci"].items():
                L.append(f"- {r} {tag}: {nm}: " + "; ".join(f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in v))
        L.append(f"- {r}: mask px where the model is invalid, max over leads (best / last-5 epochs): {int(rr['best']['mask_px_model_invalid_max_per_lead'])} / {int(rr['last5']['mask_px_model_invalid_max_per_lead'])} (step-2 rows / CIs re-scored on the model's px by band_step2_rescore.py when available; re-scored model sums equal perday.npz: {rr['best'].get('rescore_model_sums_match')}); n mask l1 {int(rr['n_mask'][0]):,}")
    if out["phys_minus_const"]:
        L += ["", "Phys-tau minus const-tau, matched seed (each on its own px; day bootstrap 95 % CI), pixel MAE leads 1-5:", ""]
        for nm, v in out["phys_minus_const"].items():
            L.append(f"- {nm}: " + "; ".join(f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in v))
    L += ["", "Learned parameters (init a 0.25, tau 1.67 / 2.38 / 13.77 d):", "", "| run | tag | a < 20 | a 20-50 | a 50-150 | tau < 20 (d) | tau 20-50 | tau 50-150 | lead-1 dB basin mean (ln CHL/day) DJF / MAM / JJA / SON / all |", "|---" * 9 + "|"]
    for r, rr in out["runs"].items():
        for tag in ("best", "last5"):
            x = rr[tag]
            L.append(f"| {r} | {tag} | " + " | ".join(f"{v:.3f}" for v in x["a"]) + " | " + " | ".join(f"{v:.2f}" for v in x["tau"]) + " | " + " / ".join(f"{v:+.4f}" for v in x["srcB"]) + " |")
    L += ["", "Across epochs 1-20: Pearson r of val basin-mean lead-1 dB with val lead-5 MAE (common set); range of val lead-5 MAE:", ""]
    for r, rr in out["runs"].items():
        L.append(f"- {r}: r {rr['r_srcB_vs_l5_over_epochs']:+.2f} (n {rr['n_epochs']} epochs); l5 MAE {rr['val_l5_range_epochs'][0]:.4f}-{rr['val_l5_range_epochs'][1]:.4f}")
    if any("tau_mult" in rr["best"] for rr in out["runs"].values()):
        L += ["", "Phys runs: mean tau multiplier over ocean px of val days by season (bands < 20 / 20-50 / 50-150):", ""]
        for r, rr in out["runs"].items():
            if "tau_mult" in rr["best"]:
                for tag in ("best", "last5"):
                    L.append(f"- {r} {tag}: " + "; ".join(f"{s} " + "/".join(f"{v:.2f}" for v in rr[tag]["tau_mult"][q]) for q, s in enumerate(SEAS)))
    L += ["", "Lead-1 gradient ratio / corr at lags 1 / 4 / 16 px (best epoch; last5 mean in brackets for the model):", "", "| predictor | lag 1 | lag 4 | lag 16 |", "|---|---|---|---|"]
    for r, rr in out["runs"].items():
        L.append(f"| {r} | " + " | ".join(f"{rr['best']['grad_l1']['model'][L_][0]:.3f} / {rr['best']['grad_l1']['model'][L_][1]:.3f} ({rr['last5']['grad_l1']['model'][L_][0]:.3f} / {rr['last5']['grad_l1']['model'][L_][1]:.3f})" for L_ in LAGS) + " |")
    if rr0:
        for p in ["test1", "step1", "x0.25", "persistence"]:
            L.append(f"| {p} | " + " | ".join(f"{rr0['grad_l1'][p][L_][0]:.3f} / {rr0['grad_l1'][p][L_][1]:.3f}" for L_ in LAGS) + " |")
    for s2 in ["tr_src", "tr_src_s1"]:
        L.append(f"| {s2} (test-1 set) | " + " | ".join(f"{t1['grad_lead1'][s2][L_]['ratio']:.3f} / {t1['grad_lead1'][s2][L_]['corr']:.3f}" for L_ in LAGS) + " |")
    L += ["", "Lead-1 MAE by region (best epoch):", "", "| region | " + " | ".join(out["runs"]) + " | test1 | step1 | tr_src (test-1 set) | n |", "|---" * (len(out["runs"]) + 5) + "|"]
    for q in range(1, 8):
        nm = REG[q]
        if not rr0:
            break
        row = [rr["best"]["region_l1"]["model"][nm]["mae"] for rr in out["runs"].values()]
        n_ = rr0["region_l1"]["step1"][nm]["n"]
        L.append(f"| {nm} | " + " | ".join(f4(v) if n_ else "-" for v in row) + f" | {f4(rr0['region_l1']['test1'][nm]['mae']) if n_ else '-'} | {f4(rr0['region_l1']['step1'][nm]['mae']) if n_ else '-'} | "
                 f"{f4(t1['region_lead1']['tr_src'][nm]['mae']) if n_ else '-'} | {n_:,} |")
    (RES / "band_test2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not R:
        return
    # ---- figures: val curves
    names = list(R)
    cols = {r: C[i] for i, r in enumerate(names)}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for r, v in R.items():
        ev = v["h"]["eval"]; ep = np.arange(len(ev))
        for i, k in enumerate((0, 2, 4)):
            axes[0, i].plot(ep, [e["mae"]["model"][k] for e in ev], color=cols[r], marker=".", label=r)
        axes[1, 0].plot(ep, [e["own_mean_lead"] for e in ev], color=cols[r], marker=".", label=r)
        axes[1, 0].axvline(v["m"]["best_epoch"], color=cols[r], lw=.6, ls=":")
        axes[1, 1].plot(v["h"]["epoch"], v["h"]["train_l1"], color=cols[r], marker=".", label=r)
        axes[1, 2].plot(ep, [e["srcB_lead1_all"] for e in ev], color=cols[r], marker=".", label=r)
    ev0 = R[names[0]]["h"]["eval"][0]
    for i, k in enumerate((0, 2, 4)):
        for p, c, ls in [("step1", C[7], "--"), ("test1", C[1], ":"), ("x0.25", C[3], "-.")]:
            axes[0, i].axhline(ev0["mae"][p][k], color=c, ls=ls, lw=1, label=f"{p} (epoch-0 px)")
        axes[0, i].axhline(t1["mae"]["tr_src"][k], color=C[4], ls="-", lw=1, label="step 2 tr_src best ep (test-1 set)")
        axes[0, i].set(xlabel="epoch (0 = init)", ylabel="val pixel MAE (ln CHL)", title=f"Val MAE, lead {k + 1} d, test-1 common set")
    axes[1, 0].set(xlabel="epoch (0 = init)", ylabel="val mean-lead MAE, own set", title="Val selection criterion (dotted: best epoch)")
    axes[1, 1].set(xlabel="epoch", ylabel="train L1, mean over leads", title="Train loss by epoch")
    axes[1, 2].axhline(0, color="0.5", lw=.8)
    axes[1, 2].set(xlabel="epoch (0 = init)", ylabel="basin-mean B(t+1) - B(t) (ln CHL / day)", title="Lead-1 base source, val basin mean")
    axes[0, 0].legend(fontsize=7)
    for a_ in axes.flat: a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "band_test2_curves.png", dpi=110); plt.close(fig)
    # parameters by epoch
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for r, v in R.items():
        ev = v["h"]["eval"]; ep = np.arange(len(ev))
        for b, ls in enumerate(["-", "--", ":"]):
            axes[0].plot(ep, [e["a"][b] for e in ev], color=cols[r], ls=ls, label=f"{r}, {BANDS[b]}")
            axes[1].plot(ep, [e["tau"][b] for e in ev], color=cols[r], ls=ls, label=f"{r}, {BANDS[b]}")
    axes[0].set(xlabel="epoch (0 = init)", ylabel="velocity factor a_b on u_rea", title="Learned band velocity factors by epoch")
    axes[1].set(xlabel="epoch (0 = init)", ylabel="tau_b (days, log scale)", yscale="log", title="Learned band damping times by epoch")
    axes[0].legend(fontsize=6, ncol=2)
    for a_ in axes: a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "band_test2_params.png", dpi=110); plt.close(fig)
    # B source maps (best epoch), PD
    fz = {r: np.load(v["p"] / "final_best.npz") for r, v in R.items() if (v["p"] / "final_best.npz").exists()}
    if fz:
        lim = max(float(np.nanpercentile(np.abs(z["srcB_map"][np.isfinite(z["srcB_map"])]), 99)) for z in fz.values()) or 0.05
        fig, axes = plt.subplots(len(fz), 4, figsize=(15, 2.9 * len(fz) + .6), squeeze=False)
        for i, (r, z) in enumerate(fz.items()):
            lat, lon = z["lat"], z["lon"]
            for q in range(4):
                ax = axes[i, q]
                mm = np.where(z["ocean"] & (z["srcB_n"][q] >= 5), z["srcB_map"][q], np.nan)
                im = ax.imshow(mm, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto")
                ax.set_title(f"{r} (best ep): {SEAS[q]}", fontsize=8)
                ax.set_xlabel("longitude (deg E)", fontsize=7); ax.set_ylabel("latitude (deg N)", fontsize=7)
            fig.colorbar(im, ax=axes[i].tolist(), label="B(t+1) - B(t), ln CHL / day", fraction=.02, pad=.01)
        fig.suptitle("Season-mean lead-1 base source, val days (px with >= 5 days)")
        fig.savefig(FIG / "band_test2_source_maps.png", dpi=100); plt.close(fig)
        # partial dependence by input decile
        z0 = next(iter(fz.values())); chans = [str(c) for c in z0["src_channels"]]
        nC = len(chans); ncol = 5; nrow = int(np.ceil(nC / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow), squeeze=False)
        pd_out = {}
        for r, z in fz.items():
            if "pd_x" not in z.files:
                continue
            x, s = z["pd_x"], z["pd_s"]; pd_out[r] = {}
            for c in range(nC):
                ed = np.unique(np.quantile(x[:, c], np.linspace(0, 1, 11)))
                if len(ed) < 3:
                    continue
                b = np.clip(np.digitize(x[:, c], ed[1:-1]), 0, len(ed) - 2)
                xm = [x[b == j, c].mean() for j in range(len(ed) - 1)]; sm = [s[b == j].mean() for j in range(len(ed) - 1)]
                pd_out[r][chans[c]] = dict(x=list(map(float, xm)), s=list(map(float, sm)), range=float(np.ptp(sm)), n=int(len(s)))
                axes.flat[c].plot(xm, sm, marker=".", color=cols[r], label=r)
        for c in range(nC):
            axes.flat[c].set(title=chans[c][:34], xlabel="input (z-scored; decile means)", ylabel="mean S_B (ln CHL / day)"); axes.flat[c].grid(alpha=.3)
            axes.flat[c].axhline(0, color="0.5", lw=.6)
        for j in range(nC, nrow * ncol):
            axes.flat[j].axis("off")
        axes.flat[0].legend(fontsize=6)
        fig.suptitle("Base source S_B by input decile (lead-1 sub-steps, val, best epoch)")
        fig.tight_layout(); fig.savefig(FIG / "band_test2_source_pd.png", dpi=100); plt.close(fig)
        out["source_pd_best"] = pd_out
        # tau multiplier PD (phys)
        tz = {r: z for r, z in fz.items() if "tau_x" in z.files}
        if tz:
            tch = [str(c) for c in next(iter(tz.values()))["tau_channels"]]
            fig, axes = plt.subplots(3, len(tch), figsize=(3.3 * len(tch), 8), squeeze=False)
            tpd = {}
            for r, z in tz.items():
                x, mu = z["tau_x"], z["tau_m"]
                # drop px without valid currents (speed exactly 0 -> its z minimum), with strain 0 (a neighbour without valid currents, i.e. next to land / missing currents) or missing MLD (z exactly 0 after NaN fill)
                keep = (x[:, 1] > x[:, 1].min() + 1e-6) & (x[:, 0] > x[:, 0].min() + 1e-6) & (x[:, 2] != 0)
                tpd[r] = {"n_samples": int(len(x)), "n_dropped_invalid_current_or_mld": int((~keep).sum())}
                x, mu = x[keep], mu[keep]
                for c in range(len(tch)):
                    ed = np.unique(np.quantile(x[:, c], np.linspace(0, 1, 11)))
                    b = np.clip(np.digitize(x[:, c], ed[1:-1]), 0, len(ed) - 2)
                    xm = [x[b == j, c].mean() for j in range(len(ed) - 1)]
                    for bb in range(3):
                        mm = [mu[b == j, bb].mean() for j in range(len(ed) - 1)]
                        axes[bb, c].plot(xm, mm, marker=".", color=cols[r], label=r)
                        tpd[r][f"{tch[c]} | {BANDS[bb]}"] = dict(x=list(map(float, xm)), mult=list(map(float, mm)), n=int(len(mu)))
            for bb in range(3):
                for c in range(len(tch)):
                    axes[bb, c].set(title=f"{BANDS[bb]}: {tch[c]}", xlabel="input (decile means)", ylabel="tau multiplier"); axes[bb, c].grid(alpha=.3)
            axes[0, 0].legend(fontsize=6)
            fig.suptitle("Physics-dependent tau multiplier by input decile (val day t, ocean px with valid currents and MLD, best epoch)")
            fig.tight_layout(); fig.savefig(FIG / "band_test2_tau_pd.png", dpi=100); plt.close(fig)
            out["tau_pd_best"] = tpd
        (RES / "band_test2.json").write_text(json.dumps(out, indent=1))
    # gradients + region bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    lg = [1, 4, 16]
    for r, rr in out["runs"].items():
        axes[0].plot(lg, [rr["best"]["grad_l1"]["model"][L_][0] for L_ in LAGS], color=cols[r], marker="o", label=f"{r} (best)")
        axes[1].plot(lg, [rr["best"]["grad_l1"]["model"][L_][1] for L_ in LAGS], color=cols[r], marker="o", label=f"{r} (best)")
    for p, c, ls in [("test1", C[1], ":"), ("step1", C[7], "--"), ("persistence", C[6], "-.")]:
        axes[0].plot(lg, [rr0["grad_l1"][p][L_][0] for L_ in LAGS], color=c, ls=ls, marker="o", label=p)
        axes[1].plot(lg, [rr0["grad_l1"][p][L_][1] for L_ in LAGS], color=c, ls=ls, marker="o", label=p)
    axes[0].plot(lg, [t1["grad_lead1"]["tr_src"][L_]["ratio"] for L_ in LAGS], color=C[4], marker="o", label="step 2 tr_src")
    axes[1].plot(lg, [t1["grad_lead1"]["tr_src"][L_]["corr"] for L_ in LAGS], color=C[4], marker="o", label="step 2 tr_src")
    axes[0].axhline(1, color="0.5", lw=.8)
    axes[0].set(xscale="log", xlabel="pair lag (px, 4.5 km)", ylabel="sum |d pred| / sum |d truth|", title="Lead-1 gradient ratio by lag")
    axes[1].set(xscale="log", xlabel="pair lag (px, 4.5 km)", ylabel="uncentred corr of differences", title="Lead-1 gradient correlation by lag")
    axes[0].legend(fontsize=7)
    for a_ in axes: a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "band_test2_gradients.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    report()
