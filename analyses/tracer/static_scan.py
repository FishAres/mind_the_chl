"""Static base + advected anomaly: landscape scan of the scalar current factor a (Test 3 of the static-base hypothesis).

    ln C^(x, t+k) = B(x, t+k) + A(X_k(x), t),  A = ln C_obs(t) - B(t),  X_k from backward RK2 trajectories of a * u_rea
    (tracer.Integrator as train_tracer: 4 sub-steps/day, vmax 10 px/day, trajectory valid if the sampled (ocean & valid
    current) mask >= 0.99 everywhere; departure observed fraction >= 0.99). Bases: chl_cnn/tracer_static.py docstring
    (doy, harm from TRAIN days only; run = causal 20-day running mean; lp = 150 km low-pass of day t; run / lp held fixed
    over the lead; doy fallback where run / lp are undefined). Base "none" = plain advection (a = 0.25 -> x0.25 baseline,
    a = 1 -> x1.0).
Val = train_tracer's val starts (2017 + 2021, every 2nd start with t..t+5 consecutive days; 351 starts). Test years not
scored. Leads 1-5.
Predictors: base x a for a in 0, 0.1, ..., 1.2; step-1 model tr_rea_geo_wind (rebuilt from model.pt); persistence; each
base alone (B(t+k)); trainer DOY climatology (8-day bins).
Common set per lead: target observed and EVERY predictor valid (fixed across a and bases).
Metrics on the common set: pixel MAE; band L1 (tracer.BandPass 20-70 km, min_frac 0.5; prediction and target both
filtered with the common-set mask, scored at band-valid px); anomaly correlation ACC = Pearson r of (pred - B_doy(t+k))
and (truth - B_doy(t+k)) pooled over px and days. Per-start sums kept for day-bootstrap CIs.
Check: on the step-1 common set (target obs & step-1 & persistence & x1.0 & x0.25 valid) none@0.25 and the step-1 model
must reproduce models/tracer/tr_rea_geo_wind/metrics.json best common MAE.

Stages: --stage bases (CPU; builds ~/mind_the_chl_gap/data/static_bases), --stage gpu (flock gpu.lock), --stage report.
"""
import sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True)
BASE_DIR = Path.home() / "mind_the_chl_gap/data/static_bases"
STEP1 = ROOT / "models/tracer/tr_rea_geo_wind"
A_GRID = sorted([round(0.1 * i, 1) for i in range(13)] + [0.25])
BASES = ["none", "doy", "harm", "run", "lp"]
K = 5
OUT = RES / "static_scan.json"
NPZ = RES / "static_scan_days.npz"


def load_data(dev):
    from chl_cnn import train_tracer as TT
    a = TT.parse(["--run", "static_scan"])
    return TT.load(a, dev)


def stage_bases():
    import torch
    from chl_cnn import tracer_static as TS
    d = load_data(torch.device("cpu"))
    y = d["Y"].numpy(); obs = d["OBS"].numpy()
    B = TS.bases_cached(y, obs, d["times"], d["tr"], BASE_DIR, rebuild=True)
    print(B["info"])


def stage_gpu():
    import torch
    from chl_cnn import train_tracer as TT, tracer as TR, tracer_static as TS
    dev = torch.device("cuda")
    t0 = time.time()
    d = load_data(dev)
    H, W = d["ocean"].shape
    B = TS.bases_cached(None, None, None, None, BASE_DIR)
    doy = TS.doy_of(d["times"])
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    BD, BH = T(B["doy"]), T(B["harm"])
    BR, BL = T(B["run"]), T(B["lp"])
    DOYI = T(doy.astype("int64"))
    FB = {"run": T(B["run_fb"]), "lp": T(B["lp_fb"])}
    print(f"load {time.time() - t0:.0f}s GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    va_s = TT.starts(d["times"], d["va"], K)[::2]
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=10.0)
    bp = TR.BandPass(20.0, 70.0, min_frac=0.5, device=dev)
    cfg = json.load(open(STEP1 / "config.json"))
    m1 = TR.VelocityModel(cfg["terms"], n_static=d["STATIC"].shape[0], hidden=cfg["hidden"], a0=cfg["a_init"], b0=cfg["b_init"],
                          b_range=(cfg["b_min"], 1.5), cell=cfg["corr_cell"], corr_hidden=cfg["corr_hidden"]).to(dev)
    m1.load_state_dict(torch.load(STEP1 / "model.pt", map_location=dev)["model"]); m1.eval()
    OCf, oc = d["OCf"], d["OCEAN"]

    def base_at(name, days_t, days_tk):
        """B(t) and B(t+k) (N, H, W) for base name; days_* store indices (N,)."""
        if name == "none":
            z = torch.zeros((len(days_t), H, W), device=dev); return z, z
        if name == "doy":
            return BD[DOYI[days_t]], BD[DOYI[days_tk]]
        if name == "harm":
            return BH[DOYI[days_t]], BH[DOYI[days_tk]]
        X = BR if name == "run" else BL
        return X[days_t], X[days_t]                         # held fixed over the lead

    preds_names = [f"{b}@{a_}" for b in BASES for a_ in A_GRID] + ["step1", "persistence"] + [f"B_{b}" for b in BASES[1:]] + ["doy_clim8"]
    P = len(preds_names); pidx = {p: i for i, p in enumerate(preds_names)}
    ns = len(va_s)
    # per start, lead, predictor: [sum|e|, n, band sum|e|, band n, sp, so, spp, soo, spo]
    acc = torch.zeros(ns, K, P, 9, dtype=torch.float64, device=dev)
    chk_names = ["step1", "none@0.25", "none@1.0", "persistence"]
    chk = torch.zeros(K, len(chk_names), 2, dtype=torch.float64, device=dev)
    fbacc = torch.zeros(K, 2, 2, dtype=torch.float64, device=dev)           # fallback px on the common set at arrival (run, lp)
    ncommon = torch.zeros(K, dtype=torch.float64, device=dev)
    eb = 16
    with torch.no_grad():
        for b0 in range(0, ns, eb):
            bt = va_s[b0:b0 + eb]; nb = len(bt)
            days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev); fl = days.reshape(-1); t = days[:, 0]
            g = lambda X: X[fl].float()
            ue, ve = m1(d["STATIC"], d["DS"][fl], d["DC"][fl], g(d["U"]), g(d["V"]), g(d["UG"]), g(d["VG"]), g(d["TX"]), g(d["TY"]), OCf)
            Um = torch.stack([ue * d["CPDX"], ve * d["CPDY"]], 1).view(nb, K + 1, 2, H, W)
            ok = d["CUROK"][fl].float().view(nb, K + 1, H, W)
            U1 = torch.stack([g(d["U"]) * d["CPDX"], g(d["V"]) * d["CPDY"]], 1).view(nb, K + 1, 2, H, W)
            yt, ot = d["Y"][t], d["OBS"][t]
            for k in range(1, K + 1):
                tk = days[:, k]; truth = d["Y"][tk]; tobs = d["OBS"][tk]
                pr, vl = {}, {}
                pos, val = it.departure(Um, ok, k)
                p_, v_ = it.depart_sample(yt, ot, pos); pr["step1"], vl["step1"] = p_, val & v_
                pr["persistence"], vl["persistence"] = yt, ot
                bases = {b: base_at(b, t, tk) for b in BASES}
                for b in BASES[1:]:
                    pr[f"B_{b}"] = torch.nan_to_num(bases[b][1]); vl[f"B_{b}"] = torch.isfinite(bases[b][1])
                pr["doy_clim8"] = d["CLIM"][d["BIN"][tk]]; vl["doy_clim8"] = oc[None].expand_as(tobs)
                for a_ in A_GRID:
                    pos, val = it.departure(U1 * a_, ok, k)
                    for b in BASES:
                        p_, v_ = TS.predict(it, pos, yt, ot, *bases[b])
                        pr[f"{b}@{a_}"], vl[f"{b}@{a_}"] = p_, val & v_
                common = tobs & oc[None]
                for p in preds_names:
                    common &= vl[p]
                # step-1 common set check
                c1 = tobs & vl["step1"] & vl["persistence"] & vl["none@1.0"] & vl["none@0.25"]
                for q, p in enumerate(chk_names):
                    chk[k - 1, q] += torch.stack([((pr[p] - truth).abs().double() * c1).sum(), c1.double().sum()])
                ncommon[k - 1] += common.double().sum()
                for q, b in enumerate(["run", "lp"]):
                    fbacc[k - 1, q] += torch.stack([(FB[b][t] & common).double().sum(), common.double().sum()])
                cf = common.double()
                bt_t, vb = bp(truth, common)
                ref = torch.nan_to_num(BD[DOYI[tk]])
                ot_ = (truth - ref).double()
                for p in preds_names:
                    e = (pr[p] - truth)
                    bpp, _ = bp(pr[p], common)
                    pa = (pr[p] - ref).double()
                    vbf = vb.double()
                    s = torch.stack([(e.abs().double() * cf).sum((1, 2)), cf.sum((1, 2)),
                                     ((bpp - bt_t).abs().double() * vbf).sum((1, 2)), vbf.sum((1, 2)),
                                     (pa * cf).sum((1, 2)), (ot_ * cf).sum((1, 2)), (pa * pa * cf).sum((1, 2)),
                                     (ot_ * ot_ * cf).sum((1, 2)), (pa * ot_ * cf).sum((1, 2))], 1)
                    acc[b0:b0 + nb, k - 1, pidx[p]] = s
            print(f"  {b0 + nb}/{ns} {time.time() - t0:.0f}s", flush=True)
    acc = acc.cpu().numpy(); chk = chk.cpu().numpy()
    np.savez_compressed(NPZ, acc=acc, preds=np.array(preds_names), starts=va_s, a_grid=np.array(A_GRID),
                        fields=np.array(["abs_e", "n", "band_abs_e", "band_n", "sp", "so", "spp", "soo", "spo"]))
    ref1 = json.load(open(STEP1 / "metrics.json"))["best"]["common"]
    check = {p: [dict(mae=float(chk[k, q, 0] / chk[k, q, 1]), n=int(chk[k, q, 1])) for k in range(K)] for q, p in enumerate(chk_names)}
    check["metrics_json_step1_model"] = [dict(mae=ref1["model"][str(k)]["mae"], n=ref1["model"][str(k)]["n"]) for k in range(1, K + 1)]
    check["metrics_json_x0.25"] = [ref1["adv_x0.25"][str(k)]["mae"] for k in range(1, K + 1)]
    check["metrics_json_x1.0"] = [ref1["adv_x1.0"][str(k)]["mae"] for k in range(1, K + 1)]
    check["metrics_json_persistence"] = [ref1["persistence"][str(k)]["mae"] for k in range(1, K + 1)]
    fb = fbacc.cpu().numpy()
    meta = dict(n_starts=int(ns), wall_s=round(time.time() - t0), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                n_common=[int(x) for x in ncommon.cpu().numpy()], check_step1_common_set=check, bases_info=B["info"],
                fallback_frac_on_common_set={b: [float(fb[k, q, 0] / max(fb[k, q, 1], 1)) for k in range(K)] for q, b in enumerate(["run", "lp"])})
    (RES / "static_scan_gpu_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta)[:2000], flush=True)


# ============================== report ==============================
def metrics_from(acc):
    """acc (..., 9) sums -> dict of arrays: mae, band, acc, n."""
    s = acc
    with np.errstate(invalid="ignore", divide="ignore"):
        n = s[..., 1]; mae = s[..., 0] / n; band = s[..., 2] / s[..., 3]
        mp, mo = s[..., 4] / n, s[..., 5] / n
        cov = s[..., 8] / n - mp * mo; vp = s[..., 6] / n - mp ** 2; vo = s[..., 7] / n - mo ** 2
        r = cov / np.sqrt(vp * vo)
    return dict(mae=mae, band=band, acc=r, n=n, band_n=s[..., 3])


def boot_diff(acc, i, j, nb=1000, seed=0):
    """Day-bootstrap 95 % CI of MAE(i) - MAE(j) per lead. acc (ns, K, P, 9)."""
    rng = np.random.default_rng(seed); ns = acc.shape[0]
    out = []
    for k in range(acc.shape[1]):
        e_i, e_j, n = acc[:, k, i, 0], acc[:, k, j, 0], acc[:, k, i, 1]
        bs = []
        for _ in range(nb):
            q = rng.integers(0, ns, ns)
            bs.append((e_i[q].sum() - e_j[q].sum()) / n[q].sum())
        out.append(dict(diff=float((e_i.sum() - e_j.sum()) / n.sum()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
    return out


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from chl_cnn.plotting import OKABE_ITO as C
    z = np.load(NPZ); acc = z["acc"]; preds = list(z["preds"]); ag = list(z["a_grid"])
    meta = json.loads((RES / "static_scan_gpu_meta.json").read_text())
    tot = acc.sum(0)                                         # (K, P, 9)
    M = metrics_from(tot)
    pi = {p: i for i, p in enumerate(preds)}
    res = dict(assumptions=__doc__.split("Stages:")[0].strip(), meta=meta, a_grid=ag)
    tab = {}
    for b in BASES:
        idx = [pi[f"{b}@{a_}"] for a_ in ag]
        tab[b] = {m: M[m][:, idx].tolist() for m in ("mae", "band", "acc")}
        best = {m: [ag[int(np.nanargmin(M[m][k, idx]) if m != "acc" else int(np.nanargmax(M[m][k, idx])))] for k in range(K)] for m in ("mae", "band", "acc")}
        tab[b]["best_a"] = best
        tab[b]["best_mae"] = [float(np.nanmin(M["mae"][k, idx])) for k in range(K)]
    res["scan"] = tab
    res["n_common"] = M["n"][:, 0].tolist(); res["n_band"] = M["band_n"][:, 0].tolist()
    res["refs"] = {p: {m: M[m][:, pi[p]].tolist() for m in ("mae", "band", "acc")} for p in ["step1", "persistence", "B_doy", "B_harm", "B_run", "B_lp", "doy_clim8"]}
    # best base x a by lead-1 MAE; CIs vs step 1 and x0.25
    cand = [(M["mae"][:, pi[f"{b}@{a_}"]].mean(), b, a_) for b in BASES[1:] for a_ in ag]
    _, bb, ba = min(cand)
    res["best_mean_lead_mae"] = dict(base=bb, a=ba)
    comps = {}
    for b in BASES[1:]:
        k_best = [ag[int(np.nanargmin(M["mae"][k, [pi[f"{b}@{a_}"] for a_ in ag]]))] for k in range(K)]
        a1 = k_best[0]
        comps[f"{b}@{a1} - step1"] = boot_diff(acc, pi[f"{b}@{a1}"], pi["step1"])
        comps[f"{b}@{a1} - none@0.25"] = boot_diff(acc, pi[f"{b}@{a1}"], pi["none@0.25"])
    comps["none@0.25 - step1"] = boot_diff(acc, pi["none@0.25"], pi["step1"])
    res["diffs_lead_mae_day_bootstrap"] = comps
    OUT.write_text(json.dumps(res, indent=1))
    # tables
    L = ["# Static base + advected anomaly: scalar-factor scan (val 2017+2021, 351 starts)", "",
         f"Common set n (px) leads 1-5: {', '.join(f'{int(x):,}' for x in res['n_common'])}; band-valid n: {', '.join(f'{int(x):,}' for x in res['n_band'])}", "",
         "Step-1 common-set check (this script vs metrics.json), lead 1-5 MAE:", ""]
    ch = meta["check_step1_common_set"]
    L.append("| predictor | " + " | ".join(f"l{k}" for k in range(1, K + 1)) + " | n l1 |"); L.append("|---" * (K + 2) + "|")
    for p in ["step1", "none@0.25", "none@1.0", "persistence"]:
        L.append(f"| {p} (here) | " + " | ".join(f"{x['mae']:.4f}" for x in ch[p]) + f" | {ch[p][0]['n']:,} |")
    L.append("| step1 (metrics.json) | " + " | ".join(f"{x['mae']:.4f}" for x in ch["metrics_json_step1_model"]) + f" | {ch['metrics_json_step1_model'][0]['n']:,} |")
    L.append("| x0.25 (metrics.json) | " + " | ".join(f"{x:.4f}" for x in ch["metrics_json_x0.25"]) + " | |")
    L += ["", "MAE on the common set, best a per base and lead (a in 0..1.2 step 0.1, plus 0.25):", "",
          "| base | " + " | ".join(f"l{k} MAE (a)" for k in range(1, K + 1)) + " |", "|---" * (K + 1) + "|"]
    for b in BASES:
        L.append(f"| {b} | " + " | ".join(f"{tab[b]['best_mae'][k]:.4f} ({tab[b]['best_a']['mae'][k]})" for k in range(K)) + " |")
    for p in ["step1", "persistence", "B_doy", "B_harm", "B_run", "B_lp", "doy_clim8"]:
        L.append(f"| {p} | " + " | ".join(f"{M['mae'][k, pi[p]]:.4f}" for k in range(K)) + " |")
    L += ["", "MAE at fixed a (common set):", "", "| base | a | " + " | ".join(f"l{k}" for k in range(1, K + 1)) + " |", "|---" * (K + 2) + "|"]
    for b in BASES:
        for a_ in [0.0, 0.25, 0.5, 0.6, 1.0]:
            L.append(f"| {b} | {a_} | " + " | ".join(f"{M['mae'][k, pi[f'{b}@{a_}']]:.4f}" for k in range(K)) + " |")
    L += ["", "Best a by metric (band L1 20-70 km, ACC):", "", "| base | best a band l1-l5 | best a ACC l1-l5 | best a MAE l1-l5 |", "|---|---|---|---|"]
    for b in BASES:
        L.append(f"| {b} | {tab[b]['best_a']['band']} | {tab[b]['best_a']['acc']} | {tab[b]['best_a']['mae']} |")
    L += ["", "Day-bootstrap (1000) 95 % CI of MAE differences, leads 1-5:", ""]
    for nm, v in comps.items():
        L.append(f"- {nm}: " + "; ".join(f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in v))
    (RES / "static_scan.md").write_text("\n".join(L) + "\n")
    # figure: landscape
    fig, axes = plt.subplots(3, K, figsize=(4 * K, 10), sharex=True)
    for k in range(K):
        for q, (m, lab) in enumerate([("mae", "pixel MAE, ln CHL"), ("band", "20-70 km band L1, ln CHL"), ("acc", "anomaly corr. (vs DOY clim)")]):
            ax = axes[q, k]
            for c, b in enumerate(BASES):
                ax.plot(ag, M[m][k, [pi[f"{b}@{a_}"] for a_ in ag]], marker=".", color=C[c], label=f"base {b}")
            ax.axhline(M[m][k, pi["step1"]], color=C[7], ls="--", lw=1, label="step-1 model")
            ax.axhline(M[m][k, pi["persistence"]], color=C[5], ls=":", lw=1, label="persistence")
            ax.set(title=f"lead {k + 1} d: {lab.split(',')[0]}", xlabel="factor a on reanalysis current", ylabel=lab)
            ax.grid(alpha=.3)
            if m == "mae":
                lo = min(np.nanmin(M[m][k, [pi[f"{b}@{a_}"] for b in BASES for a_ in ag]]), M[m][k, pi["step1"]])
                ax.set_ylim(lo - 0.003, lo + 0.05)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Base + advected anomaly vs factor a, val common set (n {int(res['n_common'][0]):,} px at lead 1)")
    fig.tight_layout(); fig.savefig(FIG / "static_scan_landscape.png", dpi=110); plt.close(fig)
    # figure: best a vs lead
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for c, b in enumerate(BASES):
        ax[0].plot(range(1, K + 1), tab[b]["best_a"]["mae"], marker="o", color=C[c], label=f"base {b}")
        ax[1].plot(range(1, K + 1), np.array(tab[b]["best_mae"]) - M["mae"][:, pi["step1"]], marker="o", color=C[c], label=f"base {b}")
    ax[0].axhspan(0.45, 0.6, color="0.85", label="observed feature-motion slope 0.45-0.6")
    ax[0].set(xlabel="lead (days)", ylabel="MAE-optimal factor a", title="MAE-optimal a by lead and base")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(xlabel="lead (days)", ylabel="best-a MAE minus step-1 MAE (ln CHL)", title="Best-a MAE minus step-1 model, common set")
    for a_ in ax: a_.grid(alpha=.3); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "static_scan_best_a.png", dpi=110); plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", default="report", choices=["bases", "gpu", "report"])
    s = p.parse_args().stage
    {"bases": stage_bases, "gpu": stage_gpu, "report": report}[s]()
