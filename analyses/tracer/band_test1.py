"""Band-split tracer, TEST 1 (no training): fixed 150 km base + three band-passed anomalies advected and damped.

    ln C^(x, t+k) = B(x, t) + sum_b exp(-k / tau_b) A_b(X_k(x), t)        (chl_cnn/tracer_band.py docstring)
B = 'lp' base (tracer_static, cached in ~/mind_the_chl_gap/data/static_bases; held fixed over the lead). Bands < 20,
20-50, 50-150 km (normalised-convolution low-passes of day t at 20 and 50 km; 0.5-response edges). All bands share
one velocity in this test: a0 = no motion (damping only), x0.25 / x0.45 reanalysis current, ueff = step-1 u_eff
(tr_rea_geo_wind rebuilt from model.pt, wind term included). Integrator as train_tracer (RK2, 4 sub-steps / day, vmax 10
px/day, trajectory valid if the sampled ocean & current mask >= 0.99, departure observed fraction >= 0.99).
Damping configs: D1 (no damping), t{m1}-{m2}-{m3} = tau_measured x (m1, m2, m3), m in {0.5, 1, 2} (t1-1-1 = measured
tau 1.67 / 2.38 / 13.8 d), tc = caller's (1.6, 5, 10) d.
References: step1 (tr_rea_geo_wind), x0.25 (none@0.25), persistence, lp_alone (B), lp@0.25 (= x0.25|D1 up to rounding),
tr_src and tr_src_s1 at their saved best epoch (velocity + source along the trajectory, re-implemented from
train_tracer.forecast_src; checked against metrics.json own-set MAE). The step-2 runs saved no per-epoch checkpoints, so
the mean of their last 5 epochs can not be re-scored here (report uses history.json on train_tracer's own per-epoch
common set and says so).
Common set per (start, lead): the static_scan common set (target observed & every static_scan predictor valid: 14 a x 5
bases, step1, persistence, bases alone, doy_clim8; rebuilt here and checked against static_scan_gpu_meta n_common and
static_scan_days.npz sums) AND every predictor of this script valid. Saved packed to ~/mind_the_chl_gap/data/band_common/
for TEST 2.
Metrics on the common set: pixel MAE (per start: day bootstrap), 20-70 km band MAE (tracer.BandPass 20-70 km, min_frac
0.5, prediction and truth filtered with the common mask), gradient ratio / corr at lags 1 / 4 / 16 px (tracer.grad_acc),
lead-1 MAE by region. Val = 351 starts (2017 + 2021, every 2nd start); test years not scored.
Stages: --stage gpu (run under flock ~/mind_the_chl_gap/logs/gpu.lock), --stage report (CPU).
"""
import sys, json, time, argparse, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
BASE_DIR = Path.home() / "mind_the_chl_gap/data/static_bases"
COMMON_DIR = Path.home() / "mind_the_chl_gap/data/band_common"
MODELS = ROOT / "models/tracer"
K = 5
A_GRID = sorted([round(0.1 * i, 1) for i in range(13)] + [0.25])
SCAN_BASES = ["none", "doy", "harm", "run", "lp"]
VELS = ["a0", "x0.25", "x0.45", "ueff"]
MULTS = (0.5, 1, 2)
TAU_CALLER = (1.6, 5.0, 10.0)
REFS = ["step1", "x0.25", "persistence", "lp_alone", "lp@0.25", "tr_src", "tr_src_s1"]
REG = {1: "NW shelf", 2: "W open sea", 3: "E open sea", 4: "Anatolian shelf", 5: "Crimea/Kerch/Caucasus", 6: "Azov+Kerch", 7: "Marmara/Bosporus"}
LAGS = (1, 4, 16)
NPZ = RES / "band_test1_days.npz"


def tau_configs(tau):
    cf = {"D1": None, "tc": np.array(TAU_CALLER)}
    for m in itertools.product(MULTS, repeat=3):
        cf["t" + "-".join(f"{x:g}" for x in m)] = tau * np.array(m)
    return cf


def load_vel(d, run, dev):
    import torch
    from chl_cnn import tracer as TR
    cfg = json.load(open(MODELS / run / "config.json"))
    ck = torch.load(MODELS / run / "model.pt", map_location=dev)
    m = TR.VelocityModel(cfg["terms"], n_static=d["STATIC"].shape[0], hidden=cfg["hidden"], a0=cfg["a_init"], b0=cfg["b_init"],
                         b_range=(cfg["b_min"], 1.5), cell=cfg["corr_cell"], corr_hidden=cfg["corr_hidden"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    src = None
    if "source" in ck:
        from chl_cnn.train_tracer import SRC_CH
        src = TR.SourceModel(len(SRC_CH) + 4 + (0 if cfg.get("src_no_c") else 1), cfg["src_hidden"], cfg["g_max"]).to(dev)
        src.load_state_dict(ck["source"]); src.eval()
    return m, src, cfg, int(ck.get("epoch", -1))


def stage_gpu():
    import torch
    from chl_cnn import train_tracer as TT, tracer as TR, tracer_static as TS, tracer_band as TB
    dev = torch.device("cuda"); t0 = time.time()
    d = TT.load(TT.parse(["--run", "band_test1", "--terms", "rea,geo,wind", "--source"]), dev)
    H, W = d["ocean"].shape; OCf, oc = d["OCf"], d["OCEAN"]
    NF = len(TT.SRC_CH)
    Bc = TS.bases_cached(None, None, None, None, BASE_DIR)
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    BD, BH, BR, BL = T(Bc["doy"]), T(Bc["harm"]), T(Bc["run"]), T(Bc["lp"])
    DOYI = T(TS.doy_of(d["times"]).astype("int64"))
    print(f"load {time.time() - t0:.0f}s GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    va_s = TT.starts(d["times"], d["va"], K)[::2]
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=10.0)
    bp = TR.BandPass(20.0, 70.0, min_frac=0.5, device=dev)
    tau, tau_info = TB.tau_measured()
    CF = tau_configs(tau)
    m1, _, _, _ = load_vel(d, "tr_rea_geo_wind", dev)
    S2 = {r: load_vel(d, r, dev) for r in ["tr_src", "tr_src_s1"]}

    def base_at(name, t, tk):
        if name == "none":
            z = torch.zeros((len(t), H, W), device=dev); return z, z
        if name == "doy":
            return BD[DOYI[t]], BD[DOYI[tk]]
        if name == "harm":
            return BH[DOYI[t]], BH[DOYI[tk]]
        X = BR if name == "run" else BL
        return X[t], X[t]

    def vel(mdl, days):
        nb = days.shape[0]; fl = days.reshape(-1); g = lambda X: X[fl].float()
        ue, ve = mdl(d["STATIC"], d["DS"][fl], d["DC"][fl], g(d["U"]), g(d["V"]), g(d["UG"]), g(d["VG"]), g(d["TX"]), g(d["TY"]), OCf)
        return torch.stack([ue * d["CPDX"], ve * d["CPDY"]], 1).view(nb, K + 1, 2, H, W)

    def forecast_src(src, Upx, ok, days, k):
        """train_tracer.forecast_src (no grad), re-implemented: returns (prediction, valid)."""
        t = days[:, 0]; nb = days.shape[0]
        path = []
        pos, val = it.departure(Upx, ok, k, path=path)
        c, pv = it.depart_sample(d["Y"][t], d["OBS"][t], pos)
        Fd = d["FEAT"][days.reshape(-1)].float().view(nb, K + 1, NF, H, W)
        h = 1.0 / it.n_sub
        for p, j, w in reversed(path):
            smp = it.sample(torch.cat([Fd[:, j], Fd[:, j + 1], d["SSTATIC"][None].expand(nb, -1, -1, -1)], 1), p)
            dj, dj1 = days[:, j], days[:, j + 1]
            ds = ((1 - w) * d["DS"][dj] + w * d["DS"][dj1]).view(nb, 1, 1, 1).expand(nb, 1, H, W)
            dc = ((1 - w) * d["DC"][dj] + w * d["DC"][dj1]).view(nb, 1, 1, 1).expand(nb, 1, H, W)
            x = torch.cat([(1 - w) * smp[:, :NF] + w * smp[:, NF:2 * NF], smp[:, 2 * NF:], ds, dc, ((c - d["CMU"]) / d["CSD"])[:, None]], 1)
            c = c + h * src(x)
        return c, val & pv

    band_names = [f"{v}|{c}" for v in VELS for c in CF]
    PRED = REFS + band_names
    P = len(PRED); pidx = {p: i for i, p in enumerate(PRED)}
    ns = len(va_s); nreg = int(d["REG"].max()) + 1
    acc = torch.zeros(ns, K, P, 4, dtype=torch.float64, device=dev)          # sum|e|, n, band sum|e|, band n
    gacc = torch.zeros(K, P, len(LAGS), 6, dtype=torch.float64, device=dev)
    racc = torch.zeros(P, nreg, 2, dtype=torch.float64, device=dev)
    scan_chk_names = ["step1", "none@0.25", "lp@0.25", "lp@0.3"]
    scan_chk = torch.zeros(ns, K, len(scan_chk_names), 2, dtype=torch.float64, device=dev)
    own_chk = {r: torch.zeros(K, 2, dtype=torch.float64, device=dev) for r in ["step1", "tr_src", "tr_src_s1"]}
    n_scan = torch.zeros(K, dtype=torch.float64, device=dev); n_comm = torch.zeros(K, dtype=torch.float64, device=dev)
    drop = {nm: torch.zeros(K, dtype=torch.float64, device=dev) for nm in VELS + ["tr_src", "tr_src_s1"]}
    rec_max = 0.0; rec_n = 0; d1_dev = 0.0; a0_dev = 0.0; nan_on_common = 0
    mask_scan = np.zeros((ns, K, H, W), bool); mask_comm = np.zeros((ns, K, H, W), bool)
    W_k = {c: (None if v is None else torch.as_tensor(np.exp(-np.arange(1, K + 1)[:, None] / v[None]), device=dev, dtype=torch.float32)) for c, v in CF.items()}
    eb = 16
    with torch.no_grad():
        for b0 in range(0, ns, eb):
            bt = va_s[b0:b0 + eb]; nb = len(bt)
            days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev); fl = days.reshape(-1); t = days[:, 0]
            ok = d["CUROK"][fl].float().view(nb, K + 1, H, W)
            U1 = torch.stack([d["U"][fl].float() * d["CPDX"], d["V"][fl].float() * d["CPDY"]], 1).view(nb, K + 1, 2, H, W)
            Ustep1 = vel(m1, days)
            Us2 = {r: vel(S2[r][0], days) for r in S2}
            yt, ot = d["Y"][t], d["OBS"][t]
            Bt = BL[t]
            A, mA = TB.decompose(yt, ot, Bt)
            r_ = (torch.nan_to_num(Bt) + A.sum(1) - yt).abs()[mA]
            rec_max = max(rec_max, float(r_.max())); rec_n += int(mA.sum())
            for k in range(1, K + 1):
                tk = days[:, k]; truth = d["Y"][tk]; tobs = d["OBS"][tk]
                # ---- static_scan validity (rebuild of its common set)
                bases = {b: base_at(b, t, tk) for b in SCAN_BASES}
                scan_ok = tobs & oc[None]
                scan_pr = {}
                pos_a = {}
                for a_ in A_GRID:
                    pos, val = it.departure(U1 * a_, ok, k)
                    pos_a[a_] = (pos, val)
                    for b in SCAN_BASES:
                        p_, v_ = TS.predict(it, pos, yt, ot, *bases[b])
                        scan_ok &= val & v_
                        if (b == "none" and a_ == 0.25) or (b == "lp" and a_ in (0.25, 0.3)):
                            scan_pr[f"{b}@{a_}"] = p_
                pos, val = it.departure(Ustep1, ok, k)
                p1, v1 = it.depart_sample(yt, ot, pos); v1 = v1 & val
                scan_ok &= v1 & ot
                for b in SCAN_BASES[1:]:
                    scan_ok &= torch.isfinite(bases[b][1])
                scan_pr["step1"] = p1
                for q, nm in enumerate(scan_chk_names):
                    scan_chk[b0:b0 + nb, k - 1, q] += torch.stack([((scan_pr[nm] - truth).abs().double() * scan_ok).sum((1, 2)), scan_ok.double().sum((1, 2))], 1)
                own_chk["step1"][k - 1] += torch.stack([((p1 - truth).abs().double() * (v1 & tobs)).sum(), (v1 & tobs).double().sum()])
                # ---- predictors of this script
                pr, vl = {}, {}
                pr["step1"], vl["step1"] = p1, v1
                pr["x0.25"], vl["x0.25"] = scan_pr["none@0.25"], pos_a[0.25][1] & it.depart_sample(yt, ot, pos_a[0.25][0])[1]
                pr["persistence"], vl["persistence"] = yt, ot
                pr["lp_alone"], vl["lp_alone"] = torch.nan_to_num(Bt), torch.isfinite(Bt)
                p_, v_ = TS.predict(it, pos_a[0.25][0], yt, ot, Bt, Bt); pr["lp@0.25"], vl["lp@0.25"] = p_, v_ & pos_a[0.25][1]
                for r in S2:
                    p_, v_ = forecast_src(S2[r][1], Us2[r], ok, days, k)
                    pr[r], vl[r] = p_, v_
                    own_chk[r][k - 1] += torch.stack([((p_ - truth).abs().double() * (v_ & tobs)).sum(), (v_ & tobs).double().sum()])
                AS = {}
                for v in VELS:
                    if v == "a0":
                        pos, val = it.base[None].expand(nb, -1, -1, -1), torch.ones_like(ot)
                    elif v == "x0.25":
                        pos, val = pos_a[0.25]
                    elif v == "x0.45":
                        pos, val = it.departure(U1 * 0.45, ok, k)
                    else:
                        pos, val = it.departure(Ustep1, ok, k)
                    s_, vs_ = TB.band_sample(it, A, mA, pos)
                    AS[v] = (s_, vs_ & val)
                common = scan_ok.clone()
                for p in REFS:
                    common &= vl[p]
                for v in VELS:
                    common &= AS[v][1]
                n_scan[k - 1] += scan_ok.double().sum(); n_comm[k - 1] += common.double().sum()
                for v in VELS:
                    drop[v][k - 1] += (scan_ok & ~AS[v][1]).double().sum()
                for r in S2:
                    drop[r][k - 1] += (scan_ok & ~vl[r]).double().sum()
                mask_scan[b0:b0 + nb, k - 1] = scan_ok.cpu().numpy(); mask_comm[b0:b0 + nb, k - 1] = common.cpu().numpy()
                Bz = torch.nan_to_num(Bt)

                def band_pred(v, c):
                    s_ = AS[v][0]
                    if W_k[c] is None:
                        return Bz + s_.sum(1)
                    w = W_k[c][k - 1]
                    return Bz + (s_ * w.view(1, 3, 1, 1)).sum(1)
                # equivalence checks on the common set
                d1_dev = max(d1_dev, float((band_pred("x0.25", "D1") - pr["lp@0.25"]).abs()[common].max()))
                a0_dev = max(a0_dev, float((band_pred("a0", "D1") - yt).abs()[common].max()))
                cf = common.double()
                bt_t, vb = bp(truth, common); vbf = vb.double()
                regc = d["REG"][None].expand_as(common)[common]
                for p in PRED:
                    if p in pr:
                        x = pr[p]
                    else:
                        v, c = p.split("|"); x = band_pred(v, c)
                    e = x - truth
                    nan_on_common += int((~torch.isfinite(x) & common).sum())
                    bpp, _ = bp(x, common)
                    acc[b0:b0 + nb, k - 1, pidx[p]] = torch.stack([(e.abs().double() * cf).sum((1, 2)), cf.sum((1, 2)),
                                                                   ((bpp - bt_t).abs().double() * vbf).sum((1, 2)), vbf.sum((1, 2))], 1)
                    ga = TR.grad_acc(x, truth, common, LAGS)
                    for li, L in enumerate(LAGS):
                        gacc[k - 1, pidx[p], li] += ga[L]
                    if k == 1:
                        racc[pidx[p], :, 0] += torch.bincount(regc, weights=e[common].abs().double(), minlength=nreg)
                        racc[pidx[p], :, 1] += torch.bincount(regc, minlength=nreg).double()
            print(f"  {b0 + nb}/{ns} {time.time() - t0:.0f}s", flush=True)
    COMMON_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COMMON_DIR / "common_mask.npz", common=np.packbits(mask_comm.reshape(-1)), scan=np.packbits(mask_scan.reshape(-1)),
                        shape=np.array(mask_comm.shape), starts=va_s)
    np.savez_compressed(NPZ, acc=acc.cpu().numpy(), gacc=gacc.cpu().numpy(), racc=racc.cpu().numpy(), preds=np.array(PRED), starts=va_s,
                        scan_chk=scan_chk.cpu().numpy(), scan_chk_names=np.array(scan_chk_names), lags=np.array(LAGS))
    # checks against earlier outputs
    sc = np.load(RES / "static_scan_days.npz"); sp = list(sc["preds"]); smeta = json.load(open(RES / "static_scan_gpu_meta.json"))
    sck = scan_chk.cpu().numpy()
    chk = dict(n_scan_common_here=[int(x) for x in n_scan.cpu().numpy()], n_scan_common_static_scan=smeta["n_common"])
    for q, nm in enumerate(scan_chk_names):
        here = sck[:, :, q, 0].sum(0) / sck[:, :, q, 1].sum(0)
        j = sp.index(nm)
        there = sc["acc"][:, :, j, 0].sum(0) / sc["acc"][:, :, j, 1].sum(0)
        chk[f"scan_mae_{nm}"] = dict(here=here.tolist(), static_scan=there.tolist(), max_abs_diff=float(np.abs(here - there).max()),
                                     per_start_n_equal=bool(np.array_equal(sck[:, :, q, 1], sc["acc"][:, :, j, 1])))
    for r in own_chk:
        o = own_chk[r].cpu().numpy()
        mj = json.load(open(MODELS / ("tr_rea_geo_wind" if r == "step1" else r) / "metrics.json"))["best"]["own"]["model"]
        chk[f"own_mae_{r}"] = dict(here=(o[:, 0] / o[:, 1]).tolist(), metrics_json=[mj[str(k)]["mae"] for k in range(1, K + 1)],
                                   n_here=[int(x) for x in o[:, 1]], n_metrics_json=[mj[str(k)]["n"] for k in range(1, K + 1)])
    meta = dict(n_starts=int(ns), wall_s=round(time.time() - t0), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                n_common=[int(x) for x in n_comm.cpu().numpy()], dropped_from_scan_set={k_: [int(x) for x in v.cpu().numpy()] for k_, v in drop.items()},
                reconstruction_max_abs=rec_max, reconstruction_n_px=rec_n, max_abs_x025D1_minus_lp025=d1_dev, max_abs_a0D1_minus_persistence=a0_dev,
                nan_predictions_on_common=nan_on_common, tau_measured=tau.tolist(), tau_info=tau_info, tau_caller=list(TAU_CALLER),
                step2_epochs={r: S2[r][3] for r in S2}, checks=chk, regions=REG)
    (RES / "band_test1_gpu_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k_: meta[k_] for k_ in ["wall_s", "peak_gb", "n_common", "reconstruction_max_abs", "max_abs_x025D1_minus_lp025", "max_abs_a0D1_minus_persistence", "nan_predictions_on_common"]}))
    print(json.dumps(chk)[:3000], flush=True)


# ============================== report ==============================
def boot_diff(acc, i, j, nb=2000, seed=0, field=0):
    """Day-bootstrap 95 % CI of metric(i) - metric(j) per lead; field 0 = pixel MAE, 2 = band MAE. acc (ns, K, P, 4)."""
    rng = np.random.default_rng(seed); ns = acc.shape[0]
    Q = rng.integers(0, ns, (nb, ns))
    out = []
    for k in range(acc.shape[1]):
        ei, ej, n = acc[:, k, i, field], acc[:, k, j, field], acc[:, k, i, field + 1]
        bs = (ei[Q].sum(1) - ej[Q].sum(1)) / n[Q].sum(1)
        out.append(dict(diff=float((ei.sum() - ej.sum()) / n.sum()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
    return out


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from chl_cnn.plotting import OKABE_ITO as C
    z = np.load(NPZ); acc = z["acc"]; preds = list(z["preds"]); gacc = z["gacc"]; racc = z["racc"]
    meta = json.loads((RES / "band_test1_gpu_meta.json").read_text())
    pi = {p: i for i, p in enumerate(preds)}
    tot = acc.sum(0)
    mae = tot[..., 0] / tot[..., 1]; band = tot[..., 2] / tot[..., 3]
    n = tot[:, 0, 1]; nband = tot[:, 0, 3]
    ratio = gacc[..., 0] / gacc[..., 1]; corr = gacc[..., 2] / np.sqrt(gacc[..., 3] * gacc[..., 4])
    tau = np.array(meta["tau_measured"])
    cfgs = list(tau_configs(tau).keys())
    # best tau config per velocity (mean-lead pixel MAE; selected on the same val days)
    best_cfg = {v: min([c for c in cfgs if c != "D1"], key=lambda c: mae[:, pi[f"{v}|{c}"]].mean()) for v in VELS}
    res = dict(assumptions=__doc__.split("Stages:")[0].strip(), meta=meta, n_common=n.tolist(), n_band=nband.tolist(), best_cfg_mean_lead=best_cfg,
               mae={p: mae[:, pi[p]].tolist() for p in preds}, band_mae={p: band[:, pi[p]].tolist() for p in preds},
               grad_lead1={p: {str(L): dict(ratio=float(ratio[0, pi[p], li]), corr=float(corr[0, pi[p], li]), n_pairs=int(gacc[0, pi[p], li, 5])) for li, L in enumerate(LAGS)} for p in preds},
               region_lead1={p: {REG[r]: dict(mae=float(racc[pi[p], r, 0] / max(racc[pi[p], r, 1], 1)), n=int(racc[pi[p], r, 1])) for r in range(1, 8)} for p in preds})
    key = [("x0.25|t1-1-1", "step1"), ("x0.25|t1-1-1", "x0.25"), ("x0.25|t1-1-1", "lp@0.25"), ("x0.25|t1-1-1", "tr_src"), ("x0.25|t1-1-1", "tr_src_s1"),
           ("ueff|t1-1-1", "step1"), (f"x0.25|{best_cfg['x0.25']}", "step1"), (f"x0.25|{best_cfg['x0.25']}", "tr_src"),
           (f"x0.25|{best_cfg['x0.25']}", "x0.25|t1-1-1"), ("x0.45|t1-1-1", "x0.25|t1-1-1"), ("a0|t1-1-1", "x0.25|t1-1-1"), ("x0.25|tc", "x0.25|t1-1-1"),
           ("tr_src", "step1"), ("lp@0.25", "step1")]
    res["diffs_day_bootstrap"] = {f"{i} - {j}": dict(pixel=boot_diff(acc, pi[i], pi[j]), band=boot_diff(acc, pi[i], pi[j], field=2)) for i, j in key}
    (RES / "band_test1.json").write_text(json.dumps(res, indent=1))
    f4 = lambda x: f"{x:.4f}"
    show = ["step1", "x0.25", "lp@0.25", "persistence", "lp_alone", "tr_src", "tr_src_s1"]
    for v in VELS:
        show += [f"{v}|D1", f"{v}|t1-1-1", f"{v}|tc", f"{v}|{best_cfg[v]}"]
    show = list(dict.fromkeys(show))
    lab = {"step1": "step 1 (tr_rea_geo_wind)", "x0.25": "currents x0.25", "lp@0.25": "lp base + anomaly x0.25 (no bands)", "persistence": "persistence",
           "lp_alone": "lp base alone", "tr_src": f"step 2 tr_src (best ep {meta['step2_epochs']['tr_src']})", "tr_src_s1": f"step 2 tr_src_s1 (best ep {meta['step2_epochs']['tr_src_s1']})"}
    L = ["# Band-split tracer, test 1 (no training): val 2017 + 2021, 351 starts", "",
         f"tau measured (d): < 20 km {tau[0]:.2f}, 20-50 km {tau[1]:.2f}, 50-150 km {tau[2]:.2f}; caller set tc = {TAU_CALLER}; grid t{{m1}}-{{m2}}-{{m3}} = measured x m.",
         f"Common set n px, leads 1-5: {', '.join(f'{int(x):,}' for x in n)} (static_scan set {', '.join(f'{x:,}' for x in meta['checks']['n_scan_common_here'])}); "
         f"20-70 km band-valid n: {', '.join(f'{int(x):,}' for x in nband)}", "",
         "| predictor | MAE l1 | l2 | l3 | l4 | l5 | band 20-70 l1 | l3 | l5 | grad ratio / corr l1 lag 1 | lag 4 | lag 16 |", "|---" * 12 + "|"]
    for p in show:
        nm = lab.get(p, p)
        g = " | ".join(f"{ratio[0, pi[p], li]:.3f} / {corr[0, pi[p], li]:.3f}" for li in range(3))
        L.append(f"| {nm} | " + " | ".join(f4(mae[k, pi[p]]) for k in range(K)) + " | " + " | ".join(f4(band[k, pi[p]]) for k in (0, 2, 4)) + f" | {g} |")
    L += ["", f"Best damping config per velocity (min mean-lead pixel MAE over {len(cfgs) - 1} configs, selected on these val days): {best_cfg}", "",
          "Day-bootstrap (2000) 95 % CI of differences (pixel MAE; band 20-70 km MAE), leads 1-5:", ""]
    for nm, v in res["diffs_day_bootstrap"].items():
        L.append(f"- {nm}: pixel " + "; ".join(f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in v["pixel"]))
        L.append(f"  band " + "; ".join(f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]" for x in v["band"]))
    L += ["", "Tau grid, x0.25 velocity, MAE lead 1 / lead 5 (rows: < 20 km multiplier; cols: 20-50 x 50-150 multipliers):", "",
          "| m_lt20 | " + " | ".join(f"{m2:g},{m3:g}" for m2 in MULTS for m3 in MULTS) + " |", "|---" * 10 + "|"]
    for m1 in MULTS:
        L.append(f"| {m1:g} | " + " | ".join(f"{mae[0, pi[f'x0.25|t{m1:g}-{m2:g}-{m3:g}']]:.4f} / {mae[4, pi[f'x0.25|t{m1:g}-{m2:g}-{m3:g}']]:.4f}" for m2 in MULTS for m3 in MULTS) + " |")
    L += ["", "Lead-1 MAE by region (common set):", "", "| region | " + " | ".join(lab.get(p, p) for p in ["step1", "x0.25", "x0.25|t1-1-1", f"x0.25|{best_cfg['x0.25']}", "tr_src"]) + " | n |", "|---" * 7 + "|"]
    for r in range(1, 8):
        row = [res["region_lead1"][p][REG[r]] for p in ["step1", "x0.25", "x0.25|t1-1-1", f"x0.25|{best_cfg['x0.25']}", "tr_src"]]
        L.append(f"| {REG[r]} | " + " | ".join(f4(x["mae"]) if x["n"] else "-" for x in row) + f" | {row[0]['n']:,} |")
    ch = meta["checks"]
    L += ["", "Checks:", f"- static_scan common set rebuilt: n {ch['n_scan_common_here']} vs {ch['n_scan_common_static_scan']}"]
    for nm in ["step1", "none@0.25", "lp@0.25", "lp@0.3"]:
        L.append(f"- static_scan MAE {nm}: max |here - static_scan| {ch[f'scan_mae_{nm}']['max_abs_diff']:.2e}, per-start n equal {ch[f'scan_mae_{nm}']['per_start_n_equal']}")
    for r in ["step1", "tr_src", "tr_src_s1"]:
        o = ch[f"own_mae_{r}"]
        L.append(f"- {r} own-set MAE here vs metrics.json best: max |diff| {max(abs(a_ - b_) for a_, b_ in zip(o['here'], o['metrics_json'])):.2e}, n equal {o['n_here'] == o['n_metrics_json']}")
    L += [f"- reconstruction B + sum A - ln C: max |.| {meta['reconstruction_max_abs']:.2e} over {meta['reconstruction_n_px']:,} px",
          f"- x0.25|D1 vs lp@0.25 max |diff| on common {meta['max_abs_x025D1_minus_lp025']:.2e}; a0|D1 vs persistence {meta['max_abs_a0D1_minus_persistence']:.2e}",
          f"- NaN predictions on the common set: {meta['nan_predictions_on_common']}",
          f"- px of the static_scan set dropped (invalid predictor), leads 1-5: {meta['dropped_from_scan_set']}",
          "- step-2 last-5-epoch mean: no per-epoch checkpoints saved; not re-scored here."]
    (RES / "band_test1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    # ---- figures
    leads = np.arange(1, K + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ref = mae[:, pi["step1"]]
    rows = [("x0.25", "currents x0.25", C[3], ":"), ("persistence", "persistence", C[7], "--"), ("lp_alone", "lp base alone", C[6], "-."),
            ("tr_src", "step 2 tr_src (best ep)", C[4], "-"), ("tr_src_s1", "step 2 tr_src_s1 (best ep)", C[4], "--"),
            ("x0.25|D1", "bands x0.25, no damping", C[1], ":"), ("x0.25|t1-1-1", "bands x0.25, measured tau", C[1], "-"),
            (f"x0.25|{best_cfg['x0.25']}", f"bands x0.25, best grid tau ({best_cfg['x0.25']})", C[1], "--"),
            ("ueff|t1-1-1", "bands step-1 u_eff, measured tau", C[2], "-"), ("a0|t1-1-1", "bands no motion, measured tau", C[0], "-")]
    for p, lb, c, ls in rows:
        ax[0].plot(leads, mae[:, pi[p]] - ref, color=c, ls=ls, marker="o", ms=3, label=lb)
        ax[1].plot(leads, band[:, pi[p]] - band[:, pi["step1"]], color=c, ls=ls, marker="o", ms=3, label=lb)
    ax[0].axhline(0, color="0.5", lw=.8); ax[1].axhline(0, color="0.5", lw=.8)
    ax[0].set(xlabel="lead (days)", ylabel="pixel MAE minus step-1 MAE (ln CHL)", title=f"Pixel MAE relative to step 1, common set (n {int(n[0]):,} px at lead 1)")
    ax[1].set(xlabel="lead (days)", ylabel="20-70 km band MAE minus step 1 (ln CHL)", title="20-70 km band MAE relative to step 1, common set")
    ax[0].set_ylim(-0.025, 0.01); ax[1].set_ylim(-0.018, 0.006)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=7)
    for v, c in zip(VELS, [C[0], C[3], C[5], C[2]]):
        g = [mae[:, pi[f"{v}|{cf}"]].mean() for cf in cfgs]
        ax[2].plot(range(len(cfgs)), np.array(g) - ref.mean(), color=c, marker=".", label=f"velocity {v}")
    ax[2].axhline(0, color="0.5", lw=.8)
    ax[2].set_xticks(range(len(cfgs))); ax[2].set_xticklabels(cfgs, rotation=90, fontsize=6)
    ax[2].set(xlabel="damping config (D1 none; t m1-m2-m3 = measured tau x m; tc 1.6/5/10 d)", ylabel="mean-lead MAE minus step 1 (ln CHL)",
              title="Mean-lead MAE by damping config and velocity")
    ax[2].set_ylim(-0.01, 0.006); ax[2].legend(fontsize=7)
    for a_ in ax: a_.grid(alpha=.3)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(FIG / "band_test1_mae.png", dpi=110); plt.close(fig)
    # gradient ratio / corr at lead 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for p, lb, c, ls in rows + [("step1", "step 1", C[7], "-")]:
        ax[0].plot(LAGS, ratio[0, pi[p]], color=c, ls=ls, marker="o", ms=3, label=lb)
        ax[1].plot(LAGS, corr[0, pi[p]], color=c, ls=ls, marker="o", ms=3, label=lb)
    ax[0].axhline(1, color="0.5", lw=.8)
    ax[0].set(xscale="log", xlabel="pair lag (px, 4.5 km)", ylabel="sum |d pred| / sum |d truth|", title="Lead-1 gradient ratio by lag, common set")
    ax[1].set(xscale="log", xlabel="pair lag (px, 4.5 km)", ylabel="uncentred corr of differences", title="Lead-1 gradient correlation by lag, common set")
    ax[0].legend(fontsize=7)
    for a_ in ax: a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "band_test1_gradients.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", default="report", choices=["gpu", "report"])
    s = p.parse_args().stage
    {"gpu": stage_gpu, "report": report}[s]()
