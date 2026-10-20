"""Tracer step 1b: decomposed velocity (train_tracer.py --vel-param decomp) trained with a band-passed loss ("A",
--loss-band 24,150) or the pixel L1 loss ("B"). Val years only (2017, 2021); test years not scored.

Stage gpu (python -m analyses.tracer.step1b_report --stage gpu; flock gpu.lock): -> results/step1b_eval.npz, .json
  1 Patch-mean velocities (px/day, east / north) of every learned velocity on every store day, on the 91 patches of
    analyses/blk4/results/spectral_motion_patches.npz: Hann-weighted mean over finite px, valid if >= 50 % of the
    1024 px are finite (as spectral_motion.ueff_and_refs); u_eff = NaN where the reanalysis current is invalid. The
    step-1 u_eff (tr_rea_geo_wind) is rebuilt the same way and compared with the cache (vel_eu / vel_ev).
  2 Val forecasts (trainer's val starts, every 2nd, leads 1-5): aligned persistence (backward-RK2 departure sampling,
    as train_tracer) along A, A seed 1, B, 0.25 x and 1.0 x reanalysis current; plain persistence. Common set per lead:
    target observed and every predictor valid. Pixel MAE on the common set. Band-limited (20-70 km, tracer.BandPass,
    min_frac 0.5) MAE: every predictor and the target filtered with the common-set mask, scored at band-valid px.
    Lead-1 gradient calibration in the band: lag-1 px pairs along both axes, both ends band-valid; per start day sums
    for OLS of dtruth on dpred (with intercept and through the origin).
  3 Train data behind each alpha cell (component x region x season): train start days per season; lead-1 loss px per
    (region, season) for A (band-valid) and B (pixel-valid) at their best models.
Stage report (default, CPU): patch-shift slopes (lead 1 only; accepted masked-NCC pairs, val-year start days),
  tables results/step1b_tables.md, figures figures/step1b_*.png.
"""
import sys, json, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True)
MODELS = ROOT / "models/tracer"
RUNS = {"A": "tr1b_band_s0", "A_s1": "tr1b_band_s1", "B": "tr1b_pix_s0"}
STEP1 = "tr_rea_geo_wind"
CACHE = ROOT / "analyses/blk4/results/spectral_motion_patches.npz"
EVAL = RES / "step1b_eval.npz"
BAND = (20.0, 70.0)
P = 32
SEAS = ["DJF", "MAM", "JJA", "SON"]
REGN = {1: "NW shelf", 2: "W open", 3: "E open", 4: "Anatolian", 5: "Crimea/Caucasus", 6: "Azov", 7: "Marmara"}
COMPS = ["geo_mean", "geo_anom", "ageo_mean", "ageo_anom"]
MIN_N = 5


# ============================== GPU stage ==============================
def gpu_stage(debug=False):
    import torch
    from chl_cnn import train_tracer as TT, tracer as TR
    dev = torch.device("cuda")
    t0 = time.time()
    a = TT.parse(["--run", "step1b_eval", "--vel-param", "decomp", "--terms="])
    d = TT.load(a, dev)
    K = 5; H, W = d["ocean"].shape
    times = d["times"]; nd = len(times)
    SEASON = torch.as_tensor((((np.asarray(times).astype("datetime64[M]").astype(int) % 12) + 1) % 12) // 3, device=dev)
    OCf = d["OCf"]
    # models
    models = {}
    for lab, run in RUNS.items():
        c = json.load(open(MODELS / run / "config.json"))
        m = TR.DecompVelocity(4, int(d["REG"].max()) + 1, c["alpha_time"], c["alpha_init"], wind="wind" in c["terms"]).to(dev)
        m.load_state_dict(torch.load(MODELS / run / "model.pt", map_location=dev)["model"]); m.eval()
        assert c["clim_sigma"] == a.clim_sigma
        models[lab] = m
    import copy
    for lab, run in RUNS.items():                                # last-epoch alphas (history.json; no wind term in these runs)
        m = copy.deepcopy(models[lab])
        with torch.no_grad():
            m.alpha.copy_(torch.as_tensor(json.load(open(MODELS / run / "history.json"))["alpha"][-1], device=dev))
        models[f"{lab}_ep20"] = m
    c1 = json.load(open(MODELS / STEP1 / "config.json"))
    m1 = TR.VelocityModel(c1["terms"], n_static=d["STATIC"].shape[0], hidden=c1["hidden"], a0=c1["a_init"], b0=c1["b_init"],
                          b_range=(c1["b_min"], 1.5), cell=c1["corr_cell"], corr_hidden=c1["corr_hidden"]).to(dev)
    m1.load_state_dict(torch.load(MODELS / STEP1 / "model.pt", map_location=dev)["model"]); m1.eval()

    def vel_ms(lab, fl):
        g = lambda X: X[fl].float()
        if lab == "step1":
            return m1(d["STATIC"], d["DS"][fl], d["DC"][fl], g(d["U"]), g(d["V"]), g(d["UG"]), g(d["VG"]), g(d["TX"]), g(d["TY"]), OCf)
        if lab in ("x0.25", "x1.0"):
            f = 0.25 if lab == "x0.25" else 1.0
            return g(d["U"]) * f, g(d["V"]) * f
        comps = TR.decomp_components(g(d["U"]), g(d["V"]), g(d["UG"]), g(d["VG"]), g(d["CUROK"]), d["CLIMD"][d["DOYI"][fl]])
        return models[lab](comps, d["REG"], SEASON[fl], d["DS"][fl], d["DC"][fl], g(d["TX"]), g(d["TY"]), OCf)

    out = {}
    # ---- 1 patch-mean velocities
    z = np.load(CACHE)
    pos = z["pos"]; npch = len(pos)
    crop = d["info"]["crop"]                                     # full-grid rows / cols of the crop
    win = torch.as_tensor(np.outer(np.hanning(P + 2)[1:-1], np.hanning(P + 2)[1:-1]), device=dev, dtype=torch.float32)
    Hf = 197; Wf = 275
    for lab in ["step1"] + list(models):
        pu = np.full((nd, npch), np.nan, "float32"); pv = np.full_like(pu, np.nan)
        for b0 in range(0, 128 if debug else nd, 64):
            fl = torch.arange(b0, min(b0 + 64, nd), device=dev)
            with torch.no_grad():
                ue, ve = vel_ms(lab, fl)
            ok = d["CUROK"][fl]
            Ex = torch.where(ok, ue * d["CPDX"], torch.nan); Ey = torch.where(ok, ve * d["CPDY"], torch.nan)
            full = torch.full((2, len(fl), Hf, Wf), torch.nan, device=dev)
            full[0, :, crop[0]:crop[1], crop[2]:crop[3]] = Ex; full[1, :, crop[0]:crop[1], crop[2]:crop[3]] = Ey
            for q, (r, c) in enumerate(pos):
                pt = full[:, :, r:r + P, c:c + P]
                fin = torch.isfinite(pt[0]) & torch.isfinite(pt[1])
                s = (torch.where(fin[None], pt, 0.0) * win).sum((-1, -2)); w = (fin * win).sum((-1, -2))
                good = fin.float().mean((-1, -2)) >= 0.5
                val = torch.where(good[None], s / w.clamp(min=1e-9)[None], torch.nan).cpu().numpy()
                pu[b0:b0 + len(fl), q] = val[0]; pv[b0:b0 + len(fl), q] = val[1]
        out[f"pv_{lab}_u"], out[f"pv_{lab}_v"] = pu, pv
        print(f"patch velocities {lab} {time.time() - t0:.0f}s", flush=True)
    both = np.isfinite(out["pv_step1_u"]) & np.isfinite(z["vel_eu"])
    chk = dict(step1_vs_cache_max_abs_px_day=float(max(np.abs(out["pv_step1_u"] - z["vel_eu"])[both].max(), np.abs(out["pv_step1_v"] - z["vel_ev"])[both].max())),
               n_both=int(both.sum()), nan_mismatch=int((np.isfinite(out["pv_step1_u"]) != np.isfinite(z["vel_eu"])).sum()))
    print(chk, flush=True)

    # ---- 2 val forecasts
    it = TR.Integrator(H, W, dev, n_sub=a.n_sub, vmax=a.vmax)
    bpf = TR.BandPass(*BAND, min_frac=0.5, device=dev)
    va_s = TT.starts(times, d["va"], K)[::a.val_stride]
    if debug:
        va_s = va_s[:4]
    PRED = list(models) + ["x0.25", "x1.0", "persistence"]
    acc = {s: np.zeros((len(PRED), K, 3)) for s in ("pix", "band")}            # sum|e|, sum e^2, n
    gday = np.zeros((len(va_s), len(PRED), 6))                                 # lead-1 band lag-1 pairs: sx sy sxy sxx syy n
    nvalid_own = np.zeros((len(PRED), K))
    for b0 in range(0, len(va_s), 16):
        bt = va_s[b0:b0 + 16]; B_ = len(bt)
        days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev); fl = days.reshape(-1); t = days[:, 0]
        okf = d["CUROK"][fl].float().view(B_, K + 1, H, W)
        Us = {}
        with torch.no_grad():
            for lab in PRED[:-1]:
                ue, ve = vel_ms(lab, fl)
                Us[lab] = torch.stack([ue * d["CPDX"], ve * d["CPDY"]], 1).view(B_, K + 1, 2, H, W)
            for k in range(1, K + 1):
                truth = d["Y"][days[:, k]]; tobs = d["OBS"][days[:, k]]
                pr, va = {}, {}
                for lab in PRED[:-1]:
                    pos_, v_ = it.departure(Us[lab], okf, k)
                    pr[lab], pv_ = it.depart_sample(d["Y"][t], d["OBS"][t], pos_); va[lab] = v_ & pv_
                pr["persistence"], va["persistence"] = d["Y"][t], d["OBS"][t]
                common = tobs.clone()
                for p in PRED:
                    common &= va[p]; nvalid_own[PRED.index(p), k - 1] += float((va[p] & tobs).sum())
                bt_, bv = bpf(truth, common)
                for i, p in enumerate(PRED):
                    e = (pr[p] - truth).double()
                    cf = common.double()
                    acc["pix"][i, k - 1] += [float((e.abs() * cf).sum()), float((e ** 2 * cf).sum()), float(cf.sum())]
                    bp_, _ = bpf(pr[p], common)
                    eb = (bp_ - bt_).double(); bvf = bv.double()
                    acc["band"][i, k - 1] += [float((eb.abs() * bvf).sum()), float((eb ** 2 * bvf).sum()), float(bvf.sum())]
                    if k == 1:
                        for ax in (-1, -2):
                            n_ = bp_.shape[ax]
                            dp = (torch.narrow(bp_, ax, 1, n_ - 1) - torch.narrow(bp_, ax, 0, n_ - 1)).double()
                            dt = (torch.narrow(bt_, ax, 1, n_ - 1) - torch.narrow(bt_, ax, 0, n_ - 1)).double()
                            mm = (torch.narrow(bv, ax, 1, n_ - 1) & torch.narrow(bv, ax, 0, n_ - 1)).double()
                            gday[b0:b0 + B_, i] += torch.stack([(dp * mm).sum((1, 2)), (dt * mm).sum((1, 2)), (dp * dt * mm).sum((1, 2)),
                                                                (dp * dp * mm).sum((1, 2)), (dt * dt * mm).sum((1, 2)), mm.sum((1, 2))], 1).cpu().numpy()
        print(f"val {b0 + B_}/{len(va_s)} {time.time() - t0:.0f}s", flush=True)

    # ---- 3 train data behind alpha cells (lead 1, best models of A and B)
    tr_s = TT.starts(times, d["tr"], K)
    if debug:
        tr_s = tr_s[:20]
    seas_tr = SEASON[torch.as_tensor(tr_s, device=dev)].cpu().numpy()
    nreg = int(d["REG"].max()) + 1
    cnt = {lab: np.zeros((nreg, 4)) for lab in ("A", "B", "A_ep20", "B_ep20")}
    bp24 = TR.BandPass(24, 150, min_frac=json.load(open(MODELS / RUNS["A"] / "config.json"))["band_min_frac"], device=dev)
    for b0 in range(0, len(tr_s), 16):
        bt = tr_s[b0:b0 + 16]; B_ = len(bt)
        days = torch.as_tensor(bt[:, None] + np.arange(2)[None], device=dev); fl = days.reshape(-1); t = days[:, 0]
        okf = d["CUROK"][fl].float().view(B_, 2, H, W)
        with torch.no_grad():
            for lab in cnt:
                ue, ve = vel_ms(lab, fl)
                U_ = torch.stack([ue * d["CPDX"], ve * d["CPDY"]], 1).view(B_, 2, 2, H, W)
                pos_, v_ = it.departure(U_, okf, 1)
                prd, pv_ = it.depart_sample(d["Y"][t], d["OBS"][t], pos_)
                m = v_ & pv_ & d["OBS"][days[:, 1]]
                if lab.startswith("A"):
                    _, m = bp24(prd, m)
                for q in range(4):
                    sel = torch.as_tensor(seas_tr[b0:b0 + B_] == q, device=dev)
                    if sel.any():
                        r = d["REG"][None].expand_as(m)[m & sel[:, None, None]]
                        cnt[lab][:, q] += torch.bincount(r, minlength=nreg).cpu().numpy()
    out.update(acc_pix=acc["pix"], acc_band=acc["band"], gday=gday, nvalid_own=nvalid_own, va_days=np.asarray(times)[va_s].astype("datetime64[D]").astype(str),
               **{f"cnt_{k_}": v for k_, v in cnt.items()}, train_days_per_season=np.bincount(seas_tr, minlength=4),
               reg_px=np.bincount(d["REG"][d["OCEAN"]].cpu().numpy(), minlength=nreg))
    np.savez_compressed(EVAL, **out)
    meta = dict(check=chk, pred=PRED, band_km=BAND, n_val_starts=int(len(va_s)), n_train_starts=int(len(tr_s)), seconds=round(time.time() - t0),
                runs=RUNS, step1=STEP1, decomp_info=d["info"].get("decomp"))
    (RES / "step1b_eval.json").write_text(json.dumps(meta, indent=1))
    print("done", meta, flush=True)


# ============================== report stage (CPU) ==============================
FLAG_PX = 10000          # alpha cells with fewer lead-1 train loss px are flagged


def ols_sums(S):
    """S (..., 6) = sx sy sxy sxx syy n of x = dpred, y = dtruth -> slope (intercept), slope (origin), r."""
    sx, sy, sxy, sxx, syy, n = np.moveaxis(S, -1, 0)
    cxy = sxy - sx * sy / n; cxx = sxx - sx * sx / n; cyy = syy - sy * sy / n
    return cxy / cxx, sxy / sxx, cxy / np.sqrt(cxx * cyy)


def slopes():
    from analyses.blk4 import spectral_motion as SM
    z = np.load(CACHE); e = np.load(EVAL)
    times = np.load(SM.STORE / "time.npy").astype("datetime64[D]")
    m = (z["k"] == 1) & (z["null"] == 0)
    acc = z["okpk"][m] & ~z["edge"][m] & (z["rpk"][m] >= SM.R_MIN)
    i, p = z["i"][m], z["p"][m]
    yr = times[i].astype("datetime64[Y]").astype(int) + 1970
    val = np.isin(yr, [2017, 2021])
    sel = acc & val
    pc = np.searchsorted(z["pidx"], p)
    do = np.stack([z["fx"][m], z["fy"][m]])
    V = {"reanalysis": (z["vel_cu"], z["vel_cv"]), "reanalysis x0.25": (0.25 * z["vel_cu"], 0.25 * z["vel_cv"]),
         "step-1 u_eff (tr_rea_geo_wind)": (e["pv_step1_u"], e["pv_step1_v"])}
    for lab in json.load(open(RES / "step1b_eval.json"))["pred"][:-3]:
        V[lab] = (e[f"pv_{lab}_u"], e[f"pv_{lab}_v"])
    D = {k_: np.stack([SM.lag_disp(u, i, pc, 1), SM.lag_disp(v, i, pc, 1)]) for k_, (u, v) in V.items()}
    common = sel.copy()
    for x in D.values():
        common &= np.isfinite(x).all(0)
    rows = {}
    for k_, dr in D.items():
        r = SM.compare(common, do, dr, i, 1, nb=300)
        r["median_abs_dref"] = float(np.median(np.hypot(*dr[:, common])))
        rows[k_] = r
    info = dict(n_candidates_lead1=int(m.sum()), n_accepted=int(acc.sum()), n_accepted_val=int(sel.sum()), n_common=int(common.sum()),
                n_days_common=int(len(np.unique(i[common]))), median_abs_dobs=float(np.median(np.hypot(*do[:, common]))))
    return rows, info, (do[:, common], {k_: x[:, common] for k_, x in D.items()})


def boot_grad(gd, nb=300, seed=0):
    rng = np.random.default_rng(seed)
    nd = gd.shape[0]
    bs = np.stack([ols_sums(gd[rng.integers(0, nd, nd)].sum(0))[0] for _ in range(nb)])
    return np.percentile(bs, [2.5, 97.5], axis=0)


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from chl_cnn.plotting import OKABE_ITO as C
    e = np.load(EVAL); meta = json.load(open(RES / "step1b_eval.json"))
    PRED = meta["pred"]; K = 5
    H = {lab: json.load(open(MODELS / run / "history.json")) for lab, run in RUNS.items()}
    Mx = {lab: json.load(open(MODELS / run / "metrics.json")) for lab, run in RUNS.items()}
    AL = {}
    for lab, run in RUNS.items():
        AL[lab] = np.array(json.load(open(MODELS / run / "alpha.json"))["alpha"])
        AL[f"{lab}_ep20"] = np.array(H[lab]["alpha"][-1])
    ALS = ["A", "A_ep20", "A_s1_ep20", "B", "B_ep20"]
    cntof = lambda lab: e["cnt_B_ep20" if lab == "B_ep20" else "cnt_B" if lab == "B" else "cnt_A_ep20" if lab.endswith("ep20") else "cnt_A"]
    rows, sinfo, (do, D) = slopes()
    L = ["# Tracer step 1b tables (val years 2017 + 2021; generated by analyses/tracer/step1b_report.py)", ""]
    L += ["Runs: " + ", ".join(f"{k} = {v} (best epoch {Mx[k]['best_epoch']}, {Mx[k]['wall_s'] / 60:.1f} min)" for k, v in RUNS.items()) + f"; step-1 reference {STEP1}.", ""]
    # ---- slopes
    L += ["## Observed lead-1 patch shift vs velocity (masked NCC, spectral_motion cache)", "",
          f"Accepted lead-1 pairs {sinfo['n_accepted']:,} (all years), {sinfo['n_accepted_val']:,} with a val-year start day; common set (every reference finite) {sinfo['n_common']:,} pairs on {sinfo['n_days_common']} days. "
          f"Median |d_obs| {sinfo['median_abs_dobs']:.2f} px/day. OLS d_obs on d_ref with intercept, 95 % CI from 300 day-bootstrap resamples. R = mean resultant length of the direction difference over pairs with |d_obs| and |d_ref| >= 0.5 px/day (n_R).", "",
          "| reference | slope east [CI] | slope north [CI] | R | mean angle (deg) | median abs d_ref (px/day) | n_R | n pairs |", "|---|---|---|---|---|---|---|---|"]
    f3 = lambda r: f"{r['slope']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]" if "slope" in r else f"(n {r['n']} < {MIN_N})"
    for k_, r in rows.items():
        an = r["angle"]
        L.append(f"| {k_} | {f3(r['east'])} | {f3(r['north'])} | {an.get('R', float('nan')):.2f} | {an.get('mean_deg', float('nan')):+.0f} | {r['median_abs_dref']:.2f} | {an['n']:,} | {r['east']['n']:,} |")
    # ---- MAE tables
    pn = {p: p for p in PRED}; pn.update({"x0.25": "rea x0.25", "x1.0": "rea x1.0"})
    ix = PRED.index
    for nm, key in (("Pixel MAE, common set (target observed, every predictor valid)", "acc_pix"), (f"Band-limited MAE ({BAND[0]:.0f}-{BAND[1]:.0f} km DoG, filter mask = common set; band-valid px)", "acc_band")):
        A_ = e[key]
        L += ["", f"## {nm}", "", "| lead | " + " | ".join(pn[p] for p in PRED) + " | A - pers | B - pers | A - B | A_ep20 - B_ep20 | n |", "|---" * (len(PRED) + 6) + "|"]
        for k in range(K):
            mae = A_[:, k, 0] / A_[:, k, 2]
            L.append(f"| {k + 1} | " + " | ".join(f"{x:.4f}" for x in mae) + f" | {mae[ix('A')] - mae[-1]:+.4f} | {mae[ix('B')] - mae[-1]:+.4f} | {mae[ix('A')] - mae[ix('B')]:+.4f} | {mae[ix('A_ep20')] - mae[ix('B_ep20')]:+.4f} | {int(A_[0, k, 2]):,} |")
    # ---- gradient calibration
    gs = e["gday"].sum(0); sl, so, rr = ols_sums(gs); ci = boot_grad(e["gday"])
    L += ["", f"## Gradient calibration, lead 1, {BAND[0]:.0f}-{BAND[1]:.0f} km band", "",
          "OLS of dtruth on dpred, lag-1 px pairs along both axes of the band-passed fields (both ends band-valid). CI: 300 val-day bootstrap resamples.", "",
          "| predictor | slope (intercept) [CI] | slope (origin) | r | n pairs |", "|---|---|---|---|---|"]
    for q, p in enumerate(PRED):
        L.append(f"| {pn[p]} | {sl[q]:.3f} [{ci[0, q]:.3f}, {ci[1, q]:.3f}] | {so[q]:.3f} | {rr[q]:.3f} | {int(gs[q, 5]):,} |")
    # ---- alpha tables
    tdays = e["train_days_per_season"]
    L += ["", "## Learned alpha (best model), component x region x season", "",
          f"Train start days per season (DJF / MAM / JJA / SON): {' / '.join(str(int(x)) for x in tdays)}. "
          f"n = lead-1 train loss px of the cell (A: band-valid, B: pixel-valid; best models). * = n < {FLAG_PX:,} (alpha barely constrained); region px = ocean px in the region.", ""]
    for lab in ALS:
        cnt = cntof(lab)
        L += [f"### {lab} ({RUNS[lab.replace('_ep20', '')]}, {'best epoch' if not lab.endswith('ep20') else 'epoch 20'})", "", "| component | region (px) | DJF | MAM | JJA | SON |", "|---|---|---|---|---|---|"]
        for c, cn in enumerate(COMPS):
            for rg, rn in REGN.items():
                L.append(f"| {cn} | {rn} ({int(e['reg_px'][rg])}) | " + " | ".join(f"{AL[lab][c, rg, q]:+.3f}{'*' if cnt[rg, q] < FLAG_PX else ''}" for q in range(4)) + " |")
        L += ["", "n (lead-1 train loss px) | " + " ; ".join(f"{REGN[rg]}: " + "/".join(f"{int(cnt[rg, q])}" for q in range(4)) for rg in REGN), ""]
    L += ["## Checks", "", f"- Step-1 u_eff patch means rebuilt here vs spectral_motion cache: max |diff| {meta['check']['step1_vs_cache_max_abs_px_day']:.2e} px/day (n {meta['check']['n_both']:,}; NaN-pattern mismatches {meta['check']['nan_mismatch']}).",
          f"- Decomposition climatology: {json.dumps(meta['decomp_info'])}."]
    (RES / "step1b_tables.md").write_text("\n".join(L)); print("\n".join(L))

    # ================= figures
    # 1 band-pass response
    br = np.load(RES / "bandpass_response.npz")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j, b in enumerate(["24_150", "20_70"]):
        ax.semilogx(br["fine_km"], br[b], color=C[j], label=f"DoG {b.replace('_', '-')} km, analytic")
        ax.semilogx(br["wavelength_km"], br[f"{b}_meas_clear"], "o", color=C[j], mfc="none", label=f"{b.replace('_', '-')} km, measured, no mask")
        ax.semilogx(br["wavelength_km"], br[f"{b}_meas_cloud"], "x", color=C[j], label=f"{b.replace('_', '-')} km, measured, 30 % clouds + land")
    ax.set(title="Band-pass amplitude response vs wavelength (synthetic sinusoids)", xlabel="wavelength (km)", ylabel="amplitude response (output / input)")
    ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "step1b_bandpass_response.png", dpi=130); plt.close(fig)
    # 2 slopes
    labs = list(rows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(labs))
    for j, comp in enumerate(["east", "north"]):
        v = [rows[k_][comp]["slope"] for k_ in labs]; lo = [rows[k_][comp]["ci95"][0] for k_ in labs]; hi = [rows[k_][comp]["ci95"][1] for k_ in labs]
        ax[0].errorbar(x + (j - 0.5) * 0.2, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="o", color=C[j], label=comp)
    ax[0].axhline(1, color="k", lw=0.6, ls="--")
    ax[0].set(title="Lead-1 observed patch shift on velocity: OLS slope (val)", ylabel="slope d_obs on d_ref (95 % CI)", xticks=x)
    ax[0].set_xticklabels([k_.split(" (")[0] for k_ in labs], rotation=25, ha="right", fontsize=8); ax[0].legend()
    ax[1].bar(x, [rows[k_]["angle"].get("R", np.nan) for k_ in labs], color=C[2])
    for q, k_ in enumerate(labs):
        ax[1].text(q, 0.02, f"n {rows[k_]['angle']['n']:,}", ha="center", rotation=90, fontsize=7, color="w")
    ax[1].set(title="Direction agreement R (|d_obs|, |d_ref| >= 0.5 px/day)", ylabel="mean resultant length R", xticks=x, ylim=(0, 1))
    ax[1].set_xticklabels([k_.split(" (")[0] for k_ in labs], rotation=25, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step1b_patch_slopes.png", dpi=130); plt.close(fig)
    # 3 MAE vs lead (difference to persistence), pixel and band
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    kk = np.arange(1, K + 1)
    for a_, key, nm in ((ax[0], "acc_pix", "pixel MAE"), (ax[1], "acc_band", f"{BAND[0]:.0f}-{BAND[1]:.0f} km band MAE")):
        A_ = e[key]; mae = A_[:, :, 0] / A_[:, :, 2]
        fam = {"A": C[0], "A_s1": C[5], "B": C[3], "x0.25": C[2], "x1.0": C[1]}
        for q, p in enumerate(PRED[:-1]):
            a_.plot(kk, mae[q] - mae[-1], "s--" if p.endswith("ep20") else "o-", color=fam[p.replace("_ep20", "")], label=pn[p])
        a_.axhline(0, color="k", lw=0.6)
        a_.set(title=f"Val {nm} minus persistence, common set", xlabel="lead (days)", ylabel="ΔMAE vs persistence (ln CHL)", xticks=kk); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step1b_mae_vs_lead.png", dpi=130); plt.close(fig)
    # 4 gradient calibration
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(np.arange(len(PRED)), sl, yerr=[sl - ci[0], ci[1] - sl], fmt="o", color=C[0])
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set(title=f"Lead-1 gradient calibration, {BAND[0]:.0f}-{BAND[1]:.0f} km band (val)", ylabel="OLS slope Δtruth on Δpred (95 % CI)", xticks=np.arange(len(PRED)))
    ax.set_xticklabels([pn[p] for p in PRED], rotation=25, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step1b_grad_calibration.png", dpi=130); plt.close(fig)
    # 5 alpha heatmaps
    fig, ax = plt.subplots(len(ALS), 4, figsize=(15, 3.3 * len(ALS)), squeeze=False)
    vmax = max(np.abs(AL[l][:, 1:]).max() for l in ALS)
    for i, lab in enumerate(ALS):
        cnt = cntof(lab)
        for c in range(4):
            im = ax[i, c].imshow(AL[lab][c, 1:], cmap="cividis", vmin=-vmax, vmax=vmax, aspect="auto")
            for rg in range(7):
                for q in range(4):
                    ax[i, c].text(q, rg, f"{AL[lab][c, rg + 1, q]:.2f}" + ("*" if cnt[rg + 1, q] < FLAG_PX else ""), ha="center", va="center", fontsize=7,
                                  color="k" if AL[lab][c, rg + 1, q] > 0 else "w")
            ax[i, c].set(title=f"alpha on {COMPS[c]} ({lab})", xticks=range(4), yticks=range(7), xlabel="season")
            ax[i, c].set_xticklabels(SEAS); ax[i, c].set_yticklabels(list(REGN.values()), fontsize=7)
            fig.colorbar(im, ax=ax[i, c], label="alpha (dimensionless)", shrink=0.8)
    fig.suptitle(f"Learned alpha by component, region and season (* = < {FLAG_PX:,} lead-1 train loss px)")
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(FIG / "step1b_alpha.png", dpi=110); plt.close(fig)
    # 6 training curves
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for j, lab in enumerate(RUNS):
        h = H[lab]
        ax[0].plot(h["epoch"], h["val_mean_lead"], "o-", color=C[j], label=lab)
        al = np.array(h["alpha"])                                      # (ep, C, R, 4)
        for c in range(4):
            ax[2].plot(h["epoch"], al[:, c, 1:4].mean((1, 2)), ["-", "--", ":", "-."][c], color=C[j], label=f"{COMPS[c]} ({lab})")
        if "band_own" in Mx[lab].get("best", {}):
            pass
    for j, lab in enumerate(RUNS):
        if lab != "B":
            ax[1].plot(H[lab]["epoch"], H[lab]["train_l1"], "x--", color=C[j], label=f"train band L1 ({lab})")
    ax[0].set(title="Mean-lead val pixel MAE by epoch (own set)", xlabel="epoch", ylabel="MAE (ln CHL)"); ax[0].legend(fontsize=8)
    ax[1].set(title="Mean-lead train band-passed L1 by epoch (A runs)", xlabel="epoch", ylabel="band L1 (ln CHL)"); ax[1].legend(fontsize=8)
    ax[2].set(title="Mean alpha over NW shelf, W open, E open seasons", xlabel="epoch", ylabel="alpha (dimensionless)"); ax[2].legend(fontsize=6, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "step1b_training_curves.png", dpi=130); plt.close(fig)
    # 7 scatter of observed vs reference displacement (east), A vs B vs x0.25
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    keys = ["A_ep20", "B_ep20", "reanalysis x0.25"]
    for a_, k_ in zip(ax, keys):
        a_.hexbin(D[k_][0], do[0], gridsize=60, extent=(-4, 4, -6, 6), mincnt=1, bins="log", cmap="cividis")
        a_.plot([-4, 4], [-4, 4], "k--", lw=0.6)
        a_.set(title=f"Lead-1 east shift: observed vs {k_.split(' (')[0]}", xlabel="velocity displacement east (px/day)", ylabel="observed shift east (px/day)")
    fig.tight_layout(); fig.savefig(FIG / "step1b_shift_scatter.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="report", choices=["gpu", "report"])
    ap.add_argument("--debug", action="store_true", help="gpu stage on a few days only (pipeline check)")
    aa = ap.parse_args()
    if aa.stage == "gpu":
        gpu_stage(aa.debug)
    else:
        report()
