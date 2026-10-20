"""Short training run of the learned tracer velocity with a static / slow base (static-base hypothesis, Test 3b).
Minimal standalone loop reusing train_tracer.load / starts, tracer.Integrator / VelocityModel / DecompVelocity and
tracer_static.predict; train_tracer.py itself is not changed.

    ln C^(x, t+k) = B(x, t+k) + A(X_k(x), t),  A = ln C_obs(t) - B(t),  X_k from u_eff (step-1 or step-1b parameterisation)
Velocity: --vel factor = tracer.VelocityModel with terms rea, geo, wind (as step 1 / tr_rea_geo_wind: a init 0.25, b init 0,
wind 0); --vel decomp = tracer.DecompVelocity (alpha per component x region x season, init 0.25, no wind; as step 1b).
Base: --base in doy, harm, run, lp (tracer_static; run / lp held fixed over the lead). Loss: pixel L1 over target observed &
valid prediction, mean per lead, leads 1-5 equal weights (as step 1). Adam lr 3e-3 (wind 1e-2), batch 16, train starts
as train_tracer (2,449), --epochs (default 5). Best epoch = min val mean-lead MAE on the own set (as step 1).
Final evaluation of the best model on the val starts: common set = target observed & every predictor valid, predictors
model, step-1 tr_rea_geo_wind, none@0.25 (x0.25), base@a_scan (the scan's MAE-best scalar factor for this base at
lead 1), persistence.
Run: flock ~/mind_the_chl_gap/logs/gpu.lock python -m analyses.tracer.static_train --run trs_lp_factor --base lp --vel factor
"""
import sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from chl_cnn import train_tracer as TT, tracer as TR, tracer_static as TS
from chl_cnn.runs import CATEGORIES

BASE_DIR = Path.home() / "mind_the_chl_gap/data/static_bases"
STEP1 = CATEGORIES["tracer"] / "tr_rea_geo_wind"
K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--base", default="lp", choices=TS.BASES)
    ap.add_argument("--vel", default="factor", choices=["factor", "decomp"]); ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--lr-wind", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--scan-a", type=float, default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda"); t0 = time.time()
    out = CATEGORIES["tracer"] / a.run; out.mkdir(parents=True, exist_ok=True)
    ta = TT.parse(["--run", a.run, "--terms", "rea,geo,wind" if a.vel == "factor" else "", "--vel-param", a.vel])
    d = TT.load(ta, dev)
    H, W = d["ocean"].shape; OCf, oc = d["OCf"], d["OCEAN"]
    B = TS.bases_cached(None, None, None, None, BASE_DIR)
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    DOYI = T(TS.doy_of(d["times"]).astype("int64"))
    BX = T(B[a.base])
    SEASON = T((((np.asarray(d["times"]).astype("datetime64[M]").astype(int) % 12) + 1) % 12) // 3)

    def base_at(name, X, t, tk):
        if name == "none":
            z = torch.zeros((len(t), H, W), device=dev); return z, z
        if name in ("doy", "harm"):
            return X[DOYI[t]], X[DOYI[tk]]
        return X[t], X[t]

    tr_s = TT.starts(d["times"], d["tr"], K); va_s = TT.starts(d["times"], d["va"], K)[::2]
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=10.0)
    if a.vel == "factor":
        model = TR.VelocityModel(["rea", "geo", "wind"], n_static=d["STATIC"].shape[0], hidden=16, a0=0.25, b0=0.0, b_range=(-0.5, 1.5)).to(dev)
        groups = [dict(params=[p for n_, p in model.named_parameters() if not n_.startswith("w")], lr=a.lr), dict(params=[model.wp, model.wq], lr=a.lr_wind)]
    else:
        model = TR.DecompVelocity(len(TR.DECOMP_COMPS), int(d["REG"].max()) + 1, "season", 0.25, wind=False).to(dev)
        groups = [dict(params=list(model.parameters()), lr=a.lr)]
    opt = torch.optim.Adam(groups)
    cfg1 = json.load(open(STEP1 / "config.json"))
    m1 = TR.VelocityModel(cfg1["terms"], n_static=d["STATIC"].shape[0], hidden=cfg1["hidden"], a0=cfg1["a_init"], b0=cfg1["b_init"],
                          b_range=(cfg1["b_min"], 1.5), cell=cfg1["corr_cell"], corr_hidden=cfg1["corr_hidden"]).to(dev)
    m1.load_state_dict(torch.load(STEP1 / "model.pt", map_location=dev)["model"]); m1.eval()
    if a.scan_a is None:
        sc = json.load(open(ROOT / "analyses/tracer/results/static_scan.json"))
        a.scan_a = sc["scan"][a.base]["best_a"]["mae"][0]

    def velocity(days, mdl, collect=None):
        nb = days.shape[0]; fl = days.reshape(-1); g = lambda X: X[fl].float()
        if isinstance(mdl, TR.DecompVelocity):
            comps = TR.decomp_components(g(d["U"]), g(d["V"]), g(d["UG"]), g(d["VG"]), g(d["CUROK"]), d["CLIMD"][d["DOYI"][fl]])
            ue, ve = mdl(comps, d["REG"], SEASON[fl], d["DS"][fl], d["DC"][fl], g(d["TX"]), g(d["TY"]), OCf, collect=collect)
        else:
            ue, ve = mdl(d["STATIC"], d["DS"][fl], d["DC"][fl], g(d["U"]), g(d["V"]), g(d["UG"]), g(d["VG"]), g(d["TX"]), g(d["TY"]), OCf, collect=collect)
        U = torch.stack([ue * d["CPDX"], ve * d["CPDY"]], 1).view(nb, K + 1, 2, H, W)
        return U, d["CUROK"][fl].float().view(nb, K + 1, H, W)

    def days_of(bt):
        return torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev)

    @torch.no_grad()
    def evaluate(sidx, final=False):
        model.eval()
        names = ["model"] + (["step1", "none@0.25", f"{a.base}@{a.scan_a}", "persistence"] if final else [])
        acc = {s: torch.zeros(K, len(names), 2, dtype=torch.float64, device=dev) for s in ("own", "common")}
        fac = []
        for b0 in range(0, len(sidx), 16):
            bt = sidx[b0:b0 + 16]; days = days_of(bt); t = days[:, 0]
            col = {}
            Um, ok = velocity(days, model, col)
            fac.append(torch.hypot(Um[:, 0, 0], Um[:, 0, 1])[:, oc].mean(1))
            if final:
                U1, _ = velocity(days, m1)
                fl = days.reshape(-1)
                Ur = torch.stack([d["U"][fl].float() * d["CPDX"], d["V"][fl].float() * d["CPDY"]], 1).view(len(bt), K + 1, 2, H, W)
            yt, ot = d["Y"][t], d["OBS"][t]
            for k in range(1, K + 1):
                tk = days[:, k]; truth = d["Y"][tk]; tobs = d["OBS"][tk]
                Bt, Btk = base_at(a.base, BX, t, tk)
                pos, val = it.departure(Um, ok, k)
                p_, v_ = TS.predict(it, pos, yt, ot, Bt, Btk)
                pr = {"model": p_}; vl = {"model": v_ & val}
                if final:
                    pos, val = it.departure(U1, ok, k); p_, v_ = it.depart_sample(yt, ot, pos); pr["step1"], vl["step1"] = p_, v_ & val
                    pos, val = it.departure(Ur * 0.25, ok, k); p_, v_ = it.depart_sample(yt, ot, pos); pr["none@0.25"], vl["none@0.25"] = p_, v_ & val
                    pos, val = it.departure(Ur * a.scan_a, ok, k); p_, v_ = TS.predict(it, pos, yt, ot, Bt, Btk); pr[names[3]], vl[names[3]] = p_, v_ & val
                    pr["persistence"], vl["persistence"] = yt, ot
                common = tobs & oc[None]
                for nm in names:
                    common &= vl[nm]
                for q, nm in enumerate(names):
                    e = (pr[nm] - truth).abs().double()
                    for s, m in (("own", tobs & vl[nm]), ("common", common)):
                        acc[s][k - 1, q] += torch.stack([(e * m).sum(), m.double().sum()])
        model.train()
        r = {s: {nm: [dict(mae=float(acc[s][k, q, 0] / acc[s][k, q, 1].clamp(min=1)), n=int(acc[s][k, q, 1])) for k in range(K)] for q, nm in enumerate(names)} for s in acc}
        r["mean_lead_own"] = float(np.mean([x["mae"] for x in r["own"]["model"]]))
        r["speed_px_day_ocean_mean"] = float(torch.cat(fac).mean())
        return r

    def factor_summary():
        if a.vel == "decomp":
            return dict(alpha=model.alpha.detach().cpu().tolist(), comps=TR.DECOMP_COMPS)
        with torch.no_grad():
            doys = torch.linspace(0, 2 * np.pi, 13, device=dev)[:-1]
            x = torch.cat([d["STATIC"][None].expand(12, -1, -1, -1), torch.sin(doys).view(12, 1, 1, 1).expand(12, 1, H, W), torch.cos(doys).view(12, 1, 1, 1).expand(12, 1, H, W)], 1)
            am = model.a(x)[:, oc]; bm = model.b(x)[:, oc]
        p, q = model.wind_coef(torch.sin(doys), torch.cos(doys))
        return dict(a_ocean_mean=float(am.mean()), a_ocean_sd=float(am.std()), a_monthly_mean=am.mean(1).tolist(), b_ocean_mean=float(bm.mean()),
                    wind_c=torch.hypot(p, q).tolist(), wind_theta_deg=torch.rad2deg(torch.atan2(q, p)).tolist())

    cfg = dict(vars(a), n_train_starts=len(tr_s), n_val_starts=len(va_s), base_info=B["info"])
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    hist = dict(epoch=[], train_l1=[], val_mean_lead_own=[], val_own=[], factors=[], epoch_s=[])
    ev0 = evaluate(va_s)
    print(f"init val own {[round(x['mae'], 4) for x in ev0['own']['model']]} ({time.time() - t0:.0f}s)", flush=True)
    best, best_ep = ev0["mean_lead_own"], 0
    torch.save(dict(model=model.state_dict(), epoch=0), out / "model.pt")
    for ep in range(a.epochs):
        te = time.time(); perm = rng.permutation(len(tr_s)); nb = len(perm) // a.batch; ls_sum = 0.0
        for bi in range(nb):
            bt = tr_s[perm[bi * a.batch:(bi + 1) * a.batch]]; days = days_of(bt); t = days[:, 0]
            Um, ok = velocity(days, model)
            ls = []
            for k in range(1, K + 1):
                tk = days[:, k]
                Bt, Btk = base_at(a.base, BX, t, tk)
                pos, val = it.departure(Um, ok, k)
                p_, v_ = TS.predict(it, pos, d["Y"][t], d["OBS"][t], Bt, Btk)
                m = v_ & val & d["OBS"][tk]
                ls.append((p_ - d["Y"][tk]).abs()[m].mean())
            loss = torch.stack(ls).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ls_sum += float(loss.detach())
        ev = evaluate(va_s)
        hist["epoch"].append(ep + 1); hist["train_l1"].append(ls_sum / nb); hist["val_mean_lead_own"].append(ev["mean_lead_own"])
        hist["val_own"].append([x["mae"] for x in ev["own"]["model"]]); hist["factors"].append(factor_summary()); hist["epoch_s"].append(time.time() - te)
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        if ev["mean_lead_own"] < best:
            best, best_ep = ev["mean_lead_own"], ep + 1
            torch.save(dict(model=model.state_dict(), epoch=ep + 1), out / "model.pt")
        fs = hist["factors"][-1]
        print(f"epoch {ep + 1} train {hist['train_l1'][-1]:.4f} val own {[round(x, 4) for x in hist['val_own'][-1]]} "
              + (f"a {fs['a_ocean_mean']:.3f} b {fs['b_ocean_mean']:.3f}" if a.vel == "factor" else f"alpha mean by comp {np.round(np.mean(fs['alpha'], (1, 2)), 3).tolist()}")
              + f" speed {ev['speed_px_day_ocean_mean']:.3f} ({hist['epoch_s'][-1]:.0f}s)", flush=True)
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev)["model"])
    evf = evaluate(va_s, final=True)
    metrics = dict(best_epoch=best_ep, best_val_mean_lead_own=best, final=evf, factors_best=factor_summary(), wall_s=time.time() - t0,
                   peak_gb=torch.cuda.max_memory_allocated() / 2**30, init=ev0)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("final common", {nm: [round(x["mae"], 4) for x in v] for nm, v in evf["common"].items()}, "n", [x["n"] for x in evf["common"]["model"]], flush=True)
    print(f"done best epoch {best_ep} wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
