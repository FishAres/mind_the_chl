"""Band-split tracer, TEST 2 (learned). Standalone loop; imports train_tracer.load / starts / SRC_CH, tracer.Integrator /
BandPass / grad_acc / SourceModel, tracer_band; changes none of them.

    ln C^(x, t+k) = B(x, t+k) + sum_b D_b(k) A_b(X^b_k(x), t)              (bands < 20, 20-50, 50-150 km; B = lp base)
  velocity  band b moves with a_b u_rea (a_b free scalar, init 0.25 for all bands = test-1 x0.25).
  damping   --tau-mode const: D_b(k) = exp(-k / tau_b), tau_b free (log-parametrised), init tau_measured (1.67 / 2.38 /
            13.8 d). --tau-mode phys: tau_b(x, t) = tau_b * softplus(z_b) / ln 2, z = MLP(z-scored log10(strain + 0.01
            /day), |u_rea| px/day, log10 MLD (step-2 channel), doy sin, cos), 1x1 convs hidden 16, last layer zero
            (multiplier 1 at init); the rate 1 / tau_b(x, t) is computed on the daily grid and sampled bilinearly / linearly in
            time at the band trajectory's RK2 sub-step midpoints: D_b = exp(-sum h rate).
  base      B(x, t+k) = B(x, t) + sum over sub-steps h S_B; S_B = 0.5 tanh(MLP) (tracer.SourceModel, hidden 32, zero at init),
            Eulerian at the arrival px (B is a 150 km low-pass; the step-2 source was Lagrangian), explicit Euler with 4
            sub-steps per day, S_B evaluated at the sub-step mid time; inputs = the step-2 source channels (train_tracer
            SRC_CH, time-interpolated), log10 depth, log10(1 + coast distance), doy sin / cos, and B itself (z-scored with
            the train observed ln C mean / sd). No anomaly input. --no-source: B held fixed.
Loss pixel L1 over target observed & valid prediction (all three band trajectories valid, departure observed fraction
>= 0.99 for each band, B finite), mean per lead, leads 1-5 equal weights; Adam, lr 3e-3 all groups, batch 16, train
starts as train_tracer (2,449; 153 steps / epoch), 20 epochs, constant lr. Gradient checkpointing per (band, lead) term
and per source sub-step.
Val each epoch (351 starts), scored on the TEST-1 common mask (~/mind_the_chl_gap/data/band_common) intersected with the
model's valid px; references recomputed on the same px every epoch: step1 (tr_rea_geo_wind), x0.25, persistence, lp_alone,
test1 (x0.25 bands, measured tau, = this model at init without source). Own-set MAE (target observed & model valid) for
selection: best epoch = min mean-lead own-set MAE (as step 1 / 2). Per-start sums saved per epoch (bootstrap).
Every epoch's weights are kept (ep{n}.pt). Final: lead-1 B-source season maps and S_B / tau-multiplier samples
(partial dependence) for the best epoch and the last epoch.
GPU: data loaded once on the CPU; each chunk of --epochs-per-lock epochs runs with ~/mind_the_chl_gap/logs/gpu.lock held
(fcntl.flock), then the GPU copy is freed and the lock released. Resumes from state.pt.
Usage: python -m analyses.tracer.band_train --runs trb_const_s0:const:0,trb_const_s1:const:1 [--epochs 20]
"""
import sys, json, time, argparse, fcntl, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from chl_cnn import train_tracer as TT, tracer as TR, tracer_static as TS, tracer_band as TB
from chl_cnn.runs import CATEGORIES

BASE_DIR = Path.home() / "mind_the_chl_gap/data/static_bases"
COMMON = Path.home() / "mind_the_chl_gap/data/band_common/common_mask.npz"
LOCK = Path.home() / "mind_the_chl_gap/logs/gpu.lock"
K = 5
LAGS = (1, 4, 16)
REFS = ["step1", "x0.25", "persistence", "lp_alone", "test1"]
PRED = ["model"] + REFS
NSUB = 4


def load_cpu():
    t0 = time.time()
    d = TT.load(TT.parse(["--run", "band_train", "--terms", "rea,geo,wind", "--source"]), torch.device("cpu"))
    Bc = TS.bases_cached(None, None, None, None, BASE_DIR)
    d["BL"] = torch.as_tensor(np.ascontiguousarray(Bc["lp"]))
    z = np.load(COMMON)
    shp = tuple(z["shape"]); d["MASK"] = torch.as_tensor(np.unpackbits(z["common"])[:np.prod(shp)].reshape(shp).astype(bool))
    d["MASK_STARTS"] = z["starts"]
    d["SEASON"] = torch.as_tensor((((np.asarray(d["times"]).astype("datetime64[M]").astype(int) % 12) + 1) % 12) // 3)
    # z-stats of the tau inputs (log10 strain + 0.01, speed px/day) over ocean & valid current px of every 5th train day
    H, W = d["ocean"].shape
    sub = torch.as_tensor(d["tr"][::5])
    U = torch.stack([d["U"][sub].float() * d["CPDX"], d["V"][sub].float() * d["CPDY"]], 1)
    okc = d["CUROK"][sub]
    st, sp = TB.strain_speed(U, okc)
    ls = torch.log10(st + 0.01)
    sel = okc & d["OCEAN"][None]
    d["TAUZ"] = [float(ls[sel].mean()), float(ls[sel].std()), float(sp[sel].mean()), float(sp[sel].std())]
    print(f"cpu load {time.time() - t0:.0f}s tau z-stats {d['TAUZ']}", flush=True)
    return d


def to_dev(d, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}


class Lock:
    def __enter__(self):
        self.f = open(LOCK, "a"); t = time.time()
        fcntl.flock(self.f, fcntl.LOCK_EX)
        print(f"  gpu lock acquired after {time.time() - t:.0f}s", flush=True)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def build(run_cfg, d, dev):
    tau, _ = TB.tau_measured()
    model = TB.BandModel(tau, a0=0.25, tau_mode=run_cfg["tau_mode"], src_in=len(TT.SRC_CH) + 5, g_max=0.5, source=run_cfg["source"]).to(dev)
    return model, tau


class Engine:
    """Forward model and evaluation on GPU data dg."""

    def __init__(self, dg, model, tau_meas, dev):
        self.dg, self.m, self.dev = dg, model, dev
        self.H, self.W = dg["ocean"].shape
        self.it = TR.Integrator(self.H, self.W, dev, n_sub=NSUB, vmax=10.0)
        self.bp = TR.BandPass(20.0, 70.0, min_frac=0.5, device=dev)
        self.NF = len(TT.SRC_CH)
        self.tau_meas = torch.as_tensor(tau_meas, device=dev, dtype=torch.float32)
        cfg = json.load(open(CATEGORIES["tracer"] / "tr_rea_geo_wind" / "config.json"))
        m1 = TR.VelocityModel(cfg["terms"], n_static=dg["STATIC"].shape[0], hidden=cfg["hidden"], a0=cfg["a_init"], b0=cfg["b_init"],
                              b_range=(cfg["b_min"], 1.5), cell=cfg["corr_cell"], corr_hidden=cfg["corr_hidden"]).to(dev)
        m1.load_state_dict(torch.load(CATEGORIES["tracer"] / "tr_rea_geo_wind" / "model.pt", map_location=dev)["model"]); m1.eval()
        self.m1 = m1

    def fields(self, days):
        dg = self.dg; nb = days.shape[0]; fl = days.reshape(-1); H, W = self.H, self.W
        ok = dg["CUROK"][fl].view(nb, K + 1, H, W)
        U1 = torch.stack([dg["U"][fl].float() * dg["CPDX"], dg["V"][fl].float() * dg["CPDY"]], 1).view(nb, K + 1, 2, H, W)
        return U1, ok

    def tau_inputs(self, days, U1, ok):
        dg = self.dg; nb = days.shape[0]; fl = days.reshape(-1); H, W = self.H, self.W
        st, sp = TB.strain_speed(U1.reshape(-1, 2, H, W), ok.reshape(-1, H, W))
        z = dg["TAUZ"]
        x = torch.stack([((torch.log10(st + 0.01) - z[0]) / z[1]).clamp(-5, 5), ((sp - z[2]) / z[3]).clamp(-5, 5), dg["FEAT"][fl, 2].float(),
                         dg["DS"][fl].view(-1, 1, 1).expand(-1, H, W), dg["DC"][fl].view(-1, 1, 1).expand(-1, H, W)], 1)
        return x * dg["OCf"]

    def forward(self, days, grad=True, rec=None):
        """Returns preds [(pred, valid)] for k = 1..K, Bk list (k = 0..K), extras."""
        dg, m, it = self.dg, self.m, self.it; H, W = self.H, self.W
        nb = days.shape[0]; fl = days.reshape(-1); t = days[:, 0]
        U1, ok = self.fields(days); okf = ok.float()
        yt, ot = dg["Y"][t], dg["OBS"][t]; Bt = dg["BL"][t]
        with torch.no_grad():
            A, mA = TB.decompose(yt, ot, Bt)
        Bfin = torch.isfinite(Bt); B0 = torch.nan_to_num(Bt)
        h = 1.0 / NSUB
        Bk = [B0]
        if m.src is not None:
            Fd = dg["FEAT"][fl].float().view(nb, K + 1, self.NF, H, W)
            SST = dg["SSTATIC"][None].expand(nb, -1, -1, -1)

            def src_x(c, j, w):
                ds = ((1 - w) * dg["DS"][days[:, j]] + w * dg["DS"][days[:, j + 1]]).view(nb, 1, 1, 1).expand(nb, 1, H, W)
                dc = ((1 - w) * dg["DC"][days[:, j]] + w * dg["DC"][days[:, j + 1]]).view(nb, 1, 1, 1).expand(nb, 1, H, W)
                return torch.cat([(1 - w) * Fd[:, j] + w * Fd[:, j + 1], SST, ds, dc, ((c - dg["CMU"]) / dg["CSD"])[:, None]], 1)

            def src_step(c, j, w):
                return m.src(src_x(c, j, w))
            c = B0
            for j in range(K):
                for i in range(NSUB):
                    w = (i + 0.5) * h
                    if grad:
                        S = checkpoint(src_step, c, j, w, use_reentrant=False)
                    else:
                        S = src_step(c, j, w)
                        if rec is not None and j == 0:
                            rec.setdefault("src", []).append((src_x(c, j, w), S))
                    c = c + h * S
                Bk.append(c)
        else:
            Bk = [B0] * (K + 1)
        R = None
        if m.tau_mode == "phys":
            xt = self.tau_inputs(days, U1, ok)
            mult = m.tau_mult(xt).view(nb, K + 1, 3, H, W)
            R = 1.0 / (m.tau().view(1, 1, 3, 1, 1) * mult)
            if rec is not None:
                rec["tau"] = (xt.view(nb, K + 1, -1, H, W)[:, 0], mult[:, 0])

        def band_term(a_b, lt_b, Rb, k, b):
            path = [] if Rb is not None else None
            pos, val = it.departure(U1 * a_b, okf, k, path=path)
            s, v = TB.band_sample(it, A[:, b:b + 1], mA, pos)
            if Rb is None:
                D = torch.exp(-k / torch.exp(lt_b))
            else:
                integ = torch.zeros_like(s[:, 0])
                for p, j, w in path:
                    r = it.sample(Rb[:, j:j + 2], p)
                    integ = integ + h * ((1 - w) * r[:, 0] + w * r[:, 1])
                D = torch.exp(-integ)
            return D * s[:, 0], val & v
        preds = []
        for k in range(1, K + 1):
            pred = Bk[k]; valid = Bfin.clone()
            for b in range(3):
                Rb = None if R is None else R[:, :, b]
                if grad:
                    cb, vb = checkpoint(band_term, m.a[b], m.log_tau[b], Rb, k, b, use_reentrant=False)
                else:
                    cb, vb = band_term(m.a[b], m.log_tau[b], Rb, k, b)
                pred = pred + cb; valid = valid & vb
            preds.append((pred, valid))
        return preds, Bk, dict(A=A, mA=mA, Bt=Bt, U1=U1, okf=okf)

    @torch.no_grad()
    def refs(self, days, ex, k):
        dg, it = self.dg, self.it; t = days[:, 0]; nb = days.shape[0]; fl = days.reshape(-1); H, W = self.H, self.W
        yt, ot = dg["Y"][t], dg["OBS"][t]
        out = {}
        g = lambda X: X[fl].float()
        if "_U1s" not in ex:
            ue, ve = self.m1(dg["STATIC"], dg["DS"][fl], dg["DC"][fl], g(dg["U"]), g(dg["V"]), g(dg["UG"]), g(dg["VG"]), g(dg["TX"]), g(dg["TY"]), dg["OCf"])
            ex["_U1s"] = torch.stack([ue * dg["CPDX"], ve * dg["CPDY"]], 1).view(nb, K + 1, 2, H, W)
        pos, val = it.departure(ex["_U1s"], ex["okf"], k); p_, v_ = it.depart_sample(yt, ot, pos); out["step1"] = (p_, v_ & val)
        pos, val = it.departure(ex["U1"] * 0.25, ex["okf"], k); p_, v_ = it.depart_sample(yt, ot, pos); out["x0.25"] = (p_, v_ & val)
        s, v = TB.band_sample(it, ex["A"], ex["mA"], pos)
        w = torch.exp(-k / self.tau_meas).view(1, 3, 1, 1)
        out["test1"] = (torch.nan_to_num(ex["Bt"]) + (s * w).sum(1), v & val & torch.isfinite(ex["Bt"]))
        out["persistence"] = (yt, ot)
        out["lp_alone"] = (torch.nan_to_num(ex["Bt"]), torch.isfinite(ex["Bt"]))
        return out

    @torch.no_grad()
    def evaluate(self, va_s, final=False, pd_every=50):
        dg, m = self.dg, self.m; H, W = self.H, self.W
        m.eval()
        ns = len(va_s); nreg = int(dg["REG"].max()) + 1; P = len(PRED)
        acc = torch.zeros(ns, K, P, 4, dtype=torch.float64, device=self.dev)        # on mask & model valid: sum|e|, n, band sum|e|, band n
        own = torch.zeros(K, 2, dtype=torch.float64, device=self.dev)
        gacc = torch.zeros(K, P, len(LAGS), 6, dtype=torch.float64, device=self.dev)
        racc = torch.zeros(P, nreg, 2, dtype=torch.float64, device=self.dev)
        nmask = torch.zeros(K, dtype=torch.float64, device=self.dev)
        sB = torch.zeros(4, H, W, dtype=torch.float64, device=self.dev); nB = torch.zeros_like(sB)
        taum = torch.zeros(4, 3, dtype=torch.float64, device=self.dev); taun = torch.zeros(4, dtype=torch.float64, device=self.dev)
        gen = torch.Generator(device=self.dev); gen.manual_seed(0)
        pdx, pds, tdx, tdm = [], [], [], []
        oc = dg["OCEAN"]
        for b0 in range(0, ns, 16):
            bt = va_s[b0:b0 + 16]; nb = len(bt)
            days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=self.dev); t = days[:, 0]
            rec = {} if final else None
            if m.tau_mode == "phys":
                rec = {} if rec is None else rec
            preds, Bk, ex = self.forward(days, grad=False, rec=rec)
            sea = dg["SEASON"][t]
            fin = torch.isfinite(ex["Bt"]) & oc[None]
            dB = (Bk[1] - Bk[0]).double()
            for q in range(4):
                mq = fin & (sea == q)[:, None, None]
                sB[q] += torch.where(mq, dB, 0.0).sum(0); nB[q] += mq.double().sum(0)
            if m.tau_mode == "phys":
                xt, mu = rec["tau"]
                for q in range(4):
                    mq = oc[None] & (sea == q)[:, None, None]
                    taum[q] += (mu * mq[:, None]).double().sum((0, 2, 3)); taun[q] += mq.double().sum()
                if final:
                    sel = oc[None].expand(nb, -1, -1) & (torch.rand((nb, H, W), generator=gen, device=self.dev) < 1.0 / pd_every)
                    tdx.append(xt.permute(1, 0, 2, 3)[:, sel].T.cpu()); tdm.append(mu.permute(1, 0, 2, 3)[:, sel].T.cpu())
            if final and "src" in rec:
                sel = fin & (torch.rand((nb, H, W), generator=gen, device=self.dev) < 1.0 / pd_every)
                for xr, Sr in rec["src"]:
                    pdx.append(xr.permute(1, 0, 2, 3)[:, sel].T.cpu()); pds.append(Sr[sel].cpu())
            for k in range(1, K + 1):
                tk = days[:, k]; truth = dg["Y"][tk]; tobs = dg["OBS"][tk]
                pm, vm = preds[k - 1]
                own[k - 1] += torch.stack([((pm - truth).abs().double() * (vm & tobs)).sum(), (vm & tobs).double().sum()])
                mk = dg["MASK"][b0:b0 + nb, k - 1]
                nmask[k - 1] += mk.double().sum()
                cm = mk & vm
                rf = self.refs(days, ex, k)
                for p in REFS:
                    cm = cm & rf[p][1]
                pr = {"model": pm}; pr.update({p: rf[p][0] for p in REFS})
                cf = cm.double()
                bt_t, vb = self.bp(truth, cm); vbf = vb.double()
                regc = dg["REG"][None].expand_as(cm)[cm]
                for q, p in enumerate(PRED):
                    e = pr[p] - truth
                    bpp, _ = self.bp(pr[p], cm)
                    acc[b0:b0 + nb, k - 1, q] = torch.stack([(e.abs().double() * cf).sum((1, 2)), cf.sum((1, 2)), ((bpp - bt_t).abs().double() * vbf).sum((1, 2)), vbf.sum((1, 2))], 1)
                    ga = TR.grad_acc(pr[p], truth, cm, LAGS)
                    for li, L in enumerate(LAGS):
                        gacc[k - 1, q, li] += ga[L]
                    if k == 1:
                        racc[q, :, 0] += torch.bincount(regc, weights=e[cm].abs().double(), minlength=nreg)
                        racc[q, :, 1] += torch.bincount(regc, minlength=nreg).double()
        m.train()
        a_ = acc.sum(0)
        res = dict(mae={p: (a_[:, q, 0] / a_[:, q, 1]).tolist() for q, p in enumerate(PRED)}, band_mae={p: (a_[:, q, 2] / a_[:, q, 3]).tolist() for q, p in enumerate(PRED)},
                   n=a_[:, 0, 1].tolist(), n_band=a_[:, 0, 3].tolist(), n_mask=nmask.tolist(),
                   own_mae=(own[:, 0] / own[:, 1]).tolist(), own_n=own[:, 1].tolist(),
                   grad={p: {str(k + 1): {str(L): dict(ratio=float(gacc[k, q, li, 0] / gacc[k, q, li, 1]), corr=float(gacc[k, q, li, 2] / (gacc[k, q, li, 3] * gacc[k, q, li, 4]).sqrt()), n_pairs=int(gacc[k, q, li, 5]))
                                                 for li, L in enumerate(LAGS)} for k in range(K)} for q, p in enumerate(PRED)},
                   region_lead1={p: {str(r): dict(mae=float(racc[q, r, 0] / racc[q, r, 1].clamp(min=1)), n=int(racc[q, r, 1])) for r in range(1, nreg)} for q, p in enumerate(PRED)},
                   a=m.a.detach().tolist(), tau=m.tau().detach().tolist(),
                   srcB_lead1={nm: float(sB[q].sum() / nB[q].sum().clamp(min=1)) for q, nm in enumerate(["DJF", "MAM", "JJA", "SON"])},
                   srcB_lead1_all=float(sB.sum() / nB.sum().clamp(min=1)), srcB_n=float(nB.sum()))
        res["own_mean_lead"] = float(np.mean(res["own_mae"]))
        if m.tau_mode == "phys":
            res["tau_mult_mean"] = {nm: (taum[q] / taun[q].clamp(min=1)).tolist() for q, nm in enumerate(["DJF", "MAM", "JJA", "SON"])}
        extra = dict(acc=acc.cpu().numpy())
        if final:
            with np.errstate(invalid="ignore", divide="ignore"):
                extra["srcB_map"] = np.where(nB.cpu().numpy() > 0, sB.cpu().numpy() / nB.cpu().numpy(), np.nan).astype("float32")
            extra["srcB_n"] = nB.cpu().numpy().astype("int32")
            if pdx:
                extra["pd_x"] = torch.cat(pdx).numpy().astype("float32"); extra["pd_s"] = torch.cat(pds).numpy().astype("float32")
            if tdx:
                extra["tau_x"] = torch.cat(tdx).numpy().astype("float32"); extra["tau_m"] = torch.cat(tdm).numpy().astype("float32")
        return res, extra


def train_run(name, tau_mode, seed, d, a):
    out = CATEGORIES["tracer"] / name; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    run_cfg = dict(run=name, tau_mode=tau_mode, seed=seed, source=not a.no_source, epochs=a.epochs, batch=a.batch, lr=a.lr, a_init=0.25, g_max=0.5,
                   n_sub=NSUB, leads=K, bands=TB.BAND_NAMES, edges_km=[20, 50, 150], base="lp (tracer_static, held + source)", tau_z=d["TAUZ"],
                   doc=__doc__.split("Usage:")[0].strip())
    tr_s = TT.starts(d["times"], d["tr"], K); va_s = TT.starts(d["times"], d["va"], K)[::2]
    assert np.array_equal(va_s, d["MASK_STARTS"])
    run_cfg.update(n_train_starts=int(len(tr_s)), n_val_starts=int(len(va_s)))
    (out / "config.json").write_text(json.dumps(run_cfg, indent=1))
    st_path = out / "state.pt"
    torch.manual_seed(seed)
    model, tau = build(run_cfg, d, torch.device("cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(seed)
    hist = dict(epoch=[], train_l1=[], train_l1_lead=[], epoch_s=[], eval=[])
    perday = []
    ep0 = 0
    if st_path.exists():
        s = torch.load(st_path, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); rng.bit_generator.state = s["rng"]
        hist = s["hist"]; perday = list(np.load(out / "perday.npz")["acc"]) if (out / "perday.npz").exists() else []; ep0 = s["epoch"]
        perday = perday[:len(hist["eval"])]          # epochs after the last state.pt (interrupted chunk) are recomputed
        print(f"{name}: resume at epoch {ep0}", flush=True)
    if ep0 >= a.epochs and (out / "metrics.json").exists() and json.load(open(out / "metrics.json")).get("done"):
        print(f"{name}: done already", flush=True); return
    while True:
        with Lock():
            t0 = time.time()
            dg = to_dev(d, dev)
            model.to(dev)
            for st in opt.state.values():
                for k_, v in st.items():
                    if torch.is_tensor(v):
                        st[k_] = v.to(dev)
            eng = Engine(dg, model, tau, dev)
            torch.cuda.reset_peak_memory_stats()
            print(f"{name}: on GPU {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
            if ep0 == 0 and not hist["eval"]:
                ev, ex = eng.evaluate(va_s)
                hist["eval"].append(ev); perday.append(ex["acc"])
                torch.save(dict(model=model.state_dict(), epoch=0), out / "ep0.pt")
                print(f"{name} init: own {np.round(ev['own_mae'], 4).tolist()} common model {np.round(ev['mae']['model'], 4).tolist()} test1 {np.round(ev['mae']['test1'], 4).tolist()} n {ev['n'][0]:.0f}/{ev['n_mask'][0]:.0f}", flush=True)
            ep_stop = min(a.epochs, ep0 + a.epochs_per_lock)
            for ep in range(ep0, ep_stop):
                te = time.time()
                perm = rng.permutation(len(tr_s)); nb = len(perm) // a.batch
                if a.max_batches:
                    nb = min(nb, a.max_batches)
                lsum = torch.zeros(K, device=dev)
                for bi in range(nb):
                    bt = tr_s[perm[bi * a.batch:(bi + 1) * a.batch]]
                    days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev)
                    preds, _, _ = eng.forward(days, grad=True)
                    ls = []
                    for k in range(1, K + 1):
                        pr, vl = preds[k - 1]
                        mk = vl & dg["OBS"][days[:, k]]
                        ls.append((pr - dg["Y"][days[:, k]]).abs()[mk].mean())
                    ls = torch.stack(ls); loss = ls.mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    lsum += ls.detach()
                    if bi == 0 and ep == ep0:
                        print(f"  first step peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB ({time.time() - te:.1f}s)", flush=True)
                ev, ex = eng.evaluate(va_s)
                ltr = (lsum / max(nb, 1)).tolist()
                hist["epoch"].append(ep + 1); hist["train_l1"].append(float(np.mean(ltr))); hist["train_l1_lead"].append(ltr)
                hist["epoch_s"].append(time.time() - te); hist["eval"].append(ev); perday.append(ex["acc"])
                torch.save(dict(model=model.state_dict(), epoch=ep + 1), out / f"ep{ep + 1}.pt")
                (out / "history.json").write_text(json.dumps(hist, indent=1))
                np.savez_compressed(out / "perday.npz", acc=np.stack(perday), preds=np.array(PRED))
                print(f"{name} ep {ep + 1}: train {hist['train_l1'][-1]:.4f} own {np.round(ev['own_mae'], 4).tolist()} common model {np.round(ev['mae']['model'], 4).tolist()} "
                      f"step1 {np.round(ev['mae']['step1'], 4).tolist()} a {np.round(ev['a'], 3).tolist()} tau {np.round(ev['tau'], 2).tolist()} "
                      f"dB l1 {ev['srcB_lead1_all']:+.4f} ({hist['epoch_s'][-1]:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
            ep0 = ep_stop
            model.cpu()
            for st in opt.state.values():
                for k_, v in st.items():
                    if torch.is_tensor(v):
                        st[k_] = v.cpu()
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state, hist=hist, epoch=ep0), st_path)
            if ep0 >= a.epochs:
                # final: best (min own mean-lead MAE) and last epoch extras
                own_ml = [e["own_mean_lead"] for e in hist["eval"]]
                best_ep = int(np.argmin(own_ml[1:]) + 1)
                metrics = dict(best_epoch=best_ep, selection="min mean-lead val own-set MAE over epochs 1..E", own_mean_lead=own_ml, wall_s_last_chunk=time.time() - t0)
                for tag, ep in (("best", best_ep), ("last", a.epochs)):
                    model.load_state_dict(torch.load(out / f"ep{ep}.pt", map_location="cpu")["model"]); model.to(dev)
                    eng.m = model
                    evf, exf = eng.evaluate(va_s, final=True)
                    metrics[tag] = dict(epoch=ep, eval=evf)
                    np.savez_compressed(out / f"final_{tag}.npz", **{k_: v for k_, v in exf.items() if k_ != "acc"}, ocean=d["ocean"], lat=d["lat"], lon=d["lon"],
                                        region=d["REG"].numpy(), src_channels=np.array(d["info"]["source"]["channels"][:-1] + ["B (z)"]),
                                        tau_channels=np.array(["log10 strain z", "speed z", "log10 MLD z", "doy sin", "doy cos"]), c_mean=d["CMU"], c_sd=d["CSD"])
                    model.cpu()
                metrics["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30; metrics["done"] = True
                metrics["epoch_s_mean"] = float(np.mean(hist["epoch_s"]))
                (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
            del eng, dg
            torch.cuda.empty_cache()
        if ep0 >= a.epochs:
            print(f"{name}: done", flush=True); return
        time.sleep(5)          # let a waiting job take the lock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list name:tau_mode:seed")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--epochs-per-lock", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-batches", type=int, default=0); ap.add_argument("--no-source", action="store_true")
    a = ap.parse_args()
    d = load_cpu()
    for r in a.runs.split(","):
        nm, mode, seed = r.split(":")
        train_run(nm, mode, int(seed), d, a)


if __name__ == "__main__":
    main()
