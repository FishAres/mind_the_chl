"""Synthetic checks of chl_cnn/tracer.py (GPU). Run: python -m chl_cnn.test_tracer  -> prints a table, writes
analyses/tracer/results/synthetic_checks.json.

  1 uniform constant velocity (u, v) = (2.3, -1.7) px/day: departure displacement must be -(k u, k v), k = 1..5; the
    departure-sampled field vs the analytic shifted field (bilinear interpolation error only).
  2 uniform velocity linear in time (u_j = 0.5 j px/day on day j): displacement over [0, k] = -0.25 k^2 (RK2 exact).
  3 solid-body rotation (space-varying, omega = 0.05 rad/day): departure point vs the analytic rotation, n_sub 1 / 4.
  4 recovery: target = smooth random field advected by 1.0 x a smooth random velocity field (RK2, n_sub 8); a learned
    rea factor (VelocityModel, terms rea, init 0.25) trained with L1 for 300 Adam steps must return ~1.0.
  5 a field shifted by a known uniform velocity (3 px/day east, 2 px/day north = +row) over one day, recovered by fitting
    a uniform velocity (two scalars from 0) through the integrator.
  6 departure(..., path=[]) returns the same positions as without path (bit-identical) and k * n_sub midpoints.
  7 SourceModel at init outputs exactly 0; with the last-layer bias set to atanh(s / g_max) it outputs s everywhere, and
    the forward sum over the k * n_sub sub-steps (h = 1 / n_sub) adds k s to the departure value.

Step 1b checks (python -m chl_cnn.test_tracer --step1b [--cpu]; writes analyses/tracer/results/step1b_checks.json;
bandpass_response.npz for the report figure):
  B1 BandPass response vs wavelength: plane sinusoids (8 directions) of wavelength 9-400 km on a 256 x 256 grid,
    fully observed and with 30 % blob clouds + a land block; measured amplitude = LSQ fit of the output onto cos / sin
    of the input wave at valid pixels >= 20 px from the grid edge; vs the analytic DoG response. Bands 24-150, 20-70 km.
  B2 mask edges: constant field and a large-scale linear ramp (0.5 / 100 px) under the same clouds / land: max |output|
    at valid pixels for the normalised-convolution band-pass and for a naive zero-filled DoG.
  B3 decomposition: random fields with invalid geostrophic pixels and invalid currents: max |sum of components -
    u_rea ok|; climatology of a doy sinusoid + noise from train days only (val days set to 1e3) vs the truth;
    DecompVelocity at alpha = 0.25 equals 0.25 u_rea; alpha gather by region / season.
  B4 recovery: field advected 1 day by 0.6 u_1 + 1.3 u_2 (two uniform velocities, px/day), alpha (init 0.25) fitted
    through the integrator with (i) the band-passed L1 loss (24-150 km) and (ii) pixel L1, 300 Adam steps.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import tracer as TR

ROOT = Path(__file__).resolve().parents[1]


def smooth_field(H, W, dev, seed, scale=12.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    f = torch.zeros(H, W)
    for _ in range(6):
        kx, ky, ph = (torch.rand(3, generator=g) * torch.tensor([2 * math.pi / scale, 2 * math.pi / scale, 2 * math.pi]))
        f += torch.sin(kx * xx + ky * yy + ph)
    return f.to(dev)


def main():
    dev = torch.device("cuda")
    H, W = 120, 160
    res = {}
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=50.0)
    ok = torch.ones(1, 6, H, W, device=dev)
    inner = torch.zeros(H, W, dtype=torch.bool, device=dev); inner[20:-20, 20:-20] = True

    # 1 uniform constant
    u0, v0 = 2.3, -1.7
    U = torch.zeros(1, 6, 2, H, W, device=dev); U[:, :, 0] = u0; U[:, :, 1] = v0
    f = smooth_field(H, W, dev, 0)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev).float(), torch.arange(W, device=dev).float(), indexing="ij")
    r1 = []
    for k in range(1, 6):
        pos, val = it.departure(U, ok, k)
        disp = (pos[0] - it.base)[inner]
        pred, pv = it.depart_sample(f[None], torch.ones(1, H, W, dtype=torch.bool, device=dev), pos)
        g = torch.Generator(device="cpu").manual_seed(0)
        # analytic: same sinusoids at (x - k u, y - k v)
        fa = smooth_field_shift(H, W, dev, 0, -k * u0, -k * v0)
        r1.append(dict(k=k, disp_err_max=float((disp - torch.tensor([-k * u0, -k * v0], device=dev)).abs().max()),
                       field_err_mae=float((pred[0] - fa)[inner].abs().mean()), field_sd=float(f[inner].std()), valid_inner=float(val[0][inner].float().mean())))
    res["uniform_constant"] = r1

    # 2 uniform, linear in time
    U = torch.zeros(1, 6, 2, H, W, device=dev)
    for j in range(6):
        U[:, j, 0] = 0.5 * j
    r2 = []
    for k in range(1, 6):
        pos, _ = it.departure(U, ok, k)
        dx = float((pos[0] - it.base)[inner][:, 0].mean())
        r2.append(dict(k=k, disp_x=dx, analytic=-0.25 * k * k, err=abs(dx + 0.25 * k * k)))
    res["uniform_linear_in_time"] = r2

    # 3 solid-body rotation
    om, xc, yc = 0.05, W / 2, H / 2
    U = torch.zeros(1, 6, 2, H, W, device=dev); U[:, :, 0] = -om * (yy - yc); U[:, :, 1] = om * (xx - xc)
    r3 = []
    for ns in (1, 4):
        itn = TR.Integrator(H, W, dev, n_sub=ns, vmax=50.0)
        for k in (1, 5):
            pos, val = itn.departure(U, ok, k)
            th = -om * k
            ex = xc + (xx - xc) * math.cos(th) - (yy - yc) * math.sin(th)
            ey = yc + (xx - xc) * math.sin(th) + (yy - yc) * math.cos(th)
            m = inner & val[0]
            err = torch.hypot(pos[0, ..., 0] - ex, pos[0, ..., 1] - ey)[m]
            r3.append(dict(n_sub=ns, k=k, err_max_px=float(err.max()), err_mean_px=float(err.mean()), max_disp_px=float(torch.hypot(ex - xx, ey - yy)[m].max())))
    res["solid_rotation"] = r3

    # 4 recovery of a factor on a space-varying velocity field
    torch.manual_seed(0)
    B = 4
    ug = torch.stack([smooth_field(H, W, dev, 10 + i, 30.0) * 1.0 for i in range(2 * B)]).view(B, 2, H, W)      # px/day
    C0 = torch.stack([smooth_field(H, W, dev, 100 + i, 10.0) for i in range(B)])
    ok2 = torch.ones(B, 2, H, W, device=dev)
    it8 = TR.Integrator(H, W, dev, n_sub=8, vmax=50.0)
    Ut = ug[:, None].expand(-1, 2, -1, -1, -1)
    pos, val = it8.departure(Ut, ok2, 1)
    obs = torch.ones(B, H, W, dtype=torch.bool, device=dev)
    target, tv = it8.depart_sample(C0, obs, pos)
    m = val & tv & inner[None]
    vm = TR.VelocityModel(["rea"], n_static=1).to(dev)
    opt = torch.optim.Adam(vm.parameters(), lr=3e-2)
    static = torch.zeros(1, H, W, device=dev); ocf = torch.ones(H, W, device=dev)
    zero = torch.zeros(B, H, W, device=dev)
    for step in range(300):
        ue, ve = vm(static, torch.zeros(B, device=dev), torch.ones(B, device=dev), ug[:, 0], ug[:, 1], zero, zero, zero, zero, ocf)
        Um = torch.stack([ue, ve], 1)[:, None].expand(-1, 2, -1, -1, -1)
        p, v = it.departure(Um, ok2, 1)
        pr, pv = it.depart_sample(C0, obs, p)
        mm = m & v & pv
        loss = (pr - target).abs()[mm].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        col = {}
        vm(static, torch.zeros(B, device=dev), torch.ones(B, device=dev), ug[:, 0], ug[:, 1], zero, zero, zero, zero, ocf, collect=col)
    res["recover_factor"] = dict(true=1.0, init=0.25, learned_mean=float(col["a"].mean()), learned_sd=float(col["a"].std()), final_l1=float(loss))

    # 5 recover a uniform velocity from a shifted field
    C0 = smooth_field(H, W, dev, 7, 10.0)[None]
    Ut = torch.zeros(1, 2, 2, H, W, device=dev); Ut[:, :, 0] = 3.0; Ut[:, :, 1] = 2.0
    pos, val = it.departure(Ut, ok2[:1], 1)
    target, tv = it.depart_sample(C0, obs[:1], pos)
    uv = torch.zeros(2, device=dev, requires_grad=True)
    opt = torch.optim.Adam([uv], lr=0.1)
    for step in range(300):
        Um = uv.view(1, 1, 2, 1, 1).expand(1, 2, 2, H, W)
        p, v = it.departure(Um, ok2[:1], 1)
        pr, pv = it.depart_sample(C0, obs[:1], p)
        loss = (pr - target).abs()[val & tv & v & pv & inner].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    res["recover_uniform_velocity"] = dict(true_px_per_day=[3.0, 2.0], learned=[float(uv[0]), float(uv[1])], final_l1=float(loss))

    # 6 path recording does not change the trajectory
    U = torch.zeros(1, 6, 2, H, W, device=dev); U[:, :, 0] = -om * (yy - yc); U[:, :, 1] = om * (xx - xc)
    p0, v0_ = it.departure(U, ok, 5); path = []
    p1, v1_ = it.departure(U, ok, 5, path=path)
    res["path_recording"] = dict(identical=bool(torch.equal(p0, p1) and torch.equal(v0_, v1_)), n_mid=len(path), expected=5 * it.n_sub)

    # 7 source integration with a constant source
    sm = TR.SourceModel(5, 8, g_max=0.5).to(dev)
    x = torch.randn(2, 5, H, W, device=dev)
    s0 = float(sm(x).abs().max())
    with torch.no_grad():
        sm.net[-1].bias.fill_(math.atanh(0.2 / 0.5))
    c = torch.zeros(2, H, W, device=dev); h = 1.0 / it.n_sub
    for _ in range(3 * it.n_sub):
        c = c + h * sm(x)
    res["source_constant"] = dict(init_abs_max=s0, s=0.2, k=3, accumulated_mean=float(c.mean()), expected=0.6, err_max=float((c - 0.6).abs().max()))

    out = ROOT / "analyses" / "tracer" / "results"; out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def smooth_field_shift(H, W, dev, seed, sx, sy, scale=12.0):
    """smooth_field evaluated at (x + sx, y + sy)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    f = torch.zeros(H, W)
    for _ in range(6):
        kx, ky, ph = (torch.rand(3, generator=g) * torch.tensor([2 * math.pi / scale, 2 * math.pi / scale, 2 * math.pi]))
        f += torch.sin(kx * (xx + sx) + ky * (yy + sy) + ph)
    return f.to(dev)


def blobs(H, W, frac, seed, smooth=3.0):
    from scipy.ndimage import gaussian_filter
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), smooth)
    return n > np.quantile(n, frac)                       # True = observed


def step1b(dev):
    res = {}
    H = W = 256
    yy, xx = np.meshgrid(np.arange(H, dtype="float64"), np.arange(W, dtype="float64"), indexing="ij")
    obs = blobs(H, W, 0.3, 1); land = np.zeros((H, W), bool); land[100:150, 0:60] = True
    masks = {"clear": np.ones((H, W), bool), "clouds30_land": obs & ~land}
    inner = np.zeros((H, W), bool); inner[20:-20, 20:-20] = True
    wls = np.array([9, 12, 16, 20, 24, 30, 40, 50, 70, 100, 150, 200, 300, 400.0])
    curves = {}
    for lo, hi in ((24, 150), (20, 70)):
        bp = TR.BandPass(lo, hi, device=dev)
        rr = dict(sigma_px=[bp.s1, bp.s2], analytic=bp.response(wls).tolist())
        for mn, m in masks.items():
            meas = []
            for wl in wls:
                amps = []
                for th in np.arange(8) * np.pi / 8:
                    ph = 2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / (wl / 4.5)
                    f = torch.as_tensor(np.cos(ph), dtype=torch.float32, device=dev)[None]
                    out, val = bp(f, torch.as_tensor(m, device=dev)[None])
                    v = val[0].cpu().numpy() & inner
                    A = np.stack([np.cos(ph)[v], np.sin(ph)[v]], 1)
                    coef = np.linalg.lstsq(A, out[0].cpu().numpy()[v].astype("float64"), rcond=None)[0]
                    amps.append(coef[0])
                meas.append(float(np.mean(amps)))
            rr[mn] = dict(measured=meas, max_abs_dev_from_analytic=float(np.max(np.abs(np.array(meas) - bp.response(wls)))),
                          valid_frac_of_observed=float(val[0].cpu().numpy()[m & inner].mean()))
        fine = np.geomspace(5, 1000, 200)
        curves[f"{lo}_{hi}"] = bp.response(fine); curves["fine_km"] = fine
        curves[f"{lo}_{hi}_meas_clear"] = np.array(rr["clear"]["measured"]); curves[f"{lo}_{hi}_meas_cloud"] = np.array(rr["clouds30_land"]["measured"])
        # B2 mask edges
        m = torch.as_tensor(masks["clouds30_land"], device=dev)[None]
        e = {}
        for fn, f in (("constant", np.full((H, W), 1.7)), ("ramp", 0.5 * xx / 100 + 0.3 * yy / 100)):
            ft = torch.as_tensor(f, dtype=torch.float32, device=dev)[None]
            out, val = bp(ft, m)
            # naive DoG: zero-filled, no normalisation
            st = torch.where(m, ft, torch.zeros_like(ft))[:, None]
            naive = (TR.BandPass._blur(st, bp.k1) - TR.BandPass._blur(st, bp.k2))[:, 0]
            v = val & torch.as_tensor(inner, device=dev)[None]
            e[fn] = dict(normconv_max_abs=float(out[v].abs().max()), normconv_mean_abs=float(out[v].abs().mean()),
                         naive_dog_max_abs=float(naive[v].abs().max()), naive_dog_mean_abs=float(naive[v].abs().mean()), n_valid=int(v.sum()))
        rr["mask_edges"] = e
        res[f"bandpass_{lo}_{hi}"] = rr
    curves["wavelength_km"] = wls
    np.savez(ROOT / "analyses/tracer/results/bandpass_response.npz", **curves)

    # B3 decomposition
    g = torch.Generator(device="cpu").manual_seed(0)
    N, h, w = 6, 40, 50
    u, v, ug, vg = (torch.randn(N, h, w, generator=g) * 0.2 for _ in range(4))
    ok = (torch.rand(N, h, w, generator=g) > 0.1).float()
    gok = torch.rand(N, h, w, generator=g) > 0.2
    ug, vg = ug * gok * ok, vg * gok * ok; u, v = u * ok, v * ok
    clim_day = torch.randn(N, 4, h, w, generator=g) * 0.1
    comps = TR.decomp_components(u, v, ug, vg, ok, clim_day)
    tot = comps.sum(1)
    res["decomp_sum"] = dict(max_abs_err_u=float((tot[:, 0] - u * ok).abs().max()), max_abs_err_v=float((tot[:, 1] - v * ok).abs().max()),
                             zero_where_invalid=float(comps.abs().sum((1, 2))[ok == 0].max()))
    # climatology from train days only
    T_ = 3 * 366; doy = np.arange(T_) % 366
    truth = 0.3 * np.sin(2 * np.pi * doy / 366)[:, None, None] * np.ones((1, 4, 5))
    rng = np.random.default_rng(0)
    U_ = (truth + 0.05 * rng.standard_normal(truth.shape)).astype("float32")
    tr = np.arange(0, 2 * 366); va = np.arange(2 * 366, T_)
    U_[va] = 1e3
    Z = np.zeros_like(U_); OK_ = np.ones(U_.shape, bool)
    clim, cnt = TR.decomp_clim(U_, Z, Z, Z, OK_, doy, tr, sigma_days=15)
    res["decomp_clim"] = dict(max_abs_err_ageo_u=float(np.abs(clim[:, 2] - truth[:366]).max()), note="truth amplitude 0.3, noise sd 0.05, 2 train years, val days = 1e3 excluded",
                              geo_u_max_abs=float(np.abs(clim[:, 0]).max()))
    reg = torch.randint(0, 3, (h, w), generator=g); season = torch.tensor([0, 1, 2, 3, 0, 2])
    dm = TR.DecompVelocity(4, 3)
    ue, ve = dm(comps, reg, season, torch.zeros(N), torch.ones(N), None, None, torch.ones(h, w))
    res["decomp_alpha_init"] = dict(max_abs_err_vs_0p25_u_rea=float(torch.maximum((ue - 0.25 * u * ok).abs().max(), (ve - 0.25 * v * ok).abs().max())))
    with torch.no_grad():
        dm.alpha.copy_(torch.arange(4 * 3 * 4, dtype=torch.float32).view(4, 3, 4))
    al = dm.alpha_field(reg, season, torch.zeros(N), torch.ones(N))
    exp = torch.stack([torch.stack([dm.alpha[c, reg, season[n]] for c in range(4)]) for n in range(N)])
    res["decomp_alpha_gather"] = dict(exact=bool(torch.equal(al, exp)))

    # B4 recovery through the integrator
    Hs = Ws = 96
    it = TR.Integrator(Hs, Ws, dev, n_sub=4, vmax=50.0)
    B = 4
    C0 = torch.stack([smooth_field(Hs, Ws, dev, 200 + i, 14.0) for i in range(B)])
    cm = torch.stack([torch.as_tensor(blobs(Hs, Ws, 0.2, 50 + i), device=dev) for i in range(2 * B)]).view(B, 2, Hs, Ws)
    u1 = torch.tensor([1.5, 0.5], device=dev); u2 = torch.tensor([-0.4, 1.0], device=dev)
    true = torch.tensor([0.6, 1.3], device=dev)
    U = (true[0] * u1 + true[1] * u2).view(1, 1, 2, 1, 1).expand(B, 2, 2, Hs, Ws)
    okk = torch.ones(B, 2, Hs, Ws, device=dev)
    pos, val = it.departure(U, okk, 1)
    tgt, tv = it.depart_sample(C0, cm[:, 0], pos)
    tobs = cm[:, 1] & val & tv
    bpf = TR.BandPass(24, 150, device=dev)
    for nm in ("band", "pixel"):
        al = torch.full((2,), 0.25, device=dev, requires_grad=True)
        opt = torch.optim.Adam([al], lr=0.03)
        for step in range(300):
            Um = (al[0] * u1 + al[1] * u2).view(1, 1, 2, 1, 1).expand(B, 2, 2, Hs, Ws)
            p, vv = it.departure(Um, okk, 1)
            pr, pv = it.depart_sample(C0, cm[:, 0], p)
            m = tobs & vv & pv
            if nm == "band":
                a1, vb = bpf(pr, m); a2, _ = bpf(tgt, m); loss = (a1 - a2).abs()[vb].mean()
            else:
                loss = (pr - tgt).abs()[m].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        res[f"recover_alpha_{nm}"] = dict(true=true.tolist(), init=0.25, learned=al.detach().cpu().tolist(), final_loss=float(loss))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--step1b", action="store_true"); ap.add_argument("--cpu", action="store_true")
    aa = ap.parse_args()
    if aa.step1b:
        dev = torch.device("cpu" if aa.cpu else "cuda")
        r = step1b(dev)
        (ROOT / "analyses/tracer/results/step1b_checks.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
    else:
        main()
