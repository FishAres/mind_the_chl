"""CPU unit tests of chl_cnn/wmattn.py (tiny shapes). Usage (project root): python -m analyses.io7.wmattn.test_wmattn
  1 key pool: never a non-visible candidate (day t hidden px, donor-hidden context px, day_ok False days, land), over many train-mode draws;
    with fewer visible candidates than K the extra slots are invalid
  2 no leak: garbage CHL values on every non-visible px (all days) leave eval outputs and the pool bit-identical; positive control changes them
  3 eval determinism: two forwards equal; per-sample results independent of batch order; train-mode draws differ
  4 exactness: estimate / spread / mass / entropy vs an explicit softmax over [no key] + the pool (augmented-dimension rewrite)
  5 chunked queries == fused call; forward / backward finite, every parameter gets a gradient
  6 stratified draw: roughly equal keys per (day, SST bin) stratum
  7 WmTokDecNet builds with the blend on and runs forward / backward
"""
import math
import sys

import torch
import torch.nn.functional as F

from chl_cnn import wmattn as WM

torch.manual_seed(0)
H, W, T, TI, NF, NS = 12, 14, 9, 4, 9, 3
ocean = torch.ones(H, W, dtype=torch.bool); ocean[:3, :4] = False; ocean[-2:, -3:] = False
FAILS = []


def check(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def make(B=3, vis_p=0.5, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, NF + 2, T, H, W, generator=g)
    vis = (torch.rand(B, T, H, W, generator=g) < vis_p) & ocean
    x[:, -1] = vis.float()
    x[:, -2] = torch.where(vis, x[:, -2], torch.zeros_like(x[:, -2]))
    xs = torch.randn(B, NS, H, W, generator=g)
    dok = torch.ones(B, T, dtype=torch.bool); dok[0, TI + 1] = False
    clim = torch.randn(B, T, H, W, generator=g)
    return x, xs, dok, clim


def net(K=64, qchunk=0):
    return WM.WaterMassAttn(NF, NS, T, TI, ocean, 0, 1, 6, 5, keys=K, lags=2, dim=16, heads=4, dv=8, sst_bins=4, qchunk=qchunk)


def randomise(m):
    with torch.no_grad():
        for p in (m.alpha, m.alpha_pos, m.beta, m.none_logit):
            p.normal_(0, 0.5)


def visible(x, dok):
    S = slice(TI - 2, TI + 3)
    return (x[:, -1, S] > 0.5) & dok[:, S].view(-1, 5, 1, 1) & ocean                    # (B, 5, H, W)


# 1 key pool
m = net(K=64); randomise(m); m.train()
x, xs, dok, clim = make()
vis = visible(x, dok).flatten(2)[..., m.oc_idx].reshape(x.shape[0], -1)
bad = 0
for i in range(50):
    _, rec = m(x, xs, dok, clim)
    p = rec["wm_pool"].long()
    ok = p >= 0
    bad += int((~vis.gather(1, p.clamp(min=0))[ok]).sum())
check("1a pool holds only visible candidates (50 train draws)", bad == 0, f"bad {bad}")
check("1b day_ok False day never in the pool", not bool(((rec["wm_pool"][0] // m.N) == 3).any()))
x2 = x.clone(); x2[:, -1] = 0; x2[:, -1, TI, 5, 5] = 1; x2[:, -1, TI + 2, 6, 7] = 1
_, r2 = m(x2, xs, dok, clim)
check("1c 2 visible px -> 2 valid keys, rest invalid", r2["wm_nvalid"].tolist() == [2.0] * 3 and int((r2["wm_pool"] >= 0).sum()) == 6)

# 2 no leak
m.eval()
o0, r0 = m(x, xs, dok, clim)
xg = x.clone()
hidden = x[:, -1] < 0.5
xg[:, -2] = torch.where(hidden, torch.randn_like(xg[:, -2]) * 100, xg[:, -2])         # garbage CHL on every non-visible px
xg[:, -2, 0, :, TI + 1] += 50                                                          # and on the day_ok False day of sample 0
o1, r1 = m(xg, xs, dok, clim)
check("2a non-visible CHL values change nothing (outputs)", torch.equal(o0, o1), f"max {float((o0 - o1).abs().max()):.3g}")
check("2b ... nor the pool", torch.equal(r0["wm_pool"], r1["wm_pool"]))
xp = x.clone(); xp[:, -2, TI + 2] += torch.where(x[:, -1, TI + 2] > 0.5, 1.0, 0.0)
o2, r2 = m(xp, xs, dok, clim)
check("2c positive control (visible t+2 values) changes outputs, pool fixed", (o2 - o0).abs().max() > 0 and torch.equal(r2["wm_pool"], r0["wm_pool"]))

# 3 eval determinism
o0b, _ = m(x, xs, dok, clim)
check("3a eval forward reproducible", torch.equal(o0, o0b))
perm = torch.tensor([2, 0, 1])
op, rp = m(x[perm], xs[perm], dok[perm], clim[perm])
check("3b per-sample result independent of batch order", torch.allclose(op, o0[perm], atol=1e-5) and torch.equal(rp["wm_pool"], r0["wm_pool"][perm]),
      f"max {float((op - o0[perm]).abs().max()):.3g}")
m.train(); _, ra = m(x, xs, dok, clim); _, rb = m(x, xs, dok, clim); m.eval()
check("3c train-mode draws differ", not torch.equal(ra["wm_pool"], rb["wm_pool"]))

# 4 exactness vs explicit softmax (eval, same pool)
with torch.no_grad():
    cap = {}
    orig = m._sdpa
    def spy(Q, Kp, V):
        cap.update(Q=Q, K=Kp, V=V); return orig(Q, Kp, V)
    m._sdpa = spy
    out, rec = m(x, xs, dok, clim)
    m._sdpa = orig
    Q, Kp, V = cap["Q"], cap["K"], cap["V"]
    s = Q @ Kp.transpose(-1, -2)                                                          # (B, h, N, K + 1)
    w = torch.softmax(s, -1)
    dv = m.dv
    a = V[..., :-1, dv]
    real = V[..., :-1, dv + 2] > 0.5
    w_real = w[..., :-1] * real.unsqueeze(2)
    est = (w_real * a.unsqueeze(2)).sum(-1).mean(1)
    mass = w_real.sum(-1).mean(1)
    spread = torch.sqrt(((w_real * (a.unsqueeze(2) - est.unsqueeze(1).unsqueeze(-1)) ** 2).sum(-1)).mean(1) + 1e-6)
    ent = (-(w * torch.log(w.clamp(min=1e-38))).sum(-1)).mean(1)
    o = out.flatten(2)[..., m.oc_idx]
    errs = [float((o[:, dv + i] - ref).abs().max()) for i, ref in enumerate([est, spread, mass, ent])]
    sn = s[..., -1]
    check("4a no-key score = none_logit", torch.allclose(sn, m.none_logit.view(1, -1, 1).expand_as(sn)))
    check("4b estimate / spread / mass / entropy match the explicit softmax", max(errs) < 1e-4, f"max errs {['%.2g' % e for e in errs]}")
    inv = ~(V[..., :-1, dv + 2] > 0.5)
    check("4c invalid slots get weight exactly 0", float(w[..., :-1].masked_select(inv.unsqueeze(2).expand_as(w[..., :-1])).abs().max()
                                                       if inv.any() else 0.0) == 0.0)
    # 4d the stationary kernel: alpha . cos(w (x_p - x_q)) part of the score
    print("    (x2 sparse-visibility case for 4c)")
    m._sdpa = spy; m(x2, xs, dok, clim); m._sdpa = orig
    w2 = torch.softmax(cap["Q"] @ cap["K"].transpose(-1, -2), -1)
    inv2 = ~(cap["V"][..., :-1, dv + 2] > 0.5)
    check("4c' invalid slots weight 0 (2 visible px)", float(w2[..., :-1].masked_select(inv2.unsqueeze(2).expand_as(w2[..., :-1])).abs().max()) == 0.0)

# 5 chunking, gradients
mc = net(K=64, qchunk=17); mc.load_state_dict(m.state_dict()); mc.eval()
oc_, _ = mc(x, xs, dok, clim)
check("5a query chunks == one call", torch.allclose(oc_, o0, atol=1e-5), f"max {float((oc_ - o0).abs().max()):.3g}")
for mm, nm in ((m, "fused"), (mc, "chunked")):
    mm.train(); mm.zero_grad()
    out, _ = mm(x, xs, dok, clim)
    (out ** 2).mean().backward()
    nog = [n for n, p in mm.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    check(f"5b backward ({nm}): finite, nonzero grads on every parameter", torch.isfinite(out).all() and not nog, f"missing {nog}")

# 6 stratification
mb = net(K=200); mb.eval()
xb, xsb, dokb, climb = make(B=1, vis_p=0.6, seed=3)
dokb[:] = True
xb[:, 0, TI - 2:TI + 3] = torch.where(torch.rand(1, 5, H, W) < 0.8, 0.0, 3.0) + 0.1 * torch.randn(1, 5, H, W)   # skewed SST: 20 % warm
_, rb_ = mb(xb, xsb, dokb, climb)
pool = rb_["wm_pool"][0].long()
sst = xb[0, 0, TI - 2:TI + 3].flatten(1)[:, mb.oc_idx].reshape(-1)[pool]
warm = float((sst > 1.5).float().mean())
per_day = torch.bincount(pool // mb.N, minlength=5).tolist()
check("6a warm-water share of the pool above its 20 % candidate share", warm > 0.3, f"warm share {warm:.2f}, keys per day {per_day}")

# 7 full net
from chl_cnn.wmattn import build_wm_tokdec
wm_kw = dict(sst_index=0, sss_index=1, mld_index=6, sla_index=5, keys=32, lags=2, dim=16, heads=4, dv=8, sst_bins=4)
tn = build_wm_tokdec(NF + 2, NS, T, TI, ocean, wm_kw, P=4, dim=32, ch=8, heads=4, layers=1, blend=True, sst_index=0, spatial=1,
                     backbone="dilated", width=16, out_channels=3)
y = tn(x, xs, dok, clim)
y.float().pow(2).mean().backward()
check("7 WmTokDecNet (blend + spatial 1) forward / backward", y.shape[:2] == (3, 3) and tn.use_blend and torch.isfinite(y).all(), str(tuple(y.shape)))

print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
