"""CPU unit tests of the io8 / blk8 code (chl_cnn/rwloss.py, chl_cnn/bsmem.py, --ptok-patch 12). python -m analyses.io8.test_io8 [--no-store]
Exits non-zero on a failure.

rwloss
 1. Binner cells == the postrun definition computed independently on the CPU (scipy distance_transform_edt bins and analyses/io7/keys.n_keys_days
    key counts with the persist rule) on synthetic days with store gaps, every non-visible px of every day with >= 1 visible px. Days without a
    visible px: scipy's EDT returns distances to a phantom corner (0, 0) - 1; the binner gives 32+ (none such among the IO val / test days).
 2. cap_weights: mean 1 under p_fake, max <= cap; without an active cap the reweighted fake distribution equals p_real; zero-fake cells get 0.
bsmem
 3. summaries are past-only (physics of days >= t changes nothing for day t; day t-1 does), day 0 = 0 with coverage 1, gap coverage, land 0.
 4. PhysMemTokDecNet(mem_ch 12) on t-8..t+8 with 8 features == TokDecNet(blend, spatial 2) with the same weights (memory zero-init).
 5. (store) mouth px / mean discharge of Danube, Dniester, Dnieper-Bug on the cropped blk4 grid; daily fields of a 10-day slice == the same
    days of the full computation; real-data summaries past-only (daily fields of days >= t perturbed, statistics from days < t).
patch 12
 6. TokDecNet(blend, spatial 2, P 12) on the cropped blk4 grid size (177 x 273), 8 features: forward / backward, 15 x 23 patch grid.
"""
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from chl_cnn import rwloss as RWL, bsmem as BSM, physmem as PM, train2 as T2
from chl_cnn.tokdec import TokDecNet
from analyses.io7 import keys as KEYS
from analyses.io7.postrun import DIST_RW, KEY_RW

FAIL = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    if not ok:
        FAIL.append(name)


def blobs(rng, n, H, W, p):
    """n random cloud-like masks (True = observed) with clear fraction ~p."""
    from scipy.ndimage import gaussian_filter
    z = np.stack([gaussian_filter(rng.normal(size=(H, W)), rng.uniform(1, 6)) for _ in range(n)])
    return z > np.quantile(z.reshape(n, -1), 1 - p, axis=1)[:, None, None]


def test_binner():
    rng = np.random.default_rng(0)
    H, W, n = 60, 75, 40
    times = np.arange(np.datetime64("2001-01-01"), np.datetime64("2001-01-01") + n + 3).astype("datetime64[ns]")
    times = np.delete(times, [10, 25, 26])                    # store gaps
    ocean = np.ones((H, W), bool); ocean[:12, :20] = False; ocean[-8:, 30:45] = False
    obs = blobs(rng, n, H, W, 0.45) & ocean
    obs[7] = False; obs[8, :, :] = False; obs[8, 30, 40] = True   # an empty and a near-empty day
    idx = np.arange(4, n - 4)
    donor = rng.choice(np.arange(n), size=len(idx))
    hid = obs[idx] & ~obs[donor]
    # reference (postrun / keys.py)
    nk = KEYS.n_keys_days(obs, times, idx, hid, donor, persist=True)
    ref = np.full(hid.shape, -1, int)
    for i in range(len(idx)):
        d = distance_transform_edt(~(obs[idx[i]] & ~hid[i]))
        di = np.full(d.shape, -1); ki = np.full(d.shape, -1)
        for b, (_, lo, hi) in enumerate(DIST_RW):
            di[(d > lo) & (d <= hi)] = b
        for b, (_, lo, hi) in enumerate(KEY_RW):
            ki[(nk[i] >= lo) & (nk[i] <= hi)] = b
        ref[i] = np.where((di >= 0) & (ki >= 0), di * 5 + ki, -1)
    # Binner with train_pinn.nb_masks-style context masks
    OBS = torch.as_tensor(obs); bi = torch.as_tensor(idx); DON = torch.as_tensor(donor)
    ctx = {}
    for k in range(-4, 5):
        if k == 0:
            continue
        nb = torch.as_tensor(T2.neighbour_index(times, k))
        j = nb[bi]; ok = (j >= 0).view(-1, 1, 1); j = j.clamp(min=0)
        jd = nb[DON]; jd = torch.where(jd >= 0, jd, DON)
        ctx[k] = OBS[j] & ok & OBS[jd]
    cell = RWL.Binner(torch.device("cpu")).cells(OBS[bi] & ~torch.as_tensor(hid), ctx).numpy()
    vis = obs[idx] & ~hid
    empty = ~vis.any((1, 2))                                  # no visible px: scipy's EDT measures to a phantom corner, the binner says 32+
    nonvis = ~vis & ~empty[:, None, None]
    check(f"binner cells == postrun EDT / keys.py cells on every non-visible px of days with a visible px ({int(nonvis.sum())} px, "
          f"{int((~empty).sum())} days)", np.array_equal(cell[nonvis], ref[nonvis]) and (ref[nonvis] >= 0).all())
    check(f"binner: days without a visible px ({int(empty.sum())}) -> distance bin 32+ everywhere", empty.sum() >= 1 and (cell[empty] // 5 == 5).all())
    check("binner: cells used in both classes (>= 12 distinct)", len(np.unique(cell[nonvis])) >= 12)


def test_weights():
    rng = np.random.default_rng(1)
    fake = rng.integers(1, 1000, 30).astype(float); real = rng.integers(1, 1000, 30).astype(float)
    w, rep = RWL.cap_weights(real, fake, 1e9)
    pf, pr = fake / fake.sum(), real / real.sum()
    check("no cap: reweighted fake == real", np.allclose(pf * w, pr))
    real2 = real.copy(); real2[3] = 50000; fake2 = fake.copy(); fake2[5] = 0
    w, rep = RWL.cap_weights(real2, fake2, 10.0)
    check("cap 10: mean 1 under p_fake, max <= cap, zero-fake cell weight 0", abs((fake2 / fake2.sum() * w).sum() - 1) < 1e-9 and w.max() <= 10 * (1 + 1e-9)
          and w[5] == 0 and rep["n_cells_at_cap"] >= 1 and rep["uncovered_real_mass"] > 0)


def test_bsmem_summaries():
    times = np.arange(np.datetime64("2014-01-01"), np.datetime64("2014-08-01")).astype("datetime64[ns]")
    times = np.delete(times, [40, 41, 100])
    n = len(times); rng = np.random.default_rng(0)
    Q = np.exp(rng.normal(size=(n, 3)) + 12).astype("float32"); Q[5, 0] = np.nan
    mld = (rng.normal(size=(n, 3, 4)) + 30).astype("float32"); swr = (rng.normal(size=(n, 3, 4)) * 20 + 150).astype("float32")
    prox = rng.uniform(size=(3, 3, 4)).astype("float32")
    pok = np.ones((3, 4), bool); pok[0, 3] = False
    s0, _ = BSM.summaries(Q, mld, swr, prox, times, np.arange(n), pok)
    t = 150; trp = np.arange(t)
    Q2, m2, w2 = Q.copy(), mld.copy(), swr.copy()
    Q2[t:] *= 3; m2[t:] += 10; w2[t:] -= 50
    a0, _ = BSM.summaries(Q, mld, swr, prox, times, trp, pok); a2, _ = BSM.summaries(Q2, m2, w2, prox, times, trp, pok)
    check("bsmem summaries of days <= t unchanged by physics of days >= t", np.array_equal(a0[:t + 1], a2[:t + 1]))
    trp = np.arange(t - 1)
    Q3, m3, w3 = Q.copy(), mld.copy(), swr.copy(); Q3[t - 1] *= 5; m3[t - 1] += 20; w3[t - 1] += 100
    a0, _ = BSM.summaries(Q, mld, swr, prox, times, trp, pok); a3, _ = BSM.summaries(Q3, m3, w3, prox, times, trp, pok)
    ch = [k for k in range(10) if not np.array_equal(a0[t, k], a3[t, k])]
    check(f"bsmem summaries of day t change with day t-1 physics (all 10 physics channels: changed {ch})", ch == list(range(10))
          and np.array_equal(a0[:t], a3[:t]))
    check("bsmem day 0: physics channels 0, coverage 1", np.all(s0[0, :10] == 0) and np.all(s0[0, 10:, pok] == 1))
    check("bsmem coverage: day 45 W30 = 2/30 missing, day 70 W60 = 2/60", abs(float(s0[45, 10, 1, 1]) - 2 / 30) < 1e-3
          and abs(float(s0[70, 11, 1, 1]) - 2 / 60) < 1e-3)
    check("bsmem land patch 0", np.all(s0[:, :, 0, 3] == 0))
    check("bsmem river channel = z x proximity (ratio of two patches = prox ratio)",
          np.allclose(s0[100, 0, 1, 1] * prox[0, 2, 2], s0[100, 0, 2, 2] * prox[0, 1, 1], rtol=2e-3))


KW = dict(dim=32, ch=8, heads=4, layers=2, sst_index=0, backbone="dilated", width=16, spatial=2, out_channels=3, dilations=[1, 2])


def test_bsmem_net():
    g = torch.Generator().manual_seed(0)
    offs = list(range(-8, 9)); T = len(offs); H, W, F, B = 22, 26, 8, 2
    ocean = torch.ones(H, W, dtype=torch.bool); ocean[:5, :7] = False
    x3d = torch.randn(B, F + 2, T, H, W, generator=g); vis = torch.rand(B, T, H, W, generator=g) > 0.4
    x3d[:, -1] = vis.float(); x3d[:, -2] *= vis
    xs, clim = torch.randn(B, 6, H, W, generator=g), torch.randn(B, T, H, W, generator=g)
    day_ok = torch.ones(B, T, dtype=torch.bool)
    ref = TokDecNet(F + 2, 6, T, 8, ocean, blend=True, P=4, **KW).eval()
    net = PM.PhysMemTokDecNet(F + 2, 6, offs, ocean, blend_k=8, mem_ch=BSM.N_MEM, P=4, **KW).eval()
    miss = net.load_state_dict(ref.state_dict(), strict=False)
    net.mem_in = torch.randn(B, BSM.N_MEM, net.nPy, net.nPx)
    with torch.no_grad():
        check("bsmem net (mem_ch 12, 8 features, +-8) == TokDecNet(blend, spatial 2) at init",
              all(k.startswith("mem.") for k in miss.missing_keys) and torch.equal(ref(x3d, xs, day_ok, clim), net(x3d, xs, day_ok, clim)))


def test_p12():
    H, W, F, T, B = 177, 273, 8, 17, 1
    g = torch.Generator().manual_seed(0)
    ocean = torch.rand(H, W, generator=g) > 0.2; ocean[:30, :60] = False
    x3d = torch.randn(B, F + 2, T, H, W, generator=g); xs = torch.randn(B, 6, H, W, generator=g); clim = torch.randn(B, T, H, W, generator=g)
    net = TokDecNet(F + 2, 6, T, 8, ocean, blend=True, P=12, **KW)
    y = net(x3d, xs, torch.ones(B, T, dtype=torch.bool), clim)
    y.square().mean().backward()
    check(f"P 12 on 177 x 273: patch grid {net.nPy} x {net.nPx}, output {tuple(y.shape)}, grads finite",
          (net.nPy, net.nPx) == (15, 23) and tuple(y.shape) == (B, 3, H, W) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))


def test_store():
    from analyses.blk7 import storectx
    from chl_cnn import store_blk4 as SB4
    import json
    d = storectx.load_raw("blk4")
    oc = np.asarray(d["ocean"]); rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
    ys, xs = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])), slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
    rd = lambda name, t=slice(None): np.asarray(d[name][t, ys, xs])
    ocean, lat, lon = oc[ys, xs], np.asarray(d["lat"][ys], "float64"), np.asarray(d["lon"][xs], "float64")
    rdist = np.asarray(d["river_dist"])[:, ys, xs]
    names = json.loads((SB4.store_path() / "river_names.json").read_text())
    times = d["time"]; n = len(times)
    Q, mld, swr, prox, rep = BSM.daily_fields(rd, n, ocean, lat, lon, 4, rdist, names, log=lambda s: None)
    print("  mouths / discharge:", {r: rep[r] for r in BSM.RIVERS}, {k: round(v) for k, v in rep.items() if k.startswith("Q_mean")},
          {k: v for k, v in rep.items() if "nan" in k}, flush=True)
    check("store: mouth px Danube >= 3, Dniester >= 1, Dnieper-Bug >= 1, mean Q Danube > Dnieper-Bug > Dniester > 0",
          rep["Danube"]["n_px"] >= 3 and rep["Dniester"]["n_px"] >= 1 and rep["Dnieper-Bug"]["n_px"] >= 1
          and rep["Q_mean_kg_s_Danube"] > rep["Q_mean_kg_s_Dnieper-Bug"] > rep["Q_mean_kg_s_Dniester"] > 0)
    sl = slice(500, 510)
    rds = lambda name, t=slice(None): rd(name, slice(500, 510))[t]
    BSM.CACHE, cache0 = BSM.CACHE / "_test_tmp", BSM.CACHE
    try:
        Qs, ms, ws, ps, _ = BSM.daily_fields(rds, 10, ocean, lat, lon, 4, rdist, names, log=lambda s: None)
    finally:
        import shutil
        shutil.rmtree(BSM.CACHE, ignore_errors=True); BSM.CACHE = cache0
    # mouth px from a 10-day slice could differ only if a mouth has friver 0 on all 10 days; compare Q where both define the same px
    check("store: daily fields of days 500-509 alone == the full computation", np.allclose(Qs, Q[sl], rtol=1e-5, equal_nan=True)
          and np.array_equal(ms, mld[sl], equal_nan=True) and np.array_equal(ws, swr[sl], equal_nan=True) and np.array_equal(ps, prox))
    tr = np.flatnonzero(times < np.datetime64("2016-06-01")); t = int(tr[-1]) + 1
    P = 4; H, W = ocean.shape
    ocp = np.pad(ocean, ((0, -(-H // P) * P - H), (0, -(-W // P) * P - W)))
    pok = ocp.reshape(ocp.shape[0] // P, P, ocp.shape[1] // P, P).any((1, 3))
    a0, _ = BSM.summaries(Q, mld, swr, prox, times, tr, pok)
    Q2, m2, w2 = Q.copy(), mld.copy(), swr.copy(); Q2[t:] *= 2; m2[t:] += 15; w2[t:] += 80
    a2, _ = BSM.summaries(Q2, m2, w2, prox, times, tr, pok)
    check(f"store: real-data summaries of days <= t ({str(times[t])[:10]}) unchanged by physics of days >= t", np.array_equal(a0[:t + 1], a2[:t + 1])
          and not np.array_equal(a0[t + 1:], a2[t + 1:]))
    v = {k: float((a0[tr][:, k][:, pok] != 0).mean()) for k in range(BSM.N_MEM)}
    print("  nonzero frac (train days, ocean patches):", {BSM.NAMES[k]: round(x, 3) for k, x in v.items()}, flush=True)


def main():
    test_binner(); test_weights(); test_bsmem_summaries(); test_bsmem_net(); test_p12()
    if "--no-store" not in sys.argv:
        test_store()
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
