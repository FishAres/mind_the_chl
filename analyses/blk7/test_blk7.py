"""CPU checks of the blk7 code changes (Black Sea 4 km store generalisation, 2026-10-26).

  krig_8feat      KrigStem with 8 physics features (zero-padded volume) == KrigStem with 9 where the 9th feature is 0 and the weights
                  are copied (query MLP's extra input column irrelevant): outputs equal (fp32, torch path), backward finite.
  ekman_stress    physmem.ekman_from_stress(physics.wind_stress(u, v)) == physics.ekman(u, v) w_E (same formula from stress input).
  lat_flip        physmem.daily_fields (wind as stress) on a synthetic grid with latitude ascending == the same grid stored north-to-south,
                  rows flipped back (curl, w_E, alongshore stress, SST, MLD patch means): latitude order does not change the physics.
Usage (project root): CUDA_VISIBLE_DEVICES= python -m analyses.blk7.test_blk7   (exit 1 on any failure)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from chl_cnn import krig as KR, physics as PH, physmem as PM

FAIL = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info, flush=True)
    if not ok:
        FAIL.append(name)


def krig_8feat():
    T, TI, NS, B, H, W = 9, 4, 5, 2, 18, 22
    g = torch.Generator().manual_seed(0)
    ocean = torch.rand(H, W, generator=g) > 0.25
    x8 = torch.randn(B, 8 + 2, T, H, W, generator=g)
    vis = (torch.rand(B, T, H, W, generator=g) < 0.4) & ocean
    x8[:, -1] = vis.float(); x8[:, -2] = x8[:, -2] * vis
    x9 = torch.cat([x8[:, :8], torch.zeros(B, 1, T, H, W), x8[:, 8:]], 1)
    xs = torch.randn(B, NS, H, W, generator=g)
    ok = torch.ones(B, T, dtype=torch.bool)
    cl = 0.3 * torch.randn(B, T, H, W, generator=g)
    vel = dict(mu_u=0.05, sd_u=0.3, mu_v=0.0, sd_v=0.22, dx=torch.full((H, W), 4500.0), dy=torch.full((H, W), 4500.0))
    idx = dict(sst_index=0, sss_index=1, u_index=4, v_index=5)
    torch.manual_seed(1)
    s8 = KR.KrigStem(8, NS, T, TI, ocean, vel=vel, compile=False, impl="torch", **idx)
    torch.manual_seed(1)
    s9 = KR.KrigStem(9, NS, T, TI, ocean, vel=vel, compile=False, impl="torch", **idx)
    sd8 = s8.state_dict()
    with torch.no_grad():
        for k, v in s9.state_dict().items():
            if k == "qmlp.0.weight":                            # (dim, 9 + NS + 8, 1, 1): copy the 8 real feature columns and the rest
                v[:, :8] = sd8[k][:, :8]; v[:, 9:] = sd8[k][:, 8:]
            else:
                v.copy_(sd8[k])
        for p in (s8.k2.weight, s9.k2.weight):
            p.mul_(4.0)
    o8, _ = s8(x8, xs, ok, cl)
    o9, _ = s9(x9, xs, ok, cl)
    d = float((o8 - o9).abs().max())
    check("krig_8feat equals 9 features with a zero 9th", d < 1e-5, f"max |diff| {d:.2e}, n_out {o8.shape[1]}")
    o8.square().mean().backward()
    bad = [n for n, p in s8.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    check("krig_8feat backward finite", not bad, f"bad {bad}")


def ekman_stress():
    rng = np.random.default_rng(0)
    lat, lon = np.linspace(40.5, 46.5, 30), np.linspace(28, 41, 40)
    u, v = rng.normal(0, 6, (3, 30, 40)), rng.normal(0, 6, (3, 30, 40))
    _, _, w_ref, tx, ty = PH.ekman(u, v, lat, lon, lat_min=PM.WE_LAT_MIN)
    w = PM.ekman_from_stress(tx, ty, lat, lon, PM.WE_LAT_MIN)
    d = float(np.nanmax(np.abs(w - w_ref)) / np.nanmax(np.abs(w_ref)))
    check("ekman_stress", d < 1e-12 and np.array_equal(np.isnan(w), np.isnan(w_ref)), f"max rel diff {d:.1e}")


def lat_flip():
    rng = np.random.default_rng(1)
    H, W, n, P = 24, 28, 4, 4
    lat, lon = np.linspace(41.0, 46.0, H), np.linspace(28.0, 40.0, W)
    yy, xx = np.mgrid[:H, :W]
    ocean = ((yy - 12) ** 2 / 90 + (xx - 14) ** 2 / 130) < 1.0           # an elliptic basin with a coastline on every side
    F = {k: rng.normal(0, 1, (n, H, W)) for k in ("u_wind", "v_wind", "sst", "mlotst")}
    F["u_wind"] *= 0.1; F["v_wind"] *= 0.1
    for k in F:
        F[k][:, ~ocean] = np.nan
    asc = lambda k, t=slice(None): F[k][t]
    desc = lambda k, t=slice(None): F[k][t][:, ::-1]
    with tempfile.TemporaryDirectory() as tmp:
        PM.CACHE = Path(tmp)
        a = PM.daily_fields(asc, n, ocean, lat, lon, P, log=lambda s: None, wind_is_stress=True)
        b = PM.daily_fields(desc, n, ocean[::-1].copy(), lat[::-1].copy(), lon, P, log=lambda s: None, wind_is_stress=True)
    b = b[:, :, ::-1]                                                    # patch rows back to south-to-north (H divisible by P)
    fin = np.isfinite(a)
    d = float(np.nanmax(np.abs(a - b)[fin] / (np.abs(a[fin]) + 1e-12)))
    nts = int(np.isfinite(a[:, 2]).sum())
    check("lat_flip daily fields", np.array_equal(fin, np.isfinite(b)) and d < 1e-9 and nts > 0, f"max rel diff {d:.1e}, finite tau_s patch values {nts}")


if __name__ == "__main__":
    krig_8feat(); ekman_stress(); lat_flip()
    print("ALL PASS" if not FAIL else f"FAILED {FAIL}")
    sys.exit(1 if FAIL else 0)
