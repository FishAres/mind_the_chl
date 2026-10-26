"""CPU unit tests of chl_cnn/physmem.py (io7_physmem). python -m analyses.io7.physmem.test_physmem ; exits non-zero on a failure.

1. equivalence: PhysMemTokDecNet on t-8..t+8 (blend_k 8, zero-init memory, memory input set) == TokDecNet(blend, spatial 2) with the same
   weights, exactly (the copied forward / tokens).
2. +-15 window, blend +-8: forward / backward shapes; memory gets gradients after one step; CHL on day t+12 changes the tokens (positive control); the
   blend has 17 + 1 day slots; a CHL value under day t+3's invisible px changes no token / blend.
3. summaries are past-only: changing the daily physics of days >= t leaves day t's summaries unchanged; changing day t-1 changes them;
   start of record: day 0 has coverage 0 and all-zero means.
4. window_offsets with stride2.
"""
import sys

import numpy as np
import torch

from chl_cnn import physmem as PM
from chl_cnn.tokdec import TokDecNet

FAIL = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    if not ok:
        FAIL.append(name)


def make(T_offs, H=22, W=26, F=9, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = len(T_offs)
    ocean = torch.ones(H, W, dtype=torch.bool); ocean[:5, :7] = False; ocean[-3:, -4:] = False
    x3d = torch.randn(B, F + 2, T, H, W, generator=g)
    vis = torch.rand(B, T, H, W, generator=g) > 0.4
    x3d[:, -1] = vis.float(); x3d[:, -2] = x3d[:, -2] * vis
    xs = torch.randn(B, 6, H, W, generator=g)
    day_ok = torch.ones(B, T, dtype=torch.bool); day_ok[0, 1] = False
    clim = torch.randn(B, T, H, W, generator=g)
    return ocean, x3d, xs, day_ok, clim


KW = dict(P=4, dim=32, ch=8, heads=4, layers=2, sst_index=0, backbone="dilated", width=16, spatial=2, out_channels=3, dilations=[1, 2])


def main():
    torch.manual_seed(0)
    # 1. equivalence on +-8
    offs8 = list(range(-8, 9))
    ocean, x3d, xs, day_ok, clim = make(offs8)
    ref = TokDecNet(9 + 2, 6, len(offs8), 8, ocean, blend=True, **KW).eval()
    pm = PM.PhysMemTokDecNet(9 + 2, 6, offs8, ocean, blend_k=8, **KW).eval()
    missing = pm.load_state_dict(ref.state_dict(), strict=False)
    check("state dict: only mem.* new", all(k.startswith("mem.") for k in missing.missing_keys) and not missing.unexpected_keys)
    pm.mem_in = torch.randn(x3d.shape[0], PM.N_MEM, pm.nPy, pm.nPx)
    with torch.no_grad():
        check("equivalence +-8 (exact)", torch.equal(ref(x3d, xs, day_ok, clim), pm(x3d, xs, day_ok, clim)))

    # 2. +-15 window, blend +-8
    offs = list(range(-15, 16))
    ocean, x3d, xs, day_ok, clim = make(offs)
    net = PM.PhysMemTokDecNet(9 + 2, 6, offs, ocean, blend_k=8, **KW)
    net.mem_in = torch.randn(x3d.shape[0], PM.N_MEM, net.nPy, net.nPx)
    y = net(x3d, xs, day_ok, clim)
    check("forward shape (B, 3, H, W)", tuple(y.shape) == (x3d.shape[0], 3) + tuple(ocean.shape))
    y.square().mean().backward()
    check("grad on mem output layer", net.mem[2].weight.grad is not None and net.mem[2].weight.grad.abs().sum() > 0)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2); opt.step(); opt.zero_grad()
    net(x3d, xs, day_ok, clim).square().mean().backward()
    check("grad reaches mem input conv after one step", net.mem[0].weight.grad.abs().sum() > 0)
    net.eval()
    rec = {}
    import chl_cnn.tokdec as TD
    TD.HOOKS["recorder"] = lambda d: rec.update({k: v.detach().clone() for k, v in d.items() if torch.is_tensor(v)})
    with torch.no_grad():
        y0 = net(x3d, xs, day_ok, clim); r0 = dict(rec)
        check("blend weights have 17 + 1 day slots", r0["w"].shape[1] == 18)
        x2 = x3d.clone(); i12 = offs.index(12); x2[:, -2, i12] += 1.0 * x2[:, -1, i12]
        y1 = net(x2, xs, day_ok, clim); r1 = dict(rec)
        check("t+12 CHL changes tokens (positive control)", (r1["tok"] - r0["tok"]).abs().max() > 0)
        x3 = x3d.clone(); i3 = offs.index(3); hid = x3[:, -1, i3] < 0.5
        x3[:, -2, i3] += 5.0 * hid                             # CHL value under day t+3's invisible px: the stem uses anomaly x visible
        net(x3, xs, day_ok, clim); r3 = dict(rec)
        check("t+3 invisible px value changes no token / blend", torch.equal(r3["tok"], r0["tok"]) and torch.equal(r3["blend"], r0["blend"]))
        net.mem_in = None
        y4 = net(x3d, xs, day_ok, clim)
        check("memory input changes the output once trained", not torch.equal(y4, y0))
    TD.HOOKS["recorder"] = None

    # 3. summaries past-only
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-07-01")).astype("datetime64[ns]")
    times = np.delete(times, [40, 41, 100])                    # store gaps
    n = len(times); rng = np.random.default_rng(0)
    daily = rng.normal(size=(n, 5, 3, 4)).astype("float32"); daily[:, 3] += 300; daily[:, 4] += 40
    daily[5, 0, 0, 0] = np.nan
    pok = np.ones((3, 4), bool); pok[0, 3] = False
    tr = np.arange(n)
    s0, st = PM.summaries(daily, times, tr, pok)
    t = 120
    d2 = daily.copy(); d2[t:] += rng.normal(size=d2[t:].shape).astype("float32") * 3
    s2, _ = PM.summaries(d2, times, tr, pok)
    # z-scoring stats use all days of tr; compare raw by re-using the same stats: rebuild with tr = days < t so stats are identical
    trp = np.arange(t)
    a0, _ = PM.summaries(daily, times, trp, pok); a2, _ = PM.summaries(d2, times, trp, pok)
    # SSTA climatology uses training days only (< t here), so day t's summaries must be identical
    check("summaries of day t unchanged by physics of days >= t", np.array_equal(a0[t], a2[t]) and np.array_equal(a0[:t + 1], a2[:t + 1]))
    d3 = daily.copy(); d3[t - 1, :4] += 5.0
    trp = np.arange(t - 1); a0, _ = PM.summaries(daily, times, trp, pok)      # stats / SST climatology from days < t - 1
    a3, _ = PM.summaries(d3, times, trp, pok)
    check("summaries of day t change with day t-1 physics", not np.array_equal(a0[t], a3[t]) and np.array_equal(a0[t - 1], a3[t - 1]))
    check("day 0: all summary channels 0 (no history)", np.all(s0[0, :10] == 0))
    check("missing fraction: day 0 = 1, day 31 A = 0, day 45 A (2 store gap days) = 2/30", float(s0[0, 10, 1, 1]) == 1 and float(s0[31, 10, 1, 1]) == 0
          and abs(float(s0[45, 10, 1, 1]) - 2 / 30) < 1e-3)
    check("land patch zero", np.all(s0[:, :, 0, 3] == 0))

    # 4. offsets
    o = PM.window_offsets(15, 15, 8, True)
    check("stride2 offsets", o == [-14, -12, -10] + list(range(-8, 9)) + [10, 12, 14])
    check("full offsets", PM.window_offsets(15, 15, 8, False) == list(range(-15, 16)))
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
