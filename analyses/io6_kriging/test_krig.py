"""Unit tests of the io6 stems on small random tensors (CPU, fp32): chl_cnn/krig.py (learned kriging, signed head pair, advected centring)
and chl_cnn/tokdec.py --tokdec-spatial.

  stencil      484 offsets: 28 at dt 0, 81 at |dt| 1-2, 33 at |dt| 3-4
  shapes       K1, K1-signed, K1-advect stems; KrigTokDecNet and TokDecNet(spatial=2, blend) forward / backward
  dense        streamed online softmax (chunks of 7 and 32 offsets) vs one dense masked softmax over every candidate + no-key slot:
               outputs and parameter gradients, max |diff| (fp32; criterion < 1e-5 on the outputs)
  signed       dense signed weights softmax(s+) - lambda softmax(s-) incl. the no-key slots sum to 1 - lambda; the module's channels
               (sum softmax(s+) real, lambda sum softmax(s-) real, no-key weights) give the same sum; signed estimate = dense sum w a
  no_key       pixels with no visible candidate: mass 0, estimate 0, entropy 0, attention output = merge bias; no-key logit +30 -> mass ~0,
               -30 -> mass ~1 where keys exist
  brute        visible-key count and the uniform-weight estimate (zero scores) vs numpy loops over the stencil, with land px marked visible
               in the input, a missing window day, and candidates beyond the grid border (never keys)
  advect       --krig-advect: key count vs numpy loops with nearest-neighbour shifted centres
  leak         +1 on day t's hidden px and on not-visible px of a context day: outputs bit-identical; on a visible px: outputs change
  tokdec_spatial  spatial=2 output differs from spatial=0 only via the blocks: with zero-initialised blocks the output equals the base model's

Usage (project root): python -m analyses.io6_kriging.test_krig   -> results/unit_tests.json; exit code 1 on any failure.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chl_cnn import krig as KR
from chl_cnn.tokdec import TokDecNet

RES = Path(__file__).resolve().parent / "results"
T, TI, NF, NS = 17, 8, 9, 7
SST, SSS, IU, IV = 0, 1, 7, 8


def make_inputs(B=2, H=20, W=24, seed=0, p_vis=0.35):
    g = torch.Generator().manual_seed(seed)
    ocean = torch.rand(H, W, generator=g) > 0.25
    ocean[0, :3] = False
    x3d = torch.randn(B, NF + 2, T, H, W, generator=g)
    vis = (torch.rand(B, T, H, W, generator=g) < p_vis) & ocean
    vis[:, :, ~ocean] = True                                    # land marked "visible" in the input: the stem must still ignore it
    x3d[:, -1] = vis.float()
    x3d[:, -2] = x3d[:, -2] * vis
    xs = torch.randn(B, NS, H, W, generator=g)
    day_ok = torch.ones(B, T, dtype=torch.bool)
    day_ok[0, TI + 2] = False                                   # a missing window day inside the stencil
    clim = 0.3 * torch.randn(B, T, H, W, generator=g)
    vel = dict(mu_u=0.05, sd_u=0.3, mu_v=0.0, sd_v=0.22, dx=torch.full((H, W), 27000.0), dy=torch.full((H, W), -27000.0))
    return ocean, x3d, xs, day_ok, clim, vel


def stem(ocean, vel, seed=1, sharpen=True, **kw):
    torch.manual_seed(seed)
    s = KR.KrigStem(NF, NS, T, TI, ocean, SST, SSS, IU, IV, vel, compile=False, **kw)
    if sharpen:                                                 # peaked, non-trivial weights for the comparisons
        with torch.no_grad():
            for p in (s.k2.weight, s.qmlp[2].weight):
                p.mul_(4.0)
            s.relbias.normal_(0, 1.0)
            s.none_logit.normal_(0, 1.0)
    return s


def dense_reference(st, x3d, xs, ok, cl):
    """Every candidate at once, torch.softmax over [visible candidates, no-key]; returns the stem's output channels (B, n_out, N)."""
    ctx = st._prep(x3d, xs, ok, cl)
    if st.advect:
        dts = torch.arange(-KR.MAX_LAG, KR.MAX_LAG + 1, dtype=torch.float32).view(1, -1, 1)
        sx = torch.round(st.s0 * ctx["up"].unsqueeze(1) * dts).clamp(-KR.SHIFT_MAX, KR.SHIFT_MAX).long()
        sy = torch.round(st.s0 * ctx["vp"].unsqueeze(1) * dts).clamp(-KR.SHIFT_MAX, KR.SHIFT_MAX).long()
        ctx["shift"] = torch.stack([sx, sy], 1)
    B, N, nh, hv = ctx["B"], st.N, st.nh, st.heads
    idx, off = st._chunk_index(0, st.Kp, ctx)
    V = ctx["vol"]; C = V.shape[1]; Bi, K, _ = idx.shape
    Vg = torch.gather(V, 2, idx.reshape(Bi, 1, K * N).expand(B, C, K * N)).view(B, C, K, N)
    valid = (Vg[:, 10] > 0.5) & (st.vk.view(1, K, 1) > 0.5)
    s = st.scores(ctx, Vg, off, 0, st.Kp)
    s = torch.where(valid.unsqueeze(1), s, torch.full_like(s, float("-inf")))
    nk = st.none_logit.view(1, nh, 1, 1).expand(B, nh, 1, N)
    w = torch.softmax(torch.cat([s, nk], 2), 2)                 # (B, nh, K + 1, N)
    wr, wnk = w[:, :, :K], w[:, :, K]
    a = Vg[:, 11]
    E = (wr * a.unsqueeze(1)).sum(2)
    A2 = (wr * a.unsqueeze(1) ** 2).sum(2)
    ent = -(w * torch.log(w.clamp(min=1e-30))).sum(2)
    mass = 1 - wnk
    ox, oy, ot = off[..., 0], off[..., 1], off[..., 2]
    advx = (ox - st.s_adv * ctx["up"].unsqueeze(1) * ot) / 6.0
    advy = (oy - st.s_adv * ctx["vp"].unsqueeze(1) * ot) / 6.0
    vf = torch.stack([a, (ox / 6.0).expand_as(a), (oy / 6.0).expand_as(a), (ot / 4.0).expand_as(a), advx, advy], -1)   # (B, K, N, 6)
    g = st.v2(F.gelu(st.v1(vf), approximate="tanh"))           # (B, K, N, dv): full value MLP per candidate
    vals = torch.einsum("bhkn,bknd->bhdn", wr[:, :hv], g)
    att = st.merge(vals.permute(0, 3, 1, 2).reshape(B, N, hv * st.dv)).permute(0, 2, 1)
    est = E[:, :hv].mean(1); S2 = A2[:, :hv].mean(1); Mb = mass[:, :hv].mean(1)
    spread = torch.sqrt((S2 - 2 * est * est + est * est * Mb).clamp(min=0) + 1e-6)
    ch = [att, est[:, None], spread[:, None], Mb[:, None], ent[:, :hv].mean(1)[:, None]]
    extra = {}
    if st.signed:
        lam = st.lam()
        wsg = w[:, hv] - lam * w[:, hv + 1]                       # (K + 1, N) signed weights incl. no-key slots
        ch += [(wsg[:, :K] * a).sum(1)[:, None], mass[:, hv][:, None], (lam * mass[:, hv + 1])[:, None]]
        extra = dict(signed_sum=wsg.sum(1), lam=lam)
    return torch.cat(ch, 1), dict(valid=valid, Vg=Vg, idx=idx, **extra)


def module_out(st, x3d, xs, ok, cl):
    out, rec = st(x3d, xs, ok, cl)
    return out.flatten(2)[..., st.oc_idx], rec


def brute_counts(ocean, x3d, ok, centre_shift=None):
    """numpy: visible keys per ocean px of each sample, and the sum of their anomalies (anomaly = CHL z - clim)."""
    B, _, _, H, W = x3d.shape
    vis = (x3d[:, -1].numpy() > 0.5) & ok.numpy()[:, :, None, None] & ocean.numpy()[None, None]
    st = KR.stencil_k1()
    cnt = np.zeros((B, H, W), int)
    for b in range(B):
        for y in range(H):
            for x in range(W):
                if not ocean[y, x]:
                    continue
                for dt, dy, dx in st:
                    sy = sx = 0
                    if centre_shift is not None:
                        sx, sy = centre_shift(b, y, x, dt)
                    yy, xx = y + dy + sy, x + dx + sx
                    if 0 <= yy < H and 0 <= xx < W and vis[b, TI + dt, yy, xx]:
                        cnt[b, y, x] += 1
    return cnt


def main():
    res, fails = {}, []
    torch.set_default_dtype(torch.float32)

    def check(name, ok, **info):
        res[name] = dict(passed=bool(ok), **info)
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(info)}", flush=True)
        if not ok:
            fails.append(name)

    # stencil
    st_ = KR.stencil_k1()
    per = {dt: sum(1 for o in st_ if o[0] == dt) for dt in range(-4, 5)}
    check("stencil", len(st_) == 484 and per[0] == 28 and all(per[d] == 81 for d in (-2, -1, 1, 2)) and all(per[d] == 33 for d in (-4, -3, 3, 4))
          and len(set(st_)) == 484 and (0, 0, 0) not in st_, n=len(st_), per_dt={str(k): v for k, v in per.items()})

    ocean, x3d, xs, ok, cl, vel = make_inputs()
    B, H, W = x3d.shape[0], x3d.shape[-2], x3d.shape[-1]
    # shapes
    for name, kw, n_out in (("K1", {}, 20), ("K1_signed", dict(signed=True), 23), ("K1_advect", dict(advect=True), 20)):
        s = stem(ocean, vel, chunk=32, **kw)
        out, rec = s(x3d, xs, ok, cl)
        check(f"shape_{name}", tuple(out.shape) == (B, n_out, H, W) and s.n_out == n_out and torch.isfinite(out).all()
              and bool((out[:, :, ~ocean] == 0).all()), shape=list(out.shape), n_params=sum(p.numel() for p in s.parameters()))

    # dense vs streamed: outputs and gradients
    for name, kw in (("K1", {}), ("K1_signed", dict(signed=True)), ("K1_advect", dict(advect=True))):
        ref_st = stem(ocean, vel, chunk=32, **kw)
        ref, info = dense_reference(ref_st, x3d, xs, ok, cl)
        gref = torch.randn_like(ref)
        ref_st.zero_grad(); (ref * gref).sum().backward()
        gr = {n: p.grad.clone() for n, p in ref_st.named_parameters() if p.grad is not None}
        for chunk in (7, 32):
            st = stem(ocean, vel, chunk=chunk, **kw)
            with torch.no_grad():
                for (n1, p1), (n2, p2) in zip(st.named_parameters(), ref_st.named_parameters()):
                    assert n1 == n2
                    if p1.shape == p2.shape:
                        p1.copy_(p2)
                    else:                                         # relbias padded to a different multiple of the chunk
                        k = min(p1.shape[0], p2.shape[0]); p1.zero_(); p1[:k] = p2[:k]
            out, rec = module_out(st, x3d, xs, ok, cl)
            d_out = (out - ref).abs().max().item()
            st.zero_grad(); (out * gref).sum().backward()
            gd = {}
            for n, p in st.named_parameters():
                if n in gr:
                    g1, g2 = p.grad, gr[n]
                    if g1.shape != g2.shape:
                        k = min(g1.shape[0], g2.shape[0]); g1, g2 = g1[:k], g2[:k]
                    gd[n] = (g1 - g2).abs().max().item() / max(g2.abs().max().item(), 1e-12)
            check(f"dense_vs_stream_{name}_chunk{chunk}", d_out < 1e-5 and max(gd.values()) < 1e-4, max_abs_out=d_out,
                  max_rel_grad=max(gd.values()), worst_grad=max(gd, key=gd.get), out_scale=ref.abs().max().item())
        if name == "K1_signed":
            lam = info["lam"].item()
            dev_sum = (info["signed_sum"] - (1 - lam)).abs().max().item()
            out, rec = module_out(st, x3d, xs, ok, cl)
            nv = st.heads; hv = st.dv + 4
            mod_sum = out[:, hv + 1] - out[:, hv + 2] + rec["w_nk_pos"] - lam * rec["w_nk_neg"]
            dev_mod = (mod_sum - (1 - lam)).abs().max().item()
            check("signed_weights_sum", dev_sum < 1e-6 and dev_mod < 1e-6, lam=lam, max_dev_dense=dev_sum, max_dev_module=dev_mod)

    # no-key slot (sparse input: 0.3 % of px visible, so that some px have no visible candidate)
    st = stem(ocean, vel, chunk=32)
    g7 = torch.Generator().manual_seed(7)
    x3s = x3d.clone()
    vs = (torch.rand(x3d[:, -1].shape, generator=g7) < 0.003) & ocean
    x3s[:, -1] = vs.float(); x3s[:, -2] = torch.randn(vs.shape, generator=g7) * vs
    x3d_s = (x3s, xs, ok, cl)
    out, rec = module_out(st, *x3d_s)
    nk = rec["n_keys"].flatten(1)[:, st.oc_idx]
    none = nk == 0
    att, est, spread, mass, ent = out[:, :16], out[:, 16], out[:, 17], out[:, 18], out[:, 19]
    if none.any():
        okk = (mass[none].abs().max() < 1e-7) and (est[none].abs().max() < 1e-7) and (ent[none].abs().max() < 1e-6) and \
              ((att.permute(0, 2, 1)[none] - st.merge.bias).abs().max() < 1e-6)
    else:
        okk = False
    with torch.no_grad():
        st.none_logit.fill_(30.0)
    m_hi = module_out(st, *x3d_s)[0][:, 18]
    with torch.no_grad():
        st.none_logit.fill_(-30.0)
    m_lo = module_out(st, *x3d_s)[0][:, 18]
    check("no_key_slot", bool(okk) and m_hi.max().item() < 1e-6 and (m_lo[~none] > 1 - 1e-6).all().item(), n_px_no_key=int(none.sum()),
          n_px=int(none.numel()), mass_max_at_nk_plus30=m_hi.max().item(), mass_min_at_nk_minus30_with_keys=m_lo[~none].min().item())

    # brute force: key count, uniform-weight estimate (zero scores)
    cnt = brute_counts(ocean, x3d, ok)
    st = stem(ocean, vel, chunk=32, sharpen=False)
    with torch.no_grad():
        for p in (st.k2.weight, st.k2.bias, st.relbias, st.none_logit):
            p.zero_()
    out, rec = st(x3d, xs, ok, cl)
    nkeys = rec["n_keys"].round().long().numpy()
    anom = ((x3d[:, -2] - cl) * x3d[:, -1]).numpy()
    vis = (x3d[:, -1].numpy() > 0.5) & ok.numpy()[:, :, None, None] & ocean.numpy()[None, None]
    ssum = np.zeros((B, H, W))
    for b in range(B):
        for y, x in zip(*np.nonzero(ocean.numpy())):
            for dt, dy, dx in KR.stencil_k1():
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and vis[b, TI + dt, yy, xx]:
                    ssum[b, y, x] += anom[b, TI + dt, yy, xx]
    est_bf = ssum / (cnt + 1)
    oc = ocean.numpy()
    d_est = np.abs(out[:, 16].detach().numpy() - est_bf)[:, oc].max()
    check("brute_force_counts_and_estimate", np.array_equal(nkeys[:, oc], cnt[:, oc]) and (nkeys[:, ~oc] == 0).all() and d_est < 1e-5,
          max_abs_est=float(d_est), n_keys_mean=float(cnt[:, oc].mean()), n_keys_max=int(cnt.max()), n_px_no_key=int((cnt[:, oc] == 0).sum()))

    # advected centring: counts with shifted centres
    st = stem(ocean, vel, chunk=32, advect=True)
    with torch.no_grad():                                       # strong currents so that shifts occur
        x3d_a = x3d.clone(); x3d_a[:, IU] = 6.0 * torch.randn_like(x3d_a[:, IU]); x3d_a[:, IV] = 6.0 * torch.randn_like(x3d_a[:, IV])
    out, rec = st(x3d_a, xs, ok, cl)
    u = ((x3d_a[:, IU, TI] * st.u_a + st.u_b) * ocean).numpy(); v = ((x3d_a[:, IV, TI] * st.v_a + st.v_b) * ocean).numpy()
    shift = lambda b, y, x, dt: (int(np.clip(np.round(st.s0 * np.float32(u[b, y, x]) * np.float32(dt)), -4, 4)),
                                 int(np.clip(np.round(st.s0 * np.float32(v[b, y, x]) * np.float32(dt)), -4, 4)))
    cnt_a = brute_counts(ocean, x3d_a, ok, shift)
    n_shifted = sum(1 for b in range(B) for y, x in zip(*np.nonzero(oc)) if shift(b, y, x, 4) != (0, 0))
    check("advect_counts", np.array_equal(rec["n_keys"].round().long().numpy()[:, oc], cnt_a[:, oc]), n_px_shifted_at_dt4=n_shifted,
          n_px=int(B * oc.sum()), differs_from_unshifted=bool(not np.array_equal(cnt_a, cnt)))

    # leak: day t hidden px and context-day not-visible px
    for name, kw in (("K1", {}), ("K1_signed", dict(signed=True))):
        st = stem(ocean, vel, chunk=32, **kw)
        base, _ = st(x3d, xs, ok, cl)
        x1 = x3d.clone()
        hid_t = (x3d[:, -1, TI] < 0.5)
        x1[:, -2, TI] = x1[:, -2, TI] + 1.0 * hid_t             # CHL value on day t's not-visible (hidden / cloud / land) px
        nv2 = (x3d[:, -1, TI + 2] < 0.5)
        x1[:, -2, TI + 2] = x1[:, -2, TI + 2] + 1.0 * nv2       # CHL value on t+2's not-visible px
        o1, _ = st(x1, xs, ok, cl)
        x2 = x3d.clone()
        vp = (x3d[:, -1, TI + 2] > 0.5) & ocean
        x2[:, -2, TI + 2] = x2[:, -2, TI + 2] + 1.0 * vp
        o2, _ = st(x2, xs, ok, cl)
        check(f"leak_{name}", torch.equal(o1, base) and (o2 - base).abs().max().item() > 1e-3, max_change_hidden_and_masked=(o1 - base).abs().max().item(),
              max_change_positive_control=(o2 - base).abs().max().item())

    # full models: KrigTokDecNet and TokDecNet(spatial=2) forward / backward
    torch.manual_seed(3)
    oc2, x3b, xsb, okb, clb, velb = make_inputs(B=2, H=24, W=28, seed=5)
    kw = dict(P=4, dim=32, ch=8, heads=4, layers=1, sst_index=SST, width=16, out_channels=3, dropout=0.0, dilations=None)
    for name, krig_kw in (("KrigTokDecNet", dict(signed=False)), ("KrigTokDecNet_signed", dict(signed=True))):
        net = KR.build_krig_tokdec(11, 7, T, TI, oc2, dict(sst_index=SST, sss_index=SSS, u_index=IU, v_index=IV, vel=velb, compile=False,
                                                           chunk=32, **krig_kw), **kw)
        y = net(x3b, xsb, okb, clb)
        y.square().mean().backward()
        ng = sum(1 for p in net.parameters() if p.grad is None)
        check(f"model_{name}", tuple(y.shape) == (2, 3, 24, 28) and torch.isfinite(y).all().item() and ng == 0,
              n_params=sum(p.numel() for p in net.parameters()), n_params_no_grad=ng, backbone_in=net.backbone.inp.in_channels)
    torch.manual_seed(4); base = TokDecNet(11, 7, T, TI, oc2, blend=True, **kw)
    torch.manual_seed(4); spn = TokDecNet(11, 7, T, TI, oc2, blend=True, spatial=2, **kw)
    sd = spn.state_dict(); sb = base.state_dict()
    same_init = all(torch.equal(sb[k], sd[k]) for k in sb)
    y0, y1 = base(x3b, xsb, okb, clb), spn(x3b, xsb, okb, clb)
    with torch.no_grad():
        for blk in spn.sp_blocks:
            blk.proj.weight.normal_(0, 0.05); blk.rel.normal_(0, 1.0)
    y2 = spn(x3b, xsb, okb, clb)
    y2.square().mean().backward()
    check("tokdec_spatial", same_init and torch.equal(y0, y1) and (y2 - y0).abs().max().item() > 1e-4 and spn.sp_blocks[0].rel.grad is not None,
          zero_init_equals_base=bool(torch.equal(y0, y1)), change_after_perturb=(y2 - y0).abs().max().item(),
          n_params_spatial=sum(p.numel() for p in spn.sp_blocks.parameters()), n_rel=int(spn.sp_blocks[0].rel.shape[0]))

    res["n_failed"] = len(fails)
    RES.mkdir(exist_ok=True)
    (RES / "unit_tests.json").write_text(json.dumps(res, indent=1))
    print("ALL PASSED" if not fails else f"FAILED: {fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
