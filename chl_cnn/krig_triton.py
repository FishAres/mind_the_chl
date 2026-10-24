"""Fused Triton kernels for the learned-kriging stem (chl_cnn/krig.py; io6 track, 2026-10-24).

Same model as krig._chunk_core (the PyTorch streamed reference), reorganised so that nothing per candidate is materialised:
  key pre-activation   pre(p, c) = Ck[k] + F(c) Wl - (dt_k / 6) U(p)
                       F(c) = [9 physics at c, density at c, |SST(c) - SST(p)|, |SSS(c) - SSS(p)|, 0 x 4] (16 columns),
                       Wl = the matching rows of k1.weight^T (zero-padded to 16 rows),
                       Ck[k] = b1 + W1[:, 0:3] (dx, dy, dt)_k / (6, 6, 4) + W1[:, 3:5] (dx, dy)_k / 6   (offset and advected-offset terms),
                       U(p) = s (W1[:, 3] u(p) + W1[:, 4] v(p))                                        (advection term)
  key                  key = gelu(pre) W2^T + b2  (k2; heads x dh, zero-padded to NHP x dh)
  score                s_h = q_h(p) . key_h + relb[k, h]      (q already scaled by 1 / sqrt(dh))
  value                pv = Cvk[k] + wva a(c) - (dt_k / 6) Uv(p), Cvk[k] = bv1 + Wv1[:, 1:4] (dx, dy, dt)_k / (6, 6, 4) + Wv1[:, 4:6] (dx, dy)_k / 6,
                       Uv(p) = s (Wv1[:, 4] u(p) + Wv1[:, 5] v(p)), wva = Wv1[:, 0]; value features gelu(pv), weighted per value head.
Ck, U, Cvk, Uv, Wl, W2, b2, wva, q, relb are built in PyTorch (autograd) and passed in; the backward kernel returns their gradients.
Forward: one program per (block of BN ocean px, sample) streams all K stencil offsets with an online softmax per head (running max m,
normaliser l, sums of e a, e a^2, e s and e gelu(pv)), starting from the no-key slot (m = nk, l = 1, e s = nk).
Backward (flash-style): recomputes every candidate's score and weight w = exp(s - m - log l) and uses
  dL/ds_c = w_c (dL/dw_c - D),  dL/dw_c = dE a + dA2 a^2 + dEnt (-log w_c - 1) + dG . gelu(pv_c),
  D = dE E + dA2 A2 + dEnt (Ent - 1) + dWnk Wnk + dG . G      (sum over every slot, no-key included).
Matrix products use tl.dot with input_precision CFG["PREC"]. Per-offset gradients (Ck, Cvk, relb) are atomics with a staggered offset
order per program; weight gradients accumulate per program and are added once. No back-advected centring (--krig-advect: PyTorch path).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

FW = 16                                                       # gathered location-feature columns (12 used)


@triton.jit
def _sig(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _gelu(x):                                                 # tanh approximation: 0.5 x (1 + tanh z) = x sigmoid(2 z)
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x * _sig(2.0 * z)


@triton.jit
def _gelu_grad(x):
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    sg = _sig(2.0 * z)
    return sg + x * sg * (1.0 - sg) * 2.0 * 0.7978845608028654 * (1.0 + 0.134145 * x * x)


@triton.jit
def _cand(Vb, L9, idx, nm, cc, sp, ssp, SST: tl.constexpr, SSS: tl.constexpr):
    """Gathered features of one offset for the block: F [BN, 16], visible flag, anomaly."""
    Fm = tl.load(Vb + cc[None, :] * L9 + idx[:, None], mask=nm[:, None] & (cc < 10)[None, :], other=0.0)
    dsst = tl.abs(tl.load(Vb + SST * L9 + idx, mask=nm, other=0.0) - sp)
    dsss = tl.abs(tl.load(Vb + SSS * L9 + idx, mask=nm, other=0.0) - ssp)
    Fm = tl.where(cc[None, :] == 10, dsst[:, None], Fm)
    Fm = tl.where(cc[None, :] == 11, dsss[:, None], Fm)
    vis = tl.load(Vb + 10 * L9 + idx, mask=nm, other=0.0)
    a = tl.load(Vb + 11 * L9 + idx, mask=nm, other=0.0)
    return Fm, vis, a


@triton.jit
def _fwd_kernel(V, base, delta, otv, Ck, Cvk, relb, Wl, W2T, b2, wva, q, Up, Uvp, sstp, sssp, nk,
                Om, Ol, OSa, OSa2, OSs, OSg, Ocnt, N, K, L9,
                NH: tl.constexpr, NHP: tl.constexpr, NV: tl.constexpr, DH: tl.constexpr, DIM: tl.constexpr, DV: tl.constexpr,
                BN: tl.constexpr, SST: tl.constexpr, SSS: tl.constexpr, PREC: tl.constexpr):
    QD: tl.constexpr = NHP * DH
    pid = tl.program_id(0)
    b = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    nm = n < N
    h = tl.arange(0, NHP)
    hm = h < NH
    j = tl.arange(0, DIM)
    d = tl.arange(0, DV)
    e2 = tl.arange(0, QD)
    cc = tl.arange(0, 16)
    q2 = tl.load(q + b * QD * N + e2[None, :] * N + n[:, None], mask=nm[:, None], other=0.0)          # [BN, QD]
    U2 = tl.load(Up + b * DIM * N + j[None, :] * N + n[:, None], mask=nm[:, None], other=0.0)
    Uv2 = tl.load(Uvp + b * DV * N + d[None, :] * N + n[:, None], mask=nm[:, None], other=0.0)
    sp = tl.load(sstp + b * N + n, mask=nm, other=0.0)
    ssp = tl.load(sssp + b * N + n, mask=nm, other=0.0)
    pb = tl.load(base + n, mask=nm, other=0)
    Wl2 = tl.load(Wl + cc[:, None] * DIM + j[None, :])                                                  # [16, DIM]
    W2t = tl.load(W2T + j[:, None] * QD + e2[None, :])                                                  # [DIM, QD]
    b2v = tl.load(b2 + e2)
    wv = tl.load(wva + d)
    nkv = tl.load(nk + h, mask=hm, other=0.0)
    m = tl.zeros([BN, NHP], tl.float32) + nkv[None, :]
    lsum = tl.full([BN, NHP], 1.0, tl.float32)
    Sa = tl.zeros([BN, NHP], tl.float32)
    Sa2 = tl.zeros([BN, NHP], tl.float32)
    Ss = tl.zeros([BN, NHP], tl.float32) + nkv[None, :]
    Sg = tl.zeros([BN, NHP, DV], tl.float32)
    cnt = tl.zeros([BN], tl.float32)
    Vb = V + b * 12 * L9
    for k in range(0, K):
        idx = pb + tl.load(delta + k)
        valid = (tl.load(Vb + 10 * L9 + idx, mask=nm, other=0.0) > 0.5) & nm
        if tl.sum(valid.to(tl.int32), 0) > 0:                 # offsets with no visible key in the block change nothing: skipped
            otk = tl.load(otv + k)
            Fm, vis, a = _cand(Vb, L9, idx, nm, cc, sp, ssp, SST, SSS)
            pre = tl.load(Ck + k * DIM + j)[None, :] - (otk / 6.0) * U2 + tl.dot(Fm, Wl2, input_precision=PREC)
            hk = _gelu(pre)
            key = tl.dot(hk, W2t, input_precision=PREC) + b2v[None, :]
            s = tl.sum(tl.reshape(q2 * key, [BN, NHP, DH]), 2) + tl.load(relb + k * NH + h, mask=hm, other=0.0)[None, :]
            s = tl.where(valid[:, None], s, float("-inf"))
            m_new = tl.maximum(m, s)
            alpha = tl.exp(m - m_new)
            e = tl.exp(s - m_new)
            lsum = lsum * alpha + e
            Sa = Sa * alpha + e * a[:, None]
            Sa2 = Sa2 * alpha + e * (a * a)[:, None]
            Ss = Ss * alpha + tl.where(valid[:, None], e * s, 0.0)
            pv = tl.load(Cvk + k * DV + d)[None, :] + a[:, None] * wv[None, :] - (otk / 6.0) * Uv2
            hv = _gelu(pv)
            Sg = Sg * alpha[:, :, None] + e[:, :, None] * hv[:, None, :]
            cnt += valid.to(tl.float32)
            m = m_new
    o2 = b * NH * N + h[None, :] * N + n[:, None]
    m2 = nm[:, None] & hm[None, :]
    tl.store(Om + o2, m, mask=m2)
    tl.store(Ol + o2, lsum, mask=m2)
    tl.store(OSa + o2, Sa, mask=m2)
    tl.store(OSa2 + o2, Sa2, mask=m2)
    tl.store(OSs + o2, Ss, mask=m2)
    tl.store(OSg + b * NV * DV * N + h[None, :, None] * DV * N + d[None, None, :] * N + n[:, None, None], Sg,
             mask=nm[:, None, None] & (h < NV)[None, :, None])
    tl.store(Ocnt + b * N + n, cnt, mask=nm)


@triton.jit
def _bwd_kernel(V, base, delta, otv, Ck, Cvk, relb, Wl, W2T, b2, wva, q, Up, Uvp, sstp, sssp,
                Im, Ilogl, ID, IdE, IdA2, IdEnt, IdG,
                Gq, GU, GUv, GCk, GCvk, Grelb, GWl, GW2T, Gb2, Gwva, N, K, L9,
                NH: tl.constexpr, NHP: tl.constexpr, NV: tl.constexpr, DH: tl.constexpr, DIM: tl.constexpr, DV: tl.constexpr,
                BN: tl.constexpr, SST: tl.constexpr, SSS: tl.constexpr, PREC: tl.constexpr):
    QD: tl.constexpr = NHP * DH
    pid = tl.program_id(0)
    b = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    nm = n < N
    h = tl.arange(0, NHP)
    hm = h < NH
    j = tl.arange(0, DIM)
    d = tl.arange(0, DV)
    e2 = tl.arange(0, QD)
    cc = tl.arange(0, 16)
    qoff = b * QD * N + e2[None, :] * N + n[:, None]
    q2 = tl.load(q + qoff, mask=nm[:, None], other=0.0)
    U2 = tl.load(Up + b * DIM * N + j[None, :] * N + n[:, None], mask=nm[:, None], other=0.0)
    Uv2 = tl.load(Uvp + b * DV * N + d[None, :] * N + n[:, None], mask=nm[:, None], other=0.0)
    sp = tl.load(sstp + b * N + n, mask=nm, other=0.0)
    ssp = tl.load(sssp + b * N + n, mask=nm, other=0.0)
    pb = tl.load(base + n, mask=nm, other=0)
    Wl2 = tl.load(Wl + cc[:, None] * DIM + j[None, :])
    W2t = tl.load(W2T + j[:, None] * QD + e2[None, :])                                                  # [DIM, QD]
    W2n = tl.load(W2T + j[None, :] * QD + e2[:, None])                                                  # [QD, DIM]
    b2v = tl.load(b2 + e2)
    wv = tl.load(wva + d)
    o2 = b * NH * N + h[None, :] * N + n[:, None]
    m2 = nm[:, None] & hm[None, :]
    mm = tl.load(Im + o2, mask=m2, other=0.0)
    logl = tl.load(Ilogl + o2, mask=m2, other=0.0)
    Dd = tl.load(ID + o2, mask=m2, other=0.0)
    dE = tl.load(IdE + o2, mask=m2, other=0.0)
    dA2 = tl.load(IdA2 + o2, mask=m2, other=0.0)
    dEnt = tl.load(IdEnt + o2, mask=m2, other=0.0)
    dG3 = tl.load(IdG + b * NV * DV * N + h[None, :, None] * DV * N + d[None, None, :] * N + n[:, None, None],
                  mask=nm[:, None, None] & (h < NV)[None, :, None], other=0.0)
    dq2 = tl.zeros([BN, QD], tl.float32)
    dU2 = tl.zeros([BN, DIM], tl.float32)
    dUv2 = tl.zeros([BN, DV], tl.float32)
    gWl = tl.zeros([16, DIM], tl.float32)
    gW2 = tl.zeros([DIM, QD], tl.float32)
    gb2 = tl.zeros([BN, QD], tl.float32)
    gwv = tl.zeros([BN, DV], tl.float32)
    Vb = V + b * 12 * L9
    k0 = (pid * 37 + b * 11) % K                              # staggered start: programs do not hit the same per-offset atomics together
    for kk in range(0, K):
        k = (k0 + kk) % K
        idx = pb + tl.load(delta + k)
        valid = (tl.load(Vb + 10 * L9 + idx, mask=nm, other=0.0) > 0.5) & nm
        if tl.sum(valid.to(tl.int32), 0) > 0:                 # no visible key in the block: every gradient term is 0, skipped
          otk = tl.load(otv + k)
          Fm, vis, a = _cand(Vb, L9, idx, nm, cc, sp, ssp, SST, SSS)
          pre = tl.load(Ck + k * DIM + j)[None, :] - (otk / 6.0) * U2 + tl.dot(Fm, Wl2, input_precision=PREC)
          hk = _gelu(pre)
          key = tl.dot(hk, W2t, input_precision=PREC) + b2v[None, :]
          s = tl.sum(tl.reshape(q2 * key, [BN, NHP, DH]), 2) + tl.load(relb + k * NH + h, mask=hm, other=0.0)[None, :]
          lw = s - mm - logl
          vh = valid[:, None] & hm[None, :]
          w = tl.where(vh, tl.exp(lw), 0.0)
          pv = tl.load(Cvk + k * DV + d)[None, :] + a[:, None] * wv[None, :] - (otk / 6.0) * Uv2
          hv = _gelu(pv)
          dw = dE * a[:, None] + dA2 * (a * a)[:, None] + dEnt * (-lw - 1.0) + tl.sum(dG3 * hv[:, None, :], 2)
          ds = tl.where(vh, w * (dw - Dd), 0.0)                                                         # [BN, NHP]
          tl.atomic_add(Grelb + k * NH + h, tl.sum(ds, 0), mask=hm)
          dsr = tl.reshape(tl.broadcast_to(ds[:, :, None], [BN, NHP, DH]), [BN, QD])
          dq2 += dsr * key
          dkey = dsr * q2                                                                                # [BN, QD]
          gb2 += dkey
          gW2 += tl.dot(tl.trans(hk), dkey, input_precision=PREC)
          dpre = tl.dot(dkey, W2n, input_precision=PREC) * _gelu_grad(pre)                             # [BN, DIM]
          tl.atomic_add(GCk + k * DIM + j, tl.sum(dpre, 0))
          dU2 += (-otk / 6.0) * dpre
          gWl += tl.dot(tl.trans(Fm), dpre, input_precision=PREC)
          dpv = tl.sum(w[:, :, None] * dG3, 1) * _gelu_grad(pv)
          tl.atomic_add(GCvk + k * DV + d, tl.sum(dpv, 0))
          gwv += dpv * a[:, None]
          dUv2 += (-otk / 6.0) * dpv
    tl.store(Gq + qoff, dq2, mask=nm[:, None])
    tl.store(GU + b * DIM * N + j[None, :] * N + n[:, None], dU2, mask=nm[:, None])
    tl.store(GUv + b * DV * N + d[None, :] * N + n[:, None], dUv2, mask=nm[:, None])
    tl.atomic_add(GWl + cc[:, None] * DIM + j[None, :], gWl)
    tl.atomic_add(GW2T + j[:, None] * QD + e2[None, :], gW2)
    tl.atomic_add(Gb2 + e2, tl.sum(gb2, 0))
    tl.atomic_add(Gwva + d, tl.sum(gwv, 0))


# A10G (analyses/io6_kriging/gpu_check.py). PREC: tl.dot input precision, "ieee" (fp32 FMA) or "tf32" / "tf32x3" (tensor cores)
CFG = dict(BN_FWD=16, WARPS_FWD=4, BN_BWD=16, WARPS_BWD=4, PREC="tf32")


def _np2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class KrigAttnFn(torch.autograd.Function):
    """Inputs (fp32, contiguous): V (B, 12, L9) gather volume [9 physics, density, visible, anomaly]; base (N,) int32; delta (K,) int32;
    otv (K,) dt; Ck (K, dim); Cvk (K, dv); relb (K, nh); Wl (16, dim); W2T (dim, NHP dh); b2 (NHP dh,); wva (dv,); q (B, NHP dh, N) scaled
    queries (zero-padded heads); U (B, dim, N); Uv (B, dv, N); sstp, sssp (B, N); nk (nh,); nv value heads; dh; sst / sss channel.
    Outputs: E, A2, Ent, Wnk (B, nh, N), G (B, nv, dv, N); non-differentiable cnt (B, N), m, logl (B, nh, N)."""

    @staticmethod
    def forward(ctx, V, base, delta, otv, Ck, Cvk, relb, Wl, W2T, b2, wva, q, U, Uv, sstp, sssp, nk, nv, dh, sst_i, sss_i):
        B, _, L9 = V.shape
        K, dim = Ck.shape
        dv, nh, N = Cvk.shape[1], relb.shape[1], base.shape[0]
        nhp = W2T.shape[1] // dh
        f32 = dict(device=V.device, dtype=torch.float32)
        m, l, Sa, Sa2, Ss = (torch.empty(B, nh, N, **f32) for _ in range(5))
        Sg = torch.empty(B, nv, dv, N, **f32)
        cnt = torch.empty(B, N, **f32)
        grid = (triton.cdiv(N, CFG["BN_FWD"]), B)
        _fwd_kernel[grid](V, base, delta, otv, Ck, Cvk, relb, Wl, W2T, b2, wva, q, U, Uv, sstp, sssp, nk, m, l, Sa, Sa2, Ss, Sg, cnt, N, K, L9,
                          NH=nh, NHP=nhp, NV=nv, DH=dh, DIM=dim, DV=dv, BN=CFG["BN_FWD"], SST=sst_i, SSS=sss_i, PREC=CFG["PREC"],
                          num_warps=CFG["WARPS_FWD"])
        E, A2 = Sa / l, Sa2 / l
        logl = torch.log(l)
        Ent = logl + m - Ss / l
        Wnk = torch.exp(nk.view(1, -1, 1) - m - logl)
        G = Sg / l[:, :nv].unsqueeze(2)
        ctx.save_for_backward(V, base, delta, otv, Ck, Cvk, relb, Wl, W2T, b2, wva, q, U, Uv, sstp, sssp, nk, m, logl, E, A2, Ent, Wnk, G)
        ctx.nv, ctx.dh, ctx.sst, ctx.sss = nv, dh, sst_i, sss_i
        ctx.mark_non_differentiable(cnt, m, logl)
        return E, A2, Ent, Wnk, G, cnt, m, logl

    @staticmethod
    def backward(ctx, dE, dA2, dEnt, dWnk, dG, _c, _m, _l):
        V, base, delta, otv, Ck, Cvk, relb, Wl, W2T, b2, wva, q, U, Uv, sstp, sssp, nk, m, logl, E, A2, Ent, Wnk, G = ctx.saved_tensors
        nv, dh = ctx.nv, ctx.dh
        z = lambda t, ref: torch.zeros_like(ref) if t is None else t.contiguous()
        dE, dA2, dEnt, dWnk, dG = z(dE, E), z(dA2, A2), z(dEnt, Ent), z(dWnk, Wnk), z(dG, G)
        D = dE * E + dA2 * A2 + dEnt * (Ent - 1) + dWnk * Wnk
        D[:, :nv] += (dG * G).sum(2)
        D = D.contiguous()
        B, _, L9 = V.shape
        K, dim = Ck.shape
        dv, nh, N = Cvk.shape[1], relb.shape[1], base.shape[0]
        nhp = W2T.shape[1] // dh
        gq, gU, gUv = torch.empty_like(q), torch.empty_like(U), torch.empty_like(Uv)
        gCk, gCvk, grelb, gWl, gW2T, gb2, gwva = (torch.zeros_like(t) for t in (Ck, Cvk, relb, Wl, W2T, b2, wva))
        grid = (triton.cdiv(N, CFG["BN_BWD"]), B)
        _bwd_kernel[grid](V, base, delta, otv, Ck, Cvk, relb, Wl, W2T, b2, wva, q, U, Uv, sstp, sssp, m, logl, D, dE, dA2, dEnt, dG,
                          gq, gU, gUv, gCk, gCvk, grelb, gWl, gW2T, gb2, gwva, N, K, L9,
                          NH=nh, NHP=nhp, NV=nv, DH=dh, DIM=dim, DV=dv, BN=CFG["BN_BWD"], SST=ctx.sst, SSS=ctx.sss, PREC=CFG["PREC"],
                          num_warps=CFG["WARPS_BWD"])
        # no-key slot: log w0 = nk - m - log l; dL/dw0 = dEnt (-log w0 - 1) + dWnk; ds0 = w0 (dL/dw0 - D)
        lw0 = nk.view(1, -1, 1) - m - logl
        gnk = (Wnk * (dEnt * (-lw0 - 1) + dWnk - D)).sum((0, 2))
        return (None, None, None, None, gCk, gCvk, grelb, gWl, gW2T, gb2, gwva, gq, gU, gUv, None, None, gnk, None, None, None, None)


np2 = _np2