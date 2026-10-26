"""Analog tokens (io7 track, bridge plan #25 Phase 1 run 3; run io7_analog; train_pinn.py --analog, default off = unchanged).

Cross-attention from each pooled day-t patch token of a --t3d-stem tokdec model to the same patch on K "analog" days drawn from the
training years, within +-W days of day t's day of year. Design note: analyses/io7/analog/DESIGN.md.

Bank (built once per run from the in-memory store tensors, on the model's device, float16; not saved in model.pt):
  one entry per (training day, ocean patch): [P^2 observed CHL z-anomalies vs the DOY climatology (0 where unobserved),
  P^2 observed mask, 9 physics patch means over ocean px (z-scored as the model inputs)] -> 2 P^2 + n_feat floats (41 for P 4).
  Only training-year days enter the bank (asserted against the val / test day lists).
Candidates for query day t (the same rule for training, val and test samples):
  bank day d with circular |DOY(d) - DOY(t)| <= --analog-window, year(d) != year(t) (leakage guard for training samples; no copying of
  t's own neighbours), and |d - t| > --analog-gap calendar days (no adjacent-year day near a year boundary, i.e. never a day of t's
  +-8 context window shown unmasked). K = --analog-k drawn without replacement: training: a fresh draw per sample per step (own
  numpy RNG, seed --seed + --analog-seed; not stored in resume.pt); eval (module.eval()): a fixed draw per day,
  default_rng([--analog-seed, t]), independent of batch composition.
Attention (per ocean patch, the stem's heads and width D): query LayerNorm(pooled token) -> W_q; each entry [descriptor, dDOY / W, |dDOY| / W]
  -> Linear(F + 2, D) GELU Linear(D, D) LayerNorm -> W_k, W_v; + a learned null key / value (zero init) so a patch can attend nowhere;
  entries whose analog patch has no observed pixel are masked. Output Linear(D, D) zero-initialised, added to the pooled token before
  the spatial blocks (a forward hook on TokDecNet.po, so chl_cnn/tokdec.py is untouched); at init the model equals the io6 model.
Days per batch: train_pinn.predict_z calls model.analog.set_days(bi) before each forward; the forward consumes them (error if unset).
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

RESUME_DEFAULTS = dict(analog=False, analog_k=32, analog_window=30, analog_gap=30, analog_seed=0)


def add_args(p):
    p.add_argument("--analog", action="store_true", help="--t3d-stem tokdec: analog-token cross-attention to the same patch on training-year "
                   "days within +---analog-window days of season (chl_cnn/analog.py; io7)")
    p.add_argument("--analog-k", type=int, default=32, help="--analog: analog days sampled per query day (OOM fallback: 16)")
    p.add_argument("--analog-window", type=int, default=30, help="--analog: max circular day-of-year distance of an analog day")
    p.add_argument("--analog-gap", type=int, default=30, help="--analog: analog days within this many calendar days of day t are excluded")
    p.add_argument("--analog-seed", type=int, default=0, help="--analog: eval draw seed (per day) and training RNG offset")


def check_args(p, a):
    if a.analog and a.t3d_stem != "tokdec":
        p.error("--analog needs --t3d-stem tokdec")
    if a.analog and (a.analog_k < 1 or a.analog_window < 1 or a.analog_gap < 0):
        p.error("--analog: --analog-k >= 1, --analog-window >= 1, --analog-gap >= 0")


def day_meta(times):
    """(year, day of year 0..365, day number) int64 arrays of the store days."""
    t = np.asarray(times).astype("datetime64[D]")
    yr = t.astype("datetime64[Y]")
    return yr.astype(int) + 1970, (t - yr).astype(int), t.astype(int)


def candidates(t, bank_days, meta, window, gap):
    """Bank-day store indices allowed as analogs of store day t (sorted)."""
    yr, doy, dn = meta
    d = np.abs(doy[bank_days] - doy[t])
    d = np.minimum(d, 365 - d)
    m = (d <= window) & (yr[bank_days] != yr[t]) & (np.abs(dn[bank_days] - dn[t]) > gap)
    return bank_days[m]


class AnalogBank:
    """data (n_bank, Np, F) float16 on the device; days: the bank's store indices (sorted training days); row: store index -> bank row."""

    def __init__(self, data, days, n_store, meta, P):
        self.data, self.days, self.meta, self.P = data, np.asarray(days, dtype=np.int64), meta, P
        self.row = np.full(n_store, -1, dtype=np.int64)
        self.row[self.days] = np.arange(len(self.days))

    def draw(self, days, k, window, gap, rng=None, seed=0):
        """-> store indices (B, k) (-1 = padding) and dDOY (B, k); rng None = the fixed eval draw default_rng([seed, t])."""
        out = np.full((len(days), k), -1, dtype=np.int64)
        for i, t in enumerate(np.asarray(days, dtype=np.int64)):
            c = candidates(int(t), self.days, self.meta, window, gap)
            g = rng if rng is not None else np.random.default_rng([seed, int(t)])
            pick = c if len(c) <= k else np.sort(g.choice(c, k, replace=False))
            out[i, :len(pick)] = pick
        doy = self.meta[1]
        dd = doy[np.maximum(out, 0)] - doy[np.asarray(days, dtype=np.int64)][:, None]
        dd = (dd + 182) % 365 - 182                                    # signed circular DOY offset
        return out, np.where(out >= 0, dd, 0)


@torch.no_grad()
def build_bank(model, times, tr, feats, chl_z, obs, clim, val_test=(), chunk=64):
    """Bank of the training days tr from the store tensors: feats(idx) (B, F, H, W) z-scored features, chl_z(idx) (B, H, W) z ln CHL (0
    unobserved), obs(idx) (B, H, W) bool, clim(idx) (B, H, W) DOY-climatology z; idx a LongTensor on the model's device."""
    tr = np.sort(np.asarray(tr, dtype=np.int64))
    assert not np.isin(tr, np.concatenate([np.asarray(v, dtype=np.int64) for v in val_test]) if len(val_test) else []).any(), \
        "analog bank: a val / test day is in the training list"
    dev = model.ocean_p.device
    P, H, W, Hp, Wp = model.P, model.H, model.W, model.Hp, model.Wp
    oc = model.ocean_p                                                 # (Hp, Wp) float
    n_oc = oc.reshape(model.nPy, P, model.nPx, P).sum((1, 3)).flatten()[model.patch_idx].clamp(min=1)   # (Np,)
    pad = lambda x: F.pad(x, (0, Wp - W, 0, Hp - H))
    px = lambda x: x.reshape(x.shape[0], model.nPy, P, model.nPx, P).permute(0, 1, 3, 2, 4).reshape(x.shape[0], -1, P * P)[:, model.patch_idx]
    out = []
    for s in range(0, len(tr), chunk):
        idx = torch.as_tensor(tr[s:s + chunk], device=dev)
        o = obs(idx).float()
        an = (chl_z(idx).float() - clim(idx).float()) * o
        f = pad(feats(idx).float()) * oc                               # (B, nf, Hp, Wp)
        B, nf = f.shape[:2]
        fm = px(f.reshape(B * nf, Hp, Wp)).sum(-1).view(B, nf, -1).transpose(1, 2) / n_oc.view(1, -1, 1)   # (B, Np, nf)
        out.append(torch.cat([px(pad(an)), px(pad(o)), fm], -1).half())
    return AnalogBank(torch.cat(out), tr, len(times), day_meta(times), P)


class AnalogAttn(nn.Module):
    def __init__(self, dim, heads, n_in, bank, k=32, window=30, gap=30, seed=0, train_seed=0):
        super().__init__()
        self.D, self.h, self.k, self.window, self.gap, self.seed = dim, heads, k, window, gap, seed
        self.bank = bank                                               # plain attribute: rebuilt each run, not in the state dict
        self.rng = np.random.default_rng(train_seed + 1_000_003 * (seed + 1))
        self.enc = nn.Sequential(nn.Linear(n_in + 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.nq, self.q, self.kv, self.proj = nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.null_kv = nn.Parameter(torch.zeros(2, dim))
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self._days, self.last = None, {}

    def set_days(self, days):
        self._days = np.asarray(days.detach().cpu().numpy() if torch.is_tensor(days) else days, dtype=np.int64)

    def hook(self, mod, inp, out):                                     # forward hook on TokDecNet.po: pooled (B Np, D) -> + analog term
        return self(out)

    def forward(self, pooled):
        days, self._days = self._days, None
        assert days is not None, "analog: set_days(bi) must be called before every forward"
        B = len(days); N, D = pooled.shape; Np = N // B
        assert N == B * Np
        sel, dd = self.bank.draw(days, self.k, self.window, self.gap, rng=self.rng if self.training else None, seed=self.seed)
        valid = torch.as_tensor(sel >= 0, device=pooled.device)
        r = np.where(sel >= 0, self.bank.row[np.maximum(sel, 0)], 0)
        assert (r >= 0).all(), "analog: a drawn day is not in the bank"
        rows = torch.as_tensor(r, device=pooled.device)
        K = sel.shape[1]
        e = self.bank.data[rows].float().permute(0, 2, 1, 3)              # (B, Np, K, F)
        P2 = self.bank.P ** 2
        seen = (e[..., P2:2 * P2].sum(-1) > 0) & valid.view(B, 1, K)        # (B, Np, K): analog patch has >= 1 observed px
        ddt = torch.as_tensor(dd, device=pooled.device, dtype=torch.float32) / self.window
        e = torch.cat([e, ddt.view(B, 1, K, 1).expand(B, Np, K, 1), ddt.abs().view(B, 1, K, 1).expand(B, Np, K, 1)], -1)
        kv = self.kv(self.enc(e.view(N, K, -1)))                            # (N, K, 2D)
        kv = torch.cat([self.null_kv.to(kv.dtype).view(1, 1, 2 * D).expand(N, 1, 2 * D), kv], 1)
        d = D // self.h
        k_, v_ = kv.view(N, K + 1, 2, self.h, d).permute(2, 0, 3, 1, 4)     # (N, h, K + 1, d) each
        q = self.q(self.nq(pooled)).view(N, self.h, 1, d)
        m = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=pooled.device), seen.view(N, K)], 1).view(N, 1, 1, K + 1)
        o = F.scaled_dot_product_attention(q, k_.to(q.dtype), v_.to(q.dtype), attn_mask=m)
        self.last = dict(days=days, sel=sel, seen_frac=float(seen.float().mean()))
        return pooled + self.proj(o.reshape(N, D)).to(pooled.dtype)


def attach(model, bank, k=32, window=30, gap=30, seed=0, train_seed=0):
    """Adds model.analog (AnalogAttn) to a TokDecNet and hooks it on model.po (the pooling output projection)."""
    model.add_module("analog", AnalogAttn(model.D, model.h, bank.data.shape[-1], bank, k, window, gap, seed, train_seed))
    model.analog.to(model.ocean_p.device)
    model.po.register_forward_hook(model.analog.hook)
    return model


def check_store(argv=None):
    """Bank rules on the real IO store split: no val / test day in the bank; no own-year, near (|d - t| <= gap) or out-of-season
    candidate for any store day; per-split candidate counts. Exit 1 on a violation."""
    import argparse
    from . import data as D, store as S
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=30); p.add_argument("--gap", type=int, default=30); p.add_argument("--k", type=int, default=32)
    a = p.parse_args(argv)
    times = S.load_store()["time"]
    tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    meta = day_meta(times); yr, doy, dn = meta
    bank = AnalogBank(None, tr, len(times), meta, 4)
    assert not np.isin(bank.days, np.r_[va, te]).any() and set(np.unique(yr[bank.days])).isdisjoint(D.VAL_YEARS + D.TEST_YEARS)
    bad, counts = 0, {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        n = []
        for t in idx:
            c = candidates(int(t), bank.days, meta, a.window, a.gap)
            dd = np.abs(doy[c] - doy[t]); dd = np.minimum(dd, 365 - dd)
            bad += int(np.isin(c, np.r_[va, te]).sum() + (yr[c] == yr[t]).sum() + (np.abs(dn[c] - dn[t]) <= a.gap).sum() + (dd > a.window).sum()
                       + (c == t).sum())
            n.append(len(c))
        counts[name] = (len(idx), int(np.min(n)), int(np.median(n)), int(np.max(n)))
        print(f"{name}: {len(idx)} days, candidates per day min / median / max {counts[name][1:]}", flush=True)
    sel, _ = bank.draw(va[:50], a.k, a.window, a.gap)
    sel2, _ = bank.draw(va[:50][::-1], a.k, a.window, a.gap)
    det = np.array_equal(sel, sel2[::-1])
    print(f"bank {len(bank.days)} training days, years {sorted(set(yr[bank.days].tolist()))}; rule violations {bad}; eval draw "
          f"deterministic {det}", flush=True)
    if bad or not det or min(c[1] for c in counts.values()) < a.k:
        raise SystemExit("ANALOG BANK CHECK FAILED")
    print("ANALOG BANK CHECK OK", flush=True)


if __name__ == "__main__":
    check_store()
