"""CPU unit tests of chl_cnn/analog.py on tiny synthetic shapes: python -m chl_cnn.test_analog (exit 1 on a failure).

1. bank: only training days; no val / test year; draws for training days never own-year, never within --analog-gap days, always in season.
2. eval determinism: a day's draw and the model output do not depend on the batch composition or on repeated calls.
3. default path: at init (zero output projection) the attached model gives bit-identical output to the same model without --analog;
   train_pinn.parse without --analog leaves analog off.
4. forward / backward: gradients reach every analog parameter once the output projection is non-zero; training draws vary.
5. val-day data never reaches the bank: perturbing every val / test day leaves the bank unchanged.
"""
import numpy as np
import torch

from . import analog as AN
from .tokdec import TokDecNet

torch.manual_seed(0)
H, W, P, T, NF = 13, 11, 4, 5, 9
TIMES = np.arange(np.datetime64("2000-01-01"), np.datetime64("2007-01-01"))
YR = TIMES.astype("datetime64[Y]").astype(int) + 1970
VAL, TEST = (2002,), (2004,)
tr, va, te = (np.flatnonzero(~np.isin(YR, VAL + TEST)), np.flatnonzero(np.isin(YR, VAL)), np.flatnonzero(np.isin(YR, TEST)))
n = len(TIMES)
g = torch.Generator().manual_seed(1)
ocean = torch.rand(H, W, generator=g) > 0.2
X = torch.randn(n, NF, H, W, generator=g)
OBS = (torch.rand(n, H, W, generator=g) > 0.5) & ocean
Y = torch.randn(n, H, W, generator=g)
CLIM = torch.randn(366, H, W, generator=g) * 0.1
DOY = torch.as_tensor(AN.day_meta(TIMES)[1])
fns = lambda Y_, X_: dict(feats=lambda i: X_[i], chl_z=lambda i: Y_[i] * OBS[i], obs=lambda i: OBS[i], clim=lambda i: CLIM[DOY[i]])


def net(seed=0):
    torch.manual_seed(seed)
    return TokDecNet(NF + 2, 3, T, T // 2, ocean, P=P, dim=32, ch=8, heads=4, layers=1, blend=True, sst_index=0, backbone="dilated",
                     width=16, spatial=2, out_channels=3)


def inputs(days):
    B = len(days)
    torch.manual_seed(int(days[0]))
    x3d = torch.randn(B, NF + 2, T, H, W); x3d[:, -1] = (torch.rand(B, T, H, W) > 0.5).float()
    return x3d, torch.randn(B, 3, H, W), torch.ones(B, T, dtype=torch.bool), torch.randn(B, T, H, W)


def main():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}  {name}", flush=True)
        ok &= bool(cond)

    base = net()
    bank = AN.build_bank(base, TIMES, tr, val_test=(va, te), chunk=50, **fns(Y, X))
    yr, doy, dn = bank.meta
    check(f"bank holds only training days ({len(bank.days)} of {len(tr)})", np.array_equal(bank.days, tr))
    check("no val / test year in the bank", set(yr[bank.days]).isdisjoint(VAL + TEST))
    check(f"bank shape {tuple(bank.data.shape)} = (n_train, Np, 2 P^2 + 9)", bank.data.shape == (len(tr), base.Np, 2 * P * P + NF))
    rng = np.random.default_rng(0)
    q = rng.choice(tr, 200, replace=False)
    sel, dd = bank.draw(q, 16, 30, 30, rng=rng)
    s = sel >= 0
    dd_true = np.abs(doy[sel] - doy[q][:, None]); dd_true = np.minimum(dd_true, 365 - dd_true)
    check("training draws: no own-year analog", not (yr[sel][s] == np.repeat(yr[q][:, None], 16, 1)[s]).any())
    check("training draws: none within 30 calendar days", (np.abs(dn[sel] - dn[q][:, None])[s] > 30).all())
    check("training draws: all within +-30 DOY, all from the bank", (dd_true[s] <= 30).all() and np.isin(sel[s], bank.days).all())
    check("training draws: K filled", s.all())
    qv = np.r_[va[:20], te[:20]]
    sv, _ = bank.draw(qv, 16, 30, 30)
    sv2, _ = bank.draw(qv[::-1], 16, 30, 30)
    check("eval draws deterministic and batch-order independent", np.array_equal(sv, sv2[::-1]))
    check("eval draws for val / test days: training days only", np.isin(sv, tr).all())

    # 5. val / test data never reaches the bank
    Y2, X2 = Y.clone(), X.clone(); Y2[va] += 5; Y2[te] -= 5; X2[va] += 3
    bank2 = AN.build_bank(base, TIMES, tr, val_test=(va, te), chunk=50, **fns(Y2, X2))
    check("perturbing val / test days leaves the bank unchanged", torch.equal(bank.data, bank2.data))

    # 3. default path: zero-init projection -> identical output
    days = np.array([tr[100], tr[900], va[10]])
    x = inputs(days)
    plain = net().eval()
    m = AN.attach(net().eval(), bank, k=16)
    with torch.no_grad():
        y0 = plain(*x)
        m.analog.set_days(days); y1 = m(*x)
    check("attached model at init == model without --analog (bit-identical)", torch.equal(y0, y1))
    check("plain model state dict has no analog keys", not any(k.startswith("analog") for k in plain.state_dict()))
    from .train_pinn import parse
    a0 = parse(["--run", "x", "--model", "temporal3d", "--clim", "--t3d-stem", "tokdec", "--context", "8", "--feat-context", "8"])
    check("train_pinn.parse without --analog: analog off, defaults as RESUME_DEFAULTS",
          all(getattr(a0, k) == v for k, v in AN.RESUME_DEFAULTS.items()))
    try:
        unset = False; m(*x)
    except AssertionError:
        unset = True
    check("forward without set_days raises", unset)

    # 4. forward / backward with a non-zero projection; determinism in eval
    with torch.no_grad():
        m.analog.proj.weight.normal_(0, 0.1)
    m.train()
    m.analog.set_days(days); y = m(*x)
    y.square().mean().backward()
    nog = [k for k, p in m.analog.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    check(f"backward: every analog parameter has a gradient (missing: {nog})", not nog)
    m.analog.set_days(days); sel_a = m.analog.last["sel"]; m(*x); sel_b = m.analog.last["sel"]
    m.analog.set_days(days); m(*x); sel_c = m.analog.last["sel"]
    check("training draws vary between steps", not np.array_equal(sel_b, sel_c))
    m.eval()
    with torch.no_grad():
        m.analog.set_days(days); e1 = m(*x)
        m.analog.set_days(days); e2 = m(*x)
        xr = tuple(t.flip(0) for t in x)
        m.analog.set_days(days[::-1]); e3 = m(*xr).flip(0)
        m.analog.set_days(days); e0 = plain(*x); m.analog._days = None
    check("eval output deterministic over repeated calls", torch.equal(e1, e2))
    check("eval output independent of batch order", torch.allclose(e1, e3, atol=1e-5))
    check("analog path is live (non-zero projection changes the output)", not torch.allclose(e1, e0))
    print("ALL PASS" if ok else "SOME TESTS FAILED", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
