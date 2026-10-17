"""Train the Simple CNN and score it on val/test pixels.

Inputs are environmental features (default sst, so, as in the baseline), optionally
plus two channels of CHL: observed ln(CHL) with gaps set to 0 and a 0/1 observed flag.
With --fake-clouds, extra gaps are punched into the CHL input every epoch and the
loss is taken only on those hidden pixels, so the model learns to fill clouds.

Target modes
  observed   loss on observed L3 pixels only (NaNs ignored)          [default]
  hidden     loss on fake-cloud-hidden pixels only (needs --chl-input)
  all_zero   the baseline as published: NaN -> 0, loss on every pixel incl. land

--clear-frac F keeps only training days whose L3 observed fraction (of ocean pixels) is >= F,
so the label is (nearly) complete; combined with --fake-clouds --target observed this trains on
clear days with clouds added from other days. Val loss is always the hidden-pixel MAE when fake
clouds are on, so runs are early-stopped on the same task.

Examples
  python -m chl_cnn.train --run baseline_sst_so
  python -m chl_cnn.train --run chl_in_fakeclouds --chl-input --fake-clouds
  python -m chl_cnn.train --run full_clear --years 1997-09-01 2020-12-31 --chl-input --fake-clouds --target observed --clear-frac 0.7
Outputs -> models/cnn/<run>/ : config.json, model.pt, history.json, metrics.json, preds.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import clouds as C
from .evaluate import metrics, per_day_metrics, climatology_reference
from .model import SimpleCNN, count_params

from .runs import run_dir


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--features", nargs="+", default=["sst", "so"])
    p.add_argument("--chl-input", action="store_true", help="add ln(CHL L3) + observed-flag input channels")
    p.add_argument("--fake-clouds", action="store_true", help="punch fake clouds into the CHL input; score hidden pixels")
    p.add_argument("--target", choices=["observed", "hidden", "all_zero"], default=None)
    p.add_argument("--years", nargs=2, default=list(D.YEARS), metavar=("START", "END"))
    p.add_argument("--split", choices=["chrono", "years"], default="chrono",
                   help="chrono: 70/20/10 in time order; years: val/test years interleaved (data.VAL_YEARS/TEST_YEARS)")
    p.add_argument("--clear-frac", type=float, default=None, help="train only on days with L3 observed fraction >= F")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0, help="seed of the fixed val/test fake clouds (shared by runs)")
    a = p.parse_args(argv)
    if a.fake_clouds and not a.chl_input:
        p.error("--fake-clouds needs --chl-input")
    if a.target is None:
        a.target = "hidden" if a.fake_clouds else "observed"
    if a.target == "hidden" and not a.fake_clouds:
        p.error("--target hidden needs --fake-clouds")
    return a


def masked_mae(pred, y, mask):
    mask = mask.float()
    err = (pred - y).abs() * mask
    return err.sum() / mask.sum().clamp(min=1)


def make_inputs(feat_z, chl_z, obs, hidden=None):
    """Concatenate feature channels with (optional) CHL channels. Tensors on the same device."""
    if chl_z is None:
        return feat_z
    vis = obs if hidden is None else obs & ~hidden
    return torch.cat([feat_z, (chl_z * vis).unsqueeze(1), vis.float().unsqueeze(1)], dim=1)


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_dir(a.run, "train", "simple")
    out.mkdir(parents=True, exist_ok=True)

    years = tuple(a.years)
    variables = None if years == D.YEARS else tuple(a.features)   # full-period cache holds only what is needed
    d = D.build_subset(years=years, variables=variables, verbose=False)
    n = len(d["time"])
    tr, va, te = D.split_indices(n) if a.split == "chrono" else D.split_by_years(d["time"])
    y = D.log_chl(d["chl_l3"])                       # (N,H,W) ln CHL, NaN where cloudy/land
    obs = np.isfinite(y) & d["ocean"]                # scored pixels are ocean only
    obs_frac = obs.reshape(n, -1).sum(1) / d["ocean"].sum()
    tr_all = tr
    if a.clear_frac is not None:
        tr = tr[obs_frac[tr] >= a.clear_frac]
        print(f"clear-day filter >= {a.clear_frac}: {len(tr)} / {len(tr_all)} training days")
    # features -> standardised (N,C,H,W), built channel by channel to keep one full-size copy in RAM
    norm = D.Normaliser(np.stack([d[f][tr] if d[f].ndim == 3 else np.broadcast_to(d[f], (len(tr),) + d[f].shape)
                                  for f in a.features], axis=1))
    X = np.empty((n, len(a.features)) + y.shape[1:], dtype="float32")
    for c, f in enumerate(a.features):
        X[:, c] = d[f] if d[f].ndim == 3 else d[f][None]
        del d[f]
        X[:, c] -= norm.mean[0, c]; X[:, c] /= norm.std[0, c]
    np.nan_to_num(X, nan=0.0, copy=False)
    chl_mu, chl_sd = float(np.nanmean(y[tr])), float(np.nanstd(y[tr]))

    # fixed fake clouds for val and test (same for every run with the same eval seed)
    hid_va, _ = C.fixed_hidden(obs, va, seed=a.eval_seed)
    hid_te, _ = C.fixed_hidden(obs, te, seed=a.eval_seed + 1)
    hidden = np.zeros_like(obs)
    hidden[va], hidden[te] = hid_va, hid_te

    if a.target == "all_zero":
        loss_mask = np.ones_like(obs)
    elif a.target == "hidden":
        loss_mask = hidden.copy()                      # train part is refreshed per epoch
    else:
        loss_mask = obs.copy()
    if a.fake_clouds:
        loss_mask[va] = hidden[va]                     # early-stop on the hidden-pixel task in every fake-cloud run

    T = lambda x: torch.as_tensor(x, device=dev)
    Xg, OBSg, HIDg, LMg = T(X), T(obs), T(hidden), T(loss_mask)
    del X
    Yg = torch.nan_to_num(T(y), nan=0.0)               # target with NaN -> 0 (masked out by LMg)
    CHLg = (Yg - chl_mu) / chl_sd * OBSg if a.chl_input else None
    in_ch = len(a.features) + (2 if a.chl_input else 0)
    model = SimpleCNN(in_ch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def run_epoch(idx, train):
        model.train(train)
        tot, cnt = 0.0, 0
        order = rng.permutation(idx) if train else idx
        for b in range(0, len(order), a.batch):
            bi = torch.as_tensor(order[b:b + a.batch], device=dev)
            hid = HIDg[bi]
            xb = make_inputs(Xg[bi], CHLg[bi] if CHLg is not None else None, OBSg[bi], hid if a.fake_clouds else None)
            with torch.set_grad_enabled(train):
                pred = model(xb)
                loss = masked_mae(pred, Yg[bi], LMg[bi])
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi); cnt += len(bi)
        return tot / cnt

    hist = dict(train=[], val=[])
    best, best_ep, best_state, t0 = np.inf, -1, None, time.time()
    for ep in range(a.epochs):
        if a.fake_clouds:  # new fake clouds on the training days every epoch
            donors = tr[C.donor_indices(len(tr), rng)]      # global day indices
            h = obs[tr] & ~obs[donors]
            HIDg[T(tr)] = T(h)
            if a.target == "hidden":
                LMg[T(tr)] = T(h)
        ltr = run_epoch(tr, True)
        lva = run_epoch(va, False)
        hist["train"].append(ltr); hist["val"].append(lva)
        if lva < best:
            best, best_ep, best_state = lva, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1:3d}  train {ltr:.4f}  val {lva:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if ep - best_ep >= a.patience:
            print(f"early stop; best epoch {best_ep + 1} val {best:.4f}")
            break
    model.load_state_dict(best_state)

    # predictions for every day, with the fixed val/test fake clouds applied to the CHL input
    model.eval()
    ev = np.r_[va, te]
    preds = np.full((n,) + y.shape[1:], np.nan, dtype="float32")   # only val/test days are filled (~1 GB for 1997-2020)
    with torch.no_grad():
        for b in range(0, len(ev), 32):
            bi = torch.as_tensor(ev[b:b + 32], device=dev)
            xb = make_inputs(Xg[bi], CHLg[bi] if CHLg is not None else None, OBSg[bi], HIDg[bi] if a.chl_input else None)
            preds[ev[b:b + 32]] = model(xb).cpu().numpy()
    preds[:, ~d["ocean"]] = np.nan

    clim = climatology_reference(y, tr_all)
    l4 = D.log_chl(d["chl_l4"]) if "chl_l4" in d else None
    scores = {}
    for name, idx, hid in [("val", va, hid_va), ("test", te, hid_te)]:
        scores[name] = {
            "cnn_hidden": metrics(y[idx], preds[idx], hid),
            "cnn_observed": metrics(y[idx], preds[idx], obs[idx]),
            "climatology_hidden": metrics(y[idx], clim[idx], hid),
        }
        if l4 is not None:
            scores[name]["l4_gapfree_hidden"] = metrics(y[idx], l4[idx], hid)
    m = scores["test"]
    print(f"TEST hidden-pixel MAE  cnn {m['cnn_hidden']['mae']:.3f}  clim {m['climatology_hidden']['mae']:.3f}   (n={m['cnn_hidden']['n']})")

    cfg = dict(vars(a), in_channels=in_ch, n_params=count_params(model), best_epoch=best_ep + 1,
               n_days=dict(train=len(tr), train_before_clear_filter=len(tr_all), val=len(va), test=len(te)),
               dates=dict(train=[str(d["time"][tr][0])[:10], str(d["time"][tr][-1])[:10]],
                          val=[str(d["time"][va][0])[:10], str(d["time"][va][-1])[:10]],
                          test=[str(d["time"][te][0])[:10], str(d["time"][te][-1])[:10]]),
               normaliser=norm.state(),
               chl_norm=dict(mean=chl_mu, std=chl_sd), device=str(dev))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "history.json").write_text(json.dumps(hist))
    (out / "metrics.json").write_text(json.dumps(dict(
        scores=scores, per_day_test=per_day_metrics(y[te], preds[te], hid_te),
        per_day_val=per_day_metrics(y[va], preds[va], hid_va),
        obs_frac=dict(val=obs_frac[va].tolist(), test=obs_frac[te].tolist())), indent=1))
    torch.save(model.state_dict(), out / "model.pt")
    np.savez_compressed(out / "preds.npz", pred=preds[np.r_[va, te]].astype("float32"), idx=np.r_[va, te],
                        hidden=hidden[np.r_[va, te]], val_idx=va, test_idx=te)
    print("wrote", out)


if __name__ == "__main__":
    main()
