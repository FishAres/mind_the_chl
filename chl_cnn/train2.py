"""Performance runs: bigger models, richer inputs, loss choices. 1997-2020 store, year split.

Loss policy for target-image clouds
  Pixels cloudy in the target day (real clouds) have no truth and never enter the loss.
  Pixels hidden by the fake clouds are the task: full weight.
  Pixels still visible in the input are optional: weight --vis-weight (0 = ignore; the model
  sees their value as input, so a loss there only teaches copying). Val/early stopping and
  test scores are always the hidden-pixel MAE on the same fixed fake clouds as train.py.

Inputs (all z-scored; missing -> 0 plus a 0/1 flag where a flag makes sense)
  --features    physical fields from the store (sst so air_temp u_wind v_wind sla mlotst u_curr v_curr)
  CHL(t) with fake clouds applied + visible flag                                   [always]
  --context K   CHL(t+-1..K) + flags. The donor's clouds are applied to the neighbours too
                (donor day j+-k on target day t+-k), so hidden pixels are hidden across the
                sequence the way real clouds persist; --context-clouds none skips that
  --clim        day-of-year climatology of ln CHL from training days (8-day bins, 40-day smooth)
  --coords      lat, lon, log depth, sin/cos day of year
  --residual    the network predicts the anomaly from the climatology channel

Example
  python -m chl_cnn.train2 --run perf_dilated --model dilated --features sst so --clim --coords --context 2
Outputs -> models/cnn/<run>/ (models/unet/<run>/ for --model unet) : config.json, model.pt, history.json, metrics.json, preds.npz  (same layout as train.py)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import clouds as C
from . import store as S
from .evaluate import metrics, per_day_metrics, climatology_reference
from .models import build_model, count_params

from .runs import run_dir


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--model", choices=["simple", "dilated", "unet"], default="dilated")
    p.add_argument("--width", type=int, default=None, help="dilated: channels (64); unet: base channels (32)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--features", nargs="*", default=["sst", "so"])
    p.add_argument("--context", type=int, default=0, help="neighbour days +-K as extra CHL channels")
    p.add_argument("--context-clouds", choices=["persist", "none"], default="persist")
    p.add_argument("--clim", action="store_true")
    p.add_argument("--coords", action="store_true")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--loss", choices=["l1", "huber", "mse"], default="l1")
    p.add_argument("--huber-delta", type=float, default=0.5, help="in z-score units of ln CHL")
    p.add_argument("--vis-weight", type=float, default=0.0)
    p.add_argument("--clear-frac", type=float, default=None, help="train only on days with obs fraction >= F")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--sched", choices=["none", "cosine"], default="cosine")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.width is None:
        a.width = 32 if a.model == "unet" else 64
    if a.residual and not a.clim:
        p.error("--residual needs --clim")
    return a


def doy_climatology(y, obs, idx, nbin=46, halfwin=2):
    """Per-pixel mean ln CHL by day-of-year bin (8-day bins) over days idx, circularly smoothed
    over +-halfwin bins. Pixels never observed in a bin fall back to the all-year pixel mean, then
    to the global mean. Returns (nbin, H, W) float32 and the per-day bin index."""
    doy = (np.asarray(TIMES) - np.asarray(TIMES).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
    b = np.minimum(doy // 8, nbin - 1)
    s = np.zeros((nbin,) + y.shape[1:]); c = np.zeros_like(s)
    for i in idx:
        m = obs[i]
        s[b[i]][m] += y[i][m]; c[b[i]][m] += 1
    ks = np.arange(-halfwin, halfwin + 1)
    ss = sum(np.roll(s, k, axis=0) for k in ks); cc = sum(np.roll(c, k, axis=0) for k in ks)
    with np.errstate(invalid="ignore"):
        clim = ss / cc
    allyear = s.sum(0) / np.maximum(c.sum(0), 1)
    allyear[c.sum(0) == 0] = np.nan
    clim = np.where(cc > 0, clim, allyear[None])
    clim = np.where(np.isfinite(clim), clim, np.nanmean(allyear))
    return clim.astype("float32"), b


def neighbour_index(times, k):
    """Index of the day k days after each day (or -1 if that date is not in the record)."""
    days = np.asarray(times).astype("datetime64[D]")
    lut = {d: i for i, d in enumerate(days.astype(int).tolist())}
    return np.array([lut.get(t + k, -1) for t in days.astype(int).tolist()], dtype=np.int64)


def main(argv=None):
    global TIMES
    a = parse(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = run_dir(a.run, "train2", a.model)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    d = S.load_store()
    TIMES = d["time"]
    n, H, W = d["chl_l3"].shape
    ocean = d["ocean"]
    tr, va, te = D.split_by_years(TIMES)
    y = D.log_chl(np.asarray(d["chl_l3"]))
    obs = np.isfinite(y) & ocean
    obs_frac = obs.reshape(n, -1).sum(1) / ocean.sum()
    tr_all = tr
    if a.clear_frac is not None:
        tr = tr[obs_frac[tr] >= a.clear_frac]
    chl_mu, chl_sd = float(np.nanmean(y[tr])), float(np.nanstd(y[tr]))
    print(f"{n} days  train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    # fixed fake clouds for val/test (identical to train.py for the same eval seed)
    hid_va, don_va = C.fixed_hidden(obs, va, seed=a.eval_seed)
    hid_te, don_te = C.fixed_hidden(obs, te, seed=a.eval_seed + 1)
    hidden = np.zeros_like(obs); hidden[va], hidden[te] = hid_va, hid_te
    donor = np.zeros(n, dtype=np.int64); donor[va], donor[te] = don_va, don_te

    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    # physical features -> GPU float16, z-scored on a training subsample, NaN -> 0
    Xg = torch.empty((n, len(a.features), H, W), dtype=torch.float16, device=dev)
    feat_stats = {}
    sub = tr[::10]
    for c, f in enumerate(a.features):
        xs = np.asarray(d[f][sub]).astype("float32")
        mu, sd = float(np.nanmean(xs)), float(np.nanstd(xs)) + 1e-6
        feat_stats[f] = dict(mean=mu, std=sd, nan_frac_train=float(np.isnan(xs).mean()))
        for b0 in range(0, n, 500):
            x = (np.asarray(d[f][b0:b0 + 500]).astype("float32") - mu) / sd
            Xg[b0:b0 + 500, c] = torch.as_tensor(np.nan_to_num(x, nan=0.0), device=dev).half()
        del xs
    Yg, OBSg, HIDg, DONg = torch.nan_to_num(T(y), nan=0.0), T(obs), T(hidden), T(donor)
    CHLg = (Yg - chl_mu) / chl_sd * OBSg
    OCEANg = T(ocean)

    # optional channels
    offsets = [k for k in range(-a.context, a.context + 1) if k != 0]
    NB = {k: T(neighbour_index(TIMES, k)) for k in offsets}
    clim_np, doy_bin = (doy_climatology(y, obs, tr_all) if a.clim else (None, None))
    CLIMg = (T(clim_np) - chl_mu) / chl_sd if a.clim else None
    BINg = T(doy_bin) if a.clim else None
    if a.coords:
        lat, lon = np.meshgrid(d["lat"], d["lon"], indexing="ij")
        depth = np.log10(np.maximum(-d["topo"], 0) + 1)
        stat = np.stack([(lat - lat.mean()) / lat.std(), (lon - lon.mean()) / lon.std(), (depth - depth.mean()) / depth.std()])
        STATg = T(stat.astype("float32"))
        doy = (np.asarray(TIMES) - np.asarray(TIMES).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
        ang = 2 * np.pi * doy / 365.25
        DOYg = T(np.stack([np.sin(ang), np.cos(ang)], 1).astype("float32"))
    in_ch = len(a.features) + 2 + 2 * len(offsets) + (1 if a.clim else 0) + (5 if a.coords else 0)

    def make_inputs(bi, hid):
        vis = OBSg[bi] & ~hid
        ch = [Xg[bi].float(), (CHLg[bi] * vis).unsqueeze(1), vis.float().unsqueeze(1)]
        for k in offsets:
            j = NB[k][bi]; ok = (j >= 0).view(-1, 1, 1); j = j.clamp(min=0)
            m = OBSg[j] & ok
            if a.context_clouds == "persist":          # donor's cloud sequence hides the neighbours too
                jd = NB[k][DONg[bi]]; jd = torch.where(jd >= 0, jd, DONg[bi])
                m = m & OBSg[jd]
            ch += [(CHLg[j] * m).unsqueeze(1), m.float().unsqueeze(1)]
        if a.clim:
            ch.append(CLIMg[BINg[bi]].unsqueeze(1))
        if a.coords:
            ch.append(STATg.unsqueeze(0).expand(len(bi), -1, -1, -1))
            ch.append(DOYg[bi].view(-1, 2, 1, 1).expand(-1, -1, H, W))
        return torch.cat(ch, 1)

    def predict_z(model, bi, hid):
        z = model(make_inputs(bi, hid))
        if a.residual:
            z = z + CLIMg[BINg[bi]]
        return z

    def pixel_loss(pz, yz):
        e = pz - yz
        if a.loss == "l1":
            return e.abs()
        if a.loss == "mse":
            return e ** 2
        return torch.where(e.abs() < a.huber_delta, 0.5 * e ** 2, a.huber_delta * (e.abs() - 0.5 * a.huber_delta))

    model = build_model(a.model, in_ch, width=a.width, dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs) if a.sched == "cosine" else None
    scaler = torch.amp.GradScaler(enabled=not a.no_amp)
    YZg = (Yg - chl_mu) / chl_sd
    print(f"model {a.model} in_ch {in_ch} params {count_params(model)}", flush=True)

    def train_epoch():
        model.train()
        donors = tr[C.donor_indices(len(tr), rng)]
        HIDg[T(tr)] = T(obs[tr] & ~obs[donors]); DONg[T(tr)] = T(donors)
        tot, cnt = 0.0, 0
        order = T(rng.permutation(tr))
        for b in range(0, len(tr), a.batch):
            bi = order[b:b + a.batch]
            hid = HIDg[bi]
            w = hid.float() + a.vis_weight * (OBSg[bi] & ~hid).float()
            with torch.autocast("cuda", enabled=not a.no_amp):
                pz = predict_z(model, bi, hid)
            loss = (pixel_loss(pz.float(), YZg[bi]) * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(bi); cnt += len(bi)
        return tot / cnt

    @torch.no_grad()
    def hidden_mae(idx):
        """MAE in ln CHL on the fixed hidden pixels of days idx."""
        model.eval()
        err, cnt = 0.0, 0
        for b in range(0, len(idx), 32):
            bi = T(idx[b:b + 32]); hid = HIDg[bi]
            with torch.autocast("cuda", enabled=not a.no_amp):
                pz = predict_z(model, bi, hid)
            err += ((pz.float() * chl_sd + chl_mu - Yg[bi]).abs() * hid).sum().item(); cnt += hid.sum().item()
        return err / max(cnt, 1)

    hist = dict(train=[], val=[], lr=[], epoch_s=[])
    best, best_ep, best_state = np.inf, -1, None
    for ep in range(a.epochs):
        t0 = time.time()
        ltr = train_epoch()
        lva = hidden_mae(va)
        if sched is not None:
            sched.step()
        hist["train"].append(ltr); hist["val"].append(lva); hist["lr"].append(opt.param_groups[0]["lr"]); hist["epoch_s"].append(time.time() - t0)
        if lva < best:
            best, best_ep, best_state = lva, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1:3d}  train {ltr:.4f}  val_mae {lva:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        (out / "history.json").write_text(json.dumps(hist))   # updated every epoch so curves can be watched
        if ep - best_ep >= a.patience:
            print(f"early stop; best epoch {best_ep + 1} val {best:.4f}"); break
    model.load_state_dict(best_state)

    model.eval()
    ev = np.r_[va, te]
    preds = np.full((len(ev), H, W), np.nan, dtype="float32")
    with torch.no_grad():
        for b in range(0, len(ev), 32):
            bi = T(ev[b:b + 32])
            with torch.autocast("cuda", enabled=not a.no_amp):
                pz = predict_z(model, bi, HIDg[bi])
            preds[b:b + 32] = (pz.float() * chl_sd + chl_mu).cpu().numpy()
    preds[:, ~ocean] = np.nan
    pv, pt = preds[:len(va)], preds[len(va):]
    clim_ref = climatology_reference(y, tr_all)
    scores = {}
    for name, idx, pr, hid in [("val", va, pv, hid_va), ("test", te, pt, hid_te)]:
        scores[name] = dict(cnn_hidden=metrics(y[idx], pr, hid), cnn_observed=metrics(y[idx], pr, obs[idx]),
                            climatology_hidden=metrics(y[idx], clim_ref[idx], hid))
    m = scores["test"]
    print(f"TEST hidden-pixel MAE  cnn {m['cnn_hidden']['mae']:.3f}  clim {m['climatology_hidden']['mae']:.3f}   (n={m['cnn_hidden']['n']})")

    cfg = dict(vars(a), in_channels=in_ch, n_params=count_params(model), best_epoch=best_ep + 1, best_val_mae=best,
               n_days=dict(train=len(tr), train_before_clear_filter=len(tr_all), val=len(va), test=len(te)),
               feature_stats=feat_stats, chl_norm=dict(mean=chl_mu, std=chl_sd), wall_s=time.time() - t_start,
               split=dict(val_years=list(D.VAL_YEARS), test_years=list(D.TEST_YEARS)))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "history.json").write_text(json.dumps(hist))
    (out / "metrics.json").write_text(json.dumps(dict(
        scores=scores, per_day_test=per_day_metrics(y[te], pt, hid_te), per_day_val=per_day_metrics(y[va], pv, hid_va),
        obs_frac=dict(val=obs_frac[va].tolist(), test=obs_frac[te].tolist())), indent=1))
    torch.save(model.state_dict(), out / "model.pt")
    np.savez_compressed(out / "preds.npz", pred=preds, idx=ev, hidden=hidden[ev], val_idx=va, test_idx=te)
    print("wrote", out, f"({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
