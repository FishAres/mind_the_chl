"""Forecast baselines at 1-3 day leads: persistence vs semi-Lagrangian advected persistence vs DOY climatology
vs blends vs anomaly persistence, on the IO and Black Sea stores.

Run:  python analyses/ude/forecast_baselines.py --store io|blk [--val-stride 2] [--train-stride 4] [--batch 16]
Outputs analyses/ude/results/<store>.json and analyses/ude/figures/mae_vs_lead_<store>.png.

Method (per target day t, lead k):
  persistence         C(t-k, x)
  advected            C(t-k, X), X = departure point of x integrated backward with the daily currents of days
                      t-1 .. t-k (each held fixed, 4 sub-steps/day), bilinear grid_sample of the source field;
                      accepted where the sampled source-observation mask >= 0.99 and the trajectory stayed on
                      ocean pixels with valid currents at every sub-step; else fallback to persistence.
  clim                clim(t, x)   (DOY climatology from training years)
  blend               a C(t-k,x) + (1-a) clim(t,x), a by least squares on training days (per k, per store)
  adv blend           same with the advected field (own a)
  anom                C(t-k,x) - clim(t-k,x) + clim(t,x)
  adv anom            C(t-k,X) - clim(t-k,X) + clim(t,x)
Scored on the common pixel set: observed at t and at t-k.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chl_cnn import data as D, train2 as T2, physics as P
from chl_cnn.plotting import OKABE_ITO

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
RES = HERE / "results"; RES.mkdir(exist_ok=True)
METHODS = ["persistence", "advected", "clim", "blend", "adv_blend", "anom", "adv_anom"]
LEADS = (1, 2, 3)
SUBSTEPS = 4
CHL_MAX_BLK = 100.0


def load(store):
    if store == "io":
        from chl_cnn import store as S
        d = S.load_store()
        tr, va, te = D.split_by_years(d["time"])
        regions = None
    else:
        from chl_cnn import store_blk as S
        d = S.load_store()
        tr, va, te = D.split_by_years(d["time"], (2017, 2021), (2019, 2023))
        regions = np.load(ROOT / "analyses/blk/regions.npy")
    n, H, W = d["chl_l3"].shape
    y = np.empty((n, H, W), dtype="float32"); n_out = 0; n_zero = 0
    for b0 in range(0, n, 200):
        c = np.asarray(d["chl_l3"][b0:b0 + 200]).astype("float32")
        if store == "blk":
            bad = c > CHL_MAX_BLK; n_out += int(bad.sum()); c[bad] = np.nan
        n_zero += int((c == 0).sum())
        y[b0:b0 + 200] = D.log_chl(c)
    ocean = d["ocean"].astype(bool)
    obs = np.isfinite(y) & ocean
    y[~obs] = 0.0           # NaN -> 0 (masked by obs everywhere below)
    return d, y, obs, ocean, tr, va, te, regions, dict(n_chl_gt_max=n_out, n_chl_zero=n_zero)


def to_grid(X, Y, H, W):
    """Index coordinates (B,H,W) -> grid_sample grid (B,H,W,2) with align_corners=True."""
    return torch.stack([2 * X / (W - 1) - 1, 2 * Y / (H - 1) - 1], -1)


def sample(field, grid):
    """field (B,C,H,W) -> (B,C,H,W) bilinear at grid; outside -> 0."""
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class Advect:
    sign = 1.0   # -1 flips the currents (direction check)
    scale = 1.0  # multiplies the currents
    flip = "none"  # orientation diagnostics: u | v | uv | swap

    def __init__(self, d, ocean, dev):
        lat, lon = d["lat"].astype("float64"), d["lon"].astype("float64")
        dx, dy, _, _ = P.grid_metrics(lat, lon)
        self.inv_dx = torch.as_tensor((1.0 / dx).astype("float32"), device=dev)   # cells per metre, signed
        self.inv_dy = torch.as_tensor((1.0 / dy).astype("float32"), device=dev)
        self.ocean = torch.as_tensor(ocean.astype("float32"), device=dev)
        self.H, self.W = ocean.shape
        self.d, self.dev = d, dev
        yy, xx = torch.meshgrid(torch.arange(self.H, device=dev, dtype=torch.float32),
                                torch.arange(self.W, device=dev, dtype=torch.float32), indexing="ij")
        self.X0, self.Y0 = xx, yy

    def currents(self, idx):
        """(B,2,H,W) cells/s (NaN -> 0) and validity (B,1,H,W) for store days idx."""
        u = torch.as_tensor(np.asarray(self.d["u_curr"][idx]).astype("float32"), device=self.dev)
        v = torch.as_tensor(np.asarray(self.d["v_curr"][idx]).astype("float32"), device=self.dev)
        ok = (torch.isfinite(u) & torch.isfinite(v) & (self.ocean > 0)).float()
        u, v = torch.nan_to_num(u), torch.nan_to_num(v)
        if self.flip == "swap":
            u, v = v, u
        if self.flip in ("u", "uv"):
            u = -u
        if self.flip in ("v", "uv"):
            v = -v
        u = u * self.inv_dx; v = v * self.inv_dy      # cells/s along increasing column / row index (dy signed)
        return torch.stack([u, v], 1) * ok.unsqueeze(1) * (self.sign * self.scale), ok.unsqueeze(1)

    def displacement_cells(self, idx):
        """|displacement| in cells/day at ocean pixels with valid currents, flattened."""
        vel, ok = self.currents(idx)
        disp = (vel * 86400.0).norm(dim=1)
        return disp[ok[:, 0] > 0].cpu().numpy()

    def back_trajectory(self, day_lists):
        """day_lists: (B, k) store indices of the current days t-1..t-k. Returns X, Y (B,H,W) and valid (B,H,W) bool
        (currents valid and on ocean at every sub-step)."""
        B, k = day_lists.shape
        X = self.X0.expand(B, -1, -1).clone(); Y = self.Y0.expand(B, -1, -1).clone()
        valid = torch.ones(B, self.H, self.W, dtype=torch.bool, device=self.dev)
        dt = 86400.0 / SUBSTEPS
        for j in range(k):
            vel, ok = self.currents(day_lists[:, j])
            for _ in range(SUBSTEPS):
                g = to_grid(X, Y, self.H, self.W)
                v_here = sample(vel, g); ok_here = sample(ok, g)[:, 0]
                oc_here = sample(self.ocean.expand(B, 1, -1, -1), g)[:, 0]
                valid &= (ok_here >= 0.99) & (oc_here >= 0.99)
                X = X - v_here[:, 0] * dt; Y = Y - v_here[:, 1] * dt
        return X, Y, valid


def run_days(targets, k, y, obs, clim, bins, nb_prev, adv, dev, batch, keep_err=False, keep_mask=None):
    """Return dict of per-method sum |err| and count on the common set, fallback counts, and optionally per-pixel
    abs errors on masks for region tables. targets: store indices t with t-1..t-k all present."""
    H, W = obs.shape[1:]
    sums = {m: 0.0 for m in METHODS}; cnt = 0; n_obs_t = 0; n_obs_src = 0
    fb = dict(traj=0, src_unobs=0, total=0, n=0)
    ls = dict(pers=[0.0, 0.0], adv=[0.0, 0.0])    # [sum (C-clim)(y-clim), sum (C-clim)^2]
    reg_acc = {}
    acc_sub = dict(pers=0.0, adv=0.0, n=0)     # accepted (non-fallback) pixels only
    T = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt)
    for b0 in range(0, len(targets), batch):
        bt = targets[b0:b0 + batch]
        src = nb_prev[k][bt]
        days = np.stack([nb_prev[j][bt] for j in range(1, k + 1)], 1)       # t-1 .. t-k
        Yt, Ot = T(y[bt]), T(obs[bt], torch.bool)
        Ys, Os = T(y[src]), T(obs[src], torch.bool)
        Ct, Cs = T(clim[bins[bt]]), T(clim[bins[src]])
        X, Yc, valid = adv.back_trajectory(days)
        g = to_grid(X, Yc, H, W)
        fld = torch.stack([Ys, Os.float(), Cs], 1)
        s = sample(fld, g)
        src_ok = s[:, 1] >= 0.99
        acc = valid & src_ok
        Ya = torch.where(acc, s[:, 0], Ys)             # fallback to persistence
        Ca = torch.where(acc, s[:, 2], Cs)
        common = Ot & Os
        n_obs_t += int(Ot.sum()); n_obs_src += int(Os.sum())
        fb["n"] += int(common.sum()); fb["traj"] += int((common & ~valid).sum())
        fb["src_unobs"] += int((common & valid & ~src_ok).sum()); fb["total"] += int((common & ~acc).sum())
        preds = dict(persistence=Ys, advected=Ya, clim=Ct, anom=Ys - Cs + Ct, adv_anom=Ya - Ca + Ct)
        if "a" in run_days.__dict__:
            a = run_days.a
            preds["blend"] = a["pers"] * Ys + (1 - a["pers"]) * Ct
            preds["adv_blend"] = a["adv"] * Ya + (1 - a["adv"]) * Ct
        else:
            preds["blend"] = preds["adv_blend"] = Ct
        cm = common.float()
        for m in METHODS:
            sums[m] += float(((preds[m] - Yt).abs() * cm).sum())
        cnt += int(common.sum())
        am = (common & acc).float()
        acc_sub["pers"] += float(((Ys - Yt).abs() * am).sum()); acc_sub["adv"] += float(((Ya - Yt).abs() * am).sum()); acc_sub["n"] += int(am.sum())
        dy_ = (Yt - Ct)
        for key, Yp in [("pers", Ys), ("adv", Ya)]:
            dc = (Yp - Ct)
            ls[key][0] += float((dc * dy_ * cm).sum()); ls[key][1] += float((dc * dc * cm).sum())
        if keep_mask is not None:
            for r, rm in keep_mask.items():
                mm = common & rm
                acc_r = reg_acc.setdefault(r, {m: 0.0 for m in METHODS} | {"n": 0, "fb": 0})
                acc_r["n"] += int(mm.sum()); acc_r["fb"] += int((mm & ~acc).sum())
                for m in METHODS:
                    acc_r[m] += float(((preds[m] - Yt).abs() * mm.float()).sum())
    out = dict(mae={m: sums[m] / max(cnt, 1) for m in METHODS}, n_common=cnt, n_obs_t=n_obs_t, n_obs_src=n_obs_src,
               fallback=fb, ls=ls, n_days=len(targets),
               accepted_only=dict(mae_persistence=acc_sub["pers"] / max(acc_sub["n"], 1), mae_advected=acc_sub["adv"] / max(acc_sub["n"], 1), n=acc_sub["n"]))
    if keep_mask is not None:
        out["regions"] = {r: {"mae": {m: a[m] / max(a["n"], 1) for m in METHODS}, "n": a["n"], "fallback_frac": a["fb"] / max(a["n"], 1)}
                          for r, a in reg_acc.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="io"); ap.add_argument("--val-stride", type=int, default=2)
    ap.add_argument("--train-stride", type=int, default=4); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reverse", action="store_true", help="flip the sign of the currents (direction check); results tagged _rev")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the currents by this factor")
    ap.add_argument("--flip", default="none", choices=["none", "u", "v", "uv", "swap"], help="orientation diagnostics")
    ap.add_argument("--leads", default="1,2,3")
    a = ap.parse_args()
    Advect.sign = -1.0 if a.reverse else 1.0; Advect.scale = a.scale; Advect.flip = a.flip
    leads = tuple(int(x) for x in a.leads.split(","))
    tag = a.store + ("_rev" if a.reverse else "") + (f"_scale{a.scale:g}" if a.scale != 1 else "") + (f"_flip{a.flip}" if a.flip != "none" else "")
    t0 = time.time(); dev = torch.device("cuda")
    d, y, obs, ocean, tr, va, te, regions, dq = load(a.store)
    times = d["time"]; T2.TIMES = times
    clim, bins = T2.doy_climatology(y, obs, tr)
    print(f"{a.store}: {len(times)} days, train {len(tr)} val {len(va)}; dq {dq}; nan frac on ocean {1 - obs[:, ocean].mean():.3f}; "
          f"clim built {time.time() - t0:.0f}s", flush=True)
    nb_prev = {j: T2.neighbour_index(times, -j) for j in leads}
    adv = Advect(d, ocean, dev)
    # displacement statistics on the val days used
    va_used = va[::a.val_stride]
    disp = np.concatenate([adv.displacement_cells(va_used[i:i + 16]) for i in range(0, len(va_used), 16)])
    disp_stats = dict(median=float(np.median(disp)), p90=float(np.percentile(disp, 90)), n_px=int(len(disp)))
    print("displacement cells/day", disp_stats, flush=True)
    reg_masks = None
    if regions is not None:
        names = {int(k): v for k, v in json.load(open(ROOT / "analyses/blk/region_names.json")).items()}
        reg_masks = {names[r]: torch.as_tensor(regions == r, device=dev) for r in names}
    res = dict(store=a.store, dq=dq, displacement=disp_stats, val_stride=a.val_stride, train_stride=a.train_stride, leads={})
    for k in leads:
        def usable(idx):
            ok = np.all([nb_prev[j][idx] >= 0 for j in range(1, k + 1)], 0)
            return idx[ok]
        tr_k = usable(tr[::a.train_stride]); va_k = usable(va_used)
        if "a" in run_days.__dict__:
            del run_days.a
        fit = run_days(tr_k, k, y, obs, clim, bins, nb_prev, adv, dev, a.batch)
        run_days.a = {key: fit["ls"][key][0] / fit["ls"][key][1] for key in ["pers", "adv"]}
        sc = run_days(va_k, k, y, obs, clim, bins, nb_prev, adv, dev, a.batch, keep_mask=(reg_masks if k == 1 else None))
        sc["a"] = dict(run_days.a); sc["n_train_days"] = len(tr_k); sc["n_val_days_requested"] = len(va_used)
        sc["fallback_frac"] = {kk: sc["fallback"][kk] / max(sc["fallback"]["n"], 1) for kk in ["traj", "src_unobs", "total"]}
        res["leads"][k] = sc
        print(f"k={k} val days {len(va_k)} common px {sc['n_common']} a={sc['a']} fb={sc['fallback_frac']} "
              f"mae={ {m: round(v, 4) for m, v in sc['mae'].items()} }  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(RES / f"{tag}.json", "w"), indent=1, default=float)
    if tag == a.store:
        plot(res, a.store)


def plot(res, store):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ks = sorted(res["leads"]); name = {"io": "Indian Ocean", "blk": "Black Sea"}[store]
    styles = {"persistence": "-", "advected": "--", "clim": "-", "blend": "-", "adv_blend": "--", "anom": "-", "adv_anom": "--"}
    cols = {"persistence": 0, "advected": 0, "clim": 7, "blend": 1, "adv_blend": 1, "anom": 2, "adv_anom": 2}
    for m in METHODS:
        ax.plot(ks, [res["leads"][k]["mae"][m] for k in ks], styles[m], marker="o", color=OKABE_ITO[cols[m]], label=m.replace("_", " "))
    ax.set_xlabel("lead k (days)"); ax.set_ylabel("MAE of ln CHL (common pixel set)"); ax.set_xticks(ks)
    n = res["leads"][ks[0]]["n_val_days_requested"]
    ax.set_title(f"{name}: forecast MAE vs lead, val days (every {res['val_stride']}nd, n={n})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / f"mae_vs_lead_{store}.png", dpi=150)


def plot_sweep():
    """MAE vs current scale factor (BLK) plus orientation variants, from results/blk_scale*.json and blk_flip*.json."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    scales = [0.25, 0.5, 0.75, 1.0, 1.5]
    R = {sc: json.load(open(RES / ("blk.json" if sc == 1 else f"blk_scale{sc:g}.json"))) for sc in scales}
    flips = {f: json.load(open(RES / f"blk_flip{f}.json")) for f in ["u", "v", "uv", "swap"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, k in enumerate(["1", "2"]):
        ax = axes[i]
        ax.plot(scales, [R[sc]["leads"][k]["mae"]["advected"] for sc in scales], "o-", color=OKABE_ITO[0], label="advected persistence")
        ax.axhline(R[1.0]["leads"][k]["mae"]["persistence"], color=OKABE_ITO[3], ls="--", label="persistence")
        for j, (f, r) in enumerate(flips.items()):
            ax.plot([1.0], [r["leads"][k]["mae"]["advected"]], marker="sD^v"[j], ls="none", color=OKABE_ITO[2], label=f"advected, flip {f}")
        ax.set_xlabel("current scale factor alpha"); ax.set_ylabel("MAE of ln CHL (common pixel set)")
        ax.set_title(f"Black Sea, lead {k} d: advected MAE vs current scale"); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "blk_current_scale_sweep.png", dpi=150)


if __name__ == "__main__":
    if "--plot-sweep" in sys.argv:
        plot_sweep()
    else:
        main()
