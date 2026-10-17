"""Hidden-pixel error breakdown of the best Black Sea 4 km run (and the lambda 0 reference on the same pixels)
by region, calendar month, day observed fraction (quintiles) and truth ln CHL (deciles). DOY climatology
(train days, as train_pinn.py --clim) is scored on the same pixels.

Error = pred - truth on hidden pixels (preds.npz 'hidden'). Figures on val days; tables for val and test in
analyses/blk4/results/error_breakdown.json.
Run from the repo root: python -m analyses.blk4.error_breakdown   (CPU only)
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chl_cnn.viz import context_for
from chl_cnn.plotting import OKABE_ITO

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
RUNS = {"sl_lam0p3_cont (best)": ROOT / "models/pinn/t3d/blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50",
        "lam0_b16_50ep": ROOT / "models/pinn/t3d/blk4_t3d_c4_gdl100_lam0_b16_50ep"}
MODELS = list(RUNS) + ["DOY climatology"]
NBIN, HALFWIN = 46, 2                       # train2.doy_climatology defaults
MIN_N_DAYS = 5


# ============================== data ==============================
def regions_4km(ocean4):
    """2 km region labels -> 4 km cropped grid: 2x2 blocks (last 2 km row dropped, as store_blk4), mode of the
    non-zero labels in the block (ties -> smaller label), then the train_pinn --crop slice of the 4 km ocean."""
    from chl_cnn import store_blk4 as S4
    reg2 = np.load(ROOT / "analyses/blk/regions.npy")
    H2, W2 = reg2.shape[0] // 2, reg2.shape[1] // 2
    b = S4._blocks(reg2, H2, W2)                                         # (H2, W2, 4)
    cnt = np.stack([(b == k).sum(-1) for k in range(1, 8)], -1)         # counts of labels 1..7
    reg4 = np.where(cnt.max(-1) > 0, cnt.argmax(-1) + 1, 0).astype(np.int8)
    oc = np.asarray(S4.load_store()["ocean"])
    assert oc.shape == reg4.shape
    rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
    ys = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0]))
    xs = slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
    reg = reg4[ys, xs]
    assert reg.shape == ocean4.shape and (oc[ys, xs] == ocean4).all()
    ties = int(((np.sort(cnt, -1)[..., -1] == np.sort(cnt, -1)[..., -2]) & (cnt.max(-1) > 0))[ys, xs][ocean4].sum())
    return reg, ties


def doy_clim(ctx):
    """Same as train2.doy_climatology(y, obs, tr), streamed one day at a time."""
    times = np.asarray(ctx["times"])
    doy = (times - times.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
    b = np.minimum(doy // 8, NBIN - 1)
    shp = ctx["ocean"].shape
    s = np.zeros((NBIN,) + shp); c = np.zeros_like(s)
    for i in ctx["tr"]:
        y = ctx["y"](i); m = np.isfinite(y) & ctx["ocean"]
        s[b[i]][m] += y[m]; c[b[i]][m] += 1
    ks = np.arange(-HALFWIN, HALFWIN + 1)
    ss = sum(np.roll(s, k, axis=0) for k in ks); cc = sum(np.roll(c, k, axis=0) for k in ks)
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = ss / cc
    allyear = s.sum(0) / np.maximum(c.sum(0), 1); allyear[c.sum(0) == 0] = np.nan
    clim = np.where(cc > 0, clim, allyear[None])
    clim = np.where(np.isfinite(clim), clim, np.nanmean(allyear))
    return clim.astype("float32"), b


def load_pixels():
    """Flat per-hidden-pixel table for each split: truth, error per model, region, day index."""
    ctx = context_for(RUNS["sl_lam0p3_cont (best)"])
    preds = {k: np.load(p / "preds.npz") for k, p in RUNS.items()}
    z0 = preds["sl_lam0p3_cont (best)"]
    for k, z in preds.items():
        assert (z["idx"] == z0["idx"]).all() and (z["hidden"] == z0["hidden"]).all(), k
    reg, ties = regions_4km(ctx["ocean"])
    clim, cbin = doy_clim(ctx)
    pos = {int(i): k for k, i in enumerate(z0["idx"])}
    dq = dict(region_ties=ties)
    out = {}
    for split, idx in (("val", z0["val_idx"]), ("test", z0["test_idx"])):
        cols = {c: [] for c in ["truth", "day", "region"] + MODELS}
        n_hid_not_ocean = n_truth_nan = 0; n_pred_nan = {k: 0 for k in RUNS}; empty_days = 0
        for i in idx:
            k = pos[int(i)]; hid = z0["hidden"][k]
            n_hid_not_ocean += int((hid & ~ctx["ocean"]).sum())
            y = ctx["y"](int(i))
            n_truth_nan += int((hid & ~np.isfinite(y)).sum())
            m = hid & ctx["ocean"] & np.isfinite(y)
            if not m.any():
                empty_days += 1; continue
            cols["truth"].append(y[m]); cols["day"].append(np.full(m.sum(), int(i)))
            cols["region"].append(reg[m])
            for name, z in preds.items():
                p = z["pred"][k][m].astype("float32"); n_pred_nan[name] += int((~np.isfinite(p)).sum())
                cols[name].append(p - y[m])
            cols["DOY climatology"].append(clim[cbin[int(i)]][m] - y[m])
        out[split] = {c: np.concatenate(v) for c, v in cols.items()}
        dq[split] = dict(n_days=int(len(idx)), days_without_hidden_px=empty_days, hidden_px_off_ocean=n_hid_not_ocean,
                         hidden_px_truth_nan=n_truth_nan, pred_nan_on_hidden=n_pred_nan,
                         hidden_px_unlabelled_region=int((out[split]["region"] == 0).sum()))
    n_oc = int(ctx["ocean"].sum())
    dq["ocean_px_4km"] = n_oc
    dq["ocean_px_unlabelled"] = int((ctx["ocean"] & (reg == 0)).sum())
    dq["pct_ocean_px_unlabelled"] = 100 * dq["ocean_px_unlabelled"] / n_oc
    dq["region_px_4km"] = {int(r): int(((reg == r) & ctx["ocean"]).sum()) for r in range(8)}
    dq["labelled_px_off_ocean"] = int(((reg > 0) & ~ctx["ocean"]).sum())
    return ctx, out, dq


# ============================== analysis ==============================
def group_stats(px, gid, labels):
    """MAE, bias, n px, n days per group, for each model. gid: int group per pixel (-1 = none)."""
    rows = []
    for g, lab in enumerate(labels):
        m = gid == g
        r = dict(group=lab, n_px=int(m.sum()), n_days=int(len(np.unique(px["day"][m]))))
        for mod in MODELS:
            e = px[mod][m]
            r[mod] = dict(mae=float(np.abs(e).mean()) if m.any() else None, bias=float(e.mean()) if m.any() else None)
        r["flag_n_days_lt5"] = r["n_days"] < MIN_N_DAYS
        rows.append(r)
    return rows


def breakdowns(ctx, px, names, truth_edges, of_edges):
    months = np.asarray(ctx["times"]).astype("datetime64[M]").astype(int) % 12 + 1
    of = ctx["obs_frac"]
    res = {}
    res["overall"] = group_stats(px, np.zeros(len(px["truth"]), int), ["all"])
    res["region"] = group_stats(px, px["region"].astype(int) - 1, [names[r] for r in range(1, 8)])
    res["month"] = group_stats(px, months[px["day"]] - 1, [f"{m:02d}" for m in range(1, 13)])
    q = np.clip(np.searchsorted(of_edges, of[px["day"]], side="right") - 1, 0, len(of_edges) - 2)
    res["obs_frac_quintile"] = group_stats(px, q, [f"Q{j + 1} [{of_edges[j]:.3f}, {of_edges[j + 1]:.3f}]" for j in range(5)])
    t = np.clip(np.searchsorted(truth_edges, px["truth"], side="right") - 1, 0, len(truth_edges) - 2)
    res["truth_decile"] = group_stats(px, t, [f"D{j + 1} [{truth_edges[j]:.2f}, {truth_edges[j + 1]:.2f}]" for j in range(10)])
    return res


# ============================== reporting ==============================
def md_table(rows, title):
    hdr = "| group | " + " | ".join(f"MAE {m} | bias {m}" for m in MODELS) + " | n px | n days |"
    lines = [f"\n### {title}", hdr, "|" + "---|" * (2 + 2 * len(MODELS))]
    for r in rows:
        vals = " | ".join(("-" if r[m]["mae"] is None else f"{r[m]['mae']:.4f} | {r[m]['bias']:+.4f}") for m in MODELS)
        flag = " (n days < 5)" if r["flag_n_days_lt5"] else ""
        lines.append(f"| {r['group']}{flag} | {vals} | {r['n_px']:,} | {r['n_days']} |")
    return "\n".join(lines)


def plot(res, n_days, fn):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    panels = [("region", "Region", ax[0, 0], True), ("month", "Calendar month", ax[0, 1], False),
              ("obs_frac_quintile", "Quintile of the day's observed fraction (range)", ax[1, 0], True),
              ("truth_decile", "Decile of true ln CHL (range)", ax[1, 1], True)]
    series = list(RUNS) + ["DOY climatology"]
    for key, xl, a, rot in panels:
        rows = res[key]; x = np.arange(len(rows))
        for j, mod in enumerate(series):
            mae = [np.nan if r[mod]["mae"] is None else r[mod]["mae"] for r in rows]
            bias = [np.nan if r[mod]["bias"] is None else r[mod]["bias"] for r in rows]
            ls = "-" if mod in RUNS else ":"
            a.plot(x, mae, marker="o", ls=ls, color=OKABE_ITO[j], label=f"MAE, {mod}")
            if key == "truth_decile" and mod in RUNS:
                a.plot(x, bias, marker="s", ls="--", color=OKABE_ITO[j], label=f"mean bias (pred − true), {mod}")
        if key == "truth_decile":
            a.axhline(0, color="grey", lw=0.6)
        a.set_xticks(x); a.set_xticklabels([f"{r['group']}\n(n px {r['n_px']:,})" for r in rows],
                                           rotation=30, ha="right", fontsize=7)
        ylabel = "ln CHL error (MAE; bias = mean pred − true)" if key == "truth_decile" else "hidden-pixel MAE (ln CHL)"
        a.set(xlabel=xl, ylabel=ylabel)
        a.set_title(f"Val hidden-pixel error by {xl.split(' (')[0][0].lower() + xl.split(' (')[0][1:]}")
        a.legend(fontsize=7)
    fig.suptitle(f"Black Sea 4 km, val hidden pixels (n = {res['overall'][0]['n_px']:,} px, {n_days} days)")
    plt.tight_layout(); plt.savefig(fn, dpi=120); plt.close(fig)


def main():
    names = {int(k): v for k, v in json.load(open(ROOT / "analyses/blk/region_names.json")).items()}
    ctx, px, dq = load_pixels()
    truth_edges = np.quantile(px["val"]["truth"], np.linspace(0, 1, 11))
    truth_edges[0], truth_edges[-1] = -np.inf, np.inf                    # test values outside the val range fall in D1 / D10
    out = dict(runs={k: str(v.relative_to(ROOT)) for k, v in RUNS.items()}, data_quality=dq,
               notes=["error = pred - truth on hidden pixels (preds.npz hidden & ocean & finite truth)",
                      "truth deciles: edges from val hidden-pixel truth, applied to test too (outer edges open)",
                      "obs_frac quintiles: edges from the split's scored days (observed fraction of ocean px before hiding)",
                      "region: mode of non-zero 2 km labels in the 2x2 block, ties -> smaller label",
                      "DOY climatology: train days, 8-day bins, +-2 bin smoothing (train2.doy_climatology)"],
               val_truth_decile_edges=[float(e) for e in truth_edges[1:-1]])
    for split in ("val", "test"):
        days = np.asarray(sorted(set(px[split]["day"].tolist())))
        of_edges = np.quantile(ctx["obs_frac"][days], np.linspace(0, 1, 6))
        out[split] = breakdowns(ctx, px[split], names, truth_edges, of_edges)
        out[split]["n_days_scored"] = int(len(days))
    json.dump(out, open(RES / "error_breakdown.json", "w"), indent=1)
    fn = FIG / "error_breakdown.png"
    plot(out["val"], out["val"]["n_days_scored"], fn)
    print("data quality:", json.dumps(dq, indent=1))
    for key, t in [("overall", "Overall"), ("region", "By region"), ("month", "By calendar month"),
                   ("obs_frac_quintile", "By day's observed-fraction quintile"), ("truth_decile", "By true ln CHL decile")]:
        print(md_table(out["val"][key], f"VAL: {t}"))
    print("\njson", RES / "error_breakdown.json", "\nfigure", fn)


if __name__ == "__main__":
    main()
