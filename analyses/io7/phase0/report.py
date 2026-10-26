"""Phase 0 evaluations (bridge #25; evaluation only, no training). CPU; item 2 reads the GPU recorder output of spatial_attn.py.

  1. val MAE on no-key hidden px of io6_tokdec_blend_spatial by calendar month x region (lat/lon boxes, REGIONS below; no IO region mask
     exists in the repo)
  2. spatial self-attention weights of io6_tokdec_blend_spatial vs patch distance and |dSST| (~/mind_the_chl_gap/data/io7/
     io6_tokdec_blend_spatial_spattn.npz; skipped with a note if missing)
  3. eval-only ensemble: per-px mean of MEMBERS (val, test; overall and by key split)
  4. real-cloud reweighted MAE (analyses/io7/postrun.py) for the members and the ensemble
Data generation (score_all), analysis (item1, item2), reporting (figures, FINDINGS entry) are separate functions.
Writes analyses/io7/phase0/results/phase0.json, analyses/io7/phase0/figures/*.png, one FINDINGS.md entry (replaced on rerun).
Usage (project root): python -W ignore -m analyses.io7.phase0.report [--no-findings]
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyses.io7 import postrun as PR
from chl_cnn import store as S
from chl_cnn.plotting import OKABE_ITO

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "results", HERE / "figures"
SPATTN = Path.home() / "mind_the_chl_gap" / "data" / "io7" / "io6_tokdec_blend_spatial_spattn.npz"
RUN = "io6_tokdec_blend_spatial"
MEMBERS = ["io6_tokdec_blend_spatial", "io4_tokdec_blend", "io3_ptok4", "io2_base_eval", "io2_seed1", "io2_seed2"]
ENS = "ens6_mean"
# region boxes (deg): store covers 5 S-32 N, 45-90 E
REGIONS = [("Arabian Sea", "lat >= 5, lon < 78 (incl. Gulf of Aden, Gulf of Oman, Persian Gulf, Laccadive Sea)"),
           ("Bay of Bengal", "lat >= 5, lon >= 78 (incl. Andaman Sea west of 90 E)"),
           ("Equatorial and south", "lat < 5 (5 S-5 N)")]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MIN_N = 5


# ------------------------------------------------------------------ data generation
def region_map():
    d = S.load_store()
    lat, lon = np.asarray(d["lat"]), np.asarray(d["lon"])
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    r = np.where(LA < 5, 2, np.where(LO < 78, 0, 1)).astype("int8")
    return np.where(d["ocean"], r, -1)


def score_all():
    """postrun metrics of every member (cached in analyses/io7/results/) and of the ensemble mean; also returns the ensemble preds."""
    res = {m: PR.score_run(m) for m in MEMBERS}
    P = None
    for m in MEMBERS:
        p, _ = PR.load_pred(m)
        P = p if P is None else P + p
    P /= len(MEMBERS)
    res[ENS] = PR.score_pred(ENS, P, dict(members=MEMBERS, combine="mean", preds_dtype={m: res[m]["preds_dtype"] for m in MEMBERS}))
    return res


# ------------------------------------------------------------------ analysis
def item1():
    C = PR.ctx(); c = C["c"]; nv = C["nv"]
    pred, _ = PR.load_pred(RUN)
    va = c["va"]
    ae = np.abs(pred[:nv] - c["y"][va])
    m = c["hidden"][:nv] & (C["nk"][:nv] == 0)
    month = (np.asarray(c["times"])[va].astype("datetime64[M]").astype(int) % 12)
    reg = region_map()
    mae = np.full((12, 3), np.nan); n = np.zeros((12, 3), int); nd = np.zeros((12, 3), int)
    for mo in range(12):
        dd = month == mo
        for r in range(3):
            mm = m[dd] & (reg == r)[None]
            n[mo, r] = int(mm.sum()); nd[mo, r] = int(mm.any((1, 2)).sum())
            if n[mo, r]:
                mae[mo, r] = float(ae[dd][mm].mean())
    tot = {REGIONS[r][0]: dict(mae=float(ae[m & (reg == r)[None]].mean()), n=int((m & (reg == r)[None]).sum())) for r in range(3)}
    allm = {MONTHS[mo]: dict(mae=float(ae[month == mo][m[month == mo]].mean()), n=int(m[month == mo].sum())) for mo in range(12)}
    return dict(mae=mae.tolist(), n_px=n.tolist(), n_days=nd.tolist(), by_region=tot, by_month=allm,
                n_nokey_px=int(m.sum()), n_days_val=int(nv), regions=REGIONS, months=MONTHS)


def item2():
    if not SPATTN.exists():
        return None
    z = np.load(SPATTN)
    nD, nS = len(z["dist_edges"]) - 1, len(z["sst_edges_k"]) - 1
    sw = z["sum_w"].reshape(z["sum_w"].shape[0], z["sum_w"].shape[1], nD, nS)       # (blk, head, D, S)
    npair = z["n_pairs"].reshape(nD, nS)
    Np, days = int(z["Np"]), int(z["n_days"])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = dict(n_days=days, Np=Np, uniform_weight=1.0 / Np, heads=int(z["heads"]), n_blocks=int(z["n_blocks"]),
                   dist_labels=["0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33+"],
                   sst_labels=["0", "0.25", "0.5", "0.75", "1.0", "1.25-1.5", "1.75-2.0", "2.25-3.0", ">3.0"],
                   n_pairs_dist=npair.sum(1).tolist(), n_pairs_sst=npair.sum(0).tolist(), n_pairs_dist_sst=npair.tolist())
        for b in range(sw.shape[0]):
            hm = sw[b].mean(0)                                                         # head mean (D, S)
            out[f"block{b}"] = dict(
                mean_w_dist=(hm.sum(1) / npair.sum(1)).tolist(), share_dist=(hm.sum(1) / (Np * days)).tolist(),
                mean_w_sst=(hm.sum(0) / npair.sum(0)).tolist(), share_sst=(hm.sum(0) / (Np * days)).tolist(),
                mean_w_dist_sst=(hm / npair).tolist(),
                mean_w_dist_head=(sw[b].sum(2) / npair.sum(1)).tolist(), mean_w_sst_head=(sw[b].sum(1) / npair.sum(0)).tolist())
        rel = z["rel_sum_w"] / np.maximum(z["rel_n"], 1)
        out["_rel_map"] = np.where(z["rel_n"] > 0, rel, np.nan).reshape(-1, 2 * int(z["nPy"]) - 1, 2 * int(z["nPx"]) - 1)
        out["_rel_n"] = z["rel_n"].reshape(2 * int(z["nPy"]) - 1, 2 * int(z["nPx"]) - 1)
    return out


# ------------------------------------------------------------------ reporting: figures
def fig_item1(r1):
    mae = np.array(r1["mae"]); n = np.array(r1["n_px"])
    show = np.where(n >= MIN_N, mae, np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(show, cmap="viridis", aspect="auto")
    for i in range(12):
        for j in range(3):
            ax.text(j, i, f"{mae[i, j]:.3f}\nn {n[i, j]:,}" if n[i, j] >= MIN_N else f"n {n[i, j]} (<{MIN_N})", ha="center", va="center",
                    fontsize=7, color="w" if np.isfinite(show[i, j]) and show[i, j] < np.nanmean(show) else "k")
    ax.set(xticks=range(3), xticklabels=[r[0] for r in REGIONS], yticks=range(12), yticklabels=MONTHS, xlabel="region (lat/lon box)",
           ylabel="calendar month (val years 2001, 2007, 2013, 2018)",
           title=f"{RUN}: val MAE on no-key hidden px\nby month and region (n = no-key hidden px)")
    fig.colorbar(im, ax=ax, label="MAE (ln CHL)")
    fig.tight_layout(); fig.savefig(FIG / "p0_nokey_month_region.png", dpi=130); plt.close(fig)


def fig_item2(r2):
    D, Sl = r2["dist_labels"], r2["sst_labels"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for b in range(r2["n_blocks"]):
        B = r2[f"block{b}"]
        ax[0, 0].plot(range(len(D)), B["mean_w_dist"], "o-", color=OKABE_ITO[b], label=f"block {b + 1} (head mean)")
        ax[0, 1].plot(range(len(Sl)), B["mean_w_sst"], "o-", color=OKABE_ITO[b], label=f"block {b + 1} (head mean)")
        for i, lab in enumerate(D[1:6], start=1):
            ax[1, b].plot(range(len(Sl)), np.array(B["mean_w_dist_sst"])[i], "o-", color=OKABE_ITO[i % 8], label=f"distance {lab} patches")
        ax[1, b].axhline(r2["uniform_weight"], color="0.5", ls=":", label="uniform 1/Np")
        ax[1, b].set(yscale="log", xticks=range(len(Sl)), xticklabels=Sl, xlabel="|ΔSST| query vs key patch, day t (K)",
                     ylabel="mean attention weight per (query, key) pair",
                     title=f"Block {b + 1}: mean weight vs |ΔSST| within distance bins (head mean)")
        ax[1, b].legend(fontsize=7)
    for a_, lab, xl in ((ax[0, 0], D, "patch distance (patches; 1 patch = 4 px = 1°)"), (ax[0, 1], Sl, "|ΔSST| query vs key patch, day t (K)")):
        a_.axhline(r2["uniform_weight"], color="0.5", ls=":", label="uniform 1/Np")
        a_.set(yscale="log", xticks=range(len(lab)), xticklabels=lab, xlabel=xl, ylabel="mean attention weight per (query, key) pair")
        a_.legend(fontsize=8)
    ax[0, 0].set_title(f"Spatial attention weight vs patch distance\n({r2['n_days']} val days, every ocean patch a query, Np = {r2['Np']})")
    ax[0, 1].set_title(f"Spatial attention weight vs |ΔSST|\n({r2['n_days']} val days)")
    fig.tight_layout(); fig.savefig(FIG / "p0_spatial_attn_bins.png", dpi=130); plt.close(fig)
    rm = r2["_rel_map"]
    cy, cx = rm.shape[1] // 2, rm.shape[2] // 2
    R = 12
    fig, ax = plt.subplots(1, rm.shape[0], figsize=(5.5 * rm.shape[0], 4.8))
    for b in range(rm.shape[0]):
        im = ax[b].imshow(np.log10(rm[b, cy - R:cy + R + 1, cx - R:cx + R + 1]), cmap="viridis", origin="upper",
                          extent=(-R - .5, R + .5, R + .5, -R - .5))
        ax[b].set(title=f"Block {b + 1}: mean weight by key offset (head mean)", xlabel="key − query, x (patches, east +)",
                  ylabel="key − query, y (patches, south +)")
        fig.colorbar(im, ax=ax[b], label="log10 mean attention weight")
    fig.tight_layout(); fig.savefig(FIG / "p0_spatial_attn_offsets.png", dpi=130); plt.close(fig)


def fig_items34(res):
    names = MEMBERS + [ENS]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    x = np.arange(len(names))
    for k, (key, title) in enumerate((("all", "all hidden px"), (">=1 key", "hidden px with ≥ 1 key"), ("no key", "hidden px with no key"))):
        for j, (split, mk) in enumerate((("val", "o"), ("test", "s"))):
            f = (lambda r: r[split]["mae"]) if key == "all" else (lambda r: r[split]["by_key"][key]["mae"])
            ax[k].plot(x + (j - .5) * .2, [f(res[n]) for n in names], mk, color=OKABE_ITO[j], label=f"{split} (fake-cloud px)")
            if key == "all":
                ax[k].plot(x + (j - .5) * .2, [res[n][split]["rw"]["mae"] for n in names], mk, mfc="none", color=OKABE_ITO[j + 2],
                           label=f"{split} real-cloud reweighted")
        ax[k].set(xticks=x, xticklabels=names, ylabel="MAE (ln CHL)", title=f"MAE by model: {title}")
        ax[k].tick_params(axis="x", rotation=45, labelsize=7)
        ax[k].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "p0_models_mae.png", dpi=130); plt.close(fig)
    C = PR.ctx()
    nd, nk = len(PR.DIST_RW), len(PR.KEY_RW)
    hid_share = np.bincount(C["cell"][:C["nv"]][C["c"]["hidden"][:C["nv"]]], minlength=C["nc"]) / C["c"]["hidden"][:C["nv"]].sum()
    real_share = C["tgt"]["val"] / C["tgt"]["val"].sum()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a_, sh, t in ((ax[0], hid_share, f"fake-cloud hidden px (n {C['c']['hidden'][:C['nv']].sum():,})"),
                      (ax[1], real_share, f"real-cloud px (n {C['n_real']['val']:,})")):
        im = a_.imshow(sh.reshape(nd, nk), cmap="viridis", vmin=0, vmax=max(hid_share.max(), real_share.max()))
        for i in range(nd):
            for j in range(nk):
                a_.text(j, i, f"{sh.reshape(nd, nk)[i, j]:.3f}", ha="center", va="center", fontsize=7, color="w")
        a_.set(xticks=range(nk), xticklabels=[b[0] for b in PR.KEY_RW], yticks=range(nd), yticklabels=[b[0] for b in PR.DIST_RW],
               xlabel="visible kriging-stencil keys", ylabel="distance to nearest visible px, day t (px)", title=f"Val share per cell: {t}")
        fig.colorbar(im, ax=a_, label="share of px")
    fig.tight_layout(); fig.savefig(FIG / "p0_rw_cells.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------ reporting: FINDINGS entry
def entry(r1, r2, res):
    when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    v = res[RUN]["val"]
    L = [f"## Phase 0 evaluations, bridge #25 (IO; {when}; eval only; analyses/io7/phase0/, analyses/io7/postrun.py, analyses/io7/keys.py)",
         "- Assumptions:",
         "  - Key counts (≥ 1 visible kriging-stencil key, |dt| ≤ 4, 484 offsets) computed on CPU from the visibility masks "
         "(analyses/io7/keys.py); identical to the io6_tokdec_krig recorder pass on all 48.8 M val+test ocean px-days (0 differences).",
         "  - Regions (no IO region mask in the repo): " + "; ".join(f"{a}: {b}" for a, b in REGIONS) + ".",
         "  - \"io3 attention\" = io3_ptok4 (patch tokens, arm A). Conv ±4 seeds = io2_base_eval (seed 0), io2_seed1, io2_seed2.",
         "  - Real-cloud reweighting: hidden-px errors weighted so the joint (distance to nearest day-t visible px bin × key-count bin) "
         f"distribution matches the val days' real-cloud px (ocean, not observed; n {PR.ctx()['n_real']['val']:,}); distance bins "
         f"{', '.join(b[0] for b in PR.DIST_RW)} px; key bins {', '.join(b[0] for b in PR.KEY_RW)}; cells with < {PR.MIN_N} hidden px dropped. "
         "Assumes errors are exchangeable between fake-cloud and real-cloud px within a cell. The same val target is applied to test.",
         f"- 1. {RUN} val MAE on no-key hidden px ({r1['n_nokey_px']:,} px, {r1['n_days_val']} days) by region: "
         + "; ".join(f"{k} {x['mae']:.4f} (n {x['n']:,})" for k, x in r1["by_region"].items()) + ".",
         "  - By month (all regions): " + ", ".join(f"{k} {x['mae']:.3f} (n {x['n']:,})" for k, x in r1["by_month"].items()) + ".",
         f"  - Month × region: min {np.nanmin(np.array(r1['mae'], float)):.3f}, max {np.nanmax(np.array(r1['mae'], float)):.3f}; "
         f"cells with < {MIN_N} px: {int((np.array(r1['n_px']) < MIN_N).sum())} of 36. Table: results/phase0.json item1; figure figures/p0_nokey_month_region.png."]
    if r2 is None:
        L.append("- 2. Spatial attention weights: no recorder output (GPU job p0_spattn missing or failed); not reported.")
    else:
        L.append(f"- 2. Spatial attention weights of {RUN} ({r2['n_days']} val days, every 10th val position; all {r2['Np']} ocean-patch "
                 f"queries; uniform weight 1/Np = {r2['uniform_weight']:.2e}). Mean weight per (query, key) pair, head mean:")
        for b in range(r2["n_blocks"]):
            B = r2[f"block{b}"]
            L.append(f"  - block {b + 1} by distance (patches): " + ", ".join(f"{l} {w:.2e}" for l, w in zip(r2["dist_labels"], B["mean_w_dist"]))
                     + f"; self weight share {B['share_dist'][0]:.3f}, share within ≤ 2 patches {sum(B['share_dist'][:3]):.3f}.")
            L.append(f"  - block {b + 1} by |ΔSST| (K): " + ", ".join(f"{l} {w:.2e}" for l, w in zip(r2["sst_labels"], B["mean_w_sst"])) + ".")
        L.append("  - pairs per distance bin: " + ", ".join(f"{l} {n:,.0f}" for l, n in zip(r2["dist_labels"], r2["n_pairs_dist"]))
                 + ". Figures figures/p0_spatial_attn_bins.png (incl. |ΔSST| within distance bins), p0_spatial_attn_offsets.png.")
    L.append(f"- 3-4. Val / test MAE (val n {v['n_hidden']:,} hidden px, {v['n_days']} days; test n {res[RUN]['test']['n_hidden']:,}, "
             f"{res[RUN]['test']['n_days']} days); key split n val ≥ 1 key {v['by_key']['>=1 key']['n']:,}, no key {v['by_key']['no key']['n']:,}:")
    L.append("")
    L.append("| model | val | test | val ≥ 1 key | val no key | test ≥ 1 key | test no key | val rw | test rw | preds |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in MEMBERS + [ENS]:
        r = res[n]
        L.append(f"| {n} | {r['val']['mae']:.5f} | {r['test']['mae']:.5f} | {r['val']['by_key']['>=1 key']['mae']:.5f} | "
                 f"{r['val']['by_key']['no key']['mae']:.5f} | {r['test']['by_key']['>=1 key']['mae']:.5f} | {r['test']['by_key']['no key']['mae']:.5f} | "
                 f"{r['val']['rw']['mae']:.5f} | {r['test']['rw']['mae']:.5f} | {r['preds_dtype'] if isinstance(r['preds_dtype'], str) else 'mean'} |")
    L.append("")
    e, s = res[ENS], res[RUN]
    L.append(f"- Ensemble ({ENS}, per-px mean of {len(MEMBERS)}) − {RUN}: val {e['val']['mae'] - s['val']['mae']:+.5f}, test "
             f"{e['test']['mae'] - s['test']['mae']:+.5f}; rw val {e['val']['rw']['mae'] - s['val']['rw']['mae']:+.5f}, test "
             f"{e['test']['rw']['mae'] - s['test']['rw']['mae']:+.5f}. Gradient ratio/corr lag 1 val: ensemble "
             f"{e['val']['grad']['1']['ratio']:.3f}/{e['val']['grad']['1']['corr']:.3f}, {RUN} {s['val']['grad']['1']['ratio']:.3f}/{s['val']['grad']['1']['corr']:.3f}.")
    L.append("- Real-cloud reweighting: uncovered real-cloud mass " + f"{s['val']['rw']['uncovered_real_mass']:.4f}; "
             f"cells used {s['val']['rw']['n_cells_used']} of {PR.ctx()['nc']}. Test with the test days' own real-cloud target instead: "
             + ", ".join(f"{n} {res[n]['test']['rw_own']['mae']:.5f}" for n in MEMBERS + [ENS]) + ". Figures figures/p0_models_mae.png, p0_rw_cells.png.")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-findings", action="store_true")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    res = score_all()
    r1, r2 = item1(), item2()
    fig_item1(r1); fig_items34(res)
    if r2 is not None:
        fig_item2(r2)
    j = dict(item1=r1, item2={k: v for k, v in r2.items() if not k.startswith("_")} if r2 else None,
             items34={n: {s: dict(mae=res[n][s]["mae"], rmse=res[n][s]["rmse"], bias=res[n][s]["bias"], n_hidden=res[n][s]["n_hidden"],
                                  by_key=res[n][s]["by_key"], rw=res[n][s]["rw"], rw_own=res[n][s]["rw_own"], grad=res[n][s]["grad"])
                          for s in ("val", "test")} for n in res},
             rw_def=res[RUN]["rw_def"], members=MEMBERS)
    (OUT / "phase0.json").write_text(json.dumps(j, indent=1))
    text = entry(r1, r2, res)
    print(text[:1500])
    if not a.no_findings:
        _write(text)


def _write(text):
    """Append, replacing an earlier Phase 0 entry, under the FINDINGS lock."""
    import fcntl, re
    with open(PR.FINDINGS.with_suffix(".md.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = PR.FINDINGS.read_text()
        s = re.sub(r"\n## Phase 0 evaluations, bridge #25 \(.*?(?=\n## |\Z)", "", s, flags=re.S)
        PR.FINDINGS.write_text(s.rstrip("\n") + "\n\n" + text)


if __name__ == "__main__":
    main()
