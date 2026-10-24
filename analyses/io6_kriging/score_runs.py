"""Score io6 runs with analyses/io_improve/score.py, writing analyses/io6_kriging/results/<run>.json, plus (io6 additions):
  val.by_age2   MAE by two-sided age: days to the nearest real observation within t-8..t+8 excluding t (analyses/io3_attn/checks.age_twosided,
                cached in ~/mind_the_chl_gap/data/io3/val_age_twosided.npy); bins 1 d, 2-3 d, 4-8 d, none (score.py's age bins)
  by_key        (val, test) MAE on hidden px with >= 1 visible key in the kriging stencil vs no key, for the run and, on the same px, for the
                references io4_tokdec_blend and conv +-4 seed 0 (io2_base_eval). Key counts from ~/mind_the_chl_gap/data/io6/<KRUN>_krigall.npz
                (analyses/io6_kriging/io6_eval.py --krig-all; KRUN = the run itself, or --keys-from; the counts depend only on the inputs and
                the stencil, not on the weights)
  spread        (kriging runs, val hidden px) Spearman correlation of the kriging spread with |error| overall and by distance / two-sided age
                bin; mean |error| by spread decile (decile edges over all val hidden px with >= 1 key); n per group
Usage (project root): python -m analyses.io6_kriging.score_runs RUN [RUN ...] [--keys-from KRUN]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analyses.io_improve import score
from analyses.forecast.forecast_eval import BINS as AGE_BINS
from chl_cnn.runs import find_run

HERE = Path(__file__).resolve().parent
score.RES = HERE / "results"
DATA6 = Path.home() / "mind_the_chl_gap" / "data" / "io6"
DATA3 = Path.home() / "mind_the_chl_gap" / "data" / "io3"
REFS = {"io4_tokdec_blend": "io4_tokdec_blend", "conv_seed0": "io2_base_eval"}


def two_sided_age(va, times):
    f = DATA3 / "val_age_twosided.npy"
    if f.exists():
        return np.load(f)
    from analyses.io3_attn.checks import age_twosided
    c = score.context()
    age = age_twosided(c["obs"], times, va)
    np.save(f, age)
    return age


def load_pred(run):
    z = np.load(find_run(run) / "preds.npz")
    c = score.context()
    assert np.array_equal(z["val_idx"], c["va"]) and np.array_equal(z["test_idx"], c["te"])
    return z["pred"].astype("float32")


def mae(ae, m):
    return dict(mae=float(ae[m].mean()) if m.any() else None, n=int(m.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--keys-from", default=None)
    a = p.parse_args()
    refs = None
    for run in a.runs:
        res = score.score_run(run)
        c = score.context()
        nv = len(c["va"])
        pred = load_pred(run)
        y = np.concatenate([c["y"][c["va"]], c["y"][c["te"]]])
        hid = c["hidden"]
        ae = np.abs(pred - y)
        age2 = two_sided_age(c["va"], c["times"])
        res["val"]["by_age2"] = {}
        for name, lo, hi in AGE_BINS:
            m = hid[:nv] & ((age2 == 0) if hi == 0 else ((age2 >= lo) & (age2 <= hi)))
            res["val"]["by_age2"][name] = mae(ae[:nv], m)
        kf = DATA6 / f"{run}_krigall.npz"
        if not kf.exists() and a.keys_from:
            kf = DATA6 / f"{a.keys_from}_krigall.npz"
        if kf.exists():
            K = np.load(kf)
            nk = K["n_keys"]
            assert nk.shape == pred.shape, (nk.shape, pred.shape)
            if refs is None:
                refs = {k: np.abs(load_pred(r) - y) for k, r in REFS.items()}
            res["keys_from"] = kf.name
            for split, sl in (("val", slice(0, nv)), ("test", slice(nv, None))):
                h = hid[sl]
                masks = {">=1 key": h & (nk[sl] > 0), "no key": h & (nk[sl] == 0)}
                res[split]["by_key"] = {g: dict(run=mae(ae[sl], m), **{k: mae(v[sl], m) for k, v in refs.items()}) for g, m in masks.items()}
                res[split]["by_key"]["n_keys_hidden_px"] = dict(mean=float(nk[sl][h].mean()), median=float(np.median(nk[sl][h])),
                                                                q90=float(np.percentile(nk[sl][h], 90)))
            if "spread" in K.files and (DATA6 / f"{run}_krigall.npz").exists():
                sp = K["spread"][:nv].astype("float32")
                h = hid[:nv] & (nk[:nv] > 0)
                e_, s_ = ae[:nv][h], sp[h]
                dec = np.percentile(s_, np.arange(0, 101, 10))
                bi = np.clip(np.searchsorted(dec, s_, side="right") - 1, 0, 9)
                out = dict(n=int(h.sum()), spearman_all=float(spearmanr(s_, e_).statistic), decile_edges=dec.tolist(),
                           mae_by_decile=[float(e_[bi == i].mean()) for i in range(10)], n_by_decile=[int((bi == i).sum()) for i in range(10)],
                           spread_mean_by_decile=[float(s_[bi == i].mean()) for i in range(10)], by_dist={}, by_age2={})
                dv = c["dist"][:nv].astype("float32")
                for name, lo, hi in score.DIST_BINS:
                    m = h & (dv > lo) & (dv <= hi)
                    out["by_dist"][name] = dict(spearman=float(spearmanr(sp[m], ae[:nv][m]).statistic) if m.sum() > 5 else None,
                                                n=int(m.sum()), mean_spread=float(sp[m].mean()) if m.any() else None,
                                                mae=float(ae[:nv][m].mean()) if m.any() else None)
                for name, lo, hi in AGE_BINS:
                    m = h & ((age2 == 0) if hi == 0 else ((age2 >= lo) & (age2 <= hi)))
                    out["by_age2"][name] = dict(spearman=float(spearmanr(sp[m], ae[:nv][m]).statistic) if m.sum() > 5 else None,
                                                n=int(m.sum()), mean_spread=float(sp[m].mean()) if m.any() else None,
                                                mae=float(ae[:nv][m].mean()) if m.any() else None)
                res["val"]["spread"] = out
        (score.RES / f"{run}.json").write_text(json.dumps(res, indent=1))
        print(f"{run}: by_age2 {json.dumps({k: round(v['mae'], 5) for k, v in res['val']['by_age2'].items()})}"
              + (f"  by_key val {json.dumps({g: {k: round(v['mae'], 5) if v['mae'] else None for k, v in d.items()} for g, d in res['val']['by_key'].items() if g != 'n_keys_hidden_px'})}"
                 if "by_key" in res["val"] else "  (no key split)"), flush=True)


if __name__ == "__main__":
    main()
