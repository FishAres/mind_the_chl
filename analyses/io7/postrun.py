"""Post-run scoring for io7 runs (CPU only, from saved preds.npz). Called by ~/mind_the_chl_gap/logs/io7/postrun.sh <run>.

Per split (val, test), hidden px of the fixed fake clouds:
  mae / rmse / bias, gradient ratio / corr at lags 1, 2, 4 px, MAE by distance and age bins   (analyses/io_improve/score.split_metrics)
  by_key   MAE on hidden px with >= 1 visible kriging-stencil key (|dt| <= 4, 484 offsets) vs no key. Key counts are model-independent;
           computed on CPU from the visibility masks (analyses/io7/keys.py; identical to the io6_tokdec_krig recorder pass on all val+test px).
  rw       real-cloud reweighted MAE: hidden-px errors weighted so that the joint distribution of (distance to the nearest day-t visible px
           bin x key-count bin) matches that of the real-cloud px (ocean, not observed) of the val days. Assumes errors are exchangeable
           between fake-cloud and real-cloud px within a cell. Cells with < 5 hidden px are dropped (their real-cloud mass is reported as
           uncovered) and the remaining weights renormalised. rw_own: the same with the split's own days' real-cloud distribution.
  dist_gain  MAE(conv +-4 seed 0, io2_base_eval) - MAE(run) per distance bin (positive = run lower).
References io6_tokdec_blend_spatial and io4_tokdec_blend are scored the same way (cached in results/<ref>.json) and differenced.
Writes analyses/io7/results/<run>.json; appends a FINDINGS.md entry; regenerates models/RUNS.md.
Usage (project root): python -W ignore -m analyses.io7.postrun RUN [--force] [--no-findings] [--res-dir DIR]

Black Sea 4 km (blk7, 2026-10-26): the store comes from the run's config.json. For store blk4 (--crop, --val-stride 4) the context is
analyses/blk7/storectx.score_context (scored days va[::4], te[::4]); results go to analyses/blk7/results/, the FINDINGS header is
"## blk7 postrun: <run> (Black Sea 4 km; ...)", references are the five blk7 runs (those with preds.npz) and the two blk4 conv runs
(batch 16, 50 epochs: recipe differs), dist_gain is against blk4_t3d_c4_gdl100_lam0_b16_50ep, key counts are cached in
~/mind_the_chl_gap/data/blk7/, and analyses/blk7/figures/blk7_runs.png is redrawn. Distance / key bins stay in pixels (4.5 km on blk4,
~25 km on IO). The IO profile (every other store value) is unchanged.
--seed-group NAME=run1,run2,... (io8, 2026-10-26; repeatable; default none = unchanged): scores each member like a reference (cached JSON) and adds
"seed_groups" to the results JSON and a FINDINGS line per group: mean and range (min..max) of the members' val / test MAE and real-cloud rw MAE,
the run minus the group mean, and whether the run lies inside the range (bridge #29 rule: compare against the mean and range of seeds).
--res-dir: write results JSONs there instead (reference JSONs are then recomputed there too; used for the IO identity check).
"""
import argparse
import datetime as dt
import fcntl
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from analyses.io_improve import score
from analyses.io7 import keys as KEYS

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FINDINGS = ROOT / "analyses" / "FINDINGS.md"
REFS = ("io6_tokdec_blend_spatial", "io4_tokdec_blend")
CONV0 = "io2_base_eval"
DIST_RW = [("1", 0.5, 1.5), ("2-3", 1.5, 3.5), ("4-7", 3.5, 7.5), ("8-15", 7.5, 15.5), ("16-31", 15.5, 31.5), ("32+", 31.5, np.inf)]
KEY_RW = [("0", 0, 0), ("1-4", 1, 4), ("5-19", 5, 19), ("20-59", 20, 59), ("60+", 60, 10 ** 9)]
MIN_N = 5
_C = {}
BLK7_RUNS = ("blk4_tokdec_blend_spatial", "blk4_physmem", "blk4_tokdec_blend", "blk4_tokdec_krig", "blk4_tokdec_krig_signed")
BLK4_CONV = ("blk4_t3d_c4_gdl100_lam0_b16_50ep", "blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50")
PROFILES = {
    "io": dict(tag="io7", region="IO", res=RES, refs=REFS, conv0=CONV0, conv_label="conv ±4 seed 0", note=None, keys_cache=None,
               skip_missing=False),
    "blk4": dict(tag="blk7", region="Black Sea 4 km", res=ROOT / "analyses" / "blk7" / "results", refs=BLK7_RUNS + BLK4_CONV,
                 conv0=BLK4_CONV[0], conv_label="conv ±4 b16 50 ep (blk4_t3d_c4_gdl100_lam0_b16_50ep)", keys_cache=Path.home() / "mind_the_chl_gap" / "data" / "blk7" / "n_keys_valtest_blk4_crop_s4.npy",
                 note="blk4_t3d_* references use another recipe (conv ±4, batch 16, 50 epochs; the _sl_lam0p3_..._cont50 run: λ 0.3, semi-Lagrangian, 50 epochs then two "
                      "50-epoch continuations, the last at lr 1e-4); blk7 runs: batch 8, 60-epoch cosine, lr 1e-3, λ 0. Scored days: every 4th val / test day (--val-stride 4).",
                 skip_missing=True),
}
PROF = dict(PROFILES["io"], store="io")


def use_store(store, res_dir=None):
    """Select the store profile (and clear the cached context when it changes)."""
    global RES
    if store not in PROFILES:
        raise SystemExit(f"postrun: store {store} not supported")
    if PROF.get("store") != store:
        _C.clear()
    PROF.clear(); PROF.update(PROFILES[store], store=store)
    if res_dir:
        PROF["res"] = Path(res_dir)
    RES = PROF["res"]


# ------------------------------------------------------------------ data
def ctx():
    """Store context + key counts + per-px reweighting cell ids (val+test days, all px; -1 = no cell)."""
    if _C:
        return _C
    if PROF["store"] == "io":
        c = score.context()
        nk = KEYS.load()
    else:
        from analyses.blk7 import storectx
        c = storectx.score_context(PROF["store"], crop=True, stride=4)
        nk = KEYS.load(PROF["keys_cache"], c=c)
    d = c["dist"].astype("float32")
    di = np.full(d.shape, -1, "int8")
    for i, (_, lo, hi) in enumerate(DIST_RW):
        di[(d > lo) & (d <= hi)] = i
    ki = np.full(nk.shape, -1, "int8")
    for i, (_, lo, hi) in enumerate(KEY_RW):
        ki[(nk >= lo) & (nk <= hi)] = i
    cell = np.where((di >= 0) & (ki >= 0), di.astype("int16") * len(KEY_RW) + ki, -1).astype("int16")
    nv = len(c["va"])
    real = c["ocean"][None] & ~c["obs"][np.r_[c["va"], c["te"]]]
    nc = len(DIST_RW) * len(KEY_RW)
    tgt = {s: np.bincount(cell[sl][real[sl] & (cell[sl] >= 0)], minlength=nc).astype("float64")
           for s, sl in (("val", slice(0, nv)), ("test", slice(nv, None)))}
    _C.update(c=c, nk=nk, cell=cell, nv=nv, nc=nc, tgt=tgt, n_real={s: int(real[sl].sum()) for s, sl in (("val", slice(0, nv)), ("test", slice(nv, None)))})
    return _C


def load_pred(run):
    if PROF["store"] == "io":
        pred, dtype = score.load_preds(run)
        return pred, dtype
    from chl_cnn.runs import find_run
    z = np.load(find_run(run) / "preds.npz"); c = ctx()["c"]
    assert np.array_equal(z["val_idx"], c["va"]) and np.array_equal(z["test_idx"], c["te"]), f"{run}: scored days differ from the context"
    assert np.array_equal(z["hidden"], c["hidden"]), f"{run}: different hidden set"
    return z["pred"].astype("float32"), str(z["pred"].dtype)


# ------------------------------------------------------------------ analysis
def mae(ae, m):
    n = int(m.sum())
    return dict(mae=float(ae[m].mean()) if n else None, n=n)


def reweighted(ae, hid, cell, target, nc):
    """Real-cloud reweighted MAE: sum_c p_c MAE_c over cells with >= MIN_N hidden px (p renormalised over covered cells)."""
    h = hid & (cell >= 0)
    cnt = np.bincount(cell[h], minlength=nc)
    s = np.bincount(cell[h], weights=ae[h].astype("float64"), minlength=nc)
    cov = cnt >= MIN_N
    p = target / target.sum()
    m_c = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return dict(mae=float((p[cov] * m_c[cov]).sum() / p[cov].sum()), uncovered_real_mass=float(p[~cov].sum()),
                n_cells_used=int(cov.sum()), n_hidden=int(h.sum()), n_hidden_no_cell=int((hid & (cell < 0)).sum()))


def score_pred(name, pred, info=None):
    """All postrun metrics for a (val+test) prediction array."""
    C = ctx(); c = C["c"]; nv = C["nv"]
    res = dict(run=name, **(info or {}))
    for split, sl, idx in (("val", slice(0, nv), c["va"]), ("test", slice(nv, None), c["te"])):
        y = c["y"][idx]
        r = score.split_metrics(pred[sl], y, c["obs"][idx], c["hidden"][sl], c["dist"][sl], c["age"][sl])
        ae = np.abs(pred[sl] - y)
        hid = c["hidden"][sl]; nk = C["nk"][sl]
        r["by_key"] = {">=1 key": mae(ae, hid & (nk > 0)), "no key": mae(ae, hid & (nk == 0))}
        r["rw"] = reweighted(ae, hid, C["cell"][sl], C["tgt"]["val"], C["nc"])
        r["rw_own"] = reweighted(ae, hid, C["cell"][sl], C["tgt"][split], C["nc"])
        res[split] = r
    res["rw_def"] = dict(dist_bins=[b[0] for b in DIST_RW], key_bins=[b[0] for b in KEY_RW], min_n=MIN_N,
                         n_real_px=C["n_real"], target_val=(C["tgt"]["val"] / C["tgt"]["val"].sum()).tolist())
    return res


def score_run(run, force=False):
    f = RES / f"{run}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    pred, dtype = load_pred(run)
    res = score_pred(run, pred, dict(preds_dtype=dtype))
    RES.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, indent=1))
    return res


def compare(res, refs, conv):
    out = {}
    for r, rr in refs.items():
        out[r] = {s: dict(mae=res[s]["mae"] - rr[s]["mae"], rw=res[s]["rw"]["mae"] - rr[s]["rw"]["mae"],
                          **{g: res[s]["by_key"][g]["mae"] - rr[s]["by_key"][g]["mae"] for g in (">=1 key", "no key")})
                  for s in ("val", "test")}
    out["dist_gain_vs_conv0"] = {s: {b: dict(gain=conv[s]["by_dist"][b]["mae"] - res[s]["by_dist"][b]["mae"], n=res[s]["by_dist"][b]["n"])
                                     for b in res[s]["by_dist"]} for s in ("val", "test")}
    return out


# ------------------------------------------------------------------ reporting
def fmt_n(n):
    return f"{n:,}"


def seed_context(res, groups):
    """{name: stats} for seed groups {name: {member: results}}: mean / min / max of val, test MAE and rw MAE; run - mean; inside range."""
    out = {}
    for name, mem in groups.items():
        g = dict(members=list(mem))
        for s in ("val", "test"):
            for key, get in (("mae", lambda r: r[s]["mae"]), ("rw", lambda r: r[s]["rw"]["mae"])):
                x = np.array([get(r) for r in mem.values()]); me = get(res)
                g[f"{s}_{key}"] = dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()), run=me, run_minus_mean=float(me - x.mean()),
                                       inside_range=bool(x.min() <= me <= x.max()))
        out[name] = g
    return out


def seed_lines(sc):
    L = []
    for name, g in sc.items():
        f = lambda k: (f"mean {g[k]['mean']:.6f} [{g[k]['min']:.6f}, {g[k]['max']:.6f}], run − mean {g[k]['run_minus_mean']:+.5f}"
                       f" ({'inside' if g[k]['inside_range'] else 'outside'} range)")
        L.append(f"- Seed context {name} (n {len(g['members'])}: {', '.join(g['members'])}): val MAE {f('val_mae')}; test MAE {f('test_mae')}; "
                 f"real-cloud rw val {f('val_rw')}; test {f('test_rw')}.")
    return L


def entry(run, res, refs, cmpd, cfg_note):
    v, t = res["val"], res["test"]
    when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"## {PROF['tag']} postrun: {run} ({PROF['region']}; {when}; analyses/io7/postrun.py)",
         f"- Run: {cfg_note}. Val n: {v['n_days']} days, {fmt_n(v['n_hidden'])} hidden px; test n: {t['n_days']} days, {fmt_n(t['n_hidden'])}.",
         f"- Val / test MAE {v['mae']:.6f} / {t['mae']:.6f}; RMSE {v['rmse']:.4f} / {t['rmse']:.4f}; bias {v['bias']:+.4f} / {t['bias']:+.4f}; "
         f"non-finite {v['n_nonfinite']} / {t['n_nonfinite']}."]
    for r, rr in refs.items():
        d = cmpd[r]
        L.append(f"- Δ vs {r} ({rr['val']['mae']:.6f} / {rr['test']['mae']:.6f}): val {d['val']['mae']:+.5f}, test {d['test']['mae']:+.5f}; "
                 f"≥ 1 key {d['val']['>=1 key']:+.5f} / {d['test']['>=1 key']:+.5f}; no key {d['val']['no key']:+.5f} / {d['test']['no key']:+.5f}; "
                 f"real-cloud rw {d['val']['rw']:+.5f} / {d['test']['rw']:+.5f}.")
    bk = lambda s, g: f"{res[s]['by_key'][g]['mae']:.5f} (n {fmt_n(res[s]['by_key'][g]['n'])})"
    L.append(f"- Key split val / test: ≥ 1 key {bk('val', '>=1 key')} / {bk('test', '>=1 key')}; no key {bk('val', 'no key')} / {bk('test', 'no key')}.")
    L.append(f"- Real-cloud reweighted MAE val / test: {v['rw']['mae']:.5f} / {t['rw']['mae']:.5f} (target: val real-cloud px, "
             f"n {fmt_n(res['rw_def']['n_real_px']['val'])}; uncovered mass {v['rw']['uncovered_real_mass']:.4f} / {t['rw']['uncovered_real_mass']:.4f}; test with the test days' real-cloud target {t['rw_own']['mae']:.5f}); "
             + ", ".join(f"{r} {rr['val']['rw']['mae']:.5f} / {rr['test']['rw']['mae']:.5f}" for r, rr in refs.items()) + ".")
    g = lambda s: ", ".join(f"{v_['ratio']:.3f}/{v_['corr']:.3f}" for v_ in (res[s]["grad"][k] for k in ("1", "2", "4")))
    L.append(f"- Gradient ratio/corr lags 1, 2, 4 px: val {g('val')}; test {g('test')} (val lag-1 pairs {fmt_n(v['grad']['1']['n_pairs'])}).")
    L += seed_lines(res.get("seed_groups", {}))
    dg = cmpd["dist_gain_vs_conv0"]
    if PROF["note"]:
        L.append(f"- Note: {PROF['note']}")
    L.append(f"- Val MAE gain over {PROF['conv_label']} by distance to visible px (conv − run): "
             + ", ".join(f"{b} px {x['gain']:+.5f} (n {fmt_n(x['n'])})" for b, x in dg["val"].items()) + ".")
    return "\n".join(L) + "\n"


def write_findings(run, text):
    """Append (or with an existing entry for this run, replace) under a lock."""
    lock = FINDINGS.with_suffix(".md.lock")
    with open(lock, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = FINDINGS.read_text()
        tag = PROF["tag"]
        hdr = f"## {tag} postrun: {run} ("
        if hdr in s:
            s = re.sub(rf"\n## {tag} postrun: {re.escape(run)} \(.*?(?=\n## |\Z)", "", s, flags=re.S)
        FINDINGS.write_text(s.rstrip("\n") + "\n\n" + text)


def cfg_note(run):
    from chl_cnn.runs import find_run
    p = find_run(run) / "config.json"
    if not p.exists():
        return "config.json missing"
    c = json.loads(p.read_text())
    keys = ("t3d_stem", "tokdec_blend", "tokdec_spatial", "seed", "epochs", "batch", "best_epoch")
    if c.get("store", "io") != "io":                       # blk7: store-specific settings and the other stems' flags
        keys = ("store", "crop", "val_stride", "t3d_stem", "tokdec_blend", "tokdec_spatial", "tokdec_krig", "krig_signed", "physmem",
                "physmem_stride2", "context", "seed", "epochs", "batch", "lr", "lam", "best_epoch")
    keys = keys + tuple(k for k in ("physmem", "rw_loss", "rw_cap", "bsmem", "ptok_patch") if c.get("rw_loss") or c.get("bsmem")    # io8 / blk8
                        or c.get("ptok_patch", 4) != 4 if k in c and k not in keys)
    return ", ".join(f"{k} {c.get(k)}" for k in keys if k in c)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-findings", action="store_true")
    p.add_argument("--res-dir", default=None, help="results directory (default: the store profile's)")
    p.add_argument("--seed-group", action="append", default=[], help="NAME=run1,run2,...: seed mean / range context line (io8)")
    a = p.parse_args(argv)
    from chl_cnn.runs import find_run
    cf = find_run(a.run) / "config.json"
    use_store(json.loads(cf.read_text()).get("store", "io") if cf.exists() else "io", a.res_dir)
    if (RES / f"{a.run}.json").exists() and not a.force:
        print(f"{a.run}: results exist, skipped (--force to redo)")
        return 0
    def have(r):
        try:
            return (find_run(r) / "preds.npz").exists()
        except FileNotFoundError:
            return False
    cand = PROF["refs"]
    if PROF["skip_missing"]:                               # blk7: a run retried after an OOM is <run>_b6 / <run>_b4 (logs/blk7 train_oom)
        cand = [next((x for x in (r, r + "_b6", r + "_b4") if have(x)), r) for r in cand]
    refs = {r: score_run(r) for r in cand if r != a.run and (not PROF["skip_missing"] or have(r))}
    conv = score_run(PROF["conv0"])
    res = score_run(a.run, force=a.force)
    res["vs"] = compare(res, refs, conv)
    if a.seed_group:                                       # io8: seed mean / range context
        groups = {}
        for spec in a.seed_group:
            name, mem = spec.split("=", 1)
            groups[name] = {m: (res if m == a.run else score_run(m)) for m in mem.split(",")}
        res["seed_groups"] = seed_context(res, groups)
    (RES / f"{a.run}.json").write_text(json.dumps(res, indent=1))
    v = res["val"]
    print(f"{a.run}: val {v['mae']:.6f} test {res['test']['mae']:.6f} | key>=1 {v['by_key']['>=1 key']['mae']:.5f} "
          f"nokey {v['by_key']['no key']['mae']:.5f} | rw {v['rw']['mae']:.5f}/{res['test']['rw']['mae']:.5f}")
    if not a.no_findings:
        write_findings(a.run, entry(a.run, res, refs, res["vs"], cfg_note(a.run)))
        r = subprocess.run([sys.executable, "-m", "chl_cnn.runs_index"], cwd=ROOT, capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.returncode == 0 else f"runs_index failed: {r.stderr[-300:]}")
    if PROF["store"] != "io":
        from analyses.blk7 import figures as BF
        print("figure", BF.runs_figure(RES, [a.run] + list(refs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
