"""Bridge #33 data generation (CPU, no model): per-hidden-px vectors for the IO and Black Sea (blk4) ensembles.

IO: members / X / y / lab from the io7 ensemble cache (~/mind_the_chl_gap/data/io7/ensemble_hidden_px.npz, ens11 members).
blk4: the members below, read from preds.npz (postrun.load_pred asserts the same scored days and hidden set as the blk4 context).
Per hidden px (val days then test days, C order within a day, as the io7 cache):
  dist (distance to the nearest day-t visible px, postrun / score.context), nk (kriging-stencil key count, analyses/io7/keys.py),
  cell (postrun real-cloud reweighting cell, -1 = none), lab (0 val half A, 1 val half B, 2 test);
lag-1 pairs (adjacent px along y or x on the same day, BOTH px hidden): i0, i1 (indices into the hidden-px vectors, i1 = the +1 px);
tgt: postrun's val real-cloud cell target (applied to val and test, as postrun's rw).
Writes ~/mind_the_chl_gap/data/ens33/{io,blk4}_hidden.npz.
Usage (project root): python -W ignore -m analyses.ens33.prep [--store io|blk4|both]
"""
import argparse
from pathlib import Path

import numpy as np

from analyses.io7 import postrun as PR
from analyses.io7.ensemble import report as ENS

DATA = Path.home() / "mind_the_chl_gap" / "data" / "ens33"
BLK_CORE = ["blk4_tokdec_blend_spatial", "blk4_physmem", "blk4_tokdec_blend", "blk4_tokdec_krig", "blk4_tokdec_krig_signed",
            "blk8_spatial_bsmem", "blk8_spatial_p12"]
BLK_REFS = list(PR.BLK4_CONV)                       # the two blk4_t3d_* postrun references
BLK_MEMBERS = BLK_CORE + BLK_REFS
BLK_HALVES = {"A": (2017,), "B": (2021,)}           # blk4 val years 2017, 2021: one year per half


def pairs(hid):
    """Lag-1 pairs with both px hidden; indices into hid's nonzero (C-order) vector. Returns i0, i1 (int64), axis (int8)."""
    idx = np.full(hid.shape, -1, "int64")
    idx[hid] = np.arange(int(hid.sum()))
    out = []
    for ax in (1, 2):
        n = hid.shape[ax]
        a0 = (slice(None),) * ax + (slice(0, n - 1),)
        a1 = (slice(None),) * ax + (slice(1, n),)
        m = hid[a0] & hid[a1]
        out.append((idx[a0][m], idx[a1][m], np.full(int(m.sum()), ax, "int8")))
    return tuple(np.concatenate(z) for z in zip(*out))


def common(C):
    c = C["c"]; hid = c["hidden"]
    i0, i1, ax = pairs(hid)
    return dict(dist=c["dist"][hid].astype("float16"), nk=C["nk"][hid].astype("uint16"), cell=C["cell"][hid].astype("int16"),
                i0=i0, i1=i1, pair_axis=ax, tgt=C["tgt"]["val"], n_days_val=C["nv"], n_days_test=len(c["te"]))


def prep_io():
    PR.use_store("io")
    H = ENS.hidden_matrix()
    assert list(H["members"]) == ENS.SET_B
    C = PR.ctx()
    out = common(C)
    assert len(out["dist"]) == len(H["y"]), "hidden-px count differs from the io7 ensemble cache"
    c = C["c"]; hid = c["hidden"]; idx = np.r_[c["va"], c["te"]]
    y_chk = np.concatenate([c["y"][idx[i]][hid[i]] for i in range(0, len(idx), 97)])
    y_sub = np.concatenate([H["y"][int(hid[:i].sum()):int(hid[:i + 1].sum())] for i in range(0, len(idx), 97)])
    assert np.array_equal(y_chk.astype("float32"), y_sub), "hidden-px order differs from the io7 ensemble cache"
    np.savez(DATA / "io_hidden.npz", X=H["X"], y=H["y"], lab=H["lab"], members=H["members"], **out)
    print("io", H["X"].shape, "pairs", len(out["i0"]), flush=True)


def prep_blk():
    PR.use_store("blk4")
    C = PR.ctx(); c = C["c"]; hid = c["hidden"]; nv = C["nv"]
    idx = np.r_[c["va"], c["te"]]
    y = np.concatenate([c["y"][idx[i]][hid[i]] for i in range(len(idx))]).astype("float32")
    yr = np.asarray(c["times"])[c["va"]].astype("datetime64[Y]").astype(int) + 1970
    labd = np.full(len(idx), 2, "int8")
    for k, h in enumerate("AB"):
        labd[:nv][np.isin(yr, BLK_HALVES[h])] = k
    lab = np.repeat(labd, hid.reshape(len(idx), -1).sum(1))
    X = np.empty((len(y), len(BLK_MEMBERS)), "float32")
    for j, m in enumerate(BLK_MEMBERS):
        p, _ = PR.load_pred(m)
        X[:, j] = p[hid]
    # other blk4_t3d_* runs: same scored days and hidden set?
    from chl_cnn.runs import find_run
    other = {}
    for d in sorted((find_run(BLK_REFS[0]).parent).glob("blk4_t3d_*")):
        if d.name in BLK_REFS or not (d / "preds.npz").exists():
            continue
        z = np.load(d / "preds.npz")
        other[d.name] = bool(np.array_equal(z["val_idx"], c["va"]) and np.array_equal(z["test_idx"], c["te"]) and np.array_equal(z["hidden"], hid))
    out = common(C)
    np.savez(DATA / "blk4_hidden.npz", X=X, y=y, lab=lab, members=np.array(BLK_MEMBERS),
             other_t3d_names=np.array(list(other)), other_t3d_match=np.array(list(other.values())), **out)
    print("blk4", X.shape, "pairs", len(out["i0"]), "lab", np.bincount(lab), "other blk4_t3d match", other, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="both", choices=("io", "blk4", "both"))
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if a.store in ("blk4", "both"):
        prep_blk()
    if a.store in ("io", "both"):
        prep_io()


if __name__ == "__main__":
    main()
