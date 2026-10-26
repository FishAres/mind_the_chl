"""CPU key counts for the io6 kriging stencil (chl_cnn/krig.py stencil_k1: |dt| <= 4, 484 offsets), no model needed.

n_keys(p, t) = number of stencil entries (dt, dy, dx) whose pixel p + (dy, dx) is visible in the model input on day t + dt:
  day t       observed & not hidden (fixed val/test fake-cloud mask, chl_cnn.clouds.fixed_hidden, eval seed 0 val / 1 test)
  day t + k   observed & in the store, and (--context-clouds persist, the io4..io7 default) observed on the donor day's t + k
              neighbour (train_pinn.nb_masks); padding and land are never visible.
This mirrors krig.py _prep (vis = mask channel * day_ok * ocean). check() compares it with a recorder pass's n_keys.
Cached: ~/mind_the_chl_gap/data/io7/n_keys_valtest.npy (uint16, val days then test days, all px).
Other stores (blk7, 2026-10-26): compute(c=ctx) / load(cache, c=ctx) with a strided context from analyses/blk7/storectx.score_context
(scored val / test days, their hidden masks and donors); the IO default (no arguments) is unchanged.
Usage (project root): python -m analyses.io7.keys [--check]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import correlate

from chl_cnn import clouds as C, train2 as T2
from chl_cnn.krig import stencil_k1, MAX_LAG
from analyses.io_improve import score

DATA7 = Path.home() / "mind_the_chl_gap" / "data" / "io7"
CACHE = DATA7 / "n_keys_valtest.npy"
KRIGALL = Path.home() / "mind_the_chl_gap" / "data" / "io6" / "io6_tokdec_krig_krigall.npz"


def kernels():
    st = stencil_k1()
    R = max(max(abs(y), abs(x)) for _, y, x in st)
    K = {}
    for dt, y, x in st:
        K.setdefault(dt, np.zeros((2 * R + 1, 2 * R + 1), "float32"))[y + R, x + R] = 1
    assert sum(int(k.sum()) for k in K.values()) == len(st) == 484
    return K


def n_keys_days(obs, times, idx, hid, donor, persist=True):
    """(len(idx), H, W) uint16 key counts for days idx with day-t hidden masks hid and donor day indices donor."""
    K = kernels()
    NB = {k: T2.neighbour_index(times, k) for k in range(-MAX_LAG, MAX_LAG + 1) if k != 0}
    out = np.zeros(hid.shape, "uint16")
    for i, (t, d) in enumerate(zip(idx, donor)):
        acc = correlate((obs[t] & ~hid[i]).astype("float32"), K[0], mode="constant")
        for k, nb in NB.items():
            j = nb[t]
            if j < 0:
                continue
            m = obs[j].copy()
            if persist:
                jd = nb[d] if nb[d] >= 0 else d
                m &= obs[jd]
            acc += correlate(m.astype("float32"), K[k], mode="constant")
        out[i] = np.rint(acc).astype("uint16")
    return out


def compute(persist=True, c=None):
    """c: a context with donors / hid_split (analyses/blk7/storectx.score_context); None = the IO store (score.context)."""
    if c is None:
        c = score.context()
        obs, times, va, te = c["obs"], c["times"], c["va"], c["te"]
        hv, dv = C.fixed_hidden(obs, va, seed=0, times=times, exclude=0)
        ht, dt_ = C.fixed_hidden(obs, te, seed=1, times=times, exclude=0)
    else:
        obs, times, va, te = c["obs"], c["times"], c["va"], c["te"]
        (hv, ht), (dv, dt_) = c["hid_split"], c["donors"]
    hid = np.concatenate([hv, ht])
    assert np.array_equal(hid, c["hidden"]), "fixed_hidden does not reproduce the scored hidden set"
    return np.concatenate([n_keys_days(obs, times, va, hv, dv, persist), n_keys_days(obs, times, te, ht, dt_, persist)])


def load(cache=CACHE, c=None):
    """n_keys for every val+test day (val first), from the cache or computed on CPU (~minutes). c: see compute()."""
    cache = Path(cache)
    if cache.exists():
        nk = np.load(cache)
        if c is None or nk.shape == c["hidden"].shape:
            return nk
    nk = compute(c=c)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, nk)
    return nk


def check(nk):
    """Compare with the recorder pass of io6_tokdec_krig (GPU), on ocean px."""
    z = np.load(KRIGALL)["n_keys"]
    oc = score.context()["ocean"]
    d = (nk.astype("int32") - z.astype("int32"))[:, oc]
    return dict(n_px=int(d.size), n_diff=int((d != 0).sum()), max_abs_diff=int(np.abs(d).max()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    nk = load()
    print("n_keys", nk.shape, "mean", float(nk.mean()))
    if a.check:
        print("vs recorder (io6_tokdec_krig):", check(nk))
