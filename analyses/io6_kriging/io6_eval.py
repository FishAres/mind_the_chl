"""Eval-only runs of a --t3d-stem tokdec model (io6: --tokdec-krig, --tokdec-spatial) with the chl_cnn/tokdec.py recorder. Data generation only.

The trainer's final scoring pass runs the val days then the test days (va[::val_stride], te[::val_stride]) in eval batches; forward i of
the pass is scored day i (i < n_val: val position i; else test position i - n_val).
Records:
  --positions P ...   full records of these val positions: emb, tok, pool_w, dec and whichever of krig (B, n_out, H, W), n_keys, pooled
                      (spatial tokens), w / blend / mass / ent (blend) the model produces, plus the stem input's visible mask -> <run>_io6.npz
                      (the leak test, analyses/io3_attn/leak_ptok.py --stem krig / tokdec_spatial, reads these).
  --krig-all          for every scored val and test day: n_keys (uint16: visible stencil keys per px), estimate, spread, mass (float16; channels
                      dv, dv + 1, dv + 2 of the kriging output) -> <run>_krigall.npz
  --diag-every S      learned-variogram statistics on the hidden px of every S-th val position (fixed hidden masks, chl_cnn/clouds.py), from the
                      final weights of every stencil candidate (chl_cnn/krig.py KrigStem.weights_chunks, head-mean over the value heads and per
                      head): sum of weights and number of visible candidates per stencil offset, and per |SST(q,d) - SST(p,t)| bin (K, z x the
                      trainer's SST std) -> <run>_diag.npz
Saved to ~/mind_the_chl_gap/data/io6/.

Usage (project root):
  python -m analyses.io6_kriging.io6_eval --run NAME [--positions 0 5] --n-val 1458 [--n-test 1463 --krig-all --diag-every 10] -- <train_pinn args incl. --eval-only>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from chl_cnn import tokdec, train_pinn
from chl_cnn import krig as KR

DATA = Path.home() / "mind_the_chl_gap" / "data" / "io6"
KEYS = ("emb", "tok", "pool_w", "dec", "w", "blend", "mass", "ent", "krig", "n_keys", "pooled")
# |dSST| bin edges (K): the IO store's SST is quantised to 0.25 K (checked 2026-10-24), so the bins are centred on its multiples:
# 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25-1.5 | 1.75-2.0 | 2.25-3.0 | > 3.0
SST_EDGES_K = np.array([0, 0.125, 0.375, 0.625, 0.875, 1.125, 1.625, 2.125, 3.125, np.inf])


class Rec:
    def __init__(self, positions, n_val, n_total, krig_all, diag_every, hid_val, sst_std):
        self.pos, self.n_val, self.n_total, self.k, self.out = set(positions), n_val, n_total, 0, {}
        self.krig_all, self.diag_every, self.hid_val, self.sst_std = krig_all, diag_every, hid_val, sst_std
        self.maps = dict(n_keys=[], est=[], spread=[], mass=[])
        self.diag = None

    def __call__(self, d):
        B = d["x3d"].shape[0]
        if self.k >= self.n_total:
            return
        nb = min(B, self.n_total - self.k)
        for i in range(nb):
            if self.k + i < self.n_val and self.k + i in self.pos:
                self.out[self.k + i] = {k: d[k][i].detach().float().cpu().numpy() for k in KEYS if k in d and torch.is_tensor(d[k]) and d[k].dim() > 0}
                self.out[self.k + i]["vis"] = (d["x3d"][i, -1] > 0.5).cpu().numpy()
        if self.krig_all and "krig" in d:
            kr, dv = d["krig"][:nb].detach().float(), d["net"].krig.dv
            self.maps["n_keys"].append(d["n_keys"][:nb].round().cpu().numpy().astype("uint16"))
            self.maps["est"].append(kr[:, dv].cpu().numpy().astype("float16"))
            self.maps["spread"].append(kr[:, dv + 1].cpu().numpy().astype("float16"))
            self.maps["mass"].append(kr[:, dv + 2].cpu().numpy().astype("float16"))
        if self.diag_every and "krig" in d:
            sel = [i for i in range(nb) if self.k + i < self.n_val and (self.k + i) % self.diag_every == 0]
            if sel:
                self.diagnose(d["net"].krig, sel)
        self.k += nb

    @torch.no_grad()
    def diagnose(self, st, sel):
        ctx = st._last
        sub = {k: (v[sel] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == ctx["B"] else v) for k, v in ctx.items()}
        sub["B"] = len(sel)
        hid = torch.stack([torch.as_tensor(self.hid_val[self.k + i].reshape(-1)) for i in sel]).to(st.oc_idx.device)[:, st.oc_idx]   # (b, N)
        nh, hv, nb_ = st.nh, st.heads, len(SST_EDGES_K) - 1
        if self.diag is None:
            self.diag = dict(sum_w=np.zeros((st.K, nh)), n_valid=np.zeros(st.K), sst_sum_w=np.zeros(nb_), sst_n=np.zeros(nb_),
                             sst_sum_w_t0=np.zeros(nb_), sst_n_t0=np.zeros(nb_), n_hidden_px=0, n_days=0, n_hidden_px_with_key=0,
                             offsets=st.offs[:st.K].cpu().numpy())
        D = self.diag
        D["n_hidden_px"] += int(hid.sum()); D["n_days"] += len(sel)
        edges = torch.as_tensor(SST_EDGES_K / self.sst_std, device=hid.device, dtype=torch.float32)
        any_key = torch.zeros_like(hid)
        for c0, c1, w, valid, dsst in st.weights_chunks(sub):
            k1 = min(c1, st.K)
            if c0 >= st.K:
                break
            n = k1 - c0
            w, valid, dsst = w[:, :, :n], valid[:, :n], dsst[:, :n]
            hm = hid.unsqueeze(1)                                                    # (b, 1, N)
            D["sum_w"][c0:k1] += (w * hm.unsqueeze(1)).sum((0, 3)).t().double().cpu().numpy()
            vv = valid & hm
            D["n_valid"][c0:k1] += vv.sum((0, 2)).cpu().numpy()
            any_key |= vv.any(1)
            wbar = w[:, :hv].mean(1)                                                 # (b, n, N) head-mean over the value heads
            bi = torch.bucketize(dsst, edges[1:-1])                                  # bins 0..nb_-1
            for t0 in (False, True):
                m_ = vv & (st.offs[c0:k1, 0].view(1, -1, 1) == 0) if t0 else vv
                sw = torch.zeros(nb_, device=w.device, dtype=torch.float64).index_add_(0, bi[m_], wbar[m_].double())
                sn = torch.zeros(nb_, device=w.device, dtype=torch.float64).index_add_(0, bi[m_], torch.ones_like(wbar[m_], dtype=torch.float64))
                D["sst_sum_w_t0" if t0 else "sst_sum_w"] += sw.cpu().numpy(); D["sst_n_t0" if t0 else "sst_n"] += sn.cpu().numpy()
        D["n_hidden_px_with_key"] += int(any_key.sum())


def main():
    argv = sys.argv[1:]
    i = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--positions", type=int, nargs="*", default=[])
    p.add_argument("--n-val", type=int, required=True, help="number of scored val days (len(va[::val_stride]))")
    p.add_argument("--n-test", type=int, default=0, help="number of scored test days (needed with --krig-all)")
    p.add_argument("--krig-all", action="store_true")
    p.add_argument("--diag-every", type=int, default=0)
    a = p.parse_args(argv[:i])
    targs = argv[i + 1:]
    hid_val, sst_std = None, 1.0
    if a.diag_every:
        from chl_cnn import data as D, store as S, clouds as C
        d = S.load_store(); times = d["time"]
        obs = np.isfinite(D.log_chl(np.asarray(d["chl_l3"]))) & d["ocean"]
        tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
        hv, _ = C.fixed_hidden(obs, va, seed=0, times=times)
        stride = int(targs[targs.index("--val-stride") + 1]) if "--val-stride" in targs else 1
        hid_val = hv[::stride]
        assert len(hid_val) == a.n_val
        from chl_cnn.runs import find_run
        src = json.loads((find_run(targs[targs.index("--init-from") + 1]) / "config.json").read_text())
        sst_std = float(src["feature_stats"]["sst"]["std"])                       # the trainer's z-scoring std of SST
        print(f"diag: SST std (trainer feature stats) {sst_std:.4f} K", flush=True)
        KR.HOOKS["keep_ctx"] = True
    rec = Rec(a.positions, a.n_val, a.n_val + (a.n_test if a.krig_all else 0), a.krig_all, a.diag_every, hid_val, sst_std)
    tokdec.HOOKS["recorder"] = rec
    train_pinn.main(["--run", a.run] + targs)
    assert rec.k == rec.n_total and set(rec.out) == set(a.positions), (rec.k, sorted(rec.out))
    DATA.mkdir(parents=True, exist_ok=True)
    if a.positions:
        flat = {f"{k}__{pos}": v for pos, dd in rec.out.items() for k, v in dd.items()}
        np.savez_compressed(DATA / f"{a.run}_io6.npz", positions=np.array(sorted(rec.out)), **flat)
    if a.krig_all:
        np.savez_compressed(DATA / f"{a.run}_krigall.npz", n_val=a.n_val, n_test=a.n_test, **{k: np.concatenate(v) for k, v in rec.maps.items()})
    if a.diag_every:
        np.savez(DATA / f"{a.run}_diag.npz", sst_edges_k=SST_EDGES_K, sst_std=sst_std, diag_every=a.diag_every, **rec.diag)
    print(f"recorded positions {sorted(rec.out)}, {rec.k} scored days, krig_all {a.krig_all}, diag days {rec.diag['n_days'] if rec.diag else 0}", flush=True)


if __name__ == "__main__":
    main()
