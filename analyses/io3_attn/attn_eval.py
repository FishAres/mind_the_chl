"""Eval-only runs of a trained --t3d-stem attn model with the chl_cnn/tattn.py EVAL_HOOKS (io3 track, Part A; 2026-10-23).

Data generation only: sets the hook, then runs chl_cnn.train_pinn.main(--eval-only ...) with the trainer arguments given after "--".
  --mode default   no hook (bit-identity check of the hooked tattn.py against a run made before the hooks existed)
  --mode record    recorder on (values unchanged): pooling weights (mean over heads) of every val day, accumulated on val hidden
                   pixels by day offset and by whether the day was visible at the pixel (A1a); full weight maps of every
                   --sample-every-th val day saved for A4 (~/mind_the_chl_gap/data/io3/<run>_weights.npz)
  --mode pool4     pooling softmax restricted to |offset| <= --max-offset (renormalised) (A1b)
  --mode noself    self-attention block bypassed (identity) (A3)
Hidden pixels of val day t = observed on t and not visible in day t's input (hid is a subset of obs, vis = obs & ~hid).

Usage (project root): python -m analyses.io3_attn.attn_eval --mode record --run io3_eval_tattn8_rec -- <train_pinn args>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from chl_cnn import tattn, train_pinn, data as D, store as S

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "mind_the_chl_gap" / "data" / "io3"


class Recorder:
    def __init__(self, va, obs, ocean, sample_every):
        self.va, self.obs, self.ocean = va, obs, ocean
        self.k = 0                                             # val days consumed so far (val pass runs first, in va order)
        self.sample = set(range(0, len(va), sample_every))
        self.acc = None
        self.maps, self.vis, self.hid, self.days = [], [], [], []

    def __call__(self, a, x, t_index):
        if self.k >= len(self.va):                             # test pass: nothing recorded
            return
        B, C, T, H, W = x.shape
        days = self.va[self.k:self.k + B]
        assert len(days) == B
        w = a.float().mean(1)[:, 0, :].view(B, H, W, T).permute(0, 3, 1, 2)       # (B, T, H, W), mean over heads
        vis = x[:, -1].float() > 0.5                                                # (B, T, H, W) visible in the input
        obs = torch.as_tensor(self.obs[days], device=x.device)
        hid = obs & ~vis[:, t_index]                                                 # (B, H, W)
        if self.acc is None:
            self.acc = {n: torch.zeros(T, dtype=torch.float64, device=x.device) for n in ("w_vis", "w_nov", "n_vis", "w2_vis", "w2_nov")}
            self.acc["n_hid"] = torch.zeros((), dtype=torch.float64, device=x.device)
            self.acc["w_far_sum"] = torch.zeros((), dtype=torch.float64, device=x.device)
            self.T, self.t_index = T, t_index
        h = hid.unsqueeze(1).double()
        wd, vd = w.double(), vis.double()
        self.acc["w_vis"] += (wd * vd * h).sum((0, 2, 3)); self.acc["w_nov"] += (wd * (1 - vd) * h).sum((0, 2, 3))
        self.acc["w2_vis"] += (wd ** 2 * vd * h).sum((0, 2, 3)); self.acc["w2_nov"] += (wd ** 2 * (1 - vd) * h).sum((0, 2, 3))
        self.acc["n_vis"] += (vd * h).sum((0, 2, 3)); self.acc["n_hid"] += h.sum()
        for i in range(B):
            if self.k + i in self.sample:
                self.maps.append(w[i].half().cpu().numpy()); self.vis.append(vis[i].cpu().numpy())
                self.hid.append(hid[i].cpu().numpy()); self.days.append(int(days[i]))
        self.k += B

    def save(self, run):
        acc = {k: v.cpu().numpy().tolist() for k, v in self.acc.items()}
        res = dict(run=run, n_val_days=int(self.k), t_index=self.t_index, offsets=[i - self.t_index for i in range(self.T)], **acc,
                   n_sample_days=len(self.days))
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / f"{run}_poolweights.json").write_text(json.dumps(res, indent=1))
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA / f"{run}_weights.npz", w=np.stack(self.maps), vis=np.stack(self.vis), hid=np.stack(self.hid),
                            days=np.array(self.days), ocean=self.ocean, t_index=self.t_index)


def main():
    argv = sys.argv[1:]
    i = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["default", "record", "pool4", "noself"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--max-offset", type=int, default=4)
    p.add_argument("--sample-every", type=int, default=15)
    a = p.parse_args(argv[:i])
    targs = ["--run", a.run] + argv[i + 1:]
    assert "--eval-only" in targs
    rec = None
    if a.mode == "record":
        d = S.load_store(); times = d["time"]
        y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & d["ocean"]
        _, va, _ = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
        rec = Recorder(va, obs, d["ocean"], a.sample_every)
        del y, d
        tattn.EVAL_HOOKS["recorder"] = rec
    elif a.mode == "pool4":
        tattn.EVAL_HOOKS["pool_max_offset"] = a.max_offset
    elif a.mode == "noself":
        tattn.EVAL_HOOKS["bypass_self"] = True
    print(f"attn_eval mode {a.mode}: EVAL_HOOKS {dict(tattn.EVAL_HOOKS, recorder=rec is not None)}", flush=True)
    train_pinn.main(targs)
    if rec is not None:
        assert rec.k > 0
        rec.save(a.run)
        print(f"recorded {rec.k} val days, {len(rec.days)} sample maps", flush=True)


if __name__ == "__main__":
    main()
