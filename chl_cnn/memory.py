"""Memory channels for the gap filler (train_pinn.py --memory; io2 track item 5, 2026-10-21).

Per pixel and target day t, from days outside / around the +-4-day input window, computed on the GPU per batch:
  0 last_z    z-scored ln CHL of the last valid observation among t-5 .. t-max_age (0 if none)
  1 last_age  its age in days / max_age (1 if none)
  2 next_z    z-scored ln CHL of the first valid observation among t+5 .. t+max_age (0 if none)
  3 next_age  its age in days / max_age (1 if none)
  4 med8      median z ln CHL over t-4..t+4, k != 0 (8 days; 0 if no valid day)
  5 cnt8      number of valid days / 8
  6 med30     median z ln CHL over t-15..t+15, k != 0 (30 days; 0 if no valid day)
  7 cnt30     number of valid days / 30
and, for --residual-base anom, the 30-day median of the z anomaly (ln CHL(t+k) - DOY clim(t+k), z units) with its count.

"Valid" on day t+k follows the --context-clouds persist rule of train_pinn.nb_masks: observed on t+k AND observed on the donor's
day d+k (d = the fake-cloud donor of day t; when d+k is not in the store, the donor day d itself). A day t+k missing from the store
is not valid. Day t itself is never used, so day t's hidden truth cannot reach these channels (analyses/io_improve/leak_test.py).
Medians are torch.nanmedian (the lower of the two middle values for an even count).
"""
import numpy as np
import torch

from . import train2 as T2

N_CH = 8


class Memory:
    def __init__(self, times, OBSg, DONg, chl_z, clim_z=None, max_age=60, inner=4, w_short=4, w_long=15, dev="cuda"):
        """OBSg (N,H,W) bool observed; DONg (N,) donor index of each day (read at call time: training donors change per epoch);
        chl_z(idx) -> z ln CHL (0 where unobserved); clim_z(idx) -> (B,H,W) z DOY climatology of the days idx (for the anomaly)."""
        self.OBS, self.DON, self.chl_z, self.clim_z = OBSg, DONg, chl_z, clim_z
        self.max_age, self.inner, self.w_short, self.w_long = max_age, inner, w_short, w_long
        ks = [k for k in range(-max_age, max_age + 1) if k != 0]
        self.NB = {k: torch.as_tensor(T2.neighbour_index(times, k), device=dev) for k in ks}

    def _mask(self, bi, k):
        """Neighbour index j of t+k (clamped), and its valid mask under the donor-sequence (persist) rule."""
        j = self.NB[k][bi]; ok = (j >= 0).view(-1, 1, 1); j = j.clamp(min=0)
        d = self.DON[bi]
        jd = self.NB[k][d]; jd = torch.where(jd >= 0, jd, d)
        return j, self.OBS[j] & ok & self.OBS[jd]

    @torch.no_grad()
    def __call__(self, bi, want_anom=False):
        """(B, 8, H, W) float32 channels; with want_anom also (anom30 (B,H,W) z units, 0 where none; count30 (B,H,W))."""
        cache = {}

        def get(k):
            if k not in cache:
                j, m = self._mask(bi, k)
                cache[k] = (j, m, self.chl_z(j))
            return cache[k]

        out = []
        for sign in (-1, 1):                                   # last before t-inner, first after t+inner
            val = None
            for a in range(self.inner + 1, self.max_age + 1):
                j, m, z = get(sign * a)
                if val is None:
                    val = torch.zeros_like(z); age = torch.full_like(z, float(self.max_age)); found = torch.zeros_like(m)
                new = m & ~found
                val = torch.where(new, z, val); age = torch.where(new, torch.full_like(z, float(a)), age); found = found | new
            out += [val, age / self.max_age]
        anom = cnt_a = None
        for w in (self.w_short, self.w_long):
            ks = [k for k in range(-w, w + 1) if k != 0]
            zs, ms = [], []
            for k in ks:
                j, m, z = get(k)
                zs.append(torch.where(m, z, torch.full_like(z, float("nan")))); ms.append(m)
            Z = torch.stack(zs, 1); M = torch.stack(ms, 1)
            cnt = M.sum(1).float()
            med = torch.nan_to_num(torch.nanmedian(Z, 1).values, nan=0.0)
            out += [med, cnt / len(ks)]
            if want_anom and w == self.w_long:
                A = torch.stack([torch.where(ms[i], zs[i] - self.clim_z(cache[k][0]), torch.full_like(zs[i], float("nan")))
                                 for i, k in enumerate(ks)], 1)
                anom = torch.nan_to_num(torch.nanmedian(A, 1).values, nan=0.0); cnt_a = cnt
        ch = torch.stack(out, 1)
        return (ch, anom, cnt_a) if want_anom else ch
