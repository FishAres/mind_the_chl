"""Real-cloud loss reweighting (io8 track, bridge #25 Phase 1 run 4; run io8_physmem_rwloss; train_pinn.py --rw-loss, default off).

Each day-t hidden-px term of the training data loss gets a weight w(cell) so that the (distance bin x key-count bin) distribution of the
training fake-cloud px matches that of the real-cloud px (ocean, not observed) of the training days. Bins and cell definition are those of
analyses/io7/postrun.py (real-cloud reweighted MAE):
  distance  Euclidean distance (px) to the nearest day-t visible px (visible = observed and not hidden; land is not visible):
            1 (0.5, 1.5], 2-3 (1.5, 3.5], 4-7, 8-15, 16-31, 32+   (scipy distance_transform_edt in postrun; here exact disk tests on the GPU:
            d <= r  <=>  a visible px lies at an integer offset with dy^2 + dx^2 <= r^2, so d <= 1.5 <=> d^2 <= 2, 3.5 -> 12, 7.5 -> 56,
            15.5 -> 240, 31.5 -> 992; disk kernels up to radius 31)
  keys      number of visible io6 kriging-stencil entries (chl_cnn/krig.py stencil_k1, |dt| <= 4, 484 offsets): day t visible px and the
            model-input visible masks of days t+-1..4 (train_pinn.nb_masks: observed, in the store, and with --context-clouds persist observed
            on the donor's neighbour day), as analyses/io7/keys.py: 0, 1-4, 5-19, 20-59, 60+
  cell      dist_bin * 5 + key_bin (30 cells).
Real-cloud px get their cells under the same input visibility as the fake-cloud px of that day (the day's training fake clouds hidden), as
postrun does for val / test.

Weights (fixed for the run; computed once before training, also on resume):
  distributions: --rw-draws training donor draws (as train_epoch: seasonal donors within --donor-window, --donor-exclude) from a separate
  generator (seed + 1000003), so the training draws and batch order are those of the same run without --rw-loss. Over all training days and
  draws: fake = counts of hidden px per cell, real = counts of real-cloud px per cell.
  w_c = p_real(c) / p_fake(c) (0 where p_fake = 0; that real mass is reported as uncovered), then clipped at --rw-cap and renormalised to
  mean 1 under p_fake, repeated until max w <= cap (water-filling); reported: weights, both distributions, uncovered mass, Kish effective
  sample fraction, total-variation distance between the reweighted fake and the real distribution.
Loss: day-t pixel term weight hid -> w(cell) * hid (visible px keep --vis-weight; the t+-1 heads and the gradient-difference term unchanged);
the denominator is the sum of the weights, as before. Only in training (model.training); val / test scoring is unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .krig import stencil_k1, MAX_LAG

DIST_R2 = (2, 12, 56, 240, 992)                                # d^2 thresholds of the distance bin edges 1.5, 3.5, 7.5, 15.5, 31.5
KEY_EDGES = (1, 5, 20, 60)                                     # key bins 0, 1-4, 5-19, 20-59, 60+
DIST_NAMES = ["1", "2-3", "4-7", "8-15", "16-31", "32+"]
KEY_NAMES = ["0", "1-4", "5-19", "20-59", "60+"]
N_CELL = len(DIST_NAMES) * len(KEY_NAMES)
RESUME_DEFAULTS = {"rw_loss": False, "rw_cap": 10.0, "rw_draws": 3}


def add_args(p):
    p.add_argument("--rw-loss", action="store_true", help="io8: weight the day-t hidden-px loss so the (distance x key-count) cell distribution of "
                   "training fake-cloud px matches the real-cloud px of the training days (chl_cnn/rwloss.py)")
    p.add_argument("--rw-cap", type=float, default=10.0, help="--rw-loss: clip of the cell weights (mean 1 under the fake-cloud distribution)")
    p.add_argument("--rw-draws", type=int, default=3, help="--rw-loss: training donor draws used for the two distributions")


def check_args(p, a):
    if (a.rw_cap != 10.0 or a.rw_draws != 3) and not a.rw_loss:
        p.error("--rw-cap / --rw-draws need --rw-loss")
    if a.rw_loss and (a.fake_mask != "donor" or a.recipe != "none" or a.stoch or a.no_chl_input or a.no_chl_today or a.rw_cap < 1 or a.rw_draws < 1):
        p.error("--rw-loss needs --fake-mask donor, no --recipe / --stoch / --no-chl-*, --rw-cap >= 1, --rw-draws >= 1")


def key_kernels():
    """{dt: (1, 1, 2R+1, 2R+1) float32 kernel} of the stencil (as analyses/io7/keys.kernels), R = 6."""
    st = stencil_k1()
    R = max(max(abs(y), abs(x)) for _, y, x in st)
    K = {}
    for dt, y, x in st:
        K.setdefault(dt, np.zeros((2 * R + 1, 2 * R + 1), "float32"))[y + R, x + R] = 1
    return {dt: torch.from_numpy(k)[None, None] for dt, k in K.items()}, R


def disk_kernels():
    """(5, 1, 63, 63) float32: disk i = integer offsets with dy^2 + dx^2 <= DIST_R2[i]."""
    R = 31
    yy, xx = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="ij")
    d2 = yy ** 2 + xx ** 2
    return torch.from_numpy(np.stack([(d2 <= t) for t in DIST_R2]).astype("float32")[:, None]), R


class Binner:
    """cells(vis_t, ctx) -> (B, H, W) int64 cell ids (dist_bin * 5 + key_bin; meaningful on non-visible px).
    vis_t (B, H, W) bool day-t visible; ctx {dt: (B, H, W) bool visible mask of day t + dt} for 1 <= |dt| <= 4 (missing dt = nothing visible)."""

    def __init__(self, device):
        k, self.Rk = key_kernels()
        self.kk = {dt: v.to(device) for dt, v in k.items()}
        d, self.Rd = disk_kernels()
        self.dk = d.to(device)

    @torch.no_grad()
    def cells(self, vis_t, ctx):
        v = vis_t.float().unsqueeze(1)                         # (B, 1, H, W)
        cov = F.conv2d(v, self.dk, padding=self.Rd) > 0.5      # (B, 5, H, W): a visible px within disk i
        dbin = len(DIST_R2) - cov.sum(1)                       # nested disks: 0 = within 1.5 px ... 5 = beyond 31.5 px
        nk = F.conv2d(v, self.kk[0], padding=self.Rk)[:, 0]
        for dt in range(-MAX_LAG, MAX_LAG + 1):
            if dt != 0 and dt in ctx:
                nk = nk + F.conv2d(ctx[dt].float().unsqueeze(1), self.kk[dt], padding=self.Rk)[:, 0]
        nk = torch.round(nk)
        kbin = sum((nk >= e).long() for e in KEY_EDGES)
        return dbin.long() * len(KEY_NAMES) + kbin


def cap_weights(real, fake, cap, max_iter=1000):
    """Cell weights p_real / p_fake clipped at cap and renormalised to mean 1 under p_fake (repeated until max <= cap); and a report dict."""
    real, fake = np.asarray(real, "float64"), np.asarray(fake, "float64")
    pr, pf = real / real.sum(), fake / fake.sum()
    raw = np.where(fake > 0, pr / np.where(fake > 0, pf, 1.0), 0.0)
    w = raw.copy()
    for it in range(max_iter):
        w = np.minimum(w, cap); w = w / (pf * w).sum()
        if w.max() <= cap * (1 + 1e-9):
            break
    q = pf * w                                                 # reweighted fake distribution (sums to 1)
    rep = dict(cap=cap, iterations=it + 1, weights=w.tolist(), raw_ratio=raw.tolist(), p_real=pr.tolist(), p_fake=pf.tolist(),
               n_real=int(real.sum()), n_fake=int(fake.sum()), uncovered_real_mass=float(pr[fake == 0].sum()),
               n_cells_at_cap=int((w >= cap * (1 - 1e-6)).sum()), n_cells_raw_above_cap=int((raw > cap).sum()),
               max_weight=float(w.max()), mean_weight_under_fake=float((pf * w).sum()),
               kish_ess_frac=float((pf * w).sum() ** 2 / (pf * w * w).sum()),
               tv_fake_vs_real=float(0.5 * np.abs(pf - pr).sum()), tv_reweighted_vs_real=float(0.5 * np.abs(q - pr).sum()),
               dist_bins=DIST_NAMES, key_bins=KEY_NAMES, cell="dist_bin * 5 + key_bin")
    return w, rep


def table(rep):
    """Printable 6 x 5 weight table (rows distance bins, columns key bins)."""
    w = np.asarray(rep["weights"]).reshape(len(DIST_NAMES), len(KEY_NAMES))
    pr = np.asarray(rep["p_real"]).reshape(w.shape); pf = np.asarray(rep["p_fake"]).reshape(w.shape)
    lines = ["dist \\ keys   " + "  ".join(f"{k:>18s}" for k in KEY_NAMES) + "   (w / p_real / p_fake)"]
    for i, dn in enumerate(DIST_NAMES):
        lines.append(f"{dn:>12s}  " + "  ".join(f"{w[i, j]:6.2f}/{pr[i, j]:.3f}/{pf[i, j]:.3f}" for j in range(len(KEY_NAMES))))
    return "\n".join(lines)
