"""Stochastic gap filler (train_pinn.py --stoch; io2 track item 8, 2026-10-21).

StochTemporal3DNet = Temporal3DNet (3D stem + dilated backbone) with noise injection: `noise_ch` channels of N(0, 1) noise per pixel
enter the backbone's input conv and the input of every residual block through 1x1 convs whose weights and biases start at zero.
With those at zero the network equals the deterministic model it was warm-started from (state-dict keys of the base model are
unchanged; the new keys start with "noise_"). forward(x3d, x_static, noise=None): noise None -> the plain backbone (no injection
modules at all); features() / head() split the noise-free stem from the noisy backbone so M samples share one stem pass.

fair_crps(X, y): the fair (unbiased-ensemble) CRPS of M samples X[0..M-1] for the truth y, elementwise:
  mean_i |X_i - y| - sum_{i<j} |X_i - X_j| / (M (M - 1)).
For identical samples it equals |X - y|, so the warm-started loss starts at the baseline's L1 loss.
"""
import torch
from torch import nn

from .models import Temporal3DNet, DilatedCNN


def is_noise_param(name):
    return name.startswith("noise_")


def _zero_conv(cin, cout):
    c = nn.Conv2d(cin, cout, 1)
    nn.init.zeros_(c.weight); nn.init.zeros_(c.bias)
    return c


class StochTemporal3DNet(Temporal3DNet):
    def __init__(self, *args, noise_ch=8, **kw):
        super().__init__(*args, **kw)
        bb = self.backbone
        assert isinstance(bb, DilatedCNN), "--stoch needs the dilated backbone"
        w = bb.inp.out_channels
        self.noise_ch = noise_ch
        self.noise_in = _zero_conv(noise_ch, w)
        self.noise_blocks = nn.ModuleList([_zero_conv(noise_ch, w) for _ in bb.blocks])

    def features(self, x3d, x_static=None):
        h = self.stem(x3d)
        return torch.cat([h, x_static.to(h.dtype)], 1) if x_static is not None else h

    def head(self, h, noise):
        bb = self.backbone
        noise = noise.to(h.dtype)
        x = bb.inp(h) + self.noise_in(noise)
        for blk, nb in zip(bb.blocks, self.noise_blocks):
            x = blk(x + nb(noise))
        return bb.out(x)

    def forward(self, x3d, x_static=None, noise=None):
        h = self.features(x3d, x_static)
        return self.backbone(h) if noise is None else self.head(h, noise)


def fair_crps(X, y):
    """X (M, ...) samples, y broadcastable to X[0] -> (...) elementwise fair CRPS."""
    M = X.shape[0]
    t1 = (X - y).abs().mean(0)
    t2 = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            t2 = t2 + (X[i] - X[j]).abs()
    return t1 - t2 / (M * (M - 1))
