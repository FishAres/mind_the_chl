"""Model zoo for the performance runs (train2.py).

SimpleCNN   the 3-layer baseline port (model.py), receptive field 7 x 7 pixels
DilatedCNN  residual 3x3 blocks with dilations 1,2,4,8,16,... -> receptive field ~150 px, no downsampling;
            optional spectral path (FNO-style global convolution in Fourier space) added to the first
            `spectral` blocks: 1x1 down-projection -> complex weights on the lowest `modes` x `modes` Fourier
            modes -> 1x1 up-projection, summed with the local conv path
UNet        4-level encoder/decoder (32-64-128-256-512) with skip connections; input is padded
            to a multiple of 16 and cropped back, so it works on the 149 x 181 grid

All models take (B, C, H, W) and return (B, H, W), or (B, out_channels, H, W) when out_channels > 1
(train_pinn.py asks for 3: ln CHL at t-1, t, t+1).
"""
import torch
from torch import nn
import torch.nn.functional as F

from .model import SimpleCNN, count_params  # noqa: F401  (re-exported)


class ResBlock(nn.Module):
    def __init__(self, w, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(w), nn.GELU(), nn.Conv2d(w, w, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(w), nn.GELU(), nn.Conv2d(w, w, 3, padding=dilation, dilation=dilation))

    def forward(self, x):
        return x + self.body(x)


class SpectralConv2d(nn.Module):
    """FNO spectral convolution: rfft2 -> learned complex weights on the lowest (m1, m2) modes -> irfft2 (Li et al. 2021)."""

    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (cin * cout)
        # stored as real (…, 2) so the AMP grad scaler can handle them; viewed as complex in forward
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.autocast("cuda", enabled=False):
            xf = torch.fft.rfft2(x.float(), norm="ortho")
            w1, w2 = torch.view_as_complex(self.w1), torch.view_as_complex(self.w2)
            out = torch.zeros(B, w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            m1, m2 = min(self.m1, H // 2), min(self.m2, W // 2 + 1)
            out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], w1[:, :, :m1, :m2])
            out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], w2[:, :, :m1, :m2])
            y = torch.fft.irfft2(out, s=(H, W), norm="ortho")
        return y.to(x.dtype)


class SpectralResBlock(nn.Module):
    """ResBlock plus a bottlenecked spectral path: x + conv(x) + up(spectral(down(x)))."""

    def __init__(self, w, dilation, modes=16, spec_ch=16):
        super().__init__()
        self.local = ResBlock(w, dilation)
        self.down = nn.Conv2d(w, spec_ch, 1)
        self.spec = SpectralConv2d(spec_ch, spec_ch, modes, modes)
        self.up = nn.Sequential(nn.GELU(), nn.Conv2d(spec_ch, w, 1))

    def forward(self, x):
        return self.local(x) + self.up(self.spec(self.down(x)))


class DilatedCNN(nn.Module):
    def __init__(self, in_channels, width=64, dilations=(1, 2, 4, 8, 16, 1, 2, 4), dropout=0.0, out_channels=1,
                 spectral=0, modes=16, spec_ch=16):
        super().__init__()
        self.inp = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*[(SpectralResBlock(width, d, modes, spec_ch) if i < spectral else ResBlock(width, d))
                                      for i, d in enumerate(dilations)])
        self.out = nn.Sequential(nn.BatchNorm2d(width), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(width, out_channels, 1))
        self.out_channels = out_channels

    def forward(self, x):
        y = self.out(self.blocks(self.inp(x)))
        return y.squeeze(1) if self.out_channels == 1 else y


def double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_channels, base=32, depth=4, dropout=0.0, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        chans = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_channels
        for w in chans[:-1]:
            self.enc.append(double_conv(c, w)); c = w
        self.bottom = double_conv(c, chans[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = chans[-1]
        for w in reversed(chans[:-1]):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.dec.append(double_conv(2 * w, w)); c = w
        self.drop = nn.Dropout2d(dropout)
        self.out = nn.Conv2d(c, out_channels, 1)
        self.mult = 2 ** depth

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % self.mult, (-W) % self.mult
        x = F.pad(x, (0, pw, 0, ph))
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([u(x), s], dim=1))
        y = self.out(self.drop(x))[..., :H, :W]
        return y.squeeze(1) if self.out_channels == 1 else y


def build_model(name, in_channels, width=64, dropout=0.0, out_channels=1, depth=4, spectral=0, modes=16, spec_ch=16, dilations=None):
    """dilations: DilatedCNN residual-block dilations (None = the DilatedCNN default 1,2,4,8,16,1,2,4); ignored by other models."""
    if name == "simple":
        assert out_channels == 1
        return SimpleCNN(in_channels, dropout=dropout)
    if name == "dilated":
        kw = {} if dilations is None else dict(dilations=tuple(dilations))
        return DilatedCNN(in_channels, width=width, dropout=dropout, out_channels=out_channels, spectral=spectral, modes=modes, spec_ch=spec_ch, **kw)
    if name == "unet":
        return UNet(in_channels, base=width, depth=depth, dropout=dropout, out_channels=out_channels)
    raise ValueError(name)


class TemporalStem3D(nn.Module):
    """3D-convolution front-end over adjacent days (train_pinn.py --model temporal3d).

    Input (B, C_day, T, H, W): the same per-day channel layout for every day of the window t-K..t+K.
    `depth` x [Conv3d(kernel (kt, 3, 3), same padding) -> GroupNorm -> GELU], then the day axis is collapsed with a
    learned Conv3d of kernel (T, 1, 1) (no padding) -> (B, width, H, W).
    """

    def __init__(self, c_day, T, width=32, depth=2, kt=3):
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        layers, c = [], c_day
        for _ in range(depth):
            layers += [nn.Conv3d(c, width, (kt, 3, 3), padding=(kt // 2, 1, 1)), nn.GroupNorm(groups, width), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.collapse = nn.Conv3d(width, width, (T, 1, 1))
        self.T = T

    def forward(self, x):
        assert x.shape[2] == self.T, f"expected {self.T} days, got {x.shape[2]}"
        return self.collapse(self.body(x)).squeeze(2)


class Temporal3DNet(nn.Module):
    """TemporalStem3D on the day-stacked channels, concatenated with the static 2D channels, fed to a 2D backbone
    (build_model('dilated' | 'unet', in_channels=width + n_static)). forward(x3d, x_static) -> backbone output."""

    def __init__(self, c_day, n_static, T, backbone="dilated", stem_width=32, stem_depth=2, kt=3, **backbone_kw):
        super().__init__()
        self.stem = TemporalStem3D(c_day, T, width=stem_width, depth=stem_depth, kt=kt)
        self.backbone = build_model(backbone, stem_width + n_static, **backbone_kw)
        self.out_channels = self.backbone.out_channels

    def forward(self, x3d, x_static=None):
        h = self.stem(x3d)
        if x_static is not None:
            h = torch.cat([h, x_static.to(h.dtype)], 1)
        return self.backbone(h)
