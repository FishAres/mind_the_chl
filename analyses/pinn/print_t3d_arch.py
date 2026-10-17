"""Print the Black Sea 3D-stem model (blk_t3d_c4_gdl100_*) layer by layer: parameters and per-sample activation size.
Usage: python analyses/pinn/print_t3d_arch.py   (CPU only; builds the model from the run's config.json)"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from chl_cnn.models import Temporal3DNet

c = json.load(open("models/pinn/t3d/blk_t3d_c4_gdl100_sl_lam0p3/config.json"))
t, (H, W) = c["t3d"], c["grid"]
m = Temporal3DNet(t["c_day"], t["n_static"], t["n_days"], backbone="dilated", stem_width=c["t3d_width"], stem_depth=c["t3d_depth"],
                  kt=c["t3d_kernel_t"], width=c["width"], dropout=c["dropout"], out_channels=3)
rows = []
def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        rows.append((name, type(mod).__name__, sum(p.numel() for p in mod.parameters(recurse=False)), tuple(o.shape[1:]), o.numel() * 2 / 2**20))
    return f
for name, mod in m.named_modules():
    if name and not list(mod.children()):          # leaf layers only
        mod.register_forward_hook(hook(name))
with torch.no_grad():
    m(torch.zeros(1, t["c_day"], t["n_days"], H, W), torch.zeros(1, t["n_static"], H, W))
print(f"grid {H} x {W}, input 3D (C={t['c_day']}, T={t['n_days']}), static C={t['n_static']}; activations in MB per sample at fp16")
print(f"{'layer':40s} {'type':12s} {'params':>9s}  {'output shape (C,[T],H,W)':26s} {'act MB':>7s}")
tot_p = tot_a = 0
for name, typ, p, shp, mb in rows:
    print(f"{name:40s} {typ:12s} {p:9,d}  {str(shp):26s} {mb:7.1f}"); tot_p += p; tot_a += mb
print(f"{'TOTAL':40s} {'':12s} {tot_p:9,d}  {'':26s} {tot_a:7.0f}")
print(f"check: model params {sum(p.numel() for p in m.parameters()):,d} (config n_params {c['n_params']:,d})")
