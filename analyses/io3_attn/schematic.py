"""Architecture schematic of the patch-token gap-filler (arm A, run io3_ptok4) for analyses/forecast/MODELS.md.

No GPU, no training. Numbers come from the run artefacts, not from memory:
  models/pinn/t3d/io3_ptok4/config.json   window, patch size, dims, batch, loss weights, total parameters
  models/pinn/t3d/io3_ptok4/model.pt      stem / backbone parameter split, token-embedding and grid shapes
  models/pinn/t3d/io2_tattn8/model.pt     old per-pixel attention stem (chl_cnn/tattn.py), parameter split
  analyses/io3_attn/results/partb.json    s / epoch and peak GPU memory of both runs
  the store's ocean mask                  number of ocean 4 x 4 patches
Architecture as in chl_cnn/ptok.py (PatchTokenNet, arm "interp", value "own") and chl_cnn/models.py (DilatedCNN).

Usage (project root): python analyses/io3_attn/schematic.py
Writes analyses/io3_attn/figures/ptok_schematic.{png,pdf}; prints any text overlap / overflow it finds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from chl_cnn import store as S                                     # noqa: E402
from chl_cnn.plotting import OKABE_ITO                             # noqa: E402

RUN, OLD = "io3_ptok4", "io2_tattn8"
RUN_DIR = ROOT / "models" / "pinn" / "t3d"
FIG = HERE / "figures"

BLUE, ORANGE, GREEN, VERM, PINK, SKY, YELLOW, BLACK = OKABE_ITO
GREY_C = "#333333"
FC_IN, FC_PATCH, FC_FULL = "#F2F2F2", "#DCEBF5", "#FBE9D0"            # as analyses/forecast architecture.png
LANE_PATCH, LANE_FULL = "#EEF5FB", "#FDF6EC"


# ============================================================================ data

def _params(sd):
    ok = lambda k, v: v.dtype.is_floating_point and "running" not in k
    tot = sum(v.numel() for k, v in sd.items() if ok(k, v))
    bb = sum(v.numel() for k, v in sd.items() if ok(k, v) and k.startswith("backbone"))
    return tot, bb, tot - bb


def load_numbers():
    c = json.loads((RUN_DIR / RUN / "config.json").read_text())
    co = json.loads((RUN_DIR / OLD / "config.json").read_text())
    sd = torch.load(RUN_DIR / RUN / "model.pt", map_location="cpu")["model"]
    sdo = torch.load(RUN_DIR / OLD / "model.pt", map_location="cpu")["model"]
    tot, bb, stem = _params(sd)
    toto, bbo, stemo = _params(sdo)
    assert tot == c["n_params"] and toto == co["n_params"], "parameter count mismatch with config.json"
    H, W = c["grid"]
    P = c["ptok_patch"]
    Hp, Wp = -(-H // P) * P, -(-W // P) * P
    oc = np.zeros((Hp, Wp)); oc[:H, :W] = np.asarray(S.load_store()["ocean"], dtype=float)
    n_oc = oc.reshape(Hp // P, P, Wp // P, P).sum((1, 3))
    assert sd["pos_emb"].shape[0] == n_oc.size, "patch grid mismatch with the saved position embedding"
    rows = {r["run"]: r for r in json.loads((HERE / "results" / "partb.json").read_text())["rows"]}
    return dict(
        H=H, W=W, P=P, Hp=Hp, Wp=Wp, nPy=n_oc.shape[0], nPx=n_oc.shape[1], Np=int((n_oc > 0).sum()),
        T=c["t3d"]["n_days"], K=c["context"], c_day=c["t3d"]["c_day"], n_static=c["t3d"]["n_static"], in_ch=c["in_channels"],
        n_stat=sd["embed.0.weight"].shape[1], D=c["ptok_dim"], heads=c["ptok_heads"], layers=c["ptok_layers"],
        n_pf=sd["pix.0.weight"].shape[1], width=c["width"], dil=c["dilations"], batch=c["batch"],
        vis_w=c["vis_weight"], grad_w=c["grad_weight"], nb_w=c["nb_weight"],
        tot=tot, bb=bb, stem=stem, s_ep=rows[RUN]["s_per_epoch"], gb=rows[RUN]["peak_gpu_gb"],
        old_tot=toto, old_bb=bbo, old_stem=stemo, old_batch=co["batch"], old_s_ep=rows[OLD]["s_per_epoch"],
        old_gb=rows[OLD]["peak_gpu_gb"], old_dim=co["attn_dim"], old_heads=co["attn_heads"], old_layers=co["attn_layers"],
        old_in=co["in_channels"])


# ============================================================================ drawing helpers

TEXTS = []          # (text artist, containing box (x0, y0, x1, y1) in data inches or None)


def txt(ax, x, y, s, box=None, **kw):
    kw.setdefault("fontsize", 10)
    t = ax.text(x, y, s, **kw)
    TEXTS.append((t, box))
    return t


def box(ax, x, y, w, h, fc, header=None, body="", num=None, fs=10, hfs=11, body_top=None, ec=GREY_C, lw=1.0, vcenter=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=lw, zorder=2))
    bb = (x, y, x + w, y + h)
    yt = y + h - 0.12
    if header:
        txt(ax, x + w / 2 + (0.13 if num else 0), yt, header, bb, ha="center", va="top", fontsize=hfs, weight="bold", zorder=4)
        yt -= 0.30
    if num is not None:
        ax.add_patch(Circle((x + 0.2, y + h - 0.2), 0.14, fc=BLACK, ec="none", zorder=4))
        ax.text(x + 0.2, y + h - 0.2, str(num), ha="center", va="center", fontsize=9.5, color="white", weight="bold", zorder=5)
    if body:
        txt(ax, x + w / 2, y + h / 2 if vcenter else (body_top if body_top is not None else yt), body, bb, ha="center",
            va="center" if vcenter else "top", fontsize=fs, linespacing=1.3, zorder=4)
    return bb


def arrow(ax, pts, color=GREY_C, lw=1.6, z=3):
    """Polyline arrow through pts [(x, y), ...]; head at the last point."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.add_line(Line2D(xs[:-1], ys[:-1], color=color, lw=lw, zorder=z, solid_capstyle="butt"))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))


def day_strip(ax, x0, x1, y, T, ti, kind="square", label=True, bb=None):
    """Row of T day markers between x0 and x1 at height y; day t highlighted."""
    step = (x1 - x0) / T
    s = step * 0.72
    cx = [x0 + step * (i + 0.5) for i in range(T)]
    for i, c in enumerate(cx):
        col = ORANGE if i == ti else "white"
        if kind == "square":
            ax.add_patch(Rectangle((c - s / 2, y - s / 2), s, s, fc=col, ec=GREY_C, lw=0.7, zorder=4))
        else:
            ax.add_patch(Circle((c, y), s / 2, fc=col, ec=GREY_C, lw=0.7, zorder=5))
    if label:
        for i, lab in [(0, "t−8"), (ti, "t"), (T - 1, "t+8")]:
            txt(ax, cx[i], y - s / 2 - 0.04, lab, bb, ha="center", va="top", fontsize=9, zorder=4)
    return cx, s


# ============================================================================ figure

def figure(n, path_png, path_pdf):
    FW, FH = 21.0, 12.6
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
    H, W, T, ti, D = n["H"], n["W"], n["T"], n["K"], n["D"]

    txt(ax, FW / 2, FH - 0.15, f"Patch-token gap-filler (arm A, {RUN})", ha="center", va="top", fontsize=17, weight="bold")
    txt(ax, FW / 2, FH - 0.58,
        f"chl_cnn/ptok.py PatchTokenNet + chl_cnn/models.py DilatedCNN.  train_pinn.py --model temporal3d --t3d-stem patch "
        f"--ptok-patch {n['P']} --ptok-dim {D} --ptok-heads {n['heads']} --ptok-layers {n['layers']} --ptok-arm interp "
        f"--ptok-value own --context {n['K']} --batch {n['batch']}", ha="center", va="top", fontsize=10, color="#444444")
    txt(ax, FW / 2, FH - 0.88,
        f"B = batch, T = {T} days (t−{n['K']} … t+{n['K']}), H × W = {H} × {W} (0.25° grid), patch grid {n['nPy']} × {n['nPx']} "
        f"(P = {n['P']} px), D = {D}.   Parameters {n['tot']:,}: stem {n['stem']:,}, backbone {n['bb']:,}.   "
        f"{n['s_ep']:.0f} s / epoch at batch {n['batch']}, {n['gb']:.1f} GB peak GPU memory.",
        ha="center", va="top", fontsize=10.5)

    # ---- lanes
    yP0, yP1 = 7.55, 11.45                          # patch-grid lane
    yF0, yF1 = 3.05, 7.35                           # full-resolution lane
    ax.add_patch(Rectangle((0.15, yP0), FW - 0.3, yP1 - yP0, fc=LANE_PATCH, ec="none", zorder=0))
    ax.add_patch(Rectangle((0.15, yF0), FW - 0.3, yF1 - yF0, fc=LANE_FULL, ec="none", zorder=0))
    txt(ax, 0.32, (yP0 + yP1) / 2, f"patch grid {n['nPy']} × {n['nPx']}", ha="center", va="center", rotation=90,
        fontsize=11, weight="bold", color=BLUE)
    txt(ax, 0.32, (yF0 + yF1) / 2, f"full resolution {H} × {W}", ha="center", va="center", rotation=90,
        fontsize=11, weight="bold", color=VERM)

    # columns (x0, width)
    C1, C2, C3, C4 = (0.6, 2.3), (3.2, 2.75), (6.25, 2.65), (9.2, 2.45)
    C5, C6, C7, C8 = (12.15, 1.85), (14.3, 2.0), (16.6, 1.75), (18.65, 2.2)
    bt, bh = yP0 + 0.18, yP1 - yP0 - 0.36            # patch-lane box bottom / height

    # ---- 1 inputs (full res)
    yI0, yI1 = 4.45, yF1 - 0.15
    b1 = box(ax, C1[0], yI0, C1[1], yI1 - yI0, FC_IN, "Inputs per day", num=1, fs=10,
             body=f"d = t−{n['K']} … t+{n['K']}\n9 physics fields (z)\nln CHL (z), 0 where\nnot visible\nvisible mask\n"
                  f"(B, {n['c_day']}, {T}, H, W)\n+ DOY climatology z\nof each day (B, {T}, H, W)\n\ncontext days carry\nthe donor's clouds")
    day_strip(ax, C1[0] + 0.1, C1[0] + C1[1] - 0.1, yI0 + 0.42, T, ti, bb=b1)
    bS = box(ax, C1[0], yF0 + 0.15, C1[1], 0.72, FC_IN, None, fs=9,
             body=f"static maps, day t (B, {n['n_static']}, H, W):\nclim z, lat, lon, log10 depth,\nDOY sin/cos, year")

    # ---- legend (patch lane, left)
    lx, ly = C1[0], bt + bh
    txt(ax, lx, ly - 0.05, "Arrows", ha="left", va="top", fontsize=10.5, weight="bold")
    for k, (col, lab) in enumerate([(VERM, "observed CHL pixels,\nfull resolution"),
                                    (BLUE, "patch values,\nbilinear upsampling"),
                                    (GREY_C, "other inputs and\nintermediate tensors")]):
        yy = ly - 0.62 - k * 0.78
        arrow(ax, [(lx, yy), (lx + 0.55, yy)], color=col, lw=2.2)
        txt(ax, lx + 0.65, yy, lab, ha="left", va="center", fontsize=9.5, linespacing=1.2)
    ax.add_patch(Rectangle((lx + 0.05, ly - 2.9), 0.3, 0.3, fc=ORANGE, ec=GREY_C, lw=0.7))
    txt(ax, lx + 0.5, ly - 2.75, "day t marker", ha="left", va="center", fontsize=9.5)

    # ---- 2 patch tokens (patch lane)
    b2 = box(ax, C2[0], bt, C2[1], bh, FC_PATCH, "Patch tokens", num=2, fs=9.5, body_top=bt + bh - 1.5,
             body=f"pad to {n['Hp']} × {n['Wp']}, {n['P']} × {n['P']} px patches:\n"
                  f"{n['nPy']} × {n['nPx']} grid, {n['Np']:,} ocean patches\n"
                  f"per patch and day, {n['n_stat']} stats:\nCHL-anomaly mean, std (visible px),\n"
                  f"visible fraction, 9 physics means\nMLP {n['n_stat']} → {D} → {D}\n+ day-offset + patch-position emb.\n"
                  f"→ (B, {n['Np']:,}, {T}, {D})\nfully cloudy patch: physics-only token")
    # pixel-patch graphic: 4 x 4 pixels (grey = not visible) -> token bar
    gx, gy, s = C2[0] + 0.55, bt + bh - 1.32, 0.17
    rng = np.random.default_rng(3)
    vals = rng.uniform(0.2, 0.9, (4, 4)); cloud = np.zeros((4, 4), bool); cloud[0, 2:] = True; cloud[1, 3] = True
    cm = plt.get_cmap("viridis")
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((gx + j * s, gy + (3 - i) * s), s, s, fc="#BBBBBB" if cloud[i, j] else cm(vals[i, j]),
                                   ec="white", lw=0.6, zorder=4))
    txt(ax, gx + 2 * s, gy - 0.04, "4 × 4 px", b2, ha="center", va="top", fontsize=9)
    arrow(ax, [(gx + 4 * s + 0.12, gy + 2 * s), (gx + 4 * s + 0.62, gy + 2 * s)], lw=1.2)
    tx = gx + 4 * s + 0.75
    for k in range(6):
        ax.add_patch(Rectangle((tx, gy + k * 0.115 - 0.01), 0.16, 0.115, fc=SKY, ec="white", lw=0.6, zorder=4))
    txt(ax, tx + 0.25, gy + 0.34, f"token\n({D})", b2, ha="left", va="center", fontsize=9, linespacing=1.1)

    # ---- 3 temporal self-attention (patch lane)
    b3 = box(ax, C3[0], bt, C3[1], bh, FC_PATCH, "Temporal self-attention", num=3, fs=9.5, body_top=bt + bh - 1.5,
             body=f"within each patch, over its\n{T} day tokens; no attention\nacross patches\n"
                  f"{n['layers']} pre-norm blocks, {n['heads']} heads, D = {D},\nfused SDPA, batched over patches:\n"
                  f"(B · {n['Np']:,}, {T}, {D})\ndays missing from the store\nmasked out of the keys\n"
                  f"→ LayerNorm → (B, {n['Np']:,}, {T}, {D})")
    sy = bt + bh - 1.15
    cx, s3 = day_strip(ax, C3[0] + 0.15, C3[0] + C3[1] - 0.15, sy, T, ti, kind="circle", label=False)
    for i in range(T):
        if i == ti:
            continue
        a, b_ = sorted([cx[i], cx[ti]])
        r = (b_ - a) / 2
        th = np.linspace(0, np.pi, 30)
        ax.plot((a + b_) / 2 + r * np.cos(th), sy + s3 / 2 + 0.55 * r * np.sin(th) / max(1, (cx[-1] - cx[0]) / 2) * 1.0 + 0.0,
                color=SKY, lw=0.8, zorder=3)
    txt(ax, cx[ti], sy - s3 / 2 - 0.04, "t", b3, ha="center", va="top", fontsize=9)
    txt(ax, cx[0], sy - s3 / 2 - 0.04, "t−8", b3, ha="center", va="top", fontsize=9)
    txt(ax, cx[-1], sy - s3 / 2 - 0.04, "t+8", b3, ha="center", va="top", fontsize=9)

    # ---- 4 day scores
    yS0 = bt + 0.9
    b4 = box(ax, C4[0], yS0, C4[1], bt + bh - yS0, FC_PATCH, "Patch day score", num=4, fs=10,
             body=f"q = W$_q$ tok(patch, t)\nk = W$_k$ tok(patch, d)\ns = q · k / √{D}\n(B, {n['Np']:,}, {T}) → grid\n"
                  f"(B, {T}, {n['nPy']}, {n['nPx']})")
    yU0, yU1 = 5.95, yF1 - 0.15
    bU = box(ax, C4[0], yU0, C4[1], yU1 - yU0, FC_FULL, None, fs=9.5,
             body=f"bilinear upsampling between\nocean-patch centres (land\nexcluded, normalised)\n→ (B, {T}, H, W)")
    yX0, yX1 = 4.45, 5.65
    bX = box(ax, C4[0], yX0, C4[1], yX1 - yX0, FC_FULL, "+ per-pixel term", fs=9.5, hfs=10.5,
             body=f"1×1-conv MLP {n['n_pf']} → 16 → 1 on\nvisible$_d$, |d−t|/{n['K']}, (d−t)/{n['K']},\n|SST$_d$ − SST$_t$|")
    arrow(ax, [(C4[0] + C4[1] / 2, yS0), (C4[0] + C4[1] / 2, yU1)], color=BLUE, lw=2.2)
    # sum node
    px, py = C4[0] + C4[1] + 0.25, (yX1 + yU0) / 2
    ax.add_patch(Circle((px, py), 0.16, fc="white", ec=GREY_C, lw=1.2, zorder=4))
    ax.text(px, py, "+", ha="center", va="center", fontsize=13, weight="bold", zorder=5)
    arrow(ax, [(C4[0] + C4[1], (yU0 + yU1) / 2), (px, (yU0 + yU1) / 2), (px, py + 0.16)])
    arrow(ax, [(C4[0] + C4[1], (yX0 + yX1) / 2), (px, (yX0 + yX1) / 2), (px, py - 0.16)])

    # ---- 5 softmax, 6 blend (full res)
    yM0, yM1 = 4.45, yF1 - 0.15
    ym = (yM0 + yM1) / 2
    b5 = box(ax, C5[0], yM0, C5[1], yM1 - yM0, FC_FULL, "Per-pixel\nsoftmax", num=5, fs=9.5, body_top=yM1 - 0.85,
             body=f"over context days\nd ≠ t visible at p\n+ learned \"no day\"\nlogit\n\nw: (B, {T} + 1, H, W)")
    arrow(ax, [(px + 0.16, py), (C5[0], py)])
    b6 = box(ax, C6[0], yM0, C6[1], yM1 - yM0, FC_FULL, "Blend", num=6, fs=9.5,
             body="blend = $\\sum_d\\, w_d(p)\\, a_d(p)$\n$a_d$: own visible CHL\nanomaly on day d\n\n"
                  "mass = 1 − w$_{none}$\nentropy = −Σ w log w\n\n3 × (B, 1, H, W)")
    arrow(ax, [(C5[0] + C5[1], ym), (C6[0], ym)])

    # ---- 7 backbone input + FiLM
    yC0 = yF0 + 0.15
    b7 = box(ax, C7[0], yC0, C7[1], yM1 - yC0, FC_FULL, "Backbone\ninput", num=7, fs=9.5, body_top=yM1 - 0.85,
             body=f"concat at full res:\n\nblend, mass,\nentropy (3)\n\nday t's own\nchannels ({n['c_day']})\n\n"
                  f"static ({n['n_static']})\n\n→ (B, {n['in_ch']}, H, W)")
    arrow(ax, [(C6[0] + C6[1], ym), (C7[0], ym)])
    bF = box(ax, C7[0], bt + 0.9, C8[0] + C8[1] - C7[0], bh - 0.9, FC_PATCH, "FiLM from day t's token", num=7, fs=9.5,
             body=f"tok(patch, t) → Linear {D} → {2 * n['width']} (zero-initialised)\n→ (γ, β) per patch\n"
                  f"bilinear upsampling → (B, {2 * n['width']}, H, W)\napplied after the first conv: h·(1 + γ) + β")

    # ---- 8 backbone + output (full res)
    yB0 = 5.25
    b8 = box(ax, C8[0], yB0, C8[1], yM1 - yB0, FC_FULL, "Dilated backbone", num=8, fs=9.5,
             body=f"Conv {n['in_ch']} → {n['width']}, 3×3\nthen h·(1 + γ) + β\n{len(n['dil'])} residual blocks, width {n['width']}\n"
                  f"x + [BN, GELU, Conv 3×3,\ndilation d] × 2\nd = {','.join(map(str, n['dil']))}\n"
                  f"BN, GELU, Conv {n['width']} → 3, 1×1")
    arrow(ax, [(C7[0] + C7[1], ym), (C8[0], ym)])
    fx = C8[0] + C8[1] / 2
    arrow(ax, [(fx, bt + 0.9), (fx, yM1)], color=BLUE, lw=2.2)
    bO = box(ax, C8[0], yF0 + 0.15, C8[1], yB0 - yF0 - 0.45, FC_IN, None, fs=9,
             body=f"output (B, 3, H, W):\nz of t−1, t, t+1\n+ DOY clim z (residual)\n× sd + mean → ln CHL\n\n"
                  f"loss: L1 day-t hidden px\n+ {n['vis_w']} × visible px\n+ {n['grad_w']:.0f} × gradient term\n"
                  f"+ {n['nb_w']:.0f} × L1 t±1 donor-hidden")
    arrow(ax, [(C8[0] + C8[1] / 2, yB0), (C8[0] + C8[1] / 2, bO[3])])

    # ---- cross-lane and long arrows
    xq = C1[0] + C1[1] - 0.2
    arrow(ax, [(xq, yI1), (xq, bt + 0.45), (C2[0], bt + 0.45)])
    txt(ax, xq - 0.08, bt + 0.2, "mask-aware\npooling", ha="right", va="center", fontsize=9.5, linespacing=1.1)
    arrow(ax, [(C2[0] + C2[1], bt + bh / 2), (C3[0], bt + bh / 2)])
    arrow(ax, [(C3[0] + C3[1], yS0 + 1.0), (C4[0], yS0 + 1.0)])
    # day t's token to FiLM: along the top of the patch lane
    yr = yP1 - 0.09
    arrow(ax, [(C3[0] + C3[1] - 0.3, bt + bh), (C3[0] + C3[1] - 0.3, yr), (C7[0] + 0.6, yr), (C7[0] + 0.6, bt + bh)])
    txt(ax, (C5[0] + C7[0]) / 2, yr - 0.06, "day t's token tok(patch, t)", ha="center", va="top", fontsize=9.5)
    # per-pixel features of day d -> per-pixel term
    yp = (yX0 + yX1) / 2
    arrow(ax, [(C1[0] + C1[1], yp), (C4[0], yp)])
    txt(ax, (C1[0] + C1[1] + C4[0]) / 2, yp + 0.06, "per-pixel features: visible$_d$, SST$_d$, SST$_t$",
        ha="center", va="bottom", fontsize=9.5)
    # observed pixels at full resolution (fine-detail routes)
    ya = yI0 + 0.18
    arrow(ax, [(C1[0] + C1[1], ya), (C6[0] + C6[1] / 2, ya), (C6[0] + C6[1] / 2, yM0)], color=VERM, lw=2.4)
    txt(ax, (C1[0] + C1[1] + C4[0]) / 2, ya + 0.06, "each context day's own visible CHL anomaly $a_d$",
        ha="center", va="bottom", fontsize=9.5, color=VERM, weight="bold")
    yb = yI0 - 0.35
    arrow(ax, [(C1[0] + C1[1] / 2, yI0), (C1[0] + C1[1] / 2, yb + 0.0), (C1[0] + C1[1] / 2, yb), (C7[0], yb)],
          color=VERM, lw=2.4)
    txt(ax, (C1[0] + C1[1] + C4[0]) / 2, yb + 0.06,
        f"day t's own channels (B, {n['c_day']}, H, W): physics, visible CHL z, visible mask",
        ha="center", va="bottom", fontsize=9.5, color=VERM, weight="bold")
    yc = bS[1] + 0.35
    arrow(ax, [(bS[2], yc), (C7[0], yc)])
    txt(ax, (C1[0] + C1[1] + C4[0]) / 2, yc + 0.06, f"static maps (B, {n['n_static']}, H, W)",
        ha="center", va="bottom", fontsize=9.5)

    # ---- fine-detail note (patch lane, between the score and FiLM)
    nx0, nx1 = C5[0], C7[0] - 0.3
    bN = box(ax, nx0, bt + 0.55, nx1 - nx0, bh - 1.1, "white", None, fs=10, ec=VERM, lw=1.6,
             body="Fine detail (below the 4 × 4 px\npatch) enters only through\nobserved full-resolution pixels:\n"
                  "the blend values $a_d$ (6) and\nday t's own channels (7).\n\nTokens are patch statistics;\n"
                  "their scores (4) and FiLM (7)\nreach full resolution only by\nbilinear upsampling.")

    # ---- inset: old per-pixel stem
    yO0, yO1 = 0.12, 2.75
    ax.add_patch(Rectangle((0.15, yO0), FW - 0.3, yO1 - yO0, fc="white", ec="#999999", lw=1.0, ls="--", zorder=0))
    txt(ax, 0.35, yO1 - 0.1, f"For comparison: previous per-pixel attention stem (chl_cnn/tattn.py, run {OLD})",
        ha="left", va="top", fontsize=11.5, weight="bold")
    ob = [(f"per-day stack\n(B, {n['c_day']}, {T}, H, W)\nall at full res", FC_IN, 2.2),
          (f"per-day conv embedding at full res:\nConv 3×3 {n['c_day']} → {n['old_dim']}, GroupNorm, GELU,\n"
           f"Conv 3×3; + day-offset emb.\n(B · {T}, {n['old_dim']}, H, W)", FC_FULL, 3.3),
          (f"{n['old_layers']} self-attention block, {n['old_heads']} heads,\ndim {n['old_dim']}, over {T} days per pixel\n"
           f"(B · {H} · {W}, {T}, {n['old_dim']})", FC_FULL, 3.0),
          (f"attention pooling per pixel:\nquery = learned + day t's token\nkeys / values = all {T} days,\n"
           f"day t included → (B, {n['old_dim']}, H, W)", FC_FULL, 3.1),
          (f"+ static ({n['n_static']}) → {n['old_in']} ch\n→ same dilated\nbackbone", FC_FULL, 2.0)]
    x = 0.35; yb0, hb = yO0 + 0.15, 1.75
    for k, (s, fc, w) in enumerate(ob):
        box(ax, x, yb0, w, hb, fc, None, body=s, fs=9.5, vcenter=True)
        if k:
            arrow(ax, [(x - 0.3, yb0 + hb / 2), (x, yb0 + hb / 2)])
        x += w + 0.3
    # cost table
    tx0 = x + 0.15
    rows = [("", f"arm A ({RUN})", f"old stem ({OLD})"),
            ("stem parameters", f"{n['stem']:,}", f"{n['old_stem']:,}"),
            ("backbone parameters", f"{n['bb']:,}", f"{n['old_bb']:,}"),
            ("total parameters", f"{n['tot']:,}", f"{n['old_tot']:,}"),
            ("batch", f"{n['batch']}", f"{n['old_batch']}"),
            ("s / epoch", f"{n['s_ep']:.0f}", f"{n['old_s_ep']:.0f}"),
            ("peak GPU memory", f"{n['gb']:.1f} GB", f"{n['old_gb']:.1f} GB")]
    colx = [tx0, tx0 + 1.85, tx0 + 3.55]
    for r, row in enumerate(rows):
        yy = yb0 + hb - 0.12 - r * 0.25 - (0.08 if r else 0)
        for cidx, v in enumerate(row):
            txt(ax, colx[cidx], yy, v, ha="left", va="top", fontsize=9.5, weight="bold" if r == 0 or cidx == 0 else "normal")
    ax.add_line(Line2D([tx0, FW - 0.3], [yb0 + hb - 0.36] * 2, color="#999999", lw=0.8))

    check_layout(fig)
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_pdf)
    plt.close(fig)


def check_layout(fig):
    """Print text pairs whose rendered boxes overlap, and texts that spill out of their box."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    ext = [(t, t.get_window_extent(r), bb) for t, bb in TEXTS]
    n_bad = 0
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            if ext[i][1].overlaps(ext[j][1]):
                n_bad += 1
                print(f"OVERLAP: {ext[i][0].get_text()[:40]!r} <-> {ext[j][0].get_text()[:40]!r}")
    for t, e, bb in ext:
        if bb is None:
            continue
        (x0, y0), (x1, y1) = ax.transData.transform([(bb[0], bb[1]), (bb[2], bb[3])])
        if e.x0 < x0 + 2 or e.x1 > x1 - 2 or e.y0 < y0 + 2 or e.y1 > y1 - 2:
            n_bad += 1
            print(f"OVERFLOW: {t.get_text()[:50]!r}")
    W, H = fig.canvas.get_width_height()
    for t, e, _ in ext:
        if e.x0 < 0 or e.y0 < 0 or e.x1 > W or e.y1 > H:
            n_bad += 1
            print(f"CLIPPED: {t.get_text()[:50]!r}")
    print(f"layout check: {n_bad} problem(s) in {len(ext)} texts")


def main():
    FIG.mkdir(exist_ok=True)
    n = load_numbers()
    print(json.dumps({k: v for k, v in n.items()}, default=str))
    figure(n, FIG / "ptok_schematic.png", FIG / "ptok_schematic.pdf")
    print("wrote", FIG / "ptok_schematic.png", FIG / "ptok_schematic.pdf")


if __name__ == "__main__":
    main()
