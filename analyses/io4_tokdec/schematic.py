"""Architecture schematic of the token-decoder gap-filler (io4_tokdec; Job 3 io4_tokdec_blend as a dashed skip). No GPU, no training.

Numbers come from run artefacts (the trained runs when present, else the io4 smoke runs):
  models/pinn/t3d/<run>/config.json, model.pt   window, patch, dims, batch, loss weights, parameter split, grid shapes
  analyses/io4_tokdec/results/io4_summary.json  s / epoch and peak GPU memory (else history.json / config.json of the run)
  models/pinn/t3d/io3_ptok4/                    the io3 patch-token run, for the cost table
Drawing helpers from analyses/io3_attn/schematic.py (box, arrow, txt, day_strip, check_layout). Architecture as chl_cnn/tokdec.py.

Usage (project root): python -m analyses.io4_tokdec.schematic
Writes analyses/io4_tokdec/figures/tokdec_schematic.{png,pdf}; prints any text overlap / overflow it finds.
"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

from analyses.io3_attn import schematic as SC
from analyses.io3_attn.schematic import (box, arrow, txt, day_strip, check_layout, BLUE, ORANGE, GREEN, VERM, PINK, SKY, BLACK, GREY_C,
                                         FC_IN, FC_PATCH, FC_FULL, LANE_PATCH, LANE_FULL)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_DIR = ROOT / "models" / "pinn" / "t3d"
FIG = HERE / "figures"
RUNS = [("io4_tokdec", "io4_smoke_tokdec"), ("io4_tokdec_blend", "io4_smoke_tokdec_blend")]
REF = "io3_ptok4"


def _params(sd):
    ok = lambda k, v: v.dtype.is_floating_point and "running" not in k
    tot = sum(v.numel() for k, v in sd.items() if ok(k, v))
    bb = sum(v.numel() for k, v in sd.items() if ok(k, v) and k.startswith("backbone"))
    return tot, bb, tot - bb


def _cost(run):
    c = json.loads((RUN_DIR / run / "config.json").read_text()); h = json.loads((RUN_DIR / run / "history.json").read_text())
    sd = torch.load(RUN_DIR / run / "model.pt", map_location="cpu")["model"]
    tot, bb, stem = _params(sd)
    assert tot == c["n_params"], f"{run}: parameter count mismatch with config.json"
    return dict(run=run, tot=tot, bb=bb, stem=stem, batch=c["batch"], s_ep=float(np.median(h["epoch_s"])), gb=c["peak_gpu_gb"], c=c, sd=sd)


def load_numbers():
    cost = []
    for full, smoke in RUNS:
        r = full if (RUN_DIR / full / "metrics.json").exists() else smoke
        cost.append(_cost(r))
    cost.append(_cost(REF))
    c, sd = cost[0]["c"], cost[0]["sd"]
    H, W = c["grid"]; P = c["ptok_patch"]
    Hp, Wp = -(-H // P) * P, -(-W // P) * P
    nPy, nPx = Hp // P, Wp // P
    assert sd["pos_emb"].shape[0] == nPy * nPx
    pw = sd["patch.weight"]                                     # (D, 12, 2P, 2P)
    return dict(H=H, W=W, P=P, Hp=Hp, Wp=Wp, nPy=nPy, nPx=nPx, T=c["t3d"]["n_days"], K=c["context"], c_day=c["t3d"]["c_day"],
                n_static=c["t3d"]["n_static"], in_ch=c["in_channels"], in_ch_b=cost[1]["c"]["in_channels"], n_in=pw.shape[1], kern=pw.shape[2],
                D=c["tokdec_dim"], C=c["tokdec_ch"], heads=c["tokdec_heads"], layers=c["tokdec_layers"], width=c["width"], dil=c["dilations"],
                vis_w=c["vis_weight"], grad_w=c["grad_weight"], nb_w=c["nb_weight"], dec_out=sd["dec.weight"].shape[0],
                cost=[{k: v for k, v in x.items() if k not in ("c", "sd")} for x in cost])


def figure(n, path_png, path_pdf):
    SC.TEXTS.clear()
    FW, FH = 21.0, 12.6
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
    H, W, T, ti, D, C, P = n["H"], n["W"], n["T"], n["K"], n["D"], n["C"], n["P"]
    c0 = n["cost"][0]
    txt(ax, FW / 2, FH - 0.15, "Token-decoder gap-filler (io4_tokdec; Job 3 io4_tokdec_blend adds the dashed skip)", ha="center", va="top",
        fontsize=17, weight="bold")
    txt(ax, FW / 2, FH - 0.58,
        f"chl_cnn/tokdec.py TokDecNet + chl_cnn/models.py DilatedCNN.  train_pinn.py --model temporal3d --t3d-stem tokdec --ptok-patch {P} "
        f"--tokdec-dim {D} --tokdec-ch {C} --tokdec-heads {n['heads']} --tokdec-layers {n['layers']} [--tokdec-blend] --context {n['K']} "
        f"--batch {c0['batch']}", ha="center", va="top", fontsize=10, color="#444444")
    txt(ax, FW / 2, FH - 0.88,
        f"B = batch, T = {T} days (t−{n['K']} … t+{n['K']}), H × W = {H} × {W} (0.25° grid), patch grid {n['nPy']} × {n['nPx']} (P = {P} px), "
        f"D = {D}, C = {C}.   io4_tokdec parameters {c0['tot']:,}: stem {c0['stem']:,}, backbone {c0['bb']:,}.", ha="center", va="top", fontsize=10.5)

    yP0, yP1 = 7.55, 11.45
    yF0, yF1 = 3.05, 7.35
    ax.add_patch(Rectangle((0.15, yP0), FW - 0.3, yP1 - yP0, fc=LANE_PATCH, ec="none", zorder=0))
    ax.add_patch(Rectangle((0.15, yF0), FW - 0.3, yF1 - yF0, fc=LANE_FULL, ec="none", zorder=0))
    txt(ax, 0.32, (yP0 + yP1) / 2, f"patch grid {n['nPy']} × {n['nPx']}", ha="center", va="center", rotation=90, fontsize=11, weight="bold", color=BLUE)
    txt(ax, 0.32, (yF0 + yF1) / 2, f"full resolution {H} × {W}", ha="center", va="center", rotation=90, fontsize=11, weight="bold", color=VERM)

    C1, C2, C3, C4 = (0.6, 2.3), (3.2, 2.9), (6.4, 2.6), (9.3, 2.75)
    C5, C6, C7 = (12.35, 2.55), (15.2, 1.95), (17.45, 3.4)
    bt, bh = yP0 + 0.18, yP1 - yP0 - 0.36

    # 1 inputs
    yI0, yI1 = 4.45, yF1 - 0.15
    b1 = box(ax, C1[0], yI0, C1[1], yI1 - yI0, FC_IN, "Inputs per day", num=1, fs=9.5,
             body=f"d = t−{n['K']} … t+{n['K']}, {n['n_in']} channels:\nCHL z-anomaly vs DOY\nclim. × visible mask\nvisible mask\n"
                  f"9 physics fields (z)\n× ocean mask\nocean mask\n→ (B, {n['n_in']}, {T}, H, W)\n\ncontext days carry\nthe donor's clouds")
    day_strip(ax, C1[0] + 0.1, C1[0] + C1[1] - 0.1, yI0 + 0.42, T, ti, bb=b1)
    bS = box(ax, C1[0], yF0 + 0.15, C1[1], 0.72, FC_IN, None, fs=9,
             body=f"static maps, day t (B, {n['n_static']}, H, W):\nclim z, lat, lon, log10 depth,\nDOY sin/cos, year")

    # legend
    lx, ly = C1[0], bt + bh
    txt(ax, lx, ly - 0.05, "Arrows", ha="left", va="top", fontsize=10.5, weight="bold")
    for k, (col, lab, ls) in enumerate([(VERM, "full-resolution pixels", "-"), (BLUE, "tokens / patch values", "-"),
                                        (GREY_C, "other tensors", "-"), (PINK, "Job 3 only (blend skip)", "--")]):
        yy = ly - 0.55 - k * 0.55
        if ls == "--":
            ax.add_line(Line2D([lx, lx + 0.55], [yy, yy], color=col, lw=2.2, ls="--"))
        else:
            arrow(ax, [(lx, yy), (lx + 0.55, yy)], color=col, lw=2.2)
        txt(ax, lx + 0.65, yy, lab, ha="left", va="center", fontsize=9.5)
    ax.add_patch(Rectangle((lx + 0.05, ly - 2.95), 0.3, 0.3, fc=ORANGE, ec=GREY_C, lw=0.7))
    txt(ax, lx + 0.5, ly - 2.8, "day t marker", ha="left", va="center", fontsize=9.5)

    # 2 patch embedding
    b2 = box(ax, C2[0], bt, C2[1], bh, FC_PATCH, "ViT patch embedding", num=2, fs=9.5, body_top=bt + bh - 1.55,
             body=f"shared across days; pad to {n['Hp']} × {n['Wp']}\nConv2d({n['n_in']}, {D}, kernel {n['kern']},\n"
                  f"stride {P}, padding {P // 2}): {n['kern']} × {n['kern']} windows\non a {P}-px stride → {n['nPy']} × {n['nPx']}\n"
                  f"LayerNorm + day-offset emb.\n+ patch-position emb.\nonly ocean patches kept:\n→ (B, Np, {T}, {D})")
    gx, gy, s = C2[0] + 0.25, bt + bh - 1.5, 0.08
    for i in range(12):
        for j in range(12):
            ax.add_patch(Rectangle((gx + j * s, gy + (11 - i) * s), s, s, fc="#E6E6E6", ec="white", lw=0.3, zorder=4))
    ax.add_patch(Rectangle((gx + 4 * s, gy + 4 * s), 4 * s, 4 * s, fc=SKY, ec="none", zorder=4))
    ax.add_patch(Rectangle((gx + 2 * s, gy + 2 * s), 8 * s, 8 * s, fill=False, ec=BLUE, lw=1.4, zorder=5))
    ax.add_patch(Rectangle((gx + 6 * s, gy + 2 * s), 8 * s - 2 * s, 8 * s, fill=False, ec=VERM, lw=1.0, ls="--", zorder=5))
    txt(ax, gx + 1.1, gy + 0.48, f"{P} × {P} patch (filled)\n{n['kern']} × {n['kern']} window (blue)\nneighbour window,\n{P}-px stride (dashed)",
        b2, ha="left", va="center", fontsize=8.5, linespacing=1.15)

    # 3 temporal self-attention
    b3 = box(ax, C3[0], bt, C3[1], bh, FC_PATCH, "Temporal self-attention", num=3, fs=9.5, body_top=bt + bh - 1.55,
             body=f"within each ocean patch,\nover its {T} day tokens;\nno attention across patches\n"
                  f"{n['layers']} pre-norm blocks, {n['heads']} heads,\nfused SDPA, batched over\npatches: (B · Np, {T}, {D})\n"
                  f"days missing from the store\nmasked out of the keys\n→ LayerNorm")
    sy = bt + bh - 1.15
    cx, s3 = day_strip(ax, C3[0] + 0.15, C3[0] + C3[1] - 0.15, sy, T, ti, kind="circle", label=False)
    for i in range(T):
        if i == ti:
            continue
        a, b_ = sorted([cx[i], cx[ti]]); r = (b_ - a) / 2; th = np.linspace(0, np.pi, 30)
        ax.plot((a + b_) / 2 + r * np.cos(th), sy + s3 / 2 + 0.55 * r * np.sin(th) / max(1, (cx[-1] - cx[0]) / 2), color=SKY, lw=0.8, zorder=3)
    for i, lab in [(0, "t−8"), (ti, "t"), (T - 1, "t+8")]:
        txt(ax, cx[i], sy - s3 / 2 - 0.04, lab, b3, ha="center", va="top", fontsize=9)

    # 4 pooling
    b4 = box(ax, C4[0], bt, C4[1], bh, FC_PATCH, "Attention pooling", num=4, fs=9.5, body_top=bt + bh - 0.5,
             body=f"query: day t's output token\nkeys / values: all {T} output\ntokens, day t included\n{n['heads']} heads → pooled (B, Np, {D})\n\n"
                  f"pooling weights recorded\n(by day offset, map\nautocorrelation)\n\nconcat [pooled, day-t token]\n→ (B, Np, {2 * D})")

    # 5 decoder (full res, fed from patch lane)
    yD0, yD1 = 4.45, yF1 - 0.15
    b5 = box(ax, C5[0], yD0, C5[1], yD1 - yD0, FC_FULL, "Decoder", num=5, fs=9.5, body_top=yD1 - 0.5,
             body=f"Linear {2 * D} → {n['dec_out']} per patch\n= {P} × {P} px × {C} ch\npixel shuffle → (B, {C},\n{n['Hp']}, {n['Wp']}), land patches 0\n"
                  f"crop to {H} × {W}\nConv 3×3 {C} → {C} + GELU\n(smooths patch seams)\n→ (B, {C}, H, W)")
    arrow(ax, [(C4[0] + C4[1], bt + 0.6), (C5[0] + C5[1] / 2, bt + 0.6), (C5[0] + C5[1] / 2, yD1)], color=BLUE, lw=2.2)

    # 6 backbone input
    yC0 = yF0 + 0.15
    b6 = box(ax, C6[0], yC0, C6[1], yD1 - yC0, FC_FULL, "Backbone\ninput", num=6, fs=9.5, body_top=yD1 - 0.85,
             body=f"concat at full res:\n\ndecoded\ncontext ({C})\n\nday t's own\nchannels ({n['c_day']})\n\nstatic ({n['n_static']})\n\n"
                  f"→ (B, {n['in_ch']}, H, W)\nJob 3: + 3 → {n['in_ch_b']}")
    ym = (yD0 + yD1) / 2 + 0.5
    arrow(ax, [(C5[0] + C5[1], ym), (C6[0], ym)])

    # 7 backbone + output
    yB0 = 5.25
    b7 = box(ax, C7[0], yB0, C7[1], yD1 - yB0, FC_FULL, "Dilated backbone", num=7, fs=9.5,
             body=f"Conv {n['in_ch']} → {n['width']}, 3×3 (no FiLM)\n{len(n['dil'])} residual blocks, width {n['width']}\n"
                  f"x + [BN, GELU, Conv 3×3, dilation d] × 2\nd = {','.join(map(str, n['dil']))}\nBN, GELU, Conv {n['width']} → 3, 1×1")
    arrow(ax, [(C6[0] + C6[1], ym), (C7[0], ym)])
    bO = box(ax, C7[0], yF0 + 0.15, C7[1], yB0 - yF0 - 0.45, FC_IN, None, fs=9,
             body=f"output (B, 3, H, W): z of t−1, t, t+1\n+ DOY clim z (residual) × sd + mean → ln CHL\n\n"
                  f"loss: L1 day-t hidden px + {n['vis_w']} × visible px\n+ {n['grad_w']:.0f} × gradient term + {n['nb_w']:.0f} × L1 t±1 donor-hidden")
    arrow(ax, [(C7[0] + C7[1] / 2, yB0), (C7[0] + C7[1] / 2, bO[3])])

    # cross-lane arrows
    xq = C1[0] + C1[1] - 0.2
    arrow(ax, [(xq, yI1), (xq, bt + 0.45), (C2[0], bt + 0.45)])
    txt(ax, xq - 0.08, bt + 0.2, "all 12\nchannels", ha="right", va="center", fontsize=9.5, linespacing=1.1)
    arrow(ax, [(C2[0] + C2[1], bt + bh / 2), (C3[0], bt + bh / 2)], color=BLUE, lw=2.0)
    arrow(ax, [(C3[0] + C3[1], bt + bh / 2), (C4[0], bt + bh / 2)], color=BLUE, lw=2.0)
    yb = yI0 - 0.35
    arrow(ax, [(C1[0] + C1[1] / 2, yI0), (C1[0] + C1[1] / 2, yb), (C6[0], yb)], color=VERM, lw=2.4)
    txt(ax, (C1[0] + C1[1] + C5[0]) / 2, yb + 0.06, f"day t's own channels (B, {n['c_day']}, H, W): physics, visible CHL z, visible mask",
        ha="center", va="bottom", fontsize=9.5, color=VERM, weight="bold")
    yc = bS[1] + 0.35
    arrow(ax, [(bS[2], yc), (C6[0], yc)])
    txt(ax, (C1[0] + C1[1] + C5[0]) / 2, yc + 0.06, f"static maps (B, {n['n_static']}, H, W)", ha="center", va="bottom", fontsize=9.5)

    # Job 3 skip (dashed)
    xJ0, xJ1, yJ0, yJ1 = 3.2, 11.95, 4.45, 6.45
    ax.add_patch(FancyBboxPatch((xJ0, yJ0), xJ1 - xJ0, yJ1 - yJ0, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=PINK, lw=1.6, ls="--", zorder=2))
    txt(ax, (xJ0 + xJ1) / 2, yJ1 - 0.12, "Job 3 only (io4_tokdec_blend): io3 arm-A blend from the same output tokens", (xJ0, yJ0, xJ1, yJ1),
        ha="center", va="top", fontsize=10.5, weight="bold", color=PINK)
    txt(ax, (xJ0 + xJ1) / 2, yJ1 - 0.5,
        f"patch score s = W$_q$ tok(t) · W$_k$ tok(d) / √{D}, bilinear between ocean-patch centres → (B, {T}, H, W)\n"
        f"+ per-pixel term (1×1-conv MLP on visible$_d$, |d−t|/{n['K']}, (d−t)/{n['K']}, |SST$_d$ − SST$_t$|)\n"
        f"softmax over context days d ≠ t visible at p + learned \"no day\" logit\n"
        f"blend = Σ$_d$ w$_d$(p) a$_d$(p) with a$_d$ = own visible CHL anomaly; mass = 1 − w$_{{none}}$; entropy → 3 extra channels",
        (xJ0, yJ0, xJ1, yJ1), ha="center", va="top", fontsize=9.5, linespacing=1.45)
    ax.add_line(Line2D([C4[0] + 0.4, C4[0] + 0.4], [bt, yJ1], color=PINK, lw=2.0, ls="--", zorder=3))
    ax.add_line(Line2D([C1[0] + C1[1], xJ0], [yJ0 + 0.35, yJ0 + 0.35], color=PINK, lw=2.0, ls="--", zorder=3))
    ax.add_line(Line2D([xJ1, xJ1 + 0.2, xJ1 + 0.2], [yJ0 + 0.35, yJ0 + 0.35, yJ0 - 0.15], color=PINK, lw=2.0, ls="--", zorder=3))
    ax.add_line(Line2D([xJ1 + 0.2, C6[0] - 0.15, C6[0] - 0.15], [yJ0 - 0.15, yJ0 - 0.15, ym - 0.6], color=PINK, lw=2.0, ls="--", zorder=3))
    arrow(ax, [(C6[0] - 0.15, ym - 0.6), (C6[0], ym - 0.6)], color=PINK, lw=2.0)

    # fine-detail note (patch lane)
    nx0 = C5[0]; nx1 = FW - 0.35
    box(ax, nx0, bt + 0.9, nx1 - nx0, bh - 1.2, "white", None, fs=10, ec=VERM, lw=1.6, vcenter=True,
        body=f"Context days (d ≠ t) reach the backbone only through the tokens:\n{n['kern']} × {n['kern']} px windows on a {P}-px stride, "
             f"decoded to {C} channels per {P} × {P} patch.\n\nDay t also enters at full resolution through its own channels (6).\n"
             f"No per-day blend, no FiLM.\n\nJob 3 adds each context day's own visible pixel values\nthrough the blend (dashed), "
             f"as in io3_ptok4.")

    # cost table
    yO0, yO1 = 0.12, 2.75
    ax.add_patch(Rectangle((0.15, yO0), FW - 0.3, yO1 - yO0, fc="white", ec="#999999", lw=1.0, ls="--", zorder=0))
    txt(ax, 0.35, yO1 - 0.1, "Cost (median s / epoch over the run's epochs; peak GPU memory allocated)", ha="left", va="top", fontsize=11.5, weight="bold")
    rows = [("", *[x["run"] for x in n["cost"]]),
            ("stem parameters", *[f"{x['stem']:,}" for x in n["cost"]]),
            ("backbone parameters", *[f"{x['bb']:,}" for x in n["cost"]]),
            ("total parameters", *[f"{x['tot']:,}" for x in n["cost"]]),
            ("batch", *[str(x["batch"]) for x in n["cost"]]),
            ("s / epoch", *[f"{x['s_ep']:.0f}" for x in n["cost"]]),
            ("peak GPU memory", *[f"{x['gb']:.1f} GB" for x in n["cost"]])]
    colx = [0.6, 4.0, 7.6, 11.2]
    for r, row in enumerate(rows):
        yy = yO1 - 0.55 - r * 0.28
        for cidx, v in enumerate(row):
            txt(ax, colx[cidx], yy, v, ha="left", va="top", fontsize=9.5, weight="bold" if r == 0 or cidx == 0 else "normal")
    ax.add_line(Line2D([0.5, 14.5], [yO1 - 0.82] * 2, color="#999999", lw=0.8))
    check_layout(fig)
    fig.savefig(path_png, dpi=220); fig.savefig(path_pdf); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    n = load_numbers()
    print(json.dumps(n, default=str))
    figure(n, FIG / "tokdec_schematic.png", FIG / "tokdec_schematic.pdf")
    print("wrote", FIG / "tokdec_schematic.png")


if __name__ == "__main__":
    main()
