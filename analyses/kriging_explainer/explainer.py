"""Explanatory figure: classic space-time kriging vs "learned kriging" (local space-time observation attention).

No data, no GPU, no training. Everything is a synthetic toy (fixed seed):
  - a smooth random ln-CHL anomaly field with a front and an eddy, advected along the front by a uniform current;
  - an SST field with its front on the same line (warm, low-CHL side offshore);
  - per-day cloud masks (smooth noise thresholded), the target pixel forced under cloud on day t.
Classic weights: ordinary kriging with an isotropic exponential covariance in space-time distance (1 day = 2 px).
"Learned" weights: a hand-written toy rule standing in for softmax(q.k): small |SST difference| and small
*advected* offset (anisotropic, longer along the front) give large weights.

Usage (project root): python analyses/kriging_explainer/explainer.py
Writes analyses/kriging_explainer/figures/learned_kriging.{png,pdf}; prints text overlaps / overflows it finds.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from chl_cnn.plotting import OKABE_ITO, MAP_CMAP                    # noqa: E402

BLUE, ORANGE, GREEN, VERM, PINK, SKY, YELLOW, BLACK = OKABE_ITO
OBS_C = ORANGE          # colour of observed values / keys everywhere in the figure
TARGET_C = PINK
FLOW_C = BLUE
GREY_C = "#333333"
CLOUD_C = "#C8C8C8"
SEED = 7

# ---------------------------------------------------------------- toy setting
N = 56                      # grid size (px)
DAYS = [-4, -2, 0]          # days shown (relative to t); the model's window is +-8 d
PX_PER_DAY = 2.0            # space-time metric: 1 day = 2 px
R = 14.0                    # neighbourhood radius in space-time distance (px)
K = 48                      # keys kept (k nearest visible observations)
THETA = np.deg2rad(18)      # front / current direction
U_EFF = 1.2                 # s * u-bar, px / day, along the front
P = np.array([27.0, 22.0])  # target pixel (x, y)
LATTICE = 3                 # display / candidate subsample spacing (px)


def rot(x, y):
    """(along-front, across-front) coordinates."""
    return x * np.cos(THETA) + y * np.sin(THETA), -x * np.sin(THETA) + y * np.cos(THETA)


def front_across(along):
    return 4.0 + 3.0 * np.sin(2 * np.pi * along / 45.0)


# ---------------------------------------------------------------- data generation
def make_fields(rng):
    """Return per-day dicts with chl (ln anomaly), sst, cloud mask, plus helpers."""
    big = 3 * N
    noise_c = gaussian_filter(rng.standard_normal((big, big)), 5) * 12
    noise_s = gaussian_filter(rng.standard_normal((big, big)), 6) * 6
    yy, xx = np.mgrid[0:N, 0:N].astype(float)

    def sample(arr, x, y):
        i = np.clip(np.round(y + N).astype(int), 0, big - 1)
        j = np.clip(np.round(x + N).astype(int), 0, big - 1)
        return arr[i, j]

    out = {}
    for d in DAYS:
        # Lagrangian coordinates: the field is translated along the front by U_EFF px / day
        x0 = xx - U_EFF * d * np.cos(THETA)
        y0 = yy - U_EFF * d * np.sin(THETA)
        a, c = rot(x0 - N / 2, y0 - N / 2)
        f = c - front_across(a)                                        # signed distance-ish to the front
        ex, ey = 14.0, 40.0                                             # eddy centre (Lagrangian)
        r2 = (x0 - ex) ** 2 + (y0 - ey) ** 2
        eddy = -0.9 * np.exp(-r2 / 30.0) + 0.35 * np.exp(-(np.sqrt(r2) - 8.0) ** 2 / 6.0)
        chl = 0.9 * np.tanh(-f / 2.5) + 0.35 * sample(noise_c, x0, y0) + eddy + 0.6
        sst = 18.0 + 1.4 * np.tanh(f / 2.5) + 0.25 * sample(noise_s, x0, y0) + 0.6 * np.exp(-r2 / 30.0)
        out[d] = {"chl": chl, "sst": sst}

    # clouds: independent per day, ~45 % cover; day t: target under a cloud
    for d in DAYS:
        cn = gaussian_filter(rng.standard_normal((N, N)), 4.5)
        cn = (cn - cn.mean()) / cn.std()
        if d == 0:
            cn += 3.0 * np.exp(-((xx - P[0] - 1) ** 2 + (yy - P[1] + 1) ** 2) / 70.0)
        if d == -2:
            cn += 0.6 * np.exp(-((xx - P[0] + 3) ** 2 + (yy - P[1] - 2) ** 2) / 30.0)
        out[d]["cloud"] = cn > np.quantile(cn, 0.55)
    return out


def candidate_obs(fields, rng):
    """Visible observations on a jittered lattice (a display subsample of the visible pixels)."""
    rows = []
    for d in DAYS:
        g = np.arange(1, N - 1, LATTICE, dtype=float)
        gx, gy = np.meshgrid(g, g)
        gx = gx + rng.uniform(-0.8, 0.8, gx.shape)
        gy = gy + rng.uniform(-0.8, 0.8, gy.shape)
        ix, iy = np.round(gx).astype(int), np.round(gy).astype(int)
        vis = ~fields[d]["cloud"][iy, ix]
        for x, y, i, j in zip(gx[vis], gy[vis], ix[vis], iy[vis]):
            rows.append((x, y, d, fields[d]["chl"][j, i], fields[d]["sst"][j, i]))
    return np.array(rows)     # columns: x, y, day, chl, sst


# ---------------------------------------------------------------- analysis (toy weights)
def st_dist(obs):
    return np.sqrt((obs[:, 0] - P[0]) ** 2 + (obs[:, 1] - P[1]) ** 2 + (PX_PER_DAY * obs[:, 2]) ** 2)


def select_keys(obs):
    dist = st_dist(obs)
    inside = np.where(dist <= R)[0]
    keep = inside[np.argsort(dist[inside])[:K]]
    return keep


def kriging_weights(xyz, L=8.0, nugget=0.05):
    """Ordinary kriging, isotropic exponential covariance in space-time distance."""
    n = len(xyz)
    D = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    C = np.exp(-D / L) + nugget * np.eye(n)
    p = np.array([P[0], P[1], 0.0])
    c0 = np.exp(-np.sqrt(((xyz - p) ** 2).sum(-1)) / L)
    A = np.zeros((n + 1, n + 1)); A[:n, :n] = C; A[:n, n] = 1; A[n, :n] = 1
    b = np.append(c0, 1.0)
    sol = np.linalg.solve(A, b)
    w = sol[:n]
    var = 1.0 + nugget - w @ c0 - sol[n]
    return w, var


def learned_weights(keys, sst_p):
    """Toy stand-in for softmax(q.k): anisotropic in the advected offset, plus same-water-mass term."""
    dx = keys[:, 0] - P[0] - U_EFF * keys[:, 2] * np.cos(THETA)     # advected offset dx - s*u*dt
    dy = keys[:, 1] - P[1] - U_EFF * keys[:, 2] * np.sin(THETA)
    a, c = rot(dx, dy)
    dsst = np.abs(keys[:, 4] - sst_p)
    logit = -(a / 9.0) ** 2 - (c / 3.5) ** 2 - dsst / 0.35 - 0.05 * np.abs(keys[:, 2])
    w = np.exp(logit - logit.max())
    return w / w.sum()


# ---------------------------------------------------------------- reporting helpers
TEXTS = []


def txt(ax, x, y, s, bb=None, **kw):
    kw.setdefault("fontsize", 10)
    t = ax.text(x, y, s, **kw)
    TEXTS.append((t, bb, ax))
    return t


def box(ax, x, y, w, h, header, body="", fc="#F2F2F2", ec=GREY_C, lw=1.0, ls="-", fs=9.5, hfs=10.5, hc="black",
        bc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=2, transform=ax.transAxes, mutation_aspect=1))
    bb = (x, y, x + w, y + h)
    if body:
        txt(ax, x + w / 2, y + h - 0.035, header, bb, ha="center", va="top", fontsize=hfs, weight="bold", color=hc,
            transform=ax.transAxes, zorder=4)
        txt(ax, x + w / 2, y + h - 0.035 - 0.085, body, bb, ha="center", va="top", fontsize=fs, linespacing=1.3,
            transform=ax.transAxes, zorder=4, color=bc, weight="bold" if bc != "black" else "normal")
    else:
        txt(ax, x + w / 2, y + h / 2, header, bb, ha="center", va="center", fontsize=hfs, weight="bold", color=hc,
            transform=ax.transAxes, zorder=4, linespacing=1.25)
    return bb


def arrow(ax, p0, p1, color=GREY_C, lw=1.5, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, ls=ls,
                                 transform=ax.transAxes, zorder=3, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


DAY_MARK = {0: "o", -2: "s", -4: "^"}
DAY_LAB = {0: "day t", -2: "day t−2", -4: "day t−4"}


def map_frame(ax, title):
    ax.set_xlim(0, N - 1); ax.set_ylim(0, N - 1); ax.set_aspect("equal")
    ax.set_xticks([0, 20, 40]); ax.set_yticks([0, 20, 40])
    ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=11)


def draw_field(ax, f, alpha=1.0, vmin=-1.2, vmax=2.4):
    cm = plt.get_cmap(MAP_CMAP).copy(); cm.set_bad(CLOUD_C)
    im = ax.imshow(np.ma.masked_where(f["cloud"], f["chl"]), origin="lower", cmap=cm, vmin=vmin, vmax=vmax,
                   alpha=alpha, interpolation="nearest")
    return im


def panel_a(axs, fields, obs, keys_idx):
    keyset = set(keys_idx.tolist())
    for ax, d in zip(axs, DAYS):
        im = draw_field(ax, fields[d])
        m = obs[:, 2] == d
        ax.scatter(obs[m, 0], obs[m, 1], s=7, c="white", edgecolors="black", linewidths=0.4, zorder=3)
        sel = np.array([i for i in np.where(m)[0] if i in keyset], dtype=int)
        ax.scatter(obs[sel, 0], obs[sel, 1], s=34, c=OBS_C, edgecolors="black", linewidths=0.8, zorder=4,
                   marker=DAY_MARK[d])
        rr = np.sqrt(max(R ** 2 - (PX_PER_DAY * d) ** 2, 0))
        ax.add_patch(Circle(P, rr, fill=False, ls="--", lw=1.4, ec="white", zorder=5))
        if d == 0:
            ax.scatter(*P, marker="*", s=260, c=TARGET_C, edgecolors="black", linewidths=0.9, zorder=6)
        else:
            ax.scatter(*P, marker="+", s=70, c="white", linewidths=1.6, zorder=6)
        map_frame(ax, f"{DAY_LAB[d]}  (r = {rr:.1f} px)")
        ax.set_xlabel("x (px)", fontsize=9.5)
        txt(ax, 1.5, 1.5, f"{int(m.sum())} visible, {len(sel)} keys", fontsize=9, color="white",
            bbox=dict(fc="black", alpha=0.55, ec="none", pad=1.5), zorder=7)
    axs[0].set_ylabel("y (px)", fontsize=9.5)
    return im


def panel_weights(ax, fields, keys, w, title, note, wmax, flow=False, neg=None):
    draw_field(ax, fields[0], alpha=0.35)
    if flow:
        sst = fields[0]["sst"]
        ax.contour(sst, levels=[18.0], colors="black", linewidths=1.6, linestyles="-", zorder=3)
        g = np.arange(5, N, 10, dtype=float)
        gx, gy = np.meshgrid(g, g)
        ax.quiver(gx, gy, np.full_like(gx, np.cos(THETA)), np.full_like(gy, np.sin(THETA)), color=FLOW_C,
                  scale=18, width=0.007, zorder=3)
        for d in (-2, -4):
            up = P + U_EFF * d * np.array([np.cos(THETA), np.sin(THETA)])
            ax.scatter(*up, marker="*", s=150, facecolors="white", edgecolors="black", linewidths=1.2, zorder=6)
        txt(ax, N - 2, 6.5, "white stars: p advected\nback to t−2, t−4", fontsize=9, ha="right", va="bottom",
            color="black", bbox=dict(fc="white", alpha=0.85, ec="none", pad=1.0), zorder=7)
    for d in DAYS:
        m = keys[:, 2] == d
        pos = m & (w > 0)
        ax.scatter(keys[pos, 0], keys[pos, 1], s=8 + 520 * w[pos] / wmax, c=OBS_C, edgecolors="black",
                   linewidths=0.7, marker=DAY_MARK[d], zorder=4, alpha=0.95)
        ng = m & (w <= 0)
        if ng.any():
            ax.scatter(keys[ng, 0], keys[ng, 1], s=8 + 520 * np.abs(w[ng]) / wmax, facecolors="none",
                       edgecolors=GREY_C, linewidths=0.9, marker=DAY_MARK[d], zorder=4)
    ax.scatter(*P, marker="*", s=260, c=TARGET_C, edgecolors="black", linewidths=0.9, zorder=6)
    map_frame(ax, title)
    ax.set_xlabel("x (px)", fontsize=9.5)
    neff = 1.0 / np.sum(w ** 2)
    txt(ax, 1.5, N - 2.5, note, fontsize=10, va="top", weight="bold",
        bbox=dict(fc="white", alpha=0.9, ec=GREY_C, pad=2.0), zorder=7)
    txt(ax, 1.5, 1.5, f"n_eff = 1/Σw² = {neff:.1f} of {len(w)}", fontsize=9, color="white",
        bbox=dict(fc="black", alpha=0.55, ec="none", pad=1.5), zorder=7)


def panel_d(ax):
    ax.set_axis_off()
    ax.set_title("D. Learned kriging: architecture for one target pixel (p, t)", fontsize=12, loc="left")
    FC_T, FC_O, FC_ATT, FC_FULL = "#F7E6EF", "#FDF0DC", "#E3EFF8", "#EDEDED"
    # inputs
    box(ax, 0.0, 0.60, 0.275, 0.36, "Target pixel (p, t), under cloud",
        "physics at p: SST, salinity,\ncurrents, MLD, …\nstatic fields\nsummary of day t's visible\nneighbourhood", fc=FC_T)
    box(ax, 0.0, 0.0, 0.275, 0.54, "k ≈ 32–64 visible obs (q, d)",
        "|d − t| ≤ 8 d, space-time dist. ≤ R\n(1 day ≈ 2 px), after cloud mask\n"
        "key features:\n• Δx, Δy, Δt\n• Δx − s·ū·Δt  (advected, s ≈ 0.25)\n• physics at (q, d)\n• |SST(q,d) − SST(p,t)|",
        fc=FC_O, ec=OBS_C, lw=1.6)
    # q, k, v
    box(ax, 0.33, 0.70, 0.12, 0.16, "query q", fc=FC_T)
    box(ax, 0.33, 0.26, 0.12, 0.16, "keys k_i", fc=FC_O)
    box(ax, 0.33, 0.02, 0.12, 0.16, "values v_i\n= observed\nCHL anomaly", fc=FC_O, ec=OBS_C, lw=2.0, hfs=9.5)
    box(ax, 0.30, 0.47, 0.18, 0.14, "optional: keys attend\nto each other\n(≈ declustering)", fc="white", ls="--",
        hfs=9, ec=GREY_C)
    arrow(ax, (0.275, 0.78), (0.33, 0.78))
    arrow(ax, (0.275, 0.34), (0.33, 0.34))
    arrow(ax, (0.275, 0.10), (0.33, 0.10), color=OBS_C, lw=2.2)
    arrow(ax, (0.39, 0.42), (0.39, 0.47), ls="--")
    # weights and sum
    box(ax, 0.52, 0.52, 0.17, 0.22, "w_i = softmax_i\n(q·k_i / √D)", fc=FC_ATT)
    box(ax, 0.50, 0.04, 0.21, 0.26, "Σ_i w_i v_i", "values pass\nthrough exactly", fc=FC_ATT, ec=OBS_C, lw=2.0,
        bc=OBS_C, fs=10)
    arrow(ax, (0.45, 0.78), (0.52, 0.68))
    arrow(ax, (0.45, 0.34), (0.52, 0.58))
    arrow(ax, (0.605, 0.52), (0.605, 0.28))
    arrow(ax, (0.45, 0.10), (0.50, 0.12), color=OBS_C, lw=2.2)
    # into the backbone (vertical chain on the right)
    xb, wb = 0.76, 0.24
    box(ax, xb, 0.79, wb, 0.17, "per-pixel embedding\n(+ estimate, spread)", fc=FC_ATT)
    box(ax, xb, 0.545, wb, 0.17, "concat with day t's\nfull-resolution inputs", fc=FC_FULL)
    box(ax, xb, 0.30, wb, 0.17, "dilated CNN backbone", fc=FC_FULL)
    box(ax, xb, 0.055, wb, 0.17, "ln CHL (p, t)", fc="white")
    arrow(ax, (0.71, 0.17), (0.73, 0.17))
    ax.add_line(Line2D([0.73, 0.73, xb], [0.17, 0.875, 0.875], color=GREY_C, lw=1.5, transform=ax.transAxes,
                       zorder=3))
    arrow(ax, (0.73, 0.875), (xb, 0.875))
    for y0, y1 in [(0.79, 0.715), (0.545, 0.47), (0.30, 0.225)]:
        arrow(ax, (xb + wb / 2, y0), (xb + wb / 2, y1))


def panel_e(ax):
    ax.set_axis_off()
    ax.set_title("E. Context sources by model", fontsize=12, loc="left")
    cols = ["same-\nday\nneighb.", "own px,\nother\ndays", "nearby\npx, other\ndays", "motion-\naware", "exact\nvalues"]
    rows = [
        ("arm A\n(own value)", ["CNN", "yes", "4×4\nsummary", "no", "own px"]),
        ("arm B\n(Gauss. avg)", ["CNN", "blurred", "blurred\nσ 1.5 px", "no", "no"]),
        ("per-pixel\nattention", ["CNN", "embedded", "3×3 conv\nembedded", "no", "no"]),
        ("tokens\n(4×4 patches)", ["CNN", "4×4\nsummary", "4×4\nsummary", "no", "no"]),
        ("learned\nkriging", ["keys\nΔt = 0", "yes", "yes, per\nobs.", "advected\noffset", "yes"]),
    ]
    level = {"yes": 2, "own px": 1, "CNN": 2, "keys\nΔt = 0": 2, "advected\noffset": 2, "yes, per\nobs.": 2, "no": 0}
    fc = {0: "#E6E6E6", 1: "#F8E3B8", 2: "#BFE3D6"}
    x0, wn, wc = 0.0, 0.25, 0.15
    yh, hh, hr = 1.0, 0.17, 0.15
    for j, c in enumerate(cols):
        txt(ax, x0 + wn + (j + 0.5) * wc, yh - 0.02, c, fontsize=9, ha="center", va="top", transform=ax.transAxes,
            weight="bold", linespacing=1.1)
    for i, (name, vals) in enumerate(rows):
        y = yh - hh - (i + 1) * hr
        bold = name.startswith("learned")
        txt(ax, x0 + 0.005, y + hr / 2, name, fontsize=9.5, ha="left", va="center", transform=ax.transAxes,
            weight="bold" if bold else "normal", linespacing=1.1)
        for j, v in enumerate(vals):
            lv = level.get(v, 1)
            ax.add_patch(Rectangle((x0 + wn + j * wc + 0.004, y + 0.006), wc - 0.008, hr - 0.012, fc=fc[lv], ec="none",
                                   transform=ax.transAxes, zorder=1))
            txt(ax, x0 + wn + (j + 0.5) * wc, y + hr / 2, v, fontsize=9, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.05, zorder=2)
    txt(ax, 0.0, 0.0, "CNN: day t's own pixels via the backbone (all models).\n"
        "Cells: green = full, orange = partial, grey = none.", fontsize=9, transform=ax.transAxes, va="bottom")


def check_layout(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ext = [(t, t.get_window_extent(r), bb, ax) for t, bb, ax in TEXTS]
    bad = 0
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            if ext[i][3] is ext[j][3] and ext[i][1].overlaps(ext[j][1]):
                bad += 1
                print(f"OVERLAP: {ext[i][0].get_text()[:30]!r} <-> {ext[j][0].get_text()[:30]!r}")
    for t, e, bb, ax in ext:
        if bb is None:
            continue
        (bx0, by0), (bx1, by1) = ax.transAxes.transform([(bb[0], bb[1]), (bb[2], bb[3])])
        if e.x0 < bx0 + 2 or e.x1 > bx1 - 2 or e.y0 < by0 + 2 or e.y1 > by1 - 2:
            bad += 1
            print(f"OVERFLOW: {t.get_text()[:40]!r}")
    W, H = fig.canvas.get_width_height()
    for t, e, _, _ in ext:
        if e.x0 < 0 or e.y0 < 0 or e.x1 > W or e.y1 > H:
            bad += 1
            print(f"CLIPPED: {t.get_text()[:40]!r}")
    print(f"layout check: {bad} problem(s) in {len(ext)} texts")


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    fields = make_fields(rng)
    obs = candidate_obs(fields, rng)
    kidx = select_keys(obs)
    keys = obs[kidx]
    sst_p = fields[0]["sst"][int(P[1]), int(P[0])]
    xyz = np.c_[keys[:, 0], keys[:, 1], PX_PER_DAY * keys[:, 2]]
    w_k, var_k = kriging_weights(xyz)
    w_l = learned_weights(keys, sst_p)
    wmax = max(w_k.max(), w_l.max())
    print(f"keys: {len(keys)} (per day: " + ", ".join(f"{d}: {(keys[:, 2] == d).sum()}" for d in DAYS) + ")")
    print(f"kriging: sum w = {w_k.sum():.3f}, negative = {(w_k < 0).sum()}, max = {w_k.max():.3f}, var = {var_k:.3f}")
    print(f"learned: max = {w_l.max():.3f}, n_eff = {1 / np.sum(w_l ** 2):.1f}")
    est_k, est_l = w_k @ keys[:, 3], w_l @ keys[:, 3]
    print(f"toy estimates: truth {fields[0]['chl'][int(P[1]), int(P[0])]:.2f}, kriging {est_k:.2f}, learned {est_l:.2f}")

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.92, 1.0], width_ratios=[1.0, 0.0001], hspace=0.28,
                  left=0.04, right=0.985, top=0.92, bottom=0.03)
    top = gs[0, :].subgridspec(1, 6, width_ratios=[1, 1, 1, 0.06, 1.12, 1.12], wspace=0.22)
    axa = [fig.add_subplot(top[0, i]) for i in range(3)]
    axb = fig.add_subplot(top[0, 4]); axc = fig.add_subplot(top[0, 5])
    bot = gs[1, :].subgridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.06)
    axd = fig.add_subplot(bot[0, 0]); axe = fig.add_subplot(bot[0, 1])

    im = panel_a(axa, fields, obs, kidx)
    fig.text(0.04, 0.925, "A. Setting: visible observations around a cloudy target pixel, three of the ±8 days",
             fontsize=12, ha="left")
    panel_weights(axb, fields, keys, w_k, "B. Classic kriging weights", "depends on distance only", wmax)
    panel_weights(axc, fields, keys, w_l, "C. Learned kriging weights (toy rule)",
                  "depends on content,\nposition and flow", wmax, flow=True)
    cax = fig.add_axes([0.045, 0.505, 0.15, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("ln CHL anomaly (synthetic)", fontsize=9); cb.ax.tick_params(labelsize=9)

    # shared legend for A-C
    h = [Line2D([], [], ls="", marker="*", ms=14, mfc=TARGET_C, mec="black", label="target pixel p (day t)"),
         Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="black", label="visible obs. (display subsample)"),
         Line2D([], [], ls="", marker="o", ms=8, mfc=OBS_C, mec="black", label="selected key, day t"),
         Line2D([], [], ls="", marker="s", ms=8, mfc=OBS_C, mec="black", label="key, day t−2"),
         Line2D([], [], ls="", marker="^", ms=8, mfc=OBS_C, mec="black", label="key, day t−4"),
         Line2D([], [], ls="--", color="black", label="radius R in space-time (1 d = 2 px)"),
         Rectangle((0, 0), 1, 1, fc=CLOUD_C, label="cloud"),
         Line2D([], [], color="black", lw=1.6, label="SST front (C)"),
         Line2D([], [], ls="", marker=r"$\rightarrow$", ms=14, color=FLOW_C, label="current ū (C)"),
         Line2D([], [], ls="", marker="o", ms=10, mfc="none", mec=GREY_C, label="marker size = weight; hollow = w < 0")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.6, 0.462), ncol=5, fontsize=9, frameon=False,
               handletextpad=0.4, columnspacing=1.2)

    panel_d(axd)
    panel_e(axe)

    check_layout(fig)
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / "learned_kriging.png", dpi=200)
    fig.savefig(out / "learned_kriging.pdf")
    print("wrote", out / "learned_kriging.png")


if __name__ == "__main__":
    main()
