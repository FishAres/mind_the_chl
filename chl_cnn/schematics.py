"""Schematic figures for the PINN notebook (drawn with matplotlib, Okabe-Ito colours).

python -m chl_cnn.schematics  -> notebooks/figures/pinn_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import numpy as np

from .plotting import OKABE_ITO as OI

FIG = Path(__file__).resolve().parents[1] / "notebooks" / "figures"
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW, BLACK = OI[1], OI[0], OI[3], OI[5], OI[6], OI[2], OI[4], OI[7]


def arrow(ax, p, q, color=BLACK, lw=2, style="-|>", ms=14, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, **kw))


def coastal_upwelling(path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # left: coastal upwelling, vertical section, Northern Hemisphere, coast on the right
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(-4, 2.2); ax.axis("off")
    ax.add_patch(Rectangle((0, -4), 8, 4, color=SKY, alpha=0.35))
    ax.add_patch(Rectangle((8, -4), 2, 6, color="#c8b28a"))
    ax.plot([0, 8], [0, 0], color=BLUE, lw=2)
    ax.text(9, 1.2, "coast\n(Somalia /\nOman)", ha="center", va="center", fontsize=10)
    ax.text(4, 1.7, "along-shore wind (SW monsoon jet)\ncoast on the right of the wind (NH)", ha="center", fontsize=10)
    ax.scatter([4], [1.0], s=260, facecolors="none", edgecolors=BLACK, lw=2); ax.scatter([4], [1.0], s=25, color=BLACK)
    ax.text(4.55, 1.0, "wind out of page", va="center", fontsize=9)
    arrow(ax, (6.5, -0.4), (2.5, -0.4), color=ORANGE, lw=3); ax.text(4.5, -0.9, "Ekman transport  $M_E=\\tau/(\\rho f)$  offshore", ha="center", color=ORANGE, fontsize=10)
    arrow(ax, (7.3, -3.2), (7.3, -0.6), color=GREEN, lw=3); ax.text(7.05, -2.0, "upwelling\n$w$", ha="right", color=GREEN, fontsize=10)
    arrow(ax, (1.0, -0.8), (1.0, -3.0), color=BLUE, lw=2); ax.text(1.2, -2.0, "compensating\ndownward flow", fontsize=8, color=BLUE)
    ax.text(5.5, -3.5, "cold, nutrient-rich water  ->  CHL bloom lags by days-weeks", fontsize=9, ha="center")
    ax.set_title("Coastal (Ekman) upwelling, vertical section")
    # right: Ekman pumping from wind stress curl, plan view
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 10, 10, color=SKY, alpha=0.25))
    th = np.linspace(0, 2 * np.pi, 9)[:-1]
    for t in th:
        x, y = 5 + 3.2 * np.cos(t), 5 + 3.2 * np.sin(t)
        dx, dy = -np.sin(t), np.cos(t)
        arrow(ax, (x - 0.6 * dx, y - 0.6 * dy), (x + 0.6 * dx, y + 0.6 * dy), color=BLACK, lw=1.8, ms=10)
    ax.text(5, 9.3, "cyclonic wind stress (curl > 0, NH)", ha="center", fontsize=10)
    for t in th:
        x, y = 5 + 2.2 * np.cos(t), 5 + 2.2 * np.sin(t)
        arrow(ax, (x, y), (x + 0.9 * np.cos(t), y + 0.9 * np.sin(t)), color=ORANGE, lw=1.8, ms=10)
    ax.text(5, 0.6, "Ekman transport 90$^\\circ$ right of wind -> divergence", ha="center", color=ORANGE, fontsize=10)
    ax.scatter([5], [5], s=380, facecolors="none", edgecolors=GREEN, lw=2.5); ax.scatter([5], [5], s=30, color=GREEN)
    ax.text(5, 4.0, "$w_E=\\dfrac{1}{\\rho}\\,\\nabla\\times\\left(\\dfrac{\\boldsymbol{\\tau}}{f}\\right) > 0$\nupwelling", ha="center", va="top", color=GREEN, fontsize=11)
    ax.set_title("Ekman pumping from wind-stress curl, plan view")
    fig.suptitle("How wind drives upwelling (Northern Hemisphere)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def tracer_stencil(path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal"); ax.axis("off")
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, ec="0.6", lw=0.8))
    for (i, j), c in [((2, 2), VERM), ((1, 2), BLUE), ((3, 2), BLUE), ((2, 1), BLUE), ((2, 3), BLUE)]:
        ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, color=c, alpha=0.45))
    ax.text(2, 2, "$C_{i,j}$", ha="center", va="center", fontsize=12)
    ax.text(1, 2, "$C_{i-1,j}$", ha="center", va="center", fontsize=9); ax.text(3, 2, "$C_{i+1,j}$", ha="center", va="center", fontsize=9)
    ax.text(2, 1, "$C_{i,j-1}$", ha="center", va="center", fontsize=9); ax.text(2, 3, "$C_{i,j+1}$", ha="center", va="center", fontsize=9)
    arrow(ax, (0.2, 3.9), (1.6, 3.9), color=ORANGE, lw=2.5); ax.text(0.9, 4.15, "$u$", color=ORANGE, ha="center", fontsize=12)
    arrow(ax, (4.0, 0.2), (4.0, 1.6), color=ORANGE, lw=2.5); ax.text(4.3, 0.9, "$v$", color=ORANGE, va="center", fontsize=12)
    ax.text(2, -0.4, "$\\Delta x = 0.25^\\circ \\approx 27$ km  (lon), $\\Delta t$ = 1 day", ha="center", fontsize=9, va="top")
    ax.set_title("Finite-difference stencil on the 0.25$^\\circ$ grid\n(advection: upwind/central; diffusion: 5-point Laplacian)")
    ax = axes[1]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.88, "Surface tracer budget for ln-CHL (or CHL) $C(x,y,t)$", ha="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.72, r"$\partial_t C \;+\; u\,\partial_x C + v\,\partial_y C \;=\; K_h\nabla^2 C \;+\; S(w_E,\,\mathrm{MLD},\,\mathrm{SST},\dots)$", ha="center", fontsize=14)
    for x, lab, col in [(0.12, "tendency", BLACK), (0.36, "advection", ORANGE), (0.6, "diffusion", BLUE), (0.85, "source / sink", GREEN)]:
        ax.text(x, 0.62, lab, ha="center", color=col, fontsize=10, weight="bold")
    rows = [("tendency", "from CHL(t-1), CHL(t+1) where observed; the network's own output otherwise", BLACK),
            ("advection", "u_curr, v_curr (total surface currents, 0.25 deg); typical |u| 0.2-1 m/s -> Courant ~0.6-3 per day", ORANGE),
            ("diffusion", "K_h ~ 1e2-1e3 m2/s: diffusion length in 1 day ~ 3-9 km  <<  27 km grid (weak, regularising)", BLUE),
            ("source/sink", "growth from upwelled nutrients (w_E, coastal index), light/SST; loss by grazing, mixing (MLD); learned or parameterised", GREEN)]
    for k, (name, txt, col) in enumerate(rows):
        y = 0.5 - 0.11 * k
        ax.text(0.02, y, name, color=col, fontsize=10, weight="bold", va="center")
        ax.text(0.2, y, txt, fontsize=9, va="center", wrap=True)
    ax.set_title("Terms and what the data can supply")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def pinn_architecture(path):
    fig, ax = plt.subplots(figsize=(13, 5.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2)

    def box(x, y, w, h, text, color, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec=BLACK, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    box(0.2, 3.4, 2.6, 1.4, "inputs at day t (and t±k)\nCHL L3 with clouds + mask\nSST, salinity, MLD, SLA\nwinds, currents, clim, coords", "#e6e6e6")
    box(0.2, 1.2, 2.6, 1.4, "wind stress $\\tau$ -> Ekman\n$M_E$, $w_E$ (precomputed,\n|lat| > ~5°)\ncurrents u, v", "#e6e6e6")
    box(3.6, 2.3, 2.4, 1.6, "CNN / U-Net\n$C_\\theta(x,y,t)$\n(gap-free field for\nt-1, t, t+1)", SKY, fs=10)
    arrow(ax, (2.8, 4.1), (3.6, 3.3)); arrow(ax, (2.8, 1.9), (3.6, 2.7))
    box(6.9, 4.0, 2.6, 1.0, "data loss\nfake-cloud hidden px only", ORANGE)
    box(6.9, 2.6, 2.6, 1.0, "PDE residual loss\n$\\|\\partial_t C + \\mathbf{u}\\cdot\\nabla C - K\\nabla^2 C - S\\|$", GREEN)
    box(6.9, 1.2, 2.6, 1.0, "upwelling source term\n$S = \\alpha\\, w_E^+ \\ast g(\\text{lag}) + \\dots$", YELLOW)
    box(6.9, 0.05, 2.6, 0.85, "coast / land: no-flux\nboundary, masked stencil", "#e6e6e6")
    for yy in (4.5, 3.1, 1.7, 0.45):
        arrow(ax, (6.0, 3.1), (6.9, yy), color="0.4", lw=1.2, ms=10)
    box(10.3, 2.3, 2.5, 1.6, "total loss\n$L = L_{data} + \\lambda_{pde} L_{pde}$\nwith loss balancing\n(NTK / grad-norm / annealed $\\lambda$)", "#e6e6e6")
    for yy in (4.5, 3.1, 1.7, 0.45):
        arrow(ax, (9.5, yy), (10.3, 3.1), color="0.4", lw=1.2, ms=10)
    ax.text(6.5, 0.0, "PDE residual evaluated with finite differences on the grid (no autodiff in x, y): pixels under real clouds are\nstill constrained by the PDE (no truth needed), pixels hidden by fake clouds by both data and PDE",
            ha="center", va="bottom", fontsize=8.5, style="italic")
    ax.set_title("Physics-informed gap filling on gridded data: where each loss term acts", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def vector_calculus(path):
    """Gradient, divergence, curl, Laplacian on toy fields."""
    x = np.linspace(-2, 2, 21); X, Y = np.meshgrid(x, x)
    xf = np.linspace(-2, 2, 201); Xf, Yf = np.meshgrid(xf, xf)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    # gradient of a scalar hill
    ax = axes[0]; Cf = np.exp(-(Xf ** 2 + Yf ** 2)); C = np.exp(-(X ** 2 + Y ** 2))
    ax.contourf(Xf, Yf, Cf, 20, cmap="viridis"); gx, gy = np.gradient(C, x, x, axis=(1, 0))
    ax.quiver(X[::2, ::2], Y[::2, ::2], gx[::2, ::2], gy[::2, ::2], color="w", scale=8)
    ax.set_title("gradient  $\\nabla C = (\\partial_x C, \\partial_y C)$\npoints uphill, |grad| = steepness")
    # divergence: radial field
    ax = axes[1]; U, V = X, Y
    ax.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2], color=BLUE, scale=30)
    ax.set_title("divergence  $\\nabla\\cdot\\mathbf{u} = \\partial_x u + \\partial_y v$\nhere > 0: net outflow (a source)")
    # curl: solid-body rotation
    ax = axes[2]; U, V = -Y, X
    ax.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2], color=ORANGE, scale=30)
    ax.set_title("curl  $(\\nabla\\times\\mathbf{u})_z = \\partial_x v - \\partial_y u$\nhere > 0: counter-clockwise spin")
    # Laplacian: hill has negative Laplacian at the top
    ax = axes[3]; lap = np.gradient(np.gradient(Cf, xf, axis=1), xf, axis=1) + np.gradient(np.gradient(Cf, xf, axis=0), xf, axis=0)
    im = ax.contourf(Xf, Yf, lap, 20, cmap="PuOr", vmin=-4, vmax=4); ax.contour(Xf, Yf, Cf, 6, colors="k", linewidths=0.5)
    ax.set_title("Laplacian  $\\nabla^2 C = \\partial_{xx} C + \\partial_{yy} C$\n< 0 at a peak, > 0 in a trough (contours: C)")
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Vector-calculus operators on a scalar field C (a tracer) and a vector field u (a current)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def advection_diffusion(path):
    """A 1-D blob under pure advection, pure diffusion, both, and with a source."""
    x = np.linspace(0, 100, 401); dx = x[1] - x[0]
    c0 = np.exp(-((x - 25) / 5) ** 2)
    u, K, dt, nsteps = 0.5, 0.5, 0.05, 1200            # arbitrary units, Courant 0.005: stable explicit stepping
    def run(u, K, src=0.0):
        c = c0.copy(); out = [c.copy()]
        for n in range(nsteps):
            cx = (c - np.roll(c, 1)) / dx                    # upwind (u > 0)
            cxx = (np.roll(c, -1) - 2 * c + np.roll(c, 1)) / dx ** 2
            c = c + dt * (-u * cx + K * cxx + src * c * (x > 50))
            if (n + 1) % 400 == 0:
                out.append(c.copy())
        return out
    cases = [("advection only:  $\\partial_t C = -u\\,\\partial_x C$\nshape moves at speed u", run(u, 0)),
             ("diffusion only:  $\\partial_t C = K\\,\\partial_{xx} C$\nshape spreads, stays put", run(0, K)),
             ("advection + diffusion", run(u, K)),
             ("advection + diffusion + source\n(growth where x > 50: the 'biology' term)", run(u, K, 0.02))]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    cols = [BLACK, BLUE, ORANGE, GREEN]
    for ax, (title, out) in zip(axes, cases):
        for k, c in enumerate(out):
            ax.plot(x, c, color=cols[k], lw=1.6, label=f"t = {k * 400 * dt:.0f}")
        ax.set_title(title, fontsize=10); ax.set_xlabel("x"); ax.grid(alpha=.3)
    axes[0].set_ylabel("tracer C"); axes[0].legend(fontsize=8)
    fig.suptitle("How each term of the tracer equation changes a blob of tracer over time (1-D toy, u > 0)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def courant_peclet(path):
    """Courant number: how far a parcel moves in one step relative to a cell; Peclet: advection vs diffusion."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]; ax.set_xlim(-0.7, 6.7); ax.set_ylim(-0.9, 2.6); ax.axis("off")
    for i in range(7):
        ax.add_patch(Rectangle((i - 0.5, 0), 1, 1, fill=False, ec="0.5"))
    ax.text(3, -0.4, "grid cells of width $\\Delta x$ (27 km here)", ha="center", fontsize=10)
    ax.scatter([1], [0.5], color=BLACK, zorder=3, s=40); ax.text(1, 0.72, "parcel at day t", ha="center", fontsize=9)
    for y0, cnum, col, lab in [(1.45, 0.5, BLUE, "Co = 0.5: parcel stays within its cell, neighbours suffice"),
                                (2.15, 2.5, VERM, "Co = 2.5: parcel skips two cells, a 3-point stencil cannot see it")]:
        arrow(ax, (1, y0), (1 + cnum, y0), color=col, lw=2.5)
        ax.text(0.5, y0 + 0.14, lab, fontsize=9, color=col)
    ax.set_title("Courant number  Co = $u\\,\\Delta t/\\Delta x$\n(cells travelled in one time step)", fontsize=11)
    ax = axes[1]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Peclet number  Pe = $u\\,\\Delta x / K$\n(advection vs diffusion across one cell)", fontsize=11)
    ax.text(0.02, 0.80, "Pe << 1:  diffusion dominates, fields are smooth,\n              the PDE behaves like the heat equation (easy)", fontsize=10, color=BLUE, va="top")
    ax.text(0.02, 0.55, "Pe >> 1:  advection dominates, fronts persist,\n              the PDE behaves like a wave equation (hard)", fontsize=10, color=VERM, va="top")
    ax.text(0.02, 0.28, "This grid: u = 0.5 m/s, $\\Delta x$ = 27 km, K = 100-1000 m$^2$/s\n   ->  Pe = 13-135", fontsize=10, va="top")
    ax.text(0.02, 0.10, "Co = 0.3-8 for u = 0.1-2.5 m/s at $\\Delta t$ = 1 day\n   (about half of all pixel-days have Co > 1, Section 2.2)", fontsize=10, va="top")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def coriolis_ekman(path):
    """Why moving water turns right in the NH, and the resulting Ekman transport 90 degrees off the wind."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]; ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(2 * np.cos(th), 2 * np.sin(th), color="0.6")
    ax.text(0, 2.05, "N", ha="center", va="bottom", fontsize=11); ax.text(0, -2.05, "S", ha="center", va="top", fontsize=11)
    ax.plot([-2, 2], [0, 0], color="0.6", ls="--"); ax.text(2.05, 0, "equator, f = 0", va="center", fontsize=9)
    for y0, lab, col in [(1.0, "NH: f > 0, deflection to the right", ORANGE), (-1.0, "SH: f < 0, deflection to the left", BLUE)]:
        r = np.sqrt(4 - y0 ** 2)
        arrow(ax, (-0.9, y0), (0.3, y0), color=col, lw=2.5)
        arrow(ax, (0.3, y0), (1.0, y0 - 0.55 * np.sign(y0)), color=col, lw=2.5, linestyle="--")
        ax.text(-0.9, y0 + 0.25 * np.sign(y0) + (0.1 if y0 > 0 else -0.35), lab, fontsize=9, color=col)
    ax.set_title("Coriolis parameter $f = 2\\Omega\\sin(\\mathrm{lat})$\nturns any horizontal motion, more strongly at high latitude")
    ax = axes[1]; ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.axis("off")
    arrow(ax, (-1.8, 1.7), (1.8, 1.7), color=BLACK, lw=3); ax.text(0, 1.85, "wind stress $\\tau$ (plan view, wind blowing to the right)", ha="center", fontsize=10)
    # Ekman spiral seen from above: velocity vectors at increasing depth, all drawn from one origin
    o = (-0.6, 0.9)
    for k, (ang, L, col) in enumerate(zip(np.deg2rad([-45, -70, -95, -120, -145, -170]), [1.5, 1.2, 0.95, 0.7, 0.5, 0.35],
                                          ["0.1", "0.25", "0.4", "0.55", "0.7", "0.8"])):
        arrow(ax, o, (o[0] + L * np.cos(ang), o[1] + L * np.sin(ang)), color=col, lw=1.8, ms=11)
    ax.text(o[0] + 1.1, o[1] - 1.15, "surface", fontsize=8, color="0.1"); ax.text(o[0] - 0.9, o[1] - 0.2, "deeper", fontsize=8, color="0.7")
    ax.text(-2.1, -1.4, "velocity at increasing depth (darker = shallower):\n45$^\\circ$ right of the wind at the surface, turning further\nright and weakening with depth (Ekman spiral)", fontsize=9, va="top")
    arrow(ax, (1.3, 1.2), (1.3, -0.8), color=ORANGE, lw=3.5); ax.text(1.42, 0.2, "net (depth-integrated)\ntransport $M_E = \\tau/(\\rho f)$\n90$^\\circ$ right of the wind", fontsize=10, color=ORANGE, va="center")
    ax.set_title("Wind pushes the surface; Coriolis turns it;\nthe depth-integrated flow is at right angles to the wind (NH)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    vector_calculus(FIG / "intro_vector_calculus.png")
    advection_diffusion(FIG / "intro_advection_diffusion.png")
    courant_peclet(FIG / "intro_courant_peclet.png")
    coriolis_ekman(FIG / "intro_coriolis_ekman.png")
    coastal_upwelling(FIG / "pinn_upwelling.png")
    tracer_stencil(FIG / "pinn_tracer_equation.png")
    pinn_architecture(FIG / "pinn_architecture.png")
    print("wrote", FIG)


if __name__ == "__main__":
    main()
