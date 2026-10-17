"""Physics diagnostics for the PINN study: wind stress, Ekman pumping, Courant number, and the
finite-difference tracer budget on the 0.25 deg grid. NumPy only; used by notebooks/pinn_design.ipynb.

Conventions: lat/lon 1-D arrays (deg), fields (..., lat, lon); lat decreasing in IO.zarr is handled
by using the signed grid spacing. w_E > 0 means upward (upwelling).
"""
import numpy as np

RHO_AIR, RHO_W, OMEGA, R_EARTH = 1.22, 1025.0, 7.2921e-5, 6.371e6


def drag_coefficient(speed):
    """Large & Pond (1981): 1.2e-3 below 11 m/s, (0.49 + 0.065 U) e-3 above (clipped at 4-25 m/s range)."""
    s = np.clip(speed, 4.0, 25.0)
    return np.where(s < 11.0, 1.2e-3, (0.49 + 0.065 * s) * 1e-3)


def wind_stress(u10, v10):
    spd = np.hypot(u10, v10)
    cd = drag_coefficient(spd)
    return RHO_AIR * cd * spd * u10, RHO_AIR * cd * spd * v10


def grid_metrics(lat, lon):
    """dx, dy in metres (signed, along increasing index) as 2-D arrays, and f, beta."""
    latr = np.deg2rad(lat)
    dlat = np.gradient(lat); dlon = np.gradient(lon)
    dy = (np.deg2rad(dlat) * R_EARTH)[:, None] * np.ones((1, len(lon)))
    dx = (np.deg2rad(dlon)[None, :] * R_EARTH * np.cos(latr)[:, None])
    f = 2 * OMEGA * np.sin(latr)[:, None] * np.ones((1, len(lon)))
    beta = 2 * OMEGA * np.cos(latr)[:, None] / R_EARTH * np.ones((1, len(lon)))
    return dx, dy, f, beta


def ddx(a, dx):
    return np.gradient(a, axis=-1) / dx


def ddy(a, dy):
    return np.gradient(a, axis=-2) / dy


def ekman(u10, v10, lat, lon, lat_min=3.0):
    """Ekman transport (m^2/s, volume) and pumping w_E (m/s, + upward) from 10 m winds.
    Cells with |lat| < lat_min are NaN (f -> 0). Includes the beta term."""
    tx, ty = wind_stress(u10, v10)
    dx, dy, f, beta = grid_metrics(lat, lon)
    f = np.where(np.abs(lat)[:, None] < lat_min, np.nan, f)      # equatorial band -> NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = ty / (RHO_W * f), -tx / (RHO_W * f)
        w = (ddx(ty, dx) - ddy(tx, dy)) / (RHO_W * f) + beta * tx / (RHO_W * f ** 2)
    return mx, my, w, tx, ty


def courant(u, v, lat, lon, dt=86400.0):
    dx, dy, _, _ = grid_metrics(lat, lon)
    return np.abs(u) * dt / np.abs(dx) + np.abs(v) * dt / np.abs(dy)


def tracer_terms(c0, c1, u, v, lat, lon, dt=86400.0):
    """Finite-difference terms of the tracer equation between two consecutive days.
    Returns tendency (c1-c0)/dt, advection u.grad(c) (centred, on the day-0 field with NaN
    where a neighbour is missing) and the Laplacian of c0 (per m^2)."""
    dx, dy, _, _ = grid_metrics(lat, lon)
    tend = (c1 - c0) / dt
    cx = (np.roll(c0, -1, -1) - np.roll(c0, 1, -1)) / (2 * dx)
    cy = (np.roll(c0, -1, -2) - np.roll(c0, 1, -2)) / (2 * dy)
    adv = u * cx + v * cy
    lap = ((np.roll(c0, -1, -1) - 2 * c0 + np.roll(c0, 1, -1)) / dx ** 2 +
           (np.roll(c0, -1, -2) - 2 * c0 + np.roll(c0, 1, -2)) / dy ** 2)
    return tend, adv, lap
