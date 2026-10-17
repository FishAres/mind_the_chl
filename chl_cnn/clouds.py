"""Fake clouds: hide observed CHL pixels using the cloud pattern of another day.

For day i and a donor day j != i, the hidden set is  observed_i & ~observed_j.
Those pixels have a known truth, so the model can be scored "under clouds".
Val/test donors are fixed by seed so every run is scored on the same pixels.

Donor exclusion (2026-10-16; `exclude` D, default 0 = the original draws, bit for bit): a donor day must not fall within
the calendar window [t-D, t+D] of the target day t (dates compared, not indices). Rejected donors are redrawn from the
same pool, exactly like the "never the day itself" rule (which D = 0 reduces to).
"""
import numpy as np


def _days(times):
    return np.asarray(times).astype("datetime64[D]").astype(np.int64)


def donor_indices(n, rng, pool=None, times=None, exclude=0):
    """One donor index per day, never the day itself. `pool`: candidate donor indices.
    exclude D > 0 (needs `times`, the dates of the n days): also never a day within D calendar days of the target."""
    pool = np.arange(n) if pool is None else np.asarray(pool)
    donors = rng.choice(pool, size=n)
    if exclude > 0:
        td = _days(times)
        bad = lambda: np.abs(td[donors] - td) <= exclude
    else:
        bad = lambda: donors == np.arange(n)
    same = bad()
    while same.any():
        donors[same] = rng.choice(pool, size=int(same.sum()))
        same = bad()
    return donors


def hidden_mask(obs, donors):
    """obs: (N,H,W) bool observed; returns (N,H,W) bool = observed today & cloudy on donor day."""
    return obs & ~obs[donors]


def fixed_hidden(obs, idx, seed=0, times=None, exclude=0):
    """Deterministic hidden mask for the days `idx` (donors drawn from the same days).
    exclude D > 0 (needs `times`, the dates of every store day): no donor within D calendar days of its target day."""
    rng = np.random.default_rng(seed)
    local = donor_indices(len(idx), rng, times=(np.asarray(times)[idx] if exclude > 0 else None), exclude=exclude)
    return hidden_mask(obs[idx], local), idx[local]


def seasonal_donor_indices(times, rng, window=15, exclude=0):
    """One donor per day drawn from the days whose day-of-year is within +-window days (circular, any year),
    never the day itself (exclude D > 0: never a day within D calendar days of it).
    `times`: datetime64 array of the candidate days; returns indices into that array."""
    doy = (np.asarray(times).astype("datetime64[D]") - np.asarray(times).astype("datetime64[Y]")).astype(int)
    n = len(doy)
    dist = np.abs(doy[:, None] - doy[None, :]); dist = np.minimum(dist, 365 - dist)
    if exclude > 0:
        td = _days(times); near = np.abs(td[:, None] - td[None, :]) <= exclude
    else:
        near = np.eye(n, dtype=bool)
    ok = (dist <= window) & ~near
    donors = np.empty(n, dtype=np.int64)
    for i in range(n):
        cand = np.flatnonzero(ok[i])
        donors[i] = rng.choice(cand) if len(cand) else rng.choice(np.flatnonzero(~near[i]))
    return donors
