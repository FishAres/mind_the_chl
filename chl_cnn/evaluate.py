"""Metrics on selected pixels + reference predictors."""
import numpy as np


def metrics(y_true, y_pred, mask):
    """Scores in ln(CHL) units over pixels where mask & both finite."""
    m = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    t, p = y_true[m], y_pred[m]
    if t.size == 0:
        return dict(mae=np.nan, rmse=np.nan, bias=np.nan, r=np.nan, r2=np.nan, n=0)
    e = p - t
    ss = ((t - t.mean()) ** 2).sum()
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                r=float(np.corrcoef(t, p)[0, 1]) if t.size > 1 else np.nan,
                r2=float(1 - (e ** 2).sum() / ss) if ss > 0 else np.nan, n=int(m.sum()))


def per_day_metrics(y_true, y_pred, mask):
    return [metrics(y_true[i], y_pred[i], mask[i]) for i in range(len(y_true))]


def climatology_reference(y_ln, train_idx):
    """Per-pixel mean ln(CHL) over observed training days; broadcast to all days."""
    clim = np.nanmean(y_ln[train_idx], axis=0)
    return np.broadcast_to(clim, y_ln.shape).copy()


def mae_by_bin(y_true, y_pred, mask, edges):
    """MAE binned by true ln(CHL). Returns (centres, mae, n)."""
    m = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    t, p = y_true[m], y_pred[m]
    which = np.digitize(t, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        s = which == b
        out.append((0.5 * (edges[b] + edges[b + 1]), float(np.abs(p[s] - t[s]).mean()) if s.any() else np.nan, int(s.sum())))
    return np.array(out).T
