"""Estimators run on OBSERVED data: compartments, insulation, P(s)."""
from __future__ import annotations

import numpy as np


def observed_over_expected(m: np.ndarray) -> np.ndarray:
    """Distance-normalise a contact matrix."""
    n = m.shape[0]
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    out = np.zeros_like(m, dtype=float)
    for k in range(1, n):
        mask = d == k
        exp = m[mask].mean()
        if exp > 0:
            out[mask] = m[mask] / exp
    return out


def compartment_eigenvector(m: np.ndarray, orient_by: np.ndarray | None = None) -> np.ndarray:
    """PC1 of the correlation matrix of observed/expected -- the standard caller."""
    oe = observed_over_expected(m)
    keep = oe.sum(axis=1) > 0
    if keep.sum() < 3:
        return np.zeros(m.shape[0])
    c = np.corrcoef(oe[np.ix_(keep, keep)])
    c = np.nan_to_num(c)
    w, v = np.linalg.eigh(c)
    pc1_sub = v[:, -1] * np.sqrt(max(w[-1], 0))
    pc1 = np.zeros(m.shape[0])
    pc1[keep] = pc1_sub
    # Eigenvector sign is arbitrary; orient against a prior (GC, or known A/B).
    if orient_by is not None and np.corrcoef(pc1[keep], orient_by[keep])[0, 1] < 0:
        pc1 = -pc1
    return pc1


def insulation_score(m: np.ndarray, window: int = 8) -> np.ndarray:
    """Mean contacts in a sliding diamond; minima mark TAD boundaries."""
    n = m.shape[0]
    out = np.full(n, np.nan)
    for i in range(window, n - window):
        out[i] = m[i - window:i, i:i + window].mean()
    finite = np.isfinite(out)
    if finite.sum() and np.nanmean(out[finite]) > 0:
        out[finite] = np.log2(out[finite] / np.nanmean(out[finite]) + 1e-9)
    return out


def ps_slope(m: np.ndarray, lo: int = 3, hi: int = 80) -> float:
    """Slope of log P vs log s -- real Hi-C sits near -1.0 to -1.5."""
    n = m.shape[0]
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    ds, ps = [], []
    for k in range(lo, min(hi, n)):
        vals = m[d == k]
        if vals.size and vals.mean() > 0:
            ds.append(k)
            ps.append(vals.mean())
    if len(ds) < 5:
        return np.nan
    return float(np.polyfit(np.log(ds), np.log(ps), 1)[0])


def compartment_strength(pc1: np.ndarray) -> float:
    """Amplitude of the A/B split -- what erodes with age."""
    v = pc1[np.isfinite(pc1)]
    return float(v.std()) if v.size else np.nan
