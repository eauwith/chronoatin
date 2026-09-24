"""Virtual assays.

Nothing in `experiments/` analyses simulator ground truth directly; every
measurement passes through here first. The gap between the two is the Assay
Attenuation Coefficient (benchmarks, Tier 5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SCHiCParams:
    contacts_per_cell: int = 3_000   # realistic scHi-C depth is sparse
    depth_dispersion: float = 0.35   # negative-binomial-ish over-dispersion
    random_ligation: float = 0.04    # uniform noise contacts
    doublet_rate: float = 0.06       # barcode collisions
    duplicate_rate: float = 0.15     # PCR duplicates, removed -> depth loss
    capture_halflife_bins: float = 220.0  # distance-dependent capture decay
    batch_depth_multiplier: float = 1.0
    batch_ps_shift: float = 0.0      # batch-specific P(s) slope distortion


def _capture_bias(n: int, halflife: float, ps_shift: float) -> np.ndarray:
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    bias = 0.5 ** (d / halflife)
    if ps_shift:
        with np.errstate(divide="ignore"):
            bias = bias * np.exp(-ps_shift * np.log(np.maximum(d, 1)))
    return bias


def schic(
    truth: np.ndarray,
    params: SCHiCParams,
    rng: np.random.Generator,
    contaminant: np.ndarray | None = None,
) -> np.ndarray:
    """Sparse observed contact counts for one cell. Returns an (n, n) int matrix."""
    n = truth.shape[0]
    p = truth * _capture_bias(n, params.capture_halflife_bins, params.batch_ps_shift)

    if contaminant is not None and rng.random() < params.doublet_rate:
        p = 0.5 * p / p.sum() + 0.5 * contaminant / contaminant.sum()

    p = p / p.sum()
    uniform = np.ones_like(p)
    np.fill_diagonal(uniform, 0.0)
    p = (1 - params.random_ligation) * p + params.random_ligation * uniform / uniform.sum()

    mean_depth = params.contacts_per_cell * params.batch_depth_multiplier
    shape = 1.0 / max(params.depth_dispersion, 1e-6)
    depth = int(rng.gamma(shape, mean_depth / shape))
    depth = int(depth * (1 - params.duplicate_rate))
    if depth <= 0:
        return np.zeros_like(truth, dtype=np.int32)

    flat = rng.multinomial(depth, p.ravel())
    obs = flat.reshape(p.shape).astype(np.int32)
    return np.triu(obs, 1) + np.triu(obs, 1).T


@dataclass
class RRBSParams:
    """Virtual RRBS, matching the format of selected_percent_methylation.csv."""

    mean_coverage: float = 12.0
    coverage_dispersion: float = 0.5
    dropout: float = 0.10
    conversion_error: float = 0.005


def rrbs(true_meth: np.ndarray, params: RRBSParams, rng: np.random.Generator) -> np.ndarray:
    """Observed percent methylation with binomial sampling noise and dropout.

    Returns NaN where a site dropped out -- sparsity that downstream code must
    handle rather than silently impute.
    """
    shape = 1.0 / max(params.coverage_dispersion, 1e-6)
    cov = rng.gamma(shape, params.mean_coverage / shape, true_meth.shape)
    cov = np.maximum(np.round(cov), 0).astype(int)

    p = np.clip(true_meth * (1 - params.conversion_error)
                + (1 - true_meth) * params.conversion_error, 0, 1)
    with np.errstate(invalid="ignore"):
        obs = np.where(cov > 0, rng.binomial(np.maximum(cov, 1), p) / np.maximum(cov, 1), np.nan)
    obs = np.where(cov == 0, np.nan, obs)
    obs[rng.random(obs.shape) < params.dropout] = np.nan
    return obs
