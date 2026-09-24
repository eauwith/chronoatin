"""Per-cell chromatin conformation from epigenome state.

Two mechanisms, as in the standard picture:
  * loop extrusion -- cohesin LEFs extrude until stalled by a CTCF site in
    convergent orientation, producing TADs and corner dots
  * compartmental attraction -- block-copolymer-like affinity between bins of
    like A/B identity, producing the plaid checkerboard

Both are driven by the epigenome state, so aging propagates into conformation
without any separate "aged conformation" parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformationParams:
    gamma: float = 1.15          # P(s) exponent; real Hi-C is ~1.0-1.5
    lam: float = 0.55            # compartment attraction strength
    tad_bonus: float = 0.60      # same-TAD contact bonus
    loop_bonus: float = 1.30     # corner-dot strength at stalled LEF anchors
    lef_per_100bins: float = 9.0
    extrusion_steps: int = 140
    boundary_kappa: float = 2.2  # CTCF strength -> per-cell boundary probability


def extrude(
    n: int,
    ctcf_strength: np.ndarray,
    ctcf_orient: np.ndarray,
    n_lef: int,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final (left, right) anchor positions of `n_lef` extruders.

    Convergent-CTCF rule: a site oriented +1 stalls an anchor arriving from the
    right (the left anchor); a site oriented -1 stalls the right anchor.
    """
    left = rng.integers(0, n - 1, n_lef)
    right = left + 1
    for _ in range(steps):
        block_l = (ctcf_orient[left] == 1) & (rng.random(n_lef) < ctcf_strength[left])
        left = np.where(block_l, left, np.maximum(left - 1, 0))
        block_r = (ctcf_orient[right] == -1) & (rng.random(n_lef) < ctcf_strength[right])
        right = np.where(block_r, right, np.minimum(right + 1, n - 1))
    return left, right


def sample_boundaries(
    boundary_strength: np.ndarray, kappa: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-cell boundary realisation. Single-cell TADs are variable, not fixed."""
    p = 1.0 - np.exp(-kappa * boundary_strength)
    return rng.random(boundary_strength.size) < p


def contact_map(
    affinity: np.ndarray,
    boundary_strength: np.ndarray,
    ctcf_strength: np.ndarray,
    ctcf_orient: np.ndarray,
    params: ConformationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Expected intra-chromosomal contact probability for ONE cell. (n, n)."""
    n = affinity.size
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])

    with np.errstate(divide="ignore"):
        logp = -params.gamma * np.log(np.maximum(d, 1))

    a = (affinity - affinity.mean()) / (affinity.std() + 1e-9)
    logp = logp + params.lam * np.outer(a, a)

    is_bnd = sample_boundaries(boundary_strength, params.boundary_kappa, rng)
    domain = np.cumsum(is_bnd)
    same = domain[:, None] == domain[None, :]
    logp = logp + params.tad_bonus * same - 0.25 * params.tad_bonus * (~same)

    n_lef = max(1, int(params.lef_per_100bins * n / 100))
    left, right = extrude(n, ctcf_strength, ctcf_orient, n_lef, params.extrusion_steps, rng)
    loops = np.zeros((n, n))
    np.add.at(loops, (left, right), 1.0)
    loops = loops + loops.T
    logp = logp + params.loop_bonus * np.tanh(loops)

    p = np.exp(logp)
    np.fill_diagonal(p, 0.0)
    return p / p.sum()
