"""Per-cell epigenome state and its division-coupled update.

State is (n_cells, n_bins) per mark. One `step` is one cell division:
replication dilutes marks, reader-writer spreading restores them subject to
boundary permeability, and CpG methylation additionally accumulates
epimutations at a constant per-division rate.

Aging enters in exactly three places (see DESIGN.md §3):
  1. `maintain` decays with cumulative divisions (heterochromatin maintenance)
  2. `lad` tethering decays (lamin B1 decline)
  3. epimutation rate is constant per division, so VARIANCE grows linearly in
     divisions while the mean barely moves

(3) alone is enough to predict H2b. If the simulator reproduces rising
heterogeneity with no age-specific programme, that is the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

MARKS = ("k9", "k27", "ac")


@dataclass
class EpigenomeParams:
    spread_width: int = 5
    # Restoration must overcome a 2x replication dilution, so maintain > 1:
    # for mark x, x/2 + maintain*(x/2)*(1-x/2) must return to ~x at equilibrium.
    maintain: float = 1.85         # reader-writer restoration efficiency, young
    maintain_halflife: int = 1400  # divisions for maintenance to fall by half
    decay: float = 0.03
    noise: float = 0.010
    # Random walk: SD after d divisions ~ epimut_rate * sqrt(d). At rate 0.0025
    # and 1000 divisions that is ~0.08 -- drift without saturating against [0,1].
    epimut_rate: float = 0.0025
    island_drift: float = 3.0e-5   # focal hypermethylation in A / CpG islands
    repeat_drift: float = 4.5e-5   # global hypomethylation in B / repeats
    lad_halflife: int = 1100       # divisions for lamina tethering to halve
    sahf_gain: float = 0.35        # H3K9me3 gain in senescent cells (SAHF)
    senescence_halflife: int = 2200  # divisions until 50% of cells are senescent


@dataclass
class EpigenomeState:
    k9: np.ndarray
    k27: np.ndarray
    ac: np.ndarray
    meth: np.ndarray
    lad: np.ndarray
    divisions: int = 0
    senescent: np.ndarray = field(default=None)  # (n_cells,) bool

    @property
    def n_cells(self) -> int:
        return self.k9.shape[0]

    @property
    def n_bins(self) -> int:
        return self.k9.shape[1]

    def copy(self) -> "EpigenomeState":
        return EpigenomeState(
            self.k9.copy(), self.k27.copy(), self.ac.copy(),
            self.meth.copy(), self.lad.copy(), self.divisions,
            None if self.senescent is None else self.senescent.copy(),
        )


def _spread(x: np.ndarray, width: int, permeability: np.ndarray) -> np.ndarray:
    """Neighbour average, attenuated where boundaries are strong.

    `permeability` is (n_bins,) in [0, 1]; a strong CTCF boundary is near 0 and
    blocks mark propagation across it.
    """
    gated = x * permeability[None, :]
    return uniform_filter1d(gated, size=width, axis=1, mode="nearest")


def initialise(n_cells: int, tissue, rng: np.random.Generator) -> EpigenomeState:
    """Young, healthy epigenome consistent with the tissue's compartment prior."""
    n = tissue.a_prior.size
    a = tissue.a_prior

    def jitter(mu: np.ndarray, sd: float) -> np.ndarray:
        return np.clip(mu[None, :] + rng.normal(0, sd, (n_cells, n)), 0, 1)

    k9 = jitter(1 / (1 + np.exp(2.0 * a)) * 0.9, 0.05)
    k27 = jitter(1 / (1 + np.exp(1.2 * a)) * 0.5, 0.05)
    ac = jitter(1 / (1 + np.exp(-2.0 * a)) * 0.9, 0.05)
    meth = jitter(np.clip(0.75 - 0.22 / (1 + np.exp(-3.0 * a)), 0, 1), 0.04)
    lad = jitter(1 / (1 + np.exp(2.5 * a)) * 0.85, 0.05)

    return EpigenomeState(k9, k27, ac, meth, lad, 0,
                          np.zeros(n_cells, dtype=bool))


def step(
    state: EpigenomeState,
    tissue,
    genome,
    params: EpigenomeParams,
    rng: np.random.Generator,
    n_divisions: int = 1,
) -> EpigenomeState:
    """Advance the epigenome by `n_divisions` cell divisions."""
    s = state.copy()
    boundary = np.clip(genome.ctcf_strength * tissue.ctcf_mod, 0, 1)
    permeability = 1.0 - 0.85 * boundary
    a = tissue.a_prior
    island = 1 / (1 + np.exp(-3.0 * a))   # A-side: CpG-island-like
    repeat = 1 - island                    # B-side: repeat-like

    # Sequence-encoded nucleation sites. Without these every mark relaxes to the
    # same spatially-smoothed attractor and all compartment structure vanishes
    # within ~100 divisions. Spreading propagates a pattern; it cannot create one.
    nucleation = {
        "k9": 1 / (1 + np.exp(2.0 * a)),          # B-side / repeats
        "k27": 0.35 * 1 / (1 + np.exp(0.5 * a)),  # broad, weakly B-biased
        "ac": np.clip(tissue.enhancer, 0, 1),     # enhancers and active promoters
    }
    # H3K9me3 and H3K27ac are mutually exclusive; the antagonism is what keeps
    # A and B separated, so its decay is what produces compartment blurring.
    antagonist = {"k9": "ac", "k27": None, "ac": "k9"}

    for _ in range(n_divisions):
        s.divisions += 1
        # (1) maintenance efficiency decays with cumulative divisions
        maint = params.maintain * 0.5 ** (s.divisions / params.maintain_halflife)

        updated = {}
        for m in MARKS:
            x = getattr(s, m)
            half = x * 0.5                                    # replication dilution
            local = _spread(half, params.spread_width, permeability)
            drive = 0.6 * local + 0.4 * nucleation[m][None, :]
            opp = antagonist[m]
            veto = 1.0 - 0.7 * getattr(s, opp) if opp else 1.0
            restored = half + maint * drive * (1 - half) * veto - params.decay * half
            restored += rng.normal(0, params.noise, x.shape)
            updated[m] = np.clip(restored, 0, 1)
        for m, v in updated.items():
            setattr(s, m, v)

        # (3) epimutation: constant per-division rate, two-directional drift.
        # Variance grows ~linearly in divisions; the mean moves only slightly.
        drift = params.island_drift * island - params.repeat_drift * repeat
        s.meth = np.clip(
            s.meth + drift[None, :] + rng.normal(0, params.epimut_rate, s.meth.shape),
            0.02, 0.98,
        )

        # (2) lamina tethering decline
        s.lad = np.clip(s.lad * 0.5 ** (1 / params.lad_halflife), 0, 1)

    # Senescence accumulates with cumulative divisions.
    p_sen = 1.0 - 0.5 ** (s.divisions / params.senescence_halflife)
    newly = (rng.random(s.n_cells) < p_sen) & ~s.senescent
    s.senescent = s.senescent | newly

    # SAHF: senescent cells gain focal H3K9me3 — the one process running against
    # heterochromatin loss, which is why aged tissue is not uniformly decompacted.
    if s.senescent is not None and s.senescent.any():
        sel = s.senescent
        s.k9[sel] = np.clip(s.k9[sel] + params.sahf_gain * (1 - s.k9[sel]), 0, 1)

    return s


def compartment_affinity(state: EpigenomeState, normalize: bool = True) -> np.ndarray:
    """(n_cells, n_bins) A/B affinity implied by the current mark state.

    `normalize=False` preserves amplitude, which is what compartment-strength
    decay lives in; z-scoring per cell removes exactly that signal.
    """
    a = state.ac - 0.5 * (state.k9 + state.k27)
    a = a - a.mean(axis=1, keepdims=True)
    if not normalize:
        return a
    return a / (a.std(axis=1, keepdims=True) + 1e-9)
