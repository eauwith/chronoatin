"""Binned genome scaffold.

Conformation is modelled at `bin_size` (default 100 kb) because that is the scale
TADs and compartments live at. Methylation features are keyed at 1 kb windows to
match the column format already used in `Methylation Aging`
(e.g. "chr4_56044000_56045000"), and `window_to_bin` joins the two.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

BIN_SIZE = 100_000
WINDOW_SIZE = 1_000


@dataclass(frozen=True)
class Genome:
    n_chrom: int
    bins_per_chrom: int
    bin_size: int
    chrom: np.ndarray          # (n_bins,) chromosome index
    start: np.ndarray          # (n_bins,) genomic start coordinate
    ctcf_strength: np.ndarray  # (n_bins,) in [0, 1]; 0 = no site
    ctcf_orient: np.ndarray    # (n_bins,) -1, 0, +1
    gene_bin: dict[str, int]

    @property
    def n_bins(self) -> int:
        return self.n_chrom * self.bins_per_chrom

    def chrom_slice(self, c: int) -> slice:
        return slice(c * self.bins_per_chrom, (c + 1) * self.bins_per_chrom)

    def window_key(self, bin_idx: int, sub: int = 0) -> str:
        """Her column format, for the 1 kb window `sub` inside a conformation bin."""
        s = int(self.start[bin_idx]) + sub * WINDOW_SIZE
        return f"chr{int(self.chrom[bin_idx]) + 1}_{s}_{s + WINDOW_SIZE}"

    def window_to_bin(self, key: str) -> int | None:
        """Inverse of `window_key`: map a 1 kb methylation column onto its bin."""
        try:
            chrom_s, start_s, _ = key.split("_")
            c = int(chrom_s.removeprefix("chr")) - 1
            s = int(start_s)
        except (ValueError, AttributeError):
            return None
        if not 0 <= c < self.n_chrom:
            return None
        b = c * self.bins_per_chrom + s // self.bin_size
        return b if self.chrom_slice(c).start <= b < self.chrom_slice(c).stop else None


@dataclass(frozen=True)
class Tissue:
    """A tissue's normal, young chromatin identity."""

    name: str
    a_prior: np.ndarray    # (n_bins,) compartment prior; positive = A
    ctcf_mod: np.ndarray   # (n_bins,) multiplier on CTCF occupancy
    enhancer: np.ndarray   # (n_bins,) enhancer density in [0, 1]
    expression: np.ndarray # (n_bins,) log-scale expression


# Driver sequences carried over from Features-of-Carcinogenesis.
DRIVER_SEQUENCES: dict[str, list[str]] = {
    "colorectal": ["APC", "KRAS", "MLH1", "PIK3CA", "SMAD4", "TP53"],
    "lung":       ["KRAS", "EGFR", "TP53", "CDKN2A", "STK11", "KEAP1"],
    "liver":      ["TERT", "CTNNB1", "ARID1A", "TP53", "AXIN1", "CDKN2A"],
    "blood":      ["DNMT3A", "TET2", "ASXL1", "IDH2", "TP53", "NPM1", "FLT3"],
}

ALL_DRIVERS: list[str] = sorted({g for seq in DRIVER_SEQUENCES.values() for g in seq})


def _smooth(x: np.ndarray, width: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Circular moving average — cheap way to give a random vector spatial structure."""
    k = np.ones(width) / width
    return np.convolve(np.r_[x[-width:], x, x[:width]], k, mode="same")[width:-width]


def build_genome(
    n_chrom: int = 3,
    bins_per_chrom: int = 300,
    bin_size: int = BIN_SIZE,
    ctcf_density: float = 0.08,
    seed: int = 0,
) -> Genome:
    rng = np.random.default_rng(seed)
    n_bins = n_chrom * bins_per_chrom

    chrom = np.repeat(np.arange(n_chrom), bins_per_chrom)
    start = np.tile(np.arange(bins_per_chrom), n_chrom) * bin_size

    has_site = rng.random(n_bins) < ctcf_density
    ctcf_strength = np.where(has_site, rng.beta(4, 2, n_bins), 0.0)
    ctcf_orient = np.where(has_site, rng.choice([-1, 1], n_bins), 0)

    # Genes land on distinct bins, well away from chromosome edges: the
    # insulation window is 8 bins, so anything closer than that has undefined
    # insulation and would silently drop out of the H1 feature table.
    margin = 15
    interior = np.flatnonzero(
        (start > margin * bin_size) & (start < (bins_per_chrom - margin) * bin_size)
    )
    picks = rng.choice(interior, size=len(ALL_DRIVERS), replace=False)
    gene_bin = {g: int(b) for g, b in zip(ALL_DRIVERS, picks)}

    return Genome(
        n_chrom=n_chrom,
        bins_per_chrom=bins_per_chrom,
        bin_size=bin_size,
        chrom=chrom,
        start=start,
        ctcf_strength=ctcf_strength,
        ctcf_orient=ctcf_orient,
        gene_bin=gene_bin,
    )


def build_tissues(
    genome: Genome,
    names: list[str] | None = None,
    shared_fraction: float = 0.55,
    seed: int = 1,
) -> dict[str, Tissue]:
    """Tissues share a core compartment structure and deviate tissue-specifically.

    `shared_fraction` controls how much of the A/B profile is common across tissues.
    Too high and no tissue-specific signal exists to detect; too low and tissues are
    unrelated. 0.55 puts real but non-trivial tissue identity in the scaffold.
    """
    names = names or list(DRIVER_SEQUENCES)
    rng = np.random.default_rng(seed)
    n = genome.n_bins

    core = _smooth(rng.normal(size=n), 25)
    core /= core.std()

    tissues: dict[str, Tissue] = {}
    for name in names:
        own = _smooth(rng.normal(size=n), 15)
        own /= own.std()
        a = shared_fraction * core + (1 - shared_fraction) * own
        a /= a.std()

        ctcf_mod = np.clip(1.0 + 0.3 * _smooth(rng.normal(size=n), 10), 0.2, 1.8)
        enhancer = np.clip(0.5 + 0.5 * a + 0.2 * rng.normal(size=n), 0, 1)
        expression = np.clip(2.0 + 1.5 * a + 0.5 * rng.normal(size=n), 0, None)

        tissues[name] = Tissue(name, a, ctcf_mod, enhancer, expression)
    return tissues


def replication_timing(tissue: Tissue, seed: int = 7) -> np.ndarray:
    """Early (high) in A, late (low) in B.

    Correlated with compartment but NOT identical to it. This matters: if
    replication timing were a deterministic function of the compartment prior,
    the mutation-supply and selection terms of H1 would be perfectly collinear
    and the decomposition would be unidentifiable by construction.
    """
    # zlib.crc32, not hash(): built-in string hashing is salted per process
    # (PYTHONHASHSEED), which would make the simulator non-reproducible
    # across runs -- silently, and only in the replication-timing features.
    rng = np.random.default_rng(seed + zlib.crc32(tissue.name.encode()) % 100_000)
    independent = _smooth(rng.normal(size=tissue.a_prior.size), 20)
    rt = 0.75 * tissue.a_prior + 0.55 * (independent / independent.std())
    return (rt - rt.mean()) / rt.std()
