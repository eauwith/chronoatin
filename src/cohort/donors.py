"""Donor cohort generation and the analysis-ready per-cell feature table.

The statistical unit is the DONOR. Cells are pseudo-replicates. `rho_age_batch`
dials the age-batch confound from 0 (randomised, unobtainable in real aging
studies) to 1 (fully confounded, which is roughly what a real cohort collected
over two decades looks like -- cf. the `prepost2017` column in the Cayo data).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..assay.virtual import RRBSParams, SCHiCParams, rrbs, schic
from ..engine.conformation import ConformationParams, contact_map
from ..engine.epigenome import (
    EpigenomeParams,
    compartment_affinity,
    initialise,
    step,
)
from ..inference.features import (
    compartment_eigenvector,
    compartment_strength,
    insulation_score,
    ps_slope,
)

# Tissue-specific proliferative history. Blood turns over fast; liver is quiescent.
DIVISIONS_PER_YEAR = {"blood": 58.0, "colorectal": 46.0, "lung": 22.0, "liver": 14.0}


@dataclass
class Donor:
    donor_id: str
    age: float
    sex: str
    batch: int
    tissue: str
    n_cells: int = 24
    # Donor random effects. Real donors differ for reasons other than age --
    # genetic background, exposure history, collection. Without these, cells
    # within a donor are exchangeable and there is no pseudo-replication to
    # demonstrate: the donor-level and cell-level standard errors coincide.
    offset_meth: float = 0.0   # baseline methylation shift
    offset_amp: float = 1.0    # compartment amplitude scaling


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def make_donors(
    ages: np.ndarray,
    tissue: str,
    rng: np.random.Generator,
    n_cells: int = 24,
    n_batches: int = 3,
    rho_age_batch: float = 0.0,
) -> list[Donor]:
    order = np.argsort(ages)
    donors: list[Donor] = []
    for rank, i in enumerate(order):
        if rng.random() < rho_age_batch:
            batch = int(rank * n_batches / len(ages))          # age-ordered = confounded
        else:
            batch = int(rng.integers(0, n_batches))            # randomised
        donors.append(
            Donor(f"D{i:03d}", float(ages[i]), rng.choice(["F", "M"]),
                  batch, tissue, n_cells,
                  offset_meth=float(rng.normal(0, 0.025)),
                  offset_amp=float(rng.normal(1.0, 0.12)))
        )
    return donors


def simulate_donor(
    donor: Donor,
    genome,
    tissue,
    rng: np.random.Generator,
    chrom: int = 0,
    epi: EpigenomeParams | None = None,
    conf: ConformationParams | None = None,
    hic: SCHiCParams | None = None,
    meth_assay: RRBSParams | None = None,
    ground_truth: bool = False,
) -> pd.DataFrame:
    """One donor -> one row per cell of observed features."""
    epi = epi or EpigenomeParams()
    conf = conf or ConformationParams()
    hic = hic or SCHiCParams()
    meth_assay = meth_assay or RRBSParams()

    divisions = int(donor.age * DIVISIONS_PER_YEAR.get(donor.tissue, 30.0))
    state = initialise(donor.n_cells, tissue, rng)
    # Apply donor random effects before ageing, so they propagate through the
    # whole trajectory rather than being added to the endpoint.
    state.meth = np.clip(state.meth + donor.offset_meth, 0.02, 0.98)
    state.ac = np.clip(state.ac * donor.offset_amp, 0, 1)
    state.k9 = np.clip(state.k9 * donor.offset_amp, 0, 1)
    state = step(state, tissue, genome, epi, rng, n_divisions=divisions)

    sl = genome.chrom_slice(chrom)
    cs, co = genome.ctcf_strength[sl], genome.ctcf_orient[sl]
    bnd = np.clip(cs * tissue.ctcf_mod[sl], 0, 1)
    prior = tissue.a_prior[sl]

    # Batch effects act on the measurement, not the biology.
    hic = SCHiCParams(**{**hic.__dict__,
                         "batch_depth_multiplier": [1.0, 0.65, 1.3][donor.batch % 3],
                         "batch_ps_shift": [0.0, 0.12, -0.09][donor.batch % 3]})

    aff = compartment_affinity(state, normalize=False)[:, sl]
    rows = []
    for c in range(donor.n_cells):
        truth = contact_map(aff[c], bnd, cs, co, conf, rng)
        if ground_truth:
            mat = truth
        else:
            other = contact_map(aff[(c + 1) % donor.n_cells], bnd, cs, co, conf, rng)
            mat = schic(truth, hic, rng, contaminant=other)

        pc1 = compartment_eigenvector(mat, orient_by=prior)
        ins = insulation_score(mat)

        meth_true = state.meth[c][sl]
        meth_obs = meth_true if ground_truth else rrbs(meth_true, meth_assay, rng)

        rows.append({
            "donor_id": donor.donor_id, "age": donor.age, "sex": donor.sex,
            "batch": donor.batch, "tissue": donor.tissue, "cell": c,
            "senescent": bool(state.senescent[c]),
            "comp_strength": compartment_strength(pc1),
            "comp_fidelity": float(np.corrcoef(pc1, prior)[0, 1]),
            "insulation_mean": float(np.nanmean(ins)),
            "insulation_sd": float(np.nanstd(ins)),
            "ps_slope": ps_slope(mat),
            "meth_mean": float(np.nanmean(meth_obs)),
            "meth_sd": float(np.nanstd(meth_obs)),
            "meth_entropy": float(np.nanmean(binary_entropy(meth_obs[np.isfinite(meth_obs)]))),
            "contacts": int(mat.sum() // 2),
        })
    return pd.DataFrame(rows)


def simulate_cohort(
    genome, tissues: dict, tissue_name: str, ages: np.ndarray, rng: np.random.Generator,
    n_cells: int = 24, rho_age_batch: float = 0.0, ground_truth: bool = False, **kw
) -> pd.DataFrame:
    donors = make_donors(ages, tissue_name, rng, n_cells=n_cells,
                         rho_age_batch=rho_age_batch)
    return pd.concat(
        [simulate_donor(d, genome, tissues[tissue_name], rng,
                        ground_truth=ground_truth, **kw) for d in donors],
        ignore_index=True,
    )
