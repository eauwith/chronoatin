"""E5 -- power and attenuation surfaces.

The deliverable with the shortest path to being useful to someone else:
how many donors, and how much sequencing depth, does an effect need to be
detectable? Answering that before spending money is the point of the
environment.

    python experiments/h3_power_attenuation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assay.virtual import SCHiCParams
from src.benchmarks.ladder import tier4_reality_discriminator
from src.cohort.donors import simulate_cohort
from src.genome.scaffold import build_genome, build_tissues
from src.inference.aging import donor_summary, location_scale

TISSUE = "blood"
FEATURE = "meth_sd"


def main(seed: int = 5, n_cells: int = 10, n_reps: int = 4) -> None:
    genome = build_genome()
    tissues = build_tissues(genome)

    print("=" * 74)
    print("E5  Power: donors needed to detect the age effect on", FEATURE)
    print("=" * 74)
    print(f"\n{'n_donors':<11}{'detected (p<0.05)':<21}{'mean |slope|':<15}{'median p'}")
    rows = []
    for n_donors in (6, 10, 16, 24):
        hits, slopes, ps = 0, [], []
        for r in range(n_reps):
            rng = np.random.default_rng(seed + 100 * r + n_donors)
            ages = np.linspace(2, 27, n_donors)
            cells = simulate_cohort(genome, tissues, TISSUE, ages, rng, n_cells=n_cells)
            res = location_scale(donor_summary(cells), FEATURE)
            if np.isfinite(res["mean_p"]):
                hits += res["mean_p"] < 0.05
                slopes.append(abs(res["mean_slope"]))
                ps.append(res["mean_p"])
        rows.append({"n_donors": n_donors, "power": hits / n_reps,
                     "mean_abs_slope": np.mean(slopes) if slopes else np.nan,
                     "median_p": np.median(ps) if ps else np.nan})
        print(f"{n_donors:<11}{hits}/{n_reps}{'':<17}"
              f"{rows[-1]['mean_abs_slope']:<15.5f}{rows[-1]['median_p']:.4f}")

    print("\n" + "=" * 74)
    print("Depth: does the effect survive realistic scHi-C sparsity?")
    print("=" * 74)
    print(f"\n{'contacts/cell':<16}{'comp_strength slope':<22}{'meth_sd slope'}")
    for depth in (500, 1_500, 5_000, 20_000):
        rng = np.random.default_rng(seed)
        ages = np.linspace(2, 27, 12)
        cells = simulate_cohort(genome, tissues, TISSUE, ages, rng, n_cells=n_cells,
                                hic=SCHiCParams(contacts_per_cell=depth))
        d = donor_summary(cells)
        cs = location_scale(d, "comp_strength")["mean_slope"]
        ms = location_scale(d, "meth_sd")["mean_slope"]
        print(f"{depth:<16,}{cs:<22.6f}{ms:.6f}")
    print("\n  meth_sd is an implicit negative control: RRBS depth is independent of")
    print("  scHi-C depth, so that column should be flat. Residual movement across")
    print("  rows is seed noise from a shared RNG stream, not a depth effect.")

    print("\n" + "=" * 74)
    print("Tier 4: reality discriminator")
    print("=" * 74)
    rng = np.random.default_rng(seed)
    ages = np.linspace(2, 27, 10)
    a = simulate_cohort(genome, tissues, TISSUE, ages, rng, n_cells=n_cells)
    b = simulate_cohort(genome, tissues, TISSUE, ages, rng, n_cells=n_cells,
                        hic=SCHiCParams(contacts_per_cell=1_200))
    feats = ["comp_strength", "insulation_mean", "ps_slope", "meth_mean",
             "meth_sd", "meth_entropy"]
    out = tier4_reality_discriminator(a, b, feats, seed=seed)
    print(f"\n  AUC = {out['auc']:.3f}  ({out['verdict']})")
    print("  top discriminating features:")
    for name, coef in out["top_features"]:
        print(f"    {name:<20}{coef:+.3f}")
    print("\n  Here 'real' is a held-out cohort at different depth, so the AUC measures")
    print("  depth sensitivity. Swap in Cayo/ONPRC-derived features and the same")
    print("  number becomes a genuine sim-to-real gap, with the coefficients naming")
    print("  which axis of reality the simulator gets wrong.")


if __name__ == "__main__":
    main()
