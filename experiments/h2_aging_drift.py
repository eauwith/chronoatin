"""E2 -- H2: how does chromatin change with age, in the absence of cancer?

Primary hypothesis is H2b: cells drift APART with age faster than they drift.
Age is fitted on both the location and the scale.

Also reports:
  * the pseudo-replication trap (cell-level vs donor-level inference)
  * the Assay Attenuation Coefficient (Tier 5)

    python experiments/h2_aging_drift.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.ladder import tier5_attenuation
from src.cohort.donors import simulate_cohort
from src.genome.scaffold import build_genome, build_tissues
from src.inference.aging import cell_level_naive, donor_summary, run

TISSUE = "blood"
AGES = np.array([2., 4., 6., 9., 12., 15., 18., 21., 24., 27.])


def main(seed: int = 11, n_cells: int = 14) -> None:
    genome = build_genome()
    tissues = build_tissues(genome)

    print("=" * 74)
    print(f"E2  H2: chromatin aging in {TISSUE}, {len(AGES)} donors x {n_cells} cells")
    print("=" * 74)

    rng = np.random.default_rng(seed)
    cells = simulate_cohort(genome, tissues, TISSUE, AGES, rng, n_cells=n_cells)
    donors = donor_summary(cells)

    print("\n--- location-scale model: age on the mean AND on the log dispersion ---")
    res = run(donors)
    show = res[["feature", "mean_slope", "mean_p", "scale_slope", "scale_p",
                "scale_pct_per_year"]]
    print(show.round(4).to_string(index=False))

    sig_mean = (res.mean_p < 0.05).sum()
    sig_scale = (res.scale_p < 0.05).sum()
    print(f"\n  features with significant MEAN shift : {sig_mean}/{len(res)}")
    print(f"  features with significant SCALE shift: {sig_scale}/{len(res)}")
    print("  H2b predicts the scale column carries more signal than the mean column.")

    print("\n--- the pseudo-replication trap ---")
    print("  Cells are not independent. Treating them as such is the most common")
    print("  way single-cell aging findings fail to replicate.")
    print(f"  {'feature':<18}{'donor-level p':<16}{'cell-level p':<16}{'inflation'}")
    for f in ["comp_strength", "meth_entropy", "insulation_sd"]:
        dp = float(res.loc[res.feature == f, "mean_p"].iloc[0])
        cp = cell_level_naive(cells, f)["p"]
        ratio = dp / cp if cp > 0 else np.inf
        print(f"  {f:<18}{dp:<16.4g}{cp:<16.4g}{ratio:,.0f}x")

    print("\n--- Tier 5: Assay Attenuation Coefficient ---")
    rng = np.random.default_rng(seed)
    truth_cells = simulate_cohort(genome, tissues, TISSUE, AGES, rng,
                                  n_cells=n_cells, ground_truth=True)
    truth_res = run(donor_summary(truth_cells))
    print(f"  {'feature':<18}{'truth slope':<15}{'assayed slope':<16}{'AAC'}")
    rows = []
    for f in truth_res.feature:
        if f not in set(res.feature):
            continue
        for col, se_col, label in (("mean_slope", "mean_se", "mean"),
                                   ("scale_slope", "scale_se", "scale")):
            t = float(truth_res.loc[truth_res.feature == f, col].iloc[0])
            se = float(truth_res.loc[truth_res.feature == f, se_col].iloc[0])
            a = float(res.loc[res.feature == f, col].iloc[0])
            rows.append((f, label, t, a, tier5_attenuation(t, a, truth_se=se)))
    for f, label, t, a, aac in rows:
        if label == "scale":
            continue
        shown = f"{aac:+.2f}" if np.isfinite(aac) else "n/a (no truth effect)"
        print(f"  {f:<18}{t:<15.5f}{a:<16.5f}{shown}")
    print("\n  AAC 0 = measurement destroys nothing, 1 = destroys the whole effect,")
    print("  >1 = the measurement REVERSES the sign. Any |AAC| > 1 is a warning that")
    print("  the estimator is not measuring what it claims at this depth.")

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "h2_location_scale.csv", index=False)
    donors.to_csv(out / "h2_donor_summary.csv", index=False)
    print(f"\n  wrote {out/'h2_location_scale.csv'}")


if __name__ == "__main__":
    main()
