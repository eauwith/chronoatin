"""E1 -- H1: does tissue-specific conformation predict driver acquisition order?

Reports the tissue-swap null FIRST, because it is the null that can kill the
hypothesis. Also sweeps tissue distinctness to show where the test has power.

    python experiments/h1_driver_order.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.genome.scaffold import build_genome, build_tissues
from src.inference.driver_order import build_driver_table, evaluate


def main(seed: int = 3) -> None:
    genome = build_genome()

    print("=" * 74)
    print("E1  H1: tissue conformation -> driver acquisition order")
    print("=" * 74)

    tissues = build_tissues(genome)

    # A single seed is not a result at n=25 pairs. Repeat it.
    taus, swaps, p_swap, p_perm = [], [], [], []
    for k in range(8):
        rng = np.random.default_rng(seed + k)
        table = build_driver_table(genome, tissues, rng, n_cells=8)
        res = evaluate(table, rng, n_perm=240)
        taus.append(res.loc[0, "tau"])
        swaps.append(res.loc[res.model.str.startswith("N1"), "tau"].iloc[0])
        p_swap.append(res.attrs["p_vs_tissue_swap"])
        p_perm.append(res.attrs["p_vs_permutation"])

    print(f"\n{len(table)} (driver, tissue) pairs, {table.tissue.nunique()} tissues, "
          f"leave-one-tissue-out CV, 8 seeds\n")
    print("  last-seed breakdown:")
    print(res.round(3).to_string(index=False))

    taus, swaps = np.array(taus), np.array(swaps)
    p_swap, p_perm = np.array(p_swap), np.array(p_perm)
    print(f"\n  across 8 seeds:")
    print(f"    tau full        : {taus.mean():.3f} +/- {taus.std():.3f} "
          f"[{taus.min():.3f}, {taus.max():.3f}]")
    print(f"    tau tissue-swap : {swaps.mean():.3f} +/- {swaps.std():.3f}")
    print(f"    p vs tissue-swap: median {np.median(p_swap):.3f}, "
          f"significant in {(p_swap < 0.05).sum()}/8 seeds   <-- the decisive test")
    print(f"    p vs label perm : median {np.median(p_perm):.3f}, "
          f"significant in {(p_perm < 0.05).sum()}/8 seeds")

    if (p_swap < 0.05).sum() < 7:
        print("\n  VERDICT: the result is SEED-UNSTABLE at this panel size. The model")
        print("  reliably beats label permutation -- it has learned something -- but")
        print("  whether it beats tissue-swap depends on the draw. Reporting a single")
        print("  seed here would be reporting noise in either direction.")

    print("\n" + "-" * 74)
    print("Power: where does this test have any ability to detect tissue-specificity?")
    print("-" * 74)
    print(f"{'shared_frac':<13}{'r(tissues)':<12}{'tau_full':<11}{'tau_swap':<11}{'p_swap'}")
    for sf in (0.85, 0.55, 0.25, 0.05):
        rng = np.random.default_rng(seed)
        ts = build_tissues(genome, shared_fraction=sf)
        r = np.corrcoef(ts["lung"].a_prior, ts["liver"].a_prior)[0, 1]
        t = build_driver_table(genome, ts, rng, n_cells=6)
        e = evaluate(t, rng, n_perm=240)
        tf = e.loc[0, "tau"]
        tsw = e.loc[e.model.str.startswith("N1"), "tau"].iloc[0]
        print(f"{sf:<13.2f}{r:<12.2f}{tf:<11.3f}{tsw:<11.3f}{e.attrs['p_vs_tissue_swap']:.3f}")
    print("\n  Real tissues share a lot of compartment structure (r ~ 0.6-0.8).")
    print("  At 25 driver-tissue pairs this test only has power when r < ~0.3,")
    print("  so H1 needs a larger panel (PCAWG timing across more cancer types)")
    print("  or locus-level rather than gene-level resolution.")


if __name__ == "__main__":
    main()
