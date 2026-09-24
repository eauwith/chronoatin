"""Run the full environment end to end.

    python experiments/run_all.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.ladder import report, tier1_known_physics
from src.engine.conformation import ConformationParams, contact_map
from src.engine.epigenome import compartment_affinity, initialise
from src.genome.scaffold import build_genome, build_tissues

import h1_driver_order, h2_aging_drift, h3_power_attenuation  # noqa: E402


def tier0_determinism() -> bool:
    """Tier 0: the same seed must give the same answer, every process."""
    def once():
        g = build_genome()
        t = build_tissues(g)["lung"]
        rng = np.random.default_rng(42)
        sl = g.chrom_slice(0)
        s = initialise(3, t, rng)
        aff = compartment_affinity(s, normalize=False)[0][sl]
        bnd = np.clip(g.ctcf_strength[sl] * t.ctcf_mod[sl], 0, 1)
        return contact_map(aff, bnd, g.ctcf_strength[sl], g.ctcf_orient[sl],
                           ConformationParams(), rng).sum()
    return bool(np.isclose(once(), once()))


def main() -> None:
    print("=" * 74)
    print("TIER 0-1  simulator validity")
    print("=" * 74)
    print(f"\n  determinism under fixed seed: {'PASS' if tier0_determinism() else 'FAIL'}")

    g = build_genome()
    t = build_tissues(g)["lung"]
    rng = np.random.default_rng(0)
    sl = g.chrom_slice(0)
    s = initialise(6, t, rng)
    aff = compartment_affinity(s, normalize=False)[:, sl].mean(axis=0)
    bnd = np.clip(g.ctcf_strength[sl] * t.ctcf_mod[sl], 0, 1)
    truth = contact_map(aff, bnd, g.ctcf_strength[sl], g.ctcf_orient[sl],
                        ConformationParams(), rng)
    checks = report(tier1_known_physics(truth, bnd, t.a_prior[sl]))
    print()
    print(checks.to_string(index=False))
    if not checks.passed.all():
        print("\n  TIER 1 FAILED -- this is not a chromatin model. Stopping.")
        sys.exit(1)

    print()
    h1_driver_order.main()
    print()
    h2_aging_drift.main()
    print()
    h3_power_attenuation.main()


if __name__ == "__main__":
    main()
