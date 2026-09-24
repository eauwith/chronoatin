# Chromatin

An in-silico experimental environment for single-cell chromatin over the lifespan.

It exists to answer two questions that are hard to answer at the bench, and one that
cannot be answered at the bench at all:

1. **H1** — does chromatin conformation vary systematically by tissue in a way that
   predicts the order in which cancer drivers are acquired?
2. **H2** — in the absence of cancer, how does chromatin change with age? Specifically:
   do cells drift *apart* faster than they drift?
3. **H3 / E5** — how much of a true epigenetic aging effect survives realistic
   single-cell measurement, and how many donors does detecting it actually need?

The third is the one that justifies the project. See [DESIGN.md](DESIGN.md) for the full
scientific design, the sim-to-real limits, and the benchmark ladder.

## Install and run

```bash
pip install -r requirements.txt
python experiments/run_all.py
```

Individual experiments:

```bash
python experiments/h1_driver_order.py      # H1 + power sweep over tissue distinctness
python experiments/h2_aging_drift.py       # H2 location-scale + AAC + pseudo-replication
python experiments/h3_power_attenuation.py # donor-count and depth requirements
```

Everything runs on synthetic data with no downloads. Real-data adapters are described
in DESIGN.md §8 and match schemas already in use in `Methylation Aging`.

## What it does

```
genome/      1 kb windows keyed like selected_percent_methylation.csv, CTCF sites
engine/      epigenome state -> conformation, advanced by cell division
assay/       virtual scHi-C and RRBS -- sparsity, doublets, dropout, batch
cohort/      donors with random effects and a dial-able age-batch confound
inference/   compartment/insulation callers, location-scale age models, driver-order
benchmarks/  the validation ladder, Tier 0-5
experiments/ H1, H2, E5 as reproducible scripts
```

**Nothing analyses ground truth except the attenuation benchmark.** Every measurement
passes through `assay/` first, because the gap between the two is the result.

Aging enters the model in exactly three places — heterochromatin maintenance decay,
lamina tethering decay, and a constant per-division epimutation rate — and nowhere else.
That constraint is what makes it falsifiable rather than merely flexible.

## Findings from the synthetic core

These are statements about **estimators**, not about chromatin. That distinction is the
whole methodological point (DESIGN.md §6).

- **A constant per-division epimutation rate is sufficient to produce rising
  cell-to-cell heterogeneity with age** (SD 0.039 → 0.106 over 1600 divisions) with
  almost no shift in the mean. No age-specific programme is required to generate the
  variance signal.
- **Compartment amplitude erodes while the compartment *pattern* is preserved**
  (strength 0.58 → 0.31; r with tissue prior stays ≈ 0.92). Aging blurs A/B rather
  than scrambling it.
- **H1 is seed-unstable at 25 driver–tissue pairs.** τ_full = 0.485 ± 0.128 vs
  τ_tissue-swap = 0.305 ± 0.097; beats label permutation in 7/8 seeds but beats the
  tissue-swap null in only 5/8. The test only has reliable power when tissues are
  genuinely distinct (r < ~0.3), and real tissues share far more than that. H1 as
  stated needs a larger panel or locus-level resolution.
- **Methylation-based aging signals survive realistic assay noise; conformation-based
  ones do not.** Assay Attenuation Coefficients: `meth_mean` 0.07, `meth_sd` 0.53,
  versus `comp_strength` 1.23 — above 1, meaning the measurement *reverses the sign*
  of the true effect at scHi-C depth.
- **Pseudo-replication inflates p-values by ~10⁵.** Methylation entropy vs age:
  p = 0.23 at the donor level, p = 2.4 × 10⁻⁶ treating cells as independent.
- **Donor-count requirement: ~16.** Detecting the age effect on `meth_sd` at 10
  cells/donor: 0/4 runs at 6 donors, 2/4 at 10, 4/4 at 16, 4/4 at 24. The slope
  estimate stays flat (~0.00065) across all four — the estimator is unbiased, only
  the standard error moves, which is the internal check that this is a power curve
  and not a bias curve.
- **P(s) slope is the most depth-sensitive statistic there is.** The Tier 4
  discriminator separates two cohorts differing *only* in sequencing depth at
  AUC 0.855, and `ps_slope` carries a coefficient of +1.51 against −0.59 for the
  next feature. Practical consequence: **never compare simulated to real chromatin
  on P(s) slope without depth-matching first** — it will dominate the comparison
  and report a sim-to-real gap that is really a depth difference.
- **Compartment strength does not converge with depth** (slope −0.00017 at 500
  contacts/cell, then +0.000068 → +0.000030 from 1.5k to 20k). It sign-flips at low
  depth and decays rather than stabilising — the same failure the AAC of 1.23 reports,
  seen from a second direction.

## Guards

`run_all.py` refuses to proceed past Tier 1. A simulator that does not produce a P(s)
slope in [−1.5, −0.8], a PC1 correlated with its own compartment prior, and insulation
minima at CTCF sites is not a chromatin model, whatever else it reproduces.

Tier 0 checks determinism under a fixed seed — added after a real bug in which
replication-timing features were derived from Python's `hash()`, which is salted per
process, making the simulator silently non-reproducible across runs.
