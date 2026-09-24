# Chronatin — design document

An in-silico experimental environment for testing whether chromatin conformation
varies systematically by tissue in a way that predicts cancer driver acquisition
order, and for modelling how chromatin drifts with age in the absence of cancer.

Built to sit on top of existing work rather than beside it:

| Existing asset | Role here |
|---|---|
| `Methylation Aging` (Cayo blood RRBS 572×216 longitudinal, ONPRC liver 94) | real-data anchor + calibration target |
| `Origin-Predict` (`LOSOHarness`, `NestedCVHarness`, `MetricsEvaluator`) | CV infrastructure, reused not rebuilt |
| `Features-of-Carcinogenesis` (per-tissue driver sequences, Hi-C panel) | driver-order labels + visual lineage |
| CHIP preprint (clonal dynamics under aging/inflammation) | clonal module of the lifespan engine |
| CEDAR / mutational signatures | the supply-vs-selection decomposition in H1 |

---

## 1. Premise

Two literatures point in opposite directions and are rarely tested against each other
in the same model.

**Mutation supply.** Closed, late-replicating, B-compartment, lamina-associated DNA
accumulates somatic mutations *faster* — reduced mismatch-repair access, differential
nucleotide-excision repair. Chromatin organisation of the cell-of-origin predicts a
large fraction of regional cancer mutation density.

**Selection.** A driver only confers fitness if the locus is transcriptionally live in
that lineage — open, A-compartment, enhancer-contacted. A mutation in a gene the tissue
never expresses is invisible to selection.

So the naive hypothesis "open chromatin → early driver" and its negation
"closed chromatin → early driver" are *both* defensible, because they are statements
about different terms. The interesting, testable claim is the decomposition:

> observed driver rank ≈ f(mutation supply | conformation) ⊗ g(selection coefficient | conformation)

Chronatin exists to fit that decomposition per tissue, with the two terms separately
identified, and to ask whether the tissue-matched conformation carries information the
tissue-swapped conformation does not.

---

## 2. Hypotheses, stated so they can fail

### H1 — tissue conformation predicts driver acquisition order

*For driver gene d in tissue t, the acquisition rank r(d,t) is predicted by features of
the locus measured in **normal, young** cells of tissue t, and the tissue-matched
features outperform tissue-swapped features.*

- **Unit of analysis:** (driver, tissue) pair. **Labels:** acquisition rank, from the
  curated sequences in `Features-of-Carcinogenesis`, upgraded where possible to PCAWG
  evolutionary-timing calls (clonal-early / clonal-late / subclonal).
- **Features:** compartment eigenvalue, insulation at locus, replication timing, ATAC,
  LAD status, enhancer-contact count, CTCF loop anchoring — plus non-conformation
  covariates that must be controlled: gene length, trinucleotide-context mutation rate,
  expression level, essentiality.
- **Primary statistic:** Kendall τ between predicted and observed rank, under
  **leave-one-tissue-out** CV. Leave-one-tissue-out is non-negotiable: leave-one-gene-out
  would let a universal gene-level prior masquerade as a conformation effect.
- **Effect size:** Δτ vs null, bootstrap CI over (driver, tissue) pairs.

**The nulls are the experiment.** In rank order of how much they matter:

| Null | What it kills if it wins |
|---|---|
| **N1 tissue-swap** — same conformation features, tissue labels permuted | the entire "systematically by tissue" claim |
| N2 universal driver-frequency prior, no tissue information | tissue-specificity |
| N3 expression-only | conformation adds nothing beyond transcription |
| N4 supply-only (replication timing + mutation rate) | selection term is unnecessary |
| N5 matched non-drivers — genes matched on expression, replication timing, GC, length, mutation rate | you are measuring gene-ness, not driver-ness |

N1 is the one to report first. If tissue-matched conformation does not beat
tissue-swapped conformation, H1 is dead regardless of absolute τ, and no amount of
model capacity rescues it.

### H2 — chromatin drifts with age, and drifts *apart* faster than it drifts

Aging chromatin is not a single monotone loss. At least four documented processes run
concurrently, two of them in opposing directions:

1. heterochromatin loss — H3K9me3 decline, LAD detachment, lamin B1 decline
2. compartment blurring — A/B eigenvalue amplitude compression
3. TAD boundary weakening — insulation decay, CTCF occupancy loss
4. *against* the above: senescence-associated heterochromatin foci (SAHF) — focal *gain*
   of compaction in senescent cells, which accumulate with age

And at the methylation layer, the familiar two-directional drift: global hypomethylation
of repeats alongside focal CpG-island hypermethylation.

**H2a (mean):** compartment strength, insulation, and P(s) slope shift monotonically with age.
**H2b (variance) — the primary hypothesis:** *cell-to-cell dispersion increases with age
faster than any mean shifts*, i.e. age enters the scale parameter more strongly than the
location parameter.

H2b is the one to lead with, because it is the direct single-cell extension of
`Age and early life adversity shape heterogeneity of the epigenome across tissues`
(Science 2026) and of the aging score already in `Methylation Aging`, which is
literally `Variability_std − Mean_std − Median_std` — variability with a positive sign,
the means subtracted off.

**Model:** heteroscedastic / location-scale mixed model. Cells nested in donors, donor
random intercept, age as fixed effect on *both* μ and log σ. Adjust for sex, batch,
`prepost2017`, mapped reads, and cell-type composition. The statistical unit is the
**donor**, never the cell — cells are pseudo-replicates, and treating them as
independent is the single most common way single-cell aging results evaporate.

### H2→H1 bridge — the hypothesis worth the project

*Does age-related boundary decay occur preferentially at loci that are early drivers in
that same tissue?*

If yes, aging does not merely permit cancer — it pre-opens the tissue-specific route, and
driver order is partly a readout of the order in which age dismantles insulation. This is
a directional, pre-registerable test: correlate per-locus age×insulation slope in tissue t
against driver earliness rank in tissue t. It is the claim that makes this gerontology
rather than cancer genomics with an age covariate.

### H3 — simulability

*A mechanistic simulator calibrated on young cells only, run forward, reproduces aged
cells.* Calibrate on the young half of the age distribution; predict the old half; report
posterior predictive checks, not point fits. Where it fails, name the axis — that failure
is a finding, not a bug (§6).

---

## 3. Architecture

Nine layers. The ordering matters: **nothing analyses ground truth except the
attenuation benchmark.** Every experiment runs through the virtual assay.

```
genome/      binned scaffold — 1 kb windows keyed exactly like
             selected_percent_methylation.csv (chr4_56044000_56045000),
             CTCF sites with orientation, genes, replication timing
engine/
  epigenome  per-cell per-bin state: H3K9me3, H3K27me3, H3K27ac, CpG methylation,
             lamina association. Division-coupled: replication dilution →
             reader-writer spreading → boundary-limited propagation → epimutation
  conform    loop extrusion (LEF, CTCF-blocked, convergent rule) + block-copolymer
             compartment attraction → per-cell contact map
  lifespan   divisions over the lifespan, epimutation accumulation, lamin B1 decline,
             senescence entry with SAHF, clonal dynamics (CHIP module)
assay/       VIRTUAL ASSAY — scHi-C, scATAC, snm3C, RRBS. Contact budget,
             distance-dependent capture, random ligation, PCR duplicates, doublets,
             ambient contamination, dropout, batch effects, PMI degradation
cohort/      donor generator — ages, sex, tissue, batch, composition shift, and a
             dial-able age↔batch confound from ρ=0 to ρ=1
inference/   compartment calling, insulation, per-cell scores, location-scale age
             models, driver-order models
benchmarks/  the validation ladder (§7), each with a pass/fail threshold
experiments/ H1, H2, H3 as reproducible scripts
viz/         three.js lifespan-slider page, continuing the genome-3d /
             genome-under-cancer lineage
```

**The critical coupling loop**, and the reason this is a model rather than a plot:

```
epigenome state → compartment affinity + boundary strength
               → conformation
               → accessibility + replication timing
               → mutation supply + expression
               → selection on drivers
               → clonal composition
               → back to epigenome state
```

Aging enters at exactly three places, and nowhere else — this is what makes the model
falsifiable rather than merely flexible:

1. heterochromatin maintenance efficiency (spreading coefficient) decays with divisions
2. lamina tethering probability decays (lamin B1 decline)
3. per-CpG epimutation rate is constant per division, so *variance* accumulates linearly
   in divisions while the mean barely moves

Note that (3) alone predicts H2b. If the simulator reproduces increasing heterogeneity
with age using only a constant per-division epimutation rate and no age-specific
machinery, that is a real result: it says age-related epigenetic heterogeneity needs no
programme, only mitotic time.

---

## 4. Single-cell: what the bench cannot do

Ranked by how badly each one damages an aging study specifically.

1. **Cross-sectional only, and destructive.** You cannot watch a cell age. Every
   single-cell dataset is a snapshot; trajectories are *inferred*. For gerontology this is
   the central obstruction — pseudotime is not time, and an ordering derived from an
   aging axis cannot then be used as evidence about that axis without circularity.
2. **Donors are the statistical unit, and N is small.** 20,000 cells from 6 donors is
   n=6. Power comes from donors, not cells. Most naive single-cell aging findings are
   donor-level batch effects with cell-level error bars.
3. **Age is never randomised, and is confounded with everything.** Post-mortem interval,
   ischemic time, cause of death, collection era, storage duration. In GTEx-like material,
   PMI correlates with age *and* independently degrades chromatin. Your own
   `prepost2017` column is this problem, already named.
4. **Composition shift masquerades as cell-intrinsic change.** Aged tissue differs partly
   because the cell-type mixture changed — immune infiltration, clonal expansion,
   senescent accumulation. A "tissue ages" result is frequently "tissue re-composes".
   Worse, annotation itself drifts with age, so the correction is entangled with the effect.
5. **Extreme sparsity, and a hard biological ceiling.** scHi-C yields ~10⁴–10⁶ contacts
   against ~10⁹ possible bin pairs. scATAC is near-binary because a diploid locus has two
   alleles — you cannot get a "high" count from one cell. Single-cell chromatin data is
   not noisy continuous data; it is sparse binary data, and methods that assume otherwise
   silently fabricate structure.
6. **Dissociation selects against exactly the cells you want.** Fragile, large, adherent
   and senescent cells are preferentially lost. In an aging study the dropout is
   correlated with the phenotype.
7. **Cell-cycle and copy-number confound conformation.** An S-phase cell has a different
   contact map and different copy number. Without a cell-cycle regressor, replication
   timing leaks into every compartment call.
8. **Modalities are not co-assayed without cost.** snm3C gives methylation+conformation,
   Paired-Tag gives histone+RNA, but each combination sacrifices depth, and the joint
   distribution you actually want is rarely measured in the same cell.
9. **No haplotype resolution.** Homologous chromosomes differ; unphased contacts are
   ambiguous, and allele-specific compartment differences average away.
10. **Cost caps the number of ages.** You get young-vs-old, not a time course. Two points
    cannot distinguish linear drift from a midlife transition — which your own hippocampus
    paper says is exactly where the signal is.

## 5. What the computational environment actually buys

Mapped one-to-one onto the list above, because the claim must be specific.

| Limitation | Computational remedy | What it does *not* fix |
|---|---|---|
| 1 cross-sectional | simulator is natively longitudinal; lineages are tracked | the sim's trajectory is your model's, not nature's |
| 2 low donor N | donors as an explicit random-effects level → **power curves**: how many donors do I need to detect Δinsulation of x? | does not create real donors |
| 3 confounding | dial ρ(age,batch) from 0→1 and measure when your estimator breaks | does not tell you real ρ |
| 4 composition | simulate composition shift explicitly; test whether the estimator recovers cell-intrinsic truth | real composition shifts are unknown |
| 5 sparsity | ground truth is dense; the **virtual assay** subsamples to realistic depth, so you can measure exactly how much of a finding survives | the real capture model is itself a hypothesis |
| 6 dissociation bias | make dropout phenotype-correlated on purpose and quantify the induced bias | real bias magnitude unmeasured |
| 7 cell cycle | cell-cycle phase is known ground truth; test regressors against it | — |
| 8 co-assay | all modalities in the same simulated cell, for free | the real cross-modality coupling is assumed |
| 9 phasing | haplotypes explicit | — |
| 10 few ages | dense age grids, arbitrarily many | — |

Remedy 2 and remedy 5 are the two that justify the project on their own.
A power curve for donor count, and a measured answer to *"does this effect survive
3,000 contacts per cell?"*, are deliverables a wet-lab collaborator can act on before
spending money.

---

## 6. What cannot be replicated — the honest limits

This section is the portfolio differentiator. Anyone can write a simulator; the
distinguishing skill is stating precisely what it licenses you to conclude.

1. **Unknown mechanism.** The simulator contains only the biology you put in. Any age
   effect driven by something absent — transposon reactivation, acetyl-CoA availability
   gating H3K27ac, ECM stiffening transmitted through the LINC complex — is invisible,
   and its absence will not announce itself. **Simulated data validates internal logic,
   never nature.**
2. **Parameter transfer.** Cohesin residence time, LEF processivity, per-CpG epimutation
   rate per division, tissue-specific stem-cell division rates in vivo — measured in few
   systems, usually transformed cell lines, usually mouse. Every one enters with large,
   usually unquantified uncertainty.
3. **Organismal context is absent entirely.** Immune surveillance, inflammaging, paracrine
   SASP, circadian and hormonal cycling, nutrient state, microbiome, tissue mechanics.
   A cell-autonomous simulator is a model of a cell in a dish that ages, which is not
   the thing gerontology is about. Your own work is about *early life adversity* and
   *environment* — precisely the axis a cell-autonomous model cannot represent.
4. **The real selection landscape.** Driver fitness in vivo depends on niche competition
   and immune editing. You can parameterise s, you cannot derive it.
5. **Coarse-graining erases scales.** Real chromatin is a crowded, active, partly
   phase-separated polymer at nucleosome resolution. Any tractable model is coarse-grained
   to 1 kb–100 kb beads, and some biology lives below that floor.
6. **The noise model is a hypothesis too.** Virtual scHi-C encodes what we *believe* the
   artifacts are. Unknown artifacts cannot be simulated, and the ones we do simulate may
   have the wrong functional form.
7. **Tails are untrustworthy.** Clonal sweeps and single-founder events are heavy-tailed
   and poorly constrained. The simulator gives you a distribution; believe its centre,
   not its 99th percentile.

The correct framing in any write-up: **the simulator is an instrument for measuring the
behaviour of estimators, not an oracle about chromatin.** Every claim should take the form
"under assumptions A, an estimator of this form recovers/fails to recover effect E at
depth D with N donors" — never "chromatin does X".

---

## 7. Benchmarks — measuring the interfering effect

A seven-tier ladder. Each tier is a runnable check with a threshold, in the spirit of the
`check-build.mjs` guards in `Genome-Under-A-Fast` — encoding this project's actual
constraints rather than generic testing.

**Tier 0 — internal consistency.** Polymer non-crossing, copy-number conservation, mark
mass balance, determinism under fixed seed. Fails = bug.

**Tier 1 — known physics.** Non-negotiable summary statistics that real Hi-C always shows:
P(s) decaying with slope ≈ −1 to −1.5 over 0.1–10 Mb; plaid compartment checkerboard with
PC1 explaining a sane variance fraction; TAD size distribution in the right decade;
insulation minima co-located with CTCF sites; loop dots at convergent CTCF pairs.
If the simulator fails Tier 1 it is not a chromatin model.

**Tier 2 — held-out real data.** Calibrate on young donors of tissue A → predict old
donors of tissue A. Then calibrate on tissue A → predict tissue B. Posterior predictive
checks. Your ONPRC liver and Cayo blood are two tissues with different sampling
structures, which makes this a genuine transfer test rather than a refit.

**Tier 3 — negative controls.** Permute age within donor. Tissue-swap (N1). Matched
non-driver gene sets (N5). A conformation-blind baseline. Report these *before* the
positive result, not in a supplement.

**Tier 4 — the reality discriminator.** Train a classifier to distinguish simulated cells
from real cells on summary features.
- AUC ≈ 0.5 → indistinguishable on the features tested
- AUC high → read the **feature importances**, which name exactly which axis of reality
  the simulator failed to reproduce

This converts "is my simulation realistic?" — unanswerable — into "on which measured
axis is it unrealistic, and by how much?" — answerable, and directly reportable.
It is the single most useful benchmark in the ladder, and it improves as you add real data.

**Tier 5 — Assay Attenuation Coefficient (AAC).** Run the identical analysis twice: once
on simulator ground truth, once after the virtual assay.

```
AAC = 1 − (effect size after virtual assay) / (effect size on ground truth)
```

AAC is a direct, numerical measure of how much of a true effect the measurement
technology destroys, as a function of contacts-per-cell, donor N, and dropout. Sweeping it
produces the practical deliverable: *this effect needs ≥ X contacts/cell and ≥ Y donors to
be detectable.* This is the quantitative answer to "to what extent does the interference
matter".

**Tier 6 — sensitivity and multiverse.** Sobol indices over simulator parameters: which
conclusions are parameter-fragile? Multiverse analysis over analysis choices (bin size,
normalisation, compartment caller, insulation window). A conclusion that survives only one
cell of the multiverse is not a conclusion.

**Tier 7 — prospective.** Pre-register a prediction the simulator makes that no existing
analysis has asked, then test it on data held out from calibration entirely.

---

## 8. Real-data integration

The synthetic core stays fully portable — no downloads, runs in CI, safe to publish.
Real data enters through adapters that match schemas already on disk.

**Cayo blood (RRBS, 572 samples / 216 animals, longitudinal).** The longitudinal
structure is the project's most valuable asset and the answer to limitation 1: repeat
sampling of the same individual gives *observed* within-individual drift, against which
simulated drift can be calibrated rather than assumed. Grouped splits by `animal_id`
(already done correctly in `Methylation Aging`). Covariates available and required:
`age_at_sample`, `sex`, `batch`, `prepost2017`, `mapped_reads`, `parous`; `residAge`
gives epigenetic age acceleration as a ready-made outcome.

**ONPRC liver (94 samples, 235,810 sites).** Second tissue for the cross-tissue transfer
test in Tier 2.

**The join key is already right.** Features are keyed as 1 kb genomic windows
(`chr4_56044000_56045000`), so methylation features map directly onto the binned genome
scaffold with no re-keying — the same bins carry compartment, insulation and replication
timing. This is why the model is methylation-anchored with conformation derived, rather
than the reverse.

**Public conformation data** for tissue-specific compartment/insulation features:
ENCODE and 4DN Hi-C, plus scATAC atlases. Human for H1 (driver labels are human);
macaque where an ortholog mapping exists, via the liftover already implemented in
`cpg_liftover.py`.

---

## 9. Experiment protocols

**E1 — H1 driver order.** Features from normal young pseudobulk per tissue → rank model →
leave-one-tissue-out CV → Kendall τ → nulls N1–N5 → donor-level permutation → BH FDR
across tissues → bootstrap CI on Δτ. Report N1 first.

**E2 — H2 aging drift.** Cohort across a dense age grid → virtual assay → compartment
strength, insulation, P(s) slope, per-cell entropy → location-scale mixed model with age
on both μ and log σ → report the σ coefficient as primary.

**E3 — H2→H1 bridge.** Per-locus age×insulation slope in tissue t vs driver earliness in
tissue t. Directional, pre-registered.

**E4 — H3 simulability.** Calibrate young → predict old → Tier 2 + Tier 4.

**E5 — power and attenuation.** Sweep donor N × contacts-per-cell × ρ(age,batch) → power
surfaces and AAC curves. This is the deliverable with the shortest path to being useful
to someone else.

---

## 10. Portfolio framing

The claim this project supports on a CV is narrow and defensible:

> Built an in-silico environment that measures how much of a true epigenetic aging effect
> survives realistic single-cell measurement, and used it to derive donor-count and
> sequencing-depth requirements for detecting tissue-specific chromatin aging.

That is a methods contribution, it is honest about what simulation can license, and it
sits exactly on the CEDAR axis — early detection is a question about *whether the signal
survives the assay*, which is what the AAC measures.

The three-panel interactive page (lifespan slider, per-cell heterogeneity fan, attenuation
curve) continues the `Genome-Under-A-Fast` / `Features-of-Carcinogenesis` line, driven by
real simulator output rather than interpolated constants.
