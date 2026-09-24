"""H1: does tissue-specific conformation predict driver acquisition order?

Evaluation is LEAVE-ONE-TISSUE-OUT. Leave-one-gene-out would let a universal
gene-level prior (TP53 is usually late) masquerade as a conformation effect.

The nulls are the experiment. N1 (tissue-swap) is the one that matters: if
tissue-matched conformation does not beat tissue-swapped conformation, the
"systematically by tissue" claim is dead regardless of absolute tau.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..engine.conformation import ConformationParams, contact_map
from ..engine.epigenome import EpigenomeParams, compartment_affinity, initialise
from ..genome.scaffold import DRIVER_SEQUENCES, replication_timing
from ..inference.features import compartment_eigenvector, insulation_score

CONFORMATION_FEATURES = ["pc1", "insulation", "rep_timing", "enhancer", "expression", "ctcf_local"]
COVARIATES = ["gene_length", "bg_mutation_rate"]


def tissue_locus_features(genome, tissue, rng, n_cells: int = 8,
                          ground_truth: bool = False, hic=None) -> pd.DataFrame:
    """Per-bin features from NORMAL, YOUNG pseudobulk of one tissue."""
    from ..assay.virtual import SCHiCParams, schic

    epi, conf = EpigenomeParams(), ConformationParams()
    hic = hic or SCHiCParams()
    state = initialise(n_cells, tissue, rng)
    aff = compartment_affinity(state, normalize=False)
    rt = replication_timing(tissue)

    pc1 = np.zeros(genome.n_bins)
    ins = np.full(genome.n_bins, np.nan)
    for c in range(genome.n_chrom):
        sl = genome.chrom_slice(c)
        cs, co = genome.ctcf_strength[sl], genome.ctcf_orient[sl]
        bnd = np.clip(cs * tissue.ctcf_mod[sl], 0, 1)
        stack = []
        for k in range(n_cells):
            truth = contact_map(aff[k][sl], bnd, cs, co, conf, rng)
            stack.append(truth if ground_truth else schic(truth, hic, rng))
        pseudobulk = np.mean(stack, axis=0)
        pc1[sl] = compartment_eigenvector(pseudobulk, orient_by=tissue.a_prior[sl])
        ins[sl] = insulation_score(pseudobulk)

    ctcf_local = np.convolve(genome.ctcf_strength, np.ones(9) / 9, mode="same")
    return pd.DataFrame({
        "bin": np.arange(genome.n_bins), "tissue": tissue.name,
        "pc1": pc1, "insulation": ins, "rep_timing": rt,
        "enhancer": tissue.enhancer, "expression": tissue.expression,
        "ctcf_local": ctcf_local,
    })


@dataclass
class TruthWeights:
    """Generative weights for simulated driver order.

    w_selection > 0  : open/A-compartment loci are selected earlier
    w_supply    > 0  : LATE-replicating loci mutate more, so are acquired earlier
    The two partly oppose, which is the decomposition H1 is built to fit.
    """
    w_selection: float = 1.0
    w_supply: float = 0.6
    w_enhancer: float = 0.4
    noise: float = 0.55


def build_driver_table(genome, tissues, rng, weights: TruthWeights | None = None,
                       mode: str = "simulated", n_cells: int = 8,
                       ground_truth: bool = False) -> pd.DataFrame:
    """One row per (driver, tissue) with observed features and a rank label."""
    weights = weights or TruthWeights()
    gene_length = {g: rng.lognormal(0, 0.5) for g in genome.gene_bin}
    bg_mut = {g: rng.gamma(2, 0.5) for g in genome.gene_bin}

    feats = pd.concat([tissue_locus_features(genome, tissues[t], rng, n_cells, ground_truth)
                       for t in DRIVER_SEQUENCES], ignore_index=True)

    rows = []
    for t, seq in DRIVER_SEQUENCES.items():
        ft = feats[feats.tissue == t].set_index("bin")
        tis = tissues[t]
        rt = replication_timing(tis)
        for g in seq:
            b = genome.gene_bin[g]
            r = ft.loc[b]
            rows.append({
                "tissue": t, "gene": g, "bin": b,
                "pc1": r.pc1, "insulation": r.insulation, "rep_timing": r.rep_timing,
                "enhancer": r.enhancer, "expression": r.expression,
                "ctcf_local": r.ctcf_local,
                "gene_length": gene_length[g], "bg_mutation_rate": bg_mut[g],
                "_latent": (weights.w_selection * tis.a_prior[b]
                            + weights.w_supply * (-rt[b])
                            + weights.w_enhancer * tis.enhancer[b]
                            + rng.normal(0, weights.noise)),
                "_curated_rank": seq.index(g),
            })
    df = pd.DataFrame(rows)

    if mode == "simulated":
        # Earliest driver = highest latent acquisition propensity.
        df["rank"] = df.groupby("tissue")["_latent"].rank(ascending=False) - 1
    elif mode == "curated":
        df["rank"] = df["_curated_rank"]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    df["rank_z"] = df.groupby("tissue")["rank"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    return df


def _loto_tau(df: pd.DataFrame, features: list[str], alpha: float = 1.0) -> float:
    """Leave-one-tissue-out Kendall tau, pooled over held-out tissues."""
    taus, weights = [], []
    for t in df.tissue.unique():
        tr, te = df[df.tissue != t], df[df.tissue == t]
        if len(te) < 3:
            continue
        # Impute from TRAINING medians only -- imputing from the full table
        # would leak held-out tissue information into the fit.
        med = tr[features].median()
        Xtr, Xte = tr[features].fillna(med), te[features].fillna(med)
        if not np.isfinite(Xtr.to_numpy(dtype=float)).all():
            continue
        sc = StandardScaler().fit(Xtr)
        model = Ridge(alpha=alpha).fit(sc.transform(Xtr), tr["rank_z"])
        pred = model.predict(sc.transform(Xte))
        tau = kendalltau(pred, te["rank_z"]).statistic
        if np.isfinite(tau):
            taus.append(tau)
            weights.append(len(te))
    return float(np.average(taus, weights=weights)) if taus else np.nan


def _gene_prior_tau(df: pd.DataFrame) -> float:
    """N2: predict a gene's rank in the held-out tissue from its mean rank
    elsewhere. Uses no tissue information at all."""
    taus, weights = [], []
    for t in df.tissue.unique():
        tr, te = df[df.tissue != t], df[df.tissue == t]
        prior = tr.groupby("gene")["rank_z"].mean()
        pred = te["gene"].map(prior)
        ok = pred.notna()
        if ok.sum() < 3:
            continue
        tau = kendalltau(pred[ok], te["rank_z"][ok]).statistic
        if np.isfinite(tau):
            taus.append(tau)
            weights.append(int(ok.sum()))
    return float(np.average(taus, weights=weights)) if taus else np.nan


def evaluate(df: pd.DataFrame, rng: np.random.Generator, n_perm: int = 400) -> pd.DataFrame:
    """Full model plus every null in DESIGN.md §2."""
    full = CONFORMATION_FEATURES + COVARIATES
    results = [
        {"model": "full (conformation + covariates)", "tau": _loto_tau(df, full)},
        {"model": "N3 expression-only", "tau": _loto_tau(df, ["expression"])},
        {"model": "N4 supply-only (rep timing + mut rate)",
         "tau": _loto_tau(df, ["rep_timing", "bg_mutation_rate"])},
        {"model": "N2 universal gene prior (no tissue)", "tau": _gene_prior_tau(df)},
    ]

    # N1 tissue-swap: keep each gene's rank, give it another tissue's conformation.
    swap_taus = []
    for _ in range(n_perm // 8):
        s = df.copy()
        tissues = list(s.tissue.unique())
        for g, grp in s.groupby("gene"):
            if len(grp) < 2:
                continue
            perm = rng.permutation(grp.index.to_numpy())
            s.loc[grp.index, CONFORMATION_FEATURES] = df.loc[perm, CONFORMATION_FEATURES].to_numpy()
        swap_taus.append(_loto_tau(s, full))
    swap_taus = np.array([t for t in swap_taus if np.isfinite(t)])
    results.append({"model": "N1 tissue-swap (null)", "tau": float(np.mean(swap_taus)),
                    "tau_sd": float(np.std(swap_taus))})

    # Label permutation within tissue.
    perm_taus = []
    for _ in range(n_perm // 8):
        s = df.copy()
        s["rank_z"] = s.groupby("tissue")["rank_z"].transform(
            lambda v: rng.permutation(v.to_numpy()))
        perm_taus.append(_loto_tau(s, full))
    perm_taus = np.array([t for t in perm_taus if np.isfinite(t)])
    results.append({"model": "label permutation (null)", "tau": float(np.mean(perm_taus)),
                    "tau_sd": float(np.std(perm_taus))})

    out = pd.DataFrame(results)
    full_tau = out.loc[0, "tau"]
    out["delta_vs_tissue_swap"] = out["tau"] - out.loc[
        out.model.str.startswith("N1"), "tau"].iloc[0]
    if swap_taus.size:
        out.attrs["p_vs_tissue_swap"] = float((swap_taus >= full_tau).mean())
    if perm_taus.size:
        out.attrs["p_vs_permutation"] = float((perm_taus >= full_tau).mean())
    return out
