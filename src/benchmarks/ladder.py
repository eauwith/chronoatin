"""The validation ladder (DESIGN.md §7).

Each tier is a check with a threshold, in the spirit of the check-build.mjs
guards in Genome-Under-A-Fast: encode this project's actual constraints, not
generic testing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..inference.features import (
    compartment_eigenvector,
    compartment_strength,
    insulation_score,
    ps_slope,
)


@dataclass
class Check:
    tier: int
    name: str
    value: float
    expected: str
    passed: bool


def tier1_known_physics(truth: np.ndarray, boundary: np.ndarray,
                        a_prior: np.ndarray) -> list[Check]:
    """Summary statistics real Hi-C always shows. Failing these means it is not
    a chromatin model, whatever else it reproduces."""
    slope = ps_slope(truth)
    pc1 = compartment_eigenvector(truth, orient_by=a_prior)
    r_pc1 = float(np.corrcoef(pc1, a_prior)[0, 1])
    ins = insulation_score(truth)
    ok = np.isfinite(ins)
    r_ins = float(np.corrcoef(ins[ok], boundary[ok])[0, 1])
    checks = [
        Check(1, "P(s) slope", slope, "-1.5 to -0.8", -1.5 <= slope <= -0.8),
        Check(1, "PC1 vs compartment prior", r_pc1, ">= 0.6", r_pc1 >= 0.6),
        Check(1, "insulation vs CTCF (anticorrelated)", r_ins, "<= -0.10", r_ins <= -0.10),
        Check(1, "compartment strength > 0", compartment_strength(pc1), "> 0",
              compartment_strength(pc1) > 0),
    ]
    return checks


def tier4_reality_discriminator(
    sim: pd.DataFrame, real: pd.DataFrame, features: list[str], seed: int = 0
) -> dict:
    """Can a classifier tell simulated cells from real ones?

    AUC ~ 0.5  -> indistinguishable on the features tested.
    AUC high   -> read the coefficients: they name WHICH axis of reality the
                  simulator failed to reproduce. That is the whole point --
                  "is my simulation realistic?" is unanswerable, "on which axis
                  is it unrealistic, and by how much?" is not.

    `real` may be a held-out simulated cohort while no real data is attached;
    swap in Cayo/ONPRC-derived features to make it a genuine reality check.
    """
    X = pd.concat([sim[features], real[features]], ignore_index=True)
    y = np.r_[np.zeros(len(sim)), np.ones(len(real))]
    ok = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X, y = X[ok], y[ok]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return {"auc": np.nan, "top_features": []}

    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    model = LogisticRegression(max_iter=2000)
    prob = cross_val_predict(model, Xs, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))

    coef = LogisticRegression(max_iter=2000).fit(Xs, y).coef_[0]
    order = np.argsort(-np.abs(coef))
    top = [(features[i], float(coef[i])) for i in order[:5]]
    return {"auc": auc, "top_features": top,
             "verdict": "indistinguishable" if auc < 0.6 else
                        "distinguishable - inspect top_features"}


def tier5_attenuation(effect_truth: float, effect_assayed: float,
                      truth_se: float | None = None) -> float:
    """Assay Attenuation Coefficient.

    AAC = 1 - (effect after virtual assay) / (effect on ground truth)

    0   -> the measurement destroys nothing
    1   -> the measurement destroys the entire effect
    >1  -> the measurement reverses the sign of the effect

    Returns NaN when there is no ground-truth effect to attenuate. This guard
    matters: a ratio against a near-zero denominator produces spectacular
    meaningless numbers (|AAC| in the hundreds) that look like findings.
    """
    if not np.isfinite(effect_truth) or abs(effect_truth) < 1e-12:
        return np.nan
    if truth_se is not None and np.isfinite(truth_se) and abs(effect_truth) < 2 * truth_se:
        return np.nan  # truth effect indistinguishable from zero
    return float(1.0 - effect_assayed / effect_truth)


def report(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in checks])
