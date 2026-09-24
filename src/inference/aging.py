"""H2: location-scale modelling of chromatin aging.

The primary hypothesis is about the SCALE parameter, not the mean: cells drift
apart with age faster than they drift. Fitting only a mean model is the usual
way that result gets missed.

The statistical unit is the donor. `cell_level_naive` exists to demonstrate what
happens when you forget that -- it is a cautionary baseline, not an analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURES = ["comp_strength", "comp_fidelity", "insulation_mean", "insulation_sd",
            "ps_slope", "meth_mean", "meth_sd", "meth_entropy"]


def donor_summary(cells: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Collapse cells to donors, keeping BOTH the location and the scale."""
    features = features or [f for f in FEATURES if f in cells.columns]
    keys = ["donor_id", "age", "sex", "batch", "tissue"]
    g = cells.groupby(keys, as_index=False)
    out = g[features].mean().rename(columns={f: f"{f}_mu" for f in features})
    sd = g[features].std().rename(columns={f: f"{f}_sd" for f in features})
    out = out.merge(sd, on=keys)
    out["n_cells"] = g.size()["size"].values
    out["senescent_frac"] = g["senescent"].mean()["senescent"].values
    return out


def _fit(y: np.ndarray, X: pd.DataFrame) -> tuple[float, float, float]:
    ok = np.isfinite(y) & np.isfinite(X.to_numpy()).all(axis=1)
    if ok.sum() < X.shape[1] + 2:
        return np.nan, np.nan, np.nan
    m = sm.OLS(y[ok], sm.add_constant(X[ok], has_constant="add")).fit()
    return float(m.params["age"]), float(m.bse["age"]), float(m.pvalues["age"])


def location_scale(
    donors: pd.DataFrame, feature: str, adjust: tuple[str, ...] = ("sex", "batch")
) -> dict:
    """Age effect on the mean AND on the log dispersion of `feature`."""
    X = pd.DataFrame({"age": donors["age"].to_numpy(dtype=float)})
    for c in adjust:
        if c in donors and donors[c].nunique() > 1:
            d = pd.get_dummies(donors[c].astype(str), prefix=c, drop_first=True)
            X = pd.concat([X, d.astype(float).reset_index(drop=True)], axis=1)

    mu_b, mu_se, mu_p = _fit(donors[f"{feature}_mu"].to_numpy(dtype=float), X)
    sd = donors[f"{feature}_sd"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sd = np.log(np.where(sd > 0, sd, np.nan))
    sc_b, sc_se, sc_p = _fit(log_sd, X)

    return {"feature": feature,
            "mean_slope": mu_b, "mean_se": mu_se, "mean_p": mu_p,
            "scale_slope": sc_b, "scale_se": sc_se, "scale_p": sc_p,
            "scale_pct_per_year": 100 * (np.exp(sc_b) - 1) if np.isfinite(sc_b) else np.nan}


def run(donors: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    feats = features or [c[:-3] for c in donors.columns if c.endswith("_mu")]
    return pd.DataFrame([location_scale(donors, f) for f in feats])


def cell_level_naive(cells: pd.DataFrame, feature: str) -> dict:
    """WRONG ON PURPOSE: treats cells as independent. Reported to show the
    p-value inflation that pseudo-replication produces."""
    X = pd.DataFrame({"age": cells["age"].to_numpy(dtype=float)})
    b, se, p = _fit(cells[feature].to_numpy(dtype=float), X)
    return {"feature": feature, "slope": b, "se": se, "p": p, "n": len(cells)}
