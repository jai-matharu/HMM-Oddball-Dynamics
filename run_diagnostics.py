"""Remaining model diagnostics for the supplement: VIFs with and without
centring, QQ data, and the random-intercept variance check that motivates
cluster-robust OLS rather than a mixed model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

PER_STATE_METRICS = ["FO", "transition_into_sum", "interval_mean", "lifetime_mean"]


def vif_table(formula_rhs: str, data: pd.DataFrame, label: str) -> pd.DataFrame:
    design = dmatrix(formula_rhs, data, return_type="dataframe")
    rows = []
    for i, term in enumerate(design.columns):
        if term == "Intercept":
            continue
        rows.append({"model": label, "term": term,
                     "VIF": float(variance_inflation_factor(design.to_numpy(), i))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--hmm-meg", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--predictor", default="SDQ_externalising")
    args = ap.parse_args()

    sys.path.insert(0, str(args.hmm_meg))
    sys.path.insert(0, "/home/nz01/Github/HMM-Oddball-Dynamics")
    from run_models import build_long_table

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long, subject = build_long_table(args.analysis_dir, ())
    pred = args.predictor

    # --- VIF, centred and uncentred ----------------------------------------
    vifs = []
    for metric in PER_STATE_METRICS:
        centred = (f"C(state) + C(state):center({pred}) + center(age) "
                   f"+ center(age):center({pred}) + C(sex)")
        raw = f"C(state) + C(state):{pred} + age + age:{pred} + C(sex)"
        vifs.append(vif_table(centred, long, metric).assign(centring="centred"))
        vifs.append(vif_table(raw, long, metric).assign(centring="uncentred"))
    centred = f"center({pred}) * center(age) + C(sex)"
    raw = f"{pred} * age + C(sex)"
    vifs.append(vif_table(centred, subject, "entropy_rate").assign(centring="centred"))
    vifs.append(vif_table(raw, subject, "entropy_rate").assign(centring="uncentred"))
    vif = pd.concat(vifs, ignore_index=True)
    vif.to_csv(args.out_dir / f"vif_full_{pred}.csv", index=False)
    print("wrote VIF table", vif.shape, flush=True)

    # --- within-subject dependence -----------------------------------------
    # The reported models are OLS with cluster-robust standard errors by subject.
    # A subject random intercept is the obvious alternative, so this records what
    # one would absorb. It is reported from a direct variance decomposition of the
    # residuals, not from the mixed model: `mixedlm` returns a boundary solution
    # (group variance exactly 0, singular covariance) for all four metrics, which
    # is correct only for the two compositional ones and is an artefact for the
    # two duration ones. The mixed-model estimate is kept beside it for the record.
    rows = []
    for metric in PER_STATE_METRICS:
        fit = smf.ols(f"{metric} ~ C(state) + C(state):center({pred}) + center(age) "
                      f"+ center(age):center({pred}) + C(sex)", long).fit()
        wide = long.assign(res=fit.resid).pivot(index="subject_id", columns="state",
                                                values="res")
        between = float(wide.mean(axis=1).var(ddof=1))
        within = float(wide.var(axis=1, ddof=1).mean())
        corr = np.corrcoef(wide.to_numpy().T)
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        try:
            md = smf.mixedlm(f"{metric} ~ C(state) + C(state):center({pred}) + center(age) "
                             f"+ center(age):center({pred}) + C(sex)",
                             long, groups=long["subject_id"]).fit(reml=True, method="lbfgs")
            mixed_var = float(np.atleast_1d(md.cov_re.to_numpy().flatten())[0])
        except Exception:                                          # noqa: BLE001
            mixed_var = np.nan
        rows.append({"metric": metric,
                     "between_subject_var": between, "within_subject_var": within,
                     "variance_ratio": between / (between + within),
                     "mean_within_subject_r": float(off.mean()),
                     "min_within_subject_r": float(off.min()),
                     "max_within_subject_r": float(off.max()),
                     "max_abs_residual_sum": float(wide.sum(axis=1).abs().max()),
                     "mixedlm_group_var": mixed_var})
    dep = pd.DataFrame(rows)
    dep.to_csv(args.out_dir / f"within_subject_dependence_{pred}.csv", index=False)
    print(dep.to_string(index=False), flush=True)

    # --- residuals for QQ plots --------------------------------------------
    resid = {}
    for metric in PER_STATE_METRICS:
        fit = smf.ols(f"{metric} ~ C(state) + C(state):center({pred}) + center(age) "
                      f"+ center(age):center({pred}) + C(sex)", long).fit()
        resid[metric] = fit.resid.to_numpy()
    fit = smf.ols(f"entropy_rate ~ center({pred}) * center(age) + C(sex)", subject).fit()
    resid["entropy_rate"] = fit.resid.to_numpy()
    np.savez(args.out_dir / f"residuals_{pred}.npz", **resid)
    print("wrote residuals", flush=True)


if __name__ == "__main__":
    main()
