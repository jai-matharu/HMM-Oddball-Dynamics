"""Subject-level HMM summaries against one measure, with permutation inference.

Entropy rate is part of the reported 31-term family. Switching rate and maximum
fractional occupancy sit outside every correction family by design, and are
computed here so the draft's original subject-level claims can be checked
against the 39-participant refit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit(df, outcome, pred):
    return smf.ols(f"{outcome} ~ center({pred}) * center(age) + C(sex)", df).fit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--n-permutations", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.analysis_dir / "subject_level.csv", dtype={"subject_id": str})
    pred = args.predictor
    outcomes = ["entropy_rate", "switching_rate", "max_FO"]
    terms = {f"center({pred})": "measure", "center(age)": "age",
             f"center({pred}):center(age)": "age_x_measure"}

    observed = {}
    for o in outcomes:
        f = fit(df, o, pred)
        for t, lab in terms.items():
            observed[(o, lab)] = (float(f.params[t]), float(f.tvalues[t]))

    # The measure and interaction terms are tested by permuting the measure; the
    # age main effect needs age permuted instead, since shuffling the measure
    # leaves age's t almost unchanged and would yield a null centred on the
    # observed value. Both nulls are built in the same loop.
    rng = np.random.default_rng(args.seed)
    null = {k: np.empty(args.n_permutations) for k in observed}
    for i in range(args.n_permutations):
        by_measure = df.assign(**{pred: rng.permutation(df[pred].to_numpy())})
        by_age = df.assign(age=rng.permutation(df["age"].to_numpy()))
        for o in outcomes:
            fm = fit(by_measure, o, pred)
            for t, lab in terms.items():
                if lab != "age":
                    null[(o, lab)][i] = float(fm.tvalues[t])
            null[(o, "age")][i] = float(fit(by_age, o, pred).tvalues["center(age)"])

    rows = []
    for (o, lab), (coef, tval) in observed.items():
        n = null[(o, lab)]
        p = float((np.sum(np.abs(n) >= abs(tval)) + 1) / (len(n) + 1))
        rows.append({"outcome": o, "term": lab, "coef": coef, "t": tval, "perm_p": p,
                     "permuted": "age" if lab == "age" else pred})
    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / f"subject_scalars_{pred}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
