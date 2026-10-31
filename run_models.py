"""Fit the individual-difference models for one question and correct across the family.

Per-state measures (fractional occupancy, transitions into state, interval, lifetime) are
fitted with state as a within-participant factor and each state carrying its own slope on the
individual-difference measure; entropy rate is fitted as a subject-level scalar. Inference is by
permutation of the measure across participants, refitting every model on the same permuted data
so that the joint null across terms is preserved.

Two corrections are reported:

* Benjamini-Hochberg across all 31 terms of the family - the reported correction.
* Westfall-Young step-down max-T over the 30 terms whose null comes from permuting the
  individual-difference measure - a dependence-aware sensitivity test. The 31st term
  (entropy rate on age) is excluded from the max-T set because a valid null for it requires
  permuting age rather than the measure, so it cannot share the same permutation. It is
  permuted separately and reported with BH only.

Max-T is built on t-statistics rather than coefficients, because the four per-state measures are
on incomparable scales (occupancy ~0.2, interval ~10^3 samples) and a maximum over raw
coefficients would be meaningless.

The statistical models themselves come from the HMM_MEG analysis package; --hmm-meg points at it.

Usage:
  python run_models.py --analysis-dir <analysis_39> --hmm-meg <path> --out-dir <dir> \
      --predictor SDQ_externalising [--n-permutations 10000] [--seed 0] [--exclude 99078]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NSTATES = 7
PER_STATE_METRICS = ["FO", "transition_into_sum", "interval_mean", "lifetime_mean"]


def build_long_table(analysis_dir: Path, exclude: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per participant x state with all four per-state measures and the covariates."""
    from analysis.models import transition_column_sums

    subject = pd.read_csv(analysis_dir / "subject_level.csv")
    state = pd.read_csv(analysis_dir / "state_level.csv")
    trans = pd.read_csv(analysis_dir / "transition_matrices.csv")
    for d in (subject, state, trans):
        d["subject_id"] = d["subject_id"].astype(str)

    if exclude:
        subject = subject[~subject.subject_id.isin(exclude)]
        state = state[~state.subject_id.isin(exclude)]
        trans = trans[~trans.subject_id.isin(exclude)]

    col_sums = transition_column_sums(trans)
    col_sums["subject_id"] = col_sums["subject_id"].astype(str)

    long = (state.merge(col_sums, on=["subject_id", "state"], validate="one_to_one")
                 .merge(subject, on="subject_id", validate="many_to_one"))
    missing = [m for m in PER_STATE_METRICS if m not in long.columns]
    if missing:
        raise ValueError(f"per-state measures missing from the long table: {missing}")
    if long[PER_STATE_METRICS].isna().any().any():
        n = int(long[PER_STATE_METRICS].isna().any(axis=1).sum())
        raise ValueError(f"{n} participant x state rows have a missing measure; "
                         "every participant must enter every state for this design")
    return long, subject


def fit_all(long: pd.DataFrame, subject: pd.DataFrame, predictor: str):
    """Fit the four per-state models and the entropy model; return observed coefs and t-values."""
    from analysis.models import (
        fit_state_metric_model_simple_slopes,
        fit_subject_scalar_model,
        state_slope_terms,
    )

    coefs, tvals, diagnostics = {}, {}, {}
    for metric in PER_STATE_METRICS:
        result, diag = fit_state_metric_model_simple_slopes(long, metric=metric, cognition_col=predictor)
        for state, term in enumerate(state_slope_terms(NSTATES, cognition_col=predictor), start=1):
            label = f"{metric}|state{state}"
            coefs[label] = float(result.params[term])
            tvals[label] = float(result.tvalues[term])
        diagnostics[metric] = diag

    result, diag = fit_subject_scalar_model(subject, outcome="entropy_rate", cognition_col=predictor)
    entropy_terms = {
        "entropy_rate|measure": f"center({predictor})",
        "entropy_rate|age": "center(age)",
        "entropy_rate|age_x_measure": f"center({predictor}):center(age)",
    }
    for label, term in entropy_terms.items():
        if term not in result.params.index:
            raise ValueError(f"term '{term}' not in the entropy model; got {list(result.params.index)}")
        coefs[label] = float(result.params[term])
        tvals[label] = float(result.tvalues[term])
    diagnostics["entropy_rate"] = diag
    return coefs, tvals, diagnostics, entropy_terms


def permute(long: pd.DataFrame, subject: pd.DataFrame, col: str, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle a participant-level column across participants, consistently in both tables."""
    values = subject[["subject_id", col]].drop_duplicates("subject_id")
    shuffled = pd.Series(values[col].to_numpy(),
                         index=rng.permutation(values["subject_id"].to_numpy()))
    out = []
    for d in (long, subject):
        e = d.copy()
        e[col] = e["subject_id"].map(shuffled)
        out.append(e)
    return out[0], out[1]


def two_sided_p(observed: float, null: np.ndarray) -> float:
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))


def westfall_young(observed_t: dict[str, float], null_t: dict[str, np.ndarray]) -> pd.DataFrame:
    """Single-step and step-down max-T p-values over a set of jointly permuted terms."""
    terms = sorted(observed_t, key=lambda k: -abs(observed_t[k]))
    T = np.abs(np.column_stack([null_t[t] for t in terms]))      # permutations x terms, ordered
    obs = np.array([abs(observed_t[t]) for t in terms])
    n = T.shape[0]

    single = (np.sum(T.max(axis=1)[:, None] >= obs[None, :], axis=0) + 1) / (n + 1)

    # Step-down: successive maxima taken from the least significant term inward.
    successive = np.maximum.accumulate(T[:, ::-1], axis=1)[:, ::-1]
    step = (np.sum(successive >= obs[None, :], axis=0) + 1) / (n + 1)
    step = np.maximum.accumulate(step)                            # enforce monotonicity
    return pd.DataFrame({"term": terms, "wy_single_step": single, "wy_step_down": step})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--hmm-meg", required=True, type=Path, help="path to the HMM_MEG repo")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--n-permutations", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()

    sys.path.insert(0, str(args.hmm_meg))
    from analysis.correction import correct_pvalues
    from analysis.models import fit_subject_scalar_model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long, subject = build_long_table(args.analysis_dir, tuple(args.exclude))
    print(f"n participants = {subject.subject_id.nunique()}, rows = {len(long)}, "
          f"predictor = {args.predictor}", flush=True)

    coefs, tvals, diagnostics, entropy_terms = fit_all(long, subject, args.predictor)
    measure_terms = [t for t in coefs if t != "entropy_rate|age"]

    rng = np.random.default_rng(args.seed)
    null_coef = {t: np.empty(args.n_permutations) for t in coefs}
    null_t = {t: np.empty(args.n_permutations) for t in coefs}
    for i in range(args.n_permutations):
        # One shuffle of the measure drives every model, preserving the joint null.
        pl, ps = permute(long, subject, args.predictor, rng)
        c, t, _, _ = fit_all(pl, ps, args.predictor)
        for term in measure_terms:
            null_coef[term][i] = c[term]
            null_t[term][i] = t[term]
        # The age main effect needs age permuted instead; it is not in the max-T set,
        # and only the entropy model is refitted for it.
        _, age_subject = permute(long, subject, "age", rng)
        age_result, _ = fit_subject_scalar_model(age_subject, outcome="entropy_rate",
                                                 cognition_col=args.predictor)
        null_coef["entropy_rate|age"][i] = float(age_result.params["center(age)"])
        null_t["entropy_rate|age"][i] = float(age_result.tvalues["center(age)"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.n_permutations} permutations", flush=True)

    # Inference is on the t-statistic throughout. The permutation null of a raw
    # coefficient and that of its t co-vary with the cluster-robust SE, so the two
    # give different p-values for the same term; the lifetime and interval models
    # are skewed enough for that gap to be large. The t is studentised and so
    # comparable across the four measures, which is also what max-T requires. The
    # coefficient-based p is kept beside it for transparency, not for correction.
    rows = [{"term": t, "coef": coefs[t], "t": tvals[t],
             "perm_p": two_sided_p(tvals[t], null_t[t]),
             "perm_p_coef": two_sided_p(coefs[t], null_coef[t]),
             "permuted": "age" if t == "entropy_rate|age" else args.predictor}
            for t in coefs]
    table = pd.DataFrame(rows)

    bh = correct_pvalues(dict(zip(table.term, table.perm_p)))[["term", "p_adjusted", "significant"]]
    bh = bh.rename(columns={"p_adjusted": "bh_p", "significant": "bh_significant"})
    table = table.merge(bh, on="term", how="left")
    table = table.merge(westfall_young({t: tvals[t] for t in measure_terms},
                                       {t: null_t[t] for t in measure_terms}),
                        on="term", how="left")

    table["measure"] = table.term.str.split("|").str[0]
    table["state"] = table.term.str.split("|").str[1]
    table = table.sort_values("perm_p")
    table.to_csv(args.out_dir / f"terms_{args.predictor}.csv", index=False)
    print(f"wrote terms_{args.predictor}.csv  {table.shape}")

    # The permutation nulls are kept so that any later change to the correction
    # family or scheme is a re-derivation rather than a refit.
    np.savez_compressed(args.out_dir / f"nulls_{args.predictor}.npz",
                        terms=np.array(list(coefs)),
                        null_t=np.column_stack([null_t[t] for t in coefs]),
                        null_coef=np.column_stack([null_coef[t] for t in coefs]),
                        observed_t=np.array([tvals[t] for t in coefs]),
                        observed_coef=np.array([coefs[t] for t in coefs]))

    diag_rows = []
    for metric, diag in diagnostics.items():
        norm = diag.get("normality", {})
        diag_rows.append({"model": metric, **norm})
    pd.DataFrame(diag_rows).to_csv(args.out_dir / f"diagnostics_{args.predictor}.csv", index=False)
    vif = diagnostics["entropy_rate"].get("vif")
    if vif is not None:
        vif.to_csv(args.out_dir / f"vif_{args.predictor}.csv", index=False)
    print("wrote diagnostics")


if __name__ == "__main__":
    main()
