"""Step 11h: declared additions to the ERF analysis (specified in ANALYSIS_PLAN.md, 11h).

prepare
  A1  spread of the device-to-head transforms across recordings (report only).
  A2  amplitude ~ center(head_radius) * center(age) + C(sex), per window.
  B1  amplitude ~ center(SDQ_externalising) * center(age) + C(sex), per window.
      Both: permutation of the tested variable (10,000, seed 0), BH across the five windows,
      n = 39 and n = 34 (without the five formerly contaminated recordings).
  A3/B2 predictors: each window's amplitude residualised on head radius (`_head`) and on
      externalising (`_ext`), OLS with intercept at n = 39, mean added back. Written into a
      copy of the analysis directory for the unchanged 11f runner.

summarise
  For the ten runner outputs: checkpoint 6, BH and WY counts per window, overlap with the 11f
  terms, correlation of each t profile with the externalising profile, pooled BH over 155.

Usage:
  python erf_followup.py prepare --analysis-dir <analysis_39_erf_fT> --geometry <csv>
      --hmm-meg <HMM_MEG> --out-analysis-dir <dir> --out-dir <dir>
  python erf_followup.py summarise --analysis-dir <analysis_39_erf_fT> --hmm-meg <HMM_MEG>
      --models-11f <models_erf> --models-adj <models_erf_adj> --externalising <terms csv>
      --out-dir <dir>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIVE = ["99016", "calmred02", "calmred04", "calmred05", "calmred06"]
N_PERM = 10_000
ADJUSTMENTS = {"head": "head_radius_mm", "ext": "SDQ_externalising"}


def scalar_t(y: np.ndarray, z: np.ndarray, age: np.ndarray, female: np.ndarray) -> float:
    """t of center(z) in y ~ center(z) * center(age) + C(sex), ordinary OLS."""
    zc, ac = z - z.mean(), age - age.mean()
    X = np.column_stack([np.ones_like(y), female, zc, ac, zc * ac])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return float(beta[2] / np.sqrt(cov[2, 2]))


def scalar_test(df: pd.DataFrame, outcome: str, var: str, fit_reference) -> dict:
    y = df[outcome].to_numpy(float)
    z = df[var].to_numpy(float)
    age = df["age"].to_numpy(float)
    female = (df["sex"].to_numpy() != df["sex"].min()).astype(float)
    t_obs = scalar_t(y, z, age, female)

    # Hard check against the paper's own scalar model.
    result, _ = fit_reference(df, outcome=outcome, cognition_col=var)
    t_ref = float(result.tvalues[f"center({var})"])
    if abs(t_obs - t_ref) > 1e-8 * max(1.0, abs(t_ref)):
        raise ValueError(f"{outcome} ~ {var}: t {t_obs} against the reference model's {t_ref}")

    rng = np.random.default_rng(0)
    null = np.array([scalar_t(y, rng.permutation(z), age, female) for _ in range(N_PERM)])
    p = float((np.sum(np.abs(null) >= abs(t_obs)) + 1) / (N_PERM + 1))
    return {"coef_per_unit": float(result.params[f"center({var})"]), "t": t_obs, "perm_p": p,
            "pearson_r": float(np.corrcoef(y, z)[0, 1]), "n": len(df)}


def residualise(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta + y.mean()


def prepare(args) -> None:
    sys.path.insert(0, str(args.hmm_meg))
    from analysis.correction import correct_pvalues
    from analysis.models import fit_subject_scalar_model

    subject = pd.read_csv(args.analysis_dir / "subject_level.csv", dtype={"subject_id": str})
    names = pd.read_csv(args.analysis_dir / "erf_predictor_names.csv")
    geometry = pd.read_csv(args.geometry, dtype={"subject_id": str})
    if set(geometry.subject_id) != set(subject.subject_id):
        raise ValueError("head geometry and the analysis sample hold different participants")
    subject = subject.merge(geometry[["subject_id", "head_radius_mm"]], on="subject_id",
                            validate="one_to_one")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # A1
    tr, rot = geometry.translation_from_first_mm.max(), geometry.rotation_from_first_deg.max()
    equalised = bool(tr <= 1.0 and rot <= 1.0)
    print(f"A1: device-to-head transforms differ from the first recording's by up to {tr:.3f} mm "
          f"and {rot:.3f} deg -> head position "
          f"{'equalised by maxfilter; not entered' if equalised else 'NOT equalised; stop for an author decision'}")

    # A2 and B1
    rows = []
    for check, var in (("A2", "head_radius_mm"), ("B1", "SDQ_externalising")):
        for sample, df in (("n39", subject), ("n34", subject[~subject.subject_id.isin(FIVE)])):
            block = []
            for window, pred in zip(names.window, names.predictor):
                block.append({"check": check, "variable": var, "sample": sample,
                              "window": window, "predictor": pred,
                              **scalar_test(df, pred, var, fit_subject_scalar_model)})
            bh = correct_pvalues({b["predictor"]: b["perm_p"] for b in block})
            bh = bh.set_index("term")
            for b in block:
                b["bh_p"] = float(bh.loc[b["predictor"], "p_adjusted"])
                b["bh_significant"] = bool(bh.loc[b["predictor"], "significant"])
            rows += block
    tests = pd.DataFrame(rows)
    tests.to_csv(args.out_dir / "erf_followup_scalar_tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tests[["check", "sample", "window", "pearson_r", "t", "perm_p", "bh_p",
                     "bh_significant"]].round(4).to_string(index=False))

    # A3 and B2 predictors
    args.out_analysis_dir.mkdir(parents=True, exist_ok=True)
    out = subject.copy()
    new = []
    for pred in names.predictor:
        for suffix, var in ADJUSTMENTS.items():
            col = f"{pred}_{suffix}"
            out[col] = residualise(out[pred].to_numpy(float), out[var].to_numpy(float))
            new.append(col)
            r = np.corrcoef(out[col], out[var])[0, 1]
            if abs(r) > 1e-10:
                raise ValueError(f"{col}: residual still correlates with {var} (r = {r})")
    out.to_csv(args.out_analysis_dir / "subject_level.csv", index=False)
    for name in ("state_level.csv", "transition_matrices.csv"):
        shutil.copy(args.analysis_dir / name, args.out_analysis_dir / name)
    pd.DataFrame({"predictor": new}).to_csv(args.out_analysis_dir / "adjusted_predictors.csv",
                                            index=False)
    print(f"wrote {len(new)} adjusted predictors to {args.out_analysis_dir}")


def checkpoint6(table: pd.DataFrame) -> bool:
    t = table[table.term != "entropy_rate|age"].copy()
    t = t.sort_values("t", key=np.abs, ascending=False)
    ok_p = bool((t.wy_step_down >= t.perm_p - 1e-12).all())
    ok_ss = bool((t.wy_single_step >= t.wy_step_down - 1e-12).all())
    ok_mono = bool((np.diff(t.wy_step_down.to_numpy()) >= -1e-12).all())
    return ok_p and ok_ss and ok_mono


def summarise(args) -> None:
    sys.path.insert(0, str(args.hmm_meg))
    from analysis.correction import correct_pvalues

    names = pd.read_csv(args.analysis_dir / "erf_predictor_names.csv")
    ext = pd.read_csv(args.externalising).set_index("term")
    rows, pooled = [], {k: {} for k in ADJUSTMENTS}
    for window, pred in zip(names.window, names.predictor):
        base = pd.read_csv(args.models_11f / f"terms_{pred}.csv").set_index("term")
        base_sig = set(base.index[base.bh_significant])
        for suffix in ADJUSTMENTS:
            table = pd.read_csv(args.models_adj / f"terms_{pred}_{suffix}.csv")
            ok = checkpoint6(table)
            if not ok:
                raise ValueError(f"{pred}_{suffix}: checkpoint 6 failed")
            table = table.set_index("term")
            sig = set(table.index[table.bh_significant])
            terms = ext.index
            r = float(np.corrcoef(table.loc[terms, "t"], ext.loc[terms, "t"])[0, 1])
            r_base = float(np.corrcoef(table.loc[terms, "t"], base.loc[terms, "t"])[0, 1])
            rows.append({"window": window, "adjusted_for": ADJUSTMENTS[suffix],
                         "checkpoint6": ok,
                         "bh_significant": len(sig), "wy_lt_05": int((table.wy_step_down < .05).sum()),
                         "smallest_bh_p": float(table.bh_p.min()),
                         "bh_significant_11f": len(base_sig), "kept_from_11f": len(sig & base_sig),
                         "lost_from_11f": "; ".join(sorted(base_sig - sig)),
                         "new_vs_11f": "; ".join(sorted(sig - base_sig)),
                         "significant_terms": "; ".join(sorted(sig)),
                         "r_t_with_externalising": r, "r_t_with_11f": r_base})
            for term, p in table.perm_p.items():
                pooled[suffix][f"{pred}::{term}"] = p
    summary = pd.DataFrame(rows)
    pooled_rows = []
    for suffix, ps in pooled.items():
        bh = correct_pvalues(ps)
        sig = bh[bh.significant]
        pooled_rows.append({"adjusted_for": ADJUSTMENTS[suffix], "n_terms": len(ps),
                            "pooled_bh_significant": len(sig),
                            "terms": "; ".join(sorted(sig.term))})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "erf_followup_summary.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(args.out_dir / "erf_followup_pooled.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.max_colwidth", 200):
        print(summary.drop(columns=["significant_terms"]).round(4).to_string(index=False))
        print()
        for _, row in summary.iterrows():
            print(f"{row.window} | {row.adjusted_for}: {row.significant_terms or '(none)'}")
        print()
        print(pd.DataFrame(pooled_rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--hmm-meg", type=Path, required=True)
    p.add_argument("--out-analysis-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    s = sub.add_parser("summarise")
    s.add_argument("--analysis-dir", type=Path, required=True)
    s.add_argument("--hmm-meg", type=Path, required=True)
    s.add_argument("--models-11f", type=Path, required=True)
    s.add_argument("--models-adj", type=Path, required=True)
    s.add_argument("--externalising", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else summarise(args)


if __name__ == "__main__":
    main()
