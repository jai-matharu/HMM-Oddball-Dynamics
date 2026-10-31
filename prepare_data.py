"""Build the tidy analysis tables from the original oddball HMM exports.

Reads the files in --data-dir (private, never in this repo) and writes three
CSVs matching the contract used by the RED_MEG analysis package:

  subject_level.csv       - one row per subject
  state_level.csv         - one row per subject x state
  transition_matrices.csv - one row per subject x from_state x to_state

Subject-level HMM metrics are recomputed from the per-participant transition
matrices rather than copied from the spreadsheet; the original columns are
carried through alongside as *_original for comparison.

Pass --durations to fill in the per-state lifetimes and intervals produced by
compute_durations.py; without it those columns are left empty, since the
duration files shipped with the original analysis carry no subject index.

Usage:
  python prepare_data.py --data-dir <dir> --out-dir <dir> [--durations <csv>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NSTATES = 7
FS_HZ = 250.0  # T is in samples; every value in Preprocessing_Steps is a multiple of 250


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1, normalised to sum to 1."""
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    mu = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    return mu / mu.sum()


def entropy_rate(P: np.ndarray, mu: np.ndarray, fs: float | None = None) -> float:
    """Entropy rate in nats per sample, or per second if fs is given."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_P = np.where(P > 0, np.log(P), 0.0)
    rate = float(np.dot(mu, -np.sum(P * log_P, axis=1)))
    return rate * fs if fs is not None else rate


def load_transition_matrices(path: Path, subject_ids) -> np.ndarray:
    """Read the 40 x 49 sheet into (nsub, 7, 7).

    The 49 columns are a MATLAB reshape of each 7x7 matrix, so they unflatten
    in column-major order. Row order matches the spreadsheet row order.
    """
    flat = pd.read_excel(path, header=None).dropna(how="all").to_numpy(dtype=float)
    if flat.shape != (len(subject_ids), NSTATES**2):
        raise ValueError(f"expected {(len(subject_ids), NSTATES ** 2)} in {path.name}, got {flat.shape}")
    P = flat.reshape(-1, NSTATES, NSTATES, order="F")
    rowsums = P.sum(axis=2)
    if not np.allclose(rowsums, 1.0, atol=1e-6):
        raise ValueError(f"transition matrices are not row-stochastic (max |rowsum-1| = {np.abs(rowsums - 1).max():.2e})")
    return P


def load_subjects(data_dir: Path) -> pd.DataFrame:
    """Demographics, cognition, SDQ and SES for the 40 participants with HMM output."""
    df = pd.read_csv(data_dir / "SESWithImputed_participants_only_40.csv")
    df = df.rename(columns={"RED ID": "subject_id", "MATRIX_Tscore": "matrix_T",
                            "MATRIX_RawScore": "matrix_raw", "Age": "age",
                            "Subjective_SES": "subjective_SES", "Income": "income",
                            "Equivalised_Income_Updated": "equivalised_income"})
    df["sex"] = df["Gender"].map({"M": 1, "F": 2, 1: 1, 2: 2, 1.0: 1, 2.0: 2})
    if df["sex"].isna().any():
        raise ValueError(f"unmapped Gender values: {sorted(df.loc[df.sex.isna(), 'Gender'].unique())}")

    sdq = {c: c.lower().replace("sdq_", "SDQ_") for c in df.columns if c.startswith("SDQ_")}
    df = df.rename(columns=sdq)
    # Standard SDQ composites: the draft's Externalising/Internalising scores.
    df["SDQ_externalising"] = df["SDQ_hyperactivity"] + df["SDQ_conduct"]
    df["SDQ_internalising"] = df["SDQ_emotion"] + df["SDQ_peerproblems"]
    return df


def load_T(data_dir: Path, subject_ids) -> pd.Series:
    """Per-participant downsampled timepoint count, keyed by RED ID."""
    T = pd.read_csv(data_dir / "T_candidate_40_with_RED_ID.csv")
    T = T.rename(columns={"RED_ID": "subject_id", "Time_ds_candidate": "T_samples"})
    T["subject_id"] = T["subject_id"].astype(str)
    return T.set_index("subject_id")["T_samples"].reindex([str(s) for s in subject_ids]).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--durations", type=Path,
                    help="durations csv from compute_durations.py")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subs = load_subjects(args.data_dir)
    ids = subs["subject_id"].to_numpy()
    P = load_transition_matrices(args.data_dir / "Individual_TransitionProbabilities_K7_original.xls", ids)

    mu = np.vstack([stationary_distribution(P[i]) for i in range(len(ids))])
    subject_level = pd.DataFrame({
        "subject_id": ids,
        "age": subs["age"].to_numpy(),
        "sex": subs["sex"].to_numpy(),
        "matrix_T": subs["matrix_T"].to_numpy(),
        "matrix_raw": subs["matrix_raw"].to_numpy(),
        "income": subs["income"].to_numpy(),
        "equivalised_income": subs["equivalised_income"].to_numpy(),
        "subjective_SES": subs["subjective_SES"].to_numpy(),
        "T_samples": load_T(args.data_dir, ids),
    })
    for c in [c for c in subs.columns if c.startswith("SDQ_")]:
        subject_level[c] = subs[c].to_numpy()

    subject_level["entropy_rate"] = [entropy_rate(P[i], mu[i], fs=FS_HZ) for i in range(len(ids))]
    subject_level["entropy_rate_per_sample"] = [entropy_rate(P[i], mu[i]) for i in range(len(ids))]
    diag = np.diagonal(P, axis1=1, axis2=2)
    # Stationary-weighted probability of leaving the current state per sample.
    subject_level["switching_rate"] = 1.0 - np.sum(mu * diag, axis=1)
    subject_level["switching_rate_unweighted"] = 1.0 - diag.mean(axis=1)
    subject_level["max_FO"] = mu.max(axis=1)

    # Columns as reported in the original spreadsheet, for comparison.
    subject_level["entropy_rate_original"] = subs["Entropy"].to_numpy()
    subject_level["switching_rate_original"] = subs["SwitchingRate"].to_numpy()
    subject_level["max_FO_original"] = subs["MaxFO"].to_numpy()

    state_level = pd.DataFrame([
        {"subject_id": ids[i], "state": s + 1,
         "FO": mu[i, s],
         "FO_original": subs[f"FO{s + 1}"].to_numpy()[i],
         "lifetime_mean": np.nan,
         "interval_mean": np.nan}
        for i in range(len(ids)) for s in range(NSTATES)
    ])

    if args.durations:
        dur = pd.read_csv(args.durations)
        dur["subject_id"] = dur["subject_id"].astype(str)
        state_level["subject_id"] = state_level["subject_id"].astype(str)
        state_level = state_level.drop(columns=["lifetime_mean", "interval_mean"]).merge(
            dur[["subject_id", "state", "n_visits", "lifetime_mean", "interval_mean",
                 "lifetime_median", "interval_median", "FO_counted"]],
            on=["subject_id", "state"], how="left", validate="one_to_one")
        if state_level[["n_visits"]].isna().any().any():
            raise ValueError("durations file does not cover every subject x state")

    transitions = pd.DataFrame([
        {"subject_id": ids[i], "from_state": f + 1, "to_state": t + 1, "probability": P[i, f, t]}
        for i in range(len(ids)) for f in range(NSTATES) for t in range(NSTATES)
    ])

    for name, df in [("subject_level", subject_level), ("state_level", state_level),
                     ("transition_matrices", transitions)]:
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv  {df.shape}")


if __name__ == "__main__":
    main()
