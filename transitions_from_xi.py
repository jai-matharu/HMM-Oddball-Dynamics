"""Per-participant transition matrices from Xi, and the metrics derived from them.

This is the route the original analysis took: HMM_Entropy.m sums the posterior
joint probabilities Xi over each participant's time course and normalises the
rows, then takes the stationary distribution and the entropy rate of that
matrix. Xi holds one entry per transition rather than per timepoint, so each
participant occupies T-1 rows.

Note on units: HMM_Entropy.m multiplied the entropy rate by 4 and labelled the
result nats/second. The data are sampled at 250 Hz (HMM_Info.m line 35), so
nats/sample converts to nats/second by multiplying by 250. Both are written out.

Xi is far larger than memory, so it is read in slices along the time axis.

Usage:
  python transitions_from_xi.py --xi <Xi_k7.mat> --T <T.csv> --ids <IDs.csv> \
      --out-dir <dir> [--fs 250] [--chunk 500000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

NSTATES = 7


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    mu = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    return mu / mu.sum()


def entropy_rate(P: np.ndarray, mu: np.ndarray) -> float:
    """Entropy rate in nats per sample."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_P = np.where(P > 0, np.log(P), 0.0)
    return float(np.dot(mu, -np.sum(P * log_P, axis=1)))


def summed_xi(xi_path: Path, T: np.ndarray, chunk: int) -> np.ndarray:
    """Sum Xi over time within each participant, giving (nsub, 7, 7).

    MATLAB saves Xi as (time, from, to); HDF5 reverses the axes, so the
    dataset is (to, from, time) and the summed slab needs transposing back.
    Each participant contributes T-1 rows.
    """
    Tnew = T - 1
    bounds = np.concatenate(([0], np.cumsum(Tnew)))
    with h5py.File(xi_path, "r") as f:
        Xi = f["Xi"]
        if Xi.shape[:2] != (NSTATES, NSTATES):
            raise ValueError(f"expected leading dims ({NSTATES}, {NSTATES}), got {Xi.shape}")
        if Xi.shape[2] != bounds[-1]:
            raise ValueError(f"Xi has {Xi.shape[2]} transitions but T-1 sums to {bounds[-1]}")

        out = np.zeros((len(T), NSTATES, NSTATES))
        for i in range(len(T)):
            lo, hi = bounds[i], bounds[i + 1]
            acc = np.zeros((NSTATES, NSTATES))
            for a in range(lo, hi, chunk):
                b = min(a + chunk, hi)
                acc += np.asarray(Xi[:, :, a:b]).sum(axis=2)
            out[i] = acc.T  # (to, from) -> (from, to)
            print(f"  subject {i + 1}/{len(T)}: {hi - lo} transitions", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi", required=True, type=Path)
    ap.add_argument("--T", required=True, type=Path)
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--chunk", type=int, default=500_000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    T = pd.read_csv(args.T, header=None).to_numpy(dtype=np.int64).ravel()
    ids = pd.read_csv(args.ids, header=None).iloc[:, 0].astype(str).to_numpy()
    if len(T) != len(ids):
        raise ValueError(f"T has {len(T)} entries but IDs has {len(ids)}")

    counts = summed_xi(args.xi, T, args.chunk)
    P = counts / counts.sum(axis=2, keepdims=True)
    rowsums = P.sum(axis=2)
    if not np.allclose(rowsums, 1.0, atol=1e-9):
        raise ValueError(f"rows not normalised (max |rowsum-1| = {np.abs(rowsums - 1).max():.2e})")

    mu = np.vstack([stationary_distribution(P[i]) for i in range(len(ids))])
    rate = np.array([entropy_rate(P[i], mu[i]) for i in range(len(ids))])
    diag = np.diagonal(P, axis1=1, axis2=2)

    subject = pd.DataFrame({
        "subject_id": ids,
        "T_samples": T,
        "entropy_rate": rate * args.fs,            # nats/second at 250 Hz
        "entropy_rate_per_sample": rate,
        "entropy_rate_times4": rate * 4.0,         # as originally reported
        "max_FO": mu.max(axis=1),
        "switching_rate_markov": 1.0 - np.sum(mu * diag, axis=1),
    })
    state = pd.DataFrame([
        {"subject_id": ids[i], "state": s + 1, "FO_stationary": mu[i, s]}
        for i in range(len(ids)) for s in range(NSTATES)
    ])
    transitions = pd.DataFrame([
        {"subject_id": ids[i], "from_state": f + 1, "to_state": t + 1, "probability": P[i, f, t]}
        for i in range(len(ids)) for f in range(NSTATES) for t in range(NSTATES)
    ])

    for name, df in [("xi_subject_level", subject), ("xi_state_level", state),
                     ("xi_transition_matrices", transitions)]:
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv  {df.shape}")


if __name__ == "__main__":
    main()
