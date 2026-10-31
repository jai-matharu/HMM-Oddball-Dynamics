"""Per-participant state durations, occupancy and switching rate from the Viterbi path.

The duration files shipped with the original analysis (LifetimesFull.csv,
IntervalsFull.csv, Lifetimes_k7.mat, Intervals_k7.mat) are state-major with no
subject index, so per-participant means cannot be read off them. They can be
derived from the Viterbi path, which is segmented per participant by the T
vector.

A "lifetime" is the length of one continuous run in a state; an "interval" is
the gap between the end of one run and the start of the next run in that same
state.

--threshold follows HMM-MAR, which applies it to a different quantity in each
case: getStateLifeTimes drops visits shorter than the threshold, while
getStateIntervalTimes drops gaps shorter than the threshold (merging the visits
either side) and leaves every visit in place. The two filters therefore act on
different things, which is why the original duration files hold more intervals
than lifetimes. HMM_Info.m used 10 for both. The default of 0 applies none.

Usage:
  python compute_durations.py --vpath <viterbipath_k7.mat> --T <T.csv> \
      --ids <IDs.csv> --out-dir <dir> [--threshold 0] [--fs 250]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

NSTATES = 7


def load_vpath(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["vpath"]).ravel().astype(np.int8)


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end indices (inclusive) of each contiguous True run."""
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def durations_for_subject(path: np.ndarray, threshold: int) -> list[dict]:
    rows = []
    for s in range(1, NSTATES + 1):
        starts, ends = runs(path == s)
        lengths = ends - starts + 1

        # Lifetimes: visits at least `threshold` long. A visit still running at
        # the end of the segment is kept only if strictly longer, as in
        # getStateLifeTimes/aux_k.
        keep = lengths >= threshold
        if len(lengths) and ends[-1] == len(path) - 1:
            keep[-1] = lengths[-1] > threshold
        lifetimes = lengths[keep]

        # Intervals: gaps between consecutive visits, taken over *all* visits
        # and filtered on the gap itself. Leading and trailing gaps are not
        # intervals, as in getStateIntervalTimes/aux_k.
        gaps = starts[1:] - ends[:-1] - 1 if len(starts) > 1 else np.array([])
        intervals = gaps[gaps >= threshold] if len(gaps) else gaps

        rows.append({
            "state": s,
            "n_visits_all": len(lengths),
            "n_visits": len(lifetimes),
            "n_intervals": len(intervals),
            # NaN rather than 0 where a participant has no qualifying visit.
            "lifetime_mean": lifetimes.mean() if len(lifetimes) else np.nan,
            "lifetime_median": np.median(lifetimes) if len(lifetimes) else np.nan,
            "interval_mean": intervals.mean() if len(intervals) else np.nan,
            "interval_median": np.median(intervals) if len(intervals) else np.nan,
            "FO_counted": (path == s).sum() / len(path),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vpath", required=True, type=Path)
    ap.add_argument("--T", required=True, type=Path)
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=int, default=0, help="minimum visit length in samples")
    ap.add_argument("--fs", type=float, default=250.0, help="sample rate, for the _ms columns")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    T = pd.read_csv(args.T, header=None).to_numpy(dtype=np.int64).ravel()
    ids = pd.read_csv(args.ids, header=None).iloc[:, 0].astype(str).to_numpy()
    vpath = load_vpath(args.vpath)
    if len(T) != len(ids):
        raise ValueError(f"T has {len(T)} entries but IDs has {len(ids)}")
    if vpath.size != T.sum():
        raise ValueError(f"vpath has {vpath.size} timepoints but T sums to {T.sum()}")

    bounds = np.concatenate(([0], np.cumsum(T)))
    duration_rows, subject_rows, transition_rows = [], [], []

    for i, sid in enumerate(ids):
        path = vpath[bounds[i]:bounds[i + 1]]
        for row in durations_for_subject(path, args.threshold):
            duration_rows.append({"subject_id": sid, **row})

        # Switching rate and the transition matrix, both counted off the path.
        switches = int((np.diff(path) != 0).sum())
        counts = np.zeros((NSTATES, NSTATES))
        np.add.at(counts, (path[:-1] - 1, path[1:] - 1), 1)
        rowsums = counts.sum(axis=1, keepdims=True)
        # A state the participant never enters has no outgoing transitions.
        P = np.divide(counts, rowsums, out=np.full_like(counts, np.nan), where=rowsums > 0)
        for f in range(NSTATES):
            for t in range(NSTATES):
                transition_rows.append({"subject_id": sid, "from_state": f + 1,
                                        "to_state": t + 1, "probability": P[f, t]})
        subject_rows.append({"subject_id": sid, "T_samples": int(T[i]),
                             "n_switches": switches,
                             "switching_rate_vpath": switches / (len(path) - 1)})

    durations = pd.DataFrame(duration_rows)
    for c in ["lifetime_mean", "lifetime_median", "interval_mean", "interval_median"]:
        durations[f"{c}_ms"] = durations[c] / args.fs * 1000.0

    suffix = f"_thresh{args.threshold}" if args.threshold else ""
    for name, df in [(f"durations{suffix}", durations),
                     (f"vpath_subject_level{suffix}", pd.DataFrame(subject_rows)),
                     (f"vpath_transition_matrices{suffix}", pd.DataFrame(transition_rows))]:
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv  {df.shape}")


if __name__ == "__main__":
    main()
