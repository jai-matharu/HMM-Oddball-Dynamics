"""Build the tidy analysis tables for the 39-participant analysis sample.

Demographic, cognitive and behavioural variables come from the original
spreadsheet. Every HMM metric comes from the 39-participant fit in
Outputs/HMM_Outputs/K7_2, via compute_durations.py (Viterbi path) and
transitions_from_xi.py (Xi). The spreadsheet's own Entropy/SwitchingRate/MaxFO/
FO1-7 columns were produced by the earlier 40-participant fit, so they are
carried through only as *_orig40 for provenance and are not analysis variables.

Writes:
  subject_level.csv       - one row per participant
  state_level.csv         - one row per participant x state
  transition_matrices.csv - one row per participant x from_state x to_state

Usage:
  python prepare_data.py --data-dir <original exports> --hmm-dir <tidy_K7_2> \
      --out-dir <dir> [--threshold 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NSTATES = 7

# 99114 is excluded from the analysis sample.
#
# The exclusion originates in the dissertation's own HMM_Compute.m, which
# removes data(32)/T(32) under the comment "Remove participant 99114" and
# re-fits the HMM on the remaining 39 into Outputs/HMM_Outputs/K7_2. No reason
# is recorded there. The manuscript attributes it to poor digitisation of the
# scalp locations giving an inadequate co-registration solution, which the
# preprocessing records contradict: 99114 has its own T1 with extracted scalp,
# skull and cortex surfaces (sub-073) and is marked coregistered, beamformed
# and parcellated, with a beamformed file and a parcellated HMM input on disk.
#
# What the data show is a state time course dominated by a single state. In the
# 40-participant fit 99114 has FO 0.651 in state 1 against a group mean of
# 0.033, supplying 50% of all state-1 occupancy across the sample, and is the
# most extreme participant of the 40 on entropy rate (z = -3.50), switching
# rate (z = -3.38) and max FO (0.651, z = +2.73). Recording length (373,250
# samples) and age (10.0) are unremarkable.
EXCLUDED = ("99114",)


def load_behaviour(data_dir: Path) -> pd.DataFrame:
    """Demographics, cognition, SDQ and SES, as recorded in the spreadsheet."""
    df = pd.read_csv(data_dir / "SESWithImputed_participants_only_40.csv")
    df = df.rename(columns={"RED ID": "subject_id", "MATRIX_Tscore": "matrix_T",
                            "MATRIX_RawScore": "matrix_raw", "Age": "age",
                            "Subjective_SES": "subjective_SES", "Income": "income",
                            "Equivalised_Income_Updated": "equivalised_income"})
    df["subject_id"] = df["subject_id"].astype(str)
    df["sex"] = df["Gender"].map({"M": 1, "F": 2, 1: 1, 2: 2, 1.0: 1, 2.0: 2})
    if df["sex"].isna().any():
        raise ValueError(f"unmapped Gender values: {sorted(df.loc[df.sex.isna(), 'Gender'].unique())}")

    df = df.rename(columns={c: c.lower().replace("sdq_", "SDQ_")
                            for c in df.columns if c.startswith("SDQ_")})
    # The draft's Externalising and Internalising scores.
    df["SDQ_externalising"] = df["SDQ_hyperactivity"] + df["SDQ_conduct"]
    df["SDQ_internalising"] = df["SDQ_emotion"] + df["SDQ_peerproblems"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="directory holding the original exports")
    ap.add_argument("--hmm-dir", required=True, type=Path,
                    help="directory holding the K7_2 derived metrics")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=int, default=10,
                    help="minimum-duration threshold whose durations to use (0 for none)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    beh = load_behaviour(args.data_dir)
    beh = beh[~beh.subject_id.isin(EXCLUDED)].reset_index(drop=True)

    suffix = f"_thresh{args.threshold}" if args.threshold else ""
    xi_sub = pd.read_csv(args.hmm_dir / "xi_subject_level.csv")
    xi_state = pd.read_csv(args.hmm_dir / "xi_state_level.csv")
    dur = pd.read_csv(args.hmm_dir / f"durations{suffix}.csv")
    vpath_sub = pd.read_csv(args.hmm_dir / "vpath_subject_level.csv")
    trans = pd.read_csv(args.hmm_dir / "xi_transition_matrices.csv")
    for d in (xi_sub, xi_state, dur, vpath_sub, trans):
        d["subject_id"] = d["subject_id"].astype(str)

    expected = set(beh.subject_id)
    for name, d in [("xi_subject_level", xi_sub), ("durations", dur),
                    ("vpath_subject_level", vpath_sub)]:
        got = set(d.subject_id)
        if got != expected:
            raise ValueError(f"{name} covers {len(got)} participants, not the "
                             f"{len(expected)} in the analysis sample; "
                             f"missing {sorted(expected - got)}, extra {sorted(got - expected)}")

    subject_level = (beh[["subject_id", "age", "sex", "matrix_T", "matrix_raw",
                          "income", "equivalised_income", "subjective_SES"]
                         + [c for c in beh.columns if c.startswith("SDQ_")]]
                     .merge(xi_sub[["subject_id", "T_samples", "entropy_rate",
                                    "entropy_rate_per_sample", "max_FO"]],
                            on="subject_id", validate="one_to_one")
                     .merge(vpath_sub[["subject_id", "n_switches", "switching_rate_vpath"]],
                            on="subject_id", validate="one_to_one")
                     .rename(columns={"switching_rate_vpath": "switching_rate"}))

    # Columns as published from the earlier 40-participant fit, for provenance only.
    orig = beh[["subject_id", "Entropy", "SwitchingRate", "MaxFO"]].rename(
        columns={"Entropy": "entropy_rate_orig40", "SwitchingRate": "switching_rate_orig40",
                 "MaxFO": "max_FO_orig40"})
    subject_level = subject_level.merge(orig, on="subject_id", validate="one_to_one")

    state_level = (xi_state.rename(columns={"FO_stationary": "FO"})
                   .merge(dur[["subject_id", "state", "n_visits", "FO_counted",
                               "lifetime_mean", "lifetime_median",
                               "interval_mean", "interval_median"]],
                          on=["subject_id", "state"], validate="one_to_one"))
    fo_orig = beh.melt(id_vars="subject_id", value_vars=[f"FO{s}" for s in range(1, NSTATES + 1)],
                       var_name="state", value_name="FO_orig40")
    fo_orig["state"] = fo_orig["state"].str.removeprefix("FO").astype(int)
    state_level = state_level.merge(fo_orig, on=["subject_id", "state"], validate="one_to_one")

    for name, df in [("subject_level", subject_level), ("state_level", state_level),
                     ("transition_matrices", trans)]:
        df = df.sort_values([c for c in ["subject_id", "state", "from_state", "to_state"]
                             if c in df.columns])
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv  {df.shape}")
    print(f"analysis sample: n = {len(subject_level)}  (excluded: {', '.join(EXCLUDED)})")


if __name__ == "__main__":
    main()
