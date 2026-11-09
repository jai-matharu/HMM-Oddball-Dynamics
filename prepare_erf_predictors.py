"""Step 11f preparation: put the ERF amplitudes where `run_models.py` expects a predictor.

S4.6 of the supplement specifies the models already: the same 31 terms as every other
question, with the participant-level event-related measure in place of the
individual-difference measure, each measure corrected as its own family. So nothing new is
fitted here - the amplitudes from 11e are written into a copy of the analysis tables as
predictor columns and the existing runner is pointed at them.

Column names are rewritten because the window names carry characters a formula cannot
hold: `center(mag_Standard-Deviant1_536-672ms)` would be read as a subtraction.

Writes --out-dir as a drop-in replacement for the analysis directory, with the other
tables copied unchanged, plus `erf_predictor_names.csv` recording the mapping.

Usage:
  python prepare_erf_predictors.py --analysis-dir <analysis_39> --amplitudes <erf_amplitudes.csv>
                                   --out-dir <dir>
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

CONTRAST_NAMES = {"Standard-Deviant1": "SvD1", "Standard-Deviant2": "SvD2",
                  "Deviant1-Deviant2": "D1vD2"}


def safe_name(window: str) -> str:
    """A formula-safe predictor name for a window, keeping what identifies it."""
    match = re.fullmatch(r"(mag|grad)_([A-Za-z0-9-]+?)_(-?\d+)-(-?\d+)ms", window)
    if match is None:
        raise ValueError(f"cannot parse the window name {window!r}")
    sensors, contrast, start, end = match.groups()
    if contrast not in CONTRAST_NAMES:
        raise ValueError(f"unknown contrast {contrast!r}")
    return f"erf_{sensors}_{CONTRAST_NAMES[contrast]}_{start}_{end}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--amplitudes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    amplitudes = pd.read_csv(args.amplitudes, dtype={"subject_id": str})
    windows = [c for c in amplitudes.columns
               if c != "subject_id" and not c.endswith("__uncorrected")]
    renamed = {window: safe_name(window) for window in windows}
    if len(set(renamed.values())) != len(renamed):
        raise ValueError("two windows map to the same predictor name")

    subject = pd.read_csv(args.analysis_dir / "subject_level.csv", dtype={"subject_id": str})
    merged = subject.merge(amplitudes[["subject_id"] + windows].rename(columns=renamed),
                           on="subject_id", how="left", validate="one_to_one")
    if merged[list(renamed.values())].isna().any().any():
        missing = sorted(set(subject.subject_id) - set(amplitudes.subject_id))
        raise ValueError(f"no amplitude for {missing}")
    merged.to_csv(args.out_dir / "subject_level.csv", index=False)

    for name in ("state_level.csv", "transition_matrices.csv", "ses_imputation_flags.csv"):
        source = args.analysis_dir / name
        if source.exists():
            shutil.copy2(source, args.out_dir / name)

    mapping = pd.DataFrame({"window": windows,
                            "predictor": [renamed[w] for w in windows]})
    mapping.to_csv(args.out_dir / "erf_predictor_names.csv", index=False)
    print(f"{len(merged)} participants, {len(windows)} predictors")
    print(mapping.to_string(index=False))


if __name__ == "__main__":
    main()
