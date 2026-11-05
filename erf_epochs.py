"""Step 11a: inventory the ERF epochs and separate real stimuli from phantom triggers.

Five recordings (99016, calmred02, calmred04, calmred05, calmred06) carry surplus
`Deviant2` epochs that were averaged into the stored ERFs. Every surplus epoch begins
20-28 ms after a `Standard` onset and none after a `Deviant1`; the standards they shadow
are indistinguishable from unshadowed standards, so no extra sound was played and the
surplus epochs are spurious trigger pulses, not pseudoword trials.

Real stimuli sit on a fixed lattice: a constant SOA near 1.22 s with no gaps, so every
onset is phase + k * period for integer k. A phantom sits tens of milliseconds off it.
The lattice is fitted to the `Standard` onsets alone, which are at design in all 40
recordings, and every epoch is then judged against it. Judging by lattice position rather
than by proximity to a neighbour matters because a phantom whose shadowed standard was
already dropped in preprocessing is left isolated: proximity misses it, the lattice does
not.

Writes:
  erf_epoch_inventory.csv  - one row per epoch, with its lattice index and residual
  erf_epoch_summary.csv    - one row per recording, with the checkpoint 1 quantities

Usage:
  python erf_epochs.py --spm-dir <SPM_Files> --out-dir <dir> [--tolerance 0.010]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

CONDITIONS = ("Standard", "Deviant1", "Deviant2")


def read_epochs(path: Path) -> pd.DataFrame:
    """Labels, onsets and bad flags for one epoched SPM MEEG file."""
    D = loadmat(str(path), squeeze_me=True, struct_as_record=False)["D"]
    trials = np.atleast_1d(D.trials)
    return pd.DataFrame({
        "label": [str(t.label) for t in trials],
        "onset": [float(t.onset) for t in trials],
        "bad": [int(getattr(t, "bad", 0)) for t in trials],
    }).sort_values("onset", ignore_index=True)


def chain_standards(onsets: np.ndarray, period: float, tol: float = 0.020,
                    pause_after: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk the standards, giving each a block and a beat number.

    Beat numbers are chained from one standard to the next rather than taken by dividing
    a whole session at once. A local step only has to be right to within half a period, so
    an error in the starting estimate cannot accumulate: dividing the span instead puts
    every stimulus past the point where the drift exceeds half a period onto the wrong
    beat. The SOA is near 1.2206 s and onsets are quantised to the 4 ms sample grid, so
    that drift arrives well before a 1,200-stimulus session is out.

    The walk steps *over* a stray rather than through it. A mistimed stimulus has a
    non-integer gap on both sides, and chaining through one is a coin flip that slips the
    beat for everything after it: calmred02 has a stray whose gaps are 3.048 s and 3.052 s,
    each about two and a half beats, while the two together are five beats to within 3 ms.
    Holding the anchor until a standard lands on a beat keeps the chain true across it.

    A run of consecutive misses is a pause, after which the stream resumes out of phase.
    Length does not identify one: 99016 resumes 278 ms out of phase after a gap of only
    2.8 beats, while plenty of honest gaps are longer. The tolerance is in milliseconds
    rather than in fractions of a period, because the error that matters is absolute;
    99134's 5.0 s gap is a tenth of a period from four beats but 117 ms away in time.
    """
    n = len(onsets)
    block = np.zeros(n, dtype=int)
    beat = np.zeros(n, dtype=float)
    anchored = np.zeros(n, dtype=bool)
    anchored[0] = True

    current, anchor_time, anchor_beat, missed = 0, onsets[0], 0.0, []
    i = 1
    while i < n:
        gap = onsets[i] - anchor_time
        steps = np.round(gap / period)
        if abs(gap - steps * period) <= tol:
            anchor_time, anchor_beat = onsets[i], anchor_beat + steps
            block[i], beat[i], anchored[i] = current, anchor_beat, True
            missed, i = [], i + 1
        elif len(missed) + 1 >= pause_after:
            current += 1
            i = missed[0]
            anchor_time, anchor_beat = onsets[i], 0.0
            block[i], beat[i], anchored[i] = current, 0.0, True
            missed, i = [], i + 1
        else:
            missed.append(i)
            block[i], beat[i] = current, anchor_beat + steps
            i += 1
    return block, beat, anchored


def fit_lattice(onsets: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """One shared period and a phase per block, from onsets known to lie on the lattice.

    The starting estimate is the median of the shorter gaps, which are single-SOA steps
    between adjacent standards; the longer ones are standards separated by a deviant.
    Period and phases are then solved together by least squares over the anchored
    standards only, the period shared across blocks because it belongs to the paradigm
    rather than to the block.
    """
    gaps = np.diff(onsets)
    period = float(np.median(gaps[gaps <= np.percentile(gaps, 40)]))
    for _ in range(3):
        block, beat, anchored = chain_standards(onsets, period)
        n_blocks = int(block.max()) + 1
        design = np.zeros((int(anchored.sum()), 1 + n_blocks))
        design[:, 0] = beat[anchored]
        design[np.arange(design.shape[0]), 1 + block[anchored]] = 1.0
        coefficients, *_ = np.linalg.lstsq(design, onsets[anchored], rcond=None)
        period = float(coefficients[0])
    bounds = np.array([[onsets[block == b].min(), onsets[block == b].max()]
                       for b in range(n_blocks)])
    return period, bounds, coefficients[1:]


def place_on_lattice(onsets: np.ndarray, period: float, bounds: np.ndarray,
                     phases: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block, beat and distance to that beat, for every onset."""
    outside = np.maximum(np.maximum(bounds[:, 0][None, :] - onsets[:, None],
                                    onsets[:, None] - bounds[:, 1][None, :]), 0.0)
    block = np.argmin(outside, axis=1)
    index = np.round((onsets - phases[block]) / period)
    return block, index.astype(int), onsets - (phases[block] + index * period)


def flag_spurious(block: np.ndarray, index: np.ndarray, residual: np.ndarray,
                  tolerance: float) -> np.ndarray:
    """Mark the epochs that cannot be real stimuli.

    A beat carries one stimulus, so a second epoch on a beat another already occupies is
    spurious whatever its lag, and the closer of the two to the beat is the real one. That
    catches the phantoms that fall near enough to a beat to pass the distance test on
    their own: their lag behind the standard they shadow runs from a few milliseconds to
    about fifty, so distance alone would let the earliest of them through.
    """
    spurious = np.abs(residual) > tolerance
    order = np.lexsort((np.abs(residual), index, block))
    beat = np.column_stack([block, index])[order]
    duplicate = np.r_[False, (beat[1:] == beat[:-1]).all(axis=1)]
    spurious[order[duplicate]] = True
    return spurious


def flag_crowded_deviants(labels: np.ndarray, block: np.ndarray, beat: np.ndarray,
                          spurious: np.ndarray, min_spacing: int = 3) -> np.ndarray:
    """Drop pseudowords that stand closer to another deviant than the design allows.

    A handful of phantoms survive the tests above: their shadowed standard was dropped
    during preprocessing, which leaves the beat free, and their lag is small enough to
    look like a real onset. Nothing in their timing gives them away.

    The sequence does. Across the 35 untouched recordings, 10,280 consecutive deviants
    are never closer than three beats, with no exceptions, so a pair at one or two beats
    cannot be two real stimuli. Every affected recording contains such pairs and no
    unaffected one does. The rule is applied only to `Deviant2`, because that is the
    condition the spurious triggers land in: `Deviant1` is at its design count in all 40
    recordings and no surplus epoch was ever seen to follow one.
    """
    spurious = spurious.copy()
    while True:
        alive = np.flatnonzero(~spurious & (labels != "Standard"))
        order = alive[np.lexsort((beat[alive], block[alive]))]
        crowded = np.flatnonzero((np.diff(block[order]) == 0)
                                 & (np.diff(beat[order]) < min_spacing))
        dropped = False
        for i in crowded:
            for candidate in (order[i + 1], order[i]):
                if labels[candidate] == "Deviant2":
                    spurious[candidate] = True
                    dropped = True
                    break
            if dropped:
                break
        if not dropped:
            return spurious


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.010,
                        help="seconds; an epoch further than this from a lattice point "
                             "is off-lattice")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.spm_dir.glob("edfilteredspmeeg_*_cleanica_sss.mat"))
    inventory, summary = [], []
    for path in paths:
        sid = re.search(r"edfilteredspmeeg_(.+?)_cleanica", path.name).group(1)
        epochs = read_epochs(path)

        standards = epochs.loc[epochs.label == "Standard", "onset"].to_numpy()
        period, bounds, phases = fit_lattice(standards)
        block, index, residual = place_on_lattice(epochs.onset.to_numpy(), period,
                                                  bounds, phases)
        labels = epochs.label.to_numpy()
        spurious = flag_spurious(block, index, residual, args.tolerance)
        crowded = flag_crowded_deviants(labels, block, index, spurious)
        epochs = epochs.assign(subject_id=sid, block=block, beat=index, residual=residual,
                               off_beat=spurious, spurious=crowded,
                               dropped_as_crowded=crowded & ~spurious)
        inventory.append(epochs)

        counts = {c: int((epochs.label == c).sum()) for c in CONDITIONS}
        dropped = {c: int(((epochs.label == c) & epochs.spurious).sum()) for c in CONDITIONS}
        real = epochs.loc[~epochs.spurious]
        std_residual = float(np.abs(real.loc[real.label == "Standard", "residual"]).max())
        kept_d2 = counts["Deviant2"] - dropped["Deviant2"]
        pauses = bounds[1:, 0] - bounds[:-1, 1]
        summary.append({
            "subject_id": sid, "period_s": period, "n_blocks": len(bounds),
            "longest_pause_s": float(pauses.max()) if len(pauses) else 0.0,
            "max_standard_residual_ms": 1000 * std_residual,
            "crowded_Deviant2": int(epochs.dropped_as_crowded.sum()),
            **{f"n_{c}": counts[c] for c in CONDITIONS},
            **{f"dropped_{c}": dropped[c] for c in CONDITIONS},
            "kept_Deviant2": kept_d2,
            "kept_matches_Deviant1": kept_d2 == counts["Deviant1"] - dropped["Deviant1"],
        })

    inventory = pd.concat(inventory, ignore_index=True)
    summary = pd.DataFrame(summary)
    inventory.to_csv(args.out_dir / "erf_epoch_inventory.csv", index=False)
    summary.to_csv(args.out_dir / "erf_epoch_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.max_rows", 60):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- checkpoint 1 -----------------------------------------------------
    AFFECTED = ["99016", "calmred02", "calmred04", "calmred05", "calmred06"]
    affected = summary.subject_id.isin(AFFECTED)
    print("\n=== checkpoint 1 ===")
    worst = summary.max_standard_residual_ms.max()
    print(f"surviving standards within {1000 * args.tolerance:.0f} ms of a beat in all "
          f"{len(summary)}: {worst <= 1000 * args.tolerance}  (worst {worst:.2f} ms)")
    clean = summary.loc[~affected]
    dropped_columns = [f"dropped_{c}" for c in CONDITIONS]
    print(f"unaffected recordings dropping nothing: "
          f"{int((clean[dropped_columns].sum(axis=1) == 0).sum())} of {len(clean)}; "
          f"most dropped by one of them: {int(clean[dropped_columns].sum(axis=1).max())}")
    # The two deviants are equally frequent by design, so in an affected recording the
    # surviving pseudowords should come back to the word-deviant count. It is not a test
    # for the other 35: there the two counts differ by however many epochs of each
    # condition preprocessing happened to discard, which is nobody's error.
    print(f"affected recordings where surviving Deviant2 equals surviving Deviant1: "
          f"{int(summary.loc[affected, 'kept_matches_Deviant1'].sum())} "
          f"of {int(affected.sum())}")
    failures = summary.loc[affected & ~summary.kept_matches_Deviant1, "subject_id"].tolist()
    print("affected recordings failing the count check:", failures if failures else "none")
    print(f"dropped by the sequence-spacing rule: "
          f"{int(summary.loc[affected, 'crowded_Deviant2'].sum())} in the affected five, "
          f"{int(summary.loc[~affected, 'crowded_Deviant2'].sum())} in the other 35")


if __name__ == "__main__":
    main()
