"""Step 11b: re-average the ERF epochs, with the phantom triggers left out.

The stored `medfilteredspmeeg_*` averages were made by `spm_eeg_average` with `robust = 0`
and no trial marked bad, so each is a plain mean over every epoch of its condition. For
five recordings that mean includes spurious `Deviant2` epochs that are standard-trial data
(see `erf_epochs.py`), which is what this step removes.

Two averages are computed per recording and condition: one over all epochs, which should
reproduce the stored file exactly, and one over the epochs `erf_epochs.py` kept. The first
exists only to test the re-averaging - if it does not reproduce a recording's stored
average, the second cannot be trusted either.

Writes to --out-dir:
  erf_averages.npz        - corrected (n_subjects, 3, 306, 251) and the all-epoch average,
                            with channel labels, types and positions and the time vector
  erf_average_checks.csv  - one row per recording and condition: epochs in each average
                            and the largest absolute departure from the stored file

Usage:
  python erf_average.py --spm-dir <SPM_Files> --inventory <erf_epoch_inventory.csv>
                        --out-dir <dir>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

CONDITIONS = ("Standard", "Deviant1", "Deviant2")


def read_header(path: Path):
    """The trial list, channel table and data layout of one SPM MEEG file."""
    D = loadmat(str(path), squeeze_me=True, struct_as_record=False)["D"]
    trials = np.atleast_1d(D.trials)
    channels = np.atleast_1d(D.channels)
    return {
        "labels": np.array([str(t.label) for t in trials]),
        "onsets": np.array([float(t.onset) for t in trials]),
        "repl": np.array([int(getattr(t, "repl", 1)) for t in trials]),
        "dim": np.asarray(D.data.dim, dtype=int),
        "dat": Path(str(D.data.fname)),
        "offset": int(D.data.offset),
        "fsample": float(D.Fsample),
        "time_onset": float(D.timeOnset),
        "channel_label": np.array([str(c.label) for c in channels]),
        "channel_type": np.array([str(c.type) for c in channels]),
        "channel_bad": np.array([int(c.bad) for c in channels]),
        "chanpos": np.asarray(D.sensors.meg.chanpos, dtype=float),
        "sensor_label": np.array([str(s) for s in np.atleast_1d(D.sensors.meg.label)]),
    }


def open_data(header, directory: Path) -> np.memmap:
    """The epochs as (channels, samples, trials), read in place rather than loaded.

    A session is 363 MB of float32 written in Fortran order, so the condition means are
    accumulated straight off the disk.
    """
    path = directory / header["dat"].name
    return np.memmap(path, dtype="<f4", mode="r", offset=header["offset"],
                     shape=tuple(header["dim"]), order="F")


def order_as_inventory(header, subject: pd.DataFrame) -> np.ndarray:
    """The trial index the inventory gives each of its rows, checked against the file.

    The inventory carries the index into the `.dat` rather than being matched back by
    onset, because an onset does not identify a trial: in three recordings a phantom is
    quantised onto the same sample as the standard it shadows.
    """
    order = subject.trial.to_numpy()
    if len(np.unique(order)) != len(order) or len(order) != len(header["labels"]):
        raise ValueError("inventory trial indices are not a permutation of the trials")
    if not np.array_equal(header["labels"][order], subject.label.to_numpy()):
        raise ValueError("labels do not match the inventory")
    if not np.allclose(header["onsets"][order], subject.onset.to_numpy(), atol=1e-9):
        raise ValueError("onsets do not match the inventory")
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(args.inventory, dtype={"subject_id": str})
    paths = sorted(args.spm_dir.glob("edfilteredspmeeg_*_cleanica_sss.mat"))

    subjects, corrected, every, checks, dropped_epochs = [], [], [], [], []
    channels = None
    for path in paths:
        sid = re.search(r"edfilteredspmeeg_(.+?)_cleanica", path.name).group(1)
        subject = inventory.loc[inventory.subject_id == sid].reset_index(drop=True)
        header = read_header(path)
        order = order_as_inventory(header, subject)
        data = open_data(header, args.spm_dir)

        stored = read_header(args.spm_dir / f"medfilteredspmeeg_{sid}_cleanica_sss.mat")
        stored_data = open_data(stored, args.spm_dir)

        this_corrected = np.zeros((len(CONDITIONS), header["dim"][0], header["dim"][1]))
        this_every = np.zeros_like(this_corrected)
        for c, condition in enumerate(CONDITIONS):
            in_condition = subject.label.to_numpy() == condition
            all_trials = order[in_condition]
            kept = order[in_condition & ~subject.spurious.to_numpy()]
            this_every[c] = data[:, :, np.sort(all_trials)].mean(axis=2)
            this_corrected[c] = data[:, :, np.sort(kept)].mean(axis=2)

            # The stored file's own trial order, not an assumed one.
            where = np.flatnonzero(stored["labels"] == condition)
            if len(where) != 1:
                raise ValueError(f"{sid}: {condition} appears {len(where)} times in the "
                                 "stored average")
            reference = np.asarray(stored_data[:, :, where[0]], dtype=float)
            difference = np.abs(this_every[c] - reference)
            scale = np.abs(reference).max()

            # How large was each epoch that 11a dropped? An epoch mistimed by tens of
            # milliseconds could still be an ordinary response, in which case dropping it
            # costs a trial; if instead it is far outside the condition's amplitude
            # distribution, dropping it is no loss at all. Measured, not assumed.
            epochs = np.asarray(data[:, :, np.sort(kept)], dtype=float)
            peaks = np.abs(epochs).max(axis=(0, 1))
            # Peak amplitude is the wrong yardstick for a change in an average: removing
            # one epoch from 883 moves the mean by about a 883rd of that epoch, and a
            # single epoch is mostly noise, so a change that looks large beside the
            # averaged peak can still be far below the noise on the average. The scale
            # that decides whether the change matters is the standard error of the mean.
            sem = epochs.std(axis=2, ddof=1) / np.sqrt(epochs.shape[2])
            usable = sem > 0
            relative_to_sem = float((np.abs(this_corrected[c] - this_every[c])[usable]
                                     / sem[usable]).max())
            del epochs
            for trial in np.sort(np.setdiff1d(all_trials, kept)):
                dropped_epochs.append({
                    "subject_id": sid, "condition": condition, "trial": int(trial),
                    "epoch_peak": float(np.abs(data[:, :, trial]).max()),
                    "kept_median_peak": float(np.median(peaks)),
                    "kept_max_peak": float(peaks.max()),
                    "residual_ms": float(1000 * subject.residual[subject.trial == trial].iloc[0]),
                })
            checks.append({
                "subject_id": sid, "condition": condition,
                "n_epochs": int(in_condition.sum()), "n_kept": len(kept),
                "stored_repl": int(stored["repl"][where[0]]),
                "max_abs_difference": float(difference.max()),
                "relative_to_peak": float(difference.max() / scale),
                "max_abs_correction": float(np.abs(this_corrected[c]
                                                   - this_every[c]).max() / scale),
                "correction_in_sem": relative_to_sem,
                "peak": float(scale),
            })

        subjects.append(sid)
        corrected.append(this_corrected)
        every.append(this_every)
        if channels is None:
            channels = header
        del data, stored_data

    checks = pd.DataFrame(checks)
    checks["dropped"] = checks.n_epochs - checks.n_kept
    checks.to_csv(args.out_dir / "erf_average_checks.csv", index=False)
    dropped_epochs = pd.DataFrame(dropped_epochs)
    dropped_epochs["relative_to_kept_median"] = (dropped_epochs.epoch_peak
                                                 / dropped_epochs.kept_median_peak)
    dropped_epochs.to_csv(args.out_dir / "erf_dropped_epochs.csv", index=False)

    n_samples = int(channels["dim"][1])
    np.savez_compressed(
        args.out_dir / "erf_averages.npz",
        subject_id=np.array(subjects),
        condition=np.array(CONDITIONS),
        corrected=np.stack(corrected).astype(np.float32),
        all_epochs=np.stack(every).astype(np.float32),
        n_kept=checks.pivot(index="subject_id", columns="condition",
                            values="n_kept").reindex(subjects)[list(CONDITIONS)].to_numpy(),
        time=channels["time_onset"] + np.arange(n_samples) / channels["fsample"],
        channel_label=channels["channel_label"],
        channel_type=channels["channel_type"],
        channel_bad=channels["channel_bad"],
        chanpos=channels["chanpos"],
        sensor_label=channels["sensor_label"],
    )

    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.max_rows", 140):
        print(checks.to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    # --- checkpoint 2 -----------------------------------------------------
    AFFECTED = ["99016", "calmred02", "calmred04", "calmred05", "calmred06"]
    # 99114 is outside the HMM analysis sample, so its averages do not enter anything.
    OUTSIDE = ["99114"]
    affected = checks.subject_id.isin(AFFECTED)
    print("\n=== checkpoint 2 ===")
    worst = checks.loc[checks.relative_to_peak.idxmax()]
    print(f"the all-epoch average reproduces the stored medfilteredspmeeg file: worst "
          f"departure {worst.relative_to_peak:.3e} of peak, {worst.max_abs_difference:.3e} "
          f"fT ({worst.subject_id} {worst.condition}) - float32 rounding")
    print("epochs averaged equals the stored repl in every recording and condition: "
          f"{bool((checks.n_epochs == checks.stored_repl).all())}")

    # The real test of the re-averaging: where 11a dropped nothing, the corrected average
    # must be the stored one exactly.
    untouched = checks.loc[checks.dropped == 0]
    print(f"\nwhere no epoch was dropped ({len(untouched)} of {len(checks)} recording x "
          f"condition): largest movement {untouched.max_abs_correction.max():.3e} of peak")

    moved = checks.loc[(checks.dropped > 0) & ~affected]
    inside = moved.loc[~moved.subject_id.isin(OUTSIDE)]
    print(f"\nunaffected recordings where 11a dropped an epoch: {len(moved)} rows, "
          f"{moved.subject_id.nunique()} recordings")
    print(f"  movement, median {inside.max_abs_correction.median():.3e} of peak, "
          f"largest {inside.max_abs_correction.max():.3e} "
          f"({inside.loc[inside.max_abs_correction.idxmax(), 'subject_id']} "
          f"{inside.loc[inside.max_abs_correction.idxmax(), 'condition']}), "
          f"within the analysis sample")
    print(f"  the same movement against the standard error of the average: median "
          f"{inside.correction_in_sem.median():.3f}, largest "
          f"{inside.correction_in_sem.max():.3f} SEM")
    for sid in OUTSIDE:
        row = moved.loc[moved.subject_id == sid]
        if len(row):
            print(f"  {sid}, outside the analysis sample: "
                  f"{row.max_abs_correction.max():.3e} of peak")

    print(f"\nthe five affected recordings, Deviant2 movement: "
          + ", ".join(f"{r.subject_id} {r.max_abs_correction:.3f}"
                      for r in checks.loc[affected & (checks.condition == "Deviant2")]
                      .itertuples()))
    held = checks.loc[affected & (checks.condition != "Deviant2")]
    print(f"\nthe five affected recordings, Deviant2 movement in SEM: "
          + ", ".join(f"{r.subject_id} {r.correction_in_sem:.1f}"
                      for r in checks.loc[affected & (checks.condition == "Deviant2")]
                      .itertuples()))
    print(f"the five affected recordings, Standard and Deviant1 movement: "
          f"largest {held.max_abs_correction.max():.3e} of peak, from the "
          f"{int(held.dropped.sum())} mistimed standards 11a dropped in three of them; "
          f"{int((held.dropped == 0).sum())} of {len(held)} are unchanged")

    # --- what the dropped epochs were ------------------------------------
    print("\n=== the dropped epochs ===")
    ratio = dropped_epochs.relative_to_kept_median
    print(f"{len(dropped_epochs)} epochs dropped in all. Peak amplitude relative to the "
          f"median kept epoch of the same condition:")
    print(f"  median {ratio.median():.2f}, 90th centile {ratio.quantile(0.9):.2f}, "
          f"max {ratio.max():.2f}")
    print(f"  above the largest kept epoch of their own condition: "
          f"{int((dropped_epochs.epoch_peak > dropped_epochs.kept_max_peak).sum())} "
          f"of {len(dropped_epochs)}")
    phantom = dropped_epochs.subject_id.isin(AFFECTED) & (dropped_epochs.condition
                                                          == "Deviant2")
    print(f"  phantoms ({int(phantom.sum())}): median {ratio[phantom].median():.2f}")
    print(f"  mistimed real stimuli ({int((~phantom).sum())}): median "
          f"{ratio[~phantom].median():.2f}, max {ratio[~phantom].max():.2f}")
    worst_drop = dropped_epochs.loc[ratio.idxmax()]
    print(f"  largest: {worst_drop.subject_id} {worst_drop.condition} trial "
          f"{int(worst_drop.trial)}, {worst_drop.relative_to_kept_median:.1f}x the median "
          f"kept epoch, {worst_drop.residual_ms:+.1f} ms off its beat")


if __name__ == "__main__":
    main()
