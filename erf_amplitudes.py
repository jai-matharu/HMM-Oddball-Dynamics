"""Step 11e: participant-level amplitudes over the clusters found in 11d.

One amplitude per participant per window: the mean of that participant's difference wave
over the cluster's own sensor x sample mask, which is the cluster as the test defined it
rather than a rectangle drawn around it.

A window exists only where the test's smallest cluster p survived the BH correction across
the six tests of 11d - the resolution recorded in the working document before this step
ran. A test that failed BH contributes nothing.

The check this step exists for: the five recordings that carried the phantom triggers must
not stand apart from the other 34 once the phantoms are gone. If they do, something of the
artefact is still in the averages.

Writes to --out-dir:
  erf_amplitudes.csv         - participant x window, corrected and uncorrected
  erf_amplitude_checks.csv   - per window: distribution shape, and the five against the rest

Usage:
  python erf_amplitudes.py --averages <erf_averages.npz> --clusters <erf_clusters.npz>
                           --tests <erf_cluster_tests.csv> --cluster-table <erf_clusters.csv>
                           --out-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AFFECTED = ("99016", "calmred02", "calmred04", "calmred05", "calmred06")
CHANNEL_TYPES = {"mag": "MEGMAG", "grad": "MEGPLANAR"}


def difference_waves(store, order, post, first: str, second: str, which: str) -> np.ndarray:
    """One participant x sample x sensor difference wave stack, as 11d built it."""
    conditions = [str(c) for c in store["condition"]]
    averages = store[which]
    return (averages[:, conditions.index(first)][:, order][:, :, post]
            - averages[:, conditions.index(second)][:, order][:, :, post]).transpose(0, 2, 1)


def displacement(values: np.ndarray, affected: np.ndarray,
                 n_permutations: int = 10_000, seed: int = 11) -> tuple[float, float]:
    """Difference in mean between the five and the rest, with an exact-style permutation p.

    A five-against-34 comparison has no business leaning on a t distribution, so the null
    is built by relabelling which five participants are called affected.
    """
    observed = values[affected].mean() - values[~affected].mean()
    rng = np.random.default_rng(seed)
    n = len(values)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        picked = np.zeros(n, dtype=bool)
        picked[rng.choice(n, affected.sum(), replace=False)] = True
        null[i] = values[picked].mean() - values[~picked].mean()
    p = (1 + np.sum(np.abs(null) >= np.abs(observed))) / (1 + n_permutations)
    return float(observed), float(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--averages", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--cluster-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    averages = np.load(args.averages, allow_pickle=False)
    clusters = np.load(args.clusters, allow_pickle=False)
    tests = pd.read_csv(args.tests)
    table = pd.read_csv(args.cluster_table)

    subjects = np.array([str(s) for s in clusters["subject_id"]])
    all_subjects = np.array([str(s) for s in averages["subject_id"]])
    keep = np.array([list(all_subjects).index(s) for s in subjects])
    time = clusters["time"]
    post = np.isin(np.round(averages["time"], 6), np.round(time, 6))

    # Channel order has to match 11d's, which followed the adjacency template.
    import mne
    labels = [str(label) for label in averages["channel_label"]]
    kinds = ["mag" if str(t) == "MEGMAG" else "grad" for t in averages["channel_type"]]
    info = mne.create_info(labels, 250.0, kinds)
    orders = {}
    for channel_type in CHANNEL_TYPES:
        _, names = mne.channels.find_ch_adjacency(info, channel_type)
        orders[channel_type] = ([labels.index(str(name)) for name in names],
                                [str(name) for name in names])

    defined = tests.loc[tests.smallest_p_BH < args.alpha]
    print(f"tests whose smallest cluster p survives BH: {len(defined)} of {len(tests)}")
    for row in tests.itertuples():
        verdict = "defines windows" if row.smallest_p_BH < args.alpha else "defines none"
        print(f"  {row.channel_type:>4} {row.contrast:<19} p {row.smallest_p:.4f}  "
              f"BH {row.smallest_p_BH:.4f}  {verdict}")

    amplitudes, checks = {}, []
    for test in defined.itertuples():
        order, names = orders[test.channel_type]
        first, second = test.contrast.split("-")
        waves = {which: difference_waves(averages, order, post, first, second, which)[keep]
                 for which in ("corrected", "all_epochs")}
        masks = clusters[f"mask_{test.channel_type}_{test.contrast}"]
        rows = table.loc[(table.channel_type == test.channel_type)
                         & (table.contrast == test.contrast)
                         & table.surviving].sort_values("p").reset_index(drop=True)
        if len(masks) != len(rows):
            raise ValueError(f"{test.channel_type} {test.contrast}: {len(masks)} masks "
                             f"against {len(rows)} surviving rows")

        for i, mask in enumerate(masks):
            name = (f"{test.channel_type}_{test.contrast}_"
                    f"{rows.start_ms[i]:.0f}-{rows.end_ms[i]:.0f}ms")
            values = {which: wave[:, mask].mean(axis=1) for which, wave in waves.items()}
            amplitudes[name] = values["corrected"]
            amplitudes[f"{name}__uncorrected"] = values["all_epochs"]

            affected = np.isin(subjects, AFFECTED)
            gap, p = displacement(values["corrected"], affected)
            gap_before, p_before = displacement(values["all_epochs"], affected)
            shift = values["corrected"][affected] - values["all_epochs"][affected]
            checks.append({
                "window": name, "channel_type": test.channel_type,
                "contrast": test.contrast, "cluster_p": float(rows.p[i]),
                "sign": rows.sign[i], "n_sensors": int(rows.n_sensors[i]),
                "n_samples": int(rows.n_samples[i]),
                "mean": float(values["corrected"].mean()),
                "sd": float(values["corrected"].std(ddof=1)),
                "skew": float(stats.skew(values["corrected"])),
                "kurtosis": float(stats.kurtosis(values["corrected"])),
                "shapiro_p": float(stats.shapiro(values["corrected"]).pvalue),
                "beyond_3sd": int((np.abs(stats.zscore(values["corrected"])) > 3).sum()),
                "five_minus_rest": gap,
                "five_minus_rest_in_sd": gap / float(values["corrected"].std(ddof=1)),
                "five_minus_rest_p": p,
                "five_minus_rest_before": gap_before,
                "five_minus_rest_p_before": p_before,
                "correction_to_the_five_in_sd": float(np.abs(shift).max()
                                                      / values["corrected"].std(ddof=1)),
            })

    frame = pd.DataFrame(amplitudes, index=pd.Index(subjects, name="subject_id"))
    frame.to_csv(args.out_dir / "erf_amplitudes.csv")
    checks = pd.DataFrame(checks)
    checks.to_csv(args.out_dir / "erf_amplitude_checks.csv", index=False)

    # --- checkpoint 5 -----------------------------------------------------
    print("\n=== checkpoint 5 ===")
    print(f"{len(checks)} windows, {len(subjects)} participants.\n")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print("distribution of each window's amplitudes:")
        print(checks[["window", "cluster_p", "mean", "sd", "skew", "kurtosis",
                      "shapiro_p", "beyond_3sd"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nthe five formerly contaminated against the other 34:")
        print(checks[["window", "five_minus_rest", "five_minus_rest_in_sd",
                      "five_minus_rest_p", "five_minus_rest_p_before",
                      "correction_to_the_five_in_sd"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwindows where the five are displaced at p < {args.alpha}: "
          f"{int((checks.five_minus_rest_p < args.alpha).sum())} of {len(checks)}")
    print(f"the same before the correction: "
          f"{int((checks.five_minus_rest_p_before < args.alpha).sum())} of {len(checks)}")
    print(f"largest displacement after correction: "
          f"{checks.five_minus_rest_in_sd.abs().max():.3f} SD "
          f"({checks.loc[checks.five_minus_rest_in_sd.abs().idxmax(), 'window']})")
    print(f"windows whose amplitudes depart from normality at p < 0.05 (Shapiro): "
          f"{int((checks.shapiro_p < 0.05).sum())} of {len(checks)}; "
          f"participants beyond 3 SD anywhere: {int(checks.beyond_3sd.sum())}")


if __name__ == "__main__":
    main()
