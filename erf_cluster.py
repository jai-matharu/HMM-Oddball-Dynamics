"""Step 11d: cluster-based permutation tests on the re-derived ERFs.

Run exactly as specified in Step 11c of the working document before any test was run:
three contrasts, magnetometers and gradiometers separately, two-tailed one-sample
sign-flipping cluster permutation over sensors x time, 0-800 ms, 10,000 permutations,
cluster-forming |t| at the two-tailed parametric 0.05 on the sample's own degrees of
freedom, MNE's shipped neuromag306 adjacency, seed 11.

The six tests are definitional rather than confirmatory: their job is to locate the
windows that Step 11e measures per participant. They see only the within-participant
condition contrast and nothing about any participant-level variable, so the associations
tested in 11f are not selected on their own outcome.

Writes to --out-dir:
  erf_clusters.csv      - one row per cluster, with its permutation p, extent and peak
  erf_cluster_tests.csv - one row per test, with its smallest cluster p and the BH value
  erf_clusters.npz      - the t maps and the surviving clusters' masks, for 11e and the figure

Usage:
  python erf_cluster.py --averages <erf_averages.npz> --sample <subject_level.csv>
                        --out-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.stats import spatio_temporal_cluster_1samp_test
from scipy import stats
from statsmodels.stats.multitest import multipletests

CONTRASTS = (("Standard", "Deviant1"), ("Standard", "Deviant2"), ("Deviant1", "Deviant2"))
CHANNEL_TYPES = {"mag": "MEGMAG", "grad": "MEGPLANAR"}
# Kept only for the descriptive comparison the checkpoint asks for. Nothing is selected
# with them.
PUBLISHED_WINDOWS = {"mismatch (548-600)": (0.548, 0.600),
                     "mismatch (512-668)": (0.512, 0.668),
                     "P3 (692-800)": (0.692, 0.800)}


def build_info(labels: np.ndarray, types: np.ndarray, fsample: float) -> mne.Info:
    """A minimal MNE Info, enough for `find_ch_adjacency` to match by channel name."""
    kinds = ["mag" if t == "MEGMAG" else "grad" for t in types]
    return mne.create_info([str(label) for label in labels], fsample, kinds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--averages", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True,
                        help="subject_level.csv of the analysis sample")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    store = np.load(args.averages, allow_pickle=False)
    subjects = np.array([str(s) for s in store["subject_id"]])
    conditions = [str(c) for c in store["condition"]]
    time = store["time"]

    sample = pd.read_csv(args.sample, dtype={"subject_id": str}).subject_id.to_numpy()
    keep = np.isin(subjects, sample)
    if sorted(subjects[keep]) != sorted(sample):
        raise ValueError("the averages do not cover the analysis sample")
    averages = store["corrected"][keep]                     # subjects x 3 x chan x time
    n_subjects = int(keep.sum())
    print(f"analysis sample {n_subjects}; left out of the averages: "
          f"{sorted(set(subjects) - set(sample))}")

    post = time >= 0.0
    threshold = float(stats.t.ppf(1 - args.alpha / 2, n_subjects - 1))
    print(f"window {1000 * time[post][0]:.0f} to {1000 * time[post][-1]:.0f} ms, "
          f"{int(post.sum())} samples; cluster-forming |t| >= {threshold:.4f} "
          f"on df = {n_subjects - 1}")

    info = build_info(store["channel_label"], store["channel_type"], 250.0)
    rows, tests, saved = [], [], {}
    for channel_type, spm_type in CHANNEL_TYPES.items():
        adjacency, adjacency_names = mne.channels.find_ch_adjacency(info, channel_type)
        # The adjacency comes back in the template's own channel order, so the data are
        # reordered to match it rather than the other way round.
        order = [list(store["channel_label"]).index(name) for name in adjacency_names]
        if not all(store["channel_type"][i] == spm_type for i in order):
            raise ValueError(f"{channel_type}: adjacency names do not match SPM types")

        for first, second in CONTRASTS:
            name = f"{first}-{second}"
            difference = (averages[:, conditions.index(first)][:, order][:, :, post]
                          - averages[:, conditions.index(second)][:, order][:, :, post])
            X = np.ascontiguousarray(difference.transpose(0, 2, 1), dtype=float)

            t_obs, clusters, cluster_p, _ = spatio_temporal_cluster_1samp_test(
                X, threshold=threshold, n_permutations=args.n_permutations, tail=0,
                adjacency=adjacency, n_jobs=args.n_jobs, seed=args.seed,
                out_type="mask", verbose="error")
            print(f"\n{channel_type:>4} {name:<19} {len(clusters)} candidate clusters; "
                  f"smallest p {cluster_p.min() if len(cluster_p) else 1.0:.4f}")

            masks = []
            for i in np.argsort(cluster_p):
                mask = clusters[i]
                mass = float(t_obs[mask].sum())
                samples = np.flatnonzero(mask.any(axis=1))
                sensors = np.flatnonzero(mask.any(axis=0))
                peak = np.unravel_index(np.abs(np.where(mask, t_obs, 0)).argmax(),
                                        mask.shape)
                row = {
                    "channel_type": channel_type, "contrast": name,
                    "p": float(cluster_p[i]),
                    "surviving": bool(cluster_p[i] < args.alpha),
                    "sign": "positive" if mass > 0 else "negative",
                    "cluster_mass": mass, "n_sensors": len(sensors),
                    "n_samples": len(samples),
                    "start_ms": float(1000 * time[post][samples[0]]),
                    "end_ms": float(1000 * time[post][samples[-1]]),
                    "peak_t": float(t_obs[peak]),
                    "peak_ms": float(1000 * time[post][peak[0]]),
                    "peak_sensor": str(adjacency_names[peak[1]]),
                }
                for label, (start, end) in PUBLISHED_WINDOWS.items():
                    window = (time[post] >= start) & (time[post] <= end)
                    row[f"overlap_{label}"] = float(
                        mask[window].any(axis=1).sum() / max(window.sum(), 1))
                rows.append(row)
                if cluster_p[i] < args.alpha:
                    masks.append(mask)

            saved[f"t_{channel_type}_{name}"] = t_obs.astype(np.float32)
            saved[f"mask_{channel_type}_{name}"] = (np.stack(masks) if masks
                                                    else np.zeros((0,) + t_obs.shape,
                                                                  dtype=bool))
            tests.append({"channel_type": channel_type, "contrast": name,
                          "n_candidate_clusters": len(clusters),
                          "smallest_p": float(cluster_p.min()) if len(cluster_p) else 1.0,
                          "n_surviving": int((cluster_p < args.alpha).sum())})

    rows = pd.DataFrame(rows)
    tests = pd.DataFrame(tests)
    _, bh, _, _ = multipletests(tests.smallest_p.to_numpy(), alpha=args.alpha,
                                method="fdr_bh")
    tests["smallest_p_BH"] = bh
    rows.to_csv(args.out_dir / "erf_clusters.csv", index=False)
    tests.to_csv(args.out_dir / "erf_cluster_tests.csv", index=False)
    np.savez_compressed(args.out_dir / "erf_clusters.npz",
                        time=time[post], channel_type=np.array(list(CHANNEL_TYPES)),
                        subject_id=subjects[keep], threshold=threshold, **saved)

    # --- checkpoint 4 -----------------------------------------------------
    print("\n=== checkpoint 4 ===")
    print(f"{n_subjects} participants, {args.n_permutations} permutations, seed "
          f"{args.seed}, |t| >= {threshold:.4f}, window 0-800 ms, as specified in 11c.")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print("\nper test:")
        print(tests.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        surviving = rows.loc[rows.surviving]
        print(f"\nsurviving clusters ({len(surviving)}):")
        if len(surviving):
            columns = ["channel_type", "contrast", "p", "sign", "start_ms", "end_ms",
                       "n_sensors", "n_samples", "peak_t", "peak_ms", "peak_sensor"]
            print(surviving[columns].to_string(index=False,
                                               float_format=lambda v: f"{v:.3f}"))
            print("\nfraction of each published window a surviving cluster covers "
                  "(descriptive only; nothing was selected with these):")
            print(surviving[["channel_type", "contrast", "p"]
                            + [f"overlap_{k}" for k in PUBLISHED_WINDOWS]]
                  .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        else:
            print("  none - per 11c, no window is defined and no threshold is loosened.")


if __name__ == "__main__":
    main()
