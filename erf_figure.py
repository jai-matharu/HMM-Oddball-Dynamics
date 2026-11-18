"""Step 11g: the event-related figure.

One row per window defined in 11d/11e: the grand-average waveforms of the two conditions
over the cluster's sensors (mean +/- SEM across participants, cluster span shaded), and the
topography of t averaged over the cluster's samples with the cluster's sensors marked.
Gradiometer pairs are combined by root-mean-square, so their maps show |t|. A last panel
gives t for every term of the five 11f runs, outlining the terms significant under one BH
correction across all of them.

Reads the averages in physical units written by `erf_rescale.py` (fT, fT/mm), and the
cluster and model outputs of the run on those averages.

Usage:
  python erf_figure.py --erf-dir <dir with erf_averages_fT.npz and the 11d outputs>
                       --models-dir <11f terms_*.csv>
                       --hmm-meg <HMM_MEG repo> --output-dir <figures dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle

CONDITION_LABEL = {"Standard": "Standard", "Deviant1": "Deviant 1", "Deviant2": "Deviant 2"}
CHANNEL_TYPES = {"mag": "MEGMAG", "grad": "MEGPLANAR"}
UNIT = {"mag": "fT", "grad": "fT/mm"}
TYPE_LABEL = {"mag": "Magnetometers", "grad": "Gradiometers"}
MEASURES = (("FO", "Occupancy"), ("transition_into_sum", "Transitions into"),
            ("interval_mean", "Interval"), ("lifetime_mean", "Lifetime"))
ENTROPY = (("entropy_rate|measure", "M"), ("entropy_rate|age", "A"),
           ("entropy_rate|age_x_measure", "A×M"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erf-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--hmm-meg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="figure_9_event_related")
    args = parser.parse_args()

    sys.path.insert(0, str(args.hmm_meg))
    from analysis.viz import style

    style.apply_style()
    averages = np.load(args.erf_dir / "erf_averages_fT.npz", allow_pickle=False)
    clusters = np.load(args.erf_dir / "erf_clusters.npz", allow_pickle=False)
    tests = pd.read_csv(args.erf_dir / "erf_cluster_tests.csv")
    table = pd.read_csv(args.erf_dir / "erf_clusters.csv")

    subjects_all = [str(s) for s in averages["subject_id"]]
    subjects = [str(s) for s in clusters["subject_id"]]
    keep = np.array([subjects_all.index(s) for s in subjects])
    conditions = [str(c) for c in averages["condition"]]
    data = averages["corrected"][keep]                      # subj x cond x chan x time
    time_ms = 1000 * averages["time"]
    post = np.isin(np.round(averages["time"], 6), np.round(clusters["time"], 6))
    labels = [str(label) for label in averages["channel_label"]]
    kinds = ["mag" if str(t) == "MEGMAG" else "grad" for t in averages["channel_type"]]

    # Channel order of the saved masks follows the adjacency template, as in 11d and 11e.
    info_plain = mne.create_info(labels, 250.0, kinds)
    orders = {}
    for key in CHANNEL_TYPES:
        _, names = mne.channels.find_ch_adjacency(info_plain, key)
        orders[key] = [str(name) for name in names]

    # Standard Neuromag layout positions, identical for every participant, rather than one
    # participant's sensor geometry, which would carry that child's head position.
    layout = mne.channels.read_layout("Vectorview-mag")
    lay_names = [name.replace(" ", "") for name in layout.names]
    mag_names = orders["mag"]
    centres = layout.pos[:, :2] + layout.pos[:, 2:] / 2
    mag_pos = np.array([centres[lay_names.index(name)] for name in mag_names])
    mag_pos = mag_pos - mag_pos.mean(axis=0)
    mag_pos = mag_pos / np.abs(mag_pos).max() * 0.085

    windows = []
    for test in tests.loc[tests.smallest_p_BH < 0.05].itertuples():
        masks = clusters[f"mask_{test.channel_type}_{test.contrast}"]
        rows = table.loc[(table.channel_type == test.channel_type)
                         & (table.contrast == test.contrast)
                         & table.surviving].sort_values("p").reset_index(drop=True)
        t_map = clusters[f"t_{test.channel_type}_{test.contrast}"]
        for i, mask in enumerate(masks):
            windows.append(dict(channel_type=test.channel_type, contrast=test.contrast,
                                mask=mask, row=rows.iloc[i], t_map=t_map,
                                p_bh=float(test.smallest_p_BH)))
    contrast_order = ["Standard-Deviant1", "Standard-Deviant2", "Deviant1-Deviant2"]
    windows.sort(key=lambda w: (list(CHANNEL_TYPES).index(w["channel_type"]),
                                contrast_order.index(w["contrast"]),
                                w["row"].sign != "positive"))

    colours = {"Standard": style.COLORS["blue"], "Deviant1": style.COLORS["orange"],
               "Deviant2": style.COLORS["vermillion"]}

    n = len(windows)
    fig = plt.figure(figsize=(14.0, 3.3 * n + 5.0))
    grid = GridSpec(n + 2, 2, figure=fig, width_ratios=[1.75, 1.0],
                    height_ratios=[1.0] * n + [0.18, 1.55], hspace=0.62, wspace=0.12)
    letters = "ABCDEFGH"

    for k, w in enumerate(windows):
        key = w["channel_type"]
        names = orders[key]
        idx = [labels.index(name) for name in names]
        sensors = np.flatnonzero(w["mask"].any(axis=0))
        samples = np.flatnonzero(w["mask"].any(axis=1))
        start, end = w["row"].start_ms, w["row"].end_ms
        first, second = w["contrast"].split("-")

        ax = fig.add_subplot(grid[k, 0])
        ax.axvspan(start, end, color=style.NEUTRAL_GREY, alpha=0.25, linewidth=0, zorder=0)
        style.reference_line(ax, 0)
        style.reference_line(ax, 0, axis="x")
        for condition in (first, second):
            wave = data[:, conditions.index(condition)][:, idx][:, sensors].mean(axis=1)
            mean = wave.mean(axis=0)
            sem = wave.std(axis=0, ddof=1) / np.sqrt(wave.shape[0])
            ax.fill_between(time_ms, mean - sem, mean + sem, color=colours[condition],
                            alpha=0.25, linewidth=0, zorder=2)
            ax.plot(time_ms, mean, color=colours[condition], linewidth=1.5, zorder=3)
        ax.set_xlim(time_ms[0], time_ms[-1])
        ax.set_ylabel(f"Field ({UNIT[key]})")
        if k == n - 1:
            ax.set_xlabel("Time from stimulus onset (ms)")
        sign = "" if key == "mag" else (" (+)" if w["row"].sign == "positive" else " (−)")
        ax.set_title(f"{TYPE_LABEL[key]}: {CONDITION_LABEL[first]} − "
                     f"{CONDITION_LABEL[second]}{sign}, {start:.0f}–{end:.0f} ms, "
                     f"{len(sensors)} sensors", loc="left", fontsize=13)
        ax.text(-0.075, 1.12, letters[k], transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="bottom")

        topo = fig.add_subplot(grid[k, 1])
        t_mean = w["t_map"][samples].mean(axis=0)
        if key == "mag":
            values = t_mean
            marked = np.zeros(len(mag_names), dtype=bool)
            marked[sensors] = True
            limit = float(np.abs(values).max())
            im, _ = mne.viz.plot_topomap(values, mag_pos, axes=topo, show=False, sphere=0.095,
                                         cmap=style.DIVERGING_CMAP, vlim=(-limit, limit),
                                         mask=marked, contours=0, sensors=False,
                                         mask_params=dict(marker="o", markersize=3.5,
                                                          markerfacecolor="black",
                                                          markeredgecolor="black"))
            label = "t"
        else:
            in_cluster = np.zeros(len(names), dtype=bool)
            in_cluster[sensors] = True
            values, marked = np.zeros(len(mag_names)), np.zeros(len(mag_names), dtype=bool)
            for j, mag in enumerate(mag_names):
                pair = [names.index(mag[:-1] + s) for s in ("2", "3")]
                values[j] = np.sqrt(np.mean(t_mean[pair] ** 2))
                marked[j] = in_cluster[pair].any()
            im, _ = mne.viz.plot_topomap(values, mag_pos, axes=topo, show=False, sphere=0.095,
                                         cmap=style.SEQUENTIAL_CMAP,
                                         vlim=(0, float(values.max())), mask=marked,
                                         contours=0, sensors=False,
                                         mask_params=dict(marker="o", markersize=3.5,
                                                          markerfacecolor="black",
                                                          markeredgecolor="black"))
            label = "|t|, pairs combined"
        bar = fig.colorbar(im, ax=topo, fraction=0.05, pad=0.02, shrink=0.8)
        bar.set_label(label, fontsize=11)
        bar.ax.tick_params(labelsize=10)

    legend_ax = fig.add_subplot(grid[n, :])
    legend_ax.axis("off")
    handles = [Patch(facecolor=colours[c], label=CONDITION_LABEL[c]) for c in colours]
    handles.append(Patch(facecolor=style.NEUTRAL_GREY, alpha=0.4, label="Cluster span"))
    legend_ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)

    # --- the 155 association tests --------------------------------------------------
    terms = [f"{m}|state{s}" for m, _ in MEASURES for s in range(1, 8)]
    terms += [t for t, _ in ENTROPY]
    # The ERF-HMM question is one correction family: BH across every term of every window.
    from analysis.correction import correct_pvalues

    matrix, pvalues, row_labels = [], [], []
    for position, w in enumerate(windows):
        start, end = w["row"].start_ms, w["row"].end_ms
        stem = f"erf_{w['channel_type']}_{w['contrast'].replace('Standard', 'S').replace('Deviant', 'D').replace('-', 'v')}_{start:.0f}_{end:.0f}"
        frame = pd.read_csv(args.models_dir / f"terms_{stem}.csv").set_index("term")
        matrix.append(frame.loc[terms, "t"].to_numpy())
        pvalues.append(frame.loc[terms, "perm_p"].to_numpy())
        letter = letters[position]
        row_labels.append(f"{letter}  {start:.0f}–{end:.0f} ms")
    matrix, pvalues = np.array(matrix), np.array(pvalues)
    keys = [f"{r}::{c}" for r in range(matrix.shape[0]) for c in range(matrix.shape[1])]
    pooled = correct_pvalues(dict(zip(keys, pvalues.ravel()))).set_index("term")
    significant = pooled.loc[keys, "significant"].to_numpy(bool).reshape(matrix.shape)
    print(f"pooled BH over {len(keys)} terms: {int(significant.sum())} significant, "
          f"smallest adjusted p {pooled.p_adjusted.min():.4f}")

    ax = fig.add_subplot(grid[n + 1, :])
    limit = float(np.ceil(np.abs(matrix).max()))
    im = ax.imshow(matrix, cmap=style.DIVERGING_CMAP, vmin=-limit, vmax=limit,
                   aspect="auto")
    for r, c in zip(*np.nonzero(significant)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                               edgecolor=style.TEXT_PRIMARY, linewidth=2.0))
    ax.set_yticks(range(len(row_labels)), row_labels)
    ticks = [str(s) for _ in MEASURES for s in range(1, 8)] + [label for _, label in ENTROPY]
    ax.set_xticks(range(len(terms)), ticks, fontsize=10)
    for g, (_, name) in enumerate(MEASURES):
        ax.text(g * 7 + 3, -0.9, name, ha="center", va="bottom", fontsize=12,
                fontweight="bold")
        if g:
            ax.axvline(g * 7 - 0.5, color="white", linewidth=3)
    ax.axvline(27.5, color="white", linewidth=3)
    ax.text(29, -0.9, "Entropy rate", ha="center", va="bottom", fontsize=12,
            fontweight="bold")
    ax.set_xlabel("HMM state   (entropy rate: M, amplitude; A, age; A×M, interaction)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.075, 1.16, letters[n], transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="bottom")
    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label("t", fontsize=11)

    written = style.save_figure(fig, args.output_dir / args.stem)
    for path in written:
        print("wrote", path)


if __name__ == "__main__":
    main()
