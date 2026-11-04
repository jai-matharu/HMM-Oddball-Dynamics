"""Paper figures for the oddball HMM reanalysis, in the project style.

Style is imported from the HMM_MEG package (`analysis/viz/style.py`) so that the
two papers' figures are visually identical: Wong palette, serif type, L-shaped
axes, PNG + PDF output.

State spatial maps are rebuilt here from the 39-participant fit, because none
exist for it. The fit used `onpower = 1`, `order = 0`, `zeromean = 0` and
`standardise = 1`, so a state's observation model mean is a per-parcel amplitude
envelope in units of SD. Maps show each state's mean minus the occupancy-weighted
mean across states and then minus its own spatial mean, i.e. where that state's
envelope runs high or low across the cortex. The offset removed by the second
step is printed beside each state, since for states 1, 3 and 6 it is the larger
part of the contrast: their envelopes are shifted in one direction at every
parcel, and without the demeaning their maps are a single flat colour.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS = 250.0
ALL_STATES = list(range(1, 8))

BEHAVIOURAL = [
    ("age", "Age"),
    ("matrix_T", "Matrix reasoning T"),
    ("SDQ_externalising", "SDQ externalising"),
    ("SDQ_internalising", "SDQ internalising"),
    ("SDQ_prosocial", "SDQ prosocial"),
    ("equivalised_income", "Equivalised income"),
    ("subjective_SES", "Subjective SES"),
    ("entropy_rate", "Entropy rate"),
    ("switching_rate", "Switching rate"),
    ("max_FO", "Max FO"),
]

MEASURE_LABELS = {
    "FO": "Fractional occupancy",
    "transition_into_sum": "Transitions into state",
    "lifetime_mean": "Lifetime",
    "interval_mean": "Interval",
    "entropy_rate": "Entropy rate",
}


# --- data -------------------------------------------------------------------

def load(analysis_dir: Path, models_dir: Path, predictor: str):
    subject = pd.read_csv(analysis_dir / "subject_level.csv", dtype={"subject_id": str})
    state = pd.read_csv(analysis_dir / "state_level.csv", dtype={"subject_id": str})
    trans = pd.read_csv(analysis_dir / "transition_matrices.csv", dtype={"subject_id": str})

    col_sums = (trans.groupby(["subject_id", "to_state"], as_index=False)["probability"].sum()
                     .rename(columns={"to_state": "state", "probability": "transition_into_sum"}))
    merged = (state.merge(col_sums, on=["subject_id", "state"], validate="one_to_one")
                   .merge(subject, on="subject_id", validate="many_to_one"))
    merged["lifetime_ms"] = merged["lifetime_mean"] / FS * 1000
    merged["interval_ms"] = merged["interval_mean"] / FS * 1000

    terms = pd.read_csv(models_dir / f"terms_{predictor}.csv")
    return subject, merged, trans, terms


def mean_transition_matrix(trans: pd.DataFrame) -> np.ndarray:
    m = (trans.groupby(["from_state", "to_state"])["probability"].mean()
              .unstack().reindex(index=ALL_STATES, columns=ALL_STATES))
    return m.to_numpy()


# --- shared drawing ---------------------------------------------------------

def scatter(ax, x, y, xlabel, ylabel, colour, style):
    """Scatter with a least-squares fit and its 95% confidence band."""
    ax.scatter(x, y, s=34, color=colour, edgecolor=style.TEXT_PRIMARY, linewidth=0.4,
               alpha=0.85, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, slope * grid + intercept, color=style.TEXT_PRIMARY, linewidth=1.5, zorder=4)
    n = len(x)
    resid = y - (slope * x + intercept)
    se = np.sqrt(np.sum(resid ** 2) / (n - 2))
    band = se * np.sqrt(1 / n + (grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(grid, slope * grid + intercept - 1.96 * band,
                    slope * grid + intercept + 1.96 * band,
                    color=style.NEUTRAL_GREY, alpha=0.25, linewidth=0, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def state_scatter_grid(df, states, value_column, ylabel, predictor, xlabel,
                       stem, output_dir, style, ncols=3):
    style.apply_style()
    palette = style.state_palette(7)
    nrows = int(np.ceil(len(states) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), squeeze=False)
    for i, s in enumerate(states):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["state"] == s]
        scatter(ax, sub[predictor].to_numpy(), sub[value_column].to_numpy(),
                xlabel, ylabel, palette[s - 1], style)
        ax.set_title(f"State {s}", fontsize=14)
    for i in range(len(states), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.tight_layout()
    style.save_figure(fig, output_dir / stem)
    plt.close(fig)


# --- figures ----------------------------------------------------------------

def figure1_correlations(subject, output_dir, style):
    style.apply_style()
    cols = [c for c, _ in BEHAVIOURAL if c in subject.columns]
    labels = [l for c, l in BEHAVIOURAL if c in subject.columns]
    corr = subject[cols].corr()
    n = len(cols)

    fig, ax = plt.subplots(figsize=style.FIGURE_SIZES["full_page"])
    im = ax.imshow(corr.to_numpy(), cmap=style.DIVERGING_CMAP, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            r = corr.iloc[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(r) > 0.6 else style.TEXT_PRIMARY)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", fontsize=14, fontweight="bold")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    style.save_figure(fig, output_dir / "figure_1_correlations")
    plt.close(fig)


def figure2_state_maps(merged, state_maps_npz, parcellation, output_dir, style):
    """Relative amplitude-envelope maps for the seven states, on an inflated surface."""
    from matplotlib.colors import LinearSegmentedColormap
    from nilearn import datasets, image, surface
    from nilearn.plotting import plot_surf_stat_map
    import nibabel as nib

    # FSL's blue-lightblue / red-yellow pair, the convention these maps are read
    # in. It saturates at both ends and runs dark next to the threshold, so a
    # supra-threshold region reads as a region instead of fading into the grey.
    cold_hot = LinearSegmentedColormap.from_list("cold_hot", [
        (0.00, "#33ffff"), (0.22, "#0099ff"), (0.47, "#000c7a"),
        (0.53, "#7a0000"), (0.78, "#ff6600"), (1.00, "#ffff33")])
    threshold_fraction = 0.35      # of each state's own scale
    scale_percentile = 98.0        # saturate the top 2% of vertices

    style.apply_style()
    maps = np.load(state_maps_npz)["mean"]                    # 7 x 38
    fo = merged.groupby("state")["FO"].mean().reindex(ALL_STATES).to_numpy()
    relative = maps - np.average(maps, axis=0, weights=fo)     # occupancy-weighted contrast
    offsets = relative.mean(axis=1)                            # each state's global shift
    pattern = relative - offsets[:, None]

    labels_img = nib.load(parcellation)
    labels = labels_img.get_fdata().astype(int)
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")     # 164k vertices per hemisphere

    # Project each state's parcel values into the volume, then onto the surface.
    # Sampling between the white and pial meshes follows the cortical ribbon, so
    # the 8 mm parcel grid reaches the surface without gaps.
    surfaces, peaks = [], []
    for k in range(7):
        vol = np.zeros(labels.shape, dtype=float)
        for parcel in range(1, 39):
            vol[labels == parcel] = pattern[k, parcel - 1]
        img = image.smooth_img(nib.Nifti1Image(vol, labels_img.affine), 4)
        surfaces.append({
            hemi: surface.vol_to_surf(img, getattr(fsaverage, f"pial_{hemi}"),
                                      inner_mesh=getattr(fsaverage, f"white_{hemi}"),
                                      interpolation="linear")
            for hemi in ("left", "right")})
        top = int(np.argmax(np.abs(pattern[k]))) + 1
        ijk = np.array(np.where(labels == top)).mean(axis=1)
        peaks.append(nib.affines.apply_affine(labels_img.affine, ijk))

    # Binarised curvature, as FreeSurfer and Workbench draw it: two flat greys
    # under the data rather than continuous shading through it, so the colour is
    # not dimmed by whatever happens to lie beneath it.
    background = {}
    for hemi in ("left", "right"):
        curv = surface.load_surf_data(getattr(fsaverage, f"curv_{hemi}"))
        background[hemi] = np.where(curv > 0, 0.35, 0.65)

    # Every state carries its own scale and its own colourbar. A shared scale
    # would put states 3 and 6 (spatial SD 0.045 and 0.036) near the middle of a
    # range set by state 1 (0.230) and leave them almost uncoloured.
    views = [("left", "lateral"), ("left", "medial"),
             ("right", "medial"), ("right", "lateral")]
    palette = style.state_palette(7)
    fig, axes = plt.subplots(7, 4, figsize=(13.0, 17.0),
                             subplot_kw={"projection": "3d"})
    scales = []
    for k in range(7):
        values = np.concatenate([surfaces[k][h] for h in ("left", "right")])
        scale = float(np.percentile(np.abs(values), scale_percentile))
        scales.append(scale)
        for c, (hemi, view) in enumerate(views):
            plot_surf_stat_map(
                getattr(fsaverage, f"infl_{hemi}"), surfaces[k][hemi],
                hemi=hemi, view=view, bg_map=background[hemi], cmap=cold_hot,
                vmin=-scale, vmax=scale, threshold=threshold_fraction * scale,
                colorbar=False, bg_on_data=False, axes=axes[k, c], figure=fig)
            axes[k, c].set_title("")
            if k == 0:
                axes[k, c].set_title(f"{hemi.capitalize()} {view}", fontsize=12,
                                     fontweight="bold", pad=0)
        x, y, z = peaks[k]
        axes[k, 0].text2D(-0.17, 0.5, f"State {k + 1}", transform=axes[k, 0].transAxes,
                          rotation=90, ha="center", va="center",
                          fontsize=15, fontweight="bold", color=palette[k])
        axes[k, 0].text2D(-0.04, 0.5,
                          f"FO {fo[k]:.3f}\noffset {offsets[k]:+.2f} SD\n"
                          f"peak ({x:.0f}, {y:.0f}, {z:.0f})",
                          transform=axes[k, 0].transAxes, rotation=90,
                          ha="center", va="center", fontsize=9,
                          color=style.TEXT_MUTED)

    fig.subplots_adjust(left=0.11, right=0.88, top=0.96, bottom=0.015,
                        wspace=-0.10, hspace=-0.30)
    for k in range(7):
        box = axes[k, 3].get_position()
        cax = fig.add_axes([0.90, box.y0 + 0.30 * box.height, 0.010, 0.40 * box.height])
        norm = plt.Normalize(vmin=-scales[k], vmax=scales[k])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cold_hot), cax=cax)
        cbar.set_ticks([-scales[k], 0.0, scales[k]])
        cbar.set_ticklabels([f"{-scales[k]:.2f}", "0", f"{scales[k]:+.2f}"])
        cbar.ax.tick_params(labelsize=8, length=2)
        cbar.outline.set_visible(False)
    fig.text(0.965, 0.5, "Relative amplitude envelope (SD)", rotation=90,
             ha="center", va="center", fontsize=11, fontweight="bold")
    # A 164k-vertex mesh is ~330k triangles per panel, which as vector paths
    # would make the PDF several hundred megabytes. The meshes are rasterised at
    # 300 dpi; the labels, ticks and colourbars stay vector.
    for ax in axes.ravel():
        for collection in ax.collections:
            collection.set_rasterized(True)
    fig.set_dpi(300)
    style.save_figure(fig, output_dir / "figure_2_state_maps")
    plt.close(fig)


def figure3_temporal(merged, output_dir, style):
    style.apply_style()
    palette = style.state_palette(7)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    panels = [("FO", "Fractional occupancy", "A", False),
              ("lifetime_ms", "Lifetime (ms)", "B", False),
              ("interval_ms", "Interval (ms)", "C", True)]
    for ax, (column, label, letter, log) in zip(axes, panels):
        data = [merged.loc[merged["state"] == s, column].to_numpy() for s in ALL_STATES]
        parts = ax.violinplot(data, positions=ALL_STATES, widths=0.75,
                              showextrema=False, showmedians=False)
        for body, colour in zip(parts["bodies"], palette):
            body.set_facecolor(colour)
            body.set_edgecolor(style.TEXT_PRIMARY)
            body.set_linewidth(0.6)
            body.set_alpha(0.55)
        for s, colour in zip(ALL_STATES, palette):
            values = merged.loc[merged["state"] == s, column].to_numpy()
            jitter = np.random.default_rng(s).normal(0, 0.055, len(values))
            ax.scatter(s + jitter, values, s=12, color=colour,
                       edgecolor=style.TEXT_PRIMARY, linewidth=0.3, alpha=0.85, zorder=3)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(ALL_STATES, [str(s) for s in ALL_STATES])
        ax.set_xlabel("State")
        ax.set_ylabel(label)
        ax.set_title(letter, loc="left", fontsize=16, fontweight="bold")
    fig.tight_layout()
    style.save_figure(fig, output_dir / "figure_3_temporal_descriptives")
    plt.close(fig)


def figure4_transitions(merged, trans, output_dir, style):
    style.apply_style()
    palette = style.state_palette(7)
    matrix = mean_transition_matrix(trans)
    occupancy = merged.groupby("state")["FO"].mean().reindex(ALL_STATES).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=style.FIGURE_SIZES["double_column"])

    # A: directed graph, states on a circle, edge width by off-diagonal probability
    ax = axes[0]
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 8)[:7]
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    off = matrix.copy()
    np.fill_diagonal(off, 0.0)
    # Only the strongest 40% of edges are drawn, and width and opacity are scaled
    # across the drawn range rather than across zero: the off-diagonals span a
    # narrow band, so an absolute scale renders every edge at the same weight.
    shown = off[off > 0]
    threshold = np.percentile(shown, 60)
    lo, hi = threshold, off.max()
    for i in range(7):
        for j in range(7):
            if i == j or off[i, j] < threshold:
                continue
            weight = (off[i, j] - lo) / (hi - lo) if hi > lo else 1.0
            start, end = pos[i], pos[j]
            d = end - start
            n = d / np.linalg.norm(d)
            start = start + n * 0.15
            end = end - n * 0.15
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45",
                                        color=style.TEXT_SECONDARY,
                                        linewidth=0.6 + 3.4 * weight,
                                        alpha=0.30 + 0.60 * weight,
                                        shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=0.16"))
    ax.text(0.5, -0.02, f"strongest 40% of between-state transitions "
                        f"({lo:.3f}-{hi:.3f})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=style.TEXT_MUTED)
    for i in range(7):
        ax.scatter(*pos[i], s=300 + 5200 * occupancy[i], color=palette[i],
                   edgecolor=style.TEXT_PRIMARY, linewidth=0.8, zorder=5)
        ax.text(*pos[i], str(i + 1), ha="center", va="center", fontsize=13,
                fontweight="bold", zorder=6,
                color="white" if i not in (4, 6) else style.TEXT_PRIMARY)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("A", loc="left", fontsize=16, fontweight="bold")

    # B: mean transition matrix, self-transitions masked
    ax = axes[1]
    masked = np.ma.masked_array(matrix, mask=np.eye(7, dtype=bool))
    im = ax.imshow(masked, cmap=style.SEQUENTIAL_CMAP, vmin=0, vmax=masked.max())
    for i in range(7):
        for j in range(7):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > 0.6 * masked.max() else style.TEXT_PRIMARY)
    ax.set_xticks(range(7), [str(s) for s in ALL_STATES])
    ax.set_yticks(range(7), [str(s) for s in ALL_STATES])
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability", fontsize=13, fontweight="bold")
    cbar.outline.set_visible(False)
    ax.set_title("B", loc="left", fontsize=16, fontweight="bold")

    fig.tight_layout()
    style.save_figure(fig, output_dir / "figure_4_transition_structure")
    plt.close(fig)


def figure6_terms(terms, output_dir, style, predictor_label):
    """Every term of the 31-term family, with both corrections marked."""
    style.apply_style()
    palette = style.state_palette(7)
    order = ["FO", "transition_into_sum", "lifetime_mean", "interval_mean", "entropy_rate"]
    t = terms.copy()
    t["order"] = t["measure"].map({m: i for i, m in enumerate(order)})
    t = t.sort_values(["order", "state"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 11.0))
    ypos = np.arange(len(t))[::-1]
    for y, row in zip(ypos, t.itertuples()):
        state_no = None
        if isinstance(row.state, str) and row.state.startswith("state"):
            state_no = int(row.state.replace("state", ""))
        colour = palette[state_no - 1] if state_no else style.NEUTRAL_GREY
        ax.plot([0, row.t], [y, y], color=colour, linewidth=1.8, alpha=0.8, zorder=2)
        ax.scatter(row.t, y, s=70 if row.bh_significant else 40, color=colour,
                   edgecolor=style.TEXT_PRIMARY,
                   linewidth=1.1 if row.bh_significant else 0.4,
                   zorder=4)
        wy = row.wy_step_down
        if pd.notna(wy) and wy < 0.05:
            ax.scatter(row.t, y, s=175, facecolor="none", edgecolor=style.TEXT_PRIMARY,
                       linewidth=0.9, zorder=3)

    scalar_labels = {"measure": predictor_label, "age": "age",
                     "age_x_measure": f"age \u00d7 {predictor_label}"}
    labels = [f"{MEASURE_LABELS[r.measure]}" + (f"  \u00b7  state {r.state[-1]}"
              if isinstance(r.state, str) and r.state.startswith("state")
              else f"  \u00b7  {scalar_labels.get(r.state, r.state)}")
              for r in t.itertuples()]
    ax.set_yticks(ypos, labels, fontsize=10)
    style.reference_line(ax, 0, axis="x")
    ax.set_xlabel(f"t  ({predictor_label})")
    ax.set_ylim(-1, len(t))

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=9,
               markerfacecolor=style.NEUTRAL_GREY, markeredgecolor=style.TEXT_PRIMARY,
               markeredgewidth=1.1, label="BH-FDR significant"),
        Line2D([], [], marker="o", linestyle="none", markersize=13,
               markerfacecolor="none", markeredgecolor=style.TEXT_PRIMARY,
               label="Westfall-Young step-down significant")],
        loc="upper center", bbox_to_anchor=(0.5, -0.055), ncol=2, fontsize=10)
    fig.tight_layout()
    style.save_figure(fig, output_dir / "figure_8_term_overview")
    plt.close(fig)


def figure5_externalising_effects(merged, terms, output_dir, style, predictor, predictor_label):
    """The headline finding: paired occupancy and entry effects, and their sign structure."""
    style.apply_style()
    palette = style.state_palette(7)
    fig = plt.figure(figsize=(13.0, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.38, wspace=0.28)

    # A: t per state for the two compositional measures, paired within state.
    ax = fig.add_subplot(gs[0, :])
    look = terms.set_index("term")
    width = 0.19
    for s_no in ALL_STATES:
        for offset, metric, marker in ((-width, "FO", "o"),
                                       (+width, "transition_into_sum", "s")):
            row = look.loc[f"{metric}|state{s_no}"]
            ax.plot([s_no + offset, s_no + offset], [0, row.t],
                    color=palette[s_no - 1], linewidth=2.0, alpha=0.8, zorder=2)
            ax.scatter(s_no + offset, row.t, marker=marker,
                       s=95 if row.bh_significant else 48,
                       color=palette[s_no - 1], edgecolor=style.TEXT_PRIMARY,
                       linewidth=1.2 if row.bh_significant else 0.4, zorder=4)
    style.reference_line(ax, 0, axis="y")
    ax.set_xticks(ALL_STATES, [f"State {s}" for s in ALL_STATES])
    ax.set_ylabel(f"t  ({predictor_label})")
    ax.set_title("A", loc="left", fontsize=16, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=9,
               markerfacecolor=style.NEUTRAL_GREY, markeredgecolor=style.TEXT_PRIMARY,
               label="Fractional occupancy"),
        Line2D([], [], marker="s", linestyle="none", markersize=9,
               markerfacecolor=style.NEUTRAL_GREY, markeredgecolor=style.TEXT_PRIMARY,
               label="Transitions into state")],
        loc="upper left", ncol=2, fontsize=11)

    # B: the three strongest states, occupancy against the measure.
    strongest = (look[(look.measure == "FO") & look.bh_significant]
                 .reindex(look[(look.measure == "FO") & look.bh_significant]
                          .t.abs().sort_values(ascending=False).index)
                 .head(3))
    for i, term in enumerate(strongest.index):
        s_no = int(term[-1])
        axb = fig.add_subplot(gs[1, i])
        sub = merged[merged["state"] == s_no]
        scatter(axb, sub[predictor].to_numpy(), sub["FO"].to_numpy(),
                predictor_label, "Fractional occupancy", palette[s_no - 1], style)
        row = look.loc[term]
        axb.set_title(f"State {s_no}   t = {row.t:+.2f}, $p_{{BH}}$ = {row.bh_p:.4f}",
                      fontsize=13)
        if i == 0:
            axb.text(-0.22, 1.06, "B", transform=axb.transAxes,
                     fontsize=16, fontweight="bold")
    style.save_figure(fig, output_dir / "figure_5_externalising_effects")
    plt.close(fig)


def figure6_subject_level(subject, scalars, entropy_adjusted, output_dir, style,
                          predictor, predictor_label):
    """Subject-level summaries against the measure."""
    style.apply_style()
    look = scalars[scalars.term == "measure"].set_index("outcome")
    panels = [("entropy_rate", "Entropy rate (nats/s)", "A"),
              ("switching_rate", "Switching rate", "B"),
              ("max_FO", "Maximum fractional occupancy", "C")]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for ax, (col, label, letter) in zip(axes, panels):
        scatter(ax, subject[predictor].to_numpy(), subject[col].to_numpy(),
                predictor_label, label, style.NEUTRAL_LIGHT_BLUE, style)
        row = look.loc[col]
        ax.set_title(f"{letter}    t = {row.t:+.2f}, p = {row.perm_p:.3f}",
                     loc="left", fontsize=14, fontweight="bold")
        # Only entropy rate belongs to the corrected family; quoting an
        # uncorrected p alone would overstate the other two.
        note = (f"adjusted p = {entropy_adjusted:.3f}" if col == "entropy_rate"
                else "outside the correction family")
        ax.text(0.0, -0.30, note, transform=ax.transAxes, fontsize=11,
                color=style.TEXT_MUTED)
    fig.subplots_adjust(bottom=0.26, wspace=0.30)
    style.save_figure(fig, output_dir / "figure_6_subject_level_summaries")
    plt.close(fig)


def figure7_specificity(models_dir, output_dir, style, questions):
    """The externalising-significant terms, shown for all three questions."""
    style.apply_style()
    palette = style.state_palette(7)
    tables = {q: pd.read_csv(models_dir / f"terms_{q}.csv").set_index("term")
              for q, _ in questions}
    primary = tables[questions[0][0]]
    sig = primary[primary.bh_significant]
    sig = sig.reindex(sig.t.abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ypos = np.arange(len(sig))
    markers = ["o", "^", "D"]
    for j, (q, label) in enumerate(questions):
        t = tables[q].reindex(sig.index)
        colours = [palette[int(st[-1]) - 1] if isinstance(st, str) and st.startswith("state")
                   else style.NEUTRAL_GREY for st in t.state]
        edge = [style.TEXT_PRIMARY if s else style.NEUTRAL_GREY for s in t.bh_significant]
        ax.scatter(t.t, ypos + (j - 1) * 0.24, marker=markers[j],
                   s=[85 if s else 42 for s in t.bh_significant],
                   color=colours if j == 0 else "white",
                   edgecolor=colours if j else edge,
                   linewidth=1.4 if j else 1.0, zorder=4)
    style.reference_line(ax, 0, axis="x")
    labels = [f"{MEASURE_LABELS[r.measure]}  \u00b7  state {r.state[-1]}"
              if isinstance(r.state, str) and r.state.startswith("state")
              else f"{MEASURE_LABELS[r.measure]}" for r in sig.itertuples()]
    ax.set_yticks(ypos, labels, fontsize=10)
    ax.set_xlabel("t")
    ax.set_ylim(-0.8, len(sig) - 0.2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, linestyle="none", markersize=9,
                      markerfacecolor=style.NEUTRAL_GREY if j == 0 else "white",
                      markeredgecolor=style.TEXT_PRIMARY, label=lab)
               for j, (m, (_, lab)) in enumerate(zip(markers, questions))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=11)
    fig.tight_layout()
    style.save_figure(fig, output_dir / "figure_7_specificity")
    plt.close(fig)


def figure_s5_correction(models_dir, output_dir, style):
    """Adjusted p under each correction scheme, for the primary question."""
    style.apply_style()
    fs = pd.read_csv(models_dir / "family_sensitivity.csv")
    e = fs[fs.question == "SDQ_externalising"].copy()
    e = e.reindex(e.perm_p.sort_values(ascending=False).index)
    schemes = [("bh_question", "BH, per question (31)", "o"),
               ("bh_measure", "BH, per measure", "s"),
               ("bh_pooled", "BH, all pooled (93)", "^"),
               ("wy_step_down", "Westfall-Young step-down", "D")]
    fig, ax = plt.subplots(figsize=(9.5, 11.0))
    ypos = np.arange(len(e))
    colours = [style.COLORS["blue"], style.COLORS["orange"],
               style.COLORS["bluish_green"], style.COLORS["reddish_purple"]]
    for (col, label, marker), colour in zip(schemes, colours):
        ax.scatter(e[col].clip(lower=1e-4), ypos, marker=marker, s=46, color=colour,
                   edgecolor=style.TEXT_PRIMARY, linewidth=0.3, label=label, zorder=4)
    ax.axvline(0.05, color=style.TEXT_PRIMARY, linestyle="--", linewidth=0.9, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted $p$")
    labels = [f"{MEASURE_LABELS[r.measure]}  \u00b7  state {r.state[-1]}"
              if isinstance(r.state, str) and r.state.startswith("state")
              else f"{MEASURE_LABELS[r.measure]}  \u00b7  {str(r.state).replace('_', ' ')}"
              for r in e.itertuples()]
    ax.set_yticks(ypos, labels, fontsize=9)
    ax.set_ylim(-1, len(e))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)
    fig.tight_layout()
    style.save_figure(fig, output_dir / "figure_S5_correction_sensitivity")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--models-dir", required=True, type=Path)
    ap.add_argument("--hmm-meg", required=True, type=Path)
    ap.add_argument("--state-maps", required=True, type=Path)
    ap.add_argument("--parcellation", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--predictor", default="SDQ_externalising")
    ap.add_argument("--predictor-label", default="SDQ externalising")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(args.hmm_meg))
    from analysis.viz import style

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject, merged, trans, terms = load(args.analysis_dir, args.models_dir, args.predictor)

    figure1_correlations(subject, args.output_dir, style)
    print("figure 1 done", flush=True)
    figure2_state_maps(merged, args.state_maps, args.parcellation, args.output_dir, style)
    print("figure 2 done", flush=True)
    figure3_temporal(merged, args.output_dir, style)
    print("figure 3 done", flush=True)
    figure4_transitions(merged, trans, args.output_dir, style)
    print("figure 4 done", flush=True)

    sig = terms[terms.bh_significant & terms.state.str.startswith("state", na=False)]
    fo_states = sorted(int(s[-1]) for s in sig.loc[sig.measure == "FO", "state"])
    tr_states = sorted(int(s[-1]) for s in
                       sig.loc[sig.measure == "transition_into_sum", "state"])
    figure5_externalising_effects(merged, terms, args.output_dir, style,
                                  args.predictor, args.predictor_label)
    print("figure 5 done", flush=True)

    scalars_path = args.models_dir / f"subject_scalars_{args.predictor}.csv"
    if scalars_path.exists():
        entropy_adjusted = float(
            terms.loc[terms.term == "entropy_rate|measure", "bh_p"].iloc[0])
        figure6_subject_level(subject, pd.read_csv(scalars_path), entropy_adjusted,
                              args.output_dir, style, args.predictor, args.predictor_label)
        print("figure 6 done", flush=True)
    else:
        print(f"figure 6 skipped: {scalars_path} not found", flush=True)

    figure7_specificity(args.models_dir, args.output_dir, style,
                        [("SDQ_externalising", "Externalising"),
                         ("SDQ_internalising", "Internalising"),
                         ("matrix_T", "Matrix reasoning")])
    print("figure 7 done", flush=True)
    figure6_terms(terms, args.output_dir, style, args.predictor_label)
    figure_s5_correction(args.models_dir, args.output_dir, style)
    print("figure 8 and S5 done", flush=True)

    state_scatter_grid(merged, fo_states, "FO", "Fractional occupancy",
                       args.predictor, args.predictor_label,
                       "figure_S6_occupancy_significant_states", args.output_dir, style)

    state_scatter_grid(merged, ALL_STATES, "FO", "Fractional occupancy",
                       args.predictor, args.predictor_label,
                       "figure_S1_occupancy_all_states", args.output_dir, style)
    state_scatter_grid(merged, ALL_STATES, "transition_into_sum",
                       "Probability of entering state", args.predictor, args.predictor_label,
                       "figure_S2_transitions_all_states", args.output_dir, style)
    state_scatter_grid(merged, ALL_STATES, "lifetime_ms", "Lifetime (ms)",
                       args.predictor, args.predictor_label,
                       "figure_S3_lifetime_all_states", args.output_dir, style)
    state_scatter_grid(merged, ALL_STATES, "interval_ms", "Interval (ms)",
                       args.predictor, args.predictor_label,
                       "figure_S4_interval_all_states", args.output_dir, style)
    print("supplementary figures done", flush=True)
    print(f"transitions-into states BH-significant: {tr_states}")


if __name__ == "__main__":
    main()
