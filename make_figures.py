"""Paper figures for the oddball HMM reanalysis, in the project style.

Style is imported from the HMM_MEG package (`analysis/viz/style.py`) so that the
two papers' figures are visually identical: Wong palette, serif type, L-shaped
axes, PNG + PDF output.

State spatial maps are rebuilt here from the 39-participant fit, because none
exist for it. The fit used `onpower = 1`, `order = 0`, `zeromean = 0` and
`standardise = 1`, so a state's observation model mean is a per-parcel amplitude
envelope in units of SD. Maps show each state's mean minus the occupancy-weighted
mean across states, i.e. relative power, which is the quantity that distinguishes
states from one another rather than the overall envelope level.
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
    from nilearn import datasets, image, surface
    from nilearn.plotting import plot_surf_stat_map
    import nibabel as nib

    style.apply_style()
    maps = np.load(state_maps_npz)["mean"]                    # 7 x 38
    fo = merged.groupby("state")["FO"].mean().reindex(ALL_STATES).to_numpy()
    relative = maps - np.average(maps, axis=0, weights=fo)     # occupancy-weighted contrast

    labels_img = nib.load(parcellation)
    labels = labels_img.get_fdata().astype(int)
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")

    # Project each state's parcel values into the volume, then onto the surface.
    surfaces, peaks = [], []
    for k in range(7):
        vol = np.zeros(labels.shape, dtype=float)
        for parcel in range(1, 39):
            vol[labels == parcel] = relative[k, parcel - 1]
        img = nib.Nifti1Image(vol, labels_img.affine)
        img = image.smooth_img(img, 6)
        surfaces.append({
            hemi: surface.vol_to_surf(img, getattr(fsaverage, f"pial_{hemi}"),
                                      interpolation="linear", radius=6.0)
            for hemi in ("left", "right")})
        top = int(np.argmax(np.abs(relative[k]))) + 1
        ijk = np.array(np.where(labels == top)).mean(axis=1)
        peaks.append(nib.affines.apply_affine(labels_img.affine, ijk))

    vmax = float(np.nanmax(np.abs(np.concatenate(
        [s[h] for s in surfaces for h in ("left", "right")]))))

    views = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]
    fig, axes = plt.subplots(7, 4, figsize=(13.0, 20.0),
                             subplot_kw={"projection": "3d"})
    for k in range(7):
        for c, (hemi, view) in enumerate(views):
            plot_surf_stat_map(
                getattr(fsaverage, f"infl_{hemi}"), surfaces[k][hemi],
                hemi=hemi, view=view, bg_map=getattr(fsaverage, f"sulc_{hemi}"),
                cmap=style.DIVERGING_CMAP, vmax=vmax, threshold=0.02 * vmax,
                colorbar=False, bg_on_data=True, darkness=None,
                axes=axes[k, c], figure=fig)
            axes[k, c].set_title("")
            if k == 0:
                axes[k, c].set_title(f"{hemi.capitalize()} {view}", fontsize=12,
                                     fontweight="bold", pad=0)
        x, y, z = peaks[k]
        axes[k, 0].text2D(-0.08, 0.5, f"State {k + 1}", transform=axes[k, 0].transAxes,
                          rotation=90, ha="center", va="center",
                          fontsize=15, fontweight="bold",
                          color=style.state_palette(7)[k])
        axes[k, 3].text2D(1.02, 0.5,
                          f"peak parcel\n({x:.0f}, {y:.0f}, {z:.0f})\nFO {fo[k]:.3f}",
                          transform=axes[k, 3].transAxes, ha="left", va="center",
                          fontsize=10, color=style.TEXT_MUTED)

    cax = fig.add_axes([0.35, 0.075, 0.3, 0.009])
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=style.DIVERGING_CMAP),
                        cax=cax, orientation="horizontal")
    cbar.set_label("Relative amplitude envelope (SD)", fontsize=13, fontweight="bold")
    cbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.97, bottom=0.10,
                        wspace=-0.10, hspace=-0.25)
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
    threshold = np.percentile(off[off > 0], 50)
    for i in range(7):
        for j in range(7):
            if i == j or off[i, j] < threshold:
                continue
            start, end = pos[i], pos[j]
            d = end - start
            n = d / np.linalg.norm(d)
            start = start + n * 0.13
            end = end - n * 0.13
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>", color=style.NEUTRAL_GREY,
                                        linewidth=0.4 + 7.0 * off[i, j],
                                        alpha=0.75, shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=0.16"))
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

    labels = [f"{MEASURE_LABELS[r.measure]}" + (f"  ·  state {r.state[-1]}"
              if isinstance(r.state, str) and r.state.startswith("state")
              else f"  ·  {r.state}") for r in t.itertuples()]
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
        loc="lower right", fontsize=10)
    fig.tight_layout()
    style.save_figure(fig, output_dir / "figure_6_term_overview")
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
    state_scatter_grid(merged, fo_states, "FO", "Fractional occupancy",
                       args.predictor, args.predictor_label,
                       "figure_5_occupancy_externalising", args.output_dir, style)
    print("figure 5 done", flush=True)
    figure6_terms(terms, args.output_dir, style, args.predictor_label)
    print("figure 6 done", flush=True)

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
