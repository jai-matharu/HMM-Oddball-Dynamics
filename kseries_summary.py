"""Model-order supplement: free-energy table and state maps for K = 4..14.

Reads oddball_HMM_exports/kseries/K??_run*/summary.mat (one per fit) and
K07_K7_2/summary.mat (the analysed fit), writes CSV tables beside them and one
figure per K into oddball_figures_t0/kseries/.

States of each solution are matched one-to-one to the K7_2 states by the
Hungarian algorithm on the across-parcel correlation of relative envelopes
(state mean minus that solution's occupancy-weighted mean). Matched states are
shown first, in K7_2 order; unmatched states follow in order of occupancy.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

ROOT = Path("/imaging/astle/natalia/oddball_HMM_exports/kseries")
FIGDIR = Path("/imaging/astle/natalia/oddball_figures_t0/kseries")
NPZ = Path("/imaging/astle/natalia/oddball_HMM_exports/state_maps_K7_2.npz")
PARCELLATION = Path("/imaging/astle/HMM_Dissertation/parcel38_volume.nii.gz")
MATCH_R = 0.7   # display only: below this a Hungarian pairing is labelled "closest", in grey
sys.path.insert(0, "/home/nz01/Github/HMM_MEG")
from analysis.viz import style  # noqa: E402


def load(path):
    m = sio.loadmat(path, squeeze_me=True)
    return {k: v for k, v in m.items() if not k.startswith("__")}


def relative(means, fo_group):
    return means - np.average(means, axis=0, weights=fo_group)


def corr_rows(a, b):
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return a @ b.T / a.shape[1]


# ---------------------------------------------------------------- reference
ref = load(ROOT / "K07_K7_2" / "summary.mat")
ref_fo = np.atleast_2d(ref["fo"]).mean(0)
ref_rel = relative(ref["means"], ref_fo)
npz_mean = np.load(NPZ)["mean"]
print(f"K7_2 means vs state_maps_K7_2.npz: max abs diff {np.abs(ref['means'] - npz_mean).max():.2e}")
print(f"K7_2 group FO from Gamma: {np.round(ref_fo, 4)}")
print(f"K7_2 recomputed free energy: {float(ref['fe_recomputed']):.6g}")

# ---------------------------------------------------------------- figures
def draw(K, run, s, fo, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from nilearn import datasets, image, surface
    from nilearn.plotting import plot_surf_stat_map
    import nibabel as nib

    cold_hot = LinearSegmentedColormap.from_list("cold_hot", [
        (0.00, "#33ffff"), (0.22, "#0099ff"), (0.47, "#000c7a"),
        (0.53, "#7a0000"), (0.78, "#ff6600"), (1.00, "#ffff33")])
    threshold_fraction, scale_percentile = 0.35, 98.0     # as Figure 1

    rel = relative(s["means"], fo)
    offsets = rel.mean(1)
    pattern = rel - offsets[:, None]
    C = corr_rows(ref_rel, rel)
    ri, ci = linear_sum_assignment(-C)
    matched = {c: (r, C[r, c]) for r, c in zip(ri, ci)}
    order = [c for r, c in sorted(zip(ri, ci))]
    order += [c for c in np.argsort(-fo) if c not in matched]

    labels_img = nib.load(PARCELLATION)
    labels = labels_img.get_fdata().astype(int)
    fsavg = datasets.fetch_surf_fsaverage("fsaverage")
    background = {h: np.where(surface.load_surf_data(getattr(fsavg, f"curv_{h}")) > 0, 0.35, 0.65)
                  for h in ("left", "right")}
    views = [("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral")]
    palette = style.state_palette(7)

    style.apply_style()
    fig, axes = plt.subplots(K, 4, figsize=(13.0, 2.43 * K), subplot_kw={"projection": "3d"})
    scales = []
    for row, k in enumerate(order):
        vol = np.zeros(labels.shape)
        for p in range(1, 39):
            vol[labels == p] = pattern[k, p - 1]
        img = image.smooth_img(nib.Nifti1Image(vol, labels_img.affine), 4)
        surf = {h: surface.vol_to_surf(img, getattr(fsavg, f"pial_{h}"),
                                       inner_mesh=getattr(fsavg, f"white_{h}"),
                                       interpolation="linear") for h in ("left", "right")}
        scale = float(np.percentile(np.abs(np.concatenate([surf["left"], surf["right"]])),
                                    scale_percentile))
        scales.append(scale)
        for c, (h, v) in enumerate(views):
            plot_surf_stat_map(getattr(fsavg, f"infl_{h}"), surf[h], hemi=h, view=v,
                               bg_map=background[h], cmap=cold_hot, vmin=-scale, vmax=scale,
                               threshold=threshold_fraction * scale, colorbar=False,
                               bg_on_data=False, axes=axes[row, c], figure=fig)
            axes[row, c].set_title("")
            if row == 0:
                axes[row, c].set_title(f"{h.capitalize()} {v}", fontsize=12,
                                       fontweight="bold", pad=0)
        if k in matched and matched[k][1] >= MATCH_R:
            r_state, r = matched[k]
            colour, note = palette[r_state], f"matches state {r_state + 1}, r = {r:.2f}"
        elif k in matched:
            # one-to-one assignment is forced; below MATCH_R it is not a match
            r_state, r = matched[k]
            colour, note = style.TEXT_MUTED, f"closest: state {r_state + 1}, r = {r:.2f}"
        else:
            colour, note = style.TEXT_MUTED, "no one-to-one match"
        axes[row, 0].text2D(-0.17, 0.5, f"State {row + 1}", transform=axes[row, 0].transAxes,
                            rotation=90, ha="center", va="center", fontsize=15,
                            fontweight="bold", color=colour)
        axes[row, 0].text2D(-0.04, 0.5, f"FO {fo[k]:.3f}\noffset {offsets[k]:+.2f} SD\n{note}",
                            transform=axes[row, 0].transAxes, rotation=90, ha="center",
                            va="center", fontsize=8, color=style.TEXT_MUTED)
    for ax in axes.ravel():
        ax.set_facecolor("none")
        ax.patch.set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.88, top=1 - 0.68 / (2.43 * K),
                        bottom=0.015, wspace=-0.10, hspace=-0.30)
    for row in range(K):
        box = axes[row, 3].get_position()
        cax = fig.add_axes([0.90, box.y0 + 0.30 * box.height, 0.010, 0.40 * box.height])
        norm = plt.Normalize(vmin=-scales[row], vmax=scales[row])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cold_hot), cax=cax)
        cb.set_ticks([-scales[row], 0.0, scales[row]])
        cb.set_ticklabels([f"{-scales[row]:.2f}", "0", f"{scales[row]:+.2f}"])
        cb.ax.tick_params(labelsize=8, length=2)
        cb.outline.set_visible(False)
    fig.text(0.965, 0.5, "Relative amplitude envelope (SD)", rotation=90, ha="center",
             va="center", fontsize=11, fontweight="bold")
    fig.suptitle(f"K = {K}", fontsize=16, fontweight="bold", y=1 - 0.15 / (2.43 * K))
    for ax in axes.ravel():
        for coll in ax.collections:
            coll.set_rasterized(True)
    fig.set_dpi(300)
    style.save_figure(fig, path)
    plt.close(fig)


if "--selftest" in sys.argv:
    # Draw the analysed fit through this code, to compare with Figure 1.
    import matplotlib
    matplotlib.use("Agg")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    draw(7, 0, ref, ref_fo, FIGDIR / "selftest_K7_2")
    print("selftest done")
    sys.exit(0)

# ---------------------------------------------------------------- all fits
rows, fits = [], {}
for d in sorted(ROOT.glob("K??_run*")):
    s = load(d / "summary.mat")
    K, run = int(s["K"]), int(s["run"])
    fo = np.atleast_2d(s["fo"]).mean(0)
    rows.append(dict(K=K, run=run, free_energy=float(s["fe_final"]),
                     free_energy_hmmfe=float(s["fe_recomputed"]),
                     cycles=int(s["cycles"]), hours=float(s["elapsed_s"]) / 3600,
                     min_group_fo=fo.min(), max_group_fo=fo.max()))
    fits[(K, run)] = (s, fo)
runs = pd.DataFrame(rows).sort_values(["K", "run"])
runs.to_csv(ROOT / "kseries_runs.csv", index=False)
print(runs.to_string(index=False))

best = runs.loc[runs.groupby("K")["free_energy"].idxmin()].reset_index(drop=True)
spread = runs.groupby("K")["free_energy"].agg(n_runs="count", fe_min="min", fe_max="max")
best = best.merge(spread, on="K")

match_rows = []
for _, b in best.iterrows():
    s, fo = fits[(int(b.K), int(b.run))]
    C = corr_rows(ref_rel, relative(s["means"], fo))          # 7 x K
    r_best = C.max(axis=1)
    match_rows.append({"K": int(b.K), **{f"state{i + 1}_best_r": r_best[i] for i in range(7)}})
match = pd.DataFrame(match_rows)
best = best.merge(match, on="K")
best.to_csv(ROOT / "kseries_best.csv", index=False)
print(best.to_string(index=False))



if "--no-figures" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for _, b in best.iterrows():
        K, run = int(b.K), int(b.run)
        s, fo = fits[(K, run)]
        draw(K, run, s, fo, FIGDIR / f"figure_S_kseries_K{K:02d}")
        print(f"figure K={K} done", flush=True)
print("done")
