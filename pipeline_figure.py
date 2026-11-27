"""Figure 1: MEG preprocessing and analysis pipeline, coloured by the software used at
each step.

Sources for each step:
- tSSS: the maxfilter-2.2.12 command files written by RED Tools (identical settings).
- ICA: HMM_MaxfilterAndICA_*.py (automatic RED Tools pass, then a manual pass).
- Co-registration, 1-30 Hz filter: SPM12; sensor normalisation, beamforming, bad-segment
  marking (oslview) and parcellation: OSL (HMM_Beamforming.m, HMM_BadSegments.m,
  HMM_ApplyParcellation_38.m). No PCA step: the option is off in HMM_FinishPreprocessing.m.
- The event-related branch leaves after sensor normalisation: HMM_Epoching.m downsampled the
  normalised filteredspmeeg_ files, hence erf_rescale.py restoring fT and fT/mm.

    python pipeline_figure.py --hmm-meg ~/Github/HMM_MEG --output-dir figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Patch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hmm-meg", type=Path, required=True, help="HMM_MEG checkout (for the style)")
    p.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    sys.path.insert(0, str(args.hmm_meg))
    from analysis.viz import style

    c = style.COLORS
    software = {
        "maxfilter": ("MaxFilter (Elekta), called through RED Tools", c["vermillion"]),
        "mne": ("MNE-Python", c["orange"]),
        "spm": ("SPM12", c["bluish_green"]),
        "osl": ("OHBA Software Library", c["blue"]),
        "hmmmar": ("HMM-MAR", c["reddish_purple"]),
        "python": ("Python (this study)", c["sky_blue"]),
    }

    shared = [
        ("Raw recordings", None),
        ("tSSS, movement\ncompensation", "maxfilter"),
        ("Automatic and\nmanual ICA", "mne"),
        ("Co-registration\nwith MRI", "spm"),
        ("Band-pass\nfilter, 1–30 Hz", "spm"),
        ("Sensor\nnormalisation", "osl"),
    ]
    erf = [
        ("Downsampling to 250 Hz", "spm"),
        ("Epoching, −200 to 800 ms; baseline correction", "spm"),
        ("Removal of surplus and mistimed epochs", "python"),
        ("Averaging by stimulus type", "python"),
        ("Restoring physical units (fT, fT/mm)", "python"),
        ("Cluster-based permutation tests\n(three contrasts, two sensor types)", "mne"),
        ("Amplitude per participant and window", "python"),
    ]
    hmm = [
        ("Beamforming (LCMV, 8 mm grid)", "osl"),
        ("Marking of bad segments", "osl"),
        ("Parcellation (38 cortical regions)", "osl"),
        ("Detrending and standardisation", "hmmmar"),
        ("Leakage correction\n(symmetric orthogonalisation)", "hmmmar"),
        ("Amplitude envelope (Hilbert transform)", "hmmmar"),
        ("Downsampling to 250 Hz", "hmmmar"),
        ("HMM inference (7 states, 39 participants)", "hmmmar"),
    ]
    final = ("Statistical models: state measures against behaviour\n"
             "and against event-related amplitudes", "python")

    style.apply_style()
    width, height = style.FIGURE_SIZES["full_page"]
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    def panel(x, y, w, h, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=to_rgba(style.NEUTRAL_GREY, 0.12),
                                    edgecolor="none", zorder=0))
        ax.text(x + 0.18, y + h - 0.12, title, ha="left", va="top", fontsize=12,
                fontweight="bold")

    def step(x, y, w, h, label, key):
        if key is None:
            fill, edge = "white", style.TEXT_PRIMARY
        else:
            colour = software[key][1]
            fill, edge = to_rgba(colour, 0.30), colour
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11,
                linespacing=1.25, zorder=3)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=style.TEXT_PRIMARY, linewidth=0.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=11))

    # Shared sensor-space preprocessing, left to right.
    panel(0.15, 8.15, 9.7, 1.8, "Shared preprocessing (sensor space)")
    sw, sh, sy, sx0 = 1.4, 0.9, 8.35, 0.35
    sgap = (9.65 - sx0 - len(shared) * sw) / (len(shared) - 1)
    xs = [sx0 + i * (sw + sgap) for i in range(len(shared))]
    for x, (label, key) in zip(xs, shared):
        step(x, sy, sw, sh, label, key)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow((x0 + sw, sy + sh / 2), (x1, sy + sh / 2))

    # The two branches.
    col_w, bh, gap, top = 4.1, 0.54, 0.15, 7.05
    branches = [(0.15, "Event-related fields (sensor space)", erf),
                (5.15, "Hidden Markov model (source space)", hmm)]
    panel_y, panel_top = 1.55, 7.55
    centres, bottoms = [], []
    for px, title, steps in branches:
        panel(px, panel_y, 4.7, panel_top - panel_y, title)
        bx = px + 0.3
        ys = [top - bh - i * (bh + gap) for i in range(len(steps))]
        for y, (label, key) in zip(ys, steps):
            step(bx, y, col_w, bh, label, key)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            arrow((bx + col_w / 2, y0), (bx + col_w / 2, y1 + bh))
        centres.append(bx + col_w / 2)
        bottoms.append(ys[-1])

    # Fork from the last shared step into both branches.
    bus_y = 7.85
    last_cx = xs[-1] + sw / 2
    ax.plot([last_cx, last_cx], [sy, bus_y], color=style.TEXT_PRIMARY, linewidth=0.8, zorder=1)
    ax.plot([centres[0], last_cx], [bus_y, bus_y], color=style.TEXT_PRIMARY, linewidth=0.8,
            zorder=1)
    for cx in centres:
        arrow((cx, bus_y), (cx, panel_top))

    # Both branches feed the statistical models.
    fx, fy, fw, fh = 1.5, 0.75, 7.0, 0.6
    step(fx, fy, fw, fh, *final)
    for cx, b in zip(centres, bottoms):
        arrow((cx, b), (cx, fy + fh))

    handles = [Patch(facecolor=to_rgba(colour, 0.30), edgecolor=colour, linewidth=1.2,
                     label=label) for label, colour in software.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=11, handlelength=1.6, columnspacing=2.0)

    for path in style.save_figure(fig, args.output_dir / "figure_1_pipeline"):
        print(path)


if __name__ == "__main__":
    main()
