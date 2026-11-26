"""Supplementary figure: how the 52 children who completed the MEG protocol became
the analysis sample of 39.

Counts: 52 completed, 11 excluded for more than 60% bad segments and 1 for a failed
co-registration (accounting from the preprocessing records), leaving 40; 99114
excluded before fitting K7_2 (see EXCLUDED in prepare_data.py for the evidence).

    python sample_flow_figure.py --hmm-meg ~/Github/HMM_MEG --output-dir figures
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# (y of the bottom edge, height, title, n, note) in inches; the axes span the figure.
MAIN = [
    (3.60, 0.80, "Completed the MEG protocol", "n = 52", None),
    (1.95, 0.80, "Fully preprocessed", "n = 40", None),
    (0.05, 1.05, "Analysis sample", "n = 39",
     "Hidden Markov model, event-related and individual-difference analyses"),
]
EXCLUSIONS = [
    ("Excluded during preprocessing (n = 12)",
     ["Bad segments in more than 60% of the recording (n\u00a0=\u00a011)",
      "Failed co-registration of MEG and structural data (n\u00a0=\u00a01)"]),
    ("Excluded before model fitting (n = 1)",
     ["In a model fitted to all 40, one state held 65.1% of this child's recording "
      "and 50.0% of that state's occupancy across the sample"]),
]
LINE = 0.21  # inches per 11 pt line at 1.3 spacing
# Counts carry non-breaking spaces so that "(n = 11)" never wraps across lines.


def box(ax, x, y, w, h, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=0.8))


def arrow(ax, start, end, colour):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=0.8,
                                shrinkA=0, shrinkB=0, mutation_scale=12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hmm-meg", type=Path, required=True, help="HMM_MEG checkout (for the style)")
    p.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    sys.path.insert(0, str(args.hmm_meg))
    from analysis.viz import style

    style.apply_style()
    width, height = style.FIGURE_SIZES["double_column"]
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    main_x, main_w = 0.1, 4.0
    cx = main_x + main_w / 2
    for y, h, title, n, note in MAIN:
        box(ax, main_x, y, main_w, h, style.NEUTRAL_LIGHT_BLUE, style.TEXT_PRIMARY)
        top = y + h - 0.14
        ax.text(cx, top, title, ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(cx, top - 0.26, n, ha="center", va="top", fontsize=11)
        if note:
            ax.text(cx, top - 0.26 - LINE - 0.04, textwrap.fill(note, 44), ha="center",
                    va="top", fontsize=11, color=style.TEXT_SECONDARY, linespacing=1.3)

    for (y_up, *_), (y_lo, h_lo, *_) in zip(MAIN[:-1], MAIN[1:]):
        arrow(ax, (cx, y_up), (cx, y_lo + h_lo), style.TEXT_PRIMARY)

    side_x, side_w = 4.9, 5.0
    for (title, reasons), (y_up, *_), (y_lo, h_lo, *_) in zip(EXCLUSIONS, MAIN[:-1], MAIN[1:]):
        body = "\n".join(textwrap.fill(r, 58) for r in reasons)
        n_lines = body.count("\n") + 1
        side_h = 0.44 + n_lines * LINE + 0.14
        y_mid = (y_up + y_lo + h_lo) / 2
        top = y_mid + side_h / 2
        box(ax, side_x, top - side_h, side_w, side_h, "white", style.NEUTRAL_GREY)
        arrow(ax, (cx, y_mid), (side_x, y_mid), style.TEXT_PRIMARY)
        ax.text(side_x + 0.15, top - 0.12, title, ha="left", va="top",
                fontsize=11, fontweight="bold")
        ax.text(side_x + 0.15, top - 0.44, body, ha="left", va="top",
                fontsize=11, color=style.TEXT_SECONDARY, linespacing=1.3)

    for path in style.save_figure(fig, args.output_dir / "figure_S9_sample_flow"):
        print(path)


if __name__ == "__main__":
    main()
