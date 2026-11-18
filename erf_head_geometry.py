"""Step 11h: head geometry per recording, read from FIF headers only.

For each participant: the device-to-head transform, and the head radius from a sphere fitted to
the digitised extra head-shape points. Any participant with fewer than --min-points extra points
is a hard failure; there is no fallback to other point kinds.

Writes --out (CSV): subject_id, n_extra_points, head_radius_mm, sphere origin (head frame, mm),
the transform's translation (mm) and its deviation from the first recording's transform
(translation in mm, rotation in degrees), and the maxfilter frame and origin where the
processing history records them.

Usage:
  python erf_head_geometry.py --fif-dir <Clean> --ids <subject_level.csv> --out <csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.io.constants import FIFF


def rotation_angle_deg(R: np.ndarray, R_ref: np.ndarray) -> float:
    cos = (np.trace(R @ R_ref.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def maxfilter_history(info) -> dict:
    out = {}
    try:
        sss = info["proc_history"][0]["max_info"]["sss_info"]
        out["sss_frame"] = int(sss.get("frame", -1))
        origin = sss.get("origin")
        if origin is not None:
            out["sss_origin_mm"] = " ".join(f"{v * 1000:.1f}" for v in np.ravel(origin))
    except (IndexError, KeyError, TypeError):
        out["sss_frame"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fif-dir", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True, help="CSV with a subject_id column")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-points", type=int, default=20)
    args = ap.parse_args()

    ids = pd.read_csv(args.ids, dtype={"subject_id": str})["subject_id"].tolist()
    rows, transforms = [], []
    for sid in ids:
        info = mne.io.read_info(args.fif_dir / f"{sid}_cleanica_sss.fif", verbose="error")
        n_extra = sum(d["kind"] == FIFF.FIFFV_POINT_EXTRA for d in (info["dig"] or []))
        if n_extra < args.min_points:
            raise ValueError(f"{sid}: {n_extra} extra head-shape points (< {args.min_points})")
        radius, origin_head, _ = mne.bem.fit_sphere_to_headshape(
            info, dig_kinds=("extra",), units="mm", verbose="error")
        T = info["dev_head_t"]["trans"]
        transforms.append(T)
        rows.append({"subject_id": sid, "n_extra_points": n_extra,
                     "head_radius_mm": float(radius),
                     "sphere_origin_head_mm": " ".join(f"{v:.1f}" for v in origin_head),
                     "dev_head_translation_mm": " ".join(f"{v * 1000:.2f}" for v in T[:3, 3]),
                     **maxfilter_history(info)})

    ref = transforms[0]
    for row, T in zip(rows, transforms):
        row["translation_from_first_mm"] = float(np.linalg.norm(T[:3, 3] - ref[:3, 3]) * 1000)
        row["rotation_from_first_deg"] = rotation_angle_deg(T[:3, :3], ref[:3, :3])
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"{len(table)} recordings")
    print(f"  extra head-shape points: {table.n_extra_points.min()} to {table.n_extra_points.max()}")
    print(f"  head radius: {table.head_radius_mm.min():.1f} to {table.head_radius_mm.max():.1f} mm "
          f"(mean {table.head_radius_mm.mean():.1f}, sd {table.head_radius_mm.std():.1f})")
    print(f"  device-to-head transform vs the first recording: translation up to "
          f"{table.translation_from_first_mm.max():.3f} mm, rotation up to "
          f"{table.rotation_from_first_deg.max():.3f} deg")
    print(f"  maxfilter frame codes recorded: {sorted(table.sss_frame.dropna().unique().tolist())}")


if __name__ == "__main__":
    main()
