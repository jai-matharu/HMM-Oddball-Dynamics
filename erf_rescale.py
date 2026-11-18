"""Step 11b': return the re-averaged ERFs to physical units.

`HMM_Beamforming.m` ran OSL's `osl_normalise_sensor_data` on each `filteredspmeeg_` file,
which stores an online montage, `normalised_sensors`, scaling every channel of a sensor type
by 1 / sqrt(mean eigenvalue) of that type's covariance in that recording, and leaves it
active. `HMM_Epoching.m` then downsampled the same file, and SPM read it through the active
montage, so every ERF downstream is in per-recording normalised units while still labelled
fT and fT/mm.

The montage is read back from each `filteredspmeeg_` header and divided out of the averages
from 11b. Nothing else changes: within a recording the montage is one constant per sensor
type, so 11a's epoch selection and 11b's averaging are untouched by it.

Checks, each a hard failure:
  - the montage is named `normalised_sensors`, is the active one, and is diagonal over the
    MEG channels;
  - its weight is constant within each sensor type of a recording;
  - the files the ERF chain read afterwards carry no further active scaling: the montage
    weight reproduces the ratio measured independently in `scale_check.csv` to 0.5%.

Writes to --out-dir:
  erf_averages_fT.npz      - erf_averages.npz with `corrected` and `all_epochs` in fT, fT/mm
  erf_scale_factors.csv    - per recording: the montage weight per sensor type

Usage:
  python erf_rescale.py --averages <erf_averages.npz> --spm-dir <SPM_Files>
                        --scale-check <scale_check.csv> --out-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

TYPES = ("MEGMAG", "MEGPLANAR")


def montage_weights(path: Path, labels: list[str]) -> tuple[np.ndarray, dict]:
    D = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["D"]
    active = int(D.montage.Mind)
    montages = np.atleast_1d(D.montage.M)
    if active < 1:
        raise ValueError(f"{path.name}: no active montage")
    M = montages[active - 1]
    if str(M.name) != "normalised_sensors":
        raise ValueError(f"{path.name}: active montage is {M.name!r}")
    tra = np.atleast_2d(M.tra).astype(float)
    original = [str(x) for x in np.atleast_1d(M.labelorg)]
    new = [str(x) for x in np.atleast_1d(M.labelnew)]
    rows = [new.index(label) for label in labels]
    cols = [original.index(label) for label in labels]
    block = tra[np.ix_(rows, cols)]
    weights = np.diag(block).copy()
    off = block - np.diag(weights)
    row_sums = np.abs(tra[rows]).sum(axis=1)
    if np.abs(off).max() != 0 or np.abs(row_sums / np.abs(weights) - 1).max() > 1e-12:
        raise ValueError(f"{path.name}: montage is not diagonal over the MEG channels")
    return weights, {"montage": str(M.name), "active": active, "n_montages": len(montages)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--averages", type=Path, required=True)
    parser.add_argument("--spm-dir", type=Path, required=True)
    parser.add_argument("--scale-check", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    store = np.load(args.averages, allow_pickle=False)
    subjects = [str(s) for s in store["subject_id"]]
    labels = [str(label) for label in store["channel_label"]]
    types = np.array([str(t) for t in store["channel_type"]])
    measured = pd.read_csv(args.scale_check, dtype={"subject_id": str}).set_index("subject_id")

    weights, rows = [], []
    for sid in subjects:
        w, meta = montage_weights(args.spm_dir / f"filteredspmeeg_{sid}_cleanica_sss.mat", labels)
        row = {"subject_id": sid, **meta}
        for t in TYPES:
            values = w[types == t]
            if values.max() / values.min() - 1 > 1e-6:
                raise ValueError(f"{sid} {t}: montage weight varies within the type")
            row[f"{t}_weight"] = float(values[0])
            ratio = measured.loc[sid, f"{t}_slope_median"] / values[0]
            row[f"{t}_measured_over_montage"] = float(ratio)
            if abs(ratio - 1) > 0.005:
                raise ValueError(f"{sid} {t}: montage {values[0]:.6g} against measured "
                                 f"{measured.loc[sid, f'{t}_slope_median']:.6g}")
        weights.append(w)
        rows.append(row)
    weights = np.array(weights)                                   # subject x channel

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "erf_scale_factors.csv", index=False)

    out = {key: store[key] for key in store.files}
    for key in ("corrected", "all_epochs"):
        out[key] = (store[key] / weights[:, None, :, None]).astype(np.float32)
    np.savez_compressed(args.out_dir / "erf_averages_fT.npz", **out)

    print(f"{len(subjects)} recordings; every active montage is 'normalised_sensors', "
          f"diagonal, one weight per sensor type")
    for t in TYPES:
        w = table[f"{t}_weight"]
        r = table[f"{t}_measured_over_montage"]
        print(f"  {t}: weight {w.min():.6g} to {w.max():.6g} ({w.max() / w.min():.1f}-fold); "
              f"measured/montage {r.min():.4f} to {r.max():.4f}")
    mag = np.abs(out["corrected"][:, :, types == "MEGMAG"]).max(axis=(1, 2, 3))
    print(f"  peak |field| over magnetometers, median across recordings: {np.median(mag):.1f} fT")


if __name__ == "__main__":
    main()
