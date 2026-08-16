"""Section 3.4.3 quality control: physical-bounds enforcement and flag counts.

Section 3.4.3 states that physically unrealistic values in the CMIP6 fields are
detected and replaced, and that per-cell flag counts were not tabulated. Both
halves needed work. The check was not implemented at all, and the violations are
real: equidistant CDF matching is a tail-sensitive transform, and correcting a
bounded variable against an unbounded empirical CDF pushes a small fraction of
values past the bound. Cloud fraction reaches -5.93 per cent and 100.89 per cent
in the corrected fields; aerosol optical depth goes marginally negative.

The fractions are small - 0.64 per cent of cloud-fraction values, 0.0018 per cent
of aerosol - but a negative cloud fraction is not a small error in kind, and it
is fed to models trained on a strictly non-negative predictor. That is the same
out-of-range extrapolation that Section 7.11 shows tree ensembles handle by
saturating.

This script enforces the bounds and writes the per-cell counts that Section 3.4.3
promised the appendix. Originals are preserved under cmip6_bias_corrected_preqc/
so the effect of the correction can be measured rather than assumed.

    python apply_qc_bounds.py            # report only, changes nothing
    python apply_qc_bounds.py --apply    # back up, clamp, write counts
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
CORRECTED = os.path.join(ROOT, "data/processed/cmip6_bias_corrected")
BACKUP = os.path.join(ROOT, "data/processed/cmip6_bias_corrected_preqc")
EVAL = os.path.join(ROOT, "data/processed/evaluation")
OUT_NC = os.path.join(EVAL, "qc_flag_counts.nc")
OUT_CSV = os.path.join(EVAL, "qc_flag_counts.csv")

# (lower, upper); None means unbounded on that side. These are physical limits,
# not tuning choices: a cloud fraction outside [0, 100] per cent, a negative
# optical depth, a negative specific humidity or a negative irradiance have no
# physical interpretation.
PHYSICAL_BOUNDS = {
    "clt": (0.0, 100.0),
    "huss": (0.0, None),
    "rsds": (0.0, None),
    "od550aer": (0.0, None),
    "ps": (0.0, None),
    "tas": (0.0, None),
}


def clamp(ds):
    """Enforce physical bounds. Returns (clamped dataset, per-cell count maps)."""
    out = ds.copy()
    counts = {}
    for var, (lo, hi) in PHYSICAL_BOUNDS.items():
        if var not in ds.data_vars:
            continue
        a = ds[var].values
        viol = np.zeros(a.shape, dtype=bool)
        if lo is not None:
            viol |= a < lo
        if hi is not None:
            viol |= a > hi
        # per-cell count: how many time steps were flagged at each grid cell
        counts[var] = viol.sum(axis=0).astype(np.int32)
        out[var] = (ds[var].dims, np.clip(a, lo, hi))
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="back up originals and write the clamped fields")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(CORRECTED) if f.endswith("_corrected.nc"))
    if not files:
        raise SystemExit(f"no corrected files in {CORRECTED}")

    rows = []
    per_cell_total = {}
    coords = None

    for fn in files:
        path = os.path.join(CORRECTED, fn)
        ds = xr.open_dataset(path)
        clamped, counts = clamp(ds)
        if coords is None:
            coords = {"lat": ds["lat"].values, "lon": ds["lon"].values}
        n_time = ds.sizes["time"]

        for var, cmap in counts.items():
            total = int(cmap.sum())
            per_cell_total[var] = per_cell_total.get(var, 0) + cmap
            a = ds[var].values
            lo, hi = PHYSICAL_BOUNDS[var]
            worst = 0.0
            if total:
                if lo is not None and a.min() < lo:
                    worst = max(worst, lo - float(a.min()))
                if hi is not None and a.max() > hi:
                    worst = max(worst, float(a.max()) - hi)
            rows.append({
                "file": fn, "variable": var,
                "n_values": int(a.size), "n_flagged": total,
                "pct_flagged": 100.0 * total / a.size,
                "cells_affected": int((cmap > 0).sum()),
                "worst_excursion_beyond_bound": worst,
                "n_time": n_time,
            })

        if args.apply:
            os.makedirs(BACKUP, exist_ok=True)
            bpath = os.path.join(BACKUP, fn)
            if not os.path.exists(bpath):
                shutil.copy2(path, bpath)
            ds.close()
            tmp = path + ".tmp"
            clamped.to_netcdf(tmp)
            os.replace(tmp, path)
        else:
            ds.close()

    df = pd.DataFrame(rows)
    os.makedirs(EVAL, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    if per_cell_total:
        xr.Dataset(
            {f"qc_flags_{v}": (("lat", "lon"), c) for v, c in per_cell_total.items()},
            coords=coords,
        ).to_netcdf(OUT_NC)

    print("=" * 84)
    print(" SECTION 3.4.3 QC — physical-bounds violations in the bias-corrected CMIP6 fields")
    print("=" * 84)
    summary = df.groupby("variable").agg(
        n_values=("n_values", "sum"), n_flagged=("n_flagged", "sum"),
        worst=("worst_excursion_beyond_bound", "max"))
    summary["pct"] = 100.0 * summary["n_flagged"] / summary["n_values"]
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("Per-cell flag counts (the appendix diagnostic Section 3.4.3 specifies):")
    for v, c in per_cell_total.items():
        if c.sum():
            print(f"  {v:<10} {int((c > 0).sum())}/{c.size} cells flagged at least once; "
                  f"max {int(c.max())} time steps at a single cell")
    print()
    print(f"Saved: {OUT_CSV}")
    if per_cell_total:
        print(f"Saved: {OUT_NC}")
    if args.apply:
        print(f"Clamped in place; originals preserved in {BACKUP}")
    else:
        print("Report only. Re-run with --apply to enforce the bounds.")


if __name__ == "__main__":
    main()
