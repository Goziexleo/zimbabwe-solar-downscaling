"""How much genuine sub-0.25 degree information does a field on the fine grid carry?

The test: degrade a 0.1 degree field onto the 0.25 degree coarse grid, then
interpolate it back up. Whatever survives that round trip was never finer than
0.25 degrees to begin with. What is destroyed is the genuine sub-grid content.

A null result is only meaningful alongside evidence that the test can produce a
non-null one, so elevation - a field that genuinely varies below 0.25 degrees -
is always evaluated as a positive control in the same run.

Usage:
    python compute_information_content.py
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from ml_dataset_common import FINE_LAT, FINE_LON

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_PATH = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
FIELDS_PATH = os.path.join(ROOT, "data/processed/evaluation/validation_spatial_fields.nc")
CLEARSKY_PATH = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
TOPO_PATH = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
OUT_CSV = os.path.join(ROOT, "data/processed/evaluation/information_content.csv")


def retained_after_round_trip(field_2d, coarse_lat, coarse_lon):
    """Return (pct_variance_lost, correlation) for one 2-D fine-grid field."""
    da = xr.DataArray(
        field_2d, dims=["lat", "lon"], coords={"lat": FINE_LAT, "lon": FINE_LON}
    )
    round_tripped = (
        da.interp(lat=coarse_lat, lon=coarse_lon, method="linear")
          .interp(lat=FINE_LAT, lon=FINE_LON, method="cubic")
          .values
    )
    valid = np.isfinite(round_tripped) & np.isfinite(field_2d)
    a, b = field_2d[valid], round_tripped[valid]
    pct_lost = 100.0 * np.var(a - b) / np.var(a)
    return pct_lost, float(np.corrcoef(a, b)[0, 1])


def main():
    ds_val = xr.open_dataset(VAL_PATH)
    coarse_lat, coarse_lon = ds_val["lat"].values, ds_val["lon"].values

    candidates = []

    # --- the deliverable and its intermediate ---
    if os.path.exists(FIELDS_PATH):
        ds_f = xr.open_dataset(FIELDS_PATH)
        for var, label in [
            ("ghi_true", "GHI target (time-mean)"),
            ("ghi_rf", "GHI predicted, Random Forest"),
            ("ghi_unet", "GHI predicted, U-Net"),
        ]:
            if var in ds_f:
                candidates.append((label, ds_f[var].values.mean(axis=0)))
    candidates.append(
        ("CSI target (time-mean)", ds_val["clear_sky_index"].values.mean(axis=0))
    )

    # --- the clear-sky normaliser ---
    if os.path.exists(CLEARSKY_PATH):
        cs = xr.open_dataset(CLEARSKY_PATH)["clearsky_ghi"].values
        candidates.append(("Clear-sky ceiling (January)", cs[0]))

    # --- positive control: a field that genuinely varies below 0.25 deg ---
    topo = xr.open_dataset(TOPO_PATH)
    lat_name = "lat" if "lat" in topo.coords else "y"
    lon_name = "lon" if "lon" in topo.coords else "x"
    topo = topo.rename({lat_name: "lat", lon_name: "lon"}).sortby("lat")
    elev = topo["elevation"].interp(lat=FINE_LAT, lon=FINE_LON, method="linear").values
    candidates.append(("CONTROL: elevation", elev))

    rows = []
    for label, field in candidates:
        pct_lost, corr = retained_after_round_trip(field, coarse_lat, coarse_lon)
        rows.append({
            "field": label,
            "pct_variance_below_0.25deg": pct_lost,
            "round_trip_correlation": corr,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("=" * 92)
    print(" SUB-0.25 DEGREE INFORMATION CONTENT (variance destroyed by a coarse round trip)")
    print("=" * 92)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("=" * 92)

    control = df[df["field"].str.startswith("CONTROL")]["pct_variance_below_0.25deg"].iloc[0]
    if control < 0.1:
        print("\nWARNING: the positive control lost almost no variance either, so this run "
              "cannot distinguish 'no fine-scale information' from 'the test is not working'.")
    else:
        print(f"\nControl check passed: elevation loses {control:.4f}% of its variance, so the "
              "test does detect genuine sub-grid structure when it is present.")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
