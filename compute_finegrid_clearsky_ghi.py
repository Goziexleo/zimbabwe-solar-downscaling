"""Compute the fine (0.1 deg) grid climatological clear-sky GHI ceiling.

Section 3.4.1 requires ERA5 ssrd interpolated to the 0.1 deg target grid,
and Section 3.5.4 normalises GHI by a "cell-specific, month-specific
clear-sky GHI computed from the PVLIB Ineichen model" to build the clear-sky
index (CSI) target. This script evaluates that clear-sky ceiling directly at
the fine target grid (the same grid used by the topographic covariates),
independently of year: solar geometry for a given calendar month is
effectively identical from year to year, so one climatological field per
month is computed and reused for every year in the training/validation
periods rather than recomputed 40 times.

Both Ineichen inputs are genuinely cell-specific, as Section 3.5.4 states.
An earlier version of this script passed altitude=1000.0 m and
linke_turbidity=3.0 as domain-wide constants, which made the field only
month-specific (through solar zenith angle) and not cell-specific at all.
Station altitude is now taken from the SRTM-derived elevation of Section
3.5.2 - the domain spans roughly 124 m in the Lowveld to 1840 m in the
Eastern Highlands, so a single 1000 m value misstates the air mass over most
of the country - and Linke turbidity is read per cell and per month from the
climatology bundled with PVLIB, which over Zimbabwe runs from about 3.9 in
July to about 5.5 in January rather than a flat 3.0.

Note that this changes the CSI intermediate the models learn, but not the
reconstructed GHI target: build_fine_grid_csi divides by this field and
csi_to_ghi multiplies by it again, and the [0, 1.1] CSI clip does not bind
(observed CSI range is roughly 0.23 to 0.57), so the normalisation cancels
exactly in GHI space. The correction is therefore one of physical
defensibility of the intermediate quantity, not a route to finer spatial
detail in the final product.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import h5py
import pvlib

from ml_dataset_common import FINE_LAT, FINE_LON, safe_nan_to_num

TOPO_PATH = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")


def linke_turbidity_grid(lat_1d, lon_1d):
    """Per-cell, per-month Linke turbidity from the climatology shipped with
    PVLIB. pvlib.clearsky.lookup_linke_turbidity only accepts scalar
    coordinates, so the documented index convention is reproduced here in
    vectorised form and checked against the library function below."""
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # latitude: 2160 rows spanning +90 to -90; longitude: 4320 columns spanning -180 to +180
    lat_idx = np.clip(np.around((lat_2d - (90.0 - 1 / 24.0)) * -12.0).astype(int), 0, 2159)
    lon_idx = np.clip(np.around((lon_2d - (-180.0 + 1 / 24.0)) * 12.0).astype(int), 0, 4319)

    # h5py cannot take paired fancy indices, so read the bounding box that
    # covers the domain (a few thousand cells) and index it with numpy.
    lat0, lat1 = int(lat_idx.min()), int(lat_idx.max())
    lon0, lon1 = int(lon_idx.min()), int(lon_idx.max())
    h5_path = os.path.join(os.path.dirname(pvlib.__file__), "data", "LinkeTurbidities.h5")
    with h5py.File(h5_path, "r") as f:
        block = f["LinkeTurbidity"][lat0:lat1 + 1, lon0:lon1 + 1, :]

    return block[lat_idx - lat0, lon_idx - lon0, :] / 20.0

# --- Configuration ---
output_dir = os.path.abspath("./data/processed/era5/csi_finegrid")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "clearsky_ghi_finegrid_climatology.nc")

REFERENCE_YEAR = 2001  # non-leap year; only month/day-of-year matter for solar geometry

print("Using canonical core-domain fine (0.1 deg) target grid (Section 3.2.1, no buffer)...")
fine_lat = FINE_LAT
fine_lon = FINE_LON
n_lat, n_lon = len(fine_lat), len(fine_lon)
print(f"Fine grid: {n_lat} lat x {n_lon} lon = {n_lat * n_lon} cells")

lon_2d, lat_2d = np.meshgrid(fine_lon, fine_lat)
lat_flat = lat_2d.flatten()
lon_flat = lon_2d.flatten()

# --- Cell-specific station altitude (Section 3.5.2 SRTM elevation) ---
print("Loading SRTM-derived elevation for cell-specific altitude...")
ds_topo = xr.open_dataset(TOPO_PATH)
topo_lat_name = "lat" if "lat" in ds_topo.coords else "y"
topo_lon_name = "lon" if "lon" in ds_topo.coords else "x"
ds_topo = ds_topo.rename({topo_lat_name: "lat", topo_lon_name: "lon"}).sortby("lat")
elevation = ds_topo["elevation"].interp(lat=fine_lat, lon=fine_lon, method="linear").values
if np.isnan(elevation).any():
    # The fine grid sits strictly inside the DEM footprint, so any NaN here
    # would signal a grid/DEM mismatch rather than an expected edge effect.
    raise RuntimeError(f"Elevation has {np.isnan(elevation).mean():.4%} NaN on the fine grid.")
elev_flat = elevation.flatten()
pressure_flat = pvlib.atmosphere.alt2pres(elev_flat)
print(f"  Elevation {elev_flat.min():.0f} to {elev_flat.max():.0f} m (mean {elev_flat.mean():.0f} m)")

# --- Cell-specific, month-specific Linke turbidity ---
print("Reading per-cell Linke turbidity climatology (PVLIB)...")
turbidity = linke_turbidity_grid(fine_lat, fine_lon)
_probe = pvlib.clearsky.lookup_linke_turbidity(
    pd.DatetimeIndex([f"{REFERENCE_YEAR}-01-15"]), float(fine_lat[0]), float(fine_lon[0]),
    interp_turbidity=False,
).iloc[0]
assert abs(_probe - turbidity[0, 0, 0]) < 1e-9, (
    f"Vectorised turbidity lookup ({turbidity[0, 0, 0]:.4f}) disagrees with "
    f"pvlib.lookup_linke_turbidity ({_probe:.4f})."
)
print(f"  Turbidity January {turbidity[:, :, 0].min():.2f}-{turbidity[:, :, 0].max():.2f}, "
      f"July {turbidity[:, :, 6].min():.2f}-{turbidity[:, :, 6].max():.2f}")

daytime_hours = np.linspace(6.0, 18.0, 13)
monthly_clearsky = np.zeros((12, n_lat, n_lon))

for month in range(1, 13):
    print(f"  Computing month {month:02d}...")
    accumulated_cs_ghi = np.zeros(lat_flat.shape)
    turbidity_flat = turbidity[:, :, month - 1].flatten()

    for hour in daytime_hours:
        hour_dt = pd.Timestamp(
            year=REFERENCE_YEAR, month=month, day=15,
            hour=int(hour), minute=int((hour % 1) * 60),
        )
        time_repeated = pd.DatetimeIndex([hour_dt] * len(lat_flat))

        solpos = pvlib.solarposition.get_solarposition(time_repeated, lat_flat, lon_flat)
        sza_flat = np.clip(solpos["apparent_zenith"].values, 0.0, 89.9)

        rel_airmass = pvlib.atmosphere.get_relative_airmass(sza_flat)
        abs_airmass = pvlib.atmosphere.get_absolute_airmass(rel_airmass, pressure_flat)

        dni_extra = pvlib.irradiance.get_extra_radiation(hour_dt)
        clearsky = pvlib.clearsky.ineichen(
            sza_flat, abs_airmass, linke_turbidity=turbidity_flat,
            altitude=elev_flat, dni_extra=dni_extra,
        )
        ghi_hr = clearsky["ghi"].values if hasattr(clearsky["ghi"], "values") else clearsky["ghi"]
        accumulated_cs_ghi += safe_nan_to_num(ghi_hr, "compute_finegrid_clearsky_ghi:ghi_hr")

    mean_cs_ghi = accumulated_cs_ghi / len(daytime_hours)
    mean_cs_ghi = np.where(mean_cs_ghi < 1.0, 1.0, mean_cs_ghi)
    monthly_clearsky[month - 1] = mean_cs_ghi.reshape(lat_2d.shape)

ds_out = xr.Dataset(
    {"clearsky_ghi": (("month", "lat", "lon"), monthly_clearsky)},
    coords={"month": np.arange(1, 13), "lat": fine_lat, "lon": fine_lon},
)
ds_out["clearsky_ghi"].attrs["units"] = "W m-2"
ds_out["clearsky_ghi"].attrs["description"] = (
    "Daytime-averaged (06-18h local) monthly-climatological clear-sky GHI "
    "(PVLIB Ineichen model) at the 0.1 deg fine target grid, Section 3.5.4. "
    "Station altitude is the per-cell SRTM elevation and Linke turbidity is "
    "the per-cell, per-month PVLIB climatology."
)
ds_out.to_netcdf(output_path)
print(f"\nSuccess! Fine-grid clear-sky GHI climatology saved to: {output_path}")
