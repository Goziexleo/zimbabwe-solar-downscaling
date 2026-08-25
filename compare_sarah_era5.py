"""SARAH against the ERA5-derived GHI, on the 0.1 degree target grid.

Every skill figure in this project is measured against ERA5, which is also what
the models were trained on. SARAH is the first reference here that is neither:
an independent satellite retrieval at 0.05 degrees, finer than the target grid.
This quantifies the difference over the full 1985-2024 record.

It exists because of a decision in the suitability analysis. Section 3.9 needs a
present-day irradiance layer, and there are two candidates - the ERA5-derived
climatology the models reproduce, and SARAH. Picking one without measuring the
gap would put an unexamined choice underneath every suitability map.

Two grid facts matter. SARAH is cell-centred (25.025, 25.075, ...) while the
target grid is node-centred (25.0, 25.1, ...), so NONE of the coordinates
coincide and this interpolates rather than block-averages. And the order was
placed without margin, so the target perimeter - 300 of 5,751 cells - lies
0.025 degrees outside SARAH's cell-centre hull; those cells are reported
separately rather than silently extrapolated.

    python compare_sarah_era5.py
"""

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data/processed")
OUT_NC = os.path.join(PROC, "evaluation/sarah_era5_comparison.nc")
OUT_CSV = os.path.join(PROC, "evaluation/sarah_era5_monthly.csv")


def era5_ghi_stack():
    """ERA5-derived 0.1 deg GHI, month by month, 1985-2024."""
    cs = xr.open_dataset(os.path.join(
        PROC, "era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc"))
    times, fields = [], []
    for f in ("ml_training_dataset.nc", "ml_validation_dataset.nc"):
        d = xr.open_dataset(os.path.join(PROC, "ml_ready", f))
        t = pd.DatetimeIndex(d.time.values)
        csi = d["clear_sky_index"].values
        for i, ts in enumerate(t):
            fields.append(csi[i] * cs.clearsky_ghi.isel(month=ts.month - 1).values)
            times.append(ts)
        lat, lon = d.fine_lat.values, d.fine_lon.values
    idx = np.argsort(times)
    return (xr.DataArray(np.array(fields)[idx],
                         coords={"time": pd.DatetimeIndex(times)[idx],
                                 "lat": lat, "lon": lon},
                         dims=("time", "lat", "lon")))


def sarah_stack(lat, lon):
    files = sorted(glob.glob(os.path.join(ROOT, "data/raw/sarah/SISmm*.nc")))
    stamp = lambda f: pd.Timestamp(re.search(r"SISmm(\d{8})", f).group(1))
    d = xr.open_mfdataset(files, combine="nested", concat_dim="time",
                          preprocess=lambda x: x[["SIS"]])
    d = d.assign_coords(time=pd.DatetimeIndex([stamp(f) for f in files]))
    print("  SARAH stack %s -> interpolating to the target grid" % (d.SIS.shape,))
    return d.SIS.interp(lat=lat, lon=lon, method="linear").compute()


def main():
    era5 = era5_ghi_stack()
    lat, lon = era5.lat.values, era5.lon.values
    print("ERA5-derived stack:", era5.shape)
    sar = sarah_stack(lat, lon)

    # month-end (ERA5) against month-start (SARAH): align on calendar month
    era5 = era5.assign_coords(ym=("time", pd.PeriodIndex(
        pd.DatetimeIndex(era5.time.values), freq="M").astype(str)))
    sar = sar.assign_coords(ym=("time", pd.PeriodIndex(
        pd.DatetimeIndex(sar.time.values), freq="M").astype(str)))
    common = sorted(set(era5.ym.values) & set(sar.ym.values))
    print("  months in common: %d (%s .. %s)" % (len(common), common[0], common[-1]))
    e = era5.isel(time=[list(era5.ym.values).index(m) for m in common]).values
    s = sar.isel(time=[list(sar.ym.values).index(m) for m in common]).values

    interior = ~np.isnan(s).any(axis=0)
    print("  cells with SARAH everywhere: %d of %d (%.1f%%)"
          % (interior.sum(), interior.size, 100 * interior.sum() / interior.size))

    diff = e - s                                        # ERA5 minus SARAH
    bias_map = np.nanmean(diff, axis=0)
    with np.errstate(invalid="ignore"):
        ratio_map = np.nanmean(e, axis=0) / np.nanmean(s, axis=0)

    ei, si = e[:, interior], s[:, interior]
    mb = np.nanmean(ei - si)
    print("\n%s\n RESULT\n%s" % ("=" * 72, "=" * 72))
    print("  ERA5 mean %.2f | SARAH mean %.2f W m-2" % (np.nanmean(ei), np.nanmean(si)))
    print("  mean bias (ERA5 - SARAH) %+.2f W m-2  (%+.1f%%)"
          % (mb, 100 * mb / np.nanmean(si)))
    print("  RMSE %.2f | spatial-mean correlation %.4f"
          % (np.sqrt(np.nanmean((ei - si) ** 2)),
             np.corrcoef(np.nanmean(ei, axis=1), np.nanmean(si, axis=1))[0, 1]))
    print("  bias range across cells: %+.2f to %+.2f W m-2"
          % (np.nanmin(bias_map[interior]), np.nanmax(bias_map[interior])))
    print("  ratio range across cells: %.3f to %.3f"
          % (np.nanmin(ratio_map[interior]), np.nanmax(ratio_map[interior])))
    print("  ratio sd across cells: %.4f  <- if small, the offset is near-uniform"
          % np.nanstd(ratio_map[interior]))

    months = pd.PeriodIndex(common, freq="M")
    rows = pd.DataFrame({
        "month": common,
        "era5_wm2": [np.nanmean(x[interior]) for x in e],
        "sarah_wm2": [np.nanmean(x[interior]) for x in s],
    })
    rows["bias"] = rows.era5_wm2 - rows.sarah_wm2
    rows["ratio"] = rows.era5_wm2 / rows.sarah_wm2
    rows.to_csv(OUT_CSV, index=False)

    print("\n  by calendar month (domain mean):")
    bym = rows.assign(m=months.month).groupby("m")[["era5_wm2", "sarah_wm2", "bias", "ratio"]].mean()
    for m, r in bym.iterrows():
        print("    %2d  ERA5 %6.2f  SARAH %6.2f  bias %+6.2f  ratio %.3f"
              % (m, r.era5_wm2, r.sarah_wm2, r.bias, r.ratio))

    out = xr.Dataset(
        {"bias_era5_minus_sarah": (("lat", "lon"), bias_map),
         "ratio_era5_over_sarah": (("lat", "lon"), ratio_map),
         "sarah_mean": (("lat", "lon"), np.nanmean(s, axis=0)),
         "era5_mean": (("lat", "lon"), np.nanmean(e, axis=0)),
         "sarah_complete": (("lat", "lon"), interior)},
        coords={"lat": lat, "lon": lon})
    out.attrs["months"] = "%s..%s" % (common[0], common[-1])
    out.to_netcdf(OUT_NC)
    print("\nSaved %s\n      %s" % (OUT_NC, OUT_CSV))


if __name__ == "__main__":
    main()
