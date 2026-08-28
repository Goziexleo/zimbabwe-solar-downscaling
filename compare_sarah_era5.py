"""SARAH against the ERA5-derived GHI, on the 0.1 degree target grid.

Every skill figure in this project is measured against ERA5, which is also what
the models were trained on. SARAH is the first reference here that is neither:
an independent satellite retrieval at 0.05 degrees, finer than the target grid.
This quantifies the difference over the full 1985-2024 record.

It exists because of a decision in the suitability analysis. Section 3.9 needs a
present-day irradiance layer, and there are two candidates - the ERA5-derived
climatology the models reproduce, and SARAH. Picking one without measuring the
gap would put an unexamined choice underneath every suitability map.

It is RESUMABLE. Each month's interpolated field is cached to disk as it is
computed, so an interruption costs one month rather than the whole run - the
first attempt at this lost twenty-five minutes to an OS update because it held
everything in memory and wrote nothing until the end. Re-running picks up from
whatever is already cached. The interpolation itself is deliberately left as it
is; this is about surviving interruption, not about going faster.

Two grid facts matter. SARAH is cell-centred (25.025, 25.075, ...) while the
target grid is node-centred (25.0, 25.1, ...), so NONE of the coordinates
coincide and this interpolates rather than block-averages.

And the order was placed without margin, so the target perimeter - 300 of 5,751
cells - lies 0.025 degrees outside the RANGE OF SARAH CELL CENTRES. This is not
missing data. The SARAH cell centred on 25.025 spans 25.00 to 25.05, so it
physically covers the target node at 25.0; linear interpolation simply will not
extrapolate the last half cell. Those nodes are therefore clamped to the nearest
interior value, which for a cell-centred field is not an approximation but the
correct answer: the node lies inside that cell. Half a degree of margin on the
order would have avoided the question.

One month is dropped rather than clamped. 1985-02 is 23.5 per cent NaN at native
resolution and is built from about 21 daily averages, so its gaps are real
retrieval gaps across the interior, not the edge artefact. Left in, it alone
dragged apparent whole-record coverage from 94.8 to 72.5 per cent through an
any-NaN test.

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
CACHE = os.path.join(PROC, "evaluation/_sarah_regrid_cache")
# 23.5% NaN at native resolution, ~21 daily averages behind it
DROP_MONTHS = {"1985-02"}


def clamp_edges(a):
    """Fill NaN from the nearest valid cell. See the note on the perimeter."""
    from scipy.ndimage import distance_transform_edt
    bad = np.isnan(a)
    if not bad.any():
        return a
    _, (iy, ix) = distance_transform_edt(bad, return_indices=True)
    return a[iy, ix]


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
    """One interpolated field per month, cached to disk as it goes."""
    os.makedirs(CACHE, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "data/raw/sarah/SISmm*.nc")))
    stamp = lambda f: pd.Timestamp(re.search(r"SISmm(\d{8})", f).group(1))
    times, fields = [], []
    cached = done = 0
    for n, f in enumerate(files, 1):
        ts = stamp(f)
        if ts.strftime("%Y-%m") in DROP_MONTHS:
            continue
        cp = os.path.join(CACHE, "%s.npy" % ts.strftime("%Y-%m"))
        if os.path.exists(cp):
            fields.append(np.load(cp)); cached += 1
        else:
            with xr.open_dataset(f) as d:
                v = d["SIS"].interp(lat=lat, lon=lon,
                                    method="linear").values.squeeze()
            np.save(cp, v.astype(np.float32))
            fields.append(v); done += 1
            if done % 20 == 0:
                print("    interpolated %d new (%d/%d files)" % (done, n, len(files)),
                      flush=True)
        times.append(ts)
    print("  SARAH: %d months (%d from cache, %d newly interpolated, %d dropped)"
          % (len(fields), cached, done, len(DROP_MONTHS)), flush=True)
    arr = np.array(fields)
    n_edge = int(np.isnan(arr[0]).sum())
    arr = np.array([clamp_edges(x) for x in arr])
    print("  clamped %d perimeter cells per month (half-cell, ~2.8 km)"
          % n_edge, flush=True)
    return xr.DataArray(arr,
                        coords={"time": pd.DatetimeIndex(times), "lat": lat, "lon": lon},
                        dims=("time", "lat", "lon"))


def main():
    era5 = era5_ghi_stack()
    lat, lon = era5.lat.values, era5.lon.values
    print("ERA5-derived stack:", era5.shape, flush=True)
    sar = sarah_stack(lat, lon)

    # month-end (ERA5) against month-start (SARAH): align on calendar month
    era5 = era5.assign_coords(ym=("time", pd.PeriodIndex(
        pd.DatetimeIndex(era5.time.values), freq="M").astype(str)))
    sar = sar.assign_coords(ym=("time", pd.PeriodIndex(
        pd.DatetimeIndex(sar.time.values), freq="M").astype(str)))
    common = sorted(set(era5.ym.values) & set(sar.ym.values))
    print("  months in common: %d (%s .. %s)" % (len(common), common[0], common[-1]), flush=True)
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
