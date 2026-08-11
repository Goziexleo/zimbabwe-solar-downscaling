"""Generate 0.1 deg topographic covariates from the SRTM 90 m DEM (Section 3.5.2).

Replaces the previous implementation, which had three defects:

1.  SVF was 100% NaN over the entire domain. `approximate_svf` divided by
    `np.max(dem) - np.min(dem)`, and because the DEM was opened with
    `masked=True` the nodata cells became NaN, so both `np.max` and `np.min`
    returned NaN. Every output cell was therefore NaN, and every downstream
    script applies `safe_nan_to_num(..., "generate_topographic_features:...")`, so SVF entered all four
    models as an all-zero constant channel. It contributed nothing.

2.  The SVF formula was a 5x5 local elevation-contrast proxy, not the
    "radial sampling algorithm with 36 azimuth directions" specified in
    Section 3.5.2. This module implements the specified horizon-angle
    algorithm and the Dozier & Frew (1990) sky-view integral.

3.  Aggregation to 0.1 deg used `.interp(method="linear")`, which samples the
    90 m field at the target cell centres. Section 3.5.2 specifies a mean over
    the cell. Point sampling discards ~14,000 of the ~14,400 90 m cells inside
    each 0.1 deg cell and badly under-represents slope in complex terrain.
    This module computes a true areal mean by block aggregation.

Also emits `slope_frac_gt15`, the fraction of 90 m cells within each 0.1 deg
cell that exceed 15 degrees. Section 3.9.2 excludes terrain steeper than 15
degrees; applying that threshold to an 11 km *mean* slope is not the same
test, and the sub-grid fraction is the defensible quantity for the MCE mask.

Output grid, dimension names, and variable names are unchanged, so this is a
drop-in replacement for the downstream dataset builders and training scripts.

Usage:
    python generate_topographic_features.py [--search-radius-m 100]

NOTE ON SEARCH RADIUS: Section 3.5.2 states a 100 m maximum search radius.
At 90 m pixel spacing that is a single cell, which makes the 36-azimuth
horizon search very nearly degenerate and produces an SVF field that is
~1.0 almost everywhere. A radius of 1000-5000 m is the physically meaningful
range for terrain shading and is what the Eastern Highlands discussion in
Chapter 4 would need. The default is left at the chapter-stated 100 m so the
code matches the written methodology; override deliberately and amend the
chapter text if you change it.

References:
    Horn, B.K.P. (1981) Hill shading and the reflectance map.
    Dozier, J. and Frew, J. (1990) Rapid calculation of terrain parameters
        for radiation modeling from digital elevation data.
    Zaksek, K., Ostir, K. and Kokalj, Z. (2011) Sky-view factor as a
        relief visualization technique.
"""

import argparse
import os

import numpy as np
import rasterio
import xarray as xr
from ml_dataset_common import safe_nan_to_num

# --- Configuration ---
DEM_PATH = os.path.abspath("./data/raw/srtm/zimbabwe_dem_90m.tif")
OUTPUT_DIR = os.path.abspath("./data/processed/topography")
OUTPUT_NC = os.path.join(OUTPUT_DIR, "zimbabwe_topographic_features_0.1deg.nc")

# Zimbabwe study domain with the 0.25 deg buffer of Section 3.2.1. This is the
# buffered grid, deliberately wider than the canonical fine output grid in
# ml_dataset_common.py; the buffer exists to damp edge effects during
# interpolation and convolution.
LAT_MIN, LAT_MAX = -22.75, -14.75
LON_MIN, LON_MAX = 24.75, 33.85
TARGET_RES = 0.1

N_AZIMUTHS = 36  # Section 3.5.2
SLOPE_EXCLUSION_DEG = 15.0  # Section 3.9.2

EARTH_R = 6371000.0
# DEM rows processed per pass. Kept small because the horizon search holds
# several full-block float32 buffers alive at once; 256 rows x ~10,800 columns
# is ~11 MB per buffer, which stays comfortable on a 4 GB machine.
ROW_BLOCK = 256


def metric_cell_sizes(lat_deg, res_deg):
    """Metres per pixel in the x and y directions at a given latitude.

    The DEM is on a geographic grid, so the east-west spacing shrinks with
    the cosine of latitude while the north-south spacing is constant.
    """
    dy = np.deg2rad(res_deg) * EARTH_R
    dx = np.deg2rad(res_deg) * EARTH_R * np.cos(np.deg2rad(lat_deg))
    return dx, dy


def horn_slope(block, dx_col, dy):
    """Slope in degrees via Horn's (1981) 3x3 operator.

    `block` is (rows, cols) with a one-row halo top and bottom already
    attached. `dx_col` is a (rows, 1) array of east-west pixel sizes in
    metres, one per row, so the latitude dependence of the geographic grid
    is handled exactly rather than assumed square.

    Returns an array matching `block` minus the halo rows.
    """
    z = block
    # Neighbour stencil, edge-replicated on the left and right margins.
    zl = np.concatenate([z[:, :1], z[:, :-1]], axis=1)
    zr = np.concatenate([z[:, 1:], z[:, -1:]], axis=1)

    a, b, c = zl[:-2], z[:-2], zr[:-2]      # north row
    d, f = zl[1:-1], zr[1:-1]               # centre row
    g, h, i = zl[2:], z[2:], zr[2:]         # south row

    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * dx_col)
    dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * dy)

    return np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))


def sky_view_factor(block, dx_col, dy, radius_cells, n_az=N_AZIMUTHS):
    """Sky-view factor by 36-azimuth horizon search (Section 3.5.2).

    For each azimuth the horizon angle is the maximum elevation angle to any
    sampled cell along that bearing, floored at zero (a cell can see at least
    the hemisphere above a level horizon). The sky-view factor is then the
    Dozier & Frew (1990) integral, which evaluates to 1.0 on flat ground:

        SVF = (1 / N) * sum_k cos^2(H_k)

    `block` carries a `radius_cells` halo top and bottom. Returns an array
    matching `block` minus the halo.
    """
    halo = radius_cells
    nrows_out = block.shape[0] - 2 * halo
    ncols = block.shape[1]
    centre = block[halo:halo + nrows_out, :]

    # Preallocated once and reused across all azimuths and radii. Allocating
    # inside the loop is what made the 1000 m run exhaust memory.
    acc = np.zeros((nrows_out, ncols), dtype=np.float32)
    max_tan = np.empty((nrows_out, ncols), dtype=np.float32)
    buf = np.empty((nrows_out, ncols), dtype=np.float32)

    for k in range(n_az):
        phi = 2.0 * np.pi * k / n_az
        # Screen convention: rows increase southward, columns increase eastward.
        u_col = np.sin(phi)
        u_row = -np.cos(phi)

        max_tan.fill(0.0)

        for r in range(1, radius_cells + 1):
            di = int(round(u_row * r))
            dj = int(round(u_col * r))
            if di == 0 and dj == 0:
                continue

            # Shift the padded block by (di, dj), replicating at the column
            # edges, writing straight into the reusable buffer.
            src_rows = block[halo + di:halo + di + nrows_out, :]
            if dj > 0:
                buf[:, :ncols - dj] = src_rows[:, dj:]
                buf[:, ncols - dj:] = src_rows[:, -1:]
            elif dj < 0:
                buf[:, -dj:] = src_rows[:, :dj]
                buf[:, :-dj] = src_rows[:, :1]
            else:
                buf[:] = src_rows

            dist = np.sqrt((dj * dx_col) ** 2 + (di * dy) ** 2)
            with np.errstate(invalid="ignore"):
                buf -= centre
                buf /= dist
            np.nan_to_num(buf, copy=False, nan=0.0)
            np.maximum(max_tan, buf, out=max_tan)

        np.clip(max_tan, 0.0, None, out=max_tan)
        # cos^2(arctan(t)) == 1 / (1 + t^2)
        max_tan *= max_tan
        max_tan += 1.0
        acc += 1.0 / max_tan

    svf = acc / n_az
    # Restore NaN where the source cell itself is void so the mask propagates.
    svf[np.isnan(centre)] = np.nan
    return svf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-radius-m", type=float, default=100.0,
                        help="Maximum horizon search radius in metres "
                             "(Section 3.5.2 states 100 m; 1000-5000 m is the "
                             "physically meaningful range)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DEM_PATH):
        raise SystemExit(f"DEM not found at {DEM_PATH}")

    # --- Target grid (cell centres) ---
    target_lat = np.round(np.arange(LAT_MIN, LAT_MAX + TARGET_RES / 2, TARGET_RES), 4)
    target_lon = np.round(np.arange(LON_MIN, LON_MAX + TARGET_RES / 2, TARGET_RES), 4)
    n_tlat, n_tlon = target_lat.size, target_lon.size

    # Accumulators for the areal mean. Sums and counts rather than a running
    # mean so that void cells simply reduce the denominator.
    fields = ["elevation", "slope", "svf"]
    sums = {f: np.zeros((n_tlat, n_tlon), dtype=np.float64) for f in fields}
    counts = {f: np.zeros((n_tlat, n_tlon), dtype=np.int64) for f in fields}
    steep_sum = np.zeros((n_tlat, n_tlon), dtype=np.float64)
    steep_count = np.zeros((n_tlat, n_tlon), dtype=np.int64)

    with rasterio.open(DEM_PATH) as src:
        nrows, ncols = src.shape
        res_y, res_x = abs(src.res[1]), abs(src.res[0])
        top, left = src.bounds.top, src.bounds.left
        nodata = src.nodata

        mean_lat = 0.5 * (LAT_MIN + LAT_MAX)
        dx_mean, dy = metric_cell_sizes(mean_lat, res_x)
        radius_cells = max(1, int(round(args.search_radius_m / dx_mean)))

        print(f"DEM {nrows} x {ncols} at {res_x:.6f} deg (~{dx_mean:.0f} m EW, "
              f"{dy:.0f} m NS)")
        print(f"SVF: {N_AZIMUTHS} azimuths, search radius "
              f"{args.search_radius_m:.0f} m = {radius_cells} cell(s)")
        if radius_cells <= 2:
            print("  WARNING: the search radius spans <= 2 pixels, so the "
                  "horizon search is close to degenerate and SVF will be "
                  "~1.0 across most of the domain. See the module docstring.")

        # Column geometry is fixed, so precompute the target-column index once.
        dem_lon = left + (np.arange(ncols) + 0.5) * res_x
        col_bin = np.floor((dem_lon - (target_lon[0] - TARGET_RES / 2)) / TARGET_RES).astype(np.int64)
        col_valid = (col_bin >= 0) & (col_bin < n_tlon)

        halo = max(radius_cells, 1)

        for start in range(0, nrows, ROW_BLOCK):
            stop = min(start + ROW_BLOCK, nrows)
            r0 = max(0, start - halo)
            r1 = min(nrows, stop + halo)

            raw = src.read(1, window=rasterio.windows.Window(0, r0, ncols, r1 - r0))
            block = raw.astype(np.float32)
            if nodata is not None:
                block[raw == nodata] = np.nan
            del raw

            # Pad to a full halo when the block sits at the raster edge.
            pad_top = halo - (start - r0)
            pad_bot = halo - (r1 - stop)
            if pad_top > 0 or pad_bot > 0:
                block = np.pad(block, ((max(pad_top, 0), max(pad_bot, 0)), (0, 0)),
                               mode="edge")

            dem_lat = top - (np.arange(start, stop) + 0.5) * res_y
            dx_col = (np.deg2rad(res_x) * EARTH_R
                      * np.cos(np.deg2rad(dem_lat)))[:, None].astype(np.float32)

            # Horn slope needs only a one-row halo, so slice the padded block.
            slope_blk = horn_slope(block[halo - 1:block.shape[0] - halo + 1, :],
                                   dx_col, np.float32(dy))
            svf_blk = sky_view_factor(block, dx_col, np.float32(dy), radius_cells)
            elev_blk = block[halo:block.shape[0] - halo, :]

            row_bin = np.floor(
                ((target_lat[0] - TARGET_RES / 2) - dem_lat) / -TARGET_RES
            ).astype(np.int64)
            row_valid = (row_bin >= 0) & (row_bin < n_tlat)

            rv = np.where(row_valid)[0]
            cv = np.where(col_valid)[0]
            if rv.size == 0 or cv.size == 0:
                continue

            flat_bin = (row_bin[rv][:, None] * n_tlon + col_bin[cv][None, :]).ravel()
            minlen = n_tlat * n_tlon

            for name, arr in (("elevation", elev_blk), ("slope", slope_blk),
                              ("svf", svf_blk)):
                sub = arr[np.ix_(rv, cv)].ravel()
                ok = ~np.isnan(sub)
                sums[name] += np.bincount(flat_bin[ok], weights=sub[ok],
                                          minlength=minlen).reshape(n_tlat, n_tlon)
                counts[name] += np.bincount(flat_bin[ok],
                                            minlength=minlen).reshape(n_tlat, n_tlon)

            sub = slope_blk[np.ix_(rv, cv)].ravel()
            ok = ~np.isnan(sub)
            steep_sum += np.bincount(
                flat_bin[ok], weights=(sub[ok] > SLOPE_EXCLUSION_DEG).astype(np.float64),
                minlength=minlen).reshape(n_tlat, n_tlon)
            steep_count += np.bincount(flat_bin[ok],
                                       minlength=minlen).reshape(n_tlat, n_tlon)

            print(f"  rows {start}-{stop} of {nrows}", flush=True)

    out = {}
    for name in fields:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(counts[name] > 0, sums[name] / counts[name], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_steep = np.where(steep_count > 0, steep_sum / steep_count, np.nan)

    ds = xr.Dataset(
        {
            "elevation": (("lat", "lon"), out["elevation"]),
            "slope": (("lat", "lon"), out["slope"]),
            "svf": (("lat", "lon"), out["svf"]),
            "slope_frac_gt15": (("lat", "lon"), frac_steep),
        },
        coords={"lat": target_lat, "lon": target_lon},
    )

    ds["elevation"].attrs = {"units": "m", "long_name": "Areal-mean elevation",
                             "source": "SRTM 90 m void-filled (CGIAR-CSI)"}
    ds["slope"].attrs = {"units": "degree", "long_name": "Areal-mean terrain slope",
                         "method": "Horn (1981) 3x3 operator at 90 m, cell mean to 0.1 deg"}
    ds["svf"].attrs = {"units": "1", "long_name": "Sky-view factor",
                       "method": f"{N_AZIMUTHS}-azimuth horizon search, Dozier & Frew (1990) integral",
                       "search_radius_m": args.search_radius_m}
    ds["slope_frac_gt15"].attrs = {
        "units": "1",
        "long_name": "Fraction of 90 m cells with slope above 15 degrees",
        "note": "Sub-grid steepness for the Section 3.9.2 exclusion mask"}
    ds.attrs = {"title": "Zimbabwe topographic covariates at 0.1 deg",
                "convention": "cell-centre coordinates, areal-mean aggregation"}

    ds.to_netcdf(OUTPUT_NC)
    print(f"\nWritten: {OUTPUT_NC}")
    for name in list(fields) + ["slope_frac_gt15"]:
        a = ds[name].values
        print(f"  {name:16s} nan {np.isnan(a).mean():.4f}  "
              f"min {np.nanmin(a):.4f}  max {np.nanmax(a):.4f}  "
              f"mean {np.nanmean(a):.4f}")


if __name__ == "__main__":
    main()
