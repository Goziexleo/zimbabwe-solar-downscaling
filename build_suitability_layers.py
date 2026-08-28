"""Assemble every Section 3.9 criterion layer on the 0.1 degree target grid.

Section 3.9 specifies seven criteria and four exclusion categories. This script
puts all of them on the 71 x 81 grid the downscaled product uses and writes one
NetCDF; compute_suitability.py then does the AHP weighting and the overlay.

Two deviations from the chapter, both deliberate and both recorded here rather
than discovered later:

  Water bodies. Section 3.9.2 specifies the ESA CCI water class AND the
  HydroSHEDS river network for riparian zones wider than 200 m. HydroSHEDS was
  never acquired, so only the ESA CCI water and flooded classes are used. At
  0.1 degrees a 200 m river is a sub-cell feature anyway - it would move a cell
  fraction, not a cell - so the practical loss is small, but the chapter
  currently promises something this does not do.

  Present-period irradiance. Section 3.9.1 asks for "the ensemble-mean
  downscaled GHI for the present period (1985 to 2024)". No such product
  exists: 1985-2010 is the training period, where the ERA5-derived target IS
  the field, and the projections start in 2026.

  TWO present layers are therefore written, not one, because the choice is a
  real decision rather than a technicality. ghi_present_era5 is the ERA5-derived
  climatology reconstructed as CSI x clearsky - the quantity every model was
  fitted to reproduce, and the only one commensurable with the projections, so
  present-to-future CHANGE must use it or it differences two measurement
  systems. ghi_present_sarah is the independent satellite retrieval, which is
  the better estimate of the resource that actually exists and so is the more
  defensible input to a siting decision about the present. compare_sarah_era5.py
  measures the gap; whether it matters to the ranking depends on whether the
  offset is spatially uniform, since Section 3.9.3 min-max normalises each
  criterion and a uniform offset largely cancels.

Distances are computed in an azimuthal equidistant projection centred on the
domain, which keeps them true to the metre over a few hundred km, rather than
in degrees.

    python build_suitability_layers.py
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data/raw")
PROC = os.path.join(ROOT, "data/processed")
OUT = os.path.join(PROC, "suitability/criterion_layers.nc")

AEQD = "+proj=aeqd +lat_0=-18.5 +lon_0=29 +datum=WGS84 +units=m +no_defs"

# Section 3.9.3's five categories, mapped onto the ESA CCI classes actually
# present. Class 60 is "closed to open (>15%)", the parent of 61 (closed, 0.14%
# of the domain) and 62 (open, 19.7%); in this domain it is overwhelmingly open,
# so it takes the open-woodland score. Water, flooded and urban classes get 0
# here and are excluded outright by the mask, so the score is never read.
LC_SCORE = {
    120: 1.0, 121: 1.0, 122: 1.0, 130: 1.0,          # shrubland, grassland
    60: 0.7, 62: 0.7, 70: 0.7, 80: 0.7, 90: 0.7,     # open woodland
    10: 0.5, 11: 0.5, 12: 0.5, 20: 0.5, 30: 0.5,     # rain-fed cropland
    50: 0.2, 61: 0.2,                                # closed woodland
    40: 0.3, 100: 0.3, 110: 0.3, 140: 0.3,           # other vegetated
    150: 0.3, 151: 0.3, 152: 0.3, 153: 0.3,
    200: 0.3, 201: 0.3, 202: 0.3,                    # bare
}
LC_URBAN = {190}
LC_WATER = {160, 170, 180, 210}
SLOPE_EXCLUDE_DEG = 15.0
URBAN_BUFFER_KM = 1.0


def target_grid():
    d = xr.open_dataset(os.path.join(PROC, "ml_ready/ml_validation_dataset.nc"))
    return d.fine_lat.values, d.fine_lon.values


def present_ghi(lat, lon):
    """ERA5-derived 0.1 deg GHI climatology over 1985-2024, as CSI x clearsky."""
    cs = xr.open_dataset(os.path.join(
        PROC, "era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc"))
    tot = np.zeros((len(lat), len(lon)))
    n = 0
    for f in ("ml_training_dataset.nc", "ml_validation_dataset.nc"):
        d = xr.open_dataset(os.path.join(PROC, "ml_ready", f))
        months = pd.DatetimeIndex(d.time.values).month
        csi = d["clear_sky_index"].values
        for i, m in enumerate(months):
            tot += csi[i] * cs.clearsky_ghi.isel(month=m - 1).values
            n += 1
    print("  present GHI from %d monthly fields (1985-2024)" % n)
    return tot / n


def sarah_present(lat, lon):
    """SARAH SIS climatology over 1985-2024, on the target grid.

    Returns NaN on the perimeter, where the order's zero margin leaves the
    target 0.025 degrees outside SARAH's cell-centre hull. Left as NaN rather
    than extrapolated so the suitability step has to decide what to do about it.
    """
    # compare_sarah_era5.py already interpolates the whole record onto this
    # grid and saves the climatology as sarah_mean. Reuse it rather than
    # repeating 480 interpolations for the same answer.
    cmp_nc = os.path.join(PROC, "evaluation/sarah_era5_comparison.nc")
    if os.path.exists(cmp_nc):
        d = xr.open_dataset(cmp_nc)
        if np.allclose(d.lat.values, lat) and np.allclose(d.lon.values, lon):
            print("  SARAH present reused from sarah_era5_comparison.nc")
            return d.sarah_mean.values
    files = sorted(glob.glob(os.path.join(RAW, "sarah/SISmm*.nc")))
    if not files:
        print("  no SARAH files - skipping the SARAH present layer")
        return None
    tot = None
    for f in files:
        d = xr.open_dataset(f)[["SIS"]].interp(lat=lat, lon=lon, method="linear")
        v = d.SIS.values.squeeze()
        tot = v if tot is None else tot + v
    print("  SARAH present from %d monthly fields" % len(files))
    return tot / len(files)


def slope_layers(lat, lon):
    """Slope and the >15 deg fraction, interpolated onto the target grid.

    The topography file sits on a 0.1 deg grid offset by half a cell from the
    target (-22.75 against -22.0), so this interpolates rather than selects.
    """
    t = xr.open_dataset(os.path.join(
        PROC, "topography/zimbabwe_topographic_features_0.1deg.nc"))
    i = t.interp(lat=lat, lon=lon, method="linear")
    return i.slope.values, i.slope_frac_gt15.values


def _cell_edges(centres):
    step = float(np.abs(np.diff(centres)).mean())
    return centres - step / 2.0, centres + step / 2.0, step


def landcover_layers(lat, lon):
    """Mean suitability score, urban fraction and water fraction per cell."""
    f = glob.glob(os.path.join(RAW, "landcover/*.nc"))[0]
    d = xr.open_dataset(f)
    v = d.lccs_class.values.squeeze().astype(np.int16)
    lc_lat, lc_lon = d.lat.values, d.lon.values

    lut = np.zeros(256, dtype=np.float32)
    for k, s in LC_SCORE.items():
        lut[k] = s
    score_px = lut[np.clip(v, 0, 255)]
    water_px = np.isin(v, list(LC_WATER)).astype(np.float32)

    # Section 3.9.2 excludes urban pixels "plus a 1 km buffer". The buffer is
    # applied here at the land-cover resolution, where it is meaningful: 300 m
    # pixels, so 1 km is a radius of about 3.3 pixels. Doing it after
    # aggregation to 0.1 degrees would be meaningless, since one cell is 11 km.
    from scipy.ndimage import binary_dilation
    urban_raw = np.isin(v, list(LC_URBAN))
    px_m = 300.0
    r = int(np.ceil(URBAN_BUFFER_KM * 1000.0 / px_m))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2) <= r ** 2
    urban_buf = binary_dilation(urban_raw, structure=disk)
    print("   urban %0.3f%% of pixels, %0.3f%% after the %.0f km buffer"
          % (100 * urban_raw.mean(), 100 * urban_buf.mean(), URBAN_BUFFER_KM))
    urban_px = urban_buf.astype(np.float32)

    lat0, lat1, _ = _cell_edges(lat)
    lon0, lon1, _ = _cell_edges(lon)
    # index of each source pixel in the target grid
    iy = np.searchsorted(np.sort(lat0), lc_lat, side="right") - 1
    ix = np.searchsorted(lon0, lc_lon, side="right") - 1
    order = np.argsort(lat0)
    score = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
    urban = np.zeros_like(score); water = np.zeros_like(score)
    for a in range(len(lat)):
        ra = order[a]
        rows = (lc_lat >= lat0[ra]) & (lc_lat < lat1[ra])
        if not rows.any():
            continue
        for b in range(len(lon)):
            cols = (lc_lon >= lon0[b]) & (lc_lon < lon1[b])
            if not cols.any():
                continue
            blk = np.ix_(rows, cols)
            valid = ~(urban_px[blk].astype(bool) | water_px[blk].astype(bool))
            score[ra, b] = score_px[blk][valid].mean() if valid.any() else 0.0
            urban[ra, b] = urban_px[blk].mean()
            water[ra, b] = water_px[blk].mean()
    return score, urban, water


def population(lat, lon):
    """WorldPop head count summed into each 0.1 deg cell."""
    import rasterio
    from rasterio.warp import Resampling
    p = os.path.join(RAW, "worldpop/zwe_ppp_2020_constrained.tif")
    with rasterio.open(p) as r:
        arr = r.read(1).astype(np.float64)
        arr[arr == r.nodata] = 0.0
        b = r.bounds
        py = np.linspace(b.top, b.bottom, r.height, endpoint=False)
        px = np.linspace(b.left, b.right, r.width, endpoint=False)
    lat0, lat1, _ = _cell_edges(lat)
    lon0, lon1, _ = _cell_edges(lon)
    iy = np.clip(np.searchsorted(lat0, py, side="right") - 1, -1, len(lat) - 1)
    ix = np.clip(np.searchsorted(lon0, px, side="right") - 1, -1, len(lon) - 1)
    out = np.zeros((len(lat), len(lon)))
    good_y = iy >= 0
    for j, row in enumerate(arr):
        if not good_y[j]:
            continue
        np.add.at(out[iy[j]], ix[ix >= 0], row[ix >= 0])
    return out


def _to_xy(gdf):
    return gdf.to_crs(AEQD)


def _densified_points(geoms, step_m=200.0):
    """Vertices along each line/boundary at <= step_m spacing, for a KD-tree."""
    pts = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        parts = g.geoms if hasattr(g, "geoms") else [g]
        for part in parts:
            line = part.exterior if part.geom_type == "Polygon" else part
            if line is None:
                continue
            n = max(int(line.length / step_m) + 1, 2)
            for t in np.linspace(0, line.length, n):
                p = line.interpolate(t)
                pts.append((p.x, p.y))
    return np.asarray(pts) if pts else np.empty((0, 2))


def distance_km(points_xy, gx, gy):
    if len(points_xy) == 0:
        return np.full(gx.shape, np.nan)
    tree = cKDTree(points_xy)
    d, _ = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
    return (d / 1000.0).reshape(gx.shape)


def main():
    import geopandas as gpd
    from shapely.geometry import box

    lat, lon = target_grid()
    print("target grid %d x %d, lat %.1f..%.1f lon %.1f..%.1f"
          % (len(lat), len(lon), lat.min(), lat.max(), lon.min(), lon.max()))

    LON, LAT = np.meshgrid(lon, lat)
    centres = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs="EPSG:4326")
    cxy = _to_xy(centres)
    gx = np.asarray(cxy.geometry.x).reshape(LAT.shape)
    gy = np.asarray(cxy.geometry.y).reshape(LAT.shape)

    ds = xr.Dataset(coords={"lat": lat, "lon": lon})

    print("GHI ...")
    ds["ghi_present_era5"] = (("lat", "lon"), present_ghi(lat, lon))
    sar = sarah_present(lat, lon)
    if sar is not None:
        ds["ghi_present_sarah"] = (("lat", "lon"), sar)
    for f in sorted(glob.glob(os.path.join(PROC, "mme_aggregations_xgb/*.nc"))):
        tag = os.path.basename(f).replace("mme_suitability_", "").replace(".nc", "")
        ds["ghi_" + tag] = (("lat", "lon"), xr.open_dataset(f).ghi_mean.values)
    print("   %d future slices" % (len(ds.data_vars) - 1))

    print("slope ...")
    sl, slf = slope_layers(lat, lon)
    ds["slope"] = (("lat", "lon"), sl)
    ds["slope_frac_gt15"] = (("lat", "lon"), slf)

    print("land cover ...")
    lc, urb, wat = landcover_layers(lat, lon)
    ds["landcover_score"] = (("lat", "lon"), lc)
    ds["urban_fraction"] = (("lat", "lon"), urb)
    ds["water_fraction"] = (("lat", "lon"), wat)

    print("population ...")
    ds["population"] = (("lat", "lon"), population(lat, lon))

    print("OSM distances ...")
    import pyogrio
    pbf = os.path.join(RAW, "osm/zimbabwe-latest.osm.pbf")
    lines = pyogrio.read_dataframe(pbf, layer="lines",
                                   columns=["highway", "other_tags"])
    roads = lines[lines.highway.isin(
        ["motorway", "trunk", "primary", "secondary", "tertiary"])]
    grid = lines[lines.other_tags.fillna("").str.contains('"power"=>"line"')]
    pts = pyogrio.read_dataframe(pbf, layer="points", columns=["place"])
    settle = pts[pts.place.isin(["city", "town", "village"])]
    print("   roads %d | power lines %d | settlements %d"
          % (len(roads), len(grid), len(settle)))

    for name, gdf in (("roads", roads), ("grid", grid)):
        xy = _densified_points(_to_xy(gdf).geometry.values)
        ds["dist_" + name] = (("lat", "lon"), distance_km(xy, gx, gy))
        print("   dist_%s: %.1f..%.1f km" % (name, np.nanmin(ds["dist_"+name]),
                                             np.nanmax(ds["dist_"+name])))
    sxy = _to_xy(settle)
    ds["dist_settlements"] = (("lat", "lon"), distance_km(
        np.column_stack([sxy.geometry.x, sxy.geometry.y]), gx, gy))

    print("protected areas and national boundary ...")
    wdpa = gpd.read_file(os.path.join(RAW, "protected areas/WDPA_ZWE_polygons.gpkg"))
    gadm = gpd.read_file(os.path.join(RAW, "gadm/gadm41_ZWE.gpkg"), layer="ADM_ADM_0")
    wdpa_u = wdpa.to_crs("EPSG:4326").geometry.union_all()
    zw = gadm.to_crs("EPSG:4326").geometry.union_all()
    lat0, lat1, _ = _cell_edges(lat)
    lon0, lon1, _ = _cell_edges(lon)
    pa = np.zeros((len(lat), len(lon)))
    inzw = np.zeros_like(pa)
    for a in range(len(lat)):
        for b in range(len(lon)):
            cell = box(lon0[b], lat0[a], lon1[b], lat1[a])
            area = cell.area
            pa[a, b] = cell.intersection(wdpa_u).area / area
            inzw[a, b] = cell.intersection(zw).area / area
    ds["protected_fraction"] = (("lat", "lon"), pa)
    ds["in_zimbabwe"] = (("lat", "lon"), inzw)

    ds.attrs["note"] = ("Section 3.9 criterion layers. Water uses ESA CCI only; "
                        "HydroSHEDS was not acquired. Present GHI is the "
                        "ERA5-derived 1985-2024 climatology.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds.to_netcdf(OUT)
    print("\nSaved %s\n  variables: %s" % (OUT, list(ds.data_vars)))


if __name__ == "__main__":
    main()
