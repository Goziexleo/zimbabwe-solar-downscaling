"""Where the robust set actually is: provinces, true area, and the top cell.

Chapter 4 carried a hardcoded province breakdown summing to 41 cells, left over
from a superseded reading of the grid-distance decay parameter that put the
robust set at 42 rather than 145. It also placed the highest-scoring cell on the
"Kadoma-Chegutu section" of the corridor, and reported the set's area as
17,545 km2 by assuming a 121 km2 cell.

Everything here is computed instead:
  - each robust cell assigned to a GADM level-1 province by point-in-polygon
  - area from the true size of each 0.1 degree cell at its own latitude, not a
    single nominal figure
  - the top-scoring cell located, with its nearest sizeable town named from
    GADM level-2 rather than asserted

Writes robust_set_geography.csv and prints the sentence Chapter 4 should carry.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

SUIT = os.path.abspath("./data/processed/suitability/suitability_index.nc")
GADM = os.path.abspath("./data/raw/gadm/gadm41_ZWE.gpkg")
OUT_DIR = os.path.abspath("./data/processed/evaluation")

R_EARTH = 6371.0088          # km, mean radius
CELL_DEG = 0.1


def cell_area_km2(lat_deg, d=CELL_DEG):
    """True area of a d x d degree cell centred at lat_deg.

    A single nominal area overstates the total: at 19 S a 0.1 degree cell is
    about 117 km2, not the 121 km2 that a cos-free calculation gives.
    """
    dlat = np.deg2rad(d)
    lat1, lat2 = np.deg2rad(lat_deg - d / 2), np.deg2rad(lat_deg + d / 2)
    return (R_EARTH ** 2) * np.deg2rad(d) * (np.sin(lat2) - np.sin(lat1)) * (dlat / dlat)


def main():
    ds = xr.open_dataset(SUIT)
    rob = ds["robustly_suitable"].values.astype(bool)
    si = ds["si_primary"].values
    lat, lon = ds["lat"].values, ds["lon"].values
    LON, LAT = np.meshgrid(lon, lat)

    n = int(rob.sum())
    print("robust cells: %d" % n)

    prov = gpd.read_file(GADM, layer="ADM_ADM_1")[["NAME_1", "geometry"]]
    dist = gpd.read_file(GADM, layer="ADM_ADM_2")[["NAME_2", "NAME_1", "geometry"]]

    jj, ii = np.where(rob)
    pts = gpd.GeoDataFrame(
        {"lat": LAT[jj, ii], "lon": LON[jj, ii], "si": si[jj, ii]},
        geometry=[Point(x, y) for x, y in zip(LON[jj, ii], LAT[jj, ii])],
        crs=prov.crs,
    )
    pts["area_km2"] = cell_area_km2(pts["lat"].values)

    joined = gpd.sjoin(pts, prov, how="left", predicate="within")
    # A cell centre can fall just outside every polygon on the coastline-free
    # borders GADM draws; attach those to the nearest province rather than
    # dropping them, so the counts always sum to the robust total.
    miss = joined["NAME_1"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss.values], prov, how="left")
        joined.loc[miss, "NAME_1"] = near["NAME_1"].values
        print("  %d cell(s) assigned to the nearest province" % int(miss.sum()))

    counts = (joined.groupby("NAME_1")
              .agg(cells=("si", "size"), area_km2=("area_km2", "sum"),
                   mean_si=("si", "mean"))
              .sort_values("cells", ascending=False).reset_index())
    assert int(counts["cells"].sum()) == n, "province counts must sum to the robust set"

    counts.to_csv(os.path.join(OUT_DIR, "robust_set_geography.csv"), index=False)
    print("\n%-22s %6s %12s %9s" % ("province", "cells", "area km2", "mean SI"))
    for r in counts.itertuples():
        print("%-22s %6d %12.0f %9.4f" % (r.NAME_1, r.cells, r.area_km2, r.mean_si))
    print("%-22s %6d %12.0f" % ("TOTAL", counts["cells"].sum(), counts["area_km2"].sum()))

    # The top-scoring cell, over the retained set rather than the robust subset,
    # which is what Chapter 4 quotes.
    keep = ~np.isnan(si)
    k = np.nanargmax(np.where(keep, si, np.nan))
    ty, tx = np.unravel_index(k, si.shape)
    tlat, tlon = float(LAT[ty, tx]), float(LON[ty, tx])
    # Containment first; only fall back to distance, and then in a projected
    # CRS, because distances in degrees are not distances.
    pt = gpd.GeoDataFrame(geometry=[Point(tlon, tlat)], crs=dist.crs)
    hit = gpd.sjoin(pt, dist, how="left", predicate="within")
    if hit["NAME_2"].notna().any():
        name2, name1 = hit.iloc[0]["NAME_2"], hit.iloc[0]["NAME_1"]
        how = "within"
    else:
        m = 32736  # UTM zone 36S, covers Zimbabwe
        d_m = dist.to_crs(m)
        k2 = d_m.distance(pt.to_crs(m).geometry.iloc[0]).idxmin()
        name2, name1, how = d_m.loc[k2, "NAME_2"], d_m.loc[k2, "NAME_1"], "nearest"
    print("\ntop cell: SI %.3f at %.2f E, %.2f S -> %s district, %s (%s)"
          % (si[ty, tx], tlon, abs(tlat), name2, name1, how))

    lons = LON[rob]
    print("\n--- sentence for Chapter 4 ---")
    print("The robust set spans %.1f to %.1f degrees east. By province it falls in %s."
          % (lons.min(), lons.max(),
             ", ".join("%s (%d cell%s)" % (r.NAME_1, r.cells, "" if r.cells == 1 else "s")
                       for r in counts.itertuples())))
    print("It covers {:,.0f} km2, and the highest-scoring cell lies in {} district."
          .format(counts["area_km2"].sum(), name2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
