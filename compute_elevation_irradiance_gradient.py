"""Measure how observed irradiance varies with elevation inside Zimbabwe.

Chapter 2 asserted twice that the lowveld carries the country's highest annual
GHI and is therefore the remote ground a planner is pulled towards. Neither
assertion is sourced, and both are checkable against the same observed fields
Chapter 4 validates against, so they are checked here rather than hedged.

The answer is the opposite of the claim. Annual-mean GHI rises weakly with
elevation in both the CM SAF SARAH retrieval and ERA5, and the lowveld carries
the lowest mean of the three physiographic zones. That is consistent with
Section 4.8.1, where the robust suitable set falls along the central watershed
rather than in the lowveld, and with Section 5.2.4, where irradiance proves too
narrowly spread to separate candidate sites at all.

Bands follow the thesis's own definitions in Chapter 2: lowveld below 600 m,
highveld above 1,200 m, middleveld between.

    python compute_elevation_irradiance_gradient.py
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
CMP = os.path.join(EVAL, "sarah_era5_comparison.nc")
TOPO = os.path.join(ROOT, "data/processed/topography",
                    "zimbabwe_topographic_features_0.1deg.nc")
OUT = os.path.join(EVAL, "elevation_irradiance_gradient.csv")

LOWVELD_M = 600.0
HIGHVELD_M = 1200.0


def main():
    cmp = xr.open_dataset(CMP)
    topo = xr.open_dataset(TOPO)

    # The topography box is wider than the analysis grid and offset by half a
    # cell, so it is interpolated onto the irradiance grid rather than indexed.
    elev = topo["elevation"].interp(lat=cmp.lat, lon=cmp.lon, method="linear").values
    fields = {"SARAH": cmp["sarah_mean"].values, "ERA5": cmp["era5_mean"].values}
    mask = zimbabwe_mask(shape=elev.shape)

    bands = [("lowveld", elev < LOWVELD_M),
             ("middleveld", (elev >= LOWVELD_M) & (elev <= HIGHVELD_M)),
             ("highveld", elev > HIGHVELD_M)]

    rows = []
    for product, fld in fields.items():
        valid = mask & np.isfinite(fld) & np.isfinite(elev)
        r = float(np.corrcoef(elev[valid], fld[valid])[0, 1])
        for name, band in bands:
            sel = valid & band
            rows.append(dict(product=product, band=name, cells=int(sel.sum()),
                             ghi_mean=float(fld[sel].mean()),
                             ghi_sd=float(fld[sel].std()),
                             elev_mean=float(elev[sel].mean()),
                             corr_elev_ghi_national=r))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("Observed annual-mean GHI by physiographic band, inside Zimbabwe\n")
    print("%-8s %-11s %6s %9s %9s" % ("product", "band", "cells", "GHI", "elev"))
    for _, w in df.iterrows():
        print("%-8s %-11s %6d %9.1f %9.0f"
              % (w["product"], w["band"], w["cells"], w["ghi_mean"], w["elev_mean"]))

    print()
    for product in fields:
        s = df[df["product"] == product]
        lo = float(s[s.band == "lowveld"].ghi_mean.iloc[0])
        hi = float(s[s.band == "highveld"].ghi_mean.iloc[0])
        r = float(s.corr_elev_ghi_national.iloc[0])
        print("%-6s corr(elevation, GHI) = %+.3f; lowveld %.1f vs highveld %.1f W/m2"
              % (product, r, lo, hi))
        # The claim under test: the lowveld is NOT the highest-irradiance band.
        assert lo < hi, ("%s: lowveld mean exceeds highveld mean, so the Chapter 2 "
                         "claim would stand and the correction must be revisited" % product)
        assert r > 0, "%s: elevation-GHI correlation is not positive" % product

    span = df.groupby("product").ghi_mean.agg(["min", "max"])
    for product, w in span.iterrows():
        print("%-6s band means span %.1f to %.1f W/m2, a range of %.1f per cent"
              % (product, w["min"], w["max"], 100.0 * (w["max"] - w["min"]) / w["min"]))

    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
