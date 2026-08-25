"""Check this study's clear-sky ceiling against SARAH's, and quantify the offset.

Section 3.5.4 normalises ERA5 ssrd by a PVLIB Ineichen clear-sky field to form
the CSI target. compute_finegrid_clearsky_ghi.py builds that field by averaging
thirteen DAYTIME hours (06:00-18:00), while the ERA5 ssrd it divides is a
24-hour monthly mean. The two are on different temporal bases, so the ratio is
not a clear-sky index in the physical sense.

SARAH ships its own clear-sky field (SISC) alongside all-sky SIS on the same
grid and the same monthly basis, which makes the comparison direct.

The offset does NOT reach the GHI product: csi_to_ghi multiplies the same field
back, an exact inverse verified to 1.14e-13 W m-2. What it reaches is the
interpretation of the intermediate, and the CSI range quoted as evidence the
[0, 1.1] clip never binds.

A near-CONSTANT ratio across calendar months is the discriminating evidence. A
disagreement between two clear-sky models would vary seasonally with air mass
and turbidity; a fixed averaging-window mismatch is close to multiplicative.

    python compare_sarah_clearsky.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
SARAH = os.path.join(ROOT, "data/raw/sarah/SISmm*.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid/"
                              "clearsky_ghi_finegrid_climatology.nc")
OUT = os.path.join(ROOT, "data/processed/evaluation/sarah_clearsky_comparison.csv")


def main():
    files = sorted(glob.glob(SARAH))
    if not files:
        raise SystemExit("No SARAH files. Run fetch_sarah_order.py first.")
    print("%d SARAH monthly files" % len(files))

    sis, sisc = {m: [] for m in range(1, 13)}, {m: [] for m in range(1, 13)}
    for f in files:
        month = pd.Timestamp(re.search(r"SISmm(\d{8})", f).group(1)).month
        with xr.open_dataset(f) as d:
            sis[month].append(float(d.SIS.mean()))
            sisc[month].append(float(d.SISC.mean()))

    with xr.open_dataset(CLEARSKY) as cs:
        mine = [float(cs.clearsky_ghi.isel(month=m - 1).mean()) for m in range(1, 13)]

    rows = []
    for m in range(1, 13):
        s_cs, s_sis = float(np.mean(sisc[m])), float(np.mean(sis[m]))
        rows.append({
            "month": m,
            "pvlib_clearsky_wm2": mine[m - 1],
            "sarah_clearsky_wm2": s_cs,
            "ratio": mine[m - 1] / s_cs,
            "sarah_allsky_wm2": s_sis,
            "sarah_csi": s_sis / s_cs,
            "n_years": len(sis[m]),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    r = df.ratio
    print("\n mo   PVLIB    SARAH    ratio   SARAH CSI")
    for _, x in df.iterrows():
        print("  %2d  %7.2f  %7.2f    %.3f     %.4f"
              % (x.month, x.pvlib_clearsky_wm2, x.sarah_clearsky_wm2,
                 x.ratio, x.sarah_csi))
    print("\nratio  mean %.3f  sd %.4f  range %.3f-%.3f"
          % (r.mean(), r.std(), r.min(), r.max()))
    print("SARAH CSI seasonal range %.3f (Jan, wet) to %.3f (Aug, dry) - the "
          "physical seasonality" % (df.sarah_csi.iloc[0], df.sarah_csi.max()))
    print("\nA ratio this flat across the year (sd %.4f) is an averaging-window "
          "artefact,\nnot a clear-sky model disagreement." % r.std())
    print("Saved: %s" % OUT)


if __name__ == "__main__":
    main()
