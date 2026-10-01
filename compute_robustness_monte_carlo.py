"""How robust is the robust set, really?

Section 3.9.6 varies the criterion WEIGHTS over four hand-chosen vectors and
calls the 145 cells that stay in the top two tiers under all four "robust". That
tests one parameter along four points. Three other choices are at least as
arbitrary and were never perturbed:

  d_ref      the distance at which the exponential proximity score falls to 0.37.
             Re-reading this one parameter moved the robust set between 42 and
             145 cells, which is the clearest evidence that it matters.
  thresholds the 0.75 and 0.60 tier cuts.
  weights    four vectors is a thin sample of a six-dimensional simplex.

This draws all four jointly. For each of N draws the weights are sampled from a
Dirichlet centred on the primary vector, each d_ref is scaled by a factor between
a half and two, and the two tier cuts are jittered. A cell's robustness is the
fraction of draws in which it lands in the top two tiers, which is a far more
informative statement than a yes/no against four vectors.

Writes robustness_monte_carlo.csv and robustness_frequency.nc.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from zimbabwe_mask import describe as mask_describe
from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
LAYERS = os.path.join(ROOT, "data/processed/suitability/criterion_layers.nc")
SUIT = os.path.join(ROOT, "data/processed/suitability/suitability_index.nc")
OUT = os.path.join(ROOT, "data/processed/evaluation")

CRITERIA = ["ghi", "slope", "landcover", "roads", "grid", "settlements", "population"]
PRIMARY = np.array([0.35, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05])
D_REF = {"roads": 10.0, "grid": 10.0, "settlements": 5.0}
VERY_HIGH, HIGH = 0.75, 0.60

N_DRAWS = int(os.environ.get("MC_DRAWS", "2000"))
CONCENTRATION = float(os.environ.get("MC_CONC", "40"))   # Dirichlet spread
D_REF_LO, D_REF_HI = 0.5, 2.0
THRESH_JITTER = 0.05


def minmax(a, mask, invert=False):
    v = np.where(mask, a, np.nan)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or hi == lo:
        return np.zeros_like(a)
    s = (a - lo) / (hi - lo)
    return np.clip(1 - s if invert else s, 0, 1)


def main():
    ds = xr.open_dataset(LAYERS)
    suit = xr.open_dataset(SUIT)

    # The retained set, rebuilt exactly as Section 3.9.2 defines it. The index
    # file stores a score for every cell and marks exclusion only through the
    # tier, and tier 4 also holds retained cells that simply score low, so
    # neither a NaN test nor "tier != 4" recovers the right mask.
    AREAL = 0.5
    SLOPE_MAX = 15.0
    wet = ds["water_fraction"].values.copy()
    if "river_fraction" in ds:
        wet = np.clip(wet + ds["river_fraction"].values, 0, 1)
    excluded = ((ds["in_zimbabwe"].values <= AREAL)
                | (ds["protected_fraction"].values > AREAL)
                | (ds["urban_fraction"].values > AREAL)
                | (ds["slope"].values > SLOPE_MAX)
                | (wet > AREAL)
                | np.isnan(ds["ghi_present_sarah"].values))
    keep = ~excluded
    mask = zimbabwe_mask(keep.shape)
    print("mask: %s" % mask_describe())
    print("retained cells: %d" % int(keep.sum()))

    # Criteria that do not depend on a sampled parameter are built once.
    fixed = {
        "ghi": minmax(ds["ghi_present_sarah"].values, keep),
        "slope": minmax(ds["slope"].values, keep, invert=True),
        "landcover": np.nan_to_num(ds["landcover_score"].values),
        "population": minmax(ds["population"].values, keep),
    }
    dist = {k: ds["dist_" + k].values for k in D_REF}

    rng = np.random.default_rng(42)
    shape = keep.shape
    hits = np.zeros(shape, dtype=np.int32)

    for _ in range(N_DRAWS):
        w = rng.dirichlet(CONCENTRATION * PRIMARY)
        scale = {k: rng.uniform(D_REF_LO, D_REF_HI) for k in D_REF}
        vh = VERY_HIGH + rng.uniform(-THRESH_JITTER, THRESH_JITTER)
        hi = HIGH + rng.uniform(-THRESH_JITTER, THRESH_JITTER)

        S = dict(fixed)
        for k in D_REF:
            S[k] = np.exp(-dist[k] / (D_REF[k] * scale[k]))

        si = np.zeros(shape)
        for wi, c in zip(w, CRITERIA):
            si += wi * S[c]
        hits += ((si >= min(vh, hi)) & keep).astype(np.int32)

    freq = np.where(keep, hits / N_DRAWS, np.nan)
    xr.Dataset({"robust_frequency": (("lat", "lon"), freq)},
               coords={"lat": ds.lat, "lon": ds.lon}).to_netcdf(
        os.path.join(OUT, "robustness_frequency.nc"))

    four = int(suit["robustly_suitable"].values.sum())
    rows = []
    for thr in (1.00, 0.99, 0.95, 0.90, 0.75, 0.50):
        n = int(np.nansum(freq >= thr))
        nz = int(np.nansum((freq >= thr) & mask))
        rows.append(dict(frequency_threshold=thr, cells=n, cells_in_zimbabwe=nz))
    df = pd.DataFrame(rows)
    df["four_scheme_robust_set"] = four
    df.to_csv(os.path.join(OUT, "robustness_monte_carlo.csv"), index=False)

    print("\n%d draws over weights, three d_ref values and both tier cuts" % N_DRAWS)
    print("%-22s %10s %12s" % ("retained in >= x of draws", "cells", "in Zimbabwe"))
    for r in rows:
        print("%-22.2f %10d %12d" % (r["frequency_threshold"], r["cells"],
                                     r["cells_in_zimbabwe"]))
    print("\nthe four-scheme robust set reported in Section 4.8.1 is %d cells" % four)
    sel = suit["robustly_suitable"].values.astype(bool)
    if sel.any():
        print("those cells survive a median %.0f per cent of the draws "
              "(5th percentile %.0f per cent)"
              % (100 * np.nanmedian(freq[sel]), 100 * np.nanpercentile(freq[sel], 5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
