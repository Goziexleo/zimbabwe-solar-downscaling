"""Section 3.9: AHP weights, weighted linear combination, and the sensitivity
analysis, from the layers build_suitability_layers.py writes.

On the AHP matrix, which is the one part of this that cannot be computed.
Section 3.9.4 says the weights come from "a pairwise comparison matrix completed
by the researcher in consultation with the thesis supervisors", and reports that
its consistency ratio is below 0.10. Table 3.5 records the resulting weights;
the matrix itself is nowhere in the project. What is coded below is a Saaty-scale
matrix RECONSTRUCTED to reproduce Table 3.5's weights, with its CR computed
honestly. That demonstrates the published weights are attainable from a
consistent set of judgements - it is not a record of the elicitation, and it
should not be presented as one. If the original matrix exists, replace PAIRWISE
with it and the reported CR becomes the real one.

Deriving the matrix from the weights themselves (a_ij = w_i/w_j) would give
CR = 0 by construction and prove nothing, so integer Saaty judgements are used
and the resulting weights are checked against Table 3.5 rather than assumed.
They reproduce it to within 0.008 on every criterion.

Note the CR this produces is 0.008, far below the 0.10 threshold and lower than
a genuine elicitation would usually give - real pairwise judgements from a person
typically land somewhere around 0.03 to 0.08. A CR that small is itself a signal
that the matrix was reverse-engineered from a target weight vector rather than
elicited, and it should be described that way rather than offered as evidence of
careful judgement.

    python compute_suitability.py
"""

import itertools
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data/processed")
LAYERS = os.path.join(PROC, "suitability/criterion_layers.nc")
OUT_NC = os.path.join(PROC, "suitability/suitability_index.nc")
OUT_DIR = os.path.join(PROC, "evaluation")

CRITERIA = ["ghi", "slope", "landcover", "roads", "grid", "settlements", "population"]

# Table 3.5
AHP_WEIGHTS = dict(zip(CRITERIA, [0.35, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05]))

# Reconstructed Saaty judgements, upper triangle, in CRITERIA order.
PAIRWISE = {
    ("ghi", "slope"): 2, ("ghi", "landcover"): 2, ("ghi", "roads"): 4,
    ("ghi", "grid"): 4, ("ghi", "settlements"): 7, ("ghi", "population"): 7,
    ("slope", "landcover"): 2, ("slope", "roads"): 2, ("slope", "grid"): 2,
    ("slope", "settlements"): 4, ("slope", "population"): 4,
    ("landcover", "roads"): 1, ("landcover", "grid"): 2,
    ("landcover", "settlements"): 3, ("landcover", "population"): 3,
    ("roads", "grid"): 1, ("roads", "settlements"): 2, ("roads", "population"): 2,
    ("grid", "settlements"): 2, ("grid", "population"): 2,
    ("settlements", "population"): 1,
}
# Saaty's random index, n = 1..10
RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

D_REF_KM = {"roads": 10.0, "grid": 10.0, "settlements": 5.0}

# One rule for every areal exclusion: a 0.1 degree cell is excluded when MORE
# THAN HALF its area falls in an excluded category. The first version of this
# used > 0.5 for protected areas and water but > 0.0 for urban, which was an
# arbitrary asymmetry rather than a decision. The threshold is defensible at
# this resolution: a cell is about 121 km2 and a utility-scale plant needs
# roughly 2 to 5 km2, so a minority of excluded land still leaves ample room and
# belongs in the SCORE rather than in a disqualification. Section 3.9.2's
# categories are all areal, so they all take the same test.
AREAL_EXCLUSION = 0.5
SLOPE_EXCLUDE_DEG = 15.0
TIERS = [("very high", 0.75), ("high", 0.60), ("moderate", 0.45), ("low", 0.30)]

SCHEMES = {
    "AHP (primary)": AHP_WEIGHTS,
    "irradiance-dominant": None,     # GHI 0.50, rest scaled
    "infrastructure-dominant": None,  # roads+grid 0.40, GHI 0.25
    "equal": {c: 1.0 / 7 for c in CRITERIA},
}


def ahp():
    """Priority vector and consistency ratio from PAIRWISE."""
    n = len(CRITERIA)
    A = np.ones((n, n))
    for (a, b), v in PAIRWISE.items():
        i, j = CRITERIA.index(a), CRITERIA.index(b)
        A[i, j], A[j, i] = v, 1.0 / v
    w = np.real(np.linalg.eig(A)[1][:, np.argmax(np.real(np.linalg.eig(A)[0]))])
    w = np.abs(w) / np.abs(w).sum()
    lam = np.real(np.linalg.eig(A)[0]).max()
    ci = (lam - n) / (n - 1)
    return w, ci / RI[n - 1], lam


def build_schemes():
    s = dict(SCHEMES)
    w = dict(AHP_WEIGHTS)
    rest = 1 - w["ghi"]
    irr = {c: (0.50 if c == "ghi" else w[c] * (0.50 / rest) * (rest / rest))
           for c in CRITERIA}
    scale = (1 - 0.50) / sum(v for c, v in w.items() if c != "ghi")
    irr = {c: (0.50 if c == "ghi" else w[c] * scale) for c in CRITERIA}
    s["irradiance-dominant"] = irr

    infra = dict(w)
    infra["ghi"] = 0.25
    infra["roads"] = infra["grid"] = 0.20
    others = [c for c in CRITERIA if c not in ("ghi", "roads", "grid")]
    rem = 1 - 0.25 - 0.40
    tot = sum(w[c] for c in others)
    for c in others:
        infra[c] = w[c] * rem / tot
    s["infrastructure-dominant"] = infra
    return s


def minmax(a, mask, invert=False):
    v = np.where(mask, a, np.nan)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or hi == lo:
        return np.zeros_like(a)
    s = (a - lo) / (hi - lo)
    return np.clip(1 - s if invert else s, 0, 1)


# Section 3.9.3 is ambiguous and the ambiguity is worth this much comment, because it
# moves the headline result. The section prints the decay as Score = e^(-d/d_ref) but
# also says the score falls below 0.14 beyond d_ref. Those disagree: e^-1 is 0.368, and
# only e^(-2d/d_ref) puts d_ref at 0.135. This follows the 0.14 statement.
#
# It is not a free choice. Under the printed formula the robust set grows from 42 cells
# to 145 and the highest tier from 21 to 54. The qualitative findings survive either way
# - infrastructure still dominates resource, at a grid proximity factor of 6.1 rather
# than 12.5, and the classification is still weight-sensitive in 87.8 per cent of cells
# - but the counts are not robust to it, so Section 4.8.1 reports both.
DECAY_EXPONENT_FACTOR = 2.0


def decay(d_km, d_ref):
    """e^(-2d/d_ref): the reading in which d_ref scores 0.135, per Section 3.9.3."""
    return np.exp(-d_km / d_ref * DECAY_EXPONENT_FACTOR)


def classify(si, excluded):
    out = np.full(si.shape, 4, dtype=np.int8)   # 4 = unsuitable
    for k, (_, lo) in enumerate(TIERS):
        out[(si >= lo) & (out == 4)] = k
    out[excluded] = 4
    return out


def kappa(a, b, valid):
    x, y = a[valid], b[valid]
    cats = sorted(set(x) | set(y))
    n = len(x)
    po = (x == y).mean()
    pe = sum(((x == c).mean() * (y == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ds = xr.open_dataset(LAYERS)
    w, cr, lam = ahp()
    print("AHP consistency: lambda_max %.4f, CR %.4f %s"
          % (lam, cr, "(< 0.10, acceptable)" if cr < 0.10 else "(FAILS 0.10)"))
    print("  reconstructed vs Table 3.5:")
    for c, v in zip(CRITERIA, w):
        print("    %-12s %.3f  vs  %.2f   (%+.3f)" % (c, v, AHP_WEIGHTS[c], v - AHP_WEIGHTS[c]))
    print("  max deviation %.3f" % max(abs(v - AHP_WEIGHTS[c]) for c, v in zip(CRITERIA, w)))

    # Present uses SARAH (7.17). CHANGE between present and future must not,
    # because the projections come from an ERA5-trained chain and differencing
    # SARAH against them would put the SARAH-ERA5 offset inside the change
    # signal. So the future periods are scored on their own downscaled layers
    # and change is taken against ghi_present_era5, never against SARAH.
    periods = [("present", "ghi_present_sarah" if "ghi_present_sarah" in ds
                else "ghi_present_era5")]
    periods += [("present_era5_basis", "ghi_present_era5")]
    periods += [(v.replace("ghi_", ""), v) for v in sorted(ds.data_vars)
                if v.startswith("ghi_ssp")]
    print("\nperiods: %d" % len(periods))
    for tag, v in periods:
        print("   %-28s <- %s" % (tag, v))
    ghi_source = periods[0][1]

    # ---- exclusion mask (Section 3.9.2) ----
    inzw = ds.in_zimbabwe.values > AREAL_EXCLUSION
    ex_pa = ds.protected_fraction.values > AREAL_EXCLUSION
    ex_urb = ds.urban_fraction.values > AREAL_EXCLUSION
    # Section 3.9.2 treats open water, marshland and riparian zones as one
    # category, so the ESA CCI water classes and the HydroSHEDS corridor are
    # unioned before the areal test.
    wet = ds.water_fraction.values.copy()
    if "river_fraction" in ds:
        wet = np.clip(wet + ds.river_fraction.values, 0, 1)
    ex_water = wet > AREAL_EXCLUSION
    # Section 3.9.2 as written: cell-mean slope above 15 degrees. Kept literal
    # rather than retuned. At 0.1 degrees it excludes almost nothing, because
    # averaging over 121 km2 smooths every peak - the 95th percentile cell-mean
    # slope inside Zimbabwe is 7.3 degrees. Changing the rule after seeing that
    # it barely binds would be tuning the method to produce a wanted outcome.
    # The Eastern Highlands are handled by the slope CRITERION at weight 0.20,
    # not by the exclusion, and Section 3.9.2's claim about them is corrected
    # rather than the threshold being moved.
    ex_slope = ds.slope.values > SLOPE_EXCLUDE_DEG
    excluded = (~inzw) | ex_pa | ex_urb | ex_slope | ex_water
    if ghi_source in ds:
        excluded |= np.isnan(ds[ghi_source].values)
    print("\nexclusions (of %d cells):" % excluded.size)
    for name, m in (("outside Zimbabwe", ~inzw), ("protected area", ex_pa),
                    ("urban", ex_urb), ("slope > 15 deg", ex_slope),
                    ("water", ex_water)):
        print("  %-18s %5d  %5.1f%%" % (name, m.sum(), 100 * m.sum() / m.size))
    print("  %-18s %5d  %5.1f%%  <- union" % ("EXCLUDED", excluded.sum(),
                                              100 * excluded.sum() / excluded.size))
    alt = (ds.urban_fraction.values > 0.0).sum()
    print("     (urban at >0%% instead of >%.0f%% would exclude %d cells, not %d - "
          "the threshold matters)" % (100 * AREAL_EXCLUSION, alt, ex_urb.sum()))
    if "river_fraction" in ds:
        riv = ds.river_fraction.values
        print("     (HydroSHEDS riparian corridor: %.3f%% of the mean cell, max "
              "%.2f%% - at 0.1 deg a 200 m corridor is 1.8%% of a cell width, so "
              "it cannot bind an areal-majority rule)"
              % (100 * riv.mean(), 100 * riv.max()))
    steep = (ds.slope_frac_gt15.values > 0.2) & inzw
    print("     (cells with >20%% of their AREA above 15 deg: %d, of which %.0f%% "
          "east of 32E - the Eastern Highlands the slope exclusion does not reach)"
          % (steep.sum(), 100 * (np.meshgrid(ds.lon.values, ds.lat.values)[0][steep] > 32).mean()))
    keep = ~excluded
    print("  %-18s %5d  %5.1f%%" % ("retained", keep.sum(), 100 * keep.sum() / keep.size))

    # ---- standardised criterion scores (Section 3.9.3) ----
    S = {
        "ghi": minmax(ds[ghi_source].values, keep),
        "slope": minmax(ds.slope.values, keep, invert=True),
        "landcover": np.nan_to_num(ds.landcover_score.values),
        "roads": decay(ds.dist_roads.values, D_REF_KM["roads"]),
        "grid": decay(ds.dist_grid.values, D_REF_KM["grid"]),
        "settlements": decay(ds.dist_settlements.values, D_REF_KM["settlements"]),
        "population": minmax(ds.population.values, keep),
    }

    schemes = build_schemes()
    out = xr.Dataset(coords={"lat": ds.lat, "lon": ds.lon})
    tiers = {}

    # Every period is standardised on the SAME min-max range as the present, so
    # the tiers mean the same thing across periods. Rescaling each period by its
    # own range would hide the change entirely: a uniformly brighter future
    # would re-normalise back to the same scores.
    ref = ds[ghi_source].values
    lo, hi = np.nanmin(ref[keep]), np.nanmax(ref[keep])
    print("\nGHI standardised on the present range %.1f..%.1f W m-2 for every period"
          % (lo, hi))

    print("\nweighting schemes (present):")
    for name, wts in schemes.items():
        assert abs(sum(wts.values()) - 1) < 1e-9, name
        si = sum(wts[c] * S[c] for c in CRITERIA)
        si = np.where(excluded, 0.0, si)
        t = classify(si, excluded)
        tiers[name] = t
        key = name.split()[0].replace("(", "")
        out["si_" + key] = (("lat", "lon"), si)
        out["tier_" + key] = (("lat", "lon"), t)
        print("  %-24s SI %.3f mean on retained | very high %d, high %d"
              % (name, si[keep].mean(), (t == 0).sum(), (t == 1).sum()))

    print("\nall periods under the primary AHP weights:")
    w = schemes["AHP (primary)"]
    rows = []
    for tag, var in periods:
        g = np.clip((ds[var].values - lo) / (hi - lo), 0, 1)
        si = sum(w[c] * (g if c == "ghi" else S[c]) for c in CRITERIA)
        si = np.where(excluded, 0.0, si)
        t = classify(si, excluded)
        out["si_" + tag] = (("lat", "lon"), si)
        out["tier_" + tag] = (("lat", "lon"), t)
        rows.append(dict(period=tag, mean_ghi=float(ds[var].values[keep].mean()),
                         mean_si=float(si[keep].mean()),
                         very_high=int((t == 0).sum()), high=int((t == 1).sum())))
        print("  %-30s GHI %6.2f | SI %.4f | very high %4d | high %4d"
              % (tag, rows[-1]["mean_ghi"], rows[-1]["mean_si"],
                 rows[-1]["very_high"], rows[-1]["high"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "suitability_by_period.csv"),
                              index=False)

    # change, taken ERA5-basis against ERA5-trained projections
    base = out["si_present_era5_basis"].values
    print("\nchange in SI against the ERA5-basis present (same measurement system):")
    for tag, _ in periods:
        if tag.startswith("ssp"):
            d_si = out["si_" + tag].values - base
            out["dsi_" + tag] = (("lat", "lon"), d_si)
            print("  %-30s mean %+.4f | cells improving %5d of %d"
                  % (tag, d_si[keep].mean(), int((d_si[keep] > 0).sum()), keep.sum()))

    prim = tiers["AHP (primary)"]
    print("\nCohen's kappa against the AHP classification (retained cells):")
    for name, t in tiers.items():
        if name != "AHP (primary)":
            print("  %-24s %.4f" % (name, kappa(prim, t, keep)))

    robust = np.all([(t <= 1) for t in tiers.values()], axis=0) & keep
    shifts = np.any([(t != prim) for t in tiers.values()], axis=0) & keep
    out["robustly_suitable"] = (("lat", "lon"), robust)
    out["weight_sensitive"] = (("lat", "lon"), shifts)
    print("\nrobustly suitable (very high or high under ALL four schemes): %d cells (%.1f%% of retained)"
          % (robust.sum(), 100 * robust.sum() / keep.sum()))
    print("weight-sensitive (tier changes under any scheme):             %d cells (%.1f%% of retained)"
          % (shifts.sum(), 100 * shifts.sum() / keep.sum()))

    out.attrs["irradiance_layer"] = ghi_source
    out.attrs["ahp_cr"] = float(cr)
    os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
    out.to_netcdf(OUT_NC)
    pd.DataFrame([dict(scheme=k, **v) for k, v in schemes.items()]).to_csv(
        os.path.join(OUT_DIR, "suitability_weights.csv"), index=False)
    print("\nSaved %s" % OUT_NC)


if __name__ == "__main__":
    main()
