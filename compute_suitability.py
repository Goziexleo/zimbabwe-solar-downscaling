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


def decay(d_km, d_ref):
    """Section 3.9.3's distance decay; at d = 2*d_ref the score is ~0.135."""
    return np.exp(-d_km / d_ref * 2.0)


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

    ghi_source = "ghi_present_sarah" if "ghi_present_sarah" in ds else "ghi_present_era5"
    print("\nirradiance layer: %s" % ghi_source)

    # ---- exclusion mask (Section 3.9.2) ----
    inzw = ds.in_zimbabwe.values > 0.5
    ex_pa = ds.protected_fraction.values > 0.5
    ex_urb = ds.urban_fraction.values > 0.0
    ex_slope = ds.slope.values > 15.0
    ex_water = ds.water_fraction.values > 0.5
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
    print("\nweighting schemes:")
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
