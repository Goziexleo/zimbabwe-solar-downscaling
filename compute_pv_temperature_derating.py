"""Ask whether the projected irradiance gain survives the temperature penalty.

The suitability mapping scores irradiance and says nothing about module
temperature, yet the same scenarios that raise GHI also raise air temperature,
and crystalline-silicon output falls with cell temperature. If the penalty
exceeds the gain then the headline direction of the projection chapter does not
carry over to delivered energy, which is what a siting decision is about.

The calculation is deliberately first order and its assumptions are stated:

  cell temperature   T_cell = T_air + k G_poa, the NOCT form, with
                     k = (NOCT - 20)/800 for NOCT = 45 C
  plane-of-array     G_poa approximated as the daytime mean, taken as twice the
                     24-hour monthly mean the projections carry. This is the
                     same averaging-window factor Section 3.5.4 discusses.
  yield              P proportional to G (1 + gamma (T_cell - 25)), evaluated
                     over a gamma range spanning common c-Si modules
  temperature        bias-corrected CMIP6 tas, bilinearly upsampled to the
                     target grid exactly as Section 3.5.1 upsamples the other
                     coarse predictors

Deltas are formed against each model's own historical run, the convention
Section 4.7.1 adopts, and averaged over the three GCMs.

    python compute_pv_temperature_derating.py
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data/processed")
EVAL = os.path.join(PROC, "evaluation")
BC = os.path.join(PROC, "cmip6_bias_corrected")
OUT = os.path.join(EVAL, "pv_temperature_derating.csv")

GCMS = {"ACCESS-CM2": "ghi_ACCESS_CM2",
        "CNRM-CM6-1": "ghi_CNRM_CM6_1",
        "MPI-ESM1-2-HR": "ghi_MPI_ESM1_2_HR"}
SCENARIOS = ["ssp245", "ssp585"]
PERIODS = {"near_term_2026_2050": (2026, 2050),
           "mid_term_2051_2075": (2051, 2075),
           "long_term_2076_2100": (2076, 2100)}

NOCT_C = 45.0
K_POA = (NOCT_C - 20.0) / 800.0      # K per W/m2 of plane-of-array irradiance
DAYTIME_FACTOR = 2.0                 # 24-hour mean -> daytime mean
T_STC_C = 25.0
GAMMAS = [-0.0030, -0.0040, -0.0045]  # per K, common c-Si range


def to_celsius(t):
    return t - 273.15 if np.nanmean(t) > 200.0 else t


def yield_factor(ghi, tair_c, gamma):
    """Relative output per unit irradiance, including the temperature penalty."""
    tcell = tair_c + K_POA * DAYTIME_FACTOR * ghi
    return 1.0 + gamma * (tcell - T_STC_C)


def main():
    hist = xr.open_dataset(os.path.join(EVAL, "xgb_historical_run.nc"))
    mask = zimbabwe_mask(shape=(hist.sizes["lat"], hist.sizes["lon"]))

    rows = []
    for scen in SCENARIOS:
        for period, (y0, y1) in PERIODS.items():
            for gcm, hvar in GCMS.items():
                gh = xr.open_dataset(os.path.join(
                    PROC, "projections_xgb",
                    "downscaled_ghi_%s_%s_2026_2100.nc" % (gcm, scen)))["ghi"]
                gf = gh.sel(time=slice("%d-01-01" % y0, "%d-12-31" % y1)).mean("time").values
                g0 = hist[hvar].mean("time").values

                tf_ds = xr.open_dataset(os.path.join(
                    BC, "%s_%s_EDCDFm_corrected.nc" % (gcm, scen)))["tas"]
                t0_ds = xr.open_dataset(os.path.join(
                    BC, "%s_historical_EDCDFm_corrected.nc" % gcm))["tas"]
                # Upsample the coarse predictor grid to the target grid.
                tf = tf_ds.sel(time=slice("%d-01-01" % y0, "%d-12-31" % y1)).mean("time") \
                          .interp(lat=gh.lat, lon=gh.lon, method="linear").values
                t0 = t0_ds.mean("time").interp(lat=gh.lat, lon=gh.lon, method="linear").values
                tf, t0 = to_celsius(tf), to_celsius(t0)

                ok = mask & np.isfinite(gf) & np.isfinite(g0) & np.isfinite(tf) & np.isfinite(t0)
                d_ghi = 100.0 * (np.mean(gf[ok]) - np.mean(g0[ok])) / np.mean(g0[ok])
                d_tair = float(np.mean(tf[ok]) - np.mean(t0[ok]))

                row = dict(scenario=scen, period=period, gcm=gcm,
                           ghi_hist=float(np.mean(g0[ok])),
                           ghi_change_pct=float(d_ghi), tair_change_K=d_tair)
                for gamma in GAMMAS:
                    pf = gf[ok] * yield_factor(gf[ok], tf[ok], gamma)
                    p0 = g0[ok] * yield_factor(g0[ok], t0[ok], gamma)
                    row["yield_change_pct_gamma%.4f" % gamma] = \
                        float(100.0 * (np.mean(pf) - np.mean(p0)) / np.mean(p0))
                rows.append(row)

    df = pd.DataFrame(rows)
    ens = df.groupby(["scenario", "period"]).mean(numeric_only=True).reset_index()
    ens.insert(2, "gcm", "ensemble mean")
    out = pd.concat([df, ens], ignore_index=True)
    out.to_csv(OUT, index=False)

    mid = "yield_change_pct_gamma-0.0040"
    print("Ensemble mean over Zimbabwe, against each model's own historical run\n")
    print("%-8s %-22s %10s %9s %12s" % ("scen", "period", "dGHI %", "dT K", "dYield %"))
    for _, w in ens.iterrows():
        print("%-8s %-22s %10.2f %9.2f %12.2f"
              % (w["scenario"], w["period"], w["ghi_change_pct"],
                 w["tair_change_K"], w[mid]))

    print("\nSign of the change, irradiance alone against delivered yield:")
    flips = 0
    for _, w in ens.iterrows():
        g, y = w["ghi_change_pct"], w[mid]
        flip = (g > 0) and (y < 0)
        flips += flip
        print("  %-8s %-22s GHI %+.2f%%  ->  yield %+.2f%%%s"
              % (w["scenario"], w["period"], g, y, "   SIGN REVERSES" if flip else ""))
    print("\n%d of %d period-scenario combinations reverse sign at gamma = -0.0040 /K."
          % (flips, len(ens)))

    lo, hi = "yield_change_pct_gamma-0.0030", "yield_change_pct_gamma-0.0045"
    w = ens[ens.period == "long_term_2076_2100"]
    for _, r in w.iterrows():
        print("  %s long term: yield %+.2f%% to %+.2f%% across the gamma range"
              % (r["scenario"], r[lo], r[hi]))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
