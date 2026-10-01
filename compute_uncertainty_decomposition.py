"""Section 3.8.3 full uncertainty decomposition: combine
  - sigma_GCM: inter-model (structural) spread across the 3 GCMs, already
    computed by compute_mme_aggregations.py as 'ghi_uncertainty'
  - sigma_SSP: scenario uncertainty, half the |SSP585 - SSP245| ensemble-mean
    difference
  - sigma_ARCH: spread across the downscaling architectures that are admissible
    for projection
  - sigma_DS: the downscaling error that SURVIVES a 25-year mean

via root-sum-of-squares, for each of the 3 planning horizons.

On sigma_DS. This term previously carried the model's monthly validation RMSE
(9.2422 W/m-2 for XGBoost) straight into a budget whose other three terms are
spreads of 25-YEAR MEAN changes. That is a category error, and an examiner
identified it: random month-to-month error largely averages out of a 300-month
mean, so quoting the monthly figure inflated the downscaling share to 67.8 per
cent of total variance and suppressed every other term.

The error is therefore split into the part that survives averaging and the part
that does not. For each cell, b(x) is the time-mean bias over the validation
record and r(x,t) the residual about it, so that

    RMSE_monthly^2 = <b^2> + <r^2>            (exactly; asserted below)

A 25-year mean retains b(x) in full and divides the variance of r by an
effective sample size that accounts for month-to-month autocorrelation:

    sigma_DS(x)^2 = b(x)^2 + sd_r(x)^2 / N_eff,    N_eff = N (1-rho) / (1+rho)

For XGBoost this gives 1.54 W m-2 rather than 9.24. The term is now a FIELD
rather than a domain constant, because the bias has spatial structure that a
single number discards. The monthly RMSE is still reported, as
sigma_ds_monthly_rmse, so the correction is auditable rather than silent.

On sigma_ARCH. The Random Forest fails the scenario-discrimination screen
(Section 4.4) and is not admissible for projection. A member cannot be both
disqualified and a measure of the uncertainty in choosing between members, so
the headline sigma_ARCH spans the architectures that pass. The all-four value is
also reported, as sigma_arch_rms_all, so the effect of the choice is visible.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd

from zimbabwe_mask import describe as mask_describe
from zimbabwe_mask import zimbabwe_mask

PERIODS = ["near_term_2026_2050", "mid_term_2051_2075", "long_term_2076_2100"]
SCENARIOS = ["ssp245", "ssp585"]

FIELDS_PATH = os.path.abspath("./data/processed/evaluation/validation_spatial_fields.nc")

MONTHS_IN_MEAN = 300.0      # a 25-year planning horizon, monthly fields

# Architectures admissible for projection, i.e. those that pass the
# scenario-discrimination screen of Section 4.4. Read from that screen's own
# output so the two cannot disagree.
SCREEN_PATH = os.path.abspath("./data/processed/evaluation/scenario_discrimination.csv")
SCREEN_KEY = {"Random Forest": "rf", "XGBoost": "xgb", "CNN": "cnn", "U-Net": "unet"}


def screened_architectures():
    scr = pd.read_csv(SCREEN_PATH)
    passed = {SCREEN_KEY[r.model] for r in scr.itertuples() if bool(r.passes_screen)}
    failed = {SCREEN_KEY[r.model] for r in scr.itertuples() if not bool(r.passes_screen)}
    print(f"screen: admissible {sorted(passed)}; disqualified {sorted(failed)}")
    return passed


def downscaling_sigma():
    """sigma_DS per model: the part of the validation error that survives a
    25-year mean, as a field.

    Returns {key: (sigma_field_2d, monthly_rmse, systematic_rms, n_eff)}.
    The identity RMSE_monthly^2 = <b^2> + <r^2> is asserted, because it is the
    whole basis for separating the two parts and a silent failure here would
    reintroduce the error this function exists to fix.
    """
    ds = xr.open_dataset(FIELDS_PATH)
    truth = ds["ghi_true"].values
    out = {}
    for key, var in [("xgb", "ghi_xgb"), ("unet", "ghi_unet"), ("rf", "ghi_rf"),
                     ("cnn", "ghi_cnn")]:
        e = ds[var].values - truth                      # (time, lat, lon)
        monthly = float(np.sqrt(np.mean(e ** 2)))
        b = e.mean(axis=0)                              # systematic, per cell
        r = e - b[None, :, :]                           # random about it
        sb, sr = float(np.sqrt(np.mean(b ** 2))), float(np.sqrt(np.mean(r ** 2)))
        assert abs(np.hypot(sb, sr) - monthly) < 1e-9, (
            f"{key}: systematic/random split does not reconstruct the monthly RMSE")

        # Pooled lag-1 autocorrelation of the residual, for the effective
        # sample size of a 300-month mean.
        r0, r1 = r[:-1], r[1:]
        rho = float(np.mean(np.sum(r0 * r1, axis=0) /
                            np.sqrt(np.sum(r0 ** 2, axis=0) * np.sum(r1 ** 2, axis=0))))
        n_eff = MONTHS_IN_MEAN * (1.0 - rho) / (1.0 + rho)

        sd_r = r.std(axis=0)                            # per-cell random sd
        sigma_field = np.sqrt(b ** 2 + sd_r ** 2 / n_eff)
        out[key] = (sigma_field, monthly, sb, n_eff)
        print(f"  sigma_DS {key}: monthly RMSE {monthly:.4f} -> "
              f"25-year {float(np.sqrt(np.mean(sigma_field**2))):.4f} W m-2 "
              f"(systematic {sb:.4f}, rho {rho:.3f}, N_eff {n_eff:.0f})")
    return out


print("mask: " + mask_describe())
print("Downscaling error that survives a 25-year mean:")
DS_SIGMA = downscaling_sigma()
ADMISSIBLE = screened_architectures()

# Models that get their own decomposition file: the deployed XGBoost, the U-Net
# as the shared-weight contrast, and the Random Forest, retained because
# Section 3.8.4 deployed it until the bootstrap showed its spatial advantage was
# not established and Section 7.11 showed it cannot separate the emission
# scenarios. Keeping its row makes that reversal auditable rather than silent.
MME_DIRS = {
    "xgb": os.path.abspath("./data/processed/mme_aggregations_xgb"),
    "unet": os.path.abspath("./data/processed/mme_aggregations"),
    "rf": os.path.abspath("./data/processed/mme_aggregations_rf"),
}

# Architectures contributing to sigma_arch. Kept separate from MME_DIRS because
# a spread wants as many members as are available, whereas a decomposition file
# is only wanted for the models actually reported. All four are present: the CNN
# and XGBoost on projections generated from their persisted checkpoints.
ARCH_DIRS = {
    "unet": os.path.abspath("./data/processed/mme_aggregations"),
    "rf": os.path.abspath("./data/processed/mme_aggregations_rf"),
    "cnn": os.path.abspath("./data/processed/mme_aggregations_cnn"),
    "xgb": os.path.abspath("./data/processed/mme_aggregations_xgb"),
}
ARCH_DIRS = {k: v for k, v in ARCH_DIRS.items() if os.path.isdir(v)}
print(f"sigma_arch estimated from {len(ARCH_DIRS)} architectures: {sorted(ARCH_DIRS)}")

OUTPUT_DIR = os.path.abspath("./data/processed/evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_rows = []

for model_key, mme_dir in MME_DIRS.items():
    sigma_ds_arr, sigma_ds_monthly, sigma_ds_syst, n_eff = DS_SIGMA[model_key]
    period_datasets = {}

    for period in PERIODS:
        scenario_data = {}
        for scenario in SCENARIOS:
            path = os.path.join(mme_dir, f"mme_suitability_{scenario}_{period}.nc")
            scenario_data[scenario] = xr.open_dataset(path)

        # sigma_GCM: average the inter-model spread across the two scenarios
        # into one representative structural-uncertainty field per period.
        sigma_gcm = xr.concat(
            [scenario_data[s]["ghi_uncertainty"] for s in SCENARIOS], dim="scenario"
        ).mean(dim="scenario")

        # sigma_SSP: half the absolute ensemble-mean difference between the
        # two emission scenarios.
        sigma_ssp = 0.5 * np.abs(scenario_data["ssp585"]["ghi_mean"] - scenario_data["ssp245"]["ghi_mean"])

        # sigma_DS: the downscaling error that survives a 25-year mean, as a
        # field. The validation grid and the projection grid are the same 71x81
        # mesh under different coordinate names, so the array drops straight in;
        # the shape is asserted rather than assumed.
        assert sigma_ds_arr.shape == sigma_gcm.shape, (
            f"sigma_DS grid {sigma_ds_arr.shape} != projection grid {sigma_gcm.shape}")
        sigma_ds_field = xr.zeros_like(sigma_gcm) + sigma_ds_arr

        # sigma_ARCH: spread across downscaling architectures, computed the same
        # way as sigma_GCM but treating the four trained architectures as the
        # ensemble members. Without this term the decomposition silently assumes
        # the choice of downscaling model contributes no uncertainty, which the
        # projections contradict.
        arch_means = {}
        for other_key, other_dir in ARCH_DIRS.items():
            fields = [
                xr.open_dataset(
                    os.path.join(other_dir, f"mme_suitability_{sc}_{period}.nc")
                )["ghi_mean"]
                for sc in SCENARIOS
            ]
            arch_means[other_key] = xr.concat(fields, dim="scenario").mean(dim="scenario")

        # Headline term spans only the architectures admissible for projection.
        adm = [v for k, v in arch_means.items() if k in ADMISSIBLE]
        sigma_arch = xr.concat(adm, dim="architecture").std(dim="architecture")
        # Reported alongside it so the effect of excluding the failed model is
        # visible rather than buried in the method.
        sigma_arch_all = xr.concat(list(arch_means.values()),
                                   dim="architecture").std(dim="architecture")

        sigma_total = np.sqrt(
            sigma_gcm**2 + sigma_ssp**2 + sigma_ds_field**2 + sigma_arch**2
        )

        ds_period = xr.Dataset(
            {
                "sigma_gcm": sigma_gcm,
                "sigma_ssp": sigma_ssp,
                "sigma_ds": sigma_ds_field,
                "sigma_arch": sigma_arch,
                "sigma_arch_all": sigma_arch_all,
                "sigma_total": sigma_total,
            }
        )
        period_datasets[period] = ds_period

        # Variance-based contribution shares (Hawkins & Sutton style), over the
        # cells inside Zimbabwe. Two fifths of the analysis box is in
        # neighbouring countries, so a share averaged over the whole box is not
        # a statement about Zimbabwe's projection uncertainty.
        MASK = zimbabwe_mask(sigma_gcm.shape)
        sel = lambda a: np.asarray(a.values if hasattr(a, "values") else a)[MASK]
        mean_gcm2 = float((sel(sigma_gcm) ** 2).mean())
        mean_ssp2 = float((sel(sigma_ssp) ** 2).mean())
        mean_ds2 = float((sel(sigma_ds_field) ** 2).mean())
        mean_arch2 = float((sel(sigma_arch) ** 2).mean())
        mean_total2 = mean_gcm2 + mean_ssp2 + mean_ds2 + mean_arch2
        # The same budget with the disqualified Random Forest left in, so the
        # prose can show that the conclusion does not depend on that choice.
        mean_arch2_all = float((sel(sigma_arch_all) ** 2).mean())
        total2_all = mean_gcm2 + mean_ssp2 + mean_ds2 + mean_arch2_all

        # Report the ROOT-MEAN-SQUARE of each field, not its spatial mean.
        # The variance shares are necessarily computed from mean(sigma^2), and
        # mean(sigma^2) = mean(sigma)^2 + Var(sigma). Tabulating the spatial
        # mean therefore produced columns that could not be squared and summed
        # to reproduce either sigma_total or the percentages - except for
        # sigma_DS, which is spatially constant and so reconciled, which is
        # exactly what made the mismatch confusing. With RMS, the columns
        # square and sum to sigma_total exactly.
        rms = lambda a: float(np.sqrt((sel(a) ** 2).mean()))
        summary_rows.append({
            "model": model_key,
            "period": period,
            "sigma_gcm_rms": rms(sigma_gcm),
            "sigma_ssp_rms": rms(sigma_ssp),
            "sigma_arch_rms": rms(sigma_arch),
            "sigma_ds": rms(sigma_ds_field),
            "sigma_ds_monthly_rmse_fullbox": sigma_ds_monthly,
            "sigma_ds_systematic_rms_fullbox": sigma_ds_syst,
            "sigma_ds_n_eff": n_eff,
            "sigma_arch_rms_all": rms(sigma_arch_all),
            "sigma_total_rms": rms(sigma_total),
            "n_arch_members": len(adm),
            "n_arch_members_all": len(arch_means),
            "pct_var_gcm": 100 * mean_gcm2 / mean_total2,
            "pct_var_ssp": 100 * mean_ssp2 / mean_total2,
            "pct_var_arch": 100 * mean_arch2 / mean_total2,
            "pct_var_ds": 100 * mean_ds2 / mean_total2,
            "pct_var_arch_all": 100 * mean_arch2_all / total2_all,
            "pct_var_gcm_all": 100 * mean_gcm2 / total2_all,
            "pct_var_ds_all": 100 * mean_ds2 / total2_all,
        })

    ds_combined = xr.concat(
        [period_datasets[p].expand_dims(period=[p]) for p in PERIODS], dim="period"
    )
    out_path = os.path.join(OUTPUT_DIR, f"uncertainty_decomposition_{model_key}.nc")
    ds_combined.to_netcdf(out_path)
    print(f"Saved: {out_path}")

df = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUTPUT_DIR, "uncertainty_decomposition_summary.csv")
df.to_csv(csv_path, index=False)

print("\n" + "=" * 110)
print(" SECTION 3.8.3 UNCERTAINTY DECOMPOSITION (RMS over the domain, W m-2, and % of total variance)")
print("=" * 110)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("=" * 110)
print(f"""
Notes on reading this table:
  - Columns are ROOT-MEAN-SQUARE over the cells inside Zimbabwe, so they square
    and sum to sigma_total exactly and the percentages follow from them. The
    analysis box is 42.8 per cent outside the country and those cells are
    excluded here, unlike in an earlier version.
  - sigma_DS is CONSTANT across horizons: it is a property of the trained model,
    not of lead time. Its share falls from near to long term only because
    sigma_GCM and sigma_arch GROW - the downscaling error does not improve.
  - sigma_DS is the part of the validation error that survives a 25-year mean,
    not the monthly RMSE. Both are tabulated: sigma_ds against
    sigma_ds_monthly_rmse. Using the monthly figure, as an earlier version did,
    overstates this term by about a factor of six and crowds out the others.
  - sigma_DS remains a different KIND of quantity from the others. It is a
    historical error measured against a reference; sigma_GCM, sigma_SSP and
    sigma_arch are spreads across possible futures. The comparison is
    informative but the two are not the same thing.
  - sigma_arch spans only the architectures that pass the Section 4.4 screen.
    Compare sigma_arch_rms with sigma_arch_rms_all to see what the disqualified
    Random Forest was contributing.
  - sigma_arch is estimated from {len(ADMISSIBLE)} admissible architectures
    ({len(ARCH_DIRS)} trained in all) and sigma_GCM from 3 GCMs, one member
    each. Both are small-sample spread estimates and should be read as
    indicative of magnitude, not as precise quantities. With one member per GCM
    neither internal variability nor its separation from model structure can be
    estimated, so there is no internal-variability term in this budget.
""")
print(f"Saved summary table: {csv_path}")
