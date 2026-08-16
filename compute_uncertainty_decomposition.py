"""Section 3.8.3 full uncertainty decomposition: combine
  - sigma_GCM: inter-model (structural) spread across the 3 GCMs, already
    computed by compute_mme_aggregations.py as 'ghi_uncertainty'
  - sigma_SSP: scenario uncertainty, half the |SSP585 - SSP245| ensemble-mean
    difference
  - sigma_DS: downscaling/model uncertainty, the model's own validation RMSE
    (Table 3.3), propagated uniformly across the domain
via root-sum-of-squares, for the deployed XGBoost and for the two models
retained as comparisons (the U-Net as the shared-weight contrast, the Random
Forest because Section 3.8.4 deployed it before the bootstrap and the
scenario-discrimination test reversed that choice), for each of the 3 planning
horizons.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd

PERIODS = ["near_term_2026_2050", "mid_term_2051_2075", "long_term_2076_2100"]
SCENARIOS = ["ssp245", "ssp585"]

FIELDS_PATH = os.path.abspath("./data/processed/evaluation/validation_spatial_fields.nc")


def validation_rmse():
    """sigma_DS, read from the saved validation fields rather than hardcoded.

    This value was previously a literal in this file and went stale at least
    once, silently attributing a superseded model's error to the current
    projections. Deriving it from the same file the Taylor statistics use means
    it cannot disagree with Table 3.3.
    """
    ds = xr.open_dataset(FIELDS_PATH)
    truth = ds["ghi_true"].values
    out = {}
    for key, var in [("xgb", "ghi_xgb"), ("unet", "ghi_unet"), ("rf", "ghi_rf")]:
        out[key] = float(np.sqrt(np.mean((ds[var].values - truth) ** 2)))
    return out


VALIDATION_RMSE = validation_rmse()
print("sigma_DS read from validation fields: " +
      ", ".join(f"{k}={v:.4f}" for k, v in VALIDATION_RMSE.items()))

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
    sigma_ds = VALIDATION_RMSE[model_key]
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

        # sigma_DS: validation RMSE, uniform across the domain.
        sigma_ds_field = xr.full_like(sigma_gcm, sigma_ds)

        # sigma_ARCH: spread across downscaling architectures, computed the same
        # way as sigma_GCM but treating the four trained architectures as the
        # ensemble members. Without this term the decomposition silently assumes
        # the choice of downscaling model contributes no uncertainty, which the
        # projections contradict.
        arch_means = []
        for other_key, other_dir in ARCH_DIRS.items():
            fields = [
                xr.open_dataset(
                    os.path.join(other_dir, f"mme_suitability_{s}_{period}.nc")
                )["ghi_mean"]
                for s in SCENARIOS
            ]
            arch_means.append(xr.concat(fields, dim="scenario").mean(dim="scenario"))
        sigma_arch = xr.concat(arch_means, dim="architecture").std(dim="architecture")

        sigma_total = np.sqrt(
            sigma_gcm**2 + sigma_ssp**2 + sigma_ds_field**2 + sigma_arch**2
        )

        ds_period = xr.Dataset(
            {
                "sigma_gcm": sigma_gcm,
                "sigma_ssp": sigma_ssp,
                "sigma_ds": sigma_ds_field,
                "sigma_arch": sigma_arch,
                "sigma_total": sigma_total,
            }
        )
        period_datasets[period] = ds_period

        # Domain-mean variance-based contribution shares (Hawkins & Sutton style)
        mean_gcm2 = float((sigma_gcm**2).mean())
        mean_ssp2 = float((sigma_ssp**2).mean())
        mean_ds2 = float((sigma_ds_field**2).mean())
        mean_arch2 = float((sigma_arch**2).mean())
        mean_total2 = mean_gcm2 + mean_ssp2 + mean_ds2 + mean_arch2

        # Report the ROOT-MEAN-SQUARE of each field, not its spatial mean.
        # The variance shares are necessarily computed from mean(sigma^2), and
        # mean(sigma^2) = mean(sigma)^2 + Var(sigma). Tabulating the spatial
        # mean therefore produced columns that could not be squared and summed
        # to reproduce either sigma_total or the percentages - except for
        # sigma_DS, which is spatially constant and so reconciled, which is
        # exactly what made the mismatch confusing. With RMS, the columns
        # square and sum to sigma_total exactly.
        rms = lambda a: float(np.sqrt((a ** 2).mean()))
        summary_rows.append({
            "model": model_key,
            "period": period,
            "sigma_gcm_rms": rms(sigma_gcm),
            "sigma_ssp_rms": rms(sigma_ssp),
            "sigma_arch_rms": rms(sigma_arch),
            "sigma_ds": sigma_ds,
            "sigma_total_rms": rms(sigma_total),
            "n_arch_members": len(ARCH_DIRS),
            "pct_var_gcm": 100 * mean_gcm2 / mean_total2,
            "pct_var_ssp": 100 * mean_ssp2 / mean_total2,
            "pct_var_arch": 100 * mean_arch2 / mean_total2,
            "pct_var_ds": 100 * mean_ds2 / mean_total2,
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
  - Columns are ROOT-MEAN-SQUARE over the domain, so they square and sum to
    sigma_total exactly and the percentages follow directly from them.
  - sigma_DS is a CONSTANT across horizons: it is the model's validation RMSE
    carried forward, not something that varies with lead time. Its share falls
    from near-term to long-term only because sigma_GCM and sigma_arch GROW -
    the downscaling error does not improve over time.
  - sigma_DS is also a different KIND of quantity from the others. It is a
    historical error measured against a reference; sigma_GCM, sigma_SSP and
    sigma_arch are spreads across possible futures. The comparison is
    informative but the two are not the same thing.
  - sigma_arch is estimated from {len(ARCH_DIRS)} architectures and sigma_GCM
    from 3 GCMs. Both are small-sample spread estimates and should be read as
    indicative of magnitude, not as precise quantities.
""")
print(f"Saved summary table: {csv_path}")
