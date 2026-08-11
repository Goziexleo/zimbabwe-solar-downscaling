"""Equidistant CDF Matching (EDCM) bias correction (Section 3.4.4).

Consolidated single source of truth for CMIP6 bias correction: covers all 6
Table 3.1 predictors (previously only 4-5 were corrected, and od550aer's
AERmon realm was never matched by the file-glob pattern, silently dropping
it). generate_future_projections.py reads directly from this script's output
(data/processed/cmip6_bias_corrected/{model}_{ssp}_EDCDFm_corrected.nc).
"""

import numpy as np
import xarray as xr
import os
import glob
import time
from scipy.interpolate import interp1d

from ml_dataset_common import COARSE_PREDICTOR_VARS

# --- Configuration ---
era5_file = "./data/processed/era5/predictors/era5_monthly_predictors_1985_2010.nc"
cmip6_raw_dir = "./data/raw/cmip6"
output_dir = "./data/processed/cmip6_bias_corrected"
os.makedirs(output_dir, exist_ok=True)

models = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
scenarios = ["ssp245", "ssp585"]

# "historical" is emitted as an extra pseudo-scenario so the perfect-prognosis
# transfer test (test_perfect_prognosis.py) has bias-corrected CMIP6 predictors
# for a period where ERA5 truth exists. For it, the target period IS the
# calibration period, so the marginal distribution of each corrected variable
# matches ERA5 almost by construction. That makes the transfer test optimistic
# about MARGINALS - but it leaves the inter-variable correlations, the spatial
# covariance and the temporal sequencing entirely as CMIP6 produced them, which
# is what the test is actually probing.
if os.environ.get("EDCM_INCLUDE_HISTORICAL", "0") == "1":
    scenarios = scenarios + ["historical"]
variables = COARSE_PREDICTOR_VARS  # clt, tas, ps, huss, rsds, od550aer
REALM_BY_VAR = {
    "clt": "Amon", "tas": "Amon", "ps": "Amon", "huss": "Amon", "rsds": "Amon",
    "od550aer": "AERmon",
}

print("Loading ERA5 (+ MERRA-2 od550aer, already merged) reference baseline...")
ds_obs = xr.open_dataset(era5_file, engine='netcdf4')
missing = set(variables) - set(ds_obs.data_vars)
if missing:
    raise ValueError(
        f"{era5_file} is missing {sorted(missing)}. Run merge_predictor_stack.py first."
    )


def get_coord_names(ds):
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    return lat_name, lon_name


def get_files(model, experiment, var):
    realm = REALM_BY_VAR[var]
    pattern = os.path.join(cmip6_raw_dir, f"{var}_{realm}_{model}_{experiment}_*.nc")
    all_files = sorted(glob.glob(pattern))
    return [f for f in all_files if os.path.getsize(f) > 0]


def edcdf_matching_1d(obs_hist, sim_hist, sim_fut, bins=100):
    """Equidistant CDF matching (Lanzante et al., 2020): preserves the GCM's
    absolute projected change signal while correcting distributional shape."""
    obs_hist = obs_hist[~np.isnan(obs_hist)]
    sim_hist = sim_hist[~np.isnan(sim_hist)]
    if len(obs_hist) == 0 or len(sim_hist) == 0:
        return np.full_like(sim_fut, np.nan)

    quantiles = np.linspace(0, 1, bins + 1)
    obs_h_q = np.quantile(obs_hist, quantiles)
    sim_h_q = np.quantile(sim_hist, quantiles)

    sim_f_sorted = np.sort(sim_fut)
    sim_f_cdf = np.linspace(0, 1, len(sim_fut))
    _, unique_idx = np.unique(sim_f_sorted, return_index=True)
    if len(unique_idx) > 1:
        f_sim_f = interp1d(sim_f_sorted[unique_idx], sim_f_cdf[unique_idx], kind='linear', fill_value="extrapolate")
        probabilities = np.clip(f_sim_f(sim_fut), 0, 1)
    else:
        probabilities = np.full_like(sim_fut, 0.5)

    f_obs_h_inv = interp1d(quantiles, obs_h_q, kind='linear', fill_value="extrapolate")
    f_sim_h_inv = interp1d(quantiles, sim_h_q, kind='linear', fill_value="extrapolate")
    return sim_fut + f_obs_h_inv(probabilities) - f_sim_h_inv(probabilities)


# --- Pipeline Execution ---
for model in models:
    for ssp in scenarios:
        out_path = os.path.join(output_dir, f"{model}_{ssp}_EDCDFm_corrected.nc")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"\nSkipping {model} - {ssp} (already completed)")
            continue

        print(f"\nProcessing {model} - {ssp}")
        corrected_vars = []
        skip_scenario = False

        for var in variables:
            print(f"  Bias correcting: {var}")
            hist_files = get_files(model, "historical", var)
            fut_files = hist_files if ssp == "historical" else get_files(model, ssp, var)

            if not hist_files or not fut_files:
                print(f"    Warning: Missing files for {var}. Skipping variable.")
                continue

            try:
                ds_h = xr.open_mfdataset(
                    hist_files, combine='by_coords', engine='netcdf4', chunks=None,
                    data_vars='minimal', coords='minimal', compat='override',
                ).sel(time=slice('1985-01-01', '2010-12-31'))

                fut_slice = (slice('1985-01-01', '2010-12-31') if ssp == "historical"
                             else slice('2026-01-01', '2100-12-31'))
                ds_f = xr.open_mfdataset(
                    fut_files, combine='by_coords', engine='netcdf4', chunks=None,
                    data_vars='minimal', coords='minimal', compat='override',
                ).sel(time=fut_slice)
            except Exception as e:
                print(f"    ERROR: Failed to open dataset for {var} in {model}: {e}")
                skip_scenario = True
                break

            lat_h, lon_h = get_coord_names(ds_h)
            lat_o, lon_o = get_coord_names(ds_obs[var])

            t0 = time.time()
            sim_h_da = ds_h[var].rename({lat_h: 'lat', lon_h: 'lon'}).interp(
                lat=ds_obs[var][lat_o], lon=ds_obs[var][lon_o], method="linear")
            sim_f_da = ds_f[var].rename({lat_h: 'lat', lon_h: 'lon'}).interp(
                lat=ds_obs[var][lat_o], lon=ds_obs[var][lon_o], method="linear")

            # Materialise to plain numpy ONCE: indexing sim_h_da/sim_f_da.values
            # inside the pixel loop would otherwise re-run the interpolation
            # from scratch on every one of the ~950 grid-cell iterations.
            obs_vals = ds_obs[var].values
            sim_h_vals = sim_h_da.values
            sim_f_vals = sim_f_da.values
            print(f"    Regridded to ERA5 grid in {time.time() - t0:.1f}s", flush=True)

            t0 = time.time()
            data_out = np.zeros_like(sim_f_vals)
            for i in range(sim_f_vals.shape[1]):
                for j in range(sim_f_vals.shape[2]):
                    data_out[:, i, j] = edcdf_matching_1d(
                        obs_vals[:, i, j],
                        sim_h_vals[:, i, j],
                        sim_f_vals[:, i, j],
                    )
            print(f"    Pixel-wise EDCM matching in {time.time() - t0:.1f}s", flush=True)

            corrected_vars.append(xr.DataArray(data_out, coords=sim_f_da.coords, dims=sim_f_da.dims, name=var))

        if corrected_vars and not skip_scenario:
            missing_vars = set(variables) - {da.name for da in corrected_vars}
            if missing_vars:
                print(f"  Warning: {model}/{ssp} missing corrected variables {sorted(missing_vars)}; "
                      "generate_future_projections.py requires all 6 predictors.")
            ds_output = xr.merge(corrected_vars, compat='override')
            ds_output.to_netcdf(out_path)
            print(f"Saved: {out_path}")
        elif skip_scenario:
            print(f"  Skipping saving {model} - {ssp} due to file errors.")

print("\nProcessing Complete.")
