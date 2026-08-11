"""Shared helpers for building the ML-ready training/validation datasets.

Implements the two-stage regridding of Section 3.4.1: atmospheric predictors
(Table 3.1) stay on the coarse ERA5 0.25 deg grid (dims "lat"/"lon"), matching
the (C+3)-channel, 0.25 deg input formulation of Section 3.6.5. The ERA5 ssrd
field is separately bicubically interpolated to the fine 0.1 deg target grid
(dims "fine_lat"/"fine_lon") and normalised by the fine-grid clear-sky GHI
climatology (Section 3.5.4) to produce the true super-resolution CSI target
required by the Section 3.6.1 problem formulation.
"""

import os

import numpy as np
import pandas as pd

# Canonical fine (0.1 deg) target/output grid: the CORE Zimbabwe domain,
# deliberately NOT the 0.25 deg buffered topography grid. The buffer exists
# "to reduce edge effects during spatial interpolation and convolution
# operations" (Section 3.2.1) - it is a computational aid, not part of the
# reported target/output domain. Using the full buffered topo grid as the
# target grid is a real bug: it extends up to 0.85 deg beyond the coarse
# ERA5/GCM grid's actual coverage, so bicubic interpolation of ssrd onto
# those extrapolated cells returns NaN, which was then silently zeroed -
# fabricating a hard-zero CSI band around the entire domain in every
# training/validation target and every projected map.
#
# Section 3.2.1 nominally specifies 15.0-22.5 S, 25.0-33.5 E, but the actual
# ERA5 CDS download request (download_era5_predictors2.py) used area
# [-15, 25, -22, 33] - 0.5 deg short of the stated domain at the south and
# east edges. Rather than re-downloading, the fine grid bounds are set to
# exactly match the coarse data's *actual* coverage, so interpolating onto
# it never extrapolates beyond the source grid's convex hull.
CORE_LAT_MIN, CORE_LAT_MAX = -22.0, -15.0
CORE_LON_MIN, CORE_LON_MAX = 25.0, 33.0
FINE_RESOLUTION = 0.1
FINE_LAT = np.round(np.arange(CORE_LAT_MIN, CORE_LAT_MAX + FINE_RESOLUTION / 2, FINE_RESOLUTION), 4)
FINE_LON = np.round(np.arange(CORE_LON_MIN, CORE_LON_MAX + FINE_RESOLUTION / 2, FINE_RESOLUTION), 4)

# Variables STORED in the ML-ready datasets. rsds is retained here even though
# it is no longer a model input, because the Section 3.7.3 interpolation
# baseline is defined as the coarse irradiance field bilinearly interpolated to
# the fine grid, and that baseline needs it.
COARSE_PREDICTOR_VARS = ["clt", "tas", "ps", "huss", "rsds", "od550aer"]

# Variables actually FED TO THE MODELS. rsds is deliberately excluded.
#
# rsds is populated by reusing the ERA5 ssrd field (merge_predictor_stack.py),
# and ssrd is also the field the clear-sky index target is derived from. Giving
# a model the target's own source variable makes the task largely circular: a
# bilinear interpolation of rsds alone reproduces the fine-grid target at
# RMSE ~0.23 W m-2, so no learned model can add value over trivial
# interpolation, and any apparent skill mostly reflects the model recovering
# its own input. This was first observed at daily resolution and was masked at
# monthly resolution by a one-month misalignment in the validation predictor
# file (since fixed). Dropping rsds makes the task a genuine perfect-prognosis
# problem: infer surface irradiance from atmospheric state, not from a copy of
# the irradiance itself.
#
# Set MODEL_PREDICTORS="clt,tas,ps,huss,rsds,od550aer" to restore rsds for the
# ablation that separates the effect of dropping it from the effect of fixing
# the one-month misalignment.
MODEL_PREDICTOR_VARS = [
    v.strip() for v in
    os.environ.get("MODEL_PREDICTORS", "clt,tas,ps,huss,od550aer").split(",")
    if v.strip()
]

# Coarse-grid feature set for U-Net, RF, and XGBoost: 5 atmospheric predictors
# + 3 topographic covariates (Section 3.5.2) + 3 astronomical features
# (Section 3.5.3), per Sections 3.6.3/3.6.4's explicit "together with the
# topographic and astronomical features" wording. 11 features.
COARSE_FEATURE_VARS = MODEL_PREDICTOR_VARS + [
    "elevation", "slope", "svf",
    "solar_zenith_angle", "sin_doy", "cos_doy",
]

# CNN-specific (C+3) input formula (Section 3.6.5): atmospheric predictors +
# elevation/slope/svf only, no astronomical channels. 8 features.
CNN_FEATURE_VARS = MODEL_PREDICTOR_VARS + ["elevation", "slope", "svf"]


def compute_astronomical_features(lat_1d, lon_1d, times):
    """Day-of-year cyclic encoding (Section 3.5.3) and analytic solar zenith
    angle, broadcast onto a (time, lat, lon) grid."""
    doy = pd.DatetimeIndex(times).dayofyear.values
    sin_d = np.sin(2.0 * np.pi * doy / 365.25)
    cos_d = np.cos(2.0 * np.pi * doy / 365.25)

    lat_3d = np.tile(lat_1d[None, :, None], (len(times), 1, len(lon_1d)))
    declination = 23.45 * np.sin(np.radians(284 + doy[:, None, None] * 365 / 365.25))
    sza = np.arccos(np.clip(
        np.sin(np.radians(lat_3d)) * np.sin(np.radians(declination)) +
        np.cos(np.radians(lat_3d)) * np.cos(np.radians(declination)),
        -1.0, 1.0,
    ))
    sza_deg = np.degrees(sza)
    sin_doy = np.tile(sin_d[:, None, None], (1, len(lat_1d), len(lon_1d)))
    cos_doy = np.tile(cos_d[:, None, None], (1, len(lat_1d), len(lon_1d)))
    return sza_deg, sin_doy, cos_doy


def _clearsky_index_and_stack(times, clearsky_fine_ds):
    """Auto-detects whether clearsky_fine_ds is the monthly (12-entry
    "month" coord) or daily (365-entry "doy" coord) climatology and returns
    the correctly-indexed (time, fine_lat, fine_lon) clear-sky stack."""
    if "doy" in clearsky_fine_ds.coords:
        doy = pd.DatetimeIndex(times).dayofyear.values
        doy = np.minimum(doy, 365)  # leap-year Dec 31 (day 366) -> reuse day 365
        clearsky_all = clearsky_fine_ds["clearsky_ghi"].values  # (365, fine_lat, fine_lon)
        return clearsky_all[doy - 1]
    months = pd.DatetimeIndex(times).month.values
    clearsky_all = clearsky_fine_ds["clearsky_ghi"].values  # (12, fine_lat, fine_lon)
    return clearsky_all[months - 1]


def build_fine_grid_csi(ssrd_coarse_da, times, fine_lat, fine_lon, clearsky_fine_ds):
    """Bicubic-interpolate coarse ssrd to the fine 0.1 deg grid and normalise
    by the fine-grid clear-sky GHI climatology (monthly or daily, auto-
    detected) to obtain the CSI training/validation target at true target
    resolution (Sections 3.4.1, 3.5.4). Returns (csi_fine, ssrd_fine) each
    shaped (time, fine_lat, fine_lon).
    """
    ssrd_fine = ssrd_coarse_da.interp(lat=fine_lat, lon=fine_lon, method="cubic")
    ssrd_fine_vals = np.clip(np.nan_to_num(ssrd_fine.values, nan=0.0), 0.0, None)

    clearsky_stack = _clearsky_index_and_stack(times, clearsky_fine_ds)
    clearsky_stack = np.where(clearsky_stack < 1.0, 1.0, clearsky_stack)

    csi_fine = np.clip(ssrd_fine_vals / clearsky_stack, 0.0, 1.1)
    return csi_fine, ssrd_fine_vals


NAN_FRACTION_LIMIT = float(os.environ.get("NAN_FRACTION_LIMIT", "0.01"))


def safe_nan_to_num(arr, name, limit=None, fill=0.0):
    """np.nan_to_num that reports what it silenced, and refuses to silence a lot.

    Two of the three most damaging bugs in this project were a bare
    np.nan_to_num turning a loud failure into a plausible-looking field: a
    sky-view factor that was 100% NaN became an all-zero constant channel that
    four models then trained on, and an out-of-range interpolation that
    returned NaN became a fabricated hard-zero band across a quarter of the
    domain. Neither moved any aggregate metric. Zero-filling is still the right
    behaviour for genuinely isolated gaps, so it is kept - but it now says how
    much it filled, and raises when the fraction is large enough that the
    result is not a gap-fill but a masked failure.
    """
    arr = np.asarray(arr)
    nan_mask = ~np.isfinite(arr)
    frac = float(nan_mask.mean()) if arr.size else 0.0
    if frac > 0:
        threshold = NAN_FRACTION_LIMIT if limit is None else limit
        msg = (f"[safe_nan_to_num] {name}: {nan_mask.sum():,} of {arr.size:,} "
               f"values non-finite ({frac:.4%}), filling with {fill}")
        if frac > threshold:
            raise ValueError(
                msg + f" - EXCEEDS the {threshold:.2%} limit. This is very likely a "
                "real failure (wrong grid, missing source, out-of-range "
                "interpolation) rather than isolated missing data. Investigate "
                "before raising the limit, and pass limit=... only if the gaps "
                "are genuinely expected."
            )
        print(msg)
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


TRAIN_PATH_FOR_CLIMATOLOGY = os.environ.get(
    "ML_TRAIN_PATH", "./data/processed/ml_ready/ml_training_dataset.nc"
)
_TRAIN_CLIM_CACHE = {}


def _training_climatology(clearsky_fine_ds):
    """Per-cell, per-calendar-month mean GHI over the 1985-2010 TRAINING record.

    Returns a (12, fine_lat, fine_lon) array. Cached, since every evaluation
    script calls it and it requires opening and converting the training set.
    """
    key = os.path.abspath(TRAIN_PATH_FOR_CLIMATOLOGY)
    if key in _TRAIN_CLIM_CACHE:
        return _TRAIN_CLIM_CACHE[key]

    import xarray as xr  # local import: keeps this module importable without xarray

    ds_train = xr.open_dataset(key)
    train_times = ds_train.time.values
    train_ghi = csi_to_ghi(
        np.nan_to_num(ds_train["clear_sky_index"].values, nan=0.0),
        train_times, clearsky_fine_ds,
    )
    months = pd.DatetimeIndex(train_times).month
    clim = np.stack([
        train_ghi[months == m].mean(axis=0) if (months == m).any()
        else np.zeros(train_ghi.shape[1:])
        for m in range(1, 13)
    ], axis=0)
    _TRAIN_CLIM_CACHE[key] = clim
    return clim


def evaluation_baselines(ds_val, true_ghi, times, clearsky_fine_ds):
    """The two reference forecasts every model is scored against (Section 3.7.3).

    Returns (interp_ghi, clim_ghi).

    interp: the coarse irradiance field bilinearly interpolated to the fine
    grid. This is the naive spatial disaggregation baseline. Because rsds is a
    reuse of the same ssrd the target is built from, it is a near-perfect
    reproduction of the target and is effectively unbeatable - reporting skill
    against it is what exposes the circularity, not what measures model quality.

    clim: the per-cell, per-calendar-month mean of the target over the
    evaluation period. This is the appropriate reference for a
    perfect-prognosis task, where the question is whether the model explains
    anything beyond the seasonal cycle it could have memorised.
    """
    interp = np.nan_to_num(
        ds_val["rsds"].interp(
            lat=ds_val["fine_lat"].values, lon=ds_val["fine_lon"].values, method="linear"
        ).values,
        nan=0.0,
    )

    # The climatology must be constructible without seeing the evaluation
    # period, or it is not a forecast. It is therefore built from the 1985-2010
    # TRAINING record and applied to 2011-2024. An earlier version averaged the
    # validation target itself, which made it the best attainable climatology
    # for that period and so an unfairly strong reference - understating model
    # skill rather than inflating it, but leaking either way.
    months = pd.DatetimeIndex(times).month
    clim = np.empty_like(true_ghi)
    train_clim = _training_climatology(clearsky_fine_ds)
    for m in range(1, 13):
        sel = months == m
        if sel.any():
            clim[sel] = train_clim[m - 1][np.newaxis, :, :]

    # delta: the interpolated coarse field rescaled so its climatological mean
    # matches the TRAINING-period target climatology (Section 3.7.3's second
    # reference). Scaling toward the validation mean would leak, exactly as the
    # climatology did.
    delta = np.empty_like(true_ghi)
    for m in range(1, 13):
        sel = months == m
        if not sel.any():
            continue
        src_mean = interp[sel].mean(axis=0, keepdims=True)
        tgt_mean = train_clim[m - 1][np.newaxis, :, :]
        scale = np.divide(tgt_mean, src_mean, out=np.ones_like(tgt_mean),
                          where=np.abs(src_mean) > 1e-6)
        delta[sel] = interp[sel] * scale
    return interp, clim, delta


def skill_score(rmse_model, rmse_reference):
    return 1.0 - (rmse_model / rmse_reference) if rmse_reference > 0 else 0.0


def csi_to_ghi(csi_array, times, clearsky_fine_ds):
    """Convert a (time, fine_lat, fine_lon) CSI array back to physical GHI
    (W m-2) using the real month- or day-indexed fine-grid clear-sky
    climatology, rather than a crude single constant (Section 3.5.4)."""
    clearsky_stack = _clearsky_index_and_stack(times, clearsky_fine_ds)
    return csi_array * clearsky_stack
