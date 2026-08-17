"""Regression tests for the failure modes that actually occurred.

Three severe bugs in this project were silent: none moved an aggregate metric,
and all three were found by inspecting fields by eye. Each test below encodes
one of them, so a recurrence fails loudly instead.

    pytest tests/ -v

Tests that need built datasets skip cleanly when those are absent, so the suite
is runnable on a fresh clone.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data/processed/ml_ready/ml_training_dataset.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
TOPO = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")

needs_data = pytest.mark.skipif(
    not (os.path.exists(TRAIN) and os.path.exists(VAL)),
    reason="ml_ready datasets not built",
)


@pytest.fixture(scope="module")
def train():
    return xr.open_dataset(TRAIN)


@pytest.fixture(scope="module")
def val():
    return xr.open_dataset(VAL)


# --------------------------------------------------------------- §6.11 ----
@needs_data
@pytest.mark.parametrize("split", ["train", "val"])
def test_temporal_alignment(split, train, val):
    """rsds must be better aligned with ssrd at lag 0 than at +/-1 month.

    Catches the misalignment of §6.11, where a nearest-timestamp join across two
    different month-stamping conventions put the previous month's irradiance in
    every validation slot. Aggregate metrics did not move; this assertion does.
    """
    ds = train if split == "train" else val
    s, r = ds["ssrd"].values, ds["rsds"].values
    c0 = np.corrcoef(s.ravel(), r.ravel())[0, 1]
    cm = np.corrcoef(s[1:].ravel(), r[:-1].ravel())[0, 1]
    cp = np.corrcoef(s[:-1].ravel(), r[1:].ravel())[0, 1]
    assert c0 > cm and c0 > cp, (
        f"{split}: rsds/ssrd correlate better at a lag than at zero "
        f"(lag0={c0:.4f}, lag-1={cm:.4f}, lag+1={cp:.4f}). Temporal join is shifted."
    )


@needs_data
def test_predictor_months_match_target_months(val):
    """Every predictor slot must carry its own calendar month."""
    months = pd.DatetimeIndex(val.time.values).month
    assert len(months) == val.sizes["time"]
    assert set(months) == set(range(1, 13)), "validation set does not span all months"


# ---------------------------------------------------------------- §6.3 ----
@needs_data
def test_no_fabricated_constant_region(val):
    """No large contiguous constant block in the target.

    Catches §6.3, where the fine grid extended past the coarse grid's coverage
    and a defensive nan_to_num turned out-of-range NaN into a hard-zero band
    across roughly a quarter of the domain.
    """
    csi = val["clear_sky_index"].values
    zero_fraction = float((np.abs(csi) < 1e-9).mean())
    assert zero_fraction < 0.01, (
        f"{zero_fraction:.2%} of the target is exactly zero — likely a fabricated region"
    )
    edge = np.concatenate([csi[:, 0, :].ravel(), csi[:, -1, :].ravel(),
                           csi[:, :, 0].ravel(), csi[:, :, -1].ravel()])
    assert float((np.abs(edge) < 1e-9).mean()) < 0.01, "hard-zero band on the domain edge"


# ---------------------------------------------------------------- §6.4 ----
@needs_data
@pytest.mark.parametrize("var", ["elevation", "slope", "svf"])
def test_no_constant_predictor_channel(var, train):
    """Every predictor must vary across the domain.

    Catches §6.4, where sky-view factor was 100% NaN, was silently zeroed, and
    entered four models as an all-zero constant channel that nothing detected.
    """
    a = train[var].values
    assert np.isfinite(a).all(), f"{var} contains non-finite values"
    assert np.nanstd(a) > 0, f"{var} is constant across the domain — a dead channel"


# ------------------------------------------------------- CSI <-> GHI ------
@needs_data
@pytest.mark.skipif(not os.path.exists(CLEARSKY), reason="clear-sky climatology not built")
def test_csi_ghi_round_trip_is_exact(val):
    """Dividing by and multiplying back the clear-sky field must be an identity."""
    from ml_dataset_common import csi_to_ghi
    cs = xr.open_dataset(CLEARSKY)
    csi = np.nan_to_num(val["clear_sky_index"].values)
    ghi = csi_to_ghi(csi, val.time.values, cs)
    stack = ghi / np.where(csi == 0, 1, csi)
    back = np.where(csi == 0, 0, ghi / np.where(stack == 0, 1, stack))
    assert np.allclose(back[csi != 0], csi[csi != 0], atol=1e-9)


@needs_data
def test_csi_within_physical_bounds(val):
    csi = val["clear_sky_index"].values
    assert csi.min() >= 0.0 and csi.max() <= 1.1


# --------------------------------------------------------- grids ---------
@needs_data
def test_fine_grid_inside_coarse_grid(val):
    """The target grid must not extend past the predictor grid — cause of §6.3."""
    assert val["fine_lat"].values.min() >= val["lat"].values.min() - 1e-9
    assert val["fine_lat"].values.max() <= val["lat"].values.max() + 1e-9
    assert val["fine_lon"].values.min() >= val["lon"].values.min() - 1e-9
    assert val["fine_lon"].values.max() <= val["lon"].values.max() + 1e-9


# ---------------------------------------------------------------- §6.2 ----
@needs_data
@pytest.mark.parametrize("coord", ["lat", "fine_lat"])
def test_latitude_ascending(coord, train, val):
    """Catches §6.2, which would have trained a vertically flipped model."""
    for ds in (train, val):
        a = ds[coord].values
        assert np.all(np.diff(a) > 0), f"{coord} is not ascending"


@needs_data
def test_expected_shapes(train, val):
    assert train.sizes["time"] == 312
    assert val.sizes["time"] == 168
    assert train.sizes["fine_lat"] == 71 and train.sizes["fine_lon"] == 81


# ------------------------------------------------- feature-set integrity --
def test_rsds_excluded_from_model_inputs():
    """rsds is the target's own source; it must never re-enter as a feature."""
    from ml_dataset_common import COARSE_FEATURE_VARS, CNN_FEATURE_VARS, MODEL_PREDICTOR_VARS
    assert "rsds" not in MODEL_PREDICTOR_VARS
    assert "rsds" not in COARSE_FEATURE_VARS
    assert "rsds" not in CNN_FEATURE_VARS


@needs_data
@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "data/processed/models/pixelwise_rf/manifest.json")),
    reason="RF models not persisted",
)
def test_rf_manifest_matches_builder(val):
    """The persisted models' feature order must match what the builder produces."""
    from pixelwise_common import build_fine_pixel_dataset
    with open(os.path.join(ROOT, "data/processed/models/pixelwise_rf/manifest.json")) as f:
        manifest = json.load(f)
    _, _, feature_vars = build_fine_pixel_dataset(val, TOPO)
    assert feature_vars == manifest["feature_vars"]


# ------------------------------------------------------- §3.4.3 QC -------
@pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, "data/processed/cmip6_bias_corrected")),
    reason="bias-corrected CMIP6 fields not built",
)
def test_bias_corrected_fields_respect_physical_bounds():
    """No predictor fed to the models may be physically impossible.

    EDCM is tail-sensitive: matching a bounded variable against an empirical
    CDF pushed 0.64% of cloud-fraction values outside [0, 100] (as low as
    -5.93%) and a handful of optical depths below zero. Section 3.4.3 claimed
    this was handled; it was not, until apply_qc_bounds.py. Nothing else in the
    pipeline would notice - the models simply receive an out-of-range predictor
    and, being trees, saturate on it.
    """
    import glob
    from apply_qc_bounds import PHYSICAL_BOUNDS

    files = sorted(glob.glob(os.path.join(
        ROOT, "data/processed/cmip6_bias_corrected/*_corrected.nc")))
    if not files:
        pytest.skip("no bias-corrected files")
    offenders = []
    for path in files:
        ds = xr.open_dataset(path)
        for var, (lo, hi) in PHYSICAL_BOUNDS.items():
            if var not in ds.data_vars:
                continue
            a = ds[var].values
            if lo is not None and np.nanmin(a) < lo:
                offenders.append(f"{os.path.basename(path)}:{var} min={np.nanmin(a):.4f} < {lo}")
            if hi is not None and np.nanmax(a) > hi:
                offenders.append(f"{os.path.basename(path)}:{var} max={np.nanmax(a):.4f} > {hi}")
        ds.close()
    assert not offenders, "physically impossible predictor values:\n  " + "\n  ".join(offenders)


# ------------------------------------------------------ loud NaN guard ----
def test_safe_nan_to_num_raises_above_threshold():
    """The helper must refuse to silence a large NaN fraction."""
    from ml_dataset_common import safe_nan_to_num
    mostly_nan = np.full(1000, np.nan)
    with pytest.raises(ValueError, match="EXCEEDS"):
        safe_nan_to_num(mostly_nan, "test:all_nan")
    small = np.arange(1000.0)
    small[0] = np.nan
    out = safe_nan_to_num(small, "test:one_nan")
    assert out[0] == 0.0 and np.isfinite(out).all()


# ------------------------------------------------- documentation drift ----
@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "brief/viva-brief.html")),
                    reason="interview brief not present")
def test_brief_has_no_stale_numbers_or_sentences():
    """The brief must agree with the canonical CSVs cell by cell, carry no
    superseded sentence, and be regenerable into the committed markdown.

    Six audit rounds found the same failure in this document: a table corrected
    while a sentence summarising it was left behind. Those are wrong CLAIMS built
    on right numbers, so a numeric check cannot see them - hence the retired
    SENTENCE list. Cell-level rather than substring matching, because "9.24"
    appears seventeen times and a presence check passed a corrupted table.
    """
    from check_brief_consistency import check
    missing, resurrected, fmt = check()
    problems = []
    problems += ["%s: expected %r, brief has %r" % (lab, exp, got)
                 for lab, exp, got in missing]
    problems += ["superseded sentence present: %r — %s" % (p, why)
                 for p, why in resurrected]
    problems += fmt
    assert not problems, "interview brief is out of date:\n  " + "\n  ".join(problems)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "PROJECT_STATUS.md")),
                    reason="status document not present")
def test_status_document_has_no_stale_numbers():
    """PROJECT_STATUS.md must agree with the canonical evaluation CSVs.

    Four audit rounds found the same failure: a number corrected in a table
    while the sentence beneath it kept the old value. Every instance was caught
    by an external reader rather than by anything in the project. This asserts
    the two conditions that would have caught them - canonical values present,
    superseded values absent outside an explanatory context.
    """
    from check_status_consistency import check
    missing, resurrected = check()
    problems = []
    if missing:
        problems += [f"canonical value absent: {lab} (expected '{val}')"
                     for lab, val in missing]
    if resurrected:
        problems += [f"superseded value present at line {ln}: '{val}' — {why}"
                     for val, why, ln, _ in resurrected]
    assert not problems, "PROJECT_STATUS.md is out of date:\n  " + "\n  ".join(problems)
