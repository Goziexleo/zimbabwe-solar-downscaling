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
def test_chapter3_has_no_stale_numbers_or_claims():
    """Chapter 3 must agree with the canonical CSVs, carry no superseded figure
    or design claim, and keep its citation fields structurally intact.

    Skips cleanly when the chapter is absent - it lives outside the repository -
    so a fresh clone still runs green. Everything this catches was previously
    found by eye: three figures that survived a deployment change, and a section
    describing a Model Output Statistics pipeline that was never built. The field
    check exists because the edits here are run-level surgery on live Zotero
    fields, and a broken begin/end pair silently destroys the bibliography.

    Section 3.7.4's ablation table is checked cell by cell rather than by
    substring, because its figures also appear in the paragraph beneath it - a
    presence check over the whole document passes a corrupted cell.
    """
    from check_chapter3_consistency import check, CHAPTER
    missing, resurrected, fields, ablation = check()
    if missing is None:
        pytest.skip("Chapter 3 not found at %s" % CHAPTER)
    problems = ["canonical value absent: %s (expected %r)" % (lab, v) for lab, v in missing]
    problems += ["superseded text present: %r — %s" % (p, why) for p, why in resurrected]
    problems += fields + ablation
    assert not problems, "Chapter 3 is out of date:\n  " + "\n  ".join(problems)


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


def test_reconstructed_ahp_cr_is_never_published():
    """No document may quote the reconstructed matrix's consistency ratio.

    compute_suitability.PAIRWISE was fitted to reproduce Table 3.5's weights,
    so its CR measures the fit and not the researcher's judgement. Chapter 4
    nonetheless stated "the consistency ratio of the pairwise comparison matrix
    is 0.0076, below the 0.10 acceptability threshold" - a validation statistic
    for a matrix built backwards from its own answer, which is the same
    circularity as training on a transform of the target.

    Chapter 3 never quoted a value, only that one was computed. Chapter 4
    invented the number, so this guards the generator rather than the chapter:
    the .docx is a build product, and a hardcoded CR in the generator is what
    would put it back.
    """
    import compute_suitability as cs

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems = []

    if not cs.PAIRWISE_IS_ELICITED:
        _, cr, _ = cs.ahp()
        cr_txt = "%.4f" % cr
        for name in ("build_chapter4.py", "build_chapter5.py"):
            path = os.path.join(root, name)
            if not os.path.exists(path):
                continue
            src = open(path, encoding="utf-8").read()
            if cr_txt in src or cr_txt.rstrip("0") in src:
                problems.append(f"{name} hardcodes the reconstructed CR {cr_txt}")
            low = src.lower()
            for phrase in ("consistency ratio of the pairwise",
                           "consistency ratio is 0.",
                           "consistency ratio of 0."):
                if phrase in low:
                    problems.append(f"{name} quotes a consistency ratio: '{phrase}'")

    assert not problems, (
        "The reconstructed AHP consistency ratio is not a result:\n  "
        + "\n  ".join(problems))


def test_scheme_kappa_is_read_from_disk_not_hardcoded():
    """Chapter 4's Cohen's kappa values must come from the CSV.

    They were literals - 0.697, 0.317, -0.111 - because compute_suitability.py
    only ever printed them. The decay resolution changed every one of them
    (to 0.725, 0.476, -0.091) and the literals stayed, so the generator whose
    stated purpose is that "no figure in the prose can drift from the analysis"
    was itself carrying three stale numbers. They are persisted now.
    """
    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = os.path.join(root, "data/processed/evaluation/suitability_schemes.csv")
    assert os.path.exists(csv), "suitability_schemes.csv is missing; re-run compute_suitability.py"

    sch = pd.read_csv(csv).set_index("scheme")
    assert "kappa_vs_primary" in sch.columns
    assert pd.isna(sch.loc["primary", "kappa_vs_primary"]), \
        "the primary scheme has no kappa against itself"

    def code_only(path):
        """Source with comments stripped.

        The first version of this check flagged its own explanatory comment in
        make_suitability_maps.py, which names the stale numbers in order to
        record why they are read from disk now. A guard that cannot tell code
        from prose fails on documentation, which teaches people to delete the
        documentation.
        """
        import io
        import tokenize
        out = []
        with open(path, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type != tokenize.COMMENT:
                    out.append(tok.string)
        return "\n".join(out)

    src = code_only(os.path.join(root, "build_chapter4.py"))
    stale = [v for v in ("0.697", "0.317", "0.111") if v in src]
    assert not stale, f"build_chapter4.py hardcodes superseded kappa values: {stale}"

    live = ["%.3f" % sch.loc[s, "kappa_vs_primary"]
            for s in ("irradiance-dominant", "infrastructure-dominant", "equal")]
    assert not [v for v in live if v.lstrip("-") in src], (
        "build_chapter4.py hardcodes the current kappa values; read them from the CSV "
        "so the next regeneration cannot leave them behind")

    # Same class of defect, and it reached the reader more directly: figure 09's
    # own title said "42 cells" and "90.8%" while the generated caption beside it
    # said 145 and 87.8%. The figure appears in Chapter 4 and Chapter 5.
    maps = code_only(os.path.join(root, "make_suitability_maps.py"))
    rb = pd.read_csv(os.path.join(root, "data/processed/evaluation",
                                  "suitability_robustness.csv")).iloc[0]
    baked = [v for v in ("42 cells", "90.8", "1.8%",
                         "%d cells" % rb.robust,
                         "%.1f" % (100 * rb.weight_sensitive / rb.assessed))
             if v in maps]
    assert not baked, f"make_suitability_maps.py bakes robustness numbers into titles: {baked}"


def test_figure_titles_carry_no_baked_results():
    """No rendered figure label may hardcode a result.

    Three were found by looking at the rendered PNGs, which nothing else in
    this project does - every guard here reads text:

      * figure 09's title said "90.8%" and "42 cells" against a generated
        caption saying 87.8% and 145;
      * figure 05's title said the U-Net ratio ran "0.72 to 0.01" and the CNN
        "0.91-1.29", while the annotations beside it were computed from the cut
        sweep - so the panel contradicted itself after the honest retrain
        (0.86-0.01 and 0.92-1.59);
      * figure 01's title said "U-Net's zero mean is cancellation, not
        accuracy", true of the leakage-selected U-Net and false once its mean
        bias became +2.50.

    All three now read from disk. This asserts the superseded values are gone
    and that today's values are not baked in to replace them.
    """
    import pandas as pd
    import tokenize

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def code_only(path):
        out = []
        with open(path, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type != tokenize.COMMENT:
                    out.append(tok.string)
        return "\n".join(out)

    figs = code_only(os.path.join(root, "make_figures.py"))
    problems = []

    for v in ("0.72 to 0.01", "0.91-1.29", "zero mean is cancellation"):
        if v in figs:
            problems.append(f"make_figures.py still carries the superseded {v!r}")

    sens = pd.read_csv(os.path.join(root, "data/processed/evaluation",
                                    "effective_resolution_cut_sensitivity.csv")
                       ).set_index("field")
    live = ["%.2f" % sens.loc[m, c]
            for m in ("U-Net", "CNN") for c in ("min", "max")]
    for v in set(live):
        # 1.00 and the like are legitimate axis limits, so only flag the
        # distinctive ones the title would quote.
        if v not in ("1.00", "0.00") and f'"{v}' in figs:
            problems.append(f"make_figures.py bakes the cut-sweep value {v}")

    assert not problems, "\n  ".join(["Figure labels must be computed:"] + problems)


def test_scenario_discrimination_is_read_from_disk():
    """Table 4.5 must come from the CSV, and the CSV must exist.

    The screen is the evidence for the deployment decision, and it lived as
    twelve literals typed into build_chapter4.py with nothing on disk behind
    them. Random Forest and XGBoost were right - neither was ever retrained.
    Every CNN and U-Net figure was stale, because the honest retrain
    regenerated their projections and nothing regenerated the table:
    +1.393/+4.053/+9.643 against a true +1.272/+3.851/+9.478, and a CNN
    ordering score printed as 100.0% that is really 99.9%.
    """
    import pandas as pd
    import tokenize

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = os.path.join(root, "data/processed/evaluation/scenario_discrimination.csv")
    assert os.path.exists(csv), (
        "scenario_discrimination.csv is missing; run compute_scenario_discrimination.py")

    scr = pd.read_csv(csv).set_index("model")
    for m in ("Random Forest", "XGBoost", "CNN", "U-Net"):
        assert m in scr.index, f"{m} missing from the screen"

    # The screen's verdict on the Random Forest is load-bearing for §4.4: it is
    # the reason the second-most-accurate model is not deployed.
    rf = scr.loc["Random Forest"]
    assert not rf.grows, "the Random Forest no longer fails the screen; §4.4 needs rewriting"
    assert scr.loc["XGBoost"].passes_screen, "XGBoost no longer passes its own screen"

    def code_only(path):
        out = []
        with open(path, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type != tokenize.COMMENT:
                    out.append(tok.string)
        return "\n".join(out)

    stale = ("1.393", "4.053", "9.643", "1.075", "1.998", "4.631")
    problems = []
    for name in ("build_chapter4.py", "build_chapter5.py"):
        src = code_only(os.path.join(root, name))
        problems += [f"{name} hardcodes the superseded {v}" for v in stale if v in src]
        # and today's values must not be baked in to replace them
        for m in ("CNN", "U-Net"):
            for h in ("near", "mid", "long"):
                v = "%.3f" % scr.loc[m, "sep_%s_term" % h]
                if v in src:
                    problems.append(f"{name} hardcodes the current {m} {h}-term value {v}")
    assert not problems, "\n  ".join(["Table 4.5 must be read from the CSV:"] + problems)


def test_withdrawn_dropout_attribution_stays_withdrawn():
    """The U-Net's damping must not be re-attributed to dropout.

    Chapters 4 and 5 stated that removing spatial dropout "raises the spectral
    ratio beyond wavenumber 10 from 0.085 to 0.771" and concluded that dropout
    causes the damping. Three things were wrong: 0.085 was a pre-retrain
    deployed-model figure paired with a sweep figure; the sweep's own baseline
    carries the same dropout rate and shows a ratio of 1.24, an excess rather
    than a deficit, so it never reproduced the defect; and removing dropout
    leaves the ratio no closer to unity (0.23 against 0.24).

    This asserts the data still says so, and that the withdrawn phrasing has not
    returned. If a future sweep produces a baseline that does reproduce the
    damping, this test fails and the attribution can be revisited on evidence.
    """
    import pandas as pd
    import tokenize

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uo = pd.read_csv(os.path.join(root, "data/processed/evaluation",
                                  "unet_optimisation.csv")).set_index("variant")

    base_r = uo.loc["baseline", "test_spec_ratio"]
    drop_r = uo.loc["drop_0", "test_spec_ratio"]
    assert base_r > 1.0, (
        f"the sweep baseline now shows damping (ratio {base_r:.3f}); it did not before, "
        "so the dropout attribution can be re-examined")
    assert abs(1 - drop_r) >= abs(1 - base_r) - 0.05, (
        "removing dropout now moves the spectral ratio materially closer to unity; "
        "§4.5's withdrawal should be revisited")

    def code_only(path):
        out = []
        with open(path, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type != tokenize.COMMENT:
                    out.append(tok.string)
        return "\n".join(out)

    problems = []
    for name in ("build_chapter4.py", "build_chapter5.py"):
        src = code_only(os.path.join(root, name))
        if "0.085" in src:
            problems.append(f"{name} carries the superseded 0.085 spectral ratio")
        for phrase in ("dropout smooths the predicted field",
                       "refutes the smoothness-prior",
                       "isolates it to the regularisation setting",
                       "removes the smoothing failure"):
            if phrase in src:
                problems.append(f"{name} restates the withdrawn attribution: {phrase!r}")
    assert not problems, "\n  ".join(["The dropout attribution is withdrawn:"] + problems)


def test_every_appendix_c_table_is_referenced():
    """An appendix table nothing points at is padding, not evidence.

    Appendix C is generated from the evaluation CSVs that fed no table in the
    document. On the round that created it, nine of its ten tables were
    reachable only through the List of Tables, which is how an appendix comes to
    read as bulk. The pointers now live in two places - Chapters 4 and 5 are
    regenerated, so theirs are in the generators, while Chapters 2 and 3 are
    edited in place - and this checks the result rather than either source.

    Matches "Table C.n" and the plural "Tables C.n and C.m": the first version
    of this check read only the singular and reported the two grid tables as
    orphans when they were in fact cited together.
    """
    import re
    from check_chapter3_consistency import CHAPTER
    if not os.path.exists(CHAPTER):
        pytest.skip("dissertation not found at %s" % CHAPTER)
    import docx
    ps = [p.text.strip() for p in docx.Document(CHAPTER).paragraphs]
    head = [i for i, t in enumerate(ps) if t.startswith("Appendix C")]
    if not head:
        pytest.skip("Appendix C not present")
    start = head[0]
    lot = [i for i, t in enumerate(ps) if t == "List of Tables"]
    # Skip the front-matter lists, which name every table by construction.
    body = "\n".join(ps[(lot[0] + 40 if lot else 0):start])
    tables = sorted(int(m.group(1)) for t in ps[start:]
                    for m in [re.match(r"^Table C\.(\d+)\.", t)] if m)
    refs = set(re.findall(r"Tables?\s+C\.(\d+)", body))
    refs |= set(re.findall(r"Tables?\s+C\.\d+\s+and\s+C\.(\d+)", body))
    orphans = [n for n in tables if str(n) not in refs]
    assert tables, "Appendix C has a heading but no numbered tables"
    assert not orphans, (
        "Appendix C tables cited nowhere in the chapters: %s"
        % ", ".join("C.%d" % n for n in orphans))


def test_cut_table_is_a_weighted_average_of_the_spectra():
    """Table 4.6's cuts must lie inside Table C.7's per-wavenumber ratios.

    The power retained beyond a cut is the truth-weighted average of the
    per-wavenumber ratios from that cut upward, so it cannot fall outside their
    range. For seven rounds it did: the cut table held a SHARE ratio - each
    field's power beyond the cut as a fraction of its own total, model over
    truth - while its caption and the chapter described power retained relative
    to the target. The U-Net read 1.21 beyond wavenumber 3, reported as a slight
    excess, when no per-wavenumber ratio from 3 upward exceeds 1.01 and its
    actual power there is 0.83 of the truth's.

    Nothing internal caught it because both tables were self-consistent and
    came from the same arrays; only the relation between them was wrong. This
    checks the relation. The share ratio is still emitted, under share_k>=N, and
    is deliberately not checked here - it is a shape measure and may exceed the
    per-wavenumber range.
    """
    import pandas as pd
    EVAL = os.path.join(ROOT, "data/processed/evaluation")
    sp_path = os.path.join(EVAL, "power_spectra.csv")
    cut_path = os.path.join(EVAL, "effective_resolution_cut_sensitivity.csv")
    if not (os.path.exists(sp_path) and os.path.exists(cut_path)):
        pytest.skip("spectra not computed")
    sp = pd.read_csv(sp_path)
    cut = pd.read_csv(cut_path).set_index("field")
    truth = sp["CSI Truth (ERA5)"]
    cols = [c for c in cut.columns if c.startswith("k>=")]
    assert cols, "the cut table has no k>=N columns"
    problems = []
    for field in cut.index:
        col = "CSI %s" % field
        if col not in sp.columns:
            continue
        ratio = sp[col] / truth
        for c in cols:
            k = int(c[3:])
            m = sp.wavenumber >= k
            lo, hi = ratio[m].min(), ratio[m].max()
            v = float(cut.loc[field, c])
            if not (lo - 1e-9 <= v <= hi + 1e-9):
                problems.append(
                    "%s %s = %.4f is outside the per-wavenumber range "
                    "[%.4f, %.4f]; the cut table is not a weighted average of "
                    "the spectra" % (field, c, v, lo, hi))
    assert not problems, "\n  ".join(["Table 4.6 contradicts Table C.7:"] + problems)


def test_no_em_dashes_or_spaced_hyphens_in_the_prose():
    """House punctuation: no em dashes, no spaced hyphens used as dashes.

    Both appeared this round, introduced by generator wording rather than by
    hand, and the v8 re-check counted them. This checks rather than rewrites,
    because the right replacement depends on the sentence - parentheses for an
    apposition, a comma or colon for a single break - and a mechanical
    substitution produces worse prose than the dash it removes.

    The bibliography is excluded: Wilby et al. (2002) is titled "SDSM - a
    decision support tool ..." with an em dash, and a cited title is quoted as
    published.
    """
    import re
    from check_chapter3_consistency import CHAPTER
    if not os.path.exists(CHAPTER):
        pytest.skip("dissertation not found at %s" % CHAPTER)
    import docx
    d = docx.Document(CHAPTER)
    problems = []
    for i, p in enumerate(d.paragraphs):
        if p.style.name == "Bibliography":
            continue
        t = p.text
        for m in re.finditer(r"—", t):
            problems.append("para %d em dash: ...%s..."
                            % (i, t[max(0, m.start() - 50):m.end() + 50]))
        for m in re.finditer(r"(?<=\s)-(?=\s)", t):
            problems.append("para %d spaced hyphen: ...%s..."
                            % (i, t[max(0, m.start() - 50):m.end() + 50]))
    assert not problems, "\n  ".join(
        ["Replace these with commas, colons or parentheses:"] + problems)
