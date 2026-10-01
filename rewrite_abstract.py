"""Rewrite the abstract so it agrees with the chapters it summarises.

The drafted abstract ran to 554 words and contradicted the thesis in six places,
most of which an examiner checks first:

  - "finer than any climate projection currently provides" is false; CP4-Africa
    ran at 4.5 km, CORDEX-CORE at 25 km and NEX-GDDP-CMIP6 at 0.25 degrees.
  - "Bias-corrected predictors from three CMIP6 models were mapped onto an
    ERA5-derived target over 1985 to 2010" contradicts Section 3.4.4. Training
    predictors are ERA5; CMIP6 enters only at projection. The design is perfect
    prognosis and is now named as such.
  - "Model selection turned not on accuracy" is false: the deployed model has the
    lowest error of the four, so accuracy alone selects it. The screen removed the
    random forest, which the stated criterion never chose.
  - "inverts the separation" is false: the separation collapses towards zero and
    never changes sign.
  - the architecture and GCM variance shares were the superseded 26.6 and 5.1 per
    cent, computed with a monthly RMSE standing in for a 25-year term.
  - "published as an open repository" is not yet true.

It also omitted the finding that now leads the thesis: a per-cell linear
regression matches the best architecture. Figures are read from the evaluation
CSVs rather than typed, so the abstract cannot drift from Chapter 4 again.

The abstract carries no citation fields, so its runs can be rewritten; the count
is checked anyway.
"""

import os
import shutil
import sys

import docx
import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")


def numbers():
    t = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")
    b = pd.read_csv(os.path.join(EVAL, "baselines.csv")).set_index("baseline")
    u = pd.read_csv(os.path.join(EVAL, "uncertainty_decomposition_summary.csv"))
    u = u[u.model == "xgb"].set_index("period").loc["long_term_2076_2100"]
    p = pd.read_csv(os.path.join(EVAL, "projection_baselines.csv"))
    pl = p[(p.scenario == "ssp585") & (p.horizon == "long_term_2076_2100")]
    inf = pd.read_csv(os.path.join(EVAL, "information_content.csv")).set_index("field")
    rob = pd.read_csv(os.path.join(EVAL, "suitability_robustness.csv")).iloc[0]
    sch = pd.read_csv(os.path.join(EVAL, "suitability_schemes.csv"))

    from zimbabwe_mask import zimbabwe_mask
    lay = xr.open_dataset(os.path.join(ROOT, "data/processed/suitability/criterion_layers.nc"))
    g = lay.ghi_present_sarah.values
    keep = zimbabwe_mask(g.shape) & np.isfinite(g)
    gv = g[keep]

    return dict(
        xgb=t.loc["XGBoost", "RMSE_zw"], ss=t.loc["XGBoost", "SS vs climatology_zw"],
        cnn=t.loc["CNN", "RMSE_zw"], unet=t.loc["U-Net", "RMSE_zw"],
        rf=t.loc["Random Forest", "RMSE_zw"],
        ols=b.loc["Linear regression (OLS)", "RMSE_zw"],
        pv=inf.loc["GHI target (time-mean)", "pct_variance_below_0.25deg"],
        arch=u["pct_var_arch"], gcm=u["pct_var_gcm"],
        ml=pl.ml_change_own_history.mean(), qdm=pl.qdm_change_own_history.mean(),
        nflip=int((~p.same_sign).sum()), ngcm=p.gcm.nunique(),
        robust=int(rob.robust), assessed=int(rob.assessed),
        tier=100 * rob.weight_sensitive / rob.assessed,
        glo=gv.min(), ghi=gv.max(), gsd=gv.std(),
    )


def paragraphs(n):
    return [
        "Zimbabwe's renewable energy targets require siting decisions at a finer spatial "
        "scale than general circulation models supply, which resolve surface solar "
        "radiation at grid spacings of roughly 100 to 250 km. This study trains four "
        "machine learning architectures to map atmospheric predictors onto an "
        "ERA5-derived clear-sky-index target at 0.1 degrees over Zimbabwe under a "
        "perfect-prognosis design, applies them to bias-corrected CMIP6 predictors to "
        "2100, and carries the resulting fields into a multi-criteria suitability "
        "assessment for utility-scale photovoltaic development.",

        "On a withheld 2011 to 2024 record masked to the national boundary, a pixel-wise "
        "gradient-boosted ensemble attains the lowest error of the four at %.2f W/m² and a "
        "skill score of %.2f against a training-period climatology, against %.2f, %.2f and "
        "%.2f W/m² for a convolutional network, a U-Net and a pixel-wise random forest. A "
        "per-cell linear regression on identical predictors reaches %.2f W/m²; paired "
        "year-block bootstrap intervals leave it indistinguishable from the best "
        "architecture and establish it ahead of the other three, so none of the machine "
        "learning models earns its complexity on this target."
        % (n["xgb"], n["ss"], n["cnn"], n["unet"], n["rf"], n["ols"]),

        "A round-trip spectral test accounts for that result. The training target retains "
        "%.3f per cent of its time-mean variance below the 0.25 degree resolution of its "
        "own source, so the product is a physically consistent regridding carrying a "
        "projected climate signal rather than a resolution gain, and a field that smooth "
        "is close to a linear function of its coarse predictors." % n["pv"],

        "Projected annual-mean irradiance rises by %.1f W/m² under SSP5-8.5 by 2076 to 2100 "
        "against each model's own historical run, where bias-correcting and interpolating "
        "the same models' native irradiance gives %.1f W/m², and the downscaled change "
        "reverses the sign of its driver for one of the three models. The choice of "
        "downscaling architecture accounts for %.0f per cent of projection variance at that "
        "horizon against %.0f per cent for the choice of global model, so a study reporting "
        "a single architecture would understate its own uncertainty by the larger term."
        % (n["ml"], n["qdm"], n["arch"], n["gcm"]),

        "The suitability analysis combines irradiance with six biophysical and "
        "infrastructural criteria. Its defensible output is not the five-tier map but the "
        "%d of %d assessed cells that stay highly suitable under every weighting tested, "
        "%.1f per cent of cells changing tier under at least one. Those cells are "
        "distinguished by transmission access rather than by irradiance, whose standard "
        "deviation is %.0f W/m² within a %.0f to %.0f W/m² range. Validation is "
        "out-of-sample in time but uses ERA5 as both training target and reference; "
        "refitting against a satellite retrieval is identified as the first priority for "
        "further work."
        % (n["robust"], n["assessed"], n["tier"], n["gsd"], n["glo"], n["ghi"]),

        "Keywords: perfect prognosis; statistical downscaling; CMIP6; clear-sky index; "
        "solar irradiance; machine learning; uncertainty decomposition; multi-criteria "
        "evaluation; Zimbabwe.",
    ]


def main():
    if os.path.exists(LOCK):
        print("The dissertation is open in Word. Close it first.")
        return 2

    n = numbers()
    new = paragraphs(n)
    d = docx.Document(DOC)
    x0 = d.element.xml
    before = (x0.count("ZOTERO_ITEM"), x0.count('w:fldCharType="begin"'))

    start = next(i for i, p in enumerate(d.paragraphs) if p.text.strip().lower() == "abstract")
    targets = []
    for i in range(start + 1, start + 12):
        p = d.paragraphs[i]
        if p.style.name.startswith("Heading"):
            break
        if p.text.strip():
            targets.append(p)
    print("abstract paragraphs found: %d, replacing with %d" % (len(targets), len(new)))
    if any("ZOTERO_ITEM" in p._p.xml for p in targets):
        raise SystemExit("REFUSED: the abstract carries a citation field")

    old_words = sum(len(p.text.split()) for p in targets)

    # Rewrite in place: keep each paragraph's first run (which carries the
    # formatting) and drop the rest. Spare paragraphs are emptied, not deleted,
    # so no paragraph that might hold a section break is removed.
    for k, p in enumerate(targets):
        text = new[k] if k < len(new) else ""
        if not p.runs:
            p.add_run("")
        p.runs[0].text = text
        for r in list(p.runs[1:]):
            r._r.getparent().remove(r._r)
    for extra in new[len(targets):]:
        para = d.add_paragraph(extra, style=targets[-1].style)
        targets[-1]._p.addnext(para._p)
        pf, src = para.paragraph_format, targets[-1].paragraph_format
        para.alignment = targets[-1].alignment
        pf.space_before, pf.space_after = src.space_before, src.space_after
        pf.line_spacing = src.line_spacing
        if targets[-1].runs:
            for r in para.runs:
                r.font.name = targets[-1].runs[0].font.name
                r.font.size = targets[-1].runs[0].font.size

    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_abstract.docx"))
    d.save(DOC)

    d2 = docx.Document(DOC)
    x1 = d2.element.xml
    after = (x1.count("ZOTERO_ITEM"), x1.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_abstract.docx"), DOC)
        print("CITATION FIELDS CHANGED, restored")
        return 1
    words = sum(len(w.split()) for w in new[:-1])
    print("abstract: %d words -> %d (keywords excluded)" % (old_words, words))
    print("citation fields intact (%d)" % after[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
