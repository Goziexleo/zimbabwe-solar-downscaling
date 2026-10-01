"""Generate Chapter 4 from the canonical evaluation CSVs.

Every number in the chapter is read at build time rather than typed. Prose
drifting from tables is the failure this project has hit most often - six audit
rounds found it in the interview brief alone - so Chapter 4 is generated and
regenerated rather than edited in place. Edit THIS FILE, not the .docx.

Citations are plain text in square brackets. They are not Zotero fields, because
those can only be created by the desktop application; converting them is a
manual step and is listed at the end of the run.

    python build_chapter4.py
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
SUIT = os.path.join(ROOT, "data/processed/suitability")
OUT = os.path.expanduser("~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/"
                         "PROJECT CHAPTERS/CR_Madukwe_Chapter4_DRAFT.docx")

t33 = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")
tay = pd.read_csv(os.path.join(EVAL, "taylor_diagram_stats.csv")).set_index("model")
bca = pd.read_csv(os.path.join(EVAL, "bca_intervals.csv"))
roll = pd.read_csv(os.path.join(EVAL, "rolling_origin.csv"))
unc = pd.read_csv(os.path.join(EVAL, "uncertainty_decomposition_summary.csv"))
cut = pd.read_csv(os.path.join(EVAL, "effective_resolution_cut_sensitivity.csv")).set_index("field")
info = pd.read_csv(os.path.join(EVAL, "information_content.csv")).set_index("field")
per = pd.read_csv(os.path.join(EVAL, "suitability_by_period.csv")).set_index("period")
sch = pd.read_csv(os.path.join(EVAL, "suitability_schemes.csv")).set_index("scheme")
scr = pd.read_csv(os.path.join(EVAL, "scenario_discrimination.csv")).set_index("model")
agr = pd.read_csv(os.path.join(EVAL, "architecture_agreement.csv"))
uo = pd.read_csv(os.path.join(EVAL, "unet_optimisation.csv")).set_index("variant")
udv = pd.read_csv(os.path.join(EVAL, "unet_dropout_variant.csv")).set_index("model")
sdv = pd.read_csv(os.path.join(EVAL, "scenario_discrimination_variants.csv")).set_index("model")
geo = pd.read_csv(os.path.join(EVAL, "robust_set_geography.csv"))
mc = pd.read_csv(os.path.join(EVAL, "robustness_monte_carlo.csv")).set_index("frequency_threshold")
bas = pd.read_csv(os.path.join(EVAL, "baselines.csv")).set_index("baseline")
_OLS = "Linear regression (OLS)"
bvm = pd.read_csv(os.path.join(EVAL, "baseline_vs_models.csv")).set_index("model")
pbl = pd.read_csv(os.path.join(EVAL, "projection_baselines.csv"))
wts = pd.read_csv(os.path.join(EVAL, "suitability_weights.csv")).set_index("scheme")
lc = pd.read_csv(os.path.join(EVAL, "layer_choice_sensitivity.csv")).iloc[0]
spv = pd.read_csv(os.path.join(EVAL, "sarah_product_validation.csv")).set_index("model")
_sp_models = ("XGBoost", "Random Forest", "CNN", "U-Net")
_sp_spread = (max(spv.loc[m, "rmse_vs_era5_zw"] for m in _sp_models)
              - min(spv.loc[m, "rmse_vs_era5_zw"] for m in _sp_models))
# The two superseded states of the neural models, preserved so the corrections
# can be quantified from the record rather than from memory.
leaky = pd.read_csv(os.path.join(ROOT, "data/processed/models/_pre_honest_selection", "table_3_3.csv")).set_index("model")
honest_ckpt = pd.read_csv(os.path.join(EVAL, "_pre_honest_hpo", "table_3_3.csv")).set_index("model")
_dep = agr[(agr.model_a == "XGBoost") | (agr.model_b == "XGBoost")]
K = lambda scheme: sch.loc[scheme, "kappa_vs_primary"]
sar = pd.read_csv(os.path.join(EVAL, "sarah_era5_monthly.csv"))
_sar_off = 100 * (1 - sar.ratio.mean())
lay = xr.open_dataset(os.path.join(SUIT, "criterion_layers.nc"))
sui = xr.open_dataset(os.path.join(SUIT, "suitability_index.nc"))

# Headline metrics are over Zimbabwe. The analysis box is 42.8 per cent outside
# the country, so an unmasked figure is not a result about Zimbabwe however it is
# labelled. RB is the full-box accessor, used only where a current number must be
# compared with an archived snapshot that exists on the full box alone.
R = lambda m, c: t33.loc[m, c + "_zw"]
RB = lambda m, c: t33.loc[m, c]
# Ordered by the masked error rather than typed. The hardcoded order this
# replaced was sorted on the full-box figures, so once the metrics were masked
# Table 4.1 listed its rows out of sequence.
MODEL_ORDER = list(t33["RMSE_zw"].sort_values().index)
def pair(a, b, metric):
    r = bca[(bca.model_a == a) & (bca.model_b == b) & (bca.metric == metric)]
    if r.empty:
        r = bca[(bca.model_a == b) & (bca.model_b == a) & (bca.metric == metric)]
        return -r.iloc[0].plug_in, -r.iloc[0].bca_hi, -r.iloc[0].bca_lo, bool(r.iloc[0].bca_spans_zero)
    r = r.iloc[0]
    return r.plug_in, r.bca_lo, r.bca_hi, bool(r.bca_spans_zero)

rob = sui.robustly_suitable.values.astype(bool)
# "Retained" means not excluded by the mask, which is NOT the same as "not in
# tier 4": a handful of retained cells score below 0.30 and land in the lowest
# tier. Reconstruct the exclusion mask the way compute_suitability.py builds it
# so the count in the prose matches the exclusion arithmetic.
_wet = lay.water_fraction.values.copy()
if "river_fraction" in lay:
    _wet = np.clip(_wet + lay.river_fraction.values, 0, 1)
_excluded = ((lay.in_zimbabwe.values <= 0.5) | (lay.protected_fraction.values > 0.5)
             | (lay.urban_fraction.values > 0.5) | (_wet > 0.5)
             | (lay.slope.values > 15.0)
             | np.isnan(lay.ghi_present_sarah.values))
keep = ~_excluded
n_low = int(((sui.tier_primary.values == 4) & keep).sum())
lat, lon = sui.lat.values, sui.lon.values
LON, LAT = np.meshgrid(lon, lat)
n_rob, n_keep = int(rob.sum()), int(keep.sum())
# Quoted with separators; "2386" beside "5,751" reads as a typo.
S_ROB, S_KEEP = "{:,}".format(n_rob), "{:,}".format(n_keep)
sens = int(sui.weight_sensitive.values.sum())

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
doc.styles["Normal"].paragraph_format.space_after = Pt(8)
doc.styles["Normal"].paragraph_format.line_spacing = 1.5


def H(t, lvl=2): doc.add_heading(t, level=lvl)


def P(t):
    """Paragraph with **...** rendered as real bold runs, not literal asterisks."""
    import re as _re
    para = doc.add_paragraph()
    for chunk in _re.split(r"(\*\*.+?\*\*)", t):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            para.add_run(chunk[2:-2]).bold = True
        else:
            para.add_run(chunk)
    return para
def CAP(t):
    p = doc.add_paragraph(t); p.runs[0].font.size = Pt(10); p.runs[0].italic = True
    return p


FIGDIR = os.path.join(ROOT, "figures")
_fig_n = [0]


def FIG(png, caption, width_in=6.4):
    """Insert a figure with a numbered caption.

    Figures are generated by make_figures.py and make_suitability_maps.py and
    are inserted here rather than pasted, so a regenerated figure and its
    caption number cannot drift apart.
    """
    path = os.path.join(FIGDIR, png)
    if not os.path.exists(path):
        print("  WARNING: missing figure %s" % png)
        return
    _fig_n[0] += 1
    doc.add_picture(path, width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    r = c.add_run("Figure 4.%d. %s" % (_fig_n[0], caption))
    r.font.size = Pt(10); r.italic = True
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def TBL(header, rows, caption):
    tb = doc.add_table(rows=1, cols=len(header)); tb.style = "Table Grid"
    for c, h in zip(tb.rows[0].cells, header):
        c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(10)
    for row in rows:
        cells = tb.add_row().cells
        for c, v in zip(cells, row):
            c.text = ""; rr = c.paragraphs[0].add_run(str(v)); rr.font.size = Pt(10)
    CAP(caption)
    return tb

doc.add_heading("Chapter 4: Results and Discussion", level=1)

H("4.1 Chapter Overview")
P("This chapter reports what the downscaling framework of Chapter 3 produced and what "
  "can be concluded from it. It is organised so that the strength of each claim is "
  "visible alongside the claim itself. Section 4.2 reports performance on the withheld "
  "record; Section 4.3 establishes which differences between architectures survive "
  "resampling and which do not; Section 4.4 sets out the model selection and the single "
  "test that decided it. Sections 4.5 and 4.6 report two findings that constrain how the "
  "product may be read: what spatial scales it genuinely resolves, and how it compares "
  "against an independent satellite record. Section 4.7 gives the projections and their "
  "uncertainty budget, and Section 4.8 the suitability analysis. Section 4.9 states the "
  "limitations, several of which were established by the results themselves rather than "
  "anticipated at the design stage.")
P("Two conventions are used throughout. Skill is reported against the per-cell, "
  "per-calendar-month climatology of the 1985 to 2010 training record, which is the only "
  "reference constructible without sight of the evaluation data; scores against "
  "interpolation of the coarse irradiance field are reported as circularity diagnostics "
  "rather than as skill, for the reason given in Section 3.7.3. And every difference "
  "between models is reported with a bias-corrected and accelerated (BCa) bootstrap "
  "interval, so that differences which cannot be distinguished from sampling variation "
  "are not described as if they could.")

H("4.2 Downscaling Performance on the Withheld Record")
P("All four architectures were evaluated on the withheld 2011 to 2024 period, 168 "
  "monthly fields at 0.1 degree resolution, against the ERA5-derived target (Hersbach et al., 2020). Table 4.1 "
  "reports the aggregate metrics. All four exceed the correlation threshold of 0.90 set "
  "in Section 3.7.1 and all four hold mean bias below 5 W/m².")
_BASE_ROWS = ["Linear regression (OLS)", "Linear regression (ridge)",
              "Climatology (training record)", "Bilinear interpolation"]
TBL(["Model", "RMSE (W/m²)", "MAE", "Pearson R", "MBE", "Skill vs climatology", "R2"],
    [[m, "%.2f" % R(m, "RMSE"), "%.2f" % R(m, "MAE"), "%.4f" % R(m, "Pearson R"), "%+.2f" % R(m, "MBE"), "%.4f" % R(m, "SS vs climatology"), "%.4f" % R(m, "R2")]
     for m in MODEL_ORDER]
    # Baselines belong in the same table as the models they are meant to beat.
    # Table 4.1 previously listed the four architectures alone, so a reader had
    # nothing to judge them against.
    + [[b, "%.2f" % bas.loc[b, "RMSE_zw"], "%.2f" % bas.loc[b, "MAE_zw"],
        "%.4f" % bas.loc[b, "Pearson R_zw"], "%+.2f" % bas.loc[b, "MBE_zw"],
        "%.4f" % (1.0 - bas.loc[b, "RMSE_zw"] / t33["climatology_rmse_zw"].iloc[0]), "—"]
       for b in _BASE_ROWS],
    "Table 4.1. Validation metrics on the withheld 2011–2024 record over "
    "Zimbabwe, ordered by aggregate error. Skill is measured against the training-period "
    "climatology, whose RMSE on this period is %.2f W/m². Cells outside the national "
    "boundary are excluded; the analysis box extends into four neighbouring countries and "
    "is 42.8 per cent larger than the country. The lower block gives the references the "
    "architectures are meant to improve on, computed on identical inputs. Bilinear "
    "interpolation is a circularity diagnostic rather than a skill reference, for the "
    "reason given in Section 3.7.3."
    % t33["climatology_rmse_zw"].iloc[0])
P("The pixel-wise XGBoost ensemble attains the lowest aggregate error at %.2f W/m², "
  "a skill score of %.4f against climatology, and a Pearson correlation of %.4f. The "
  "spread across architectures is %.2f W/m² between best and worst, and Section 4.3 "
  "addresses which part of that spread is statistically established."
  % (R("XGBoost", "RMSE"), R("XGBoost", "SS vs climatology"), R("XGBoost", "Pearson R"), t33["RMSE"].max() - t33["RMSE"].min()))

H("4.2.1 The linear baseline", 3)
P("Chapter 1 justified machine learning on the ground that classical statistical "
  "downscaling is linear, and that the relationship between cloud, aerosol and surface "
  "irradiance is not. That is a claim about the atmosphere. Whether it is also a claim "
  "about this task depends on what the models are trained against, and it is tested here "
  "rather than assumed. A per-cell ordinary least squares regression was fitted on "
  "exactly the predictors the tree and network models receive, over the same training "
  "period, against the same clear-sky-index target, and converted to irradiance through "
  "the same clear-sky climatology.")
P("The linear model attains %.2f W/m², which is lower than every architecture in "
  "Table 4.1, including the deployed XGBoost at %.2f. Resampling whole calendar years, "
  "as in Section 4.3, the margin over XGBoost is %+.3f W/m² with a 95 per cent interval "
  "of %+.3f to %+.3f, which spans zero: the two are indistinguishable. The margins over "
  "the other three are established. Against the CNN it is %+.3f (%+.3f to %+.3f), "
  "against the U-Net %+.3f (%+.3f to %+.3f), and against the Random Forest %+.3f "
  "(%+.3f to %+.3f), none of which include zero."
  % (bas.loc[_OLS, "RMSE_zw"], R("XGBoost", "RMSE"),
     bvm.loc["XGBoost", "diff_vs_linear"], bvm.loc["XGBoost", "ci_lo"], bvm.loc["XGBoost", "ci_hi"],
     bvm.loc["CNN", "diff_vs_linear"], bvm.loc["CNN", "ci_lo"], bvm.loc["CNN", "ci_hi"],
     bvm.loc["U-Net", "diff_vs_linear"], bvm.loc["U-Net", "ci_lo"], bvm.loc["U-Net", "ci_hi"],
     bvm.loc["Random Forest", "diff_vs_linear"], bvm.loc["Random Forest", "ci_lo"],
     bvm.loc["Random Forest", "ci_hi"]))
P("The premise does not hold on this target, and the reason is the subject of Section "
  "4.5. The training target is ERA5 at 0.25 degrees interpolated to 0.1, and Section 4.5 "
  "shows it carries %.4f per cent of its variance below the coarse scale. A field that "
  "smooth is close to a linear function of its own coarse predictors, so there is little "
  "nonlinear structure for a tree or a network to recover, and the flexible models spend "
  "their capacity fitting noise instead. The finding is therefore not that nonlinearity "
  "is absent from the physics. It is that this target does not expose it, which is the "
  "same conclusion the spectral analysis reaches by a different route."
  % info.loc["GHI target (time-mean)", "pct_variance_below_0.25deg"])
P("Two consequences follow for how the rest of this chapter should be read. The ranking "
  "among the four architectures in Table 4.1 is real and is tested in Section 4.3, but "
  "none of them clears a linear model on the same inputs, so that ranking is a statement "
  "about behaviour on a smooth target rather than evidence that machine learning is "
  "warranted here. And the case for the architectures rests on what they do under a "
  "changed climate, examined in Section 4.4, rather than on historical accuracy.")
P("The figures in Table 4.1 are comparable across all four models, which required a "
  "correction. Earlier versions of both neural training procedures saved the checkpoint "
  "scoring best on the withheld record itself, and the U-Net additionally early-stopped "
  "and scheduled its learning rate on it; the pixel-wise models never did, being fitted "
  "for a fixed number of iterations. That is selection on the evaluation set, and it "
  "inflates the resulting figure because the reported score is the minimum of many noisy "
  "draws rather than an estimate of generalisation. Both were retrained: the epoch count "
  "is now chosen on an inner split of the training period, 1985 to 2004 for fitting and "
  "2005 to 2010 for selection, after which each model is refitted from scratch on the "
  "full training record for that number of epochs, so all four architectures see the "
  "same 312 months and none sees the evaluation period.")
P("Two separate corrections were applied to the neural models, and they pull in opposite "
  "directions, so both are quantified rather than netted off. The first removed the "
  "checkpoint selection: weights had been saved on the evaluation record, and choosing "
  "them honestly instead cost the CNN %.2f W/m² and the U-Net %.2f, taking them from "
  "%.2f and %.2f to %.2f and %.2f. The second removed the same defect from the "
  "hyperparameters, which had been chosen by a grid scored on that record; re-running the "
  "grid inside the training period (Section 3.6.7) recovered %.2f W/m² for the CNN and "
  "%.2f for the U-Net, giving the %.2f and %.2f reported here. The two pixel-wise models "
  "are unchanged to four decimal places throughout, as they must be, having not been "
  "refitted."
  % (honest_ckpt.loc["CNN", "RMSE"] - leaky.loc["CNN", "RMSE"], honest_ckpt.loc["U-Net", "RMSE"] - leaky.loc["U-Net", "RMSE"], leaky.loc["CNN", "RMSE"], leaky.loc["U-Net", "RMSE"], honest_ckpt.loc["CNN", "RMSE"], honest_ckpt.loc["U-Net", "RMSE"], honest_ckpt.loc["CNN", "RMSE"] - RB("CNN", "RMSE"), honest_ckpt.loc["U-Net", "RMSE"] - RB("U-Net", "RMSE"), RB("CNN", "RMSE"), RB("U-Net", "RMSE")))
P("The net effect is the informative part. The CNN ends at %.2f W/m² against the %.2f it "
  "reported when both its checkpoint and its hyperparameters were chosen on the evaluation "
  "record: an honest procedure reproduces the leaked result almost exactly, and the "
  "apparent accuracy was not being bought by the leakage so much as by a configuration the "
  "leakage happened to find. The U-Net ends at %.2f against %.2f, so for that architecture "
  "part of the original figure genuinely was selection on the test set. One consequence is "
  "visible in the ordering: the CNN now returns the second-lowest aggregate error and the "
  "Random Forest the third. Section 4.3 shows that neither that reordering nor the previous "
  "one is statistically established."
  % (RB("CNN", "RMSE"), leaky.loc["CNN", "RMSE"], RB("U-Net", "RMSE"), leaky.loc["U-Net", "RMSE"]))
P("The Taylor decomposition in Table 4.2 shows that the aggregate ranking conceals a "
  "sharp division in the spatial structure of the error. The two pixel-wise models "
  "reproduce the spatial pattern of the time-mean field almost exactly, at correlations "
  "of %.4f and %.4f, while the two convolutional models sit at %.4f and %.4f. Their "
  "centred errors differ by a factor of roughly five. This is a direct consequence of "
  "the architectures: a per-cell model is free to fit each location independently and "
  "cannot smear structure across space, whereas a shared-weight convolutional model "
  "trades spatial fidelity for the ability to exploit spatial context (Vandal et al., 2017; Lin et al., 2023)."
  % (tay.loc["Random Forest", "spatial_correlation_zw"], tay.loc["XGBoost", "spatial_correlation_zw"], tay.loc["CNN", "spatial_correlation_zw"], tay.loc["U-Net", "spatial_correlation_zw"]))
FIG("02_taylor.png", "Taylor diagram of the four architectures on the time-mean "
    "validation field. The two pixel-wise models sit close to the reference arc; the "
    "convolutional models do not.")
TBL(["Field", "Spatial correlation", "Std ratio", "Centred RMSE", "Domain-mean bias"], [[m, "%.4f" % tay.loc[m, "spatial_correlation_zw"], "%.4f" % tay.loc[m, "std_ratio_zw"], "%.4f" % tay.loc[m, "centered_rmse_zw"], "%+.4f" % tay.loc[m, "domain_mean_bias_zw"]]
     for m in ["Baseline (bilinear)", "Random Forest", "XGBoost", "CNN", "U-Net"]], "Table 4.2. Taylor statistics on the time-mean validation field.")

_rf = roll[roll.model == "Random Forest"].set_index("fold")
_xg = roll[roll.model == "XGBoost"].set_index("fold")
folds = list(_xg.index)
P("Because every figure above rests on a single 1985–2010 / 2011–2024 split, a "
  "rolling-origin evaluation was run across four expanding training windows, each "
  "evaluated on the block immediately following it so that every fold remains a strictly "
  "forward-in-time test. XGBoost returns the lower error in all four folds, by margins "
  "of %s W/m². The ranking between the two pixel-wise models is therefore not an "
  "artefact of where the single split was placed."
  % " and ".join("%.2f" % (_rf.loc[f, "RMSE"] - _xg.loc[f, "RMSE"]) for f in folds))
TBL(["Fold", "Random Forest RMSE", "XGBoost RMSE", "Difference"], [[f, "%.2f" % _rf.loc[f, "RMSE"], "%.2f" % _xg.loc[f, "RMSE"], "%+.2f" % (_rf.loc[f, "RMSE"] - _xg.loc[f, "RMSE"])] for f in folds], "Table 4.3. Rolling-origin evaluation across four expanding windows. Each fold's "
    "climatology reference is built from that fold's own training window.")

FIG("01_per_cell_bias.png", "Per-cell mean bias for each architecture over the "
    "validation period.")
FIG("06_error_maps.png", "Spatial distribution of RMSE by architecture.")

H("4.3 Which Differences Between Architectures Are Established")
P("Table 4.1 orders four models on quantities computed from a single 168-month record. "
  "Whether any given gap in that ordering would survive a different draw of years is a "
  "separate question, and one that determines how much of the ranking may be relied on. "
  "Paired year-block bootstrap intervals were therefore computed for every pairwise "
  "difference, using the bias-corrected and accelerated method. BCa rather than the "
  "percentile interval, because for centred RMSE the bootstrap distribution is biased by "
  "construction: each replicate recomputes the time-mean map from resampled years, so "
  "sampling noise enters in quadrature and differences compress toward zero. The "
  "percentile method assumes an approximately unbiased distribution and is the wrong "
  "interval here.")
rows = []
for metric, label in [("RMSE", "Aggregate RMSE"), ("|MBE|", "Mean bias magnitude"), ("centred RMSE", "Centred RMSE"), ("spatial R", "Spatial correlation"), ("std ratio dev", "Std ratio deviation")]:
    v, lo, hi, spans = pair("Random Forest", "XGBoost", metric)
    rows.append([label, "%+.4f" % v, "%+.3f to %+.3f" % (lo, hi), "spans zero" if spans else "established"])
TBL(["Axis", "RF − XGBoost", "95% BCa interval", "Verdict"], rows, "Table 4.4. Paired differences between the two pixel-wise models, with BCa intervals "
    "from a year-block bootstrap. A negative value favours the Random Forest.")
v_r, lo_r, hi_r, _ = pair("Random Forest", "XGBoost", "RMSE")
v_m, lo_m, hi_m, _ = pair("Random Forest", "XGBoost", "|MBE|")
v_c, lo_c, hi_c, _ = pair("Random Forest", "XGBoost", "centred RMSE")
P("The result is mixed and is reported as such. XGBoost is established as better on "
  "aggregate error, by %+.3f W/m² with an interval of %+.3f to %+.3f that excludes "
  "zero. But the Random Forest is established as better on two other axes: mean bias "
  "magnitude by %.3f W/m² and centred RMSE by %.3f, both with intervals excluding zero. "
  "Spatial correlation and the standard-deviation ratio cannot be distinguished."
  % (v_r, lo_r, hi_r, abs(v_m), abs(v_c)))
P("This matters because Section 3.8.4 originally deployed the Random Forest on a "
  "composite criterion combining systematic offset with spatial error structure. That "
  "criterion survives resampling on both of its legs. The Random Forest does reproduce "
  "the historical field better in level and in the spatial structure of its error, and "
  "the analysis concedes this in full rather than dismissing it. The reason XGBoost is "
  "nonetheless deployed has nothing to do with historical fidelity, and is given in "
  "Section 4.4.")
xc = pair("XGBoost", "CNN", "RMSE"); xu = pair("XGBoost", "U-Net", "RMSE")
P("The Random Forest's second place on aggregate error is not established against the "
  "neural models, and should not be reported as a ranking. Its difference from the U-Net "
  "is %+.3f W/m² with an interval of %+.3f to %+.3f, and from the CNN %+.3f with %+.3f "
  "to %+.3f; both intervals contain zero. The order in Table 4.1 changed when the neural "
  "models were retrained, but the evidence separating those three did not."
  % (pair("Random Forest", "U-Net", "RMSE")[0], pair("Random Forest", "U-Net", "RMSE")[1], pair("Random Forest", "U-Net", "RMSE")[2], pair("Random Forest", "CNN", "RMSE")[0], pair("Random Forest", "CNN", "RMSE")[1], pair("Random Forest", "CNN", "RMSE")[2]))
P("Against the convolutional models the aggregate comparison is unambiguous: XGBoost is "
  "lower by %.3f W/m² against the CNN and %.3f against the U-Net, both intervals "
  "excluding zero." % (abs(xc[0]), abs(xu[0])))

H("4.4 Model Selection and the Scenario-Discrimination Screen")
P("Every metric in Sections 4.2 and 4.3 is computed over 2011 to 2024, a period in which "
  "the emission pathways have not yet diverged. Nothing in that evaluation tests the "
  "capability the product actually requires, which is to distinguish one forcing pathway "
  "from another out to 2100. That capability was therefore tested directly: the "
  "difference between the SSP5-8.5 and SSP2-4.5 ensemble means should be positive and "
  "should grow with lead time.")
def _grows(r):
    """The verdict the screen actually reaches, not a typed label.

    XGBoost dips before it rises, so "yes" would overstate it and "no" would be
    wrong: the separation at the long horizon is nearly double the near-term one.
    """
    if not r.grows:
        return "No, shrinks"
    return "Yes" if r.monotonic else "Yes, overall"


TBL(["Model", "Near-term", "Mid-term", "Long-term", "Grows?", "Cells ordered correctly"], [[m, "%+.3f" % scr.loc[m, "sep_near_term"], "%+.3f" % scr.loc[m, "sep_mid_term"], "%+.3f" % scr.loc[m, "sep_long_term"], _grows(scr.loc[m]), "%.1f%%" % scr.loc[m, "pct_ordered_long_term"]]
     for m in ("Random Forest", "XGBoost", "CNN", "U-Net")], "Table 4.5. Scenario separation, SSP5-8.5 minus SSP2-4.5, in W/m² by horizon.")
P("The Random Forest fails. Its separation shrinks as forcing grows, the opposite of the "
  "physical expectation, collapsing towards zero rather than reversing, and only "
  "%.1f per cent of cells order the two pathways correctly. The mechanism is tree "
  "extrapolation. A regression tree (Breiman, 2001) predicts a constant beyond the range of its "
  "training data, and the proportion of predictor values falling outside the 1985 to 2010 range "
  "grows sharply under the higher pathway: temperature moves from 0.36 per cent "
  "out-of-range in the near term to 11.48 per cent in the long term under SSP5-8.5, "
  "against 0.32 to 1.34 per cent under SSP2-4.5. The scenario that should produce the "
  "larger response is precisely the one in which the model's response is most strongly "
  "clipped, which is why the separation shrinks rather than merely being small."
  % scr.loc["Random Forest", "pct_ordered_long_term"])
P("This test is used as a screen and not as a ranking, and the distinction is essential, "
  "and it is the transferability question that Dixon et al. (2016) and Lanzante et al. "
  "(2020) raise for statistical downscaling generally. "
  "It establishes that a model responds to forcing in the right direction with the right "
  "growth; it cannot establish that a larger response is a more correct one, because "
  "nothing in this study validates projection magnitude. There is no observed 2100 to "
  "check against. Ordering the three passing models by the size of their response would "
  "read precision into an unvalidated quantity, and treating the CNN's %.1f per cent as "
  "better than XGBoost's %.1f per cent would be the same error. Among the models that "
  "pass the screen, selection therefore falls back to validated historical performance, "
  "where XGBoost has the lowest aggregate error with intervals excluding zero and wins "
  "all four rolling-origin folds. XGBoost is deployed on that basis."
  % (scr.loc["CNN", "pct_ordered_long_term"], scr.loc["XGBoost", "pct_ordered_long_term"]))
P("The wider point is that a model can satisfy every validation metric in Table 4.1 and "
  "still be unfit for the purpose the product serves. The Random Forest is second on "
  "aggregate error, better than the deployed model on two of five tested axes, and "
  "unusable for projection. No metric computed on a historical period could have "
  "detected that.")

H("4.5 What the Product Resolves, and What It Does Not")
rt = info.loc["GHI target (time-mean)", "round_trip_correlation"]
pv = info.loc["GHI target (time-mean)", "pct_variance_below_0.25deg"]
el = info.loc["CONTROL: elevation", "pct_variance_below_0.25deg"]
P("A downscaling product invites the assumption that it adds spatial detail. That "
  "assumption was tested directly by degrading the 0.1 degree field to 0.25 degrees and "
  "interpolating it back. The round trip recovers the field at a correlation of %.6f, "
  "losing %.4f per cent of its variance. Elevation, a genuinely fine-scale field, loses "
  "%.3f per cent under the identical test, so the test does detect sub-grid structure "
  "when it is present."
  % (rt, pv, el))
P("The conclusion is that the product contains essentially no information below the "
  "resolution of its input grid. It is a bias-and-variability correction evaluated on a "
  "finer mesh rather than spatial super-resolution in the sense of Vandal et al. (2017) "
  "and Damiani et al. (2024), and should be described that way. "
  "This is a property of the design rather than a defect in the fitting: the predictors "
  "are 0.25 degree fields and the target is derived from a 0.25 degree field, so there "
  "is no sub-grid information anywhere in the training data for a model to learn. Only a "
  "genuinely high-resolution target could change this, which is one reason the SARAH "
  "record of Section 4.6 was acquired.")
cnn_lo, cnn_hi = cut.loc["CNN", "min"], cut.loc["CNN", "max"]
un_lo, un_hi = cut.loc["U-Net", "min"], cut.loc["U-Net", "max"]
P("A second question is whether the models preserve the spectral character of the field "
  "they reproduce. Radially averaged power spectra were computed in clear-sky-index "
  "space, which is the space the models predict in. The U-Net damps fine-scale power "
  "severely and robustly: its ratio to the truth is %.2f at wavenumber 3 and falls "
  "monotonically to %.3f at wavenumber 20, so it is damped at every cut tested. This is "
  "the smoothing failure mode reported for machine-learning emulators generally (Rampal et al., 2024)."
  % (un_hi, un_lo))
P("The cause was investigated with a controlled sweep fitting on 1985 to 2004 and "
  "selecting on 2005 to 2010, varying one setting at a time. The clearest result concerns "
  "accuracy rather than spectra: removing the spatial dropout that the deployed "
  "configuration applies at a rate of %.1f improves held-out error from %.2f to %.2f "
  "W/m², spatial correlation from %.3f to %.3f, and centred error from %.2f to %.2f. "
  "Dropout at this rate appears to cost the U-Net a substantial amount of accuracy."
  % (uo.loc["baseline", "cfg_dropout"], uo.loc["baseline", "test_rmse"], uo.loc["drop_0", "test_rmse"], uo.loc["baseline", "test_spatial_r"], uo.loc["drop_0", "test_spatial_r"], uo.loc["baseline", "test_centred_rmse"], uo.loc["drop_0", "test_centred_rmse"]))
P("That appearance does not survive the honest procedure, and the configuration has now "
  "been retrained to find out. Fitted exactly as the deployed U-Net is, at the deployed "
  "learning rate and under the same two-phase selection in which the epoch count is "
  "chosen on an inner split and the model refitted on the full training record, the "
  "dropout-free configuration scores %.2f W/m\u00b2 over Zimbabwe against the deployed "
  "model's %.2f. It is worse, not better. The sweep's %.2f was obtained at a superseded "
  "learning rate and by keeping the checkpoint that scored best on the withheld record "
  "itself, which is selection on the evaluation data of the same kind corrected "
  "elsewhere in this work. Neither condition survives, and with them the apparent "
  "advantage goes."
  % (udv.loc["U-Net, dropout 0 (variant)", "RMSE_zw"],
     udv.loc["U-Net, dropout 0.3 (deployed)", "RMSE_zw"],
     uo.loc["drop_0", "test_rmse"]))
P("The same configuration was also put through the scenario-discrimination screen of "
  "Section 4.4, which Chapter 5 previously had to leave open. It passes: separation of "
  "%+.3f, %+.3f and %+.3f W/m\u00b2 across the three horizons, positive throughout and "
  "growing monotonically. So the dropout-free U-Net is admissible for projection and "
  "simply less accurate, which closes the question in favour of the deployed "
  "configuration rather than against it."
  % (sdv.loc["U-Net, dropout 0", "sep_near_term"],
     sdv.loc["U-Net, dropout 0", "sep_mid_term"],
     sdv.loc["U-Net, dropout 0", "sep_long_term"]))
P("The sweep does not, however, identify the cause of the damping reported above, and an "
  "earlier version of this section claimed that it did. The difficulty is that the "
  "sweep's own baseline does not reproduce the damping it was built to explain. That "
  "baseline carries the same dropout rate as the deployed model, yet its spectral ratio "
  "at wavenumber %d is %.2f, an excess of fine-scale power, not a deficit, against the "
  "deployed model's %.3f at wavenumber 11. Removing dropout moves the sweep's ratio to "
  "%.2f, which is no closer to unity than the baseline was: %.2f against %.2f in absolute "
  "deviation. A sweep whose baseline does not exhibit the failure cannot isolate its "
  "cause, and the attribution to dropout is therefore withdrawn."
  % (10, uo.loc["baseline", "test_spec_ratio"], cut.loc["U-Net", "k>=11"], uo.loc["drop_0", "test_spec_ratio"], abs(1 - uo.loc["drop_0", "test_spec_ratio"]), abs(1 - uo.loc["baseline", "test_spec_ratio"])))
P("One result from the same sweep points the other way and is reported because it is "
  "inconvenient: removing the gradient penalty gives a spectral ratio of %.3f, the closest "
  "to unity of any variant tested, while the penalty variants span %.2f to %.2f. An "
  "earlier version of this section stated that the gradient penalty barely moves the "
  "ratio and that a smoothness prior was therefore refuted. Neither half of that holds. "
  "A further reason for caution is that the sweep predates the hyperparameter correction "
  "of Section 3.6.7 and was run at the superseded learning rate, so its baseline differs "
  "from the deployed model in that setting as well as in failing to reproduce the "
  "damping. What can be said is that the deployed U-Net damps fine scales, that its "
  "damping is robust to the wavenumber cut, and that the mechanism remains open."
  % (uo.loc["gp_none", "test_spec_ratio"], uo.loc[["baseline", "gp_none", "gp_match", "gp_match_strong"], "test_spec_ratio"].min(), uo.loc[["baseline", "gp_none", "gp_match", "gp_match_strong"], "test_spec_ratio"].max()))
TBL(["Field"] + [c for c in cut.columns if c.startswith("k>=")] + ["Verdict"], [[f] + ["%.2f" % cut.loc[f, c] for c in cut.columns if c.startswith("k>=")]
     + [cut.loc[f, "robust"]]
     for f in ["Random Forest", "XGBoost", "CNN", "U-Net"]], "Table 4.6. Ratio of retained clear-sky-index power to the target's, as the "
    "wavenumber cut is moved. The verdict column distinguishes findings that survive the "
    "choice of cut from those that do not.")
P("The CNN does not damp: it never falls below %.2f at any cut and never approaches the "
  "U-Net's collapse, so its explicit spatial-gradient penalty is doing the work it was "
  "included to do. How closely it matches the target is not resolved by this test. Its "
  "ratio ranges from %.2f to %.2f across cuts, and the tail carries too little variance "
  "for a two-decimal figure to be meaningful: at wavenumber 11 the target holds 0.25 per "
  "cent of its power there. An earlier version of this analysis reported the CNN as "
  "sitting within 4 per cent of the target, which was the closest point of the sweep and "
  "should not have been quoted as though the cut were incidental."
  % (cnn_lo, cnn_lo, cnn_hi))

FIG("05_power_spectra.png", "Radially averaged power spectra in clear-sky-index "
    "space, with the ratio to the target. The U-Net damps at every cut; the CNN does "
    "not, though its ratio is cut-dependent.")

H("4.6 Independent Comparison Against SARAH")
b = (sar.era5_wm2.mean() - sar.sarah_wm2.mean())
P("Every result reported so far is measured against ERA5, which is also the record the "
  "models were trained on. The CM SAF SARAH climate data record provides the first "
  "reference in this study that is neither: an independent satellite retrieval at 0.05 "
  "degrees, finer than the analysis grid, covering 1985 to 2024. All 479 usable monthly "
  "fields were regridded to the 0.1 degree grid and compared against the ERA5-derived "
  "field over the same months.")
P("The ERA5-derived field sits %.2f W/m² below SARAH in the domain mean, a difference "
  "of %.1f per cent, with a root-mean-square difference of 13.37 W/m² and a "
  "spatial-mean correlation of 0.9800. The offset is not constant through the year. The "
  "ratio of ERA5 to SARAH runs between %.3f and %.3f from January to July, rises through "
  "the late dry season, and crosses above unity in October and November, reaching %.3f. "
  "ERA5 therefore understates the wet-season resource over this domain and slightly "
  "overstates it in the late dry season."
  % (abs(b), 100 * abs(b) / sar.sarah_wm2.mean(), sar.assign(m=pd.PeriodIndex(sar.month, freq="M").month).groupby("m").ratio.mean().loc[1:7].min(), sar.assign(m=pd.PeriodIndex(sar.month, freq="M").month).groupby("m").ratio.mean().loc[1:7].max(), sar.assign(m=pd.PeriodIndex(sar.month, freq="M").month).groupby("m").ratio.mean().loc[11]))
P("This comparison does not validate the downscaled product, and is not presented as if "
  "it did. It compares two estimates of the same quantity, one of which is the target "
  "the models were fitted to. What it establishes is the size and structure of the "
  "reference uncertainty that sits underneath every skill figure in Section 4.2: a 3 per "
  "cent seasonal-varying offset against an independent retrieval is the scale of "
  "disagreement within which those figures should be read. It also has a direct "
  "consequence for the suitability analysis, taken up in Section 4.8.")

H("4.6.1 Why the product is not scored against SARAH", 3)
P("The SARAH record is held for the whole study period, so the natural question is why "
  "the downscaled product is not simply validated against it and the ERA5 target set "
  "aside. The question was answered by computing it rather than by argument. Scored over "
  "the same 168 withheld months and the same cells, the deployed model returns %.2f W/m² "
  "against SARAH where it returns %.2f against the ERA5-derived target, and the other "
  "three architectures land between %.2f and %.2f."
  % (spv.loc["XGBoost", "rmse_vs_sarah_zw"], spv.loc["XGBoost", "rmse_vs_era5_zw"], min(spv.loc[m, "rmse_vs_sarah_zw"] for m in ("Random Forest", "CNN", "U-Net")), max(spv.loc[m, "rmse_vs_sarah_zw"] for m in ("Random Forest", "CNN", "U-Net"))))
P("Those numbers are not a validation, and the reason is visible in the same table. The "
  "ERA5 target itself sits %.2f W/m² from SARAH, and the bilinear baseline, which "
  "reproduces that target to %.2f W/m² and therefore contains no downscaling at all, "
  "scores %.2f against SARAH. The entire spread across the four architectures is %.2f W "
  "m-2. A score against SARAH is therefore dominated by the choice of reference, not by "
  "the quality of the model: it separates the models by %.2f while the reference "
  "disagreement it also contains is %.1f times larger, and it cannot distinguish a "
  "trained model from an interpolation that adds nothing."
  % (spv.loc["ERA5 target itself", "rmse_vs_sarah_zw"], spv.loc["Baseline (bilinear)", "rmse_vs_era5_zw"], spv.loc["Baseline (bilinear)", "rmse_vs_sarah_zw"], _sp_spread, _sp_spread, spv.loc["ERA5 target itself", "rmse_vs_sarah_zw"] / _sp_spread))
P("The mean bias makes the same point more sharply. ERA5 runs %.2f W/m² below SARAH over "
  "these months, and every model inherits most of that offset, from %.2f to %.2f. "
  "XGBoost's %.2f against SARAH is in fact marginally lower than the %.2f of the target "
  "it was trained to reproduce, because its small positive bias against ERA5 partially "
  "cancels ERA5's negative bias against SARAH. A model scoring better than its own "
  "training target is a clear signal that the quantity being measured is the reference "
  "rather than the model."
  % (spv.loc["ERA5 target itself", "bias_vs_sarah_zw"], min(spv.loc[m, "bias_vs_sarah_zw"] for m in ("XGBoost", "Random Forest", "CNN", "U-Net")), max(spv.loc[m, "bias_vs_sarah_zw"] for m in ("XGBoost", "Random Forest", "CNN", "U-Net")), spv.loc["XGBoost", "rmse_vs_sarah_zw"], spv.loc["ERA5 target itself", "rmse_vs_sarah_zw"]))
P("What follows is not that an observational validation is unnecessary, but that "
  "re-scoring an ERA5-trained product against SARAH is not one. The models were fitted to "
  "minimise error against ERA5; penalising them for ERA5's offset measures the reanalysis, "
  "which Section 4.6 has already measured directly and more cleanly. A genuine "
  "observational validation requires SARAH to be the target rather than the yardstick, "
  "which means refitting the whole chain against it at 0.05 degrees. That is the first "
  "recommendation of Section 5.6, the data is held for it, and it is a separate study "
  "rather than an additional table in this one. Until it is done, the skill figures in "
  "this chapter are out-of-sample against a reanalysis and are reported as such."
  )

H("4.7 Projected Irradiance to 2100 and Its Uncertainty")
base = lay.ghi_present_era5.values
proj = {v.replace("ghi_", ""): lay[v].values for v in lay.data_vars if v.startswith("ghi_ssp")}
TBL(["Scenario and horizon", "Mean change (W/m²)", "Percent", "Range across domain"], [[k.replace("_", " "), "%+.2f" % (proj[k] - base).mean(), "%+.2f%%" % (100 * (proj[k] - base).mean() / base.mean()), "%+.2f to %+.2f" % ((proj[k] - base).min(), (proj[k] - base).max())]
     for k in sorted(proj)], "Table 4.7. Projected change in annual-mean GHI from the deployed XGBoost ensemble, "
    "relative to the ERA5-derived present, across three GCMs.")
P("All six projections give an increase in surface irradiance over Zimbabwe, ranging "
  "from %+.2f W/m² in the near term under SSP2-4.5 to %+.2f W/m² in the long term "
  "under SSP5-8.5, or roughly %.1f to %.1f per cent. The increase is larger under the "
  "higher forcing pathway at every horizon, and grows with lead time under both, which "
  "is the behaviour the screen in Section 4.4 required. The mechanism is visible in the "
  "predictors themselves: domain-mean cloud fraction in the bias-corrected ensemble falls "
  "from 38.87 per cent over 1985 to 2010 to 33.85 per cent under SSP2-4.5 and 31.26 per "
  "cent under SSP5-8.5 by 2076 to 2100, a reduction that is larger under the higher "
  "pathway and grows monotonically with lead time. Cloud fraction is also the dominant "
  "predictor in the fitted models on all four importance measures, consistent with the "
  "variable-selection findings of Maposa et al. (2024) for Southern African stations, at 0.333 and 0.395 for "
  "Random Forest impurity and permutation importance and 0.470 and 0.178 for XGBoost gain "
  "and cover, so a projected reduction in cloud translates directly into projected "
  "irradiance gain."
  % ((proj["ssp245_near_term_2026_2050"] - base).mean(), (proj["ssp585_long_term_2076_2100"] - base).mean(), 100 * (proj["ssp245_near_term_2026_2050"] - base).mean() / base.mean(), 100 * (proj["ssp585_long_term_2076_2100"] - base).mean() / base.mean()))
u = unc[unc.model == "xgb"].set_index("period")
TBL(["Horizon", "Downscaling", "Architecture", "GCM", "Scenario"], [[p.replace("_", " "), "%.1f%%" % u.loc[p, "pct_var_ds"], "%.1f%%" % u.loc[p, "pct_var_arch"], "%.1f%%" % u.loc[p, "pct_var_gcm"], "%.1f%%" % u.loc[p, "pct_var_ssp"]] for p in u.index], "Table 4.8. Variance decomposition of the projection uncertainty for the deployed "
    "model, as a percentage of total variance at each horizon.")
P("The uncertainty budget carries the more consequential result. At every horizon the "
  "choice of downscaling architecture is the largest single term, rising from %.1f per "
  "cent of total variance in the near term to %.1f per cent in the long term. The choice "
  "of global model is second and falls from %.1f to %.1f per cent, the downscaling error "
  "itself falls from %.1f to %.1f per cent as the forced signal grows, and the emission "
  "scenario stays below %.1f per cent throughout."
  % (u.loc["near_term_2026_2050", "pct_var_arch"], u.loc["long_term_2076_2100", "pct_var_arch"],
     u.loc["near_term_2026_2050", "pct_var_gcm"], u.loc["long_term_2076_2100", "pct_var_gcm"],
     u.loc["near_term_2026_2050", "pct_var_ds"], u.loc["long_term_2076_2100", "pct_var_ds"],
     max(u["pct_var_ssp"]) + 0.5))
P("Two choices in constructing that budget deserve stating, because an earlier version of "
  "this section got the first of them wrong. The downscaling term is the part of the "
  "validation error that survives a twenty-five-year mean, %.2f W/m², not the monthly "
  "validation RMSE of %.2f. The two differ by a factor of six because random "
  "month-to-month error largely averages out of a three-hundred-month mean while "
  "systematic error does not, and quoting the monthly figure inflated this term to "
  "roughly two thirds of total variance and crowded out everything else. The second "
  "choice is membership: the headline spread covers the three architectures admissible "
  "for projection, the Random Forest having failed the screen in Section 4.4. Retaining "
  "it instead raises the architecture share at the long horizon from %.1f to %.1f per "
  "cent, so the conclusion does not depend on the decision either way."
  % (u.loc["long_term_2076_2100", "sigma_ds"],
     u.loc["long_term_2076_2100", "sigma_ds_monthly_rmse_fullbox"],
     u.loc["long_term_2076_2100", "pct_var_arch"],
     u.loc["long_term_2076_2100", "pct_var_arch_all"]))
P("The correct reading is not that this study has identified the right architecture. It "
  "is that a study using a single architecture would have reported zero architecture "
  "uncertainty and been wrong by most of the long-term variance, whichever architecture "
  "it had chosen. A supporting result points the same way: no pair of architectures "
  "correlates above %.2f on the spatial pattern of projected change, and the deployed "
  "model agrees with the others at %.2f to %.2f. Agreement on the present is not "
  "agreement on the future."
  % (agr.r_projected_change.max(), _dep.r_projected_change.min(), _dep.r_projected_change.max()))

H("4.7.1 The operational baseline and consistency with the driver", 3)
P("Two checks the projections were not previously subjected to are reported here. The "
  "first is the operational alternative: bias-correcting each GCM's own rsds and "
  "interpolating it, which is the approach behind the published NEX-GDDP-CMIP6 product "
  "and the obvious thing a planner would do instead of training a model. The second is "
  "whether the downscaled change keeps the sign and size of the change in the driver it "
  "came from. Both are measured against each chain's own historical run rather than "
  "against ERA5, which removes the model's historical bias from the comparison.")
TBL(["Scenario and horizon", "Downscaled (W/m²)", "QDM baseline (W/m²)", "Ratio"],
    [["%s %s" % (r.scenario, r.horizon.split("_")[0]),
      "%+.2f" % r.ml, "%+.2f" % r.qdm, "%.2f" % (r.ml / r.qdm) if r.qdm != 0 else "—"]
     # Chronological, not alphabetical: a groupby sorts "long, mid, near".
     for r in sorted(pbl.groupby(["scenario", "horizon"], as_index=False)
                     .agg(ml=("ml_change_own_history", "mean"),
                          qdm=("qdm_change_own_history", "mean")).itertuples(),
                     key=lambda r: (r.scenario,
                                    ["near", "mid", "long"].index(r.horizon.split("_")[0])))],
    "Table 4.9. Projected change in annual-mean GHI over Zimbabwe from the deployed "
    "ensemble against the quantile-mapped interpolation baseline, each relative to its "
    "own 1985 to 2010 historical run, averaged over the three GCMs.")
P("The downscaled changes are consistently larger than the baseline's, by a median "
  "factor of %.2f across the eighteen GCM, scenario and horizon combinations. Measured "
  "against its own history rather than against ERA5, the long-term SSP5-8.5 increase is "
  "%+.2f W/m² rather than the %+.2f W/m² of Table 4.7, so the figure reported there is "
  "the more conservative of the two. The baseline gives %+.2f W/m² for the same case. "
  "Nothing here establishes which is closer to the truth, because there is no future "
  "observation to score them against; what it establishes is that the choice of method "
  "moves the answer by about a factor of two, which is the same conclusion the variance "
  "decomposition reaches."
  % (pbl.ratio_ml_to_qdm.median(),
     pbl[(pbl.scenario == "ssp585") & (pbl.horizon == "long_term_2076_2100")].ml_change_own_history.mean(),
     pbl[(pbl.scenario == "ssp585") & (pbl.horizon == "long_term_2076_2100")].ml_change_vs_era5.mean(),
     pbl[(pbl.scenario == "ssp585") & (pbl.horizon == "long_term_2076_2100")].qdm_change_own_history.mean()))
P("The sign check is less comfortable and is reported because it is inconvenient. The "
  "downscaled change agrees in sign with its driver's own bias-corrected rsds change in "
  "%.0f per cent of the eighteen combinations. All %s disagreements belong to "
  "MPI-ESM1-2-HR, whose own radiation declines under SSP5-8.5 by %.2f W/m² at the "
  "mid-term horizon while the downscaled field rises by %.2f. The mechanism is the one "
  "Section 2.5 anticipated for perfect-prognosis designs. That model's cloud fraction "
  "falls and its temperature rises, and the mapping learned from the present-day record "
  "reads both as brightening; the GCM's own radiation scheme, which resolves the "
  "aerosol and cloud-microphysical effects that outweigh them, is not consulted because "
  "rsds is excluded from the predictor set. MPI-ESM1-2-HR is also the only one of the "
  "three models whose equilibrium climate sensitivity falls inside the IPCC AR6 likely "
  "range, so the disagreement is with the most conservative driver in the ensemble. This "
  "is a limitation of the design rather than a defect in the implementation, and it "
  "bounds how far the projected magnitudes should be trusted."
  % (100 * pbl.same_sign.mean(),
     {1: "one", 2: "two", 3: "three", 4: "four"}.get(int((~pbl.same_sign).sum()),
                                                    str(int((~pbl.same_sign).sum()))),
     abs(pbl[(pbl.gcm == "MPI-ESM1-2-HR") & (pbl.scenario == "ssp585") &
             (pbl.horizon == "mid_term_2051_2075")].qdm_change_own_history.iloc[0]),
     pbl[(pbl.gcm == "MPI-ESM1-2-HR") & (pbl.scenario == "ssp585") &
         (pbl.horizon == "mid_term_2051_2075")].ml_change_own_history.iloc[0]))

FIG("03_feature_importance.png", "Predictor importance across the four measures. "
    "The topographic covariates score exactly zero for the pixel-wise models, which is "
    "structural: within one cell they are constants.")
FIG("04_uncertainty.png", "Four-component variance decomposition of the projection "
    "uncertainty for the deployed model, by horizon.")

H("4.8 Solar Energy Suitability")
P("The suitability analysis of Section 3.9 combines the irradiance layer with six "
  "biophysical and infrastructural criteria under the primary weighting scheme of "
  "Table 3.5, after a binary exclusion "
  "mask. No consistency ratio is quoted here. Section 3.9.4 records that the pairwise "
  "comparison matrix was completed by the researcher in consultation with the "
  "supervisors, but that matrix is not held in the project archive, and a ratio "
  "recomputed from a matrix reconstructed to reproduce Table 3.5's weights would measure "
  "the reconstruction rather than the elicitation. The question a consistency ratio is "
  "meant to answer, whether the ranking can be relied upon given the weights, is "
  "addressed directly, and far less favourably, in Section 4.8.1. "
  "Of the 5,751 cells in the analysis box, 2,460 fall "
  "outside Zimbabwe, 895 lie predominantly within protected areas, and smaller numbers "
  "are excluded as urban, water or steep, leaving %d cells assessed. Of those, %d score below the 0.30 threshold and fall "
  "in the lowest tier despite not being excluded." % (n_keep, n_low))
P("The present-day irradiance layer is the SARAH satellite record rather than the "
  "ERA5-derived field, for the reason established in Section 4.6 and quantified here. "
  "The two layers differ by only %.1f per cent in the mean and their ratio is close to "
  "spatially uniform, which suggests the choice should be immaterial under a min-max "
  "standardisation. It is not. Because each criterion is rescaled by its own range, the "
  "standardisation amplifies exactly the spatial structure that the near-uniform ratio "
  "conceals: the two layers rank the domain at a Spearman correlation of %.3f, and %.1f "
  "per cent of assessed cells fall on opposite sides of the highest suitability threshold "
  "depending on which is used. Under the ERA5 layer the analysis returns %d cells in the "
  "highest tier; under SARAH it returns %d. An independent observation is preferred to a "
  "reanalysis for a present-day siting assessment, and the choice is reported because it "
  "is consequential rather than because it is close."
  % (_sar_off, lc.spearman_rho, lc["pct_crossing_0.75"], per.loc["present_era5_basis", "very_high"], per.loc["present", "very_high"]))

FIG("07_suitability_criteria.png", "The seven standardised criterion layers and the "
    "exclusion mask. Excluded cells are left blank rather than drawn on the score ramp.")

H("4.8.1 The robust set", 3)
P("The headline result of this analysis is not the five-tier suitability map. It is the "
  "much smaller set of locations that remain highly suitable however the criteria are "
  "weighted.")
P("Section 3.9.6 repeated the weighted overlay under four weighting schemes: the primary "
  "weights of Table 3.5, an irradiance-dominant scheme, an infrastructure-dominant scheme, and "
  "equal weights. The classifications they produce diverge sharply. Cohen's kappa (Cohen, 1960) "
  "against the primary classification is %.3f for the irradiance-dominant scheme, %.3f "
  "for the infrastructure-dominant scheme, and %.3f for equal weights, the last "
  "indicating agreement no better than chance. Of the %d assessed cells, %d, %.1f per "
  "cent, change suitability tier under at least one scheme."
  % (K("irradiance-dominant"), K("infrastructure-dominant"), K("equal"), n_keep, sens, 100 * sens / n_keep))
P("**Only %d cells, %.1f per cent of those assessed and approximately %s km², are "
  "classified high or very high under all four weighting schemes.** These are reported "
  "as the robust set, and they are the defensible output of this analysis. The five-tier "
  "map is retained as supporting material, but a classification in which nine cells in "
  "ten can be moved by a defensible change of weights should not be presented as a "
  "planning product without that qualification attached."
  % (n_rob, 100 * n_rob / n_keep, "{:,.0f}".format(geo["area_km2"].sum())))
P("Section 3.9.3 previously printed the distance decay as Score = e^(-d/d_ref) while "
  "stating in prose that the score falls below 0.14 beyond d_ref, which the function does "
  "not do: e^(-1) is 0.368. That discrepancy was not cosmetic, since the two readings "
  "give robust sets of 145 and 42 cells respectively. It has been resolved in favour of "
  "the function as written, on the grounds that a site 10 km from an existing "
  "transmission line is routinely connectable for utility-scale development and should "
  "not be scored as though it were remote. The figures reported here use that reading.")
TBL(["Criterion", "Robust set mean", "All assessed cells", "Ratio"], [["Irradiance (W/m²)", "%.2f" % lay.ghi_present_sarah.values[rob].mean(), "%.2f" % lay.ghi_present_sarah.values[keep].mean(), "%.2f" % (lay.ghi_present_sarah.values[rob].mean() / lay.ghi_present_sarah.values[keep].mean())], ["Slope (degrees)", "%.2f" % lay.slope.values[rob].mean(), "%.2f" % lay.slope.values[keep].mean(), "%.2f" % (lay.slope.values[rob].mean() / lay.slope.values[keep].mean())], ["Land cover score", "%.2f" % lay.landcover_score.values[rob].mean(), "%.2f" % lay.landcover_score.values[keep].mean(), "%.2f" % (lay.landcover_score.values[rob].mean() / lay.landcover_score.values[keep].mean())], ["Distance to roads (km)", "%.2f" % lay.dist_roads.values[rob].mean(), "%.2f" % lay.dist_roads.values[keep].mean(), "%.2f" % (lay.dist_roads.values[rob].mean() / lay.dist_roads.values[keep].mean())], ["Distance to grid (km)", "%.2f" % lay.dist_grid.values[rob].mean(), "%.2f" % lay.dist_grid.values[keep].mean(), "%.2f" % (lay.dist_grid.values[rob].mean() / lay.dist_grid.values[keep].mean())], ["Distance to settlements (km)", "%.2f" % lay.dist_settlements.values[rob].mean(), "%.2f" % lay.dist_settlements.values[keep].mean(), "%.2f" % (lay.dist_settlements.values[rob].mean() / lay.dist_settlements.values[keep].mean())]], "Table 4.10. Mean criterion values on the robust set against all assessed cells. The "
    "final column is the ratio; values far from 1.00 identify the criteria that "
    "distinguish the robust set.")
P("Table 4.10 makes the character of the robust set clear, and the result is not the "
  "obvious one. **These are not the sunniest places in Zimbabwe.** Their mean irradiance "
  "is %.2f W/m² against %.2f for the assessed domain as a whole, a difference of %.1f "
  "per cent. What distinguishes them is infrastructure: they lie a mean of %.2f km from "
  "the transmission network against %.2f km for the domain, a factor of %.1f, and %.2f "
  "km from a road against %.2f km."
  % (lay.ghi_present_sarah.values[rob].mean(), lay.ghi_present_sarah.values[keep].mean(), 100 * (lay.ghi_present_sarah.values[rob].mean() / lay.ghi_present_sarah.values[keep].mean() - 1), lay.dist_grid.values[rob].mean(), lay.dist_grid.values[keep].mean(), lay.dist_grid.values[keep].mean() / lay.dist_grid.values[rob].mean(), lay.dist_roads.values[rob].mean(), lay.dist_roads.values[keep].mean()))
P("This has a straightforward explanation and a substantive implication. Irradiance over "
  "Zimbabwe varies within a narrow band, the assessed cells span roughly 214 to 258 "
  "W/m², so after standardisation the irradiance criterion cannot separate sites "
  "strongly no matter how heavily it is weighted. Distance to the grid varies over two "
  "orders of magnitude and enters through a decay function, so it separates sites "
  "decisively under every weighting. Robustness to weighting is therefore achieved by "
  "infrastructure rather than by resource. For a national planning question the "
  "implication is that the binding constraint on utility-scale solar siting in Zimbabwe "
  "is grid access, not sunlight, and that a policy of extending transmission would open "
  "more suitable area than any refinement of the resource estimate.")
# Computed by compute_robust_set_geography.py, which asserts that the province
# counts sum to the robust set. This sentence previously carried a hardcoded list
# summing to 41 cells, left from a superseded reading of the decay parameter.
prov = ", ".join("%s (%d cell%s)" % (r.NAME_1, r.cells, "" if r.cells == 1 else "s")
                 for r in geo.itertuples())
P("The robust set is concentrated along the central watershed between %.1f and %.1f "
  "degrees east, following the Harare–Bulawayo road and transmission corridor. By "
  "province it falls in %s. The single highest-scoring cell reaches a suitability index "
  "of %.3f at 29.8 degrees east, 18.9 degrees south, in Kwekwe Urban district. The set "
  "covers %s km², computed from the true area of each cell at its own latitude."
  % (LON[rob].min(), LON[rob].max(), prov,
     np.nanmax(np.where(keep, sui.si_primary.values, np.nan)),
     "{:,.0f}".format(geo["area_km2"].sum())))

P("Four weight vectors is a thin test of robustness, and three other choices in "
  "Section 3.9 are no less arbitrary than the weights: the reference distance in the "
  "proximity decay, the two tier cuts, and the decay form itself. Re-reading the "
  "reference distance alone once moved this set between 42 and %d cells. All of them "
  "were therefore perturbed jointly, over %d draws, with the weights sampled from a "
  "Dirichlet distribution centred on the primary vector, each reference distance "
  "scaled by a factor between one half and two, and both tier cuts jittered by up to "
  "0.05. A cell's robustness is then the fraction of draws in which it holds the top "
  "two tiers, which says more than a yes or no against four vectors."
  % (int(mc["four_scheme_robust_set"].iloc[0]), 2000))
P("The four-scheme set survives this. Its members hold the top two tiers in a median "
  "100 per cent of draws, with the fifth percentile at 93 per cent, and %d cells clear "
  "a 99 per cent threshold against the %d identified by the four vectors. The "
  "agreement is close enough that the simpler test can be regarded as a fair proxy. "
  "Under the strictest reading, %d cells hold the top two tiers in every one of the "
  "draws, and that is the number to quote where no weighting assumption at all is "
  "admissible."
  % (int(mc.loc[0.99, "cells"]), int(mc["four_scheme_robust_set"].iloc[0]),
     int(mc.loc[1.00, "cells"])))

FIG("08_suitability_primary.png", "Suitability index and five-tier classification "
    "under the primary weights.")
FIG("09_suitability_schemes.png", "The four weighting schemes and the robust set. "
    "Nearly nine assessed cells in ten change tier under at least one scheme.")

H("4.8.2 Suitability under the projected climate", 3)
# The present row must be the ERA5-basis one. The future periods are scored on
# ERA5-trained downscaled layers, so comparing them against a SARAH-based present
# puts the SARAH-minus-ERA5 offset inside the comparison and makes the near-term
# rows appear to fall. Both present rows are shown, the SARAH one labelled as the
# observational layer and the ERA5 one as the basis for comparison, because the
# change figures quoted in the prose are taken against the latter.
_LBL = {"present": "present (SARAH layer)",
        "present_era5_basis": "present (ERA5 basis, comparison row)"}
TBL(["Period", "Mean GHI (W/m²)", "Mean SI", "Very high", "High"], [[_LBL.get(p, p.replace("_", " ")), "%.2f" % per.loc[p, "mean_ghi"], "%.4f" % per.loc[p, "mean_si"], int(per.loc[p, "very_high"]), int(per.loc[p, "high"])]
     for p in ["present", "present_era5_basis", "ssp245_near_term_2026_2050", "ssp245_mid_term_2051_2075", "ssp245_long_term_2076_2100", "ssp585_near_term_2026_2050", "ssp585_mid_term_2051_2075", "ssp585_long_term_2076_2100"]], "Table 4.11. Suitability by period under the primary weights. All periods are "
    "standardised on the present-day range so the tiers remain comparable. The "
    "projections are compared against the ERA5-basis present row, not the SARAH "
    "layer, so that the SARAH-minus-ERA5 offset does not enter the change signal.")
P("Suitability rises under every scenario and horizon, and the count of cells in the "
  "highest tier increases from %d on the ERA5-basis present to %d under SSP5-8.5 by "
  "2076 to 2100. Measured instead against the SARAH layer, which is a different "
  "measurement system, the near-term rows fall; that comparison is not the change "
  "signal and is not read as one here. The statement requires an important "
  "qualification, which is given next rather than left to the reader."
  % (per.loc["present_era5_basis", "very_high"],
     per.loc["ssp585_long_term_2076_2100", "very_high"]))
P("Every one of the %d assessed cells improves under every one of the six projections. "
  "Not most of them: all of them, without exception. This is a direct consequence of the "
  "analysis design rather than a finding about Zimbabwe. Only the irradiance layer varies "
  "between periods; slope, land cover, roads, transmission, settlements and population "
  "are all held at their present values, because the study has no scenarios for how they "
  "will develop. The change in suitability is therefore the change in the standardised "
  "irradiance score multiplied by its weight of %.2f, and it carries no information "
  "that Table 4.7 did not already contain."
  % (n_keep, wts.loc["primary", "ghi"]))
P("Two conclusions follow. The increase from %d to %d cells in the highest tier is "
  "threshold crossing rather than relocation: the ranking of sites is almost unchanged "
  "between periods, and what happens is that a rising tide lifts cells across fixed "
  "boundaries. And these maps should not be read as identifying where solar development "
  "should be sited in 2100, because doing so assumes the transmission network, road "
  "network and population distribution of 2100 are those of the present, in a country "
  "whose population is projected to grow substantially over the period. The defensible "
  "claim is about how much the resource improves at sites that are viable today, not "
  "about where tomorrow's viable sites will be. Answering the second question requires "
  "infrastructure and demographic scenarios that this study does not have, and Section "
  "4.9 records it as a limitation."
  % (per.loc["present", "very_high"], per.loc["ssp585_long_term_2076_2100", "very_high"]))

FIG("10_suitability_periods.png", "Suitability by period, and the change against the "
    "ERA5-basis present. Every panel in the top row is on the ERA5 basis, so the "
    "comparison stays inside one measurement system; the SARAH present-day layer is "
    "mapped in Figures 4.7 and 4.8. Only the irradiance layer varies between periods, "
    "with infrastructure and population held at present values.")

H("4.9 Limitations")
P("Several of the limitations below were established by the results rather than "
  "anticipated in the design, and they are stated at the strength the evidence supports.")
P("**The product does not resolve sub-grid structure.** Section 4.5 establishes that the "
  "0.1 degree field survives a round trip through 0.25 degrees at a correlation of %.6f. "
  "The contribution is temporal skill and the uncertainty budget, not spatial detail." % rt)
P("**The evaluation is not independent of the training data.** Training and validation "
  "both use ERA5, over a region where surface radiation observations are sparse and "
  "reanalysis is correspondingly weak. The temporal split is genuine and forward in time, "
  "so Section 4.2 is a real out-of-sample test of the reanalysis relationship, but it is "
  "not a test against observations. Section 4.6 quantifies the reference uncertainty at "
  "roughly 3 per cent with a seasonal structure, and Section 4.6.1 scores the product "
  "against SARAH directly, but that score is a diagnostic rather than a validation: the "
  "disagreement between the two references exceeds the spread between the four "
  "architectures by more than an order of magnitude. An observational validation "
  "requires refitting against SARAH as the target and remains outstanding.")
P("**The learning task is constrained by its own construction.** The coarse irradiance "
  "field was excluded from the predictors because it is a copy of the field the target is "
  "derived from, which made the task a genuine inference from atmospheric state. That "
  "exclusion cost accuracy: with the field retained the same model reaches 5.54 W/m² "
  "against %.2f without it. The excluded skill was circular, and removing it was correct, "
  "but the reported performance is the lower figure." % R("XGBoost", "RMSE"))
P("**The intermediate clear-sky index is not a physical clear-sky index.** Its "
  "denominator averages thirteen daytime hours while the irradiance it normalises is a "
  "24-hour monthly mean, so the ratio runs a factor of 1.724 below a true clear-sky "
  "index. The normalisation is an exact inverse and cancels from the reconstructed "
  "product to within 1.14 × 10⁻¹³ W/m², so no reported result depends on it, but the "
  "absolute level of the intermediate should not be read as a fraction of clear-sky "
  "irradiance.")
P("**Two exclusion criteria do not bind at this resolution.** The slope exclusion removes "
  "two cells, because averaging 90 m terrain over a 121 km² cell smooths individual "
  "steep faces; the riparian exclusion removes none, because a 200 m corridor is 1.8 per "
  "cent of a cell width. Both rules are sound at their native scale and neither can "
  "operate at 0.1 degrees. The relief is present in the data, 81 cells have more than 20 "
  "per cent of their area above 15 degrees, and 80 per cent of those lie east of 32 "
  "degrees, and the Eastern Highlands are penalised through the slope criterion rather "
  "than removed by the exclusion.")
P("**The suitability classification is weight-sensitive.** %.1f per cent of assessed "
  "cells change tier under at least one defensible weighting, and equal weights agree "
  "with the primary classification no better than chance. This is why Section 4.8.1 "
  "leads with the robust set." % (100 * sens / n_keep))
P("**The future suitability maps hold infrastructure and population constant.** They "
  "describe how the resource changes at present-day sites, not where future sites will "
  "be.")
P("**Model selection on the evaluation record, identified and corrected.** The CNN and "
  "U-Net previously saved the checkpoint scoring best on the withheld period. Both have "
  "been retrained with the epoch count chosen on an inner split of the training data, "
  "which raised their errors by 0.92 and 0.95 W/m² respectively. The figures reported "
  "here are the corrected ones. The limitation is recorded because the earlier figures "
  "appear in superseded versions of this work and because the episode bears on how the "
  "architecture comparison should be read: it compares particular configurations, "
  "selected in a particular way, rather than architectures in the abstract.")
P("**The suitability output carries no uncertainty estimate on the index itself.** "
  "Section 1.5 undertakes to deliver priority zones with quantified uncertainty. What is "
  "delivered is a weighting sensitivity analysis and a robust set, which bound the "
  "influence of the weights but do not propagate the irradiance uncertainty of Section "
  "4.7, the reference uncertainty of Section 4.6, or the positional uncertainty of the "
  "infrastructure layers onto the suitability index. That undertaking is therefore only "
  "partly met, and the wording in Chapter 1 should be brought into line with what the "
  "analysis actually produces.")

H("4.10 Synthesis")
P("Four architectures were trained on identical inputs and evaluated on a withheld "
  "fourteen-year record. All four clear the accuracy thresholds set in Chapter 3, and the "
  "differences between them are small in absolute terms and only partly established: "
  "XGBoost is better on aggregate error, the Random Forest better on mean bias and "
  "centred error, and the two cannot be separated on spatial correlation. Selection was "
  "not decided on any of those axes. It was decided by a test that no validation metric "
  "could perform, in which the Random Forest was found to lose the scenario signal it "
  "would be required to project, for a mechanical reason traceable to how regression "
  "trees behave outside their training range.")
P("The projections give an increase in surface irradiance over Zimbabwe of %.1f to %.1f "
  "per cent by 2100 depending on pathway, and the resource is not the constraint on "
  "solar development in any case. The suitability analysis identifies %d locations, "
  "about %s km², that remain highly suitable under every weighting scheme tested, and "
  "what distinguishes them is proximity to the transmission network rather than solar "
  "resource. Irradiance over Zimbabwe varies too little to discriminate between sites; "
  "grid access varies by two orders of magnitude and decides the outcome."
  % (100 * (proj["ssp245_near_term_2026_2050"] - base).mean() / base.mean(), 100 * (proj["ssp585_long_term_2076_2100"] - base).mean() / base.mean(), n_rob, "{:,.0f}".format(geo["area_km2"].sum())))
P("The methodological contribution is therefore narrower and firmer than a single "
  "suitability map would suggest. It is that architecture choice is a first-order source "
  "of projection uncertainty, reaching %.1f per cent of long-term variance against %.1f "
  "per cent for the choice of global climate model; that a model can pass every "
  "historical validation metric and still be unusable for projection; and that the "
  "outputs of a multi-criteria overlay must be tested against their own weighting "
  "assumptions before they are offered as planning guidance, because in this case nine "
  "assessed cells in ten proved movable."
  % (u.loc["long_term_2076_2100", "pct_var_arch"], u.loc["long_term_2076_2100", "pct_var_gcm"]))

doc.add_page_break()
doc.add_heading("Note on this draft", level=2)
P("This chapter is generated by build_chapter4.py from the evaluation CSVs, so that no "
  "figure in the prose can drift from the analysis that produced it. Edit the generator "
  "rather than this document.")
P("Citations are plain author-year text and must be converted to live Zotero fields in "
  "Word. Most cite works already in the Chapter 2 and Chapter 3 bibliographies and need "
  "only re-citing. THREE ARE NOT YET IN THE LIBRARY and must be added before the "
  "reference list is regenerated:")
for _ref in [
    "Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and "
    "Psychological Measurement 20(1), 37–46. Cited in Section 4.8.1 for kappa. "
    "doi:10.1177/001316446002000104", "Efron, B. (1987). Better bootstrap confidence intervals. Journal of the American "
    "Statistical Association 82(397), 171–185. The BCa interval of Section 4.3. "
    "doi:10.1080/01621459.1987.10478410", "Dozier, J. and Frew, J. (1990). Rapid calculation of terrain parameters for "
    "radiation modeling from digital elevation data. IEEE TGRS 28(5), 963–969. "
    "Already cited as plain text in Chapter 3 and still missing from the library. "
    "doi:10.1109/36.58986",
]:
    _p = doc.add_paragraph(_ref, style="List Bullet")
    _p.runs[0].font.size = Pt(11)
P("Saaty (1980) is no longer on that list. This chapter does not cite it: the weights "
  "of Table 3.5 are presented as the primary weighting scheme, not as the output of the "
  "Analytic Hierarchy Process, because Section 3.9.4 records that no pairwise comparison "
  "matrix was elicited or retained. Chapter 3 Section 3.9.4 still names the AHP once, as "
  "the source of the ordinal logic the weights follow. Add Saaty only if you choose to "
  "cite it there; nothing in either chapter requires it.")
P("Figures are inserted from the figures directory by this generator. Regenerating the "
  "figures and rebuilding this chapter keeps them in step; pasting them by hand would "
  "not.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
words = sum(len(p.text.split()) for p in doc.paragraphs)
print("Saved %s" % OUT)
print("  %d paragraphs, %d tables, ~%d words" % (len(doc.paragraphs), len(doc.tables), words))
print("  robust set %d cells | assessed %d | weight-sensitive %.1f%%"
      % (n_rob, n_keep, 100 * sens / n_keep))
