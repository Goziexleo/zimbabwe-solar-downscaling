"""Generate Chapter 5 from the same canonical sources as Chapter 4.

Conclusions must not restate results in numbers that have drifted from the
results chapter, so both chapters read the same CSVs at build time. Edit this
file, not the .docx.

    python build_chapter5.py
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
                         "PROJECT CHAPTERS/CR_Madukwe_Chapter5_DRAFT.docx")

t33 = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")
unc = pd.read_csv(os.path.join(EVAL, "uncertainty_decomposition_summary.csv"))
info = pd.read_csv(os.path.join(EVAL, "information_content.csv")).set_index("field")
per = pd.read_csv(os.path.join(EVAL, "suitability_by_period.csv")).set_index("period")
scr = pd.read_csv(os.path.join(EVAL, "scenario_discrimination.csv")).set_index("model")
uo = pd.read_csv(os.path.join(EVAL, "unet_optimisation.csv")).set_index("variant")
udv5 = pd.read_csv(os.path.join(EVAL, "unet_dropout_variant.csv")).set_index("model")
sdv5 = pd.read_csv(os.path.join(EVAL, "scenario_discrimination_variants.csv")).set_index("model")
spv5 = pd.read_csv(os.path.join(EVAL, "sarah_product_validation.csv")).set_index("model")
# True area from each cell's own latitude, not a nominal 121 km2 cell.
geo5 = pd.read_csv(os.path.join(EVAL, "robust_set_geography.csv"))
bas5 = pd.read_csv(os.path.join(EVAL, "baselines.csv")).set_index("baseline")
bvm5 = pd.read_csv(os.path.join(EVAL, "baseline_vs_models.csv")).set_index("model")
pbl5 = pd.read_csv(os.path.join(EVAL, "projection_baselines.csv"))
leaky5 = pd.read_csv(os.path.join(ROOT, "data/processed/models/_pre_honest_selection", "table_3_3.csv")).set_index("model")
# Over Zimbabwe, like every headline metric in Chapter 4. The full-box columns
# are used only where a current figure is set against an archived snapshot that
# exists on the full box alone.
_order = list(t33["RMSE_zw"].sort_values().index)
_ORD = {1: "lowest", 2: "second-lowest", 3: "third-lowest", 4: "highest"}
def rank(m):
    """Ordinal position on aggregate RMSE, so the prose cannot outlive the table."""
    return _ORD[_order.index(m) + 1]
sar = pd.read_csv(os.path.join(EVAL, "sarah_era5_monthly.csv"))
lay = xr.open_dataset(os.path.join(SUIT, "criterion_layers.nc"))
sui = xr.open_dataset(os.path.join(SUIT, "suitability_index.nc"))
u = unc[unc.model == "xgb"].set_index("period")

rob = sui.robustly_suitable.values.astype(bool)
_wet = lay.water_fraction.values.copy()
if "river_fraction" in lay:
    _wet = np.clip(_wet + lay.river_fraction.values, 0, 1)
keep = ~((lay.in_zimbabwe.values <= .5) | (lay.protected_fraction.values > .5)
         | (lay.urban_fraction.values > .5) | (_wet > .5) | (lay.slope.values > 15.0)
         | np.isnan(lay.ghi_present_sarah.values))
n_rob, n_keep = int(rob.sum()), int(keep.sum())
sens = int(sui.weight_sensitive.values.sum())
base = lay.ghi_present_era5.values
lo_ch = (lay["ghi_ssp245_near_term_2026_2050"].values - base).mean()
hi_ch = (lay["ghi_ssp585_long_term_2076_2100"].values - base).mean()
g_r, g_a = lay.dist_grid.values[rob].mean(), lay.dist_grid.values[keep].mean()
i_r, i_a = lay.ghi_present_sarah.values[rob].mean(), lay.ghi_present_sarah.values[keep].mean()

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
st.paragraph_format.space_after = Pt(8); st.paragraph_format.line_spacing = 1.5


def H(t, lvl=2): doc.add_heading(t, level=lvl)


FIGDIR = os.path.join(ROOT, "figures")
_fig_n = [0]


def FIG(png, caption, width_in=6.4):
    path = os.path.join(FIGDIR, png)
    if not os.path.exists(path):
        print("  WARNING: missing figure %s" % png); return
    _fig_n[0] += 1
    doc.add_picture(path, width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    r = c.add_run("Figure 5.%d. %s" % (_fig_n[0], caption))
    r.font.size = Pt(10); r.italic = True
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def P(t):
    import re as _re
    para = doc.add_paragraph()
    for c in _re.split(r"(\*\*.+?\*\*)", t):
        if not c:
            continue
        if c.startswith("**") and c.endswith("**"):
            para.add_run(c[2:-2]).bold = True
        else:
            para.add_run(c)
    return para

doc.add_heading("Chapter 5: Conclusions and Recommendations", level=1)

H("5.1 Overview")
P("This chapter answers the four research questions set out in Section 1.7, states what "
  "the study contributes and what it does not, and sets out what follows from it for "
  "solar energy planning in Zimbabwe and for further research. Two of the four answers "
  "are narrower than the questions anticipated, and they are given in that narrower form "
  "rather than stretched to fit. A study that reports only the findings it hoped for is "
  "not reporting findings.")

H("5.2 Answers to the Research Questions")

H("5.2.1 RQ1: downscaling accuracy and the comparison between architectures", 3)
P("Four architectures were trained on the same predictor set and evaluated on the same "
  "withheld fourteen-year record, though not under identical conditions: Sections 3.6.5 "
  "to 3.6.8 record differences in static-input resolution, coordinate channels, sampling "
  "and target formulation between the two networks, so this is a comparison of particular "
  "configurations rather than of architectures in the abstract. All four exceed the accuracy thresholds set in Section 3.7.1. "
  "The deployed pixel-wise XGBoost ensemble (Chen and Guestrin, 2016) reaches %.2f W/m² against an ERA5-derived "
  "target, a skill score of %.4f relative to a training-period climatology, and a "
  "correlation of %.4f. The spread between best and worst is %.2f W/m²."
  % (t33.loc["XGBoost", "RMSE_zw"], t33.loc["XGBoost", "SS vs climatology_zw"], t33.loc["XGBoost", "Pearson R_zw"], t33["RMSE_zw"].max() - t33["RMSE_zw"].min()))
P("The comparison between architectures is answered with more care than the question "
  "invites. Resampling establishes that XGBoost is better than the Random Forest on "
  "aggregate error, that the Random Forest is better on mean bias, and that the two "
  "cannot be separated on the spatial structure of the error or on spatial correlation. "
  "The answer is a split decision, not a ranking. The Random Forest returns the %s "
  "aggregate error. It cannot be separated from the CNN, whose interval contains zero, "
  "but it is established better than the U-Net. Only XGBoost is established as better "
  "than all three others on that axis." % rank("Random Forest"))
P("The more consequential comparison is not among the four. A per-cell ordinary least "
  "squares regression, fitted on the same predictors over the same period against the "
  "same target, reaches %.2f W/m², below every architecture in Table 4.1. Its margin "
  "over the deployed XGBoost is %+.3f W/m² with an interval of %+.3f to %+.3f that "
  "spans zero, so the two are indistinguishable; its margins over the CNN, the U-Net and "
  "the Random Forest are established. The premise on which Chapter 1 justified machine "
  "learning, that the predictor-predictand relationship is too nonlinear for linear "
  "methods, therefore does not hold on this target. Section 4.2.1 gives the reason, and "
  "it is the same one RQ2 arrives at independently: a target that carries %.4f per cent "
  "of its variance below the coarse scale is close to a linear function of its own "
  "coarse predictors, leaving little nonlinear structure to recover. The honest answer "
  "to RQ1 is that the architectures can be ranked against each other, that the ranking "
  "is partly established, and that none of them earns its complexity against a linear "
  "model on this target."
  % (bas5.loc["Linear regression (OLS)", "RMSE_zw"],
     bvm5.loc["XGBoost", "diff_vs_linear"], bvm5.loc["XGBoost", "ci_lo"],
     bvm5.loc["XGBoost", "ci_hi"],
     info.loc["GHI target (time-mean)", "pct_variance_below_0.25deg"]))
P("Two qualifications belong with that answer, and both concern how the comparison was "
  "produced rather than what it found. The convolutional models originally selected "
  "their saved weights and their hyperparameters on the evaluation record. Both were "
  "corrected: the epoch count is now chosen on an inner split of the training period, "
  "and the hyperparameter grids were re-run inside that period as well. The two "
  "corrections pull opposite ways, and Section 4.2 separates them; the net is that the "
  "CNN moves %.2f W/m² from where it stood under full leakage and the U-Net %.2f the "
  "other way, both measured between archived runs on the full analysis box. "
  "Separately, a controlled sweep shows the U-Net reaches %.2f W/m² under honest "
  "selection once its dropout setting is removed in that sweep, although Section 4.5 "
  "reports that the improvement does not survive an honest retrain. "
  "Neither fact overturns the deployment, for the reason given under RQ3, but together "
  "they mean this study compares particular configurations, selected in a particular "
  "way, rather than architectures in the abstract."
  % (abs(t33.loc["CNN", "RMSE"] - leaky5.loc["CNN", "RMSE"]),
     abs(t33.loc["U-Net", "RMSE"] - leaky5.loc["U-Net", "RMSE"]),
     uo.loc["drop_0", "test_rmse"]))  # full box both sides: the archived snapshot has no masked columns
P("Against traditional baselines the answer is unambiguous only for the admissible one. "
  "All four models beat a per-cell, per-calendar-month climatology. Scores against "
  "interpolation of the coarse irradiance field are not skill, because that field is a "
  "reuse of the source of the target; they measure the circularity of the task and are "
  "reported as such.")

H("5.2.2 RQ2: what the high-resolution fields resolve that GCM output does not", 3)
P("**This question receives a largely negative answer, and it is the most important "
  "single finding of the study.** Degrading the 0.1 degree product to 0.25 degrees and "
  "interpolating it back recovers it at a correlation of %.6f, losing %.4f per cent of "
  "its variance. Elevation loses %.3f per cent under the identical test, so the test "
  "detects sub-grid structure when it exists. The product therefore contains almost no "
  "spatial information below the resolution of its input grid."
  % (info.loc["GHI target (time-mean)", "round_trip_correlation"], info.loc["GHI target (time-mean)", "pct_variance_below_0.25deg"], info.loc["CONTROL: elevation", "pct_variance_below_0.25deg"]))
P("This is a property of the design rather than a failure of fitting. The predictors are "
  "0.25 degree fields and the target is derived from a 0.25 degree field, so no sub-grid "
  "information exists anywhere in the training data for a model to recover. No "
  "architecture, loss function or training schedule can manufacture it. What the study "
  "delivers is a bias-and-variability correction evaluated on a finer mesh, with genuine "
  "temporal skill, and it should be described in those terms rather than as spatial "
  "super-resolution in the sense of Vandal et al. (2017). The physiographic correspondence the question anticipated, fine "
  "structure aligning with Zimbabwe's relief zones, is not present to be reported, "
  "because relief-scale structure is not in the input.")
P("The one route to a different answer is a genuinely high-resolution target. The CM SAF "
  "SARAH record at 0.05 degrees was acquired for precisely this purpose and is held for "
  "the full 1985 to 2024 period, but retraining against it was not undertaken within "
  "this study and is the first recommendation of Section 5.6.")

H("5.2.3 RQ3: projected changes in the solar resource to 2100", 3)
P("All six projections give an increase in surface irradiance over Zimbabwe, from %+.2f "
  "W/m² in the near term under SSP2-4.5 to %+.2f W/m² in the long term under SSP5-8.5, "
  "or roughly %.1f to %.1f per cent. The increase is larger under the higher forcing "
  "pathway at every horizon and grows with lead time under both. The mechanism is "
  "visible in the predictors: domain-mean cloud fraction falls from 38.87 per cent over "
  "the historical period to 33.85 per cent under SSP2-4.5 and 31.26 per cent under "
  "SSP5-8.5 by 2076 to 2100, and cloud fraction is the dominant predictor in the fitted "
  "models on all four importance measures."
  % (lo_ch, hi_ch, 100*lo_ch/base.mean(), 100*hi_ch/base.mean()))
P("The uncertainty attached to those projections carries a result the question did not "
  "anticipate. At the long-term horizon the choice of downscaling architecture accounts "
  "for %.1f per cent of projection variance, against %.1f per cent for the choice of "
  "global climate model, %.1f per cent for the downscaling error and %.1f per cent for "
  "the emission scenario. A single-architecture study would have reported zero "
  "architecture uncertainty and been wrong by that margin, whichever architecture it had "
  "chosen."
  % (u.loc["long_term_2076_2100", "pct_var_arch"], u.loc["long_term_2076_2100", "pct_var_gcm"],
     u.loc["long_term_2076_2100", "pct_var_ds"], u.loc["long_term_2076_2100", "pct_var_ssp"]))
P("Two checks qualify the magnitudes. Measured against each chain's own historical run "
  "rather than against ERA5, which removes the model's historical bias from the "
  "comparison, the long-term SSP5-8.5 increase is %+.2f W/m². The operational "
  "alternative of bias-correcting each GCM's own irradiance and interpolating it, the "
  "approach behind NEX-GDDP-CMIP6, gives %+.2f W/m² for the same case, and across all "
  "eighteen combinations the downscaled change exceeds the baseline's by a median factor "
  "of %.2f. No future observation exists to decide between them, so the comparison "
  "establishes the sensitivity of the answer to method rather than the accuracy of "
  "either."
  % (pbl5[(pbl5.scenario == "ssp585") & (pbl5.horizon == "long_term_2076_2100")].ml_change_own_history.mean(),
     pbl5[(pbl5.scenario == "ssp585") & (pbl5.horizon == "long_term_2076_2100")].qdm_change_own_history.mean(),
     pbl5.ratio_ml_to_qdm.median()))
P("The second check is less comfortable. The downscaled change agrees in sign with its "
  "driver's own bias-corrected irradiance change in %.0f per cent of the eighteen "
  "combinations, and every disagreement belongs to MPI-ESM1-2-HR, whose own radiation "
  "declines under SSP5-8.5 where the downscaled field rises. The cause is structural: "
  "that model's cloud cover falls and its temperature rises, and a mapping learned from "
  "the present-day record reads both as brightening, while the GCM's own radiation "
  "scheme is never consulted because irradiance is excluded from the predictor set. It "
  "is also the only one of the three models whose climate sensitivity lies inside the "
  "IPCC AR6 likely range. The direction of change reported here should therefore be read "
  "as conditional on the perfect-prognosis assumption holding, which for one of three "
  "drivers it demonstrably does not."
  % (100 * pbl5.same_sign.mean()))
P("The deployment decision followed from a test that no accuracy metric could perform. "
  "The Random Forest, %s on aggregate error and better than the deployed model on "
  "two of five tested axes, loses the scenario signal it would be required to project: "
  "its separation between pathways shrinks with lead time rather than growing, and only "
  "%.1f per cent of cells order the two pathways correctly. The mechanism is tree "
  "extrapolation (Breiman, 2001), and the failure is categorical rather than a matter "
  "of degree. This is "
  "why the improved U-Net configuration identified after the fact does not reopen the "
  "decision on accuracy alone: Section 4.5 reports that it passes the screen but is less "
  "accurate than the deployed configuration under the same training procedure."
  % (rank("Random Forest").replace("-lowest", ""), scr.loc["Random Forest", "pct_ordered_long_term"]))

H("5.2.4 RQ4: where suitability is highest", 3)
P("Of %d assessed locations, **%d, %.1f per cent, approximately %s km², are classified "
  "highly suitable under every weighting scheme tested.** These are reported as the "
  "answer, in preference to the five-tier map, because %.1f per cent of assessed cells "
  "change tier under at least one defensible reweighting and equal weighting agrees with "
  "the primary classification no better than chance. A classification that movable is not "
  "a planning product without that qualification attached."
  % (n_keep, n_rob, 100*n_rob/n_keep, "{:,.0f}".format(geo5["area_km2"].sum()), 100*sens/n_keep))
P("**The robust locations are not the sunniest places in Zimbabwe.** Their mean "
  "irradiance exceeds the assessed domain's by %.1f per cent. What distinguishes them is "
  "proximity to the transmission network: %.2f km against %.2f km, a factor of %.1f. "
  "Irradiance over Zimbabwe spans a narrow band, so once standardised it cannot separate "
  "candidate sites however heavily it is weighted, while distance to the grid varies over "
  "two orders of magnitude and decides the outcome under every weighting. They lie along "
  "the central watershed following the Harare–Bulawayo road and transmission corridor, "
  "concentrated in the Midlands and Mashonaland West."
  % (100*(i_r/i_a - 1), g_r, g_a, g_a/g_r))
P("Suitability rises under every scenario and horizon, but that result is weaker than it "
  "appears and is reported with its qualification. Every assessed cell improves in every "
  "projection, because only the irradiance layer varies between periods while "
  "infrastructure, land cover and population are held at present values. The change is "
  "the irradiance change scaled by its weight, and the growth in the highest tier is "
  "cells crossing fixed thresholds rather than the best locations moving.")

FIG("09_suitability_schemes.png", "The four weighting schemes and the robust set. "
    "The robust locations, rather than any single classification, are this analysis's "
    "answer to RQ4.")

H("5.3 Contributions")
P("**A quantified architecture-uncertainty term for solar downscaling.** The study's "
  "firmest methodological contribution is the finding that the choice of downscaling "
  "architecture accounts for %.1f per cent of long-term projection variance, several "
  "times the contribution of the global climate model. The term is sensitive to how it "
  "is constructed, and Section 4.7 reports that sensitivity rather than a single figure: "
  "the share depends more on whether the downscaling error is taken as a monthly or a "
  "systematic quantity than on which architectures are counted. A comparable result for "
  "temperature and precipitation is reported by Lafferty and Sriver (2023); what is new "
  "here is the quantity for surface solar radiation, and the finding that it exceeds the "
  "inter-model spread. Downscaling studies conventionally "
  "report GCM and scenario spread while fitting a single architecture (Vandal et al., 2017; Lin et al., 2023), while comparative studies such as Hernanz et al. (2023) and Rampal et al. (2024) rank architectures without propagating the choice into a projection uncertainty, which silently "
  "sets the largest of the three terms to zero."
  % u.loc["long_term_2076_2100", "pct_var_arch"])
P("**A demonstration that historical validation cannot substitute for a projection test.** "
  "The Random Forest passes every accuracy threshold, is %s on aggregate error, and "
  "is better than the deployed model on two of five resampled axes, yet cannot produce "
  "the deliverable. The screen that detects this is cheap, is not standard practice, and "
  "would have changed the model selection in this study had it not been applied."
  % rank("Random Forest").replace("-lowest", ""))
P("**An explicit account of what the product does not contain.** The round-trip test "
  "establishes that the fields carry essentially no sub-grid information, and the study "
  "reports that rather than presenting resolution as achievement. The same applies to the "
  "circularity of the learning task: excluding the coarse irradiance field cost accuracy, "
  "and the reported figure is the lower one.")
P("**A suitability analysis reported against its own weighting assumptions.** The robust "
  "set, rather than the tier map, is the output. The finding that robustness is conferred "
  "by grid proximity rather than by solar resource is a substantive planning result and "
  "not a methodological aside.")
P("**Applied outputs for Zimbabwe.** Projections of monthly irradiance to 2100 at 0.1 "
  "degrees under two pathways, with a four-component uncertainty budget, and a "
  "suitability assessment with an explicit robust subset. No downscaled present-period "
  "climatology was produced: Section 3.9.1 records that the present-day suitability layer "
  "is the SARAH satellite record rather than a product of this chain.")

H("5.4 Limitations")
P("The limitations are stated in full in Section 4.9 and summarised here. The product "
  "resolves no structure below its input grid. Training and validation both use ERA5, so "
  "the evaluation is out-of-sample in time but not independent of the reference; the "
  "comparison against SARAH in Section 4.6 quantifies that reference uncertainty at about "
  "3 per cent with a seasonal structure. A validation of the product itself against SARAH "
  "has not been performed, and Section 4.6.1 shows why re-scoring would not supply one: "
  "the ERA5 target sits %.2f W/m² from SARAH while the four architectures span %.2f, so "
  "such a score is dominated by the reference and cannot even separate a trained model "
  "from an interpolation. The observational validation therefore requires refitting "
  "against SARAH as the target, which Section 5.6 recommends first. Two exclusion "
  "criteria in the suitability analysis do not bind "
  "at 0.1 degrees. The suitability index carries no propagated uncertainty. And the "
  "future suitability maps hold infrastructure and population constant, which makes them "
  "statements about the resource at today's viable sites rather than about tomorrow's."
  % (spv5.loc["ERA5 target itself", "rmse_vs_sarah_zw"],
     max(spv5.loc[m, "rmse_vs_era5_zw"] for m in ("XGBoost", "Random Forest", "CNN", "U-Net"))
     - min(spv5.loc[m, "rmse_vs_era5_zw"] for m in ("XGBoost", "Random Forest", "CNN", "U-Net"))))

H("5.5 Recommendations for Policy and Practice")
P("**Grid extension opens more suitable land than resource refinement.** This follows "
  "directly from Section 4.8.1 and is the study's clearest planning implication. Solar "
  "resource over Zimbabwe is abundant and spatially uniform enough that it does not "
  "discriminate between candidate sites; transmission access does, by a factor of %.1f "
  "among robust sites. Planning effort and capital directed at network extension will "
  "enlarge the developable estate more than any improvement in resource estimation."
  % (g_a/g_r))
P("**Treat the robust set as the siting shortlist and the tier map as context.** The %d "
  "robust locations are those that survive every defensible weighting. Sites outside that "
  "set that rank highly under the primary weights may still be viable, but their ranking "
  "depends on a weighting choice that reasonable people could make differently."
  % n_rob)
P("**Do not plan 2100 siting from the projected maps.** They assume the transmission "
  "network, roads and population of the present. They support the statement that the "
  "resource at today's viable sites improves; they do not support a claim about where "
  "future sites should be.")
P("**Read the projected increase as favourable but second-order.** An increase of %.1f "
  "to %.1f per cent in mean irradiance by 2100 is a mild tailwind for solar investment "
  "and is smaller than the uncertainty attached to it. It is not a basis for "
  "distinguishing between sites, since it is spatially broad."
  % (100*lo_ch/base.mean(), 100*hi_ch/base.mean()))

H("5.6 Recommendations for Further Research")
P("**Retrain against SARAH.** This is the highest-value next step and the only route to "
  "a product that genuinely resolves sub-grid structure. The record is held at 0.05 "
  "degrees for 1985 to 2024 over the study domain, following the approach Buster et al. "
  "(2024) take for high-resolution solar resource data. It would convert the negative answer "
  "to RQ2 into a testable positive one and would simultaneously supply the independent "
  "validation the study currently lacks.")
P("**Search the remaining hyperparameters jointly, inside the training period.** The "
  "checkpoint-selection defect in the neural models has been corrected, the epoch count "
  "is chosen on an inner split, and Section 3.6.7 reports a learning-rate and "
  "penalty-weight search conducted entirely within the training record. What has not "
  "been done is a joint search over learning rate, gradient-penalty weight, dropout and "
  "architecture width together; the one-at-a-time sweep of Section 4.5 cannot see "
  "interactions between them, and the dropout result it appeared to show did not survive "
  "an honest retrain.")
P("**Settle what causes the U-Net's damping.** One part of this recommendation has since "
  "been carried out and is reported in Section 4.5. The dropout-free configuration that "
  "the sweep favoured was retrained under the deployed procedure and put through the "
  "scenario screen. It passes the screen, with separation growing from %+.3f to %+.3f "
  "W/m², but it is less accurate than the deployed model, %.2f W/m² against %.2f: the "
  "sweep's apparent gain came from a superseded learning rate and from keeping the "
  "checkpoint that scored best on the withheld record. What remains open is the cause of "
  "the spectral damping. The sweep's baseline carries the same dropout rate as the "
  "deployed model yet shows an excess of fine-scale power (%.2f) rather than a deficit, "
  "so it never reproduced the failure it was built to explain, and Section 4.5 withdraws "
  "the earlier attribution. Establishing the mechanism needs a sweep whose baseline "
  "reproduces the deployed model's spectra."
  % (sdv5.loc["U-Net, dropout 0", "sep_near_term"],
     sdv5.loc["U-Net, dropout 0", "sep_long_term"],
     udv5.loc["U-Net, dropout 0 (variant)", "RMSE_zw"],
     udv5.loc["U-Net, dropout 0.3 (deployed)", "RMSE_zw"],
     uo.loc["baseline", "test_spec_ratio"]))
P("**Extend the suitability analysis with infrastructure and demographic scenarios.** "
  "The present analysis can say how the resource changes at fixed sites. Answering where "
  "future sites should be requires projections of the transmission network and population "
  "distribution alongside the climate projections.")
P("**Propagate uncertainty onto the suitability index.** The irradiance uncertainty of "
  "Section 4.7, the reference uncertainty of Section 4.6 and the positional uncertainty "
  "of the infrastructure layers are all quantified or quantifiable, and none currently "
  "reaches the index.")
P("**Test the architecture-uncertainty finding elsewhere.** If architecture choice "
  "dominates GCM choice for solar downscaling over Zimbabwe, the question is whether that "
  "holds for other variables, regions and resolutions. It is the finding most worth "
  "generalising and the one this study can least confirm beyond its own domain.")

H("5.7 Concluding Remarks")
P("The study set out to downscale CMIP6 solar irradiance for Zimbabwe and to convert the "
  "result into a suitability assessment. It does both, and the more useful part of what "
  "it learned is about the limits of each step.")
P("The downscaling works in time and not in space, and no better than a linear "
  "regression on the same predictors: the fields carry real temporal skill against a "
  "climatology and essentially no information below their input resolution. Accuracy "
  "alone selects the deployed model; what the scenario screen added was the removal of "
  "the Random Forest, which accuracy had never chosen and which cannot separate the "
  "emission pathways it would be asked to project. The projection uncertainty is "
  "dominated not by the climate models but by the choice of downscaling architecture, a "
  "term most studies of this kind do not report because they fit only one. And the suitability analysis, pressed on its own "
  "assumptions, reduces from a map of Zimbabwe to %d locations, distinguished not by "
  "sunlight but by their distance from an existing power line."
  % n_rob)
P("Each of those is a narrower statement than the question that prompted it. Stated "
  "narrowly, each is defensible; stated broadly, none would be. For a resource assessment "
  "intended to inform capital allocation in a country with constrained generation "
  "capacity, that trade is the right one to make.")

# Same drafting note as Chapter 4's, removed for the same reason.

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("Saved %s" % OUT)
print("  %d paragraphs, ~%d words | robust %d of %d assessed"
      % (len(doc.paragraphs), sum(len(p.text.split()) for p in doc.paragraphs), n_rob, n_keep))
