"""Build Appendix C, the supporting tables, from the evaluation CSVs.

A pre-examiner reads the document and nothing else, so every result the thesis
rests on has to be in it. Fifteen evaluation CSVs fed no table: their findings
were quoted in prose, shown as a figure, or - in one case - promised and never
delivered. Section 3.4.3 says the per-cell quality-control flag counts are
"tabulated for the appendix", and no such table existed.

Each table here substantiates a claim the body already makes, and each is built
from the CSV rather than transcribed, so the appendix cannot drift from the
results the way Table 3.4 did. Captions are numbered C.1 upward so they cannot
collide with the chapter sequence.

The whole appendix is replaced on each run: it is generated output, and editing it
in Word would be overwritten.

    python build_appendix_c.py
"""

import os
import shutil
import sys

import docx
import numpy as np
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")
HEADING = "Appendix C: Supporting Tables"


def csv(name):
    p = os.path.join(EVAL, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def fmt(v, nd=4):
    """Format a cell, keeping counts as counts.

    numpy integers are not instances of int, so a value that came from
    Series.sum() took the float branch and a total of 36,381,312 printed as
    "36381312.0000" while the same quantity read from iterrows() - a Python int
    - printed correctly. Both integer paths are handled explicitly now.
    """
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "—"
    if isinstance(v, (int, np.integer)) or (
            isinstance(v, (float, np.floating)) and float(v).is_integer()
            and abs(float(v)) >= 1000):
        return "{:,}".format(int(v))
    return ("%%.%df" % nd) % float(v)


class Builder(object):
    def __init__(self, doc):
        self.doc = doc
        self.n = 0

    def para(self, text, size=11, italic=False):
        p = self.doc.add_paragraph()
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.italic = italic
        p.paragraph_format.line_spacing = 1.5
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return p

    def table(self, header, rows, caption):
        self.n += 1
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = self.doc.styles["Table Grid"]
        for c, h in zip(t.rows[0].cells, header):
            c.text = ""
            r = c.paragraphs[0].add_run(str(h))
            r.bold = True
            r.font.size = Pt(10)
        for row in rows:
            cells = t.add_row().cells
            for c, v in zip(cells, row):
                c.text = ""
                rr = c.paragraphs[0].add_run(str(v))
                rr.font.size = Pt(10)
        cp = self.doc.add_paragraph()
        cr = cp.add_run("Table C.%d. %s" % (self.n, caption))
        cr.font.size = Pt(10)
        cp.paragraph_format.line_spacing = 1.0
        return t


def build(b):
    b.para("This appendix prints the tables behind results the chapters state in "
           "prose, show as a figure, or rely on without reproducing. Every table is "
           "generated from the same evaluation outputs the chapters read, so the two "
           "cannot diverge. Nothing here is new analysis.")

    # C.1 quality control, promised in 3.4.3
    q = csv("qc_flag_counts.csv")
    if q is not None:
        g = (q.groupby("variable")
              .agg(values=("n_values", "sum"), flagged=("n_flagged", "sum"),
                   cells=("cells_affected", "max"),
                   worst=("worst_excursion_beyond_bound", "max"))
              .reset_index().sort_values("variable"))
        rows = [[r["variable"], fmt(r["values"]), fmt(r["flagged"]),
                 "%.4f" % (100.0 * r["flagged"] / r["values"]) if r["values"] else "—",
                 fmt(r["cells"]), "%.4f" % r["worst"]] for _, r in g.iterrows()]
        rows.append(["all variables", fmt(g["values"].sum()), fmt(g["flagged"].sum()),
                     "%.4f" % (100.0 * g["flagged"].sum() / g["values"].sum()),
                     "—", "%.4f" % g["worst"].max()])
        b.table(["Predictor", "Values checked", "Flagged", "Per cent", "Cells affected",
                 "Worst excursion"], rows,
                "Quality-control flag counts for the bias-corrected CMIP6 predictors, "
                "summed across models and scenarios, as referenced in Section 3.4.3. A "
                "flag records a value outside the physical bound for that variable.")

    # C.2 / C.3 pixel-wise searches
    for tag, label in (("xgb", "XGBoost"), ("rf", "Random Forest")):
        h = csv("hpo_pixelwise_%s.csv" % tag)
        if h is None:
            continue
        keys = [c for c in h.columns
                if c not in ("cv_rmse_csi", "is_deployed", "is_a_priori", "rank",
                             "in_grid")]
        # Ties on the score are common - the Random Forest grid has seven
        # pairs scoring identically - and sorting on the score alone let
        # rank 10 print above rank 9. Rank breaks the tie.
        h = h.sort_values(["cv_rmse_csi", "rank"])
        rows = []
        for _, r in h.iterrows():
            mark = ("deployed" if r.get("is_deployed") else
                    "a priori" if r.get("is_a_priori") else "")
            rows.append([int(r["rank"])] + [fmt(r[k], 2) for k in keys]
                        + ["%.5f" % r["cv_rmse_csi"], mark])
        b.table(["Rank"] + [k.replace("_", " ") for k in keys] + ["CV RMSE", ""], rows,
                "%s hyperparameter search, five-fold temporal cross-validation inside "
                "the training period on a 150-cell subsample, as described in Section "
                "3.6.7. Error is in clear-sky-index units." % label)

    # C.4 neural search
    n = csv("neural_hpo.csv")
    if n is not None:
        rows = [[r["model"], fmt(r["lr"], 4),
                 "—" if pd.isna(r.get("lambda_gp")) else fmt(r["lambda_gp"], 3),
                 "%.6f" % r["inner_select_mse"], int(r["selected_epoch"]),
                 "deployed" if r.get("is_deployed") else ""]
                for _, r in n.sort_values(["model", "inner_select_mse"]).iterrows()]
        b.table(["Model", "Learning rate", "Penalty weight", "Inner-split MSE",
                 "Epoch", ""], rows,
                "Neural hyperparameter search. Each candidate is fitted on 1985 to 2004 "
                "and selected on 2005 to 2010, so the withheld record is never used. "
                "Losses are not comparable between the two architectures. The epoch is "
                "the one minimising inner-split error, not a cap or a stopping point: "
                "the ceiling is 100 epochs and the early-stop patience 20. All three "
                "U-Net candidates minimise at epoch 22, which reflects a shallow and "
                "noisy basin rather than a fixed epoch: in the selected run the "
                "inner-split error is 0.099 at epoch 20 and 0.099 at epoch 30 against "
                "0.086 at the minimum, so the exact epoch is weakly determined.")

    # C.5 the U-Net sweep in full
    u = csv("unet_optimisation.csv")
    if u is not None:
        rows = [[r["variant"], fmt(r["cfg_dropout"], 2), r["cfg_loss"], r["cfg_clim"],
                 int(r["cfg_width"]), "%.3f" % r["inner_rmse"], "%.3f" % r["test_rmse"],
                 "%.4f" % r["test_spatial_r"], "%.3f" % r["test_spec_ratio"]]
                for _, r in u.sort_values("test_rmse").iterrows()]
        b.table(["Variant", "Dropout", "Loss", "Climatology", "Width", "Inner RMSE",
                 "Test RMSE", "Spatial r", "Share ratio k\u226510"], rows,
                "U-Net variant sweep in full, of which Section 4.5 discusses a subset. "
                "Fitted on 1985 to 2004 and selected on 2005 to 2010. The sweep carries "
                "its own baseline configuration, not the deployed one. The final column "
                "is a share ratio, each field's power beyond wavenumber 10 as a "
                "fraction of its own total, prediction over truth, which measures where "
                "a spectrum puts its power, not how much it retains, and is therefore "
                "not comparable with Table 4.6.")

    # C.6 resampling intervals in all three forms
    bca = csv("bca_intervals.csv")
    if bca is not None:
        rows = []
        for _, r in bca.iterrows():
            # Reading only bca_spans_zero printed "yes" for comparisons the
            # percentile and basic intervals both leave spanning zero, which is
            # the opposite of what publishing three forms is for.
            holds = [t for t, c in (("percentile", "pct_spans_zero"),
                                    ("basic", "basic_spans_zero"),
                                    ("BCa", "bca_spans_zero")) if not r[c]]
            if len(holds) == 3:
                verdict = "all three"
            elif not holds:
                verdict = "none"
            elif len(holds) == 1:
                verdict = "%s only" % holds[0]
            else:
                verdict = "%s" % " + ".join(holds)
            rows.append(["%s vs %s" % (r["model_a"], r["model_b"]), r["metric"],
                         "%+.4f" % r["plug_in"],
                         "%+.3f to %+.3f" % (r["pct_lo"], r["pct_hi"]),
                         "%+.3f to %+.3f" % (r["basic_lo"], r["basic_hi"]),
                         "%+.3f to %+.3f" % (r["bca_lo"], r["bca_hi"]),
                         verdict])
        b.table(["Comparison", "Metric", "Estimate", "Percentile", "Basic",
                 "BCa", "Excludes zero under"], rows,
                "Paired year-block resampling in all three interval forms, supporting "
                "the choice of the bias-corrected and accelerated interval in Section "
                "3.8.4, which also states why the form matters at fourteen blocks. The "
                "estimate is the first model named minus the second, so a positive value "
                "means the first has the larger value: on the error metrics (RMSE, "
                "centred RMSE, |MBE|, standard-deviation ratio deviation) that favours "
                "the second model, while on spatial correlation it favours the first. "
                "Signed mean bias is directional and orders neither model on its own. "
                "The final column reports which of the three forms exclude zero rather "
                "than the BCa verdict alone.")

    # C.7 spectra
    s = csv("power_spectra.csv")
    if s is not None:
        cols = [c for c in s.columns if c.startswith("CSI ")]
        rows = [[int(r["wavenumber"]), "%.1f" % r["wavelength_km"]]
                + ["%.4f" % (r[c] / r["CSI Truth (ERA5)"]) for c in cols
                   if c != "CSI Truth (ERA5)"]
                for _, r in s.iterrows()]
        b.table(["Wavenumber", "Wavelength (km)"]
                + [c.replace("CSI ", "") for c in cols if c != "CSI Truth (ERA5)"], rows,
                "Radially averaged power spectra in clear-sky-index space, each "
                "expressed as a ratio to the target's own power at the same wavenumber: "
                "the entry is that field's power at that wavenumber divided by the "
                "target's. Table 4.6 aggregates these, reporting the power retained "
                "beyond a cut as the target-weighted average of the ratios from that cut "
                "upward, so no entry in Table 4.6 can fall outside the range of the "
                "ratios here. The sweep in Table C.5 reports a different quantity again, "
                "a share ratio, and names it as such.")

    # C.8 PV yield by GCM
    p = csv("pv_temperature_derating.csv")
    if p is not None:
        col = "yield_change_pct_gamma-0.0040"
        rows = [[r["scenario"], r["period"].split("_")[0], r["gcm"],
                 "%+.2f" % r["ghi_change_pct"], "%+.2f" % r["tair_change_K"],
                 "%+.2f" % r[col]]
                for _, r in p.sort_values(["scenario", "period", "gcm"]).iterrows()]
        b.table(["Scenario", "Horizon", "Model", "ΔGHI (%)", "ΔT (K)", "ΔYield (%)"],
                rows,
                "Projected change in delivered photovoltaic yield by driving model, at "
                "the central temperature coefficient of −0.0040 per K, supporting the "
                "statement in Section 4.7.2 that every combination gives a positive "
                "change.")

    # C.9 perfect-prognosis transfer
    frames = []
    for tag, label in (("xgb", "XGBoost"), ("rf", "Random Forest")):
        t = csv("perfect_prognosis_transfer_%s.csv" % tag)
        if t is not None:
            t = t.copy()
            t.insert(0, "Model", label)
            frames.append(t)
    if frames:
        tr = pd.concat(frames, ignore_index=True)
        rows = [[r["Model"], r["predictor source"], "%.4f" % r["RMSE"],
                 "%+.4f" % r["MBE"], "%.4f" % r["spatial R"]]
                for _, r in tr.iterrows()]
        b.table(["Model", "Predictor source", "RMSE", "MBE", "Spatial r"], rows,
                "Perfect-prognosis transfer test, interpreted in Section 4.7.1. Each "
                "of the two pixel-wise models (XGBoost, which is deployed, and the "
                "Random Forest, which is not) is applied to bias-corrected predictors "
                "from each driving model over the historical period and scored against "
                "the ERA5 field it was trained to reproduce. The comparison is between "
                "per-cell, per-calendar-month climatologies, because individual CMIP6 "
                "months do not correspond to ERA5 months (Section 3.5). RMSE and MBE are "
                "in W/m\u00b2 of GHI; the ERA5 row is the floor the test can reach. The "
                "design assumes the mapping learned from reanalysis predictors transfers "
                "to model predictors, and this is the test of that assumption.")

    # C.10 elevation and irradiance
    e = csv("elevation_irradiance_gradient.csv")
    if e is not None:
        # The national correlation is one value per product, so as a column it
        # repeated down every band row. It belongs in the caption.
        rows = [[r["product"], r["band"], "{:,}".format(int(r["cells"])),
                 "%.1f" % r["ghi_mean"], "%.1f" % r["ghi_sd"], "%.0f" % r["elev_mean"]]
                for _, r in e.iterrows()]
        corr = ", ".join(
            "%+.3f in %s" % (g["corr_elev_ghi_national"].iloc[0], prod)
            for prod, g in e.groupby("product", sort=True))
        b.table(["Product", "Physiographic band", "Cells", "Mean GHI (W/m²)",
                 "SD (W/m²)", "Mean elevation (m)"], rows,
                "Observed annual-mean irradiance by physiographic band inside Zimbabwe, "
                "in both observed products, supporting the correction in Chapter 2. "
                "Across all national cells the correlation between elevation and "
                "irradiance is %s, so elevation explains little of the spatial pattern "
                "in either product." % corr)


def main():
    if os.path.exists(LOCK):
        print("The dissertation is open in Word. Close it first.")
        return 2
    d = docx.Document(DOC)
    before = (d.element.xml.count("ZOTERO_ITEM"),
              d.element.xml.count('w:fldCharType="begin"'),
              len(d.sections))

    body = d.element.body
    kids = list(body)
    start = None
    for i, ch in enumerate(kids):
        if ch.tag.endswith("}p"):
            from docx.text.paragraph import Paragraph
            p = Paragraph(ch, d)
            if p.style.name == "Heading 1" and p.text.strip() == HEADING:
                start = i
                break
    if start is not None:
        removed = 0
        for ch in kids[start:]:
            if any(True for _ in ch.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr")):
                continue
            ch.getparent().remove(ch)
            removed += 1
        print("replaced the existing appendix (%d elements removed)" % removed)

    d.add_heading(HEADING, level=1)
    b = Builder(d)
    build(b)
    print("built %d tables" % b.n)

    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_appendixc.docx"))
    d.save(DOC)
    nd = docx.Document(DOC)
    after = (nd.element.xml.count("ZOTERO_ITEM"),
             nd.element.xml.count('w:fldCharType="begin"'), len(nd.sections))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_appendixc.docx"), DOC)
        print("STRUCTURE CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("citations and sections intact %s; document now has %d tables"
          % (after, len(nd.tables)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
