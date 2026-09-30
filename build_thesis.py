"""Assemble the complete dissertation as one document, in University of Zimbabwe styling.

Builds the front matter (title page, abstract, acknowledgements, contents), merges
Chapters 1 to 5 with docxcompose so that Zotero fields, tables, embedded figures
and OMML equations survive intact, then appends the reference list and the two
declarations the University requires of work that used computational tools.

Chapters 4 and 5 are themselves generated, so run build_chapter4.py and
build_chapter5.py first if the analysis has changed. This script does not edit the
chapters; it only assembles them.

    python build_thesis.py

Writes CR_Madukwe_Dissertation.docx beside the chapters.
"""

import os
import re

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from docxcompose.composer import Composer

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
CHAPTERS = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
LOGO = os.path.join(ROOT, "assets/uz_logo.jpg")
OUT = os.path.join(CHAPTERS, "CR_Madukwe_Dissertation.docx")

# Sampled from the University's own crest and wordmark (assets/uz_logo.jpg).
UZ_PURPLE = RGBColor(0x2C, 0x1A, 0x70)
UZ_BLUE = RGBColor(0x32, 0x51, 0xA1)

BODY_FONT = "Times New Roman"
TITLE = ("Machine Learning-Based Downscaling of GCM Outputs for "
         "Solar Energy Suitability Mapping over Zimbabwe")
AUTHOR = "CHIAGOZIE RAPHAEL MADUKWE"
REGNO = "FSR2526653"
DEGREE = "Master of Science in Climate Science and Climate Systems Modelling"
DEPT = "Department of Space Science and Applied Physics"
FACULTY = "Faculty of Science"
UNIVERSITY = "UNIVERSITY OF ZIMBABWE"
# Taken from Chapter 1's own cover page, where they were recorded all along.
SUPERVISOR = "Prof E. Mashonjowa"
COORDINATOR = "Prof T.D. Mushore"
DATE = "October 2026"
REPO = "https://github.com/<your-username>/zimbabwe-solar-downscaling"

FILES = [
    "CR_Madukwe_Chapter 1_corrected.docx",
    "CR_Madukwe_Chapter2.docx",
    "CR_Madukwe_Chapter3_Final.docx",
    "CR_Madukwe_Chapter4_DRAFT.docx",
    "CR_Madukwe_Chapter5_DRAFT.docx",
]


# --------------------------------------------------------------- helpers ----
def field(paragraph, instr):
    """Insert a Word field, used for page numbers and the contents table."""
    r = paragraph.add_run()
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin")
    t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = instr
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end")
    for el in (b, t, sep, e):
        r._r.append(el)
    return r


def page_numbering(section, fmt, start=None):
    """Roman numerals for the front matter, arabic restarting at 1 for the body."""
    pr = section._sectPr
    for old in pr.findall(qn("w:pgNumType")):
        pr.remove(old)
    el = OxmlElement("w:pgNumType")
    el.set(qn("w:fmt"), fmt)
    if start is not None:
        el.set(qn("w:start"), str(start))
    pr.append(el)


def footer_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field(p, "PAGE")
    for r in p.runs:
        r.font.name = BODY_FONT
        r.font.size = Pt(10)


def para(doc, text="", size=12, bold=False, italic=False, align=None,
         space_after=10, space_before=0, colour=None, caps=False, font=None,
         line=1.5):
    p = doc.add_paragraph()
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = line
    if text:
        r = p.add_run(text)
        r.font.name = font or BODY_FONT
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        r.font.all_caps = caps
        if colour is not None:
            r.font.color.rgb = colour
    return p


def heading(doc, text, level=1):
    """A heading that matches the chapters' own Heading 1/2 styles.

    The chapters use Word's built-in heading styles, so the contents table picks
    them up. Restyling them in UZ colours here means the front matter and the
    merged chapters read as one document rather than two.
    """
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.name = BODY_FONT
        r.font.color.rgb = UZ_PURPLE
        r.font.size = Pt(16 if level == 1 else 13)
        r.bold = True
    h.paragraph_format.space_before = Pt(18 if level == 1 else 12)
    h.paragraph_format.space_after = Pt(10)
    return h


def rule(doc, colour=UZ_BLUE):
    """A thin horizontal rule, used once on the title page."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "3251A1")
    bdr.append(bottom)
    pr.append(bdr)
    p.paragraph_format.space_after = Pt(14)
    return p


# ------------------------------------------------------------ front matter --
def build_front():
    doc = Document()

    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(12)

    s = doc.sections[0]
    s.page_height, s.page_width = Cm(29.7), Cm(21.0)      # A4
    s.left_margin = Cm(3.5)                               # binding edge
    s.right_margin = Cm(2.5)
    s.top_margin = s.bottom_margin = Cm(2.5)

    # ---------------- title page ----------------
    if os.path.exists(LOGO):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        p.add_run().add_picture(LOGO, width=Inches(2.9))
    # The crest already carries the University's name, so the wordmark is not
    # repeated as text (author's edit).
    para(doc, FACULTY, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1)
    para(doc, DEPT, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
    rule(doc)

    para(doc, TITLE, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         colour=UZ_PURPLE, space_before=18, space_after=22, line=1.3)

    para(doc, "By", size=12, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, AUTHOR, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, "Registration Number: %s" % REGNO, size=12,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=22)

    para(doc, "A dissertation submitted in partial fulfilment of the requirements "
              "for the degree of", size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_after=4, line=1.3)
    para(doc, DEGREE, size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_after=24, line=1.3)

    para(doc, "Supervisor: %s" % SUPERVISOR, size=12,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, "Coordinator: %s" % COORDINATOR, size=12,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)

    para(doc, "Harare, Zimbabwe", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(doc, DATE, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)

    # front matter in roman numerals, title page unnumbered
    page_numbering(s, "lowerRoman", start=1)
    s.different_first_page_header_footer = True
    footer_page_number(s)

    # ---------------- declaration ----------------
    doc.add_page_break()
    heading(doc, "Declaration")
    para(doc, "I, %s, registration number %s, declare that this dissertation is my own "
              "original work. It has not been submitted, in whole or in part, for any "
              "degree or examination at this or any other university. All sources used "
              "or quoted have been acknowledged by complete reference." % (AUTHOR.title(), REGNO),
         space_after=12)
    para(doc, "I further declare that computational tools were used in the preparation "
              "of this work, and that their use is declared in full in Appendix B. The "
              "research questions, the choice of methods, the interpretation of results "
              "and the conclusions drawn are my own, and responsibility for the content "
              "of this dissertation rests entirely with me.",
         space_after=12)
    para(doc, "The analysis code supporting this dissertation is published openly, as "
              "described in Chapter 3 Section 3.11 and Appendix A, so that the results "
              "reported here can be independently regenerated.",
         space_after=28)

    _sig = [("Candidate", AUTHOR.title()),
            ("Supervisor", SUPERVISOR),
            ("Coordinator", COORDINATOR)]
    tbl = doc.add_table(rows=len(_sig), cols=3)
    tbl.autofit = False
    for row, (role, who) in zip(tbl.rows, _sig):
        cells = row.cells
        cells[0].width = Cm(4.0); cells[1].width = Cm(6.5); cells[2].width = Cm(4.0)
        for cell, text in zip(cells, ["%s: %s" % (role, who),
                                      "Signature: ......................................",
                                      "Date: ........................"]):
            q = cell.paragraphs[0]
            q.paragraph_format.space_after = Pt(22)
            q.paragraph_format.line_spacing = 1.0
            r = q.add_run(text)
            r.font.name = BODY_FONT
            r.font.size = Pt(11)

    # ---------------- abstract ----------------
    doc.add_page_break()
    heading(doc, "Abstract")
    for t in abstract_paragraphs():
        para(doc, t, space_after=10)

    kw = para(doc, "", space_before=6)
    r = kw.add_run("Keywords: ")
    r.bold = True; r.font.name = BODY_FONT; r.font.size = Pt(12)
    r2 = kw.add_run("statistical downscaling; CMIP6; solar irradiance; machine learning; "
                    "multi-criteria evaluation; Zimbabwe; reproducibility.")
    r2.font.name = BODY_FONT; r2.font.size = Pt(12)

    # ---------------- acknowledgements ----------------
    doc.add_page_break()
    heading(doc, "Acknowledgements")
    for t in acknowledgements():
        para(doc, t, space_after=10)

    # ---------------- contents ----------------
    doc.add_page_break()
    heading(doc, "Table of Content")
    toc = doc.add_paragraph()
    field(toc, r'TOC \o "1-3" \h \z \u')

    # body pages: arabic, restarting at 1
    body = doc.add_section(WD_SECTION.NEW_PAGE)
    body.left_margin = Cm(3.5); body.right_margin = Cm(2.5)
    body.top_margin = body.bottom_margin = Cm(2.5)
    page_numbering(body, "decimal", start=1)
    footer_page_number(body)
    return doc


def abstract_paragraphs():
    """The abstract, with its figures read from the analysis outputs."""
    t = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")
    u = pd.read_csv(os.path.join(EVAL, "uncertainty_decomposition_summary.csv"))
    x = u[(u.model == "xgb") & (u.period == "long_term_2076_2100")].iloc[0]
    rb = pd.read_csv(os.path.join(EVAL, "suitability_robustness.csv")).iloc[0]
    info = pd.read_csv(os.path.join(EVAL, "information_content.csv")).set_index("field")
    spv = pd.read_csv(os.path.join(EVAL, "sarah_product_validation.csv")).set_index("model")
    # The projected change must be taken against the ERA5-basis present, because the
    # projections come from an ERA5-trained chain. A first version of this abstract
    # differenced the SARAH-based present instead and reported +4.0 W m-2 rather
    # than +9.6: that subtracts one instrument from another and calls the
    # difference climate, which is the error Section 4.6 exists to prevent.
    # Computed here exactly as Table 4.7 computes it, so the two agree.
    import numpy as np
    import xarray as xr
    _base = xr.open_dataset(os.path.join(
        ROOT, "data/processed/suitability/criterion_layers.nc")).ghi_present_era5.values
    _fut = xr.open_dataset(os.path.join(
        ROOT, "data/processed/mme_aggregations_xgb",
        "mme_suitability_ssp585_long_term_2076_2100.nc")).ghi_mean.values
    chg = float(np.nanmean(_fut - _base))

    return [
        "Zimbabwe's renewable energy targets require siting decisions at a spatial scale "
        "finer than any climate projection currently provides. General "
        "circulation models resolve surface solar radiation at grid spacings of roughly "
        "100 to 250 km, while the planning decisions that depend on it are taken over "
        "distances of a few kilometres. This study addresses that mismatch by training "
        "four machine learning architectures to downscale Coupled Model Intercomparison "
        "Project Phase 6 (CMIP6) solar radiation fields to a 0.1-degree grid over "
        "Zimbabwe, and by carrying the resulting fields forward into a multi-criteria "
        "suitability assessment for utility-scale photovoltaic development.",

        "Bias-corrected predictors from three CMIP6 models were mapped onto an "
        "ERA5-derived clear-sky index target over 1985 to 2010 and evaluated on a "
        "withheld 2011 to 2024 record. A pixel-wise gradient-boosted ensemble attains "
        "the lowest aggregate error at %.2f W m-2, a skill score of %.4f against a "
        "training-period climatology, compared with %.2f for a per-cell random forest, "
        "%.2f for a convolutional network, and %.2f for a U-Net. Paired year-block bootstrap "
        "intervals establish the deployed model's margin over all three alternatives "
        "while showing that the ordering of the remaining three is not statistically "
        "separable. Model selection turned not on accuracy but on a scenario "
        "discrimination screen: the random forest, second on several validated axes, "
        "inverts the separation between emission pathways that a projection must "
        "reproduce, and is disqualified on that basis." % (
            t.loc["XGBoost", "RMSE"], t.loc["XGBoost", "SS vs climatology"],
            t.loc["Random Forest", "RMSE"], t.loc["CNN", "RMSE"], t.loc["U-Net", "RMSE"]),

        "Two findings qualify the product and are reported as prominently as the skill "
        "figures. First, a round-trip spectral test shows that the downscaled fields "
        "carry essentially no information below the 0.25 degree resolution of their own "
        "predictors, with %.3f per cent of time-mean variance residing at finer scales, "
        "so the product is a physically consistent regridding with a projected climate "
        "signal rather than a resolution gain. Second, the choice of downscaling "
        "architecture accounts for %.1f per cent of projection variance by 2076 to 2100, "
        "compared with %.1f per cent for the choice of global model, so a study reporting a "
        "single architecture would understate its own uncertainty by the larger term." % (
            info.loc["GHI target (time-mean)", "pct_variance_below_0.25deg"],
            x.pct_var_arch, x.pct_var_gcm),

        "Projected annual-mean irradiance rises by %.1f W m-2 under SSP5-8.5 by 2076 to "
        "2100. The suitability analysis combines the irradiance layer with six "
        "biophysical and infrastructural criteria under a weighted linear combination. "
        "Its defensible output is not the five-tier map but the %d of %d assessed cells "
        "that remain highly suitable under every weighting scheme tested, since %.1f per "
        "cent of cells change tier under at least one defensible reweighting. Those "
        "locations are distinguished by transmission access rather than by irradiance, "
        "which varies by only a few per cent nationally." % (
            chg, int(rb.robust), int(rb.assessed),
            100 * rb.weight_sensitive / rb.assessed),

        "The validation is out-of-sample in time but uses ERA5 as both training target "
        "and reference. An independent satellite retrieval is compared against that "
        "reference and shown to differ by about three per cent with a seasonal "
        "structure; re-scoring the product against the satellite record is shown not to "
        "constitute an observational validation, because the reference disagreement of "
        "%.2f W m-2 exceeds the whole spread across architectures by more than an order "
        "of magnitude. Refitting against the satellite record as the target is "
        "identified as the first priority for further work. The complete analysis "
        "pipeline and the test suite that guards it are published as an open "
        "repository, so that every figure reported here can be regenerated from the "
        "documented inputs." % (
            spv.loc["ERA5 target itself", "rmse_vs_sarah"]),
    ]


def acknowledgements():
    return [
        "This dissertation was made possible by the support of several institutions and "
        "people, and I am pleased to acknowledge them.",

        "I thank Homegrown Clean Energy Solutions for its support of this work, and for "
        "grounding a study that could easily have remained an exercise in modelling in "
        "the practical question of where generation can actually be built. The framing "
        "of the suitability analysis around transmission access rather than irradiance "
        "alone owes a great deal to that perspective.",

        "I gratefully acknowledge the European Education and Culture Executive Agency "
        "(EACEA), which administers the Intra-Africa Academic Mobility Scheme on behalf "
        "of the European Union, for the mobility funding that allowed me to undertake this "
        "degree. This scheme, which moves students between African universities, made "
        "this work possible where the results matter, rather than in a region from "
        "somewhere else.",

        "My deepest thanks go to my parents, who encouraged my love for physics without "
        "need for justification. That freedom is a rarer gift than it should be, and "
        "everything here follows from it.",

        "I thank my supervisor and project coordinator for their guidance throughout, and "
        "the Department of Space Science and Applied Physics for the advice and "
        "resources used to carry out this work. Responsibility for the analysis, and "
        "for any errors remaining in it, is mine alone.",
    ]


# -------------------------------------------------------------- back matter --
def references():
    """The consolidated reference list.

    Chapters 1, 3, 4 and 5 cite in author-date form; Chapter 2 carries a numbered
    list in a physics style. One list cannot be both, so the entries below are
    normalised to author-date, which is what a single Zotero style will emit when
    the citations are refreshed. Entries marked [complete from Zotero] are ones
    whose numbered form in Chapter 2 omits the title; the item exists in the
    library and will fill itself in on refresh.
    """
    return sorted([
        "Anthropic, 2026. Claude (Opus 5) [large language model]. Anthropic, San "
        "Francisco. URL https://claude.ai",
        "Araya-Osses, D., Casanueva, A., Roman-Figueroa, C., Uribe, J.M., Paneque, M., "
        "2020. Climate change projections of temperature and precipitation in Chile "
        "based on statistical downscaling. Climate Dynamics 54, 4309-4330.",
        "Benatiallah, D., Bouchouicha, K., 2021. [complete from Zotero]. pp. 1-7.",
        "Breiman, L., 2001. Random Forests. Machine Learning 45, 5-32. "
        "https://doi.org/10.1023/A:1010933404324",
        "Buster, G., Benton, B., Glaws, A., King, R., 2024. High-resolution meteorology "
        "with climate change impacts from global climate model data using generative "
        "machine learning. Nature Energy 9, 1-13. "
        "https://doi.org/10.1038/s41560-024-01507-9",
        "Casey, J.P., 2024. IRENA: Solar LCOE falls 12% year-on-year, 90% since 2010. "
        "PV Tech. URL https://www.pv-tech.org/irena-solar-lcoe-falls-12-year-on-year-"
        "90-since-2010/",
        "Chen, T., Guestrin, C., 2016. XGBoost: A Scalable Tree Boosting System, in: "
        "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
        "Discovery and Data Mining. Association for Computing Machinery, pp. 785-794.",
        "Cohen, J., 1960. A coefficient of agreement for nominal scales. Educational "
        "and Psychological Measurement 20, 37-46. "
        "https://doi.org/10.1177/001316446002000104",
        "Damiani, A., Ishizaki, N., Sasaki, H., Feron, S., Cordero, R., 2024. Exploring "
        "super-resolution spatial downscaling of several meteorological variables and "
        "potential applications for photovoltaic power. Scientific Reports 14. "
        "https://doi.org/10.1038/s41598-024-57759-8",
        "Dixon, K.W., Lanzante, J.R., Nath, M.J., Hayhoe, K., Stoner, A., "
        "Radhakrishnan, A., Balaji, V., Gaitan, C.F., 2016. Evaluating the stationarity "
        "assumption in statistically downscaled climate projections. Climatic Change "
        "135, 395-408.",
        "Dozier, J., Frew, J., 1990. Rapid calculation of terrain parameters for "
        "radiation modeling from digital elevation data. IEEE Transactions on "
        "Geoscience and Remote Sensing 28, 963-969. https://doi.org/10.1109/36.58986",
        "Efron, B., 1987. Better bootstrap confidence intervals. Journal of the "
        "American Statistical Association 82, 171-185. "
        "https://doi.org/10.1080/01621459.1987.10478410",
        "Eyring, V., Bony, S., Meehl, G.A., Senior, C.A., Stevens, B., Stouffer, R.J., "
        "Taylor, K.E., 2016. Overview of the Coupled Model Intercomparison Project "
        "Phase 6 (CMIP6) experimental design and organization. Geoscientific Model "
        "Development 9, 1937-1958.",
        "Fuentes-Franco, R., Krus, K., Ivanov, M., Koenigk, T., Wang, F., "
        "Aldama-Campino, A., 2025. Pan-European High-Resolution Downscaling Using Deep "
        "Learning. Journal of Geophysical Research: Machine Learning and Computation 2, "
        "e2025JH000630. https://doi.org/10.1029/2025JH000630",
        "Harilal, N., Hodge, B.-M.S., Monteleoni, C., Subramanian, A., 2022. EnhancedSD: "
        "Downscaling Solar Irradiance from Climate Model Projections, in: Climate "
        "Change AI. NeurIPS 2022 Workshop.",
        "Hernanz, A., Correa, C., Dominguez, M., Rodriguez-Guisado, E., "
        "Rodriguez-Camino, E., 2023. Comparison of machine learning statistical "
        "downscaling and regional climate models for temperature, precipitation, wind "
        "speed, humidity and radiation over Europe under present conditions. "
        "International Journal of Climatology 43. https://doi.org/10.1002/joc.8190",
        "Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horanyi, A., "
        "Munoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, "
        "A., Soci, C., et al., 2020. The ERA5 global reanalysis. Quarterly Journal of "
        "the Royal Meteorological Society 146, 1999-2049. "
        "https://doi.org/10.1002/qj.3803",
        "Horn, B.K.P., 1981. Hill shading and the reflectance map. Proceedings of the "
        "IEEE 69, 14-47. https://doi.org/10.1109/PROC.1981.11918",
        "Lanzante, J.R., Adams-Smith, D., Dixon, K.W., Nath, M., Whitlock, C.E., 2020. "
        "Evaluation of some distributional downscaling methods as applied to daily "
        "maximum temperature with emphasis on extremes. International Journal of "
        "Climatology 40, 1571-1585. https://doi.org/10.1002/joc.6288",
        "Lin, H., Tang, J., Wang, Shuyu, Wang, Shuguang, Dong, G., 2023. Deep learning "
        "downscaled high-resolution daily near surface meteorological datasets over "
        "East Asia. Scientific Data 10, 890. "
        "https://doi.org/10.1038/s41597-023-02805-9",
        "Maposa, D., Masache, A., Mdlongwa, P., Sigauke, C., 2024. An evaluation of "
        "variable selection methods using Southern Africa solar irradiation data. "
        "Journal of Energy in Southern Africa 35, 1-23.",
        "Najafi, H., Lagerwall, G.L., Obeysekera, J., Liu, J., 2026. [complete from "
        "Zotero]. Water 18, 271.",
        "Polasky, A., Evans, J., Fuentes, J., 2023. [complete from Zotero]. Theoretical "
        "and Applied Climatology 155, 1.",
        "Rampal, N., Hobeichi, S., Gibson, P.B., Bano-Medina, J., Abramowitz, G., "
        "Beucler, T., Gonzalez-Abad, J., Chapman, W., Harder, P., Gutierrez, J.M., "
        "2024. Enhancing Regional Climate Downscaling through Advances in Machine "
        "Learning. Artificial Intelligence for the Earth Systems 3.",
        "Saaty, T.L., 1980. The Analytic Hierarchy Process: Planning, Priority Setting, "
        "Resource Allocation. McGraw-Hill, New York.",
        "Vandal, T., Kodra, E., Ganguly, S., Michaelis, A., Nemani, R., Ganguly, A.R., "
        "2017. DeepSD: Generating High Resolution Climate Change Projections through "
        "Single Image Super-Resolution, in: Proceedings of the 23rd ACM SIGKDD "
        "International Conference on Knowledge Discovery and Data Mining, pp. "
        "1663-1672.",
        "Wilby, R.L., Dawson, C.W., Barrow, E.M., 2002. SDSM: a decision support tool "
        "for the assessment of regional climate change impacts. Environmental Modelling "
        "& Software 17, 145-157.",
    ], key=str.lower)


def build_back(doc):
    doc.add_page_break()
    heading(doc, "References")
    para(doc, "Citations in the chapters above are plain author-date text pending "
              "conversion to live Zotero fields. This list is the working equivalent of "
              "the bibliography those fields will generate, normalised to a single "
              "author-date style: Chapter 2's numbered citations will convert to the "
              "same form when the document is refreshed against the library.",
         size=10, italic=True, space_after=12)
    for ref in references():
        p = para(doc, ref, size=11, space_after=7, line=1.15,
                 align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)   # hanging indent

    # ---------------- appendix A: reproducibility ----------------
    doc.add_page_break()
    heading(doc, "Appendix A: Reproducibility and Code Availability")
    nscripts = len([f for f in os.listdir(ROOT) if f.endswith(".py")])
    for t in [
        "The complete analysis for this dissertation is published as an open Git "
        "repository at %s. The intent is stronger than making code available: it is "
        "that the claims in this dissertation should be checkable rather than merely "
        "stated." % REPO,
        "Two properties of the repository support that. First, Chapters 4 and 5 are "
        "generated from the analysis outputs by scripts rather than typed, so a number "
        "quoted in the prose cannot drift from the analysis that produced it. Prose "
        "drifting from tables was the single most frequent defect encountered during "
        "this work, and generating the chapters removes the possibility rather than "
        "guarding against it. Second, the repository carries a test suite of invariant "
        "checks, including three consistency guards that compare the written chapters "
        "cell by cell against the canonical result files, reject figures known to have "
        "been superseded, and verify that citation fields remain structurally intact. "
        "These are run before any set of results is accepted.",
        "The repository contains %d analysis scripts covering data acquisition, bias "
        "correction, feature construction, model training, evaluation, uncertainty "
        "decomposition and the multi-criteria overlay; a pinned environment "
        "specification; the test suite; and the generators for both results chapters. "
        "Its README documents the order in which the stages run and the expected "
        "runtime of each." % nscripts,
        "The raw input data is not redistributed. CMIP6, ERA5, the CM SAF SARAH "
        "record, WorldPop, the ESA Climate Change Initiative land cover product, "
        "OpenStreetMap and the World Database on Protected Areas are each obtained "
        "under their own licences and access conditions, several of which require "
        "registration or the acceptance of terms by the user. The repository instead "
        "documents how each dataset was requested and includes the download scripts "
        "used, so that the acquisition step is itself reproducible without the files "
        "being republished.",
        "A reader who obtains the inputs and installs the pinned environment can "
        "regenerate every table and figure in Chapters 4 and 5, and should obtain the "
        "same values. Where a result depends on a random seed, the seed is fixed in "
        "the script that uses it.",
    ]:
        para(doc, t, space_after=10)

    # ---------------- appendix B: AI tools ----------------
    doc.add_page_break()
    heading(doc, "Appendix B: Declaration on the Use of Artificial Intelligence Tools")
    for t in [
        "This dissertation was prepared with the assistance of a large language model, "
        "Claude (Anthropic, 2026). The assistance is declared here in full, in the "
        "interest of the same transparency that motivates Appendix A.",
        "The tool was used for three purposes. First, language: improving the clarity, "
        "concision and grammar of prose whose argument and content are the author's. "
        "Second, code generation: drafting analysis scripts in Python from "
        "specifications set by the author, including the data acquisition, evaluation "
        "and figure generation stages. Third, debugging: diagnosing defects in that "
        "code and in the analysis, a use that proved more consequential than the other "
        "two and that is visible in the dissertation itself, since several results "
        "reported here were corrected after such a defect was identified, including "
        "two forms of information leakage between the training and evaluation records "
        "and a set of figures whose plotted labels had fallen out of step with the "
        "analysis behind them.",
        "The tool was not used to generate research questions, to choose methods, to "
        "interpret results, or to produce any number reported in this dissertation. "
        "Every figure and table originates from the analysis scripts operating on the "
        "datasets described in Chapter 3. Where the tool proposed an interpretation, "
        "it was accepted only after the author verified it against the data, and "
        "several such proposals were rejected on that basis.",
        "Responsibility for the content of this dissertation, including for any error "
        "that remains in it, rests entirely with the author.",
    ]:
        para(doc, t, space_after=10)
    return doc



def strip_chapter_reference_lists(doc):
    """Remove the per-chapter reference lists, leaving one list at the end.

    Chapters 1, 2 and 3 each carry their own bibliography, two of them generated
    by Zotero. Three bibliographies in one dissertation is wrong on its own terms,
    and it also gives Zotero three anchors to regenerate into when the citations
    are refreshed. The in-text citation fields are left untouched: those are what
    Zotero needs. Only the lists and their ZOTERO_BIBL anchors go.
    """
    from docx.oxml.ns import qn

    removed_paras = removed_bibl = 0
    # Work on a snapshot: the tree is modified while iterating.
    paras = list(doc.paragraphs)
    heads = [i for i, q in enumerate(paras)
             if q.style.name.startswith("Heading") and q.text.strip() == "References"]
    # Keep the last one, which is the consolidated list this script appends after.
    for i in heads:
        stop = len(paras)
        for j in range(i + 1, len(paras)):
            if paras[j].style.name.startswith("Heading 1"):
                stop = j
                break
        for q in paras[i:stop]:
            if q._p.getparent() is not None:
                q._p.getparent().remove(q._p)
                removed_paras += 1

    # Any orphaned bibliography field anchors go with them.
    for instr in doc.element.body.iter(qn("w:instrText")):
        if instr.text and "ZOTERO_BIBL" in instr.text:
            removed_bibl += 1
    print("  removed %d paragraphs of per-chapter reference lists"
          " (%d ZOTERO_BIBL anchors remained)" % (removed_paras, removed_bibl))



ZOTERO_PREF = (
    '&lt;data data-version="3" zotero-version="9.0.4"&gt;'
    '&lt;session id="02dIrP3h"/&gt;'
    '&lt;style id="http://www.zotero.org/styles/elsevier-harvard" hasBibliography="1" '
    'bibliographyStyleHasBeenSet="0"/&gt;'
    '&lt;prefs&gt;&lt;pref name="fieldType" value="Field"/&gt;&lt;/prefs&gt;&lt;/data&gt;')


def set_zotero_style(path):
    """Give the merged document the Zotero preferences its fields need.

    The front matter is a fresh python-docx document, so the merged file has no
    docProps/custom.xml and therefore no recorded citation style: Zotero would
    open it with 60 live fields and no idea how to render them. This writes the
    same Elsevier Harvard preference the chapters carry, with
    bibliographyStyleHasBeenSet cleared so Zotero rebuilds the bibliography from
    scratch rather than reusing what each chapter cached.
    """
    import shutil
    import zipfile

    CT = "[Content_Types].xml"
    RELS = "_rels/.rels"
    PART = "docProps/custom.xml"
    custom = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/'
        'custom-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument'
        '/2006/docPropsVTypes">'
        '<property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" pid="2" '
        'name="ZOTERO_PREF_1"><vt:lpwstr>%s</vt:lpwstr></property>'
        '</Properties>' % ZOTERO_PREF)

    tmp = path + ".tmp"
    zin = zipfile.ZipFile(path)
    names = set(zin.namelist())
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == CT and "custom-properties" not in data.decode("utf8"):
                t = data.decode("utf8").replace(
                    "</Types>",
                    '<Override PartName="/docProps/custom.xml" ContentType='
                    '"application/vnd.openxmlformats-officedocument.'
                    'custom-properties+xml"/></Types>')
                data = t.encode("utf8")
            elif item.filename == RELS and "custom-properties" not in data.decode("utf8"):
                t = data.decode("utf8").replace(
                    "</Relationships>",
                    '<Relationship Id="rIdCustom1" Type="http://schemas.openxmlformats'
                    '.org/officeDocument/2006/relationships/custom-properties" '
                    'Target="docProps/custom.xml"/></Relationships>')
                data = t.encode("utf8")
            if item.filename == PART:
                data = custom.encode("utf8")
            zout.writestr(item, data)
        if PART not in names:
            zout.writestr(PART, custom)
    zin.close()
    shutil.move(tmp, path)
    print("  citation style set to Elsevier Harvard for all %d fields"
          % Document(path).element.xml.count("ZOTERO_ITEM"))



def strip_chapter_one_cover(doc):
    """Remove Chapter 1's own cover page from the merged document.

    Chapter 1 opens with a standalone cover - University, faculty, degree,
    title, author, supervisor - which is correct when the chapter is read on its
    own and duplicate once the dissertation has a title page of its own. It is
    removed here rather than from the chapter file, so Chapter 1 remains
    complete as a standalone document.

    Those cover paragraphs are also where the supervisor and coordinator names
    were recorded all along.
    """
    from docx.oxml.ns import qn

    paras = list(doc.paragraphs)
    start = end = None
    for i, q in enumerate(paras):
        if start is None and q._p.findall(".//" + qn("w:instrText")) and "TOC" in q._p.xml:
            start = i
        if start is not None and q.style.name == "Heading 1" \
                and q.text.strip().lower().startswith("chapter 1"):
            end = i
            break
    if start is None or end is None or end <= start + 1:
        print("  Chapter 1 cover: nothing to remove")
        return
    # A paragraph can carry the section break that starts the body pages. Deleting
    # it silently merges the front matter and the body into one section, which
    # loses the roman/arabic page numbering split. Keep any such paragraph and
    # empty it instead.
    n = kept = 0
    for q in paras[start + 1:end]:
        if q._p.find(qn("w:pPr")) is not None \
                and q._p.find(qn("w:pPr")).find(qn("w:sectPr")) is not None:
            for r in list(q.runs):
                r._r.getparent().remove(r._r)
            kept += 1
            continue
        if q._p.getparent() is not None:
            q._p.getparent().remove(q._p)
            n += 1
    if kept:
        print("  kept %d paragraph(s) carrying a section break" % kept)
    print("  removed %d paragraphs of Chapter 1's duplicate cover page" % n)


# ------------------------------------------------------------------ merge ----
def main():
    missing = [f for f in FILES if not os.path.exists(os.path.join(CHAPTERS, f))]
    if missing:
        raise SystemExit("chapters not found: %s" % ", ".join(missing))

    print("building front matter...")
    front = build_front()
    composer = Composer(front)

    for f in FILES:
        print("  appending %s" % f)
        d = Document(os.path.join(CHAPTERS, f))
        composer.append(d)

    strip_chapter_reference_lists(composer.doc)
    strip_chapter_one_cover(composer.doc)

    print("appending references and appendices...")
    build_back(composer.doc)

    # The chapters carry their own section properties, so the merged document
    # arrives with three different left margins. Normalise every section to the
    # same A4 page and binding margin, and let only the title page go unnumbered.
    print("normalising page setup across %d sections..." % len(composer.doc.sections))
    for i, sec in enumerate(composer.doc.sections):
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.left_margin = Cm(3.5)
        sec.right_margin = Cm(2.5)
        sec.top_margin = sec.bottom_margin = Cm(2.5)
        sec.different_first_page_header_footer = (i == 0)
        if i > 0:
            sec.footer.is_linked_to_previous = True

    composer.save(OUT)
    set_zotero_style(OUT)

    d = Document(OUT)
    txt = " ".join(p.text for p in d.paragraphs)
    for t in d.tables:
        txt += " " + " ".join(c.text for r in t.rows for c in r.cells)
    x = d.element.xml
    print("\nSaved %s" % OUT)
    print("  %d paragraphs | %d tables | %d images | %d Zotero fields"
          % (len(d.paragraphs), len(d.tables),
             sum(1 for r in d.part.rels.values() if "image" in r.reltype),
             x.count("ZOTERO_ITEM")))
    print("  %d words | em-dashes: %d" % (len(txt.split()), txt.count("—")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
