"""Guard Chapter 3 against drift — the last unguarded document.

The brief and PROJECT_STATUS each have a checker. Chapter 3 had none, and it is
the document an examiner actually reads. Everything found in it this cycle was
found by eye: three stale figures (RMSE 9.11, MBE +1.26, centred RMSE 0.721)
that survived a deployment change, and two design claims describing a Model
Output Statistics pipeline that was never built.

Three checks, deliberately different in kind:

  ANCHORS    canonical values, read from the evaluation CSVs. Chapter 3 quotes
             all of them, so a rerun that moves a metric fails here until the
             chapter follows.

  RETIRED    superseded SENTENCES and figures. The part no numeric check can
             do: "would contaminate the ML training process" is not a wrong
             number, it is a wrong claim about the design, and it sat in §3.4.4
             through every numeric correction.

  FIELDS     Zotero field integrity. Chapter 3 carries 25 live citation fields
             (the docstring said 26 until the count was actually taken),
             and the edits in this project are run-level surgery. A mismatched
             begin/end pair or an orphaned field silently breaks the
             bibliography, and nothing else would notice.

Chapter 3 lives outside the repository, so the path is configurable and every
check skips cleanly when the file or python-docx is absent:

    python check_chapter3_consistency.py
    CHAPTER3_PATH=/some/other/Chapter3.docx python check_chapter3_consistency.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
# The merged dissertation, not the standalone chapter. Once reference management
# began, every edit went into the merged file and the chapter sources stopped
# being updated; a guard pointed at CR_Madukwe_Chapter3_Final.docx was therefore
# checking a document nobody writes to, and would pass while the thesis drifted.
DEFAULT = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS/"
    "CR_Madukwe_Dissertation.docx")
CHAPTER = os.environ.get("CHAPTER3_PATH", DEFAULT)

# Why the guard skipped, set by load(). A guard that skips silently is worse
# than no guard: it prints a clean bill of health for a document it never
# opened. Distinguishing the two causes matters because they have opposite
# meanings - an absent chapter is a legitimate skip, a missing python-docx is
# a broken environment that must not exit 0.
SKIP_REASON = None
MISSING_DEP = False


def load():
    """(prose, xml, tables) or a triple of None if unavailable.

    Tables come back as a list of row-lists so cells can be checked
    individually. Flattening them into the prose is not enough: Section 3.7.4's
    figures also appear in the paragraph beneath the table, so a corrupted cell
    passes a substring search over the whole document. That is the same defect
    the brief guard had, and the fix is the same - anchor the cell, not the
    number.
    """
    global SKIP_REASON, MISSING_DEP
    try:
        import docx
    except ImportError:
        MISSING_DEP = True
        SKIP_REASON = ("python-docx is not installed in this interpreter (%s).\n"
                       "  That is an environment fault, not a missing chapter: "
                       "run the guard in the\n  project environment."
                       % sys.executable)
        return None, None, None
    if not os.path.exists(CHAPTER):
        SKIP_REASON = ("chapter not found at:\n  %s\n"
                       "  Set CHAPTER3_PATH to check a copy elsewhere." % CHAPTER)
        return None, None, None
    d = docx.Document(CHAPTER)
    parts = [p.text for p in d.paragraphs]
    grids = []
    for t in d.tables:
        rows = [[re.sub(r"\s+", " ", c.text).strip() for c in r.cells] for r in t.rows]
        grids.append(rows)
        parts += [c for r in rows for c in r]
    return re.sub(r"\s+", " ", " ".join(parts)), d.element.xml, grids


def canonical_anchors():
    """(label, value) pairs the chapter must contain, read from the CSVs."""
    import pandas as pd
    a = []

    t33 = os.path.join(EVAL, "table_3_3.csv")
    if os.path.exists(t33):
        df = pd.read_csv(t33).set_index("model")
        a.append(("deployed XGBoost RMSE", "%.2f" % df.loc["XGBoost", "RMSE"]))
        a.append(("Random Forest RMSE", "%.2f" % df.loc["Random Forest", "RMSE"]))

    tay = os.path.join(EVAL, "taylor_diagram_stats.csv")
    if os.path.exists(tay):
        df = pd.read_csv(tay).set_index("model")
        for m in ["Random Forest", "XGBoost", "CNN", "U-Net"]:
            if m in df.index:
                a.append(("%s centred RMSE" % m, "%.3f" % df.loc[m, "centered_rmse"]))

    bca = os.path.join(EVAL, "bca_intervals.csv")
    if os.path.exists(bca):
        df = pd.read_csv(bca)
        key = df[(df.model_a == "Random Forest") & (df.model_b == "XGBoost")]
        for metric, label in [("centred RMSE", "BCa centred RMSE"),
                              ("|MBE|", "BCa mean bias"),
                              ("RMSE", "BCa aggregate RMSE")]:
            hit = key[key.metric == metric]
            if not hit.empty:
                a.append((label, "%.3f" % abs(hit.iloc[0]["plug_in"])))

    # Section 3.7.4's ablation. Configurations A and B are one-off refits that
    # no script re-emits, so unlike everything above these cannot be read from a
    # CSV and are transcribed from Section 7.3 of PROJECT_STATUS.md. They are
    # anchored anyway: the chapter now rests an argument on them, and a silent
    # divergence between the two documents is exactly the drift this file
    # exists to catch. Configuration C is already covered by the Table 3.3
    # anchors above.
    a += [("ablation A RMSE", "22.94"),
          ("ablation A Pearson R", "0.817"),
          ("ablation B RMSE", "5.54"),
          ("ablation B Pearson R", "0.9894"),
          ("ablation B skill", "0.7096")]
    return a


# Superseded text. Each was true of an earlier Chapter 3 and is now false.
RETIRED = [
    # figures that survived the deployment change
    ("9.11", "XGBoost RMSE while early-stopped on the validation set; now 9.24"),
    ("+1.26", "XGBoost mean bias before the early-stopping fix; now +1.20"),
    ("0.721", "XGBoost centred RMSE mis-transcribed; the Taylor value is 0.742"),
    # the AHP elicitation that was never held (Section 12d)
    ("pairwise comparison matrix completed by the researcher",
     "no pairwise matrix was elicited or retained; 3.9.4 now says so and reports no CR"),
    ("consistency ratio of the pairwise comparison matrix is computed and verified",
     "unevidenced - the matrix is not in the archive; 3.9.4 reports no consistency ratio"),
    ("confirming that the weight assignments are internally consistent",
     "no consistency ratio supports this; Section 3.9.6's sensitivity analysis is the test"),
    ("Criterion Weighting Using AHP",
     "the heading claimed a procedure 3.9.4 states was not performed; now 'Criterion Weighting'"),
    # scope claims the analysis overtook (Section 12l)
    ("remains to be applied to the downscaled products",
     "3.9 was applied; Chapter 4 Section 4.8 reports it"),
    ("but it is not analysed here",
     "SARAH is analysed: 3.9.1 uses it as the present-day layer, 4.6 and 4.6.1 score against it"),
    # method descriptions that outlived the code (Section 12j)
    ("early stopping on the validation split with a patience of 50 rounds",
     "XGBoost uses a fixed 200 rounds; early stopping on the validation split was 6.12's leakage"),
    ("early stopping on the validation MSE with a patience of 20 epochs",
     "the U-Net now selects its epoch count on an inner 2005-2010 split, not the withheld record"),
    ("These were not extracted in the present implementation",
     "RF importances were extracted; feature_importance.csv and Section 4.7 report them"),
    ("As with the Random Forest, these were not extracted in the present implementation",
     "XGBoost gain and cover were extracted too"),
    ("a Spearman rank correlation of 0.860",
     "computed on layers that no longer exist; layer_choice_sensitivity.csv gives 0.875"),
    ("27.2 per cent of cells falling on opposite sides",
     "recomputed on the assessed cells as 15.5 per cent"),
    # the AHP label, renamed throughout once the procedure was withdrawn
    ("initial AHP weights", "Table 3.5's caption now says 'initial criterion weights'"),
    ("initial AHP weight", "3.9.1 now says 'its initial weight'"),
    ("Initial AHP Weight", "Table 3.5's column header is now 'Initial Weight'"),
    ("primary AHP weights", "3.9.6 now says 'the primary weights of Table 3.5'"),
    ("primary AHP classification", "3.9.6 now says 'the primary classification'"),
    # deployment claims
    ("the pixel-wise Random Forest is deployed", "XGBoost is deployed"),
    ("deployed for the projection stage is the pixel-wise Random Forest", "XGBoost is deployed"),
    # selection-criterion claims, both superseded directions
    ("neither leg of the composite criterion", "both legs survive under BCa"),
    ("on one axis of two rather than none", "both legs survive"),
    ("Both intervals contain zero", "the paired differences are established under BCa"),
    ("both intervals span zero",
     "the same claim in different words; the paired MBE difference is established at -0.953"),
    ("approximately 1° to 2.5°",
     "measured native range is 0.94 to 1.88 degrees across the three GCMs"),
    ("downscale CMIP6 GCM solar radiation outputs to ERA5 resolution",
     "the product is 0.1 deg, finer than ERA5's 0.25 deg; RQ1 now says 0.1 deg against an ERA5-derived target"),
    ("Models INJECT small-scale power",
     "the GHI-space reading Section 7.8 overturned; judge in CSI space"),
    ("spatial upscaling factor of approximately 23x",
     "conflates the whole chain with the ML step; the ML bridges 0.25 to 0.1, a factor of 2.5"),
    # the MOS framing that described a pipeline never built
    ("would contaminate the ML training process",
     "CMIP6 is not a training input; the biases are carried into the projections"),
    ("independently before model training",
     "correction is applied before projection, not before training"),
    # scope and framing errors corrected elsewhere
    ("calibrated during the historical period (1985–2024)",
     "calibrated on 1985-2010, evaluated on a withheld 2011-2024 record"),
    ("closes all four gaps", "three closed, the fourth addressed in part"),
    ("per-cell flag counts were not separately tabulated",
     "they are tabulated, and are identically zero for ERA5"),
    # predictor-count claims: both described configuration B of the Section
    # 3.7.4 ablation - the circular one - as the deployed design
    ("C = 6 atmospheric predictors",
     "the deployed CNN takes C = 5; rsds is excluded per Section 3.5.1"),
    ("were used directly as atmospheric predictor features",
     "five of the six are model inputs; rsds is excluded per Section 3.5.1"),
    ("determines the physical meaning of the intermediate CSI",
     "it does not; the ceiling averages 13 daytime hours against a 24-hour "
     "numerator, so the ratio runs 1.724 below a true clear-sky index"),
    ("direct radiation flux (rsds)",
     "rsds is not a model input; listing it among the selected predictors "
     "describes configuration B"),
]

EXPLANATORY = (r"earlier version|previously|an earlier|was wrong|no longer|superseded|"
               r"overturned|retired|contradicted|"
               r"rather than|instead of|not\s+what|this section|had been")
WINDOW = 300


def check_fields(xml):
    """Zotero/field integrity: begin and end counts must match, and every
    ZOTERO_ITEM instruction must sit inside a field."""
    problems = []
    begins = xml.count('w:fldCharType="begin"')
    ends = xml.count('w:fldCharType="end"')
    if begins != ends:
        problems.append("field begin/end mismatch: %d begins, %d ends — a citation "
                        "field is broken" % (begins, ends))
    items = xml.count("ZOTERO_ITEM")
    if items and begins < items:
        problems.append("%d ZOTERO_ITEM instructions but only %d field begins — "
                        "orphaned citation" % (items, begins))
    return problems


def _explained(text, start, end):
    """Is the retired phrase being DISCUSSED as a correction, or asserted?

    Two signals, and the first is the reliable one.

    QUOTED. Every legitimate mention of a retired phrase in these documents
    quotes it - 'the old title said "Models INJECT small-scale power"' - while a
    recurrence asserts it bare. Quoting is what distinguishes use from mention,
    and it does not depend on how near some explanatory word happens to fall.

    EXPLANATORY IN THE SAME SENTENCE. A narrow fallback for unquoted discussion.
    Two wider versions were tried and both suppressed a real recurrence: a
    260-character window, because the unrelated words "superseded sentences" -
    describing what these guards do - sat 48 characters away; and a
    two-sentence lookback, for the same reason. If an explanation spans
    sentences, quote the phrase; that is better writing anyway.
    """
    before = text[max(0, start - 4):start]
    after = text[end:end + 4]
    QUOTES = '"\u201c\u201d\u2018\u2019\'*_'
    if any(c in QUOTES for c in before) and any(c in QUOTES for c in after):
        return True

    lo = max(text.rfind('. ', 0, start), text.rfind('\n', 0, start)) + 1
    hi = text.find('. ', end)
    hi = len(text) if hi == -1 else hi + 1
    return re.search(EXPLANATORY, text[lo:hi], re.I) is not None

# Section 3.7.4, by row label -> (RMSE, Pearson R, skill). Section 7.3 of
# PROJECT_STATUS.md is the source; configuration C must also match Table 3.3.
ABLATION = {
    "A.": ("22.94", "0.817", "-0.203"),
    "B.": ("5.54", "0.9894", "0.7096"),
    "C.": ("9.24", "0.9707", "0.5155"),
}


def check_ablation_table(grids):
    """Anchor Section 3.7.4's cells individually."""
    for rows in grids:
        if not rows or not rows[0] or not rows[0][0].startswith("Configuration"):
            continue
        problems = []
        seen = set()
        for row in rows[1:]:
            tag = row[0][:2]
            if tag not in ABLATION:
                continue
            seen.add(tag)
            for got, want, col in zip(row[1:4], ABLATION[tag],
                                      ("RMSE", "Pearson R", "skill")):
                if got != want:
                    problems.append("Section 3.7.4 config %s %s is %r, expected %r"
                                    % (tag[0], col, got, want))
        for tag in sorted(set(ABLATION) - seen):
            problems.append("Section 3.7.4 is missing configuration %s" % tag[0])
        return problems
    return ["Section 3.7.4's ablation table is missing"]


def check():
    prose, xml, grids = load()
    if prose is None:
        return None, None, None, None

    missing = [(lab, v) for lab, v in canonical_anchors() if v not in prose]

    resurrected = []
    for phrase, why in RETIRED:
        for m in re.finditer(re.escape(phrase), prose):
            if _explained(prose, m.start(), m.end()):
                continue
            resurrected.append((phrase, why))

    return missing, resurrected, check_fields(xml), check_ablation_table(grids)


def main():
    missing, resurrected, fields, ablation = check()
    print("=" * 84)
    print(" Chapter 3 consistency check")
    print("=" * 84)

    if missing is None:
        print("\nSKIPPED — %s" % SKIP_REASON)
        # A missing dependency means nothing was verified. Exit non-zero so it
        # cannot be mistaken for a pass.
        return 1 if MISSING_DEP else 0

    if missing:
        print("\nCANONICAL VALUES ABSENT (%d):" % len(missing))
        for lab, v in missing:
            print("  %-34s expected to find '%s'" % (lab, v))
    else:
        print("\nEvery canonical value from the evaluation CSVs appears in the chapter.")

    if resurrected:
        print("\nSUPERSEDED TEXT PRESENT (%d):" % len(resurrected))
        for phrase, why in resurrected:
            print("  '%s'\n      %s" % (phrase, why))
    else:
        print("No superseded figure or claim appears outside an explanatory context.")

    if ablation:
        print("\nSECTION 3.7.4 ABLATION TABLE (%d):" % len(ablation))
        for a_ in ablation:
            print("  " + a_)
    else:
        print("Section 3.7.4's ablation table matches PROJECT_STATUS Section 7.3.")

    if fields:
        print("\nCITATION FIELD PROBLEMS (%d):" % len(fields))
        for f in fields:
            print("  " + f)
    else:
        print("Zotero citation fields are structurally intact.")

    print()
    return 1 if (missing or resurrected or fields or ablation) else 0


if __name__ == "__main__":
    sys.exit(main())
