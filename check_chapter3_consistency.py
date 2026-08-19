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

  FIELDS     Zotero field integrity. Chapter 3 carries 26 live citation fields,
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
DEFAULT = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS/"
    "CR_Madukwe_Chapter3_Final.docx")
CHAPTER = os.environ.get("CHAPTER3_PATH", DEFAULT)


def load():
    """(prose, xml) or (None, None) if the chapter or python-docx is unavailable."""
    try:
        import docx
    except ImportError:
        return None, None
    if not os.path.exists(CHAPTER):
        return None, None
    d = docx.Document(CHAPTER)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        parts += [c.text for r in t.rows for c in r.cells]
    return re.sub(r"\s+", " ", " ".join(parts)), d.element.xml


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
    return a


# Superseded text. Each was true of an earlier Chapter 3 and is now false.
RETIRED = [
    # figures that survived the deployment change
    ("9.11", "XGBoost RMSE while early-stopped on the validation set; now 9.24"),
    ("+1.26", "XGBoost mean bias before the early-stopping fix; now +1.20"),
    ("0.721", "XGBoost centred RMSE mis-transcribed; the Taylor value is 0.742"),
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
]

EXPLANATORY = (r"earlier version|previously|an earlier|was wrong|no longer|superseded|"
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


def check():
    prose, xml = load()
    if prose is None:
        return None, None, None

    missing = [(lab, v) for lab, v in canonical_anchors() if v not in prose]

    resurrected = []
    for phrase, why in RETIRED:
        for m in re.finditer(re.escape(phrase), prose):
            lo, hi = max(0, m.start() - WINDOW), min(len(prose), m.end() + WINDOW)
            if re.search(EXPLANATORY, prose[lo:hi], re.I):
                continue
            resurrected.append((phrase, why))

    return missing, resurrected, check_fields(xml)


def main():
    missing, resurrected, fields = check()
    print("=" * 84)
    print(" Chapter 3 consistency check")
    print("=" * 84)

    if missing is None:
        print("\nSKIPPED — chapter not found at:\n  %s" % CHAPTER)
        print("Set CHAPTER3_PATH to check a copy elsewhere.")
        return 0

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

    if fields:
        print("\nCITATION FIELD PROBLEMS (%d):" % len(fields))
        for f in fields:
            print("  " + f)
    else:
        print("Zotero citation fields are structurally intact.")

    print()
    return 1 if (missing or resurrected or fields) else 0


if __name__ == "__main__":
    sys.exit(main())
