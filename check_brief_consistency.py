"""Guard the interview brief against drift, in numbers and in sentences.

PROJECT_STATUS.md has had check_status_consistency.py since the fourth audit
round. The brief had nothing, across three formats, and it is the document that
actually gets read in the room. Six rounds of review found the same failure
repeatedly, and the second half of it is the part a number-only check misses:

  ANCHORS    values that must appear, read from the canonical evaluation CSVs.
             Catches a rerun that moves a metric without the brief following.

  RETIRED    superseded SENTENCES that must not reappear. This is the important
             half. Rounds three to six each found a table corrected while a
             sentence summarising it was left behind - "neither leg survived",
             "one real point", "every severe bug was silent", "beaten by no
             model on any tested axis". None of those is a wrong number; each is
             a wrong claim built on a right number, so no numeric check can see
             them.

  FORMATS    the markdown must be regenerable from the HTML and identical to the
             committed copy, so the three formats cannot silently diverge.

Run directly for a report, or via pytest as part of the suite:

    python check_brief_consistency.py
"""

import html as H
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.join(ROOT, "brief/viva-brief.html")
MD = os.path.join(ROOT, "brief/CR_Madukwe_Interview_Brief.md")
STATUS = os.path.join(ROOT, "PROJECT_STATUS.md")
UPDATE = os.path.join(ROOT, "UPDATE_BRIEF.md")
DECK = os.path.join(ROOT, "brief/CR_Madukwe_Interview_Deck.pptx")


def deck_prose():
    """Every text frame and speaker note in the interview deck, as one string.

    The deck is a fourth document making the same claims, and it was unguarded:
    an injected "which made the task genuine perfect prognosis" survived here
    while the same claim was caught in the brief and in PROJECT_STATUS. Speaker
    notes are included because they are what gets said out loud.

    Returns "" when the deck or python-pptx is absent, so the guard still runs
    on a machine that only has the markdown.
    """
    try:
        from pptx import Presentation
    except ImportError:
        return ""
    if not os.path.exists(DECK):
        return ""
    parts = []
    for slide in Presentation(DECK).slides:
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text)
        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text)
    # the deck sets typographic quotes and dashes that the ASCII retired
    # phrases would otherwise slip past
    t = " ".join(parts)
    for bad, good in (("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                      ("\u2013", "-"), ("\u2014", "-")):
        t = t.replace(bad, good)
    return re.sub(r"\s+", " ", t)
EVAL = os.path.join(ROOT, "data/processed/evaluation")
MINUS = "\u2212"


def prose(path=BRIEF):
    """Document text with tags stripped and entities resolved, whitespace collapsed."""
    if not os.path.exists(path):
        return ""
    s = open(path).read()
    if path.endswith('.html'):
        s = s.split('</style>', 1)[1]
    s = re.sub(r'<[^>]+>', ' ', s)
    return re.sub(r'\s+', ' ', H.unescape(s))


def tables(path=BRIEF):
    """[(caption, {first cell: [remaining cells]})] for every table in the brief.

    Cell-level rather than substring matching. A presence-anywhere check is
    useless here: "9.24" appears seventeen times in the brief, so corrupting the
    Table 3.3 cell left an earlier version of this guard passing. The value has
    to be checked where it lives.
    """
    body = open(path).read().split('</style>', 1)[1]
    out = []
    for t in re.findall(r'<table>(.*?)</table>', body, re.S):
        cap = re.search(r'<caption>(.*?)</caption>', t, re.S)
        cap = re.sub(r'<[^>]+>', '', H.unescape(cap.group(1))).strip() if cap else ''
        rows = {}
        for tr in re.findall(r'<tr[^>]*>.*?</tr>', t, re.S):
            cells = [re.sub(r'<[^>]+>', '', H.unescape(c)).strip()
                     for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', tr, re.S)]
            if cells:
                rows[cells[0]] = cells[1:]
        out.append((cap, rows))
    return out


def find_table(fragment):
    for cap, rows in tables():
        if fragment.lower() in cap.lower():
            return rows
    return None


def canonical_anchors():
    """(label, expected, found) triples checked CELL BY CELL against the CSVs."""
    checks = []

    def cell(rows, key, idx, label, expected):
        if rows is None or key not in rows or idx >= len(rows[key]):
            checks.append((label, expected, '<row or column missing>'))
        else:
            checks.append((label, expected, rows[key][idx]))

    t33 = os.path.join(EVAL, "table_3_3.csv")
    rows = find_table("Table 3.3")
    if os.path.exists(t33) and rows is not None:
        # the deployed row carries a suffix, so match on the model name prefix
        keymap = {k.split(' —')[0].strip(): k for k in rows}
        for _, r in pd.read_csv(t33).iterrows():
            k = keymap.get(r["model"])
            if k is None:
                checks.append(("Table 3.3 row %s" % r["model"], "present", "<row missing>"))
                continue
            cell(rows, k, 0, "Table 3.3 %s RMSE" % r["model"], "%.2f" % r["RMSE"])
            cell(rows, k, 3, "Table 3.3 %s Pearson R" % r["model"], "%.4f" % r["Pearson R"])

    tay = os.path.join(EVAL, "taylor_diagram_stats.csv")
    rows = find_table("Taylor statistics")
    if os.path.exists(tay) and rows is not None:
        for _, r in pd.read_csv(tay).iterrows():
            m = str(r["model"])
            if m not in rows:
                continue
            cell(rows, m, 0, "Taylor %s spatial R" % m, "%.4f" % r["spatial_correlation"])
            cell(rows, m, 2, "Taylor %s centred RMSE" % m, "%.3f" % r["centered_rmse"])

    bca = os.path.join(EVAL, "bca_intervals.csv")
    rows = find_table("Random Forest minus XGBoost")
    if os.path.exists(bca) and rows is not None:
        df = pd.read_csv(bca)
        key = df[(df.model_a == "Random Forest") & (df.model_b == "XGBoost")]
        for metric, row_label in [("centred RMSE", "Centred RMSE"),
                                  ("|MBE|", "Mean bias, |MBE|"),
                                  ("RMSE", "Aggregate RMSE")]:
            hit = key[key.metric == metric]
            if hit.empty:
                continue
            r = hit.iloc[0]
            sign = '+' if r["plug_in"] > 0 else MINUS
            cell(rows, row_label, 0, "BCa %s estimate" % metric,
                 "%s%.3f" % (sign, abs(r["plug_in"])))

    unc = os.path.join(EVAL, "uncertainty_decomposition_summary.csv")
    rows = find_table("Four-component uncertainty decomposition")
    if os.path.exists(unc) and rows is not None:
        df = pd.read_csv(unc)
        for period, label in [("near_term_2026_2050", "Near-term"),
                              ("mid_term_2051_2075", "Mid-term"),
                              ("long_term_2076_2100", "Long-term")]:
            hit = df[(df.model == "xgb") & (df.period == period)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            cell(rows, label, 5, "decomposition %s DS share" % label, "%.2f%%" % r["pct_var_ds"])
            cell(rows, label, 6, "decomposition %s arch share" % label, "%.2f%%" % r["pct_var_arch"])

    ro = os.path.join(EVAL, "rolling_origin.csv")
    rows = find_table("Rolling origin")
    if os.path.exists(ro) and rows is not None:
        df = pd.read_csv(ro)
        for model in ["Random Forest", "XGBoost"]:
            sub = df[df.model == model].sort_values("fold")
            key = next((k for k in rows if k.startswith(model)), None)
            if key is None:
                continue
            for i, (_, r) in enumerate(sub.iterrows()):
                cell(rows, key, i, "rolling %s fold %d" % (model, i + 1), "%.2f" % r["RMSE"])

    return checks


# Superseded sentences. Each was true of an earlier version of the brief and is
# now false; each survived at least one round of correction because the table it
# summarised was fixed and it was not.
RETIRED = [
    ("every severe bug was silent",
     "refuted by the table beneath it, which lists three that moved aggregate metrics"),
    ("Aggregate metrics did not move for any of these",
     "the same claim, relocated to the cancelled-bugs box"),
    ("beaten by no model on any tested axis",
     "the Random Forest beats XGBoost on centred RMSE and mean bias"),
    ("only model that no other model is established to beat",
     "same absolute claim, refuted by the BCa table"),
    ("Neither of its legs survives resampling",
     "both legs survive under BCa"),
    ("neither leg of the criterion is established",
     "both legs are established under BCa"),
    ("supports the Random Forest on one axis of two",
     "it survives on both legs, not one"),
    ("scorecard with one loss",
     "two established losses: centred RMSE and mean bias"),
    ("grant the Random Forest one real point",
     "both of its points"),
    ("rested on that leg",
     "neither leg, since both survive"),
    ("partly survived",
     "survived on both legs"),
    ("Two independent findings removed its basis",
     "the criterion survived; it was outweighed"),
    ("both intervals span zero",
     "the paired MBE difference is established; marginal intervals are the wrong test"),
    ("a factor of four to twelve",
     "five to twelve; 8.31/1.65 = 5.04"),
    ("2.45 by cell count",
     "2.45 per axis, 6.01 by cell count"),
    ("was found internally",
     "most were found through external audit"),
    ("Two things are genuinely worse about XGBoost",
     "three, once centred RMSE and mean bias are both established"),
    ("loses on every axis the study can actually test",
     "the CNN is better on std ratio; say every axis the study can establish"),
    ("Five external audit rounds", "six"),
    # the CSI intermediate is not a clear-sky index: its denominator averages
    # 13 daytime hours while the ssrd numerator is a 24-hour mean (7.15)
    # verb-agnostic: PROJECT_STATUS said "making", the brief said "makes"
    ("the task a genuine perfect-prognosis problem",
     "training is ERA5-to-ERA5 either way, so both configurations qualify; "
     "what dropping rsds changes is whether the ML step does anything"),
    ("which made the task genuine perfect prognosis",
     "same overstatement in the deck's wording"),
    ("fixes the physical meaning of the intermediate CSI",
     "it does not; the denominator is on a different temporal basis, so the "
     "ratio runs 1.724 below a true clear-sky index"),
    # live in PROJECT_STATUS after the brief had been corrected - the paired
    # RF-XGBoost MBE difference IS established at -0.953 [-1.499, -0.475]
    ("every model's MBE interval spans zero",
     "true of the MARGINAL intervals; the paired difference is established"),
    ("both intervals span zero, and +1.20 is an eighth",
     "the paired MBE difference is established under BCa"),
    ("individually indistinguishable from zero",
     "marginal intervals; the paired difference is established"),
    # live in PROJECT_STATUS after the brief had been corrected, round seven
    ("both spanning zero (§7.10)",
     "centred RMSE IS established under BCa at -0.171 [-0.360, -0.041]"),
    ("not as an established difference from XGBoost",
     "centred RMSE is established; only spatial correlation is not"),
    ("One leg of the composite criterion survives",
     "both legs survive; mean bias is the larger"),
    ("Four audit rounds found the same failure",
     "six, and the caveat was duplicated"),
    ("from 0.25° to 0.1°",
     "0.25 deg is ERA5's grid, not CMIP6's; the GCMs run natively at 0.94-1.88 deg"),
    ("Models INJECT small-scale power",
     "the GHI-space reading Section 7.8 overturned; judge in CSI space"),
    ("downscale CMIP6 to ERA5 resolution",
     "the product is finer than ERA5: 0.1 deg against 0.25 deg"),
]

EXPLANATORY = (r"earlier version|previously|an earlier|was wrong|were wrong|no longer|"
               r"overturned|retired|contradicted|until 19 August|"
               r"superseded|corrected|said first|twice over|used to|instead of|rather than|"
               r"the old |this brief said|versions of this brief")
WINDOW = 260


# Numbers that are DERIVABLE must not be written into prose. The test count was
# maintained by hand in three documents and reached three different values - 16,
# 18 and 19 - while the suite actually held 20. No canonical-value check can catch
# that, because the count lives in the test suite rather than in any CSV. The fix
# is to forbid the claim rather than to track it.
DERIVABLE = [
    # the lookbehind excludes section references: "Table 3.3 tests whether" and
    # "§7.11 tests the scenario response" both contain "N tests" and neither is a count
    (re.compile(r"(?<![.\d])\b\d+\s+tests?\b(?!\s*/)", re.I),
     "the test count changes whenever a test is added, and quoting it produced three "
     "different numbers across three documents. Refer to `pytest tests/` or 'the suite'."),
]


def check_derivable(label, text):
    out = []
    for pat, why in DERIVABLE:
        for m in pat.finditer(text):
            lo = max(0, m.start() - 90)
            if re.search(r"different numbers|no count is quoted|changes whenever", text[lo:m.end() + 120], re.I):
                continue  # the document is explaining why it does not quote one
            out.append(("%s: %r" % (label, m.group(0).strip()), why))
    return out


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

def check():
    text = prose()
    missing, resurrected, fmt = [], [], []

    for label, expected, found in canonical_anchors():
        if expected != found:
            missing.append((label, expected, found))

    # The sentence list applies to BOTH documents. Restricting it to the brief is
    # how "both intervals span zero" and "individually indistinguishable from
    # zero" survived in PROJECT_STATUS after being corrected in the brief - the
    # authoritative record contradicting itself while the guard reported clean.
    docs = (("brief", text), ("PROJECT_STATUS", prose(STATUS)),
            ("UPDATE_BRIEF", prose(UPDATE)), ("deck", deck_prose()))
    for label, doc in docs:
        resurrected += check_derivable(label, doc)

    for label, doc in docs:
        for phrase, why in RETIRED:
            for m in re.finditer(re.escape(phrase), doc):
                if _explained(doc, m.start(), m.end()):
                    continue  # the document is discussing its own correction
                resurrected.append(("%s: %s" % (label, phrase), why))

    # the markdown must be exactly what the builder produces from the HTML
    if os.path.exists(MD):
        sys.path.insert(0, os.path.join(ROOT, "brief"))
        try:
            import build_brief_formats as B
            body = open(BRIEF).read().split('</style>', 1)[1]
            B.check_no_nested_divs(body)
            if B.to_markdown(body) != open(MD).read():
                fmt.append("CR_Madukwe_Interview_Brief.md is not what "
                           "build_brief_formats.py produces from viva-brief.html "
                           "— rebuild it")
        except SystemExit as e:
            fmt.append(str(e))
    return missing, resurrected, fmt


def main():
    missing, resurrected, fmt = check()
    print("=" * 84)
    print(" Interview brief + PROJECT_STATUS consistency check")
    print("=" * 84)

    if missing:
        print("\nCELLS THAT DISAGREE WITH THE CANONICAL CSVs (%d):" % len(missing))
        for label, expected, found in missing:
            print("  %-40s expected '%s', brief has '%s'" % (label, expected, found))
    else:
        print("\nEvery checked table cell matches the canonical evaluation CSVs.")

    if resurrected:
        print("\nSUPERSEDED SENTENCES PRESENT (%d):" % len(resurrected))
        for phrase, why in resurrected:
            print("  '%s'\n      %s" % (phrase, why))
    else:
        print("No superseded sentence appears outside an explanatory context.")

    if fmt:
        print("\nFORMAT DRIFT (%d):" % len(fmt))
        for f in fmt:
            print("  " + f)
    else:
        print("Markdown matches what the builder produces from the HTML.")

    print()
    return 1 if (missing or resurrected or fmt) else 0


if __name__ == "__main__":
    sys.exit(main())
