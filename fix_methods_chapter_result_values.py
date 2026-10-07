"""Take result values out of Chapter 3, and say why the interval form matters.

Chapter 3 carried thirteen result figures copied from Chapter 4. Chapter 4 is
regenerated from the CSVs on every chain run; Chapter 3 is not, so each
regeneration left a stale copy behind in the methods chapter. By v7 that had
produced a direct contradiction: Section 3.8.1 told the reader the U-Net returns
the lower aggregate error, which stopped being true when the U-Net was refitted,
and Section 3.8.4 quoted bootstrap intervals from two reruns ago.

The examiner found four of the thirteen. The other nine were in the same
paragraph and were found by checking every figure in Sections 3.8.1 and 3.8.4
against bca_intervals.csv:

    quoted in 3.8.4            current value
    1.345 W/m2 |MBE|           1.247
    [-1.926, -0.606]           [-1.919, -0.597]
    0.126 centred RMSE         0.139
    [-0.273, +0.056] pct       [-0.269, +0.023]
    [-0.304, +0.017] BCa       [-0.276, +0.010]
    +0.0011 spatial R          +0.0012
    [-0.0005, +0.0025]         [-0.0002, +0.0023]
    +1.218 aggregate           +0.777
    [+0.686, +2.139]           [+0.326, +1.420]

Rather than refresh them, which only resets the clock, the figures are removed
and the chapter points to Sections 4.3 and 4.4 and to Appendix C, Table C.6. A
methods chapter states what was done and why; the values belong where they are
generated. What stays is every methodological statement, including which
interval form was chosen and the reason.

One paragraph is added, because Appendix C now prints the percentile, basic and
BCa intervals side by side and nine of the thirty-six comparisons are
form-dependent. A reader can see that, so the chapter has to address it.

    python fix_methods_chapter_result_values.py
"""

import copy
import os
import shutil
import sys

import docx
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

bca = pd.read_csv(os.path.join(EVAL, "bca_intervals.csv"))
N_BLOCKS = 14  # 2011-2024 inclusive, the withheld record resampled by year


def forms(a, b, metric):
    r = bca[(bca.model_a == a) & (bca.model_b == b) & (bca.metric == metric)]
    if r.empty:
        r = bca[(bca.model_a == b) & (bca.model_b == a) & (bca.metric == metric)]
    r = r.iloc[0]
    return [t for t, c in (("percentile", "pct_spans_zero"),
                           ("basic", "basic_spans_zero"),
                           ("BCa", "bca_spans_zero")) if not bool(r[c])]


_split = {3: [], 2: [], 1: [], 0: []}
for _, r in bca.iterrows():
    _split[len(forms(r["model_a"], r["model_b"], r["metric"]))].append(
        (r["model_a"], r["model_b"], r["metric"]))
N_ALL = len(_split[3])
N_PART = len(_split[2]) + len(_split[1])
N_TOTAL = len(bca)

# The three axes the deployment turns on, and whether each is form-independent.
_DEPLOY = [("centred RMSE", "centred error"), ("spatial R", "spatial correlation"),
           ("std ratio dev", "the standard-deviation ratio")]
_robust = [lab for m, lab in _DEPLOY
           if len(forms("XGBoost", "U-Net", m)) == 3]
assert len(_robust) == 3, (
    "the deployment axes are no longer unanimous across interval forms: %s"
    % [(m, forms("XGBoost", "U-Net", m)) for m, _ in _DEPLOY])

EDITS = [
    # --- 3.8.1: the reason XGBoost falls to the third step ----------------
    ("XGBoost is deployed under the third step rather than the second: Section "
     "4.2 reports that the retrained U-Net returns the lower aggregate error, "
     "8.62 W/m² against 8.92, and Section 4.4 that the margin does not survive "
     "resampling, while XGBoost's advantage on centred error and spatial "
     "correlation does.",
     "XGBoost is deployed under the third step rather than the second. It "
     "returns the lower aggregate error of the two, but the second step requires "
     "that margin to be established and Section 4.4 reports that it is not, so "
     "the second step does not settle the choice; the third step does, because "
     "XGBoost's advantage on centred error, spatial correlation and the "
     "standard-deviation ratio is established. The figures are in Section 4.4 "
     "and Appendix C, Table C.6, and are deliberately not repeated here."),

    # --- 3.8.4: the configuration that prompted the third step ------------
    ("was formalised after the retrained U-Net of Section 4.5 returned the "
     "lower RMSE.",
     "was formalised after an intermediate configuration of the U-Net, since "
     "superseded, returned the lower RMSE. That configuration is not the one "
     "Table 4.1 reports: refitting the U-Net at the learning rate its own "
     "search selects, as Section 3.6.7 describes, moved it above XGBoost, so in "
     "the study as it stands XGBoost is the lower of the two on the point "
     "estimate as well. The rule is kept in its three-step form because the "
     "margin is still not established, which is the condition the third step "
     "exists to handle."),

    # --- 3.8.4: the a priori round's figures ------------------------------
    ("and XGBoost returned the lower aggregate error over Zimbabwe, at 8.92 "
     "W/m² against 10.13.",
     "and XGBoost returned the lower aggregate error over Zimbabwe. Both models "
     "then carried the a priori configurations of Sections 3.6.3 and 3.6.4, "
     "since replaced by the cross-validated ones, so the figures of that round "
     "are not those of Table 4.1 and are not quoted here."),

    # --- 3.8.4: the defence paragraph -------------------------------------
    ("The Random Forest's composite case rested on a structural advantage that "
     "resampling does not establish: its centred-error difference from XGBoost "
     "is -0.126 W/m² with a bias-corrected interval of -0.304 to +0.017 and its "
     "spatial-correlation difference +0.0011 at -0.0005 to +0.0025, both "
     "spanning zero. XGBoost's structural advantage over the U-Net does "
     "survive, at -1.531 (-1.659 to -1.451) and +0.0336 (+0.0248 to +0.0451).",
     "The Random Forest's composite case rested on a structural advantage that "
     "resampling does not establish: neither its centred-error difference from "
     "XGBoost nor its spatial-correlation difference excludes zero under the "
     "bias-corrected interval. XGBoost's structural advantage over the U-Net "
     "does survive on both, and on the standard-deviation ratio as well. "
     "Sections 4.3 and 4.4 report every difference with its interval and "
     "Appendix C, Table C.6, prints all three interval forms; they are not "
     "repeated here, because a methods chapter that carries result values falls "
     "out of step with the results each time they are regenerated, which is "
     "how the figures this paragraph used to quote came to describe a "
     "superseded configuration."),

    # --- 3.8.4: the first leg --------------------------------------------
    ("Measured that way the Random Forest is established better, by 1.345 W/m² "
     "with a bias-corrected interval of [-1.926, -0.606].",
     "Measured that way the Random Forest is the better of the two, as Section "
     "4.3 reports."),

    # --- 3.8.4: the interval comparison, split around the Efron field -----
    ("A percentile interval places its centred-error advantage at 0.126 W/m² "
     "with a 95 per cent interval of [-0.273, +0.056], which contains zero. A "
     "bias-corrected and accelerated interval ",
     "The percentile interval for its centred-error advantage contains zero, "
     "and so does a bias-corrected and accelerated interval "),
    (", which corrects the downward bias that arises because each bootstrap "
     "replicate recomputes the time-mean field from resampled years and so "
     "acquires sampling noise in quadrature, gives 0.126 W/m² at [-0.304, "
     "+0.017]. That also contains zero, and the Random Forest's centred-error "
     "advantage is therefore not established.",
     ", which corrects the downward bias that arises because each bootstrap "
     "replicate recomputes the time-mean field from resampled years and so "
     "acquires sampling noise in quadrature. The Random Forest's centred-error "
     "advantage is therefore not established."),

    ("Nor is its spatial-correlation advantage, at +0.0011 with a bias-corrected "
     "interval of [-0.0005, +0.0025].",
     "Nor is its spatial-correlation advantage."),

    ("the systematic offset, where the Random Forest is established better by "
     "1.345 W/m²,",
     "the systematic offset, where the Random Forest is the better of the two,"),

    ("The aggregate deficit the criterion was introduced to outweigh is itself "
     "established, at +1.218 W/m² with a bias-corrected interval of [+0.686, "
     "+2.139],",
     "The aggregate deficit the criterion was introduced to outweigh is itself "
     "established, as Section 4.3 reports,"),

    # --- 3.8.4, the two-axis paragraph: nine more, found by checking every
    # figure in the section rather than only the ones the critique listed.
    ("so the identity is exact and the tables report it twice under two names, "
     "at 0.405 W/m² for the Random Forest, 0.531 for XGBoost, 2.682 for the CNN "
     "and 2.062 for the U-Net.",
     "so the identity is exact and the tables report it twice under two names. "
     "Table 4.2 gives the values for all four architectures."),

    ("the Random Forest is separated from one network and not the other: +0.577 "
     "W/m² against the CNN with a bias-corrected interval of -0.201 to +1.566, "
     "which contains zero, and +1.510 against the U-Net at +0.892 to +2.729, "
     "which does not, so the U-Net is established the better of that pair.",
     "the Random Forest is separated from one network and not the other, as "
     "Section 4.3 reports: its difference from the CNN spans zero while its "
     "difference from the U-Net does not, so the U-Net is the better of that "
     "pair on the bias-corrected interval."),

    ("Its mean bias is +1.43 W/m² against the Random Forest's +0.08, and the "
     "Random Forest is established preferable on this axis: the paired "
     "difference in mean bias is -1.345 W/m² with a bias-corrected interval of "
     "[-1.926, -0.606].",
     "Its mean bias is the larger of the two, and on this axis the Random "
     "Forest is preferable: Table 4.1 gives both biases and Section 4.3 the "
     "paired difference with its interval."),

    ("For scale, +1.43 is roughly a seventh of a downscaling uncertainty of "
     "order 10 W/m².",
     "For scale, that bias is of order a tenth of the downscaling uncertainty "
     "reported in Section 4.7."),
]

NEW_PARA_AFTER = "No validation metric in Table 3.3 is capable of detecting this"
NEW_PARA = (
    "One qualification applies to every interval quoted in Chapter 4, and it is "
    "stated here because Appendix C, Table C.6, now prints the percentile, basic "
    "and bias-corrected intervals side by side and a reader can check it. The "
    "resampling unit is the calendar year and the withheld record is %d years "
    "long, so every interval rests on %d blocks. The BCa correction estimates "
    "its acceleration constant by jackknife over those %d blocks, and its "
    "coverage guarantee is asymptotic; at this size the correction is itself "
    "estimated noisily, and the three forms need not agree near the margin. They "
    "do not: of the %d comparisons in Table C.6, %d exclude zero under all three "
    "forms and %d under one or two, so for that second group the verdict depends "
    "on the choice of interval and is reported in Table C.6 as holding under the "
    "named forms rather than as established outright. The mean-bias comparisons "
    "are the most affected, because the absolute value of a quantity whose sign "
    "is near zero is not smooth and folds the bootstrap distribution, which is "
    "the condition the BCa transformation assumes away; where the sign is not at "
    "issue the signed difference is the sounder statistic and is what Section 4.4 "
    "uses. None of this reaches the deployment. The three axes the third step "
    "turns on — %s — exclude zero under all three interval forms, so the "
    "selection of XGBoost does not depend on which interval is read."
    % (N_BLOCKS, N_BLOCKS, N_BLOCKS, N_TOTAL, N_ALL, N_PART,
       ", ".join(_robust)))


def _key(old, new):
    """A substring of `new` that does not occur in `old`, for idempotence.

    Several replacements here keep the opening clause and change only what
    follows, so a head-of-string key matches the text being replaced and the
    edit is skipped as already done. The longest differing tail is used instead.
    """
    i = 0
    while i < min(len(old), len(new)) and old[i] == new[i]:
        i += 1
    tail = new[i:i + 80].strip()
    return tail if tail and tail not in old else None


def is_field(r):
    x = r._r.xml
    return ("fldChar" in x) or ("instrText" in x)


def apply(par, old, new):
    for r in par.runs:
        if old in r.text:
            r.text = r.text.replace(old, new, 1)
            return True
    runs = par.runs
    full = "".join(r.text for r in runs)
    at = full.find(old)
    if at < 0:
        return False
    starts, pos = [], 0
    for r in runs:
        starts.append(pos)
        pos += len(r.text)
    f = max(i for i, st in enumerate(starts) if st <= at)
    l = max(i for i, st in enumerate(starts) if st < at + len(old))
    span = runs[f:l + 1]
    if any(is_field(r) for r in span):
        return False
    acc = "".join(r.text for r in span)
    k = acc.find(old)
    span[0].text = acc[:k] + new + acc[k + len(old):]
    for r in span[1:]:
        r.text = ""
    return True


def main():
    if os.path.exists(LOCK):
        print("The dissertation is open in Word. Close it first.")
        return 2
    d = docx.Document(DOC)
    before = (d.element.xml.count("ZOTERO_ITEM"),
              d.element.xml.count('w:fldCharType="begin"'),
              len(d.sections))
    done = left = 0
    for old, new in EDITS:
        # The idempotence key must be text the replacement introduces and the
        # original does not. Keying on new[:70] reported three of these edits as
        # "already applied" when they had not run at all, because each
        # replacement opens with the same clause it replaces.
        key = _key(old, new)
        if key and any(key in p.text for p in d.paragraphs):
            print("  already applied: %s" % old[:52])
            continue
        par = next((p for p in d.paragraphs if old in p.text), None)
        if par is None:
            # No usable key (the replacement is a strict truncation of the
            # original, so every differing character is punctuation): the edit
            # is done if the replacement text is present.
            if any(new in p.text for p in d.paragraphs):
                print("  already applied: %s" % old[:52])
            else:
                left += 1
                print("  NOT FOUND: %s" % old[:52])
            continue
        if apply(par, old, new):
            done += 1
            print("  stripped: %s" % old[:52])
        else:
            left += 1
            print("  SPANS A FIELD, skipped: %s" % old[:52])

    if not any(NEW_PARA[:60] in p.text for p in d.paragraphs):
        host = next((p for p in d.paragraphs if NEW_PARA_AFTER in p.text), None)
        if host is None:
            left += 1
            print("  NOT FOUND: the host paragraph for the interval note")
        else:
            # Clone the host so the new paragraph inherits its style exactly,
            # then strip the copy to a single run carrying the new text.
            new_p = copy.deepcopy(host._p)
            for child in list(new_p):
                if child.tag in (W + "r", W + "hyperlink", W + "bookmarkStart",
                                 W + "bookmarkEnd", W + "fldSimple"):
                    new_p.remove(child)
            host._p.addnext(new_p)
            from docx.text.paragraph import Paragraph
            Paragraph(new_p, host._parent).add_run(NEW_PARA)
            done += 1
            print("  added the interval-form paragraph (%d of %d comparisons "
                  "unanimous, %d partial)" % (N_ALL, N_TOTAL, N_PART))

    if not done:
        print("nothing to change")
        return 0 if left == 0 else 1
    bak = DOC.replace(".docx", "_BACKUP_pre_methods_values.docx")
    shutil.copy(DOC, bak)
    d.save(DOC)
    dd = docx.Document(DOC)
    ax = dd.element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'),
             len(dd.sections))
    if after != before:
        shutil.copy(bak, DOC)
        print("STRUCTURE CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d changed, %d outstanding; citations/fields/sections intact %s"
          % (done, left, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
