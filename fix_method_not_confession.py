"""State the hyperparameter work in Section 3.6.7 as method rather than confession.

The sixth critique's second item. Four passages had crept back into the narrative
voice of a revision log. The leakage corrections stay, because an examiner needs
to know the checkpoints were once selected on the withheld record and no longer
are: that is a statement of procedure. What goes is the confessional framing
around the pixel-wise configurations, where the finding that matters — the
cross-validated settings beat the a priori defaults on validation — reads better
as a result than as an admission.

The numbers are unchanged and still read from the CSVs; only the voice changes.

    python fix_method_not_confession.py
"""

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

cv = pd.read_csv(os.path.join(EVAL, "pixelwise_cv_config_check.csv"))


def arm(model, label):
    return float(cv[(cv.model == model)
                    & (cv.configuration.str.startswith(label))].RMSE_zw.iloc[0])


xa, xc = arm("XGBoost", "a priori"), arm("XGBoost", "cross-validated")
ra, rc = arm("Random Forest", "a priori"), arm("Random Forest", "cross-validated")

EDITS = [
    ("That was not true of an earlier version, and the end of this section records what "
     "changed and why, because the correction bears on how the comparisons in Chapter 4 "
     "should be read.",
     "The exception is the U-Net, and the end of this section gives it."),

    ("One correction belongs here. The pixel-wise defaults were originally retained in "
     "preference to the cross-validated configurations on the ground that the latter "
     "scored worse on the full 5,751-cell validation set. Choosing between configurations "
     "by their score on the evaluation period is selection on the withheld record, of "
     "exactly the kind corrected for the neural models' checkpoints in Section 3.6.6 and "
     "for their hyperparameters earlier in this section, and it should not have stood. "
     "The comparison had also never been reported. It was therefore run: both "
     "configurations of both models were fitted on the training period and scored on the "
     "validation period over Zimbabwe. The cross-validated configuration proved the "
     "better of the two in both cases, by %.3f W/m² for XGBoost (%.3f against %.3f) "
     "and by %.3f for the Random Forest (%.3f against %.3f), so the stated reason for "
     "retaining the defaults was not merely improper but wrong."
     % (xa - xc, xc, xa, ra - rc, rc, ra),

     "One further point of procedure. No configuration is chosen by its score on the "
     "withheld record, for either family: the pixel-wise models take the configuration "
     "their cross-validation selects inside the training period, and the neural models "
     "take theirs from the inner split. The a priori configurations of Sections 3.6.3 and "
     "3.6.4 were therefore not retained, and nothing was lost by not retaining them. "
     "Fitted on the training period and scored on the validation period over Zimbabwe, "
     "the cross-validated configuration is the better of the two in both cases, by %.3f "
     "W/m² for XGBoost (%.3f against %.3f) and by %.3f for the Random Forest (%.3f "
     "against %.3f)."
     % (xa - xc, xc, xa, ra - rc, rc, ra)),
]


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
              d.element.xml.count('w:fldCharType="begin"'))
    done = left = 0
    for old, new in EDITS:
        par = next((p for p in d.paragraphs if old in p.text), None)
        if par is None:
            print("  already applied or absent: %s" % old[:50])
            continue
        if apply(par, old, new):
            done += 1
            print("  fixed: %s" % old[:50])
        else:
            left += 1
            print("  SPANS A FIELD, skipped: %s" % old[:50])
    if not done:
        return 0 if left == 0 else 1
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_voice.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_voice.docx"), DOC)
        print("CITATION FIELDS CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d fixed, %d outstanding; citation fields intact %s" % (done, left, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
