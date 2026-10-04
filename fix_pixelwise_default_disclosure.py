"""Correct Section 3.6.7's account of why the pixel-wise defaults were retained.

The fifth critique's severest finding, and the measurement makes it worse than
the critique states. Section 3.6.7 said:

    "For both pixel-wise models the configuration selected by cross-validation
    performed marginally worse on the full 5,751-cell validation set than the
    default configuration specified in Sections 3.6.3 and 3.6.4 ... The default
    configurations were therefore retained."

Two things are wrong with that. Choosing between configurations by their score on
the withheld record is selection on the evaluation data, which is the defect this
thesis corrects at length for the neural models. And the comparison it rests on
was never reported, so nobody could check it. It has now been run, by
compute_pixelwise_cv_config_check.py, fitting both configurations of both models
on the training period and scoring them on the validation period on the Zimbabwe
basis. The cross-validated configuration is BETTER for both models, marginally
for XGBoost and materially for the Random Forest. The stated reason for retaining
the defaults therefore does not hold.

The passage is rewritten to say what was measured, to concede the
selection-on-test problem rather than restate it as a finding, and to record that
the defaults are kept in this submission only because redeploying cascades
through the projection, uncertainty and suitability chains. The opening sentence,
which claimed all four models were optimised by grid search, is corrected too,
and Section 4.4 picks up the caveat where the XGBoost-U-Net decision is made.

    python fix_pixelwise_default_disclosure.py
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


def g(model, cfg):
    return float(cv[(cv.model == model) & (cv.configuration == cfg)].RMSE_zw.iloc[0])


xd, xc = g("XGBoost", "deployed"), g("XGBoost", "cross-validated")
rd, rc = g("Random Forest", "deployed"), g("Random Forest", "cross-validated")
hx = pd.read_csv(os.path.join(EVAL, "hpo_pixelwise_xgb.csv"))
dep_rank = int(hx[hx.is_deployed].iloc[0]["rank"])
n_grid = len(hx)

EDITS = [
    # Revision history that reads better as a statement of procedure.
    ("An earlier version of this search was scored directly on the validation mean "
     "squared error, which is selection on the withheld record, of the same kind "
     "corrected for the epoch count in Section 3.6.6. It has been re-run over the same "
     "grids entirely inside the training period:",
     "The search was first scored directly on the validation mean squared error, which "
     "is selection on the withheld record of the same kind corrected for the epoch count "
     "in Section 3.6.6, and has been re-run over the same grids entirely inside the "
     "training period:"),

    ("Hyperparameters for all four models were optimised by grid search over the "
     "primary hyperparameters listed in Table 3.2.",
     "Hyperparameters were searched by grid search over the primary hyperparameters "
     "listed in Table 3.2, and the search governed the deployed configuration for the "
     "two neural models only. For the two pixel-wise models the configurations in "
     "Sections 3.6.3 and 3.6.4 were retained instead, which the end of this section "
     "returns to because the reason originally given for retaining them does not hold."),

    ("One methodological caveat follows from the subsampled search. For both pixel-wise "
     "models the configuration selected by cross-validation performed marginally worse "
     "on the full 5,751-cell validation set than the default configuration specified in "
     "Sections 3.6.3 and 3.6.4, indicating that a 150-cell search sample does not "
     "transfer perfectly to the full domain. The default configurations were therefore "
     "retained for the Random Forest and XGBoost, while the two neural architectures "
     "adopted their search-selected values.",

     "One caveat follows, and it is a defect rather than a caveat. The pixel-wise "
     "defaults were retained in preference to the cross-validated configurations on the "
     "ground that the latter scored worse on the full 5,751-cell validation set. That is "
     "selection on the withheld record, of exactly the kind corrected for the neural "
     "models' checkpoints in Section 3.6.6 and for their hyperparameters earlier in this "
     "section. It is also, as it turns out, not true. The comparison was never reported, "
     "and it has now been run: both configurations of both models were fitted on the "
     "training period and scored on the validation period over Zimbabwe. The "
     "cross-validated configuration is the better of the two in both cases, by %.3f "
     "W/m² for XGBoost (%.3f against %.3f) and by %.3f for the Random Forest (%.3f "
     "against %.3f). The defaults are retained in this submission because the deployed "
     "XGBoost drives the projection, uncertainty and suitability chains of Chapter 4 and "
     "refitting it would require all three to be regenerated and rechecked, not because "
     "the comparison favours them. Two consequences follow and are stated plainly. The "
     "deployed configuration is not the one a clean procedure would have produced; and "
     "the XGBoost figure the comparison with the U-Net rests on in Section 4.4 carries "
     "this contamination, which matters because that comparison turns on a margin of "
     "about 0.3 W/m². The Random Forest's score carries it too. For completeness, "
     "the deployed XGBoost configuration ranks %d of %d on the search's own "
     "cross-validation criterion."
     % (xd - xc, xc, xd, rd - rc, rc, rd, dep_rank, n_grid)),
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
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_defaults.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_defaults.docx"), DOC)
        print("CITATION FIELDS CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d fixed, %d outstanding; citation fields intact %s" % (done, left, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
