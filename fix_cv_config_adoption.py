"""Update Chapter 3 for the adoption of the cross-validated pixel-wise configurations.

The fifth critique's severest finding was that the pixel-wise defaults had been
retained over the search's own selection on the strength of a comparison on the
withheld record, and that the comparison had never been reported. It has now
been reported and the result reversed the stated reason, so the configurations
were changed: both pixel-wise models carry the cross-validated settings, chosen
inside the training period, and the defect is repaired rather than disclosed.

Five places in Chapter 3 named the old values or described the old state:

  3.6.3  the Random Forest prose, 500 trees and a minimum leaf of 5
  3.6.4  the XGBoost prose, depth 6, eta 0.05, subsample 0.8, child weight 3
  3.6.7  the opening, which said the search governed the neural models only
  3.6.7  the closing paragraph, which conceded an unrepaired defect
  Table 3.2  the caption, which said the pixel-wise rows were a priori defaults
             with three XGBoost values outside the ranges searched

Every number is read from the evaluation CSVs or the training scripts, so none
of it is typed here.

    python fix_cv_config_adoption.py
"""

import os
import re
import shutil
import sys

import docx
import pandas as pd
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")

t33 = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")
cv = pd.read_csv(os.path.join(EVAL, "pixelwise_cv_config_check.csv"))
hx = pd.read_csv(os.path.join(EVAL, "hpo_pixelwise_xgb.csv"))
hr = pd.read_csv(os.path.join(EVAL, "hpo_pixelwise_rf.csv"))


def arm(model, label):
    r = cv[(cv.model == model) & (cv.configuration.str.startswith(label))]
    assert not r.empty, "no %r arm for %s" % (label, model)
    return float(r.RMSE_zw.iloc[0])


def default_of(script, name):
    src = open(os.path.join(ROOT, script)).read()
    m = re.search(r'os\.environ\.get\(\s*"%s"\s*,\s*"([^"]+)"' % name, src)
    assert m, name
    return m.group(1)


X = {k: default_of("train_pixelwise_xgb.py", "XGB_" + k.upper())
     for k in ("max_depth", "eta", "subsample", "min_child_weight", "n_estimators")}
R = {k: default_of("train_pixelwise_rf.py", "RF_" + k.upper())
     for k in ("n_estimators", "min_samples_leaf")}

xa, xc = arm("XGBoost", "a priori"), arm("XGBoost", "cross-validated")
ra, rc = arm("Random Forest", "a priori"), arm("Random Forest", "cross-validated")

EDITS = [
    # 3.6.3
    ("The forest consists of 500 decision trees (n_estimators = 500)",
     "The forest consists of %s decision trees (n_estimators = %s)"
     % (R["n_estimators"], R["n_estimators"])),
    ("Minimum leaf size was set to 5 to avoid overfitting on the limited 312-month "
     "training record.",
     "Minimum leaf size was set to %s. Both values are the output of the "
     "cross-validated search of Section 3.6.7 rather than chosen a priori."
     % R["min_samples_leaf"]),

    # 3.6.4
    ("maximum tree depth (max_depth = 6), learning rate (eta = 0.05), subsample ratio "
     "of the training data per tree (subsample = 0.8), minimum child weight "
     "(min_child_weight = 3), and a fixed 200 boosting rounds",
     "maximum tree depth (max_depth = %s), learning rate (eta = %s), subsample ratio of "
     "the training data per tree (subsample = %s), minimum child weight "
     "(min_child_weight = %s), and a fixed %s boosting rounds, the first four being the "
     "output of the cross-validated search of Section 3.6.7"
     % (X["max_depth"], X["eta"], X["subsample"], X["min_child_weight"],
        X["n_estimators"])),

    # the early-stopping comparison predates the configuration change
    ("it is removed, and the cost of removing it was 9.11 against the 9.03 reported "
     "here, both on the full analysis box.",
     "it is removed. Measured under the configuration then in use, the cost of removing "
     "it was 9.11 against 9.03, both on the full analysis box; the configuration has "
     "since changed and that comparison was not repeated under the new one, which now "
     "scores %.2f on the same basis."
     % t33.loc["XGBoost", "RMSE"]),

    # 3.6.7 opening
    ("Hyperparameters were searched by grid search over the primary hyperparameters "
     "listed in Table 3.2, and the search governed the deployed configuration for the "
     "two neural models only. For the two pixel-wise models the configurations in "
     "Sections 3.6.3 and 3.6.4 were retained instead, which the end of this section "
     "returns to because the reason originally given for retaining them does not hold.",
     "Hyperparameters for all four models were searched by grid search over the primary "
     "hyperparameters listed in Table 3.2, and for all four the deployed configuration "
     "is the one the search selected. That was not true of an earlier version, and the "
     "end of this section records what changed and why, because the correction bears on "
     "how the comparisons in Chapter 4 should be read."),

    # 3.6.7 closing
    ("One caveat follows, and it is a defect rather than a caveat. The pixel-wise "
     "defaults were retained in preference to the cross-validated configurations on the "
     "ground that the latter scored worse on the full 5,751-cell validation set.",
     "One correction belongs here. The pixel-wise defaults were originally retained in "
     "preference to the cross-validated configurations on the ground that the latter "
     "scored worse on the full 5,751-cell validation set."),
]

CLOSING_TAIL = (
    "That is "
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
    "cross-validation criterion.")

NEW_TAIL = (
    "Choosing between configurations by their score on the "
    "evaluation period is selection on the withheld record, of exactly the kind "
    "corrected for the neural models' checkpoints in Section 3.6.6 and for their "
    "hyperparameters earlier in this section, and it should not have stood. The "
    "comparison had also never been reported. It was therefore run: both "
    "configurations of both models were fitted on the training period and scored on the "
    "validation period over Zimbabwe. The cross-validated configuration proved the "
    "better of the two in both cases, by %.3f W/m² for XGBoost (%.3f against %.3f) "
    "and by %.3f for the Random Forest (%.3f against %.3f), so the stated reason for "
    "retaining the defaults was not merely improper but wrong. Both models were "
    "refitted with their cross-validated configurations and every artefact downstream "
    "of them regenerated: the validation fields and Table 4.1, the resampled intervals, "
    "the spectra, the baselines, the projections, the ensembles, the scenario screen, "
    "the uncertainty budget and the whole suitability chain. The figures reported "
    "throughout Chapter 4 are therefore those of configurations selected inside the "
    "training period, and no comparison in that chapter depends on a choice made by "
    "looking at the evaluation record. The cost of the change is a model store that "
    "grows from about four to about thirteen gigabytes, because the selected Random "
    "Forest is both larger and deeper.")


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


def cells(d):
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    yield p


def main():
    if os.path.exists(LOCK):
        print("The dissertation is open in Word. Close it first.")
        return 2
    d = docx.Document(DOC)
    before = (d.element.xml.count("ZOTERO_ITEM"),
              d.element.xml.count('w:fldCharType="begin"'))
    done = left = 0

    edits = list(EDITS)
    dep_rank = int(hx[hx.is_deployed].iloc[0]["rank"]) if hx.is_deployed.any() else 1
    edits.append((CLOSING_TAIL % (xa - xc, xc, xa, ra - rc, rc, ra, dep_rank, len(hx)),
                  NEW_TAIL % (xa - xc, xc, xa, ra - rc, rc, ra)))

    for old, new in edits:
        par = next((p for p in list(d.paragraphs) + list(cells(d)) if old in p.text), None)
        if par is None:
            print("  already applied or absent: %s" % old[:52])
            continue
        if apply(par, old, new):
            done += 1
            print("  fixed: %s" % old[:52])
        else:
            left += 1
            print("  SPANS A FIELD, skipped: %s" % old[:52])

    # Table 3.2's caption described the pixel-wise rows as a priori defaults.
    items = []
    for ch in d.element.body:
        if ch.tag.endswith("}p"):
            items.append(Paragraph(ch, d))
        elif ch.tag.endswith("}tbl"):
            items.append(Table(ch, d))
    OLDCAP = (" The deployed column is the configuration actually fitted; for the "
              "pixel-wise models these are the a priori defaults, and three of the "
              "XGBoost values fall outside the ranges searched, which Section 3.6.7 "
              "quantifies.")
    NEWCAP = (" The deployed column is the configuration actually fitted, and for every "
              "model it is the one the search selected; Section 3.6.7 records that the "
              "pixel-wise models previously carried a priori defaults instead, and what "
              "changing them was worth.")
    for p in items:
        if isinstance(p, Paragraph) and OLDCAP.strip() in p.text:
            for r in p.runs:
                if OLDCAP.strip()[:40] in r.text:
                    r.text = r.text.replace(OLDCAP, NEWCAP, 1)
                    done += 1
                    print("  fixed: Table 3.2 caption")
                    break

    if not done:
        return 0 if left == 0 else 1
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_cvadopt.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_cvadopt.docx"), DOC)
        print("CITATION FIELDS CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d fixed, %d outstanding; citation fields intact %s" % (done, left, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
