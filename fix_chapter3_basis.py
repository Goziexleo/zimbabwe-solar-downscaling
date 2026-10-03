"""Put Chapter 3's model-comparison figures on the same basis as Chapter 4.

Chapter 4 reports metrics over Zimbabwe, because 42.8 per cent of the analysis
box is in other countries. Chapter 3 was quoting the full-box column for the same
quantities, so the thesis carried two numbers for one thing - 9.03 against 8.92
for the deployed model's error, and four more pairs besides. An examiner found
five figures for the deployed RMSE across the document.

Model-comparison figures here move to the masked basis. The predictor ablation
stays on the full box, because it is a separate experiment whose A, B and C rows
were computed there, and it is now labelled so rather than left to look like a
third convention.
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

t = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")
y = pd.read_csv(os.path.join(EVAL, "taylor_diagram_stats.csv")).set_index("model")

EDITS = [
    ("the lower aggregate error, at 9.03 W/m² against 10.22",
     "the lower aggregate error over Zimbabwe, at %.2f W/m² against %.2f"
     % (t.loc["XGBoost", "RMSE_zw"], t.loc["Random Forest", "RMSE_zw"])),

    ("at 0.554 W/m² for the Random Forest, 0.687 for XGBoost, 3.155 for the CNN "
     "and 4.341 for the U-Net",
     "at %.3f W/m² for the Random Forest, %.3f for XGBoost, %.3f for the CNN and "
     "%.3f for the U-Net"
     % (y.loc["Random Forest", "centered_rmse_zw"], y.loc["XGBoost", "centered_rmse_zw"],
        y.loc["CNN", "centered_rmse_zw"], y.loc["U-Net", "centered_rmse_zw"])),

    ("Its mean bias is +1.42 W/m² against the Random Forest's +0.22",
     "Its mean bias is %+.2f W/m² against the Random Forest's %+.2f"
     % (t.loc["XGBoost", "MBE_zw"], t.loc["Random Forest", "MBE_zw"])),

    ("For scale, +1.42 is roughly a seventh of a downscaling uncertainty of order "
     "10 W/m²",
     "For scale, %+.2f is roughly a seventh of a downscaling uncertainty of order "
     "10 W/m²" % t.loc["XGBoost", "MBE_zw"]),

    # the ablation is a full-box experiment; say so rather than leave a third number
    ("raising the error from 5.54 to 9.03 W/m²",
     "raising the error from 5.54 to %.2f W/m², both over the full analysis box on "
     "which the ablation was run" % t.loc["XGBoost", "RMSE"]),

    ("the cost of removing it was 9.11 against the 9.03 reported here",
     "the cost of removing it was 9.11 against the %.2f reported here, both on the "
     "full analysis box" % t.loc["XGBoost", "RMSE"]),
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
    x = d.element.xml
    before = (x.count("ZOTERO_ITEM"), x.count('w:fldCharType="begin"'))
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
            print("  SPANS A FIELD: %s" % old[:50])
    if not done:
        print("nothing to do")
        return 0
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_basis.docx"))
    d.save(DOC)
    after_x = docx.Document(DOC).element.xml
    after = (after_x.count("ZOTERO_ITEM"), after_x.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_basis.docx"), DOC)
        print("CITATION FIELDS CHANGED, restored")
        return 1
    print("%d fixed, %d outstanding; citation fields intact" % (done, left))
    return 0


if __name__ == "__main__":
    sys.exit(main())
