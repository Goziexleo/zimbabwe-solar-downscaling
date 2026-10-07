"""Record the U-Net refit as done, in the three places still saying it is not.

The U-Net was refitted at 2e-4 on 7 October, the rate its own inner-split search
selects, and every artefact downstream was regenerated: Table 4.1 reports the
result, 9.01 W/m2 over Zimbabwe against 8.62 at the old rate. Chapter 3 and
Section 5.6 were written before that run and still describe it as an open item,
which is a flat contradiction with Chapter 4 rather than a stale figure:

  - Section 3.6.4 gave the deployed rate as 1e-3.
  - Section 3.6.7 said three of four models carry the configuration their search
    selected, named the U-Net as the exception, said refitting it "would" move
    Table 4.1, and said the deployed U-Net must not be claimed to carry the rate
    the search prefers. All four now do, and it does.
  - Table 3.2's deployed value for the U-Net read 1e-3.
  - Section 5.6 recommended the refit as further research.

Section 3.6.7 also carried a sentence duplicated by two successive edits: "The
exception is the U-Net, and it is recorded here rather than left for a reader to
notice. The exception is the U-Net, and the end of this section gives it."
Correcting the count removes both.

Every figure is read from neural_hpo.csv, table_3_3.csv and the training script,
so none of this can drift from the models again.

    python fix_unet_refit_recorded.py
"""

import os
import re
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

# --- the facts, from the artefacts -----------------------------------------
src = open(os.path.join(ROOT, "train_unet_downscaler.py")).read()
DEP_LR = float(re.search(
    r'os\.environ\.get\(\s*"UNET_LEARNING_RATE"\s*,\s*"([^"]+)"', src).group(1))

n = pd.read_csv(os.path.join(EVAL, "neural_hpo.csv"))
un = n[n.model == "U-Net"]
un_dep = un[un.is_deployed].iloc[0]
un_best = un.loc[un.inner_select_mse.idxmin()]
un_alt = un[~un.is_deployed].sort_values("inner_select_mse").iloc[0]
assert abs(float(un_dep.lr) - DEP_LR) < 1e-12, (un_dep.lr, DEP_LR)
assert un_best.name == un_dep.name, "the deployed U-Net is not the search's pick"

t33 = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")
UNET_RMSE = float(t33.loc["U-Net", "RMSE_zw"])
BEST = t33["RMSE_zw"].idxmin()
# The pre-refit figure is not in any live CSV - it is the number the refit
# replaced - so it is quoted from the commit that performed the refit.
UNET_WAS = 8.62

LR = "%g" % DEP_LR
ALT = "%g" % float(un_alt.lr)

EDITS = [
    # --- Section 3.6.4: the deployed rate ---------------------------------
    ("The U-Net was trained using the Adam optimiser with an initial learning "
     "rate of 1e-3, selected as described in Section 3.6.7,",
     "The U-Net was trained using the Adam optimiser with an initial learning "
     "rate of %s, selected as described in Section 3.6.7," % LR),

    # --- Section 3.6.7: four of four, and the duplicate goes with it -------
    ("and for three of the four the deployed configuration is the one the "
     "search selected. The exception is the U-Net, and it is recorded here "
     "rather than left for a reader to notice. The exception is the U-Net, and "
     "the end of this section gives it.",
     "and for all four the deployed configuration is the one the search "
     "selected."),

    # --- Section 3.6.7: the U-Net result, now adopted ----------------------
    ("For the U-Net it is not: the deployed learning rate of %s scores %.6f "
     "while %s scores %.6f, which is %.1f per cent better."
     % (ALT, float(un_alt.inner_select_mse), LR,
        float(un_dep.inner_select_mse),
        100.0 * (float(un_alt.inner_select_mse) - float(un_dep.inner_select_mse))
        / float(un_alt.inner_select_mse)),
     "For the U-Net the search prefers a learning rate of %s, which scores "
     "%.6f on the inner split against %.6f for the %s it had been trained at, "
     "and %s is the rate it now carries."
     % (LR, float(un_dep.inner_select_mse), float(un_alt.inner_select_mse),
        ALT, LR)),

    # --- Section 3.6.7: refit done, not deferred ---------------------------
    ("The U-Net margin is small, %.4f against %.4f on a squared standardised "
     "anomaly, and the U-Net is not the deployed model: Section 4.4 deploys "
     "XGBoost. Refitting it at the selected rate would move its row in Table "
     "4.1, the architecture term of the uncertainty budget and the spectra, so "
     "it is recorded as an open item in Section 5.6 rather than changed late. "
     "What should not be claimed, and is not claimed here, is that the deployed "
     "U-Net carries the rate this search prefers."
     % (float(un_alt.inner_select_mse), float(un_dep.inner_select_mse)),
     "The margin is small, %.4f against %.4f on a squared standardised anomaly, "
     "and the U-Net was refitted at the selected rate rather than left at the "
     "other: its row in Table 4.1, its contribution to the architecture term of "
     "the uncertainty budget and its spectra are all those of the refitted "
     "model. The refit made it worse on the withheld record, moving it from "
     "%.2f to %.2f W/m² over Zimbabwe, and it is deployed anyway. That is "
     "the point of selecting inside the training period: the alternative was to "
     "keep %s because it scored better on the record the model is reported "
     "against, which is the selection this study declines to make everywhere "
     "else."
     % (float(un_alt.inner_select_mse), float(un_dep.inner_select_mse),
        UNET_WAS, UNET_RMSE, ALT)),

]

# Section 5.6 is NOT edited here. Chapter 5 is regenerated by build_chapter5.py
# on every run of finalise_dissertation.sh, so a document edit to it survives
# only until the next rebuild; the recommendation is rewritten in that generator
# instead, where it derives the deployed rate from the training script. Chapter 3
# is not regenerated, which is why its corrections belong here.

CELL_EDITS = [("learning rate 0.001; dropout 0 (adopted in Section 4.5)",
               "learning rate %s; dropout 0 (adopted in Section 4.5)" % LR)]


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
              len(d.element.body.findall(
                  "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr")))
    done = left = 0
    for old, new in EDITS:
        par = next((p for p in d.paragraphs if old in p.text), None)
        if par is None:
            # Already applied if the replacement is present instead.
            if any(new[:60] in p.text for p in d.paragraphs):
                print("  already applied: %s" % old[:52])
            else:
                left += 1
                print("  NOT FOUND: %s" % old[:52])
            continue
        if apply(par, old, new):
            done += 1
            print("  fixed: %s" % old[:52])
        else:
            left += 1
            print("  SPANS A FIELD, skipped: %s" % old[:52])

    for old, new in CELL_EDITS:
        hit = False
        for t in d.tables:
            for row in t.rows:
                for c in row.cells:
                    for p in c.paragraphs:
                        if old in p.text and apply(p, old, new):
                            done += 1
                            hit = True
                            print("  fixed cell: %s" % old[:52])
        if not hit:
            if any(new in c.text for t in d.tables for row in t.rows
                   for c in row.cells):
                print("  cell already applied: %s" % old[:52])
            else:
                left += 1
                print("  CELL NOT FOUND: %s" % old[:52])

    if not done:
        print("nothing to change")
        return 0 if left == 0 else 1
    bak = DOC.replace(".docx", "_BACKUP_pre_unetrefit_record.docx")
    shutil.copy(DOC, bak)
    d.save(DOC)
    dd = docx.Document(DOC)
    ax = dd.element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'),
             len(dd.element.body.findall(
                 "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr")))
    if after != before:
        shutil.copy(bak, DOC)
        print("STRUCTURE CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d changed, %d outstanding; citations/fields/sections intact %s"
          % (done, left, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
