"""Update Section 3.6.7 for the reran searches.

Three searches were rerun on the rebuilt target: the neural learning-rate and
penalty grid, and the two pixel-wise grids re-stamped so their is_deployed column
names the configuration the training scripts actually carry rather than the
a priori one they replaced. Four claims in the section are now wrong:

  - "for all four the deployed configuration is the one the search selected" is
    true for three of four. On the rebuilt target the U-Net search prefers a
    learning rate of 0.0002 to the deployed 0.001.
  - the neural inner-split figures are from the superseded September run
  - the caveat that those figures predate the rebuild no longer applies
  - "the deployed XGBoost ranks 7 of 25" described the a priori configuration,
    because hpo_pixelwise.py hardcoded the a priori values under the name
    DEPLOYED. Both deployed configurations are in fact the searches' selections.

Every number is read from the regenerated CSVs.

    python fix_search_results_after_rerun.py
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

n = pd.read_csv(os.path.join(EVAL, "neural_hpo.csv"))
cnn, unet = n[n.model == "CNN"], n[n.model == "U-Net"]
cnn_dep = cnn[cnn.is_deployed].iloc[0]
cnn_alt = cnn[(cnn.lr == 0.0005) & (cnn.lambda_gp == 0.01)].iloc[0]
un_dep = unet[unet.is_deployed].iloc[0]
un_best = unet.sort_values("inner_select_mse").iloc[0]


def px(tag):
    d = pd.read_csv(os.path.join(EVAL, "hpo_pixelwise_%s.csv" % tag))
    return (d[d.is_deployed].iloc[0], d[d.is_a_priori].iloc[0], len(d),
            float(d.cv_rmse_csi.min()))


xd, xa, xn, xbest = px("xgb")
rd, ra, rn, rbest = px("rf")

EDITS = [
    ("and for all four the deployed configuration is the one the search selected.",
     "and for three of the four the deployed configuration is the one the search "
     "selected. The exception is the U-Net, and it is recorded here rather than "
     "left for a reader to notice."),

    ("For the CNN a learning rate of 0.001 with a loss weighting of 0.001 scores "
     "0.000400 on the inner split against 0.000424 for the 0.0005 and 0.01 that the "
     "leaked search had chosen; for the U-Net a learning rate of 0.001 scores 0.124575 "
     "against 0.128886 for 0.0002.",
     "For the CNN a learning rate of %s with a loss weighting of %s scores %.6f on the "
     "inner split against %.6f for the %s and %s that the leaked search had chosen, so "
     "the deployed configuration is the one this search selects. For the U-Net it is "
     "not: the deployed learning rate of %s scores %.6f while %s scores %.6f, which is "
     "%.1f per cent better."
     % (("%g" % cnn_dep.lr), ("%g" % cnn_dep.lambda_gp), cnn_dep.inner_select_mse,
        cnn_alt.inner_select_mse, ("%g" % cnn_alt.lr), ("%g" % cnn_alt.lambda_gp),
        ("%g" % un_dep.lr), un_dep.inner_select_mse, ("%g" % un_best.lr),
        un_best.inner_select_mse,
        100.0 * (un_dep.inner_select_mse - un_best.inner_select_mse)
        / un_dep.inner_select_mse)),

    ("One caveat attaches to the two figures just quoted: they come from a search run "
     "before the clear-sky target was rebuilt, so they are not the inner-split scores of "
     "the models actually deployed, which are 0.001245 for the CNN and 0.120408 for the "
     "U-Net. The search was not repeated after the rebuild. What it established was the "
     "ordering, and the learning rate of 0.001 it selected is the one both deployed "
     "models carry.",
     "The U-Net margin is small, %.4f against %.4f on a squared standardised anomaly, "
     "and the U-Net is not the deployed model: Section 4.4 deploys XGBoost. Refitting it "
     "at the selected rate would move its row in Table 4.1, the architecture term of the "
     "uncertainty budget and the spectra, so it is recorded as an open item in Section "
     "5.6 rather than changed late. What should not be claimed, and is not claimed here, "
     "is that the deployed U-Net carries the rate this search prefers."
     % (un_dep.inner_select_mse, un_best.inner_select_mse)),

    ("With each deployed configuration added to its own grid as an extra point, the "
     "deployed XGBoost ranks 7 of 25 at 0.03093 against 0.03001 for the best, 3.1 per "
     "cent higher, and the deployed Random Forest ranks 11 of 18 at 0.03651 against "
     "0.03441 for the best, 6.1 per cent higher.",
     "Both deployed configurations are the selections of their own grids, ranking first "
     "of %d for XGBoost at %.5f and first of %d for the Random Forest at %.5f. The a "
     "priori configurations they replaced rank %d and %d, at %.5f and %.5f, which is %.1f "
     "and %.1f per cent higher."
     % (xn, xd.cv_rmse_csi, rn, rd.cv_rmse_csi, int(xa["rank"]), int(ra["rank"]),
        xa.cv_rmse_csi, ra.cv_rmse_csi,
        100.0 * (xa.cv_rmse_csi - xbest) / xbest,
        100.0 * (ra.cv_rmse_csi - rbest) / rbest)),
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
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_searchrerun.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_searchrerun.docx"), DOC)
        print("CITATION FIELDS CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d fixed, %d outstanding; citation fields intact %s" % (done, left, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
