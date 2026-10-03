"""Clear the contradictions a second critique found between Chapter 3 and the rest.

Five passages still describe the state of the work before the clear-sky rebuild
and the uncertainty correction, and one gives an account of the weighting matrix
that Section 3.9.4 denies.

  3.4.3  kept the old sentence that per-cell flag counts are "identically zero"
         directly after the rewrite saying the bound now binds on 826 values
  3.5.4  still describes the index as running 0.25 to 0.63 and the ceiling as
         1.724 times too large, both properties of the superseded denominator
  3.8.3  still defines three components and sigma_DS as a monthly RMSE used "as a
         conservative lower bound" - a conservative estimate is an upper bound,
         and Section 4.7 now uses four components and a systematic term
  4.9    repeats the 1.724 factor
  4.8    says the pairwise matrix "was completed by the researcher in consultation
         with the supervisors"; Section 3.9.4 says no matrix was elicited or
         retained. Only one of those can be true, and the integrity-sensitive one
         is the account that claims an elicitation record exists.
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

u = pd.read_csv(os.path.join(EVAL, "uncertainty_decomposition_summary.csv"))
ux = u[(u.model == "xgb") & (u.period == "long_term_2076_2100")].iloc[0]

EDITS = [
    ("and the per-cell flag counts tabulated for the appendix are accordingly "
     "identically zero at every grid cell.",
     "and the per-cell flag counts tabulated for the appendix are accordingly zero "
     "for the ERA5 variables themselves. The clear-sky bound discussed above is a "
     "separate test and does bind, on 826 of 2,760,480 values."),

    ("since the observed CSI range of roughly 0.25 to 0.63 never reaches the "
     "[0, 1.1] clip bounds, so",
     "and although the index now reaches the upper clip for 0.030 per cent of "
     "values, so"),

    ("compared against the independently derived clear-sky field distributed with "
     "the CM SAF SARAH record, the ceiling used here is larger by a factor of 1.724",
     "the ceiling was formerly larger than a 24-hour mean by a factor of about two, "
     "which Section 3.4.3 records and the rebuilt climatology removes"),

    ("Its denominator averages thirteen daytime hours while the irradiance it "
     "normalises is a 24-hour monthly mean, so the ratio runs a factor of 1.724 "
     "below a true clear-sky index.",
     "Its denominator formerly averaged thirteen daytime hours while the irradiance "
     "it normalises is a 24-hour monthly mean, so the ratio ran about a factor of "
     "two below a true clear-sky index; Section 3.4.3 records the correction and the "
     "models were refitted against the rebuilt target."),

    ("Total uncertainty in the projected GHI fields is decomposed into three "
     "components following standard practice in climate projection uncertainty "
     "attribution.",
     "Total uncertainty in the projected GHI fields is decomposed into four "
     "components following standard practice in climate projection uncertainty "
     "attribution: the spread across global models, the spread across emission "
     "pathways, the spread across downscaling architectures, and the downscaling "
     "error itself."),

    ("Downscaling uncertainty (σDS) is estimated as the RMSE of the corresponding "
     "deployed model on the validation period, propagated uniformly across the "
     "projection domain as a conservative lower bound.",
     "Downscaling uncertainty (σDS) is the part of the deployed model's validation "
     "error that survives a twenty-five-year mean, which is the averaging period the "
     "other three terms describe. The monthly validation RMSE is not that quantity: "
     "random month-to-month error largely averages out of a three-hundred-month mean "
     "while systematic error does not, so the term is formed from the per-cell "
     "time-mean bias and the residual variance divided by an effective sample size "
     "that accounts for month-to-month autocorrelation. Section 4.7 reports the "
     "value and what using the monthly figure instead would do to the budget."),

    ("The total uncertainty envelope at each grid cell and time step is reported as "
     "the root-sum-of-squares combination of the three components",
     "The total uncertainty envelope at each grid cell and time step is reported as "
     "the root-sum-of-squares combination of the four components"),

    ("Section 3.9.4 records that the pairwise comparison matrix was completed by "
     "the researcher in consultation with the supervisors, but that matrix is not "
     "held in the project archive, and",
     "Section 3.9.4 records that no pairwise comparison matrix was elicited or "
     "retained: the weights of Table 3.5 follow the ordinal logic of the Analytic "
     "Hierarchy Process without being its output. Accordingly"),
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
            print("  already applied or absent: %s" % old[:52])
            continue
        if apply(par, old, new):
            done += 1
            print("  fixed: %s" % old[:52])
        else:
            left += 1
            print("  SPANS A FIELD: %s" % old[:52])
    if not done:
        return 0 if left == 0 else 1
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_contradictions.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_contradictions.docx"), DOC)
        print("CITATION FIELDS CHANGED, restored")
        return 1
    print("%d fixed, %d outstanding; citation fields intact" % (done, left))
    return 0


if __name__ == "__main__":
    sys.exit(main())
