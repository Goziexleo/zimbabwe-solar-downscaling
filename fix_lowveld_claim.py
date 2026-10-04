"""Correct the two Chapter 2 sentences that put the highest irradiance in the lowveld.

The third critique flagged the claim as unsourced. It is worse than unsourced:
compute_elevation_irradiance_gradient.py shows it is false in both observed
products the thesis uses, and it contradicts the thesis's own results. Section
4.8.1 puts the robust suitable set along the central watershed, and Section 5.2.4
says in terms that irradiance over Zimbabwe spans too narrow a band to separate
sites at all. Chapter 2 was the only place still asserting the opposite.

Numbers are read from the CSV rather than typed, so the passage cannot drift away
from the measurement.

    python fix_lowveld_claim.py
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

g = pd.read_csv(os.path.join(EVAL, "elevation_irradiance_gradient.csv"))


def pick(product, band, col="ghi_mean"):
    return float(g[(g["product"] == product) & (g.band == band)][col].iloc[0])


r_sarah = pick("SARAH", "lowveld", "corr_elev_ghi_national")
r_era5 = pick("ERA5", "lowveld", "corr_elev_ghi_national")
lo, mid, hi = (pick("SARAH", b) for b in ("lowveld", "middleveld", "highveld"))
cells = int(g[(g["product"] == "SARAH")].cells.sum())
spread = 100.0 * (mid - lo) / lo

EDITS = [
    ("since the highest-irradiance zones (the lowveld) are simultaneously the "
     "most remote from existing transmission lines",

     "since distance to the grid varies over two orders of magnitude across the "
     "country while irradiance spans a band of only a few per cent, which makes "
     "grid access rather than resource quality the criterion that separates one "
     "candidate site from another"),

    ("In Zimbabwe's context, this tension is particularly acute: the lowveld, "
     "which exhibits the highest annual GHI values in the country, is also the "
     "region furthest from the existing ZETDC transmission backbone.",

     "In Zimbabwe the tension is weaker than that framing implies, and it does not "
     "run in the direction usually assumed. Measured over the %s cells inside the "
     "national boundary, annual-mean GHI rises weakly with elevation rather than "
     "falling, correlating with it at %+.2f in the CM SAF SARAH retrieval and %+.2f "
     "in ERA5, and the lowveld below 600 m carries the lowest mean of the three "
     "physiographic zones: %.1f W/m² in SARAH against %.1f W/m² for the "
     "middleveld and %.1f W/m² for the highveld. The highest-irradiance ground "
     "is therefore not the most remote ground. Because the band means span only "
     "%.1f per cent while distance to the transmission network varies over two "
     "orders of magnitude, it is grid access and not resource quality that decides "
     "between sites, as Sections 4.8.1 and 5.2.4 report."
     % ("{:,}".format(cells), r_sarah, r_era5, lo, mid, hi, spread)),
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
        if any(new[:60] in p.text for p in d.paragraphs):
            print("  already applied: %s" % old[:52])
            continue
        par = next((p for p in d.paragraphs if old in p.text), None)
        if par is None:
            print("  NOT FOUND: %s" % old[:52])
            left += 1
            continue
        if apply(par, old, new):
            done += 1
            print("  fixed: %s" % old[:52])
        else:
            left += 1
            print("  SPANS A FIELD, skipped: %s" % old[:52])
    if not done:
        return 0 if left == 0 else 1
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_lowveld.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_lowveld.docx"), DOC)
        print("CITATION FIELDS CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d fixed, %d outstanding; citation fields intact %s" % (done, left, before))
    return 0


if __name__ == "__main__":
    sys.exit(main())
