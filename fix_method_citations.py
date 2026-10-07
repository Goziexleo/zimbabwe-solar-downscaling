"""Cite the methods that were used without a source.

The sixth critique's fourth item. Four methods are described and applied in the
text with no citation: quantile delta mapping, the bias-corrected and accelerated
interval, the PVLIB clear-sky implementation, and the national irradiance figure
in Chapter 1, whose attribution to Lin et al. was removed in an earlier round
without a replacement. The three datasets are cited by name but never with a
year, so a reader cannot match them to their entries.

The citations are plain text, in the same author-year form the Zotero fields
render, so they read correctly now and convert in one step later. The entries
themselves are added to the reference list by insert_missing_references.py, which
has to run after this because it only adds works it finds cited.

    python fix_method_citations.py
"""

import os
import shutil
import sys

import docx

CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")

EDITS = [
    # PVLIB: the implementation as well as the model behind it
    ("computed from the PVLIB Ineichen model, producing a dimensionless ratio to the "
     "clear-sky ceiling",
     "computed from the PVLIB implementation (Holmgren et al., 2018) of the Ineichen "
     "model (Ineichen and Perez, 2002), producing a dimensionless ratio to the clear-sky "
     "ceiling"),

    # BCa, at first use
    ("A bias-corrected and accelerated interval, which corrects the downward bias",
     "A bias-corrected and accelerated interval (Efron, 1987), which corrects the "
     "downward bias"),

    # the datasets, with years so they match their entries
    ("Global Administrative Areas (GADM) database at administrative level 0",
     "Global Administrative Areas database at administrative level 0 (GADM, 2022)"),
    ("ESA CCI Land Cover (300 m): European Space Agency Climate Change Initiative annual "
     "land cover",
     "ESA CCI Land Cover (300 m): European Space Agency Climate Change Initiative annual "
     "land cover (European Space Agency, 2022)"),

    # the national irradiance figure, left uncited when the Lin attribution was removed
    ("exceeding 2,000 kWh/m² across much of the country",
     "exceeding 2,000 kWh/m² across much of the country (Solargis, 2024)"),
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
            print("  already applied or absent: %s" % old[:52])
            continue
        if apply(par, old, new):
            done += 1
            print("  cited: %s" % old[:52])
        else:
            left += 1
            print("  SPANS A FIELD, skipped: %s" % old[:52])
    if not done:
        return 0 if left == 0 else 1
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_methodcites.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_methodcites.docx"), DOC)
        print("CITATION FIELDS CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d cited, %d outstanding; citation fields intact %s" % (done, left, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
