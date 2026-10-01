"""Correct the prose claims the citation audit found false.

Only the sentences are changed here, never the citation fields. Several of these
paragraphs carry Zotero fields across many runs, and flattening a paragraph
destroys a field, so each replacement is made inside the single run that holds
the text and the field count is asserted before and after.

Removing or swapping a citation is a Zotero operation and is listed for the
author in CITATION_AUDIT.md rather than attempted here.

Each entry is (locator, old, new). The locator is a distinctive phrase that
identifies the paragraph.
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
    ("the climate super-resolution literature that follows it includes",
     "; the climate super-resolution literature that follows it includes",
     ". It reached climate downscaling after the earlier single-image "
     "super-resolution approaches in the field, of which the best known is DeepSD"),

    # Rampal is a review, and its theme is that out-of-distribution behaviour is
    # the open problem - not that deep learning consistently wins.
    ("conducted a systematic benchmarking",
     "conducted a systematic benchmarking of ML downscaling approaches against "
     "CMIP6 baselines, demonstrating that deep learning methods consistently "
     "outperform classical statistical baselines on spatial super-resolution tasks.",
     "reviewed the machine learning downscaling literature and set out the "
     "evaluation problems the field faces, noting in particular that gains shown "
     "on historical records need not survive out of distribution."),

    # Sup3rCC is daily in, hourly out; it cannot support a monthly choice.
    ("Monthly mean fields were used as the primary",
     "Monthly mean fields were used as the primary temporal resolution for model "
     "training, consistent with the approach adopted by Buster et al. for "
     "GCM-to-high-resolution solar energy downscaling.",
     "Monthly mean fields were used as the primary temporal resolution for model "
     "training. This follows from the suitability application, which requires "
     "annual and seasonal means rather than sub-daily variability, and from the "
     "length of the CMIP6 record available for all six predictor variables; "
     "Section 3.6.9 reports a sensitivity experiment at daily resolution."),

    # Rampal again, in the synthesis.
    ("Benchmarking studies report that deep learning",
     "Benchmarking studies report that deep learning models outperform classical "
     "baselines on spatial verification metrics over complex terrain",
     "Reviews of the field report that deep learning models can outperform "
     "classical baselines on spatial verification metrics over complex terrain, "
     "while cautioning that the advantage is not reliable outside the training "
     "distribution"),

    # Neither Buster nor Lin used ERA5 as the high-resolution target: Buster used
    # NSRDB and WTK, Lin used MSWX.
    ("The use of ERA5 as a training target",
     "The use of ERA5 as a training target for ML-based GCM downscaling is now a "
     "well-established practice in the literature",
     "Reanalysis fields are routinely used as the reference in ML-based GCM "
     "downscaling, though the studies closest to this one take a satellite or "
     "station-blended product as the high-resolution target instead"),

    # The U-Net is Ronneberger, Fischer and Brox (2015). DeepSD is a stacked
    # SRCNN with no encoder-decoder and no skip connections.
    ("skip connections concatenate encoder feature maps",
     "skip connections concatenate encoder feature maps directly to corresponding "
     "decoder layers, preserving fine-grained spatial detail that would otherwise "
     "be lost through successive downsampling operations",
     "skip connections concatenate encoder feature maps directly to corresponding "
     "decoder layers, preserving fine-grained spatial detail that would otherwise "
     "be lost through successive downsampling operations. The architecture is due "
     "to Ronneberger, Fischer and Brox (2015). It reached climate downscaling "
     "after the earlier single-image super-resolution approaches in the field, of "
     "which the best known is DeepSD"),

    # Polasky is a self-organising-map study of precipitation, not of ML skill
    # and not of irradiance.
    ("Research on statistical downscaling for West African precipitation",
     "Research on statistical downscaling for West African precipitation has "
     "highlighted the model performance challenges posed by ITCZ-driven convective "
     "regimes that directly affect solar radiation variability",
     "Self-organising-map downscaling of West African precipitation reproduces the "
     "seasonal cycle inland but struggles along the coast, which indicates how "
     "strongly convective regimes condition downscaling skill in the tropics"),
]


def fields(d):
    x = d.element.xml
    return (x.count("ZOTERO_ITEM"), x.count('w:fldCharType="begin"'),
            x.count('w:fldCharType="end"'))


def is_field(run):
    x = run._r.xml
    return ("fldChar" in x) or ("instrText" in x)


def apply(par, old, new):
    """Replace the text, inside one run where possible.

    Where the text is split across runs, the replacement is written into the
    first run of the span and the rest are emptied - but only when every run in
    the span is ordinary text. A Zotero citation occupies several runs, and
    writing over one of them destroys the field, so a span touching any field run
    is refused and left for the author.
    """
    for r in par.runs:
        if old in r.text:
            r.text = r.text.replace(old, new, 1)
            return True

    # Locate the MINIMAL span by character offset. Scanning forward from run 0
    # and growing an accumulator finds the text only once the span has swallowed
    # the paragraph's citation fields, which then refuses every edit.
    runs = par.runs
    full = "".join(r.text for r in runs)
    at = full.find(old)
    if at < 0:
        return False
    end = at + len(old)
    starts, pos = [], 0
    for r in runs:
        starts.append(pos)
        pos += len(r.text)
    first = max(i for i, st in enumerate(starts) if st <= at)
    last = max(i for i, st in enumerate(starts) if st < end)
    span = runs[first:last + 1]
    if any(is_field(r) for r in span):
        return False
    acc = "".join(r.text for r in span)
    local = acc.find(old)
    span[0].text = acc[:local] + new + acc[local + len(old):]
    for r in span[1:]:
        r.text = ""
    return True


def main():
    if os.path.exists(LOCK):
        print("The dissertation is open in Word. Close it first.")
        return 2

    d = docx.Document(DOC)
    before = fields(d)
    print("before:", before)

    done = missed = 0
    for locator, old, new in EDITS:
        hits = [p for p in d.paragraphs if locator in p.text]
        if not hits:
            print("  NOT FOUND: %s" % locator[:60])
            missed += 1
            continue
        par = hits[0]
        if new in par.text:
            print("  already applied: %s" % locator[:60])
            continue
        if apply(par, old, new):
            done += 1
            print("  fixed: %s" % locator[:60])
        else:
            missed += 1
            print("  TEXT SPANS RUNS, left alone: %s" % locator[:60])

    if not done:
        print("nothing to do")
        return 0

    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_citations.docx"))
    d.save(DOC)
    after = fields(docx.Document(DOC))
    print("after: ", after)
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_citations.docx"), DOC)
        print("CITATION FIELDS CHANGED, restored from backup")
        return 1
    print("%d applied, %d left for the author; citation fields intact" % (done, missed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
