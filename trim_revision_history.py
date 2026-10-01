"""Drop the drafting narrative, keep the disclosures that bear on a result.

Seventeen passages narrated earlier versions of the work. An examiner read the
accumulation as an erratum log rather than a thesis. The candour is also a
strength, so this is not a blanket deletion: the six passages that disclose a
correction material to reading a result are kept, and the rest are reworded to
state the methodological point without the drafting history.

KEPT, because the correction changes how a number should be read:
  3.6.7  the hyperparameter search once scored on the withheld record
  3.8.4  the criterion has since been tested against sampling uncertainty
  3.3.3  the SARAH record has since been acquired
  4.5    the sweep predates the hyperparameter correction
  4.7    sigma_DS was previously the monthly RMSE
  4.9    the earlier figures appear in superseded versions of this work

REWORDED here: eight passages whose only content was that a previous draft said
something different. In each the surviving sentence makes the same technical
point, so nothing is lost but the autobiography.
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
    ("because an earlier version of this section reached a different one and the "
     "reasoning is open to challenge either way",
     "because the reasoning is open to challenge either way"),

    ("and the reason is worth stating precisely because an earlier version of this "
     "section overstated it",
     "and the reason is worth stating precisely"),

    ("An earlier version of this section dismissed this leg on the grounds that "
     "every marginal mean-bias interval in Table 3.3 contains zero, which does not "
     "follow.",
     "Dismissing this leg on the grounds that every marginal mean-bias interval in "
     "Table 3.3 contains zero does not follow."),

    ("not removed by this exclusion, as an earlier version of this section stated; they",
     "not removed by this exclusion; they"),

    ("An earlier version of this section described the score as falling below 0.14 "
     "beyond d_ref, which is not what the stated function returns:",
     "Describing the score as falling below 0.14 beyond d_ref would not match the "
     "function as stated:"),

    ("the damping reported above, and an earlier version of this section claimed "
     "that it did.",
     "the damping reported above."),

    ("An earlier version of this section stated that the gradient penalty barely "
     "moves the ratio and that a smoothness prior was therefore refuted. Neither "
     "half of that holds.",
     "It does not follow from this that the gradient penalty barely moves the "
     "ratio, nor that a smoothness prior is refuted."),

    ("An earlier version of this analysis reported the CNN as sitting within 4 per "
     "cent of the target, which was the closest point of the sweep and should not "
     "have been quoted as though the cut were incidental.",
     "Quoting the CNN as within 4 per cent of the target would take the closest "
     "point of the sweep and treat the choice of cut as incidental, which it is not."),
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
    end = at + len(old)
    starts, pos = [], 0
    for r in runs:
        starts.append(pos)
        pos += len(r.text)
    f = max(i for i, st in enumerate(starts) if st <= at)
    l = max(i for i, st in enumerate(starts) if st < end)
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
    print("before:", before)

    done = miss = 0
    for old, new in EDITS:
        par = next((p for p in d.paragraphs if old in p.text), None)
        if par is None:
            print("  absent or already done: %s" % old[:56])
            continue
        if apply(par, old, new):
            done += 1
            print("  reworded: %s" % old[:56])
        else:
            miss += 1
            print("  SPANS A FIELD, left: %s" % old[:56])

    if not done:
        print("nothing to do")
        return 0
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_revhist.docx"))
    d.save(DOC)
    y = docx.Document(DOC).element.xml
    after = (y.count("ZOTERO_ITEM"), y.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_revhist.docx"), DOC)
        print("CITATION FIELDS CHANGED, restored")
        return 1
    print("%d reworded, %d left; citation fields intact" % (done, miss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
