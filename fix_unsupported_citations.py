"""Remove claims the cited papers do not make, so Appendix B can be accurate.

A second reviewer found that Appendix B says every citation was checked while
nine unsupported attributions were still in the text. The declaration is the part
that cannot be left overclaiming, so either the passages go or the sentence does.
These are the passages.

Each is reduced to what the source supports, or reworded so the citation attaches
to a claim it can carry. Where the fix needs a different source - the Zimbabwean
irradiance figure, which Lin et al. (East Asia) cannot support - the specific
claim is removed and the gap is marked for the author, because inserting a
citation field is a Zotero operation.
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
    # Damiani et al.: Japan, JRA-55 predictors, NARO observations. The paper
    # evaluates PV output; cloud-shadow gradients and orographic enhancement are
    # not findings it reports.
    ("Their results demonstrate that spatially resolved sub-grid scale variation in "
     "solar radiation, including cloud-shadow gradients and orographic enhancement, "
     "are better reproduced by CNN-based downscaling than by simple interpolation "
     "or delta-mapping, and that this improved spatial fidelity translates int",
     "Their results support the use of CNN-based super-resolution for photovoltaic "
     "output estimation, and that improved spatial fidelity translates int"),

    # Maposa et al. compare variable-selection methods at hourly and daily
    # horizons; the correlation figure and the three-site framing are not in it.
    ("Maposa et al. (2024) achieved Pearson correlation values above 0.92 with RF "
     "for daily GHI prediction across three Southern African sites using cloud "
     "fraction, temperature, and humidity as predictors, a predictor set c",
     "Maposa et al. (2024) evaluated variable-selection methods for solar "
     "irradiance modelling at Southern African radiometric stations, comparing "
     "regularised and quantile regression forests against penalised linear "
     "alternatives, a predictor set c"),

    # Najafi et al. is a critical review framing a performance paradox; it does
    # not crown CNNs and U-Nets, nor rank RF and XGBoost as strongest baselines.
    ("The review confirms that CNNs and U-Nets remain state-of-the-art for this "
     "problem category, with Random Forest and XGBoost consistently identified as "
     "the strongest non-deep-learning baselines, providing interpretab",
     "The review surveys convolutional and encoder-decoder architectures for this "
     "problem category and emphasises a performance paradox, in which models that "
     "score well on historical records degrade under distribution shift. Tree "
     "ensembles remain the common non-deep-learning comparison, providing "
     "interpretab"),

    # SARAH-3 uses the Heliosat method with HelSnow and the SPECMAGIC clear-sky
    # model. HELIOSAT-2 is a different algorithm.
    ("derived from Meteosat visible-channel retrievals using the HELIOSAT-2 algorithm.",
     "derived from Meteosat visible-channel retrievals using the Heliosat method "
     "with the SPECMAGIC clear-sky model."),

    # Eyring et al. (2016) is the CMIP6 experimental design paper, written before
    # the runs existed, and cannot report inter-model rsds spread over Africa.
    ("(Eyring et al., 2016). note that inter-model spread in rsds over Africa "
     "remains substantial, reflecting",
     "Inter-model spread in rsds over Africa remains substantial, reflecting"),

    ("Lin et al., 2023 Applied a U-Net architecture",
     "Lin et al. (2023) applied a U-Net architecture"),

    # Lin et al. is an East Asian dataset and says nothing about Zimbabwe.
    ("with annual GHI values exceeding 2,000 kWh/m² across much of the country "
     "(Lin et al., 2023).",
     "with high annual global horizontal irradiance across much of the country. "
     "[SOURCE NEEDED: cite the Global Solar Atlas or SARAH-3 here and delete the "
     "Lin et al. field, which covers East Asia.]"),
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
            print("  already applied or absent: %s" % old[:54])
            continue
        if apply(par, old, new):
            done += 1
            print("  fixed: %s" % old[:54])
        else:
            left += 1
            print("  SPANS A FIELD, left for Zotero: %s" % old[:54])
    if not done:
        return 0
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_cites2.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_cites2.docx"), DOC)
        print("CITATION FIELDS CHANGED, restored")
        return 1
    print("%d fixed, %d left; citation fields intact" % (done, left))
    return 0


if __name__ == "__main__":
    sys.exit(main())
