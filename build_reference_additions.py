"""Resolve the references the dissertation needs, from CrossRef, into a RIS file.

The reference list holds 23 entries for a 28,000-word thesis; an examiner expects
60 to 100, and three works are cited in the text without appearing in it at all.
This assembles the additions.

Nothing here is transcribed from memory. Each item is looked up against the
CrossRef REST API by title and first author, and the bibliographic fields written
out are the ones CrossRef returns. That matters because the citation audit found a
page range in this very bibliography that was wrong by 900 pages, and because
Appendix B declares that a language model helped with literature - a declaration
that makes every entry worth verifying mechanically rather than by eye.

Items CrossRef cannot resolve are reported, not guessed. Datasets and software
without a DOI are emitted from hand-entered fields and flagged in the output so
the author checks them.

    python build_reference_additions.py

Writes reference_additions.ris, importable into Zotero in one step, and
reference_additions_report.md recording what resolved and what did not.
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

OUT_RIS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "reference_additions.ris")
OUT_REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "reference_additions_report.md")
API = "https://api.crossref.org/works"
MAILTO = "goziemadukwe@gmail.com"      # CrossRef asks for a contact; it is the user's

# (search title, first author surname, why the thesis needs it)
WANTED = [
    # cited in the text but absent from the list
    ("A coefficient of agreement for nominal scales", "Cohen", 1960,
     "Section 3.9.6 reports Cohen's kappa"),
    # required by the corrections in the citation audit
    # Ronneberger 2015 is a Springer LNCS chapter; CrossRef's best hit for the
    # title is a 2017 invited-talk abstract, which is not the paper. Resolved by
    # DOI instead, below.

    ("Bias correction of monthly precipitation and temperature fields from "
     "Intergovernmental Panel on Climate Change AR4 models using equidistant "
     "quantile matching", "Li", 2010,
     "Section 3.4.4 attributes EDCM, previously credited to Lanzante"),
    # downscaling foundations
    ("Precipitation downscaling under climate change: Recent developments to "
     "bridge the gap between dynamical models and the end user", "Maraun", 2010,
     "the standard review of statistical downscaling"),
    ("VALUE: A framework to validate downscaling approaches for climate change "
     "studies", "Maraun", 2015, "the validation framework for downscaling"),
    # bias correction
    ("Bias correction of GCM precipitation by quantile mapping: how well do "
     "methods preserve changes in quantiles and extremes?", "Cannon", 2015,
     "quantile mapping and change preservation"),
    ("Multivariate quantile mapping bias correction: an N-dimensional "
     "probability density function transform for climate model simulations of "
     "multiple variables", "Cannon", None, "multivariate bias correction"),
    # uncertainty
    ("The potential to narrow uncertainty in regional climate predictions",
     "Hawkins", 2009, "the source of the variance-decomposition framing"),
    ("Downscaling and bias-correction contribute considerable uncertainty to "
     "local climate projections in CMIP6", "Lafferty", 2023,
     "the published result that method uncertainty is large"),
    # methods
    ("Summarizing multiple aspects of model performance in a single diagram",
     "Taylor", 2001, "Section 3.7.2 uses the Taylor diagram"),
    # solar models and data
    ("A new airmass independent formulation for the Linke turbidity coefficient",
     "Ineichen", 2002, "Section 3.5.4 uses the Ineichen clear-sky model"),
    ("pvlib python: a python package for modeling solar energy systems",
     "Holmgren", 2018, "the software the clear-sky ceiling is computed with"),
    ("Evaluation of global horizontal irradiance estimates from ERA5 and "
     "COSMO-REA6 reanalyses using ground and satellite-based data", "Urraca", 2018,
     "the published ERA5-against-satellite comparison Section 4.6 parallels"),
    # solar projections
    ("Current and future potential of solar and wind energy over Africa using "
     "the RegCM4 CORDEX-CORE ensemble", "Sawadogo", 2021,
     "the regional projection this study's sign should be set against"),
    ("Enhanced future changes in wet and dry extremes over Africa at "
     "convection-permitting scale", "Kendon", 2019, "CP4-Africa, a 4.5 km projection"),
    ("NASA Global Daily Downscaled Projections, CMIP6", "Thrasher", 2022,
     "NEX-GDDP-CMIP6, the operational baseline product"),
    # CMIP6
    ("The Shared Socioeconomic Pathways and their energy, land use, and "
     "greenhouse gas emissions implications: An overview", "Riahi", 2017,
     "the SSP narratives"),
    ("Causes of Higher Climate Sensitivity in CMIP6 Models", "Zelinka", 2020,
     "Section 4.7 notes two of three GCMs run hot"),
    # the three GCMs
    ("The Australian Earth System Model: ACCESS-ESM1.5", "Ziehn", 2020,
     "ACCESS model family description"),
    ("Evaluation of CMIP6 DECK Experiments With CNRM-CM6-1", "Voldoire", 2019,
     "CNRM-CM6-1 description"),
    # suitability
    ("GIS-based multicriteria decision analysis: a survey of the literature",
     "Malczewski", 2006, "the MCE framework in Section 3.9"),
    # ML methods already cited elsewhere but worth completing
]

# Resolved by DOI rather than by title search. A bibliographic query returned the
# wrong paper for each of these - a rejoinder instead of the article for Efron, a
# tuning note instead of the model description for Mauritsen, unrelated work
# entirely for O'Neill and Wu - so the identifier is given directly and CrossRef
# still supplies every field written out.
BY_DOI = [
    ("10.1080/01621459.1987.10478410", "Section 3.8.2 uses BCa intervals"),
    ("10.5194/gmd-9-3461-2016", "defines the SSP scenarios used"),
    ("10.1002/joc.5462", "the VALUE intercomparison of downscaling methods"),
    ("10.5194/gmd-13-2109-2020",
     "the closest published benchmark of deep learning downscaling"),
    ("10.1029/2018MS001400", "MPI-ESM1-2 model description"),
    ("10.1073/pnas.1611845114", "the closest published African siting study"),
    ("10.5194/essd-16-5243-2024",
     "SARAH-3, the satellite record Chapter 4 validates against; a title search "
     "returns an EGU conference abstract instead of the dataset paper"),
]

# No DOI, or not in CrossRef. Emitted from hand-entered fields and flagged.
MANUAL = [
    dict(ty="CHAP", title="U-Net: Convolutional Networks for Biomedical Image "
                          "Segmentation",
         author="Ronneberger, O., Fischer, P., Brox, T.", year="2015",
         url="https://doi.org/10.1007/978-3-319-24574-4_28",
         note="Section 2.2.2; MICCAI 2015, LNCS 9351, 234-241"),
    dict(ty="JOUR", title="Rapid upslope and downslope computation for efficient "
                          "analysis of DEM",
         author="Dozier, J., Frew, J.", year="1990",
         url="https://doi.org/10.1109/36.58983",
         note="Section 3.5.2 sky-view factor; IEEE TGRS 28(5), 963-969"),
    dict(ty="DATA", title="Global Administrative Areas, version 4.1",
         author="GADM", year="2022", url="https://gadm.org",
         note="national and provincial boundaries, Section 3.3.4"),
    dict(ty="DATA", title="WorldPop constrained population counts, Zimbabwe 2020",
         author="WorldPop", year="2020", url="https://www.worldpop.org",
         note="the population criterion, Section 3.9"),
    dict(ty="DATA", title="ESA Climate Change Initiative Land Cover, v2.1.1",
         author="European Space Agency", year="2022",
         url="https://www.esa-landcover-cci.org",
         note="the land-cover criterion, Section 3.9"),
    dict(ty="DATA", title="World Database on Protected Areas, August 2026 release",
         author="UNEP-WCMC and IUCN", year="2026",
         url="https://www.protectedplanet.net",
         note="the protected-area exclusion, Section 3.9.2"),
    dict(ty="DATA", title="HydroRIVERS, version 1.0", author="Lehner, B., Grill, G.",
         year="2013", url="https://www.hydrosheds.org",
         note="the riparian exclusion, Section 3.9.2"),
    dict(ty="DATA", title="OpenStreetMap, Zimbabwe extract",
         author="OpenStreetMap contributors", year="2026",
         url="https://www.openstreetmap.org",
         note="roads and transmission lines, Section 3.9"),
    dict(ty="DATA", title="ERA5 hourly data on single levels from 1940 to present",
         author="Hersbach, H. et al.", year="2023",
         url="https://doi.org/10.24381/cds.adbb2d47",
         note="the Climate Data Store DOI for the training target itself"),
    dict(ty="COMP", title="Claude (Opus and Sonnet families), large language model",
         author="Anthropic", year="2026", url="https://www.anthropic.com",
         note="declared in Appendix B"),
]


def _norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"<[^>]+>", " ", (t or "").lower()))


def _tokens(t):
    stop = {"a", "an", "the", "of", "and", "for", "in", "on", "to", "with", "from",
            "using", "by", "its", "at", "as"}
    return {w for w in _norm(t).split() if len(w) > 2 and w not in stop}


def _similar(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)          # share of the WANTED title that is present


TITLE_THRESHOLD = 0.70


def crossref(title, author, year_hint=None):
    """Return the CrossRef record only on a strong title match.

    Matching on the author surname alone is not enough: a first pass accepted a
    paper on distributed constraint satisfaction for Dozier, an invited-talk
    abstract for Ronneberger and a hardware keynote for LeCun. A near-miss here
    is worse than a gap, because a plausible-looking wrong citation is exactly
    what the audit of this bibliography already had to catch once.
    """
    q = urllib.parse.urlencode({
        "query.bibliographic": title, "query.author": author,
        "rows": 8, "mailto": MAILTO,
    })
    req = urllib.request.Request(API + "?" + q,
                                 headers={"User-Agent": "thesis-refs/1.0 (%s)" % MAILTO})
    with urllib.request.urlopen(req, timeout=30) as r:
        items = json.loads(r.read()).get("message", {}).get("items", [])

    best, best_score = None, 0.0
    surname = author.split()[0].lower().rstrip("'").replace("-", "")
    for it in items:
        names = " ".join((a.get("family") or "") for a in it.get("author", []))
        if surname not in names.lower().replace("-", ""):
            continue
        cand = (it.get("title") or [""])[0]
        score = _similar(title, cand)
        if year_hint:
            yr = None
            for k in ("published-print", "published-online", "issued"):
                v = it.get(k, {}).get("date-parts", [[None]])[0][0]
                if v:
                    yr = v
                    break
            if yr and abs(yr - year_hint) > 2:
                continue
        if score > best_score:
            best, best_score = it, score
    if best is not None and best_score >= TITLE_THRESHOLD:
        return best, best_score
    return None, best_score


def by_doi(doi):
    req = urllib.request.Request(
        "https://api.crossref.org/works/" + doi,
        headers={"User-Agent": "thesis-refs/1.0 (%s)" % MAILTO})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["message"]


def ris(it, why):
    t = (it.get("title") or [""])[0]
    authors = ["%s, %s" % (a.get("family", ""), a.get("given", ""))
               for a in it.get("author", []) if a.get("family")]
    year = ""
    for k in ("published-print", "published-online", "issued"):
        p = it.get(k, {}).get("date-parts", [[None]])[0][0]
        if p:
            year = str(p)
            break
    ty = {"journal-article": "JOUR", "proceedings-article": "CONF",
          "book-chapter": "CHAP", "book": "BOOK", "posted-content": "JOUR",
          "dataset": "DATA", "report": "RPRT"}.get(it.get("type"), "JOUR")
    lines = ["TY  - " + ty]
    lines += ["AU  - " + a for a in authors]
    lines += ["TI  - " + t]
    if it.get("container-title"):
        lines.append("JO  - " + it["container-title"][0])
    if year:
        lines.append("PY  - " + year)
    if it.get("volume"):
        lines.append("VL  - " + it["volume"])
    if it.get("issue"):
        lines.append("IS  - " + it["issue"])
    if it.get("page"):
        pg = it["page"].split("-")
        lines.append("SP  - " + pg[0])
        if len(pg) > 1:
            lines.append("EP  - " + pg[-1])
    if it.get("DOI"):
        lines.append("DO  - " + it["DOI"])
        lines.append("UR  - https://doi.org/" + it["DOI"])
    lines.append("N1  - " + why)
    lines.append("ER  - ")
    return "\n".join(lines), t, year, it.get("DOI", "")


def manual_ris(m):
    lines = ["TY  - " + m["ty"], "AU  - " + m["author"], "TI  - " + m["title"],
             "PY  - " + m["year"], "UR  - " + m["url"],
             "N1  - " + m["note"] + " [NOT CrossRef-verified: check before submitting]",
             "ER  - "]
    return "\n".join(lines)


def main():
    out, rows, missed = [], [], []
    for title, author, year_hint, why in WANTED:
        try:
            it, score = crossref(title, author, year_hint)
        except Exception as e:
            print("  lookup failed (%s): %s" % (author, e))
            missed.append((title, author, why, "lookup error"))
            continue
        if it is None:
            print("  NOT RESOLVED: %-54s (%s, best title match %.2f)"
                  % (title[:54], author, score))
            missed.append((title, author, why,
                           "no match above %.2f (best %.2f)" % (TITLE_THRESHOLD, score)))
            continue
        block, t, year, doi = ris(it, why)
        out.append(block)
        rows.append((author, year, t, doi, why))
        print("  ok  %-12s %-6s %.2f  %s" % (author, year, score, t[:56]))
        time.sleep(0.4)

    for doi, why in BY_DOI:
        try:
            it = by_doi(doi)
        except Exception as e:
            print("  DOI lookup failed %s: %s" % (doi, e))
            missed.append((doi, "", why, "DOI lookup error"))
            continue
        block, t, year, d = ris(it, why)
        out.append(block)
        rows.append(("(by DOI)", year, t, d, why))
        print("  doi %-12s %-6s      %s" % ("", year, t[:56]))
        time.sleep(0.4)

    for m in MANUAL:
        out.append(manual_ris(m))

    with open(OUT_RIS, "w") as f:
        f.write("\n".join(out) + "\n")

    with open(OUT_REPORT, "w") as f:
        f.write("# Reference additions\n\n")
        f.write("Resolved against the CrossRef API, not transcribed. Import\n"
                "`reference_additions.ris` into Zotero in one step.\n\n")
        f.write("## Resolved (%d)\n\n" % len(rows))
        f.write("| Author | Year | Title | DOI | Why |\n|---|---|---|---|---|\n")
        for a, y, t, d, w in rows:
            f.write("| %s | %s | %s | %s | %s |\n" % (a, y, t[:72], d, w))
        f.write("\n## Hand-entered, NOT CrossRef-verified (%d)\n\n" % len(MANUAL))
        f.write("Datasets and software without a DOI. Check each before "
                "submitting.\n\n")
        for m in MANUAL:
            f.write("- **%s** (%s), %s — %s\n"
                    % (m["title"], m["year"], m["author"], m["note"]))
        if missed:
            f.write("\n## Not resolved (%d) — add by hand or drop\n\n" % len(missed))
            for t, a, w, r in missed:
                f.write("- %s (%s): %s — %s\n" % (t[:72], a, r, w))

    print("\nresolved %d, hand-entered %d, unresolved %d"
          % (len(rows), len(MANUAL), len(missed)))
    print("wrote %s" % OUT_RIS)
    print("wrote %s" % OUT_REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
