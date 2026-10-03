#!/usr/bin/env bash
# Rebuild the results chapters and put the dissertation back in order.
#
# The steps must run in this order. Splicing imports the generators' output
# verbatim, so the presentation pass, the table formatting and the front-matter
# lists all have to follow it; running the splice alone silently reintroduces the
# generator's units, spacing and an unformatted table.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== 1/8 regenerate the results chapters ==="
$PY build_chapter4.py >/dev/null
$PY build_chapter5.py >/dev/null

echo "=== 2/8 splice them into the dissertation ==="
$PY splice_results_chapters.py | grep -vE '^\s*$' | tail -4

# The abstract is read from the CSVs too, and must be regenerated here. Leaving
# it out is how it froze at a superseded run while every chapter moved on, and an
# examiner found five different figures for one quantity.
echo "=== 3/8 abstract, from the CSVs ==="
$PY rewrite_abstract.py | tail -2

echo "=== 4/8 presentation: spacing, alignment, units, metadata ==="
$PY fix_presentation.py | tail -3

echo "=== 5/8 factual corrections (idempotent) ==="
$PY fix_facts.py | tail -1

# Lists BEFORE table formatting: the abbreviations list is itself a table, so
# formatting first leaves it in the generator's style.
echo "=== 6/8 front-matter lists ==="
$PY add_front_matter_lists.py | tail -2

echo "=== 7/8 table formatting ==="
$PY normalise_tables.py | tail -2

echo "=== 8/8 verify ==="
$PY - <<'PYEOF'
import os, re, docx
from docx.oxml.ns import qn
D = os.path.expanduser("~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/"
                       "PROJECT CHAPTERS/CR_Madukwe_Dissertation.docx")
d = docx.Document(D); x = d.element.xml
txt = " ".join(p.text for p in d.paragraphs)
# Narrowly the false claim about the scenario signal, not every use of the word:
# "the conversion back to irradiance inverts it exactly" is correct.
bad = {"false 'inverts' claim": len(re.findall(
           r"inverts (?:the scenario|outright)", txt)),
       "W m-2": len(re.findall(r"W m-2", txt)),
       "km2": len(re.findall(r"km2", txt)),
       "bare 2386": len(re.findall(r"\b2386\b", txt)),
       "[n] markers": len(re.findall(r"\[\d{1,2}\]", txt)),
       "El Nino": len(re.findall(r"El Nino", txt)),
       "empty headings": sum(1 for p in d.paragraphs
                             if p.style.name.startswith("Heading") and not p.text.strip())}
sig = {(t.style.name, t._tbl.tblPr.find(qn("w:tblW")).get(qn("w:w")))
       for i, t in enumerate(d.tables) if i}
ok = (not any(bad.values()) and len(sig) == 1 and len(d.sections) == 2
      and x.count('w:fldCharType="begin"') == x.count('w:fldCharType="end"'))
print("  citations %d | fields %d/%d | sections %d | tables %d (%d format) | images %d | words %d"
      % (x.count("ZOTERO_ITEM"), x.count('w:fldCharType="begin"'),
         x.count('w:fldCharType="end"'), len(d.sections), len(d.tables), len(sig),
         x.count("<a:blip"), sum(len(p.text.split()) for p in d.paragraphs)))
print("  defects:", {k: v for k, v in bad.items() if v} or "none")
print("  " + ("ALL CHECKS PASS" if ok else "CHECKS FAILED"))
raise SystemExit(0 if ok else 1)
PYEOF
