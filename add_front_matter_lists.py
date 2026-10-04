"""Add the lists of tables, figures and abbreviations, and make Word refresh fields.

The dissertation has a table of contents but none of the three lists an examiner
expects, and its contents field still shows a cached entry, "Note on this draft",
for a heading that no longer exists. The cache is stale because settings.xml does
not set updateFields, so Word never offers to rebuild it.

Two things are done here:

  updateFields   set in settings.xml, so Word rebuilds the contents, the page
                 numbers and the citation fields when the file is opened. This
                 clears the stale entry without regenerating the field by hand.
  three lists    inserted after the contents control and before Chapter 1.

The lists carry caption text without page numbers. Word can only number them
from captions that use its own Caption style with SEQ numbering, and this
document numbers its captions as literal text; converting them would renumber
every cross-reference in the prose. Caption text alone is the safe form, and the
trade-off is recorded here rather than hidden.
"""

import os
import re
import shutil
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")

ABBREVIATIONS = [
    ("AHP", "Analytic Hierarchy Process"),
    ("AR6", "Sixth Assessment Report of the IPCC"),
    ("BCa", "bias-corrected and accelerated (bootstrap)"),
    ("CCI", "Climate Change Initiative (ESA)"),
    ("CDF", "cumulative distribution function"),
    ("CM SAF", "Satellite Application Facility on Climate Monitoring"),
    ("CMIP6", "Coupled Model Intercomparison Project Phase 6"),
    ("CNN", "convolutional neural network"),
    ("CORDEX", "Coordinated Regional Climate Downscaling Experiment"),
    ("CSI", "clear-sky index"),
    ("DEM", "digital elevation model"),
    ("ECMWF", "European Centre for Medium-Range Weather Forecasts"),
    ("EDCM", "equidistant cumulative distribution function matching"),
    ("ENSO", "El Niño–Southern Oscillation"),
    ("ERA5", "fifth-generation ECMWF atmospheric reanalysis"),
    ("ESA", "European Space Agency"),
    ("ESGF", "Earth System Grid Federation"),
    ("EUMETSAT", "European Organisation for the Exploitation of Meteorological Satellites"),
    ("GCM", "general circulation model"),
    ("GHI", "global horizontal irradiance"),
    ("IPCC", "Intergovernmental Panel on Climate Change"),
    ("ITCZ", "Inter-Tropical Convergence Zone"),
    ("MCE", "multi-criteria evaluation"),
    ("ML", "machine learning"),
    ("NDC", "Nationally Determined Contribution"),
    ("NREP", "National Renewable Energy Policy"),
    ("ONI", "Oceanic Niño Index"),
    ("PV", "photovoltaic"),
    ("QM", "quantile mapping"),
    ("RF", "random forest"),
    ("RMSE", "root-mean-square error"),
    ("RQ", "research question"),
    ("SADC", "Southern African Development Community"),
    ("SARAH", "Surface Solar Radiation Data Set – Heliosat"),
    ("SDSM", "Statistical DownScaling Model"),
    ("SRTM", "Shuttle Radar Topography Mission"),
    ("SSP", "Shared Socioeconomic Pathway"),
    ("SVF", "sky-view factor"),
    ("SZA", "solar zenith angle"),
    ("WLC", "weighted linear combination"),
    ("ZERA", "Zimbabwe Energy Regulatory Authority"),
    ("ZETDC", "Zimbabwe Electricity Transmission and Distribution Company"),
]


def set_update_fields(d):
    """Make Word rebuild every field, including the stale contents, on open."""
    st = d.settings.element
    for e in st.findall(qn("w:updateFields")):
        st.remove(e)
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "true")
    st.append(uf)


def main():
    if os.path.exists(LOCK):
        print("The dissertation is open in Word. Close it first.")
        return 2

    d = docx.Document(DOC)
    x = d.element.xml
    before = (x.count("ZOTERO_ITEM"), x.count('w:fldCharType="begin"'), len(d.sections))
    print("before:", before)

    # Rebuild rather than skip: captions change as chapters are regenerated, and
    # a list that silently went stale is worse than no list.
    from docx.table import Table as _T
    from docx.text.paragraph import Paragraph as _P
    items = []
    for ch in d.element.body:
        if ch.tag.endswith("}p"):
            items.append(_P(ch, d))
        elif ch.tag.endswith("}tbl"):
            items.append(_T(ch, d))
    start = next((i for i, it in enumerate(items)
                  if isinstance(it, _P) and it.text.strip() == "List of Tables"), None)
    if start is not None:
        stop = next(i for i in range(start + 1, len(items))
                    if isinstance(items[i], _P)
                    and items[i].style.name.startswith("Heading")
                    and items[i].text.strip().startswith("Chapter"))
        removed = kept = 0
        for it in items[start:stop]:
            node = it._p if isinstance(it, _P) else it._tbl
            # Never remove a paragraph carrying the section break that starts the
            # body pages. Doing so merges the front matter into the body and
            # takes the page numbering with it.
            pPr = node.find(qn("w:pPr")) if node.tag.endswith("}p") else None
            if pPr is not None and pPr.find(qn("w:sectPr")) is not None:
                for r in list(node.findall(qn("w:r"))):
                    node.remove(r)
                kept += 1
                continue
            node.getparent().remove(node)
            removed += 1
        print("removed %d stale list element(s); %d kept for a section break"
              % (removed, kept))

    if False:
        pass
    else:
        def entry(text):
            """First sentence only. Captions carry several sentences of
            explanation, which belongs under the table and not in a list of
            them; the number plus the leading description is what a reader
            scans for."""
            t = text.strip()
            head = re.match(r"^((?:Table|Figure) \d+\.\d+\.\s*[^.]*\.)", t)
            return head.group(1).strip() if head else t

        tables = [entry(p.text) for p in d.paragraphs
                  if re.match(r"^Table \d+\.\d+\.", p.text.strip())]
        figures = [entry(p.text) for p in d.paragraphs
                   if re.match(r"^Figure \d+\.\d+\.", p.text.strip())]
        print("captions found: %d tables, %d figures" % (len(tables), len(figures)))

        body = list(d.element.body)
        sdt = next(c for c in body if c.tag.endswith("}sdt"))
        anchor = sdt

        def add(text, style="Normal", bold=False, size=11, after=2):
            p = d.add_paragraph(style=style)
            r = p.add_run(text)
            r.bold = bold
            r.font.size = Pt(size)
            p.paragraph_format.space_after = Pt(after)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            return p

        made = []
        for title, items in (("List of Tables", tables), ("List of Figures", figures)):
            h = d.add_paragraph(title, style="Heading 1")
            made.append(h)
            for it in items:
                made.append(add(it, size=10, after=4))

        made.append(d.add_paragraph("List of Abbreviations", style="Heading 1"))
        tbl = d.add_table(rows=0, cols=2)
        tbl.style = d.styles["Table Grid"]
        for ab, full in ABBREVIATIONS:
            row = tbl.add_row()
            row.cells[0].text = ab
            row.cells[1].text = full
            for ci, cell in enumerate(row.cells):
                for par in cell.paragraphs:
                    par.paragraph_format.space_before = Pt(2)
                    par.paragraph_format.space_after = Pt(2)
                    par.paragraph_format.line_spacing = 1.0
                    for r in par.runs:
                        r.font.size = Pt(10)
                        r.bold = (ci == 0)
        made.append(tbl)

        # Move the whole block, in order, to just after the contents control.
        for el in made:
            node = el._p if hasattr(el, "_p") else el._tbl
            anchor.addnext(node)
            anchor = node
        print("inserted %d tables, %d figures, %d abbreviations"
              % (len(tables), len(figures), len(ABBREVIATIONS)))

    set_update_fields(d)
    print("updateFields set: Word will rebuild the contents on open")

    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_lists.docx"))
    d.save(DOC)

    d2 = docx.Document(DOC)
    y = d2.element.xml
    after = (y.count("ZOTERO_ITEM"), y.count('w:fldCharType="begin"'), len(d2.sections))
    print("after: ", after)
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_lists.docx"), DOC)
        print("STRUCTURE CHANGED, restored")
        return 1
    print("citation fields and sections intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
