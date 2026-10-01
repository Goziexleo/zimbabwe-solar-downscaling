"""Give every data table in the dissertation one format.

The merged document inherited its tables from two generators and they do not
match. Chapters 1 to 3 carry hand-made tables: style Normal Table, explicit
borders, a bold and shaded header that repeats across pages, a fixed 9026 twip
width, and font sizes running 10, 10.5 and 11. Chapters 4 and 5 are generated:
style Table Grid, borders inherited from the style, a bold but unshaded header
that does not repeat, automatic width, uniform 10pt.

Neither width is right. The text column is 8505 twips (A4 less a 1984 twip inside
margin and a 1417 twip outside margin), so the Chapter 1 to 3 tables overhang the
right margin by 521 twips and the Chapter 4 to 5 tables by about 135. Column
widths are therefore rescaled to the measure, not just the table width, because
Word lays the table out from the grid when the two disagree.

One standard is applied to all eighteen data tables:

    borders      single, 0.5pt, all four edges and both insides
    header row   bold, lightly shaded, repeats at every page break
    body text    10pt
    width        8505 twips, columns rescaled in proportion
    cell padding 2pt above and below, single-spaced

The declaration signature block is excluded and has its borders REMOVED. It is a
layout table, not a data table, and it was carrying a full grid: ruled lines
through a block of signature lines read as a defect.

Nothing outside the tables is altered: no body paragraph spacing, no heading or
caption formatting. Paragraph spacing INSIDE cells is set, because the two
families otherwise sit differently in their cells even once the borders, width
and font agree.

Zotero fields are counted before and after. If the count moves the save is rolled
back, because collapsing runs has destroyed a citation field in this project
before and the count is the only thing that catches it.

    python normalise_tables.py [--dry-run] [--out PATH]
"""

import argparse
import os
import re
import shutil
import sys

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

CHAPTERS = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CHAPTERS, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CHAPTERS, "~$_Madukwe_Dissertation.docx")

BODY_PT = Pt(10)
HEADER_SHADE = "EDEDF2"          # a light tint of the University purple
EDGE_SZ = 4                      # eighths of a point, so 0.5pt
CAPTION = re.compile(r"^\s*Table\s+\d+\.\d+\.")
SIGNATURE_FIRST_CELL = "Candidate:"


def _replace(parent, tag, el):
    for old in parent.findall(qn(tag)):
        parent.remove(old)
    if el is not None:
        parent.append(el)


def set_borders(tbl, present=True):
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "single" if present else "none")
        e.set(qn("w:sz"), str(EDGE_SZ) if present else "0")
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "auto")
        b.append(e)
    _replace(tbl._tbl.tblPr, "w:tblBorders", b)


def shade(cell, fill):
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear")
    s.set(qn("w:color"), "auto")
    s.set(qn("w:fill"), fill)
    _replace(cell._tc.get_or_add_tcPr(), "w:shd", s)


def margins(tbl, pt=5):
    """Even cell padding. Without this the two families sit differently in
    their cells even once the borders and the font agree."""
    m = OxmlElement("w:tblCellMar")
    for side, v in (("top", 20), ("start", pt * 20), ("bottom", 20), ("end", pt * 20)):
        e = OxmlElement("w:" + side)
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        m.append(e)
    _replace(tbl._tbl.tblPr, "w:tblCellMar", m)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    if trPr.find(qn("w:tblHeader")) is None:
        trPr.append(OxmlElement("w:tblHeader"))


def rescale(tbl, target):
    """Fit the table to the text column, grid included.

    Setting tblW alone is not enough: where the grid disagrees with the declared
    width Word lays the table out from the grid, so the overhang survives. The
    gridCol widths are scaled in proportion and the rounding remainder is given
    to the last column so the parts sum to the target exactly. Each cell's own
    width is then rewritten from the columns its gridSpan covers.
    """
    grid = tbl._tbl.findall(".//" + qn("w:gridCol"))
    if not grid:
        return
    cur = [max(1, int(g.get(qn("w:w")) or 1)) for g in grid]
    tot = sum(cur)
    new = [max(1, round(c * target / tot)) for c in cur]
    new[-1] += target - sum(new)
    for g, w in zip(grid, new):
        g.set(qn("w:w"), str(w))

    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), str(target))
    w.set(qn("w:type"), "dxa")
    _replace(tbl._tbl.tblPr, "w:tblW", w)

    # Autofit, not fixed. The rescaled grid is a starting hint; Word then sizes
    # columns from content while honouring the preferred width above. Twelve cells
    # across the set hold a word slightly wider than its rescaled column - the
    # widest is "(convolutional)" - and a fixed layout breaks those mid-word.
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "autofit")
    _replace(tbl._tbl.tblPr, "w:tblLayout", lay)

    for row in tbl._tbl.findall(qn("w:tr")):
        col = 0
        for tc in row.findall(qn("w:tc")):
            tcPr = tc.find(qn("w:tcPr"))
            if tcPr is None:
                tcPr = OxmlElement("w:tcPr")
                tc.insert(0, tcPr)
            span = tcPr.find(qn("w:gridSpan"))
            n = int(span.get(qn("w:val"))) if span is not None else 1
            cw = OxmlElement("w:tcW")
            cw.set(qn("w:w"), str(sum(new[col:col + n]) or new[-1]))
            cw.set(qn("w:type"), "dxa")
            _replace(tcPr, "w:tcW", cw)
            col += n


def text_width(doc):
    s = doc.sections[0]
    return s.page_width.twips - s.left_margin.twips - s.right_margin.twips


def body_items(doc):
    out = []
    for ch in doc.element.body:
        if ch.tag.endswith("}p"):
            out.append(Paragraph(ch, doc))
        elif ch.tag.endswith("}tbl"):
            out.append(Table(ch, doc))
    return out


def caption_after(items, n):
    for k in range(n + 1, min(n + 4, len(items))):
        it = items[k]
        if isinstance(it, Table):
            return None
        if it.text.strip():
            return it if CAPTION.match(it.text.strip()) else None
    return None


def integrity(path):
    d = docx.Document(path)
    x = d.element.xml
    return (x.count("ZOTERO_ITEM"), x.count('w:fldCharType="begin"'),
            x.count('w:fldCharType="end"'), len(d.tables), len(d.paragraphs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", help="write here instead of in place")
    args = ap.parse_args()
    dest = args.out or DOC

    if dest == DOC and os.path.exists(LOCK) and not args.dry_run:
        print("The dissertation is open in Word (lock file present).")
        print("Close it first, or Word overwrites this on its next save.")
        print("To see the result without touching the open file, pass --out PATH.")
        return 2

    d = docx.Document(DOC)
    target = text_width(d)
    items = body_items(d)
    tpos = [n for n, it in enumerate(items) if isinstance(it, Table)]
    print("text column %d twips; %d tables\n" % (target, len(tpos)))

    sizes, caps = set(), 0
    for ti, n in enumerate(tpos):
        t = items[n]
        first = t.rows[0].cells[0].text.strip() if t.rows else ""
        for r in t.rows:
            for c in r.cells:
                for p in c.paragraphs:
                    for run in p.runs:
                        if run.font.size:
                            sizes.add(run.font.size.pt)

        if first.startswith(SIGNATURE_FIRST_CELL):
            if not args.dry_run:
                set_borders(t, present=False)
            print("  table %-2d  signature block, borders removed" % ti)
            continue

        if not args.dry_run:
            t.style = d.styles["Table Grid"]
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            set_borders(t, present=True)
            margins(t)
            rescale(t, target)
            for ri, row in enumerate(t.rows):
                for cell in row.cells:
                    if ri == 0:
                        shade(cell, HEADER_SHADE)
                    for p in cell.paragraphs:
                        pf = p.paragraph_format
                        pf.space_before, pf.space_after = Pt(2), Pt(2)
                        pf.line_spacing = 1.0
                        for run in p.runs:
                            run.font.size = BODY_PT
                            if ri == 0:
                                run.bold = True
            repeat_header(t.rows[0])

        # Captions are REPORTED but never touched. The author asked for the tables
        # alone to be made consistent, and explicitly not the titles around them.
        cap = caption_after(items, n)
        if cap is not None:
            caps += 1
        print("  table %-2d  %-22s %dx%d  caption %s"
              % (ti, first[:22], len(t.rows), len(t.columns),
                 cap.text.strip().split(".")[1] if cap is not None else "MISSING"))

    print("\nfont sizes found before: %s -> all 10.0" % sorted(sizes))
    print("captions found: %d (left untouched)" % caps)

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    if dest == DOC:
        shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_tables.docx"))
    d.save(dest)

    before, after = integrity(DOC if dest != DOC else
                              DOC.replace(".docx", "_BACKUP_pre_tables.docx")), integrity(dest)
    ok = before == after
    print("\ncitation fields and structure: %s" % ("intact" if ok else
          "CHANGED %s -> %s" % (before, after)))
    if not ok and dest == DOC:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_tables.docx"), DOC)
        print("restored from backup; nothing applied")
        return 1
    print("wrote %s" % dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
