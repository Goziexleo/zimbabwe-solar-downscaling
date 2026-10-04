"""Replace Chapters 4 and 5 in the dissertation with freshly generated ones.

Chapters 4 and 5 are build products: build_chapter4.py and build_chapter5.py
write every number in them from the evaluation CSVs. Chapters 1 to 3 and the
front matter are hand-written and carry all 77 Zotero citation fields, so the
whole document cannot be rebuilt without destroying that work.

The field audit makes a narrower operation safe. Chapter 4 contains no citation
fields at all and Chapter 5 contains exactly one, a Breiman reference in the
paragraph on tree extrapolation. So the two results chapters can be swapped
wholesale while everything else is left untouched, and that one field is carried
across explicitly.

Two things a naive element copy gets wrong, both guarded here:

  images     copying a run that holds a drawing copies the reference but not the
             image part, so every figure renders as a broken link. Each blip's
             image is added to the target package and its r:embed rewritten.
  sections   a paragraph can carry the section break that starts the body pages.
             Removing one silently merges the front matter into the body and the
             page numbering with it, which has happened in this project before.
             The target range is asserted to contain none, and none is imported.

Afterwards the table formatting is reapplied, because the generated tables arrive
in the generator's style rather than the unified one.

    python splice_results_chapters.py [--dry-run]
"""

import argparse
import copy
import io
import os
import re
import shutil
import sys

import docx
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")
SOURCES = [os.path.join(CH, "CR_Madukwe_Chapter4_DRAFT.docx"),
           os.path.join(CH, "CR_Madukwe_Chapter5_DRAFT.docx")]

START_TEXT = "Chapter 4:"
END_TEXT = "References"


def integrity(path):
    d = docx.Document(path)
    x = d.element.xml
    return dict(zotero=x.count("ZOTERO_ITEM"),
                fld_begin=x.count('w:fldCharType="begin"'),
                fld_end=x.count('w:fldCharType="end"'),
                sections=len(d.sections),
                blips=x.count("<a:blip"),
                tables=len(d.tables))


def find_bounds(d):
    body = list(d.element.body)
    start = end = None
    for i, ch in enumerate(body):
        if not ch.tag.endswith("}p"):
            continue
        p = Paragraph(ch, d)
        t = p.text.strip()
        if start is None and t.startswith(START_TEXT) and p.style.name.startswith("Heading"):
            start = i
        if t == END_TEXT and p.style.name.startswith("Heading"):
            end = i
    if start is None or end is None or end <= start:
        raise SystemExit("could not locate the Chapter 4 and References headings")
    return body, start, end


def rebind_images(el, src_doc, tgt_doc):
    """Copy each referenced image into the target package and repoint the run.

    The image is handed over as BYTES, not as a part. Relating the source part
    directly keeps that part's own name, and the target already holds parts
    called image1.png and so on from the chapters being replaced, so the package
    ends up with two zip entries of the same name - which it did on the first
    attempt. Going through get_or_add_image lets the package assign a free name
    and deduplicate by content, so a figure that did not change is reused rather
    than stored twice.
    """
    n = 0
    for blip in el.iter(qn("a:blip")):
        rid = blip.get(qn("r:embed"))
        if not rid:
            continue
        blob = src_doc.part.related_parts[rid].blob
        new_rid, _ = tgt_doc.part.get_or_add_image(io.BytesIO(blob))
        blip.set(qn("r:embed"), new_rid)
        n += 1
    return n


def prune_orphan_images(d):
    """Drop image relationships the replaced chapters left behind."""
    used = {b.get(qn("r:embed")) for b in d.element.body.iter(qn("a:blip"))}
    used |= {b.get(qn("r:link")) for b in d.element.body.iter(qn("a:blip"))}
    rels = d.part.rels
    dead = [rid for rid, rel in list(rels.items())
            if rel.reltype == RT.IMAGE and rid not in used]
    for rid in dead:
        del rels[rid]
    return len(dead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if os.path.exists(LOCK) and not args.dry_run:
        print("The dissertation is open in Word. Close it first.")
        return 2

    before = integrity(DOC)
    print("before:", before)

    d = docx.Document(DOC)
    body, start, end = find_bounds(d)
    segment = body[start:end]
    print("replacing body[%d:%d] = %d elements" % (start, end, len(segment)))

    if any(True for e in segment for _ in e.iter(qn("w:sectPr"))):
        raise SystemExit("REFUSED: the range carries a section break")

    # The one citation field in the range, kept so it can be put back.
    #
    # Only the field's own runs are kept, not the paragraph around them. An
    # earlier version carried the whole old paragraph across, which silently
    # discarded every wording change the generator made to it: Section 5.2.3
    # went on saying the Random Forest was "third on aggregate error" for two
    # rounds after the retrain made it fourth, because the splice kept putting
    # the old sentence back. The field is now transplanted into the regenerated
    # paragraph instead, so the prose is the generator's and the field is the
    # document's.
    keeper = keeper_lead = None
    for e in segment:
        if "ZOTERO_ITEM" in e.xml and e.tag.endswith("}p"):
            kp = Paragraph(e, d)
            runs = kp.runs
            b = next((i for i, r in enumerate(runs)
                      if 'fldCharType="begin"' in r._r.xml), None)
            t = next((i for i, r in enumerate(runs)
                      if 'fldCharType="end"' in r._r.xml), None)
            assert b is not None and t is not None and t > b, (
                "the preserved field is not a contiguous begin..end block")
            keeper = [copy.deepcopy(r._r) for r in runs[b:t + 1]]
            keeper_lead = " ".join(kp.text.strip().split()[:8])
            keeper_text = "".join(r.text for r in runs[b:t + 1]).strip()
            print("preserving the citation field %r from: %s..."
                  % (keeper_text, kp.text.strip()[:60]))
    if args.dry_run:
        print("dry run, nothing written")
        return 0

    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_splice.docx"))

    anchor = body[end]                      # the References heading
    for e in segment:
        e.getparent().remove(e)

    imported = images = 0
    for src_path in SOURCES:
        src = docx.Document(src_path)
        for ch in list(src.element.body):
            if ch.tag.endswith("}sectPr"):
                continue
            if any(True for _ in ch.iter(qn("w:sectPr"))):
                continue                    # never import a section break
            new = copy.deepcopy(ch)
            images += rebind_images(new, src, d)
            anchor.addprevious(new)
            imported += 1
        print("  %-34s %d elements" % (os.path.basename(src_path), imported))

    print("imported %d elements, rebound %d images" % (imported, images))

    # Put the citation field back: find the regenerated paragraph that carries
    # the same plain-text citation and swap the old runs in for it.
    if keeper is not None:
        from docx.text.run import Run
        done = False
        for ch in list(d.element.body):
            if not ch.tag.endswith("}p") or "ZOTERO_ITEM" in ch.xml:
                continue
            par = Paragraph(ch, d)
            if not par.text.strip().startswith(keeper_lead):
                continue
            # Replace the generator's plain-text citation with the real field,
            # splitting the run it sits in so the surrounding prose is kept.
            for r in par.runs:
                if keeper_text not in r.text:
                    continue
                head, tail = r.text.split(keeper_text, 1)
                r.text = head
                anchor = r._r
                for fr in keeper:
                    anchor.addnext(fr)
                    anchor = fr
                if tail:
                    tail_el = copy.deepcopy(r._r)
                    anchor.addnext(tail_el)
                    Run(tail_el, par).text = tail
                print("transplanted the citation field into the regenerated "
                      "paragraph, keeping the generator's wording")
                done = True
                break
            if done:
                break
        if not done:
            print("NOTE: the %s field could not be re-sited; the regenerated "
                  "paragraph no longer carries the plain-text citation it "
                  "replaces. Re-insert it in Word (Section 5.2.3)." % keeper_text)

    dropped = prune_orphan_images(d)
    print("pruned %d orphaned image relationship(s)" % dropped)

    d.save(DOC)
    after = integrity(DOC)
    print("after: ", after)

    ok = (after["blips"] == before["blips"]
          and after["sections"] == before["sections"]
          and after["fld_begin"] == after["fld_end"])
    if not ok:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_splice.docx"), DOC)
        print("VERIFICATION FAILED, restored from backup")
        return 1
    print("verified: images %d, sections %d, fields balanced %d/%d"
          % (after["blips"], after["sections"], after["fld_begin"], after["fld_end"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
