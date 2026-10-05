# What is done, what you must do, and in what order

## Part 1 — Zotero, in Word

The prose claims have been corrected. The citation **fields** are yours, because
inserting and deleting them is a Zotero operation. Work through
`CITATION_AUDIT.md`; the short version:

**The reference list is no longer incomplete, but it is not yet Zotero's.**
Nine works that the text cites were missing from the list for five versions
running, and an examiner takes that mark cheaply. They have been inserted as
plain-text entries in alphabetical position, formatted to match the Zotero
output: Cohen, Dozier & Frew, Ineichen & Perez, Lafferty & Sriver, Ronneberger
et al., Skoplaki & Palyvos, and the three datasets (GADM, WorldPop, ESA CCI).
The list now shows 32 entries and nothing cited in the body is unlisted.

**Those nine are a stopgap and a Zotero refresh will delete them, which is
correct.** They sit inside the bibliography field, so the moment you import the
RIS and press Refresh, Zotero regenerates the block from your library and
replaces them with real entries. Nothing has to be removed first. Until you do
that, do not edit them by hand — edit the library instead.

**Import `reference_additions.ris`.** Zotero: File → Import → choose the file.
It carries 41 entries — 32 resolved against the CrossRef API (title, journal,
volume, pages and DOI come from CrossRef, not from me) and 9 hand-entered
datasets and software, each flagged in the note field as needing a check. See
`reference_additions_report.md` for what resolved and why each is needed.

**One entry in that file was wrong and is now fixed.** The hand-entered Dozier &
Frew record carried DOI `10.1109/36.58983`, which resolves to a Price paper on
evapotranspiration, and a title belonging to a third paper. The correct record
is `10.1109/36.58986`, *Rapid calculation of terrain parameters for radiation
modeling from digital elevation data*, IEEE TGRS 28, 963–969. Both it and
Ronneberger are now CrossRef-verified rather than hand-entered. If you had
imported the earlier file, delete that entry and re-import.

**One citation still needs re-pointing, and it is a factual error rather than a
formatting one.** Chapter 1's claim that annual GHI exceeds 2,000 kWh/m² across
much of Zimbabwe was cited to Lin et al. (2023), a downscaling study of East
Asia, which does not support it. The attribution has been removed: the sentence
now reads "...across much of the country, a resource comparable to the regions in
which machine-learning downscaling has so far been demonstrated (Lin et al.,
2023)", which is a claim that paper does support. **The irradiance figure is now
uncited.** `reference_additions.ris` includes a Global Solar Atlas entry for it;
insert that citation after the words "much of the country".

Four title searches returned a plausible but wrong paper and were resolved by DOI
instead: Efron's *Rejoinder* rather than the article, a tuning note rather than
the MPI-ESM description, and unrelated work entirely for O'Neill and Wu. That is
the same failure the citation audit found in the existing list, so treat any
reference you add by hand the same way.

**Also add to your Zotero library**

| Item | Why |
|---|---|
| Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI | Section 2.2.2 now credits them for the U-Net; the text names them but there is no field yet |
| Li, Sheffield & Wood (2010), JGR Atmospheres 115, D10, doi:10.1029/2009JD012882 | the real source of equidistant CDF matching |
| A source for Zimbabwean irradiance | the Global Solar Atlas, or Pfeifroth et al. (2024) for SARAH-3, which you already hold |
| A CORDEX reference | for the archive claim in Section 1.4.1 |
| Cohen (1960) | cited in the text, absent from the list |
| Dozier & Frew (1990) | cited in the text, absent from the list |
| Anthropic (2026), Claude | named in Appendix B, absent from the list |

**Fix in Zotero**

- Buster et al. (2024): pages are **894–906**, not 1–13.
- NREP: item type **Report**, institutional author **Ministry of Energy and Power
  Development**, March 2019. The present entry uses a third-party page title.
- Horn (1981): strip the stray `(PDF)` from the title; publication is not
  *ResearchGate*.
- Wilby et al. (2002): the title's em dash and lowercase `sdsm`.

**Replace or delete these fields in the document**

1. Section 1.1 — the Buster field on the Africa deployment sentence. **Delete**;
   it is a United States study.
2. Section 1.1 — the Lin field on the Zimbabwe irradiance figure. **Replace**
   with the irradiance source you add above.
3. Section 1.4.1 — the Harilal field on the CORDEX archive. **Replace** with the
   CORDEX reference.
4. Section 2.2.2 — the Vandal field after the U-Net description. **Replace** with
   Ronneberger et al. (2015). The sentence already reads correctly for this.
5. Section 2.5.3 — the Buster field on scenario agreement. **Delete**.
6. Section 3.4.4 — the Lanzante field on EDCM. **Replace** with Li, Sheffield
   and Wood (2010).
7. Section 3.8.1 — the Buster field on annual and dry-season means. **Delete**.

Then refresh: Zotero toolbar → **Refresh**.

## Part 2 — Word, once Zotero is done

1. Open the file. Word will ask whether to update fields; say **yes**. That
   rebuilds the table of contents, which currently still caches an entry called
   "Note on this draft" for a heading that no longer exists.
2. Check the three new front-matter lists sit where you want them, between the
   contents and Chapter 1.
3. Read Section 5.2.3. One citation, Breiman (2001), is carried across each time
   Chapters 4 and 5 are regenerated; confirm it survived.

## Part 3 — Publish the repository

Nothing has been uploaded. The repository is committed and clean, with `data/`,
`figures/`, `logs/` and `env/` excluded, no credentials, and 25 tests passing.

```bash
cd "/Users/gozie/Documents/MCSM PROJECT AGY" && git remote add origin https://github.com/goziexleo/zimbabwe-solar-downscaling.git && git push -u origin main
```

Then, in this order:

1. Make the repository **public** on GitHub.
2. Tag the commit the dissertation was built from:
   `git tag -a v1.0-dissertation -m "Version submitted" && git push origin v1.0-dissertation`
3. Archive the release on Zenodo and put the DOI in Appendix A.

Only after the push is the Declaration's claim that the code "is published
openly" true, and the Abstract no longer makes that claim at all. If you decide
not to publish, tell me and I will reword the Declaration instead.

## Rebuilding the document

After any change to a results chapter, run **`./finalise_dissertation.sh`**. It
regenerates both chapters, splices them in, reapplies the presentation pass, the
factual corrections, the front-matter lists and the table formatting, then
verifies. Running the splice alone silently reintroduces the generator's units
and spacing, which is why it is one script and not six commands.

## Part 4 — Decide

- **Abstract length: 412 words.** Most regulations cap at 250–350. Check the MCSM
  rule; tell me the number and I will cut to it. Paragraph three compresses most
  easily.
- **The domain is now disclosed, not corrected.** Section 3.2.1 states the true
  extent and Section 4.9 records the cost: 50 cells inside the boundary but south
  of the grid edge, 5,725 km2, 1.47 per cent of the country, including
  Beitbridge. Figure 3.1 shows the box against the border. The re-download was
  started and stopped at 21 files of 160 - the CDS queue was running at about 45
  minutes a file, which projects to several days. `RUNBOOK_DOMAIN_FIX.md` has the
  procedure if you ever want to run it.
