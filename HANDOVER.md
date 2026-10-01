# What is done, what you must do, and in what order

## Part 1 — Zotero, in Word

The prose claims have been corrected. The citation **fields** are yours, because
inserting and deleting them is a Zotero operation. Work through
`CITATION_AUDIT.md`; the short version:

**Add to your Zotero library**

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

## Part 4 — Decide

- **Abstract length: 412 words.** Most regulations cap at 250–350. Check the MCSM
  rule; tell me the number and I will cut to it. Paragraph three compresses most
  easily.
- **Revision-history sentences.** Seventeen passages narrate earlier versions.
  The examiner wants them all gone; your candour was also scored a strength. I
  would keep the six that disclose a correction material to reading a result —
  the leakage correction in 3.6.7, the Random Forest reversal in 3.8.4, the
  sigma_DS error in 4.7 — and delete the eleven that only narrate drafting. Say
  the word and I will do exactly that.
- **The domain.** The grid stops at 22.0 S and Zimbabwe does not.
  `RUNBOOK_DOMAIN_FIX.md` gives both paths: the full rerun, or the wording to
  disclose it. This is the largest outstanding item.
