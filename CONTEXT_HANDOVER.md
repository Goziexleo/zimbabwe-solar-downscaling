# Context handover — Zimbabwe solar downscaling dissertation

Written 11 October 2026 for a fresh Claude session. Everything here is verified against
the repository and the document, not recalled.

---

## 1. The project

MSc dissertation, **Chiagozie Raphael Madukwe** (FSR2526653), University of Zimbabwe, MSc
Climate Science and Climate Systems Modelling.

**"Machine Learning-Based Downscaling of GCM Outputs for Solar Energy Suitability Mapping
over Zimbabwe."** Four ML architectures are trained to map bias-corrected CMIP6
atmospheric predictors onto an ERA5-derived clear-sky-index target at 0.1° under a
perfect-prognosis design, applied to SSP2-4.5 and SSP5-8.5 to 2100, and carried into a
multi-criteria suitability assessment for utility-scale PV.

**Status: submitted on 8 October 2026.** The submitted file is
`~/Downloads/CR_Madukwe_Dissertation_DRAFT_rev.docx`. Work since then has been on the
repository and on the author's remaining Zotero tasks, not on the submission.

---

## 2. Where things live

| What | Where |
|---|---|
| Code, data, pipeline | `/Users/gozie/Documents/MCSM PROJECT AGY` (git repo) |
| The dissertation | `~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS/CR_Madukwe_Dissertation.docx` |
| Published repo | `https://github.com/Goziexleo/zimbabwe-solar-downscaling` (public, tag `v1.0-dissertation`) |
| Python | `/opt/anaconda3/envs/climate_stack/bin/python` — the only interpreter with the dependencies |

Always `export PYTHONPATH="$PWD" KMP_DUPLICATE_LIB_OK=TRUE` before running anything.

### Hard constraints

- **Never run `build_thesis.py`.** It would destroy the Zotero fields that exist only in
  the live `.docx`.
- **Never write to `~/Desktop/CR_Madukwe_Thesis_Chapters/`.** Standing commitment.
- **Zotero fields are the author's.** Inserting, deleting or re-pointing a citation is a
  Word operation. Scripts may edit only the plain-text runs *beside* a field, and every
  document-editing script refuses a span that touches one.
- **Check for the Word lock** (`~$_Madukwe_Dissertation.docx`) before writing; the chain
  refuses if Word holds the file.
- `data/` and `brief/` are gitignored. `brief/` was purged from git history before the
  first push.
- The machine is an **8 GB M1 Air**. Two heavy jobs at once thrash it; that killed a
  rolling-origin run. Two workers is the practical ceiling for the pixel-wise jobs.

---

## 3. Document architecture — the thing to understand first

**Chapters 1–3 are edited in place in the `.docx`. Chapters 4–5 are generated from the
evaluation CSVs by `build_chapter4.py` and `build_chapter5.py`, then spliced in.**

`./finalise_dissertation.sh` is the only correct way to rebuild, and the order matters:

```
1 regenerate Ch4 and Ch5   2 splice into the document   3 abstract from the CSVs
4 presentation pass        5 factual corrections        6 front-matter lists
7 table formatting         7b figure freshness          8 verify
```

Two consequences that have each caused a round of defects:

- **The splice used to carry the one field-bearing paragraph across verbatim**, silently
  discarding generator wording changes to it. §5.2.3 said the Random Forest was "third on
  aggregate error" for two rounds after the generator began returning "highest". It now
  transplants only the field's runs into the regenerated paragraph.
- **Chapter 3 does not regenerate.** Every Chapter 4 rebuild therefore left a stale copy
  of the results in the methods chapter — twenty-two values at one point. §3.8 now states
  which comparisons are established and *why that is the criterion*, and leaves the
  numbers to §4.3, §4.4 and Appendix C. `no_intervals_in_methods()` guards the class.

---

## 4. Current state

Document: **83 citations, 215/215 fields balanced, 32 tables, 13 figures, 38,160 words,
25 reference entries.** Five chapters plus Appendices A (reproducibility), B (AI-use
declaration) and C (supporting tables).

Repository: **109 commits, 102 tracked files**, working tree clean at `03fb3e2`.

Deployed model: **XGBoost**, pixel-wise, cross-validated configuration (`max_depth=4,
eta=0.1, subsample=0.7, min_child_weight=5`, 200 rounds, `reg_alpha=0.1`,
`reg_lambda=1.0`).

Validation over Zimbabwe, withheld 2011–2024:

| model | RMSE (W/m²) |
|---|---|
| per-cell OLS baseline | **8.80** |
| XGBoost (deployed) | 8.88 |
| U-Net | 9.01 |
| CNN | 9.56 |
| Random Forest | 9.66 |

All four architectures sit **above** the linear baseline, so the thesis's central negative
result is *"the straight line returns the lowest error of the five."*

### The selection rule, stated in §3.8.1 as three ordered steps

1. Pass the scenario-discrimination screen (separation positive and growing with lead
   time). The Random Forest fails it on monotonicity.
2. Among those passing, prefer the lowest RMSE **where the margin is established** by the
   paired year-block bootstrap.
3. Otherwise prefer the model established better on spatial structure.

§3.8.4 admits step 3 was formalised when the U-Net result arrived, and defends it
empirically: the Random Forest's structural advantage does not survive resampling,
XGBoost's over the U-Net does.

### Design ceiling, which the thesis states plainly

The ERA5 target retains about **0.002 %** of its time-mean variance below the 0.25° grid
it came from. The product is a physically consistent regridding carrying a climate signal,
not a resolution gain. §4.5, §5.3 and the abstract all say so. The U-Net is the only
architecture flagged DAMPED.

---

## 5. Critique history

An external examiner re-scored eight versions. **40 → 49 → 57 → 59 → 65 → 71 → 74 → 77.**
`PROJECT_STATUS.md` §§12q–12w and beyond record each round. The substantive turns:

- **Target leakage and the clear-sky bug.** The clear-sky ceiling averaged 13 daytime
  hours against a 24-hour ERA5 mean. Rebuilt; climatology 502.7 → 278.5 W/m².
- **The linear baseline.** A per-cell OLS on identical predictors beats or matches every
  architecture. The thesis was reframed around this negative result.
- **Dropout.** The U-Net was refit without dropout (8.62 against 11.66) and the
  configuration adopted; XGBoost kept as deployed.
- **Pixel-wise defaults were chosen on the withheld record.** §3.6.7 claimed the
  cross-validated configurations scored worse there. Measured, they scored *better* —
  0.036 for XGBoost and 0.477 for the Random Forest. Both refit on their CV
  configurations and the whole cascade regenerated.
- **The U-Net refit at its own selected learning rate** (2e-4 against 1e-3) made it
  *worse*, 8.62 → 9.01, and was reported anyway. This is the cleanest vindication of the
  rule in the thesis: choosing 1e-3 because it scored better on the withheld record would
  have been selection on the evaluation data. **Do not "fix" this number.**
- **Table 4.6 was mathematically impossible.** Power retained beyond a cut is the
  target-weighted average of the per-wavenumber ratios from that cut upward, so it cannot
  exceed the largest of them; the U-Net's 1.21 beyond wavenumber 3 was impossible when no
  ratio from 3 upward exceeds 1.0085. One array, two different quantities — the table
  computed each field's share of its *own* power. Recomputed: 0.83.
- **PV temperature derating (§4.7.2).** New. The penalty removes between a third and a
  half of the projected irradiance gain, and reverses the scenario ordering for module
  coefficients more negative than about −0.0032 per K — so module technology decides it,
  not climate.

---

## 6. Recurring failure modes — read this before trusting anything

Five classes of error have each recurred. Assume they are still latent.

1. **Scripts hardcoding superseded hyperparameters.** Found in
   `compute_rolling_origin.py`, `hpo_pixelwise.py` (missing `reg_alpha`/`reg_lambda`, so
   the grid searched a different estimator than was deployed) and `hpo_neural.py` (wrong
   `DEPLOYED` map, so `is_deployed` flagged rows the search had not selected). All now
   read from the training scripts. **Check any new consumer the same way.**
2. **Date used as a freshness proxy.** A four-hour rolling-origin rerun reproduced the
   previous numbers to three decimals because the script refitted superseded models; the
   file was new and its contents were not. It now stamps the configuration it used into a
   `config` column and the chapter compares that, not the mtime.
3. **A rebuild aborting mid-chain and nothing noticing.** `set -e` killed a rebuild at its
   configuration-check stage, so the figure stage never ran and the document carried plots
   of superseded models for a day. **Read the completion line, never a grep for stage
   headers.** `finalise_dissertation.sh` now has stage 7b for figure freshness.
4. **Idempotence guards matching the wrong thing.** A `new[:70] in text` check matched the
   *original* where a replacement began with the same words, and matched an
   already-correct occurrence elsewhere while skipping the broken one. Key idempotence on
   a distinctive tail, or on the old text being absent.
5. **Wrong-paper citations.** Four found. A prepared RIS entry for Dozier & Frew carried a
   DOI resolving to a Price paper on evapotranspiration *and* a title from a third paper.
   **Verify every DOI against CrossRef before use; never trust a title search.**

---

## 7. Guards

Three consistency checkers plus a 24-test suite, all run before any commit:

```bash
export PYTHONPATH="$PWD" KMP_DUPLICATE_LIB_OK=TRUE
python check_chapter3_consistency.py    # the dissertation
python check_status_consistency.py      # PROJECT_STATUS.md
python check_brief_consistency.py       # the viva brief and deck
python -m pytest tests/ -q
```

They cover: canonical values present, retired phrases absent, no intervals in the methods
chapter, every Appendix C table referenced, the cut table being a weighted average of the
spectra, no baked results in figure titles, the withdrawn dropout attribution staying
withdrawn, the reconstructed AHP consistency ratio never being published, and prose
typography. **The discipline is that a document is not guarded until something has been
injected into it and the guard has failed.** Several guards were injection-tested both
ways.

---

## 8. Outstanding

**The author's, in Word — these are what the examiner still scores down:**

1. Import `reference_additions.ris` (41 entries, 32 CrossRef-verified) and **Refresh**. The
   list is 25; **eight works cited in the submitted text are not in it** — Dozier, Cohen,
   Skoplaki, Ineichen, Cannon, Efron, Holmgren, GADM. A Zotero refresh on 8 October
   replaced nine stopgap entries I had inserted, which is why the count fell from 32.
2. Insert a Global Solar Atlas citation after "much of the country" in Chapter 1. The
   irradiance figure is deliberately uncited — it had been attributed to Lin et al.
   (2023), a study of East Asia, and the attribution was removed.
3. Switch the Lin field to narrative mode for the parentheses.
4. Convert the remaining plain-text citations in Chapters 4–5 to fields.
5. Open the repository URL logged out. It returns 200 unauthenticated; the examiner's 403
   was their own network.

**Open in the repository:**

- `unet_optimisation.csv` and the neural and rolling-origin reruns all completed on 6–7
  October. `HANDOVER.md` and `PROJECT_STATUS.md` §13 carry the live list.
- A **regional-model benchmark** against CORDEX is named in §5.6 as the comparison most
  likely to overturn the case for the statistical approach. Not run; needs CORDEX data.

---

## 9. A mistake made in this session, as a warning

On 11 October I inferred from a file date and an `inner_select_mse` of 0.08637 that the
deployed U-Net checkpoint had been corrupted by `hpo_neural.py`, "restored" the superseded
1e-3 model over it, and regenerated the validation fields and Table 3.3 from the wrong
model. The 9.01 was **deliberate** — commit `abdd4f5`. `PHASE_A_ONLY` raises
`SystemExit(0)` before `torch.save`, so Phase A never writes a checkpoint and the bug I
"fixed" did not exist. All of it was reverted and the guards re-pass.

**The lesson: read the git log before concluding that a number is wrong.** Nine commits of
deliberate work sat between my last session and that inference, and the commit messages
said exactly what had happened.

---

## 10. Conventions

- Deliverables are copied into the OneDrive `PROJECT CHAPTERS` folder, not left in the
  repo. `brief/build_brief_formats.py --pdf --combined --sync` does it, and
  `brief/sync_brief_tables.py` rewrites the brief's four numeric tables from the CSVs
  first — hand-patching them failed the brief guard three rounds running.
- `PROJECT_STATUS.md` is the authoritative work record; every round appends a numbered
  section. It is long on purpose.
- Numbers in generated prose are read from CSVs, never typed. Where a superseded figure
  must be quoted it is quoted as history, with the reason it is superseded beside it.
- Commit messages state what was wrong, what the evidence was and what changed. They end
  with the `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` attribution line.
