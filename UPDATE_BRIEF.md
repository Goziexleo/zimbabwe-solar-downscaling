# Project update brief

**For dropping into the Claude project so its knowledge matches the repository.**
Everything below happened after the deployment decision was reopened. If your project
knowledge predates that, most of what it holds about model selection is superseded.

---

## The one thing to know

**The deployed model changed from Random Forest to XGBoost**, and the argument for it
changed twice more after that. If a stored note says Random Forest is deployed, or that
the composite selection criterion failed, both are out of date.

| | Current position |
|---|---|
| Deployed | **Pixel-wise XGBoost**, RMSE 9.24, skill 0.5155 vs a 19.08 climatology |
| Selection rule | §3.8.1's **original** pre-registered rule: lowest validation RMSE |
| The composite criterion | **Survives on both legs** — and is outweighed, not absent |
| Why XGBoost anyway | Random Forest **inverts the emission-scenario signal**; that is categorical, not a magnitude trade |

---

## What changed, in order

**Deployment reversed.** A paired year-block bootstrap showed the composite criterion did
not separate Random Forest from XGBoost on either leg, while XGBoost's aggregate advantage
was established and held in all four rolling-origin folds. Then a scenario-discrimination
test found Random Forest's SSP5-8.5 minus SSP2-4.5 separation *shrinks* with lead time
(+0.465 → +0.167) and orders only 71.7% of cells correctly, against XGBoost's 96.7%. The
mechanism is tree extrapolation: temperature leaves the training range 11.48% of the time
at long term under SSP5-8.5 against 1.34% under SSP2-4.5, and a tree saturates outside the
range it was fitted on.

**Then the interval method was corrected, twice.** Percentile intervals are invalid for
centred RMSE, whose bootstrap distribution is biased by construction. Under BCa the Random
Forest **is** established better on centred RMSE (−0.171 [−0.360, −0.041]) and on mean bias
(−0.953 [−1.499, −0.475]). Both legs of the criterion survive. The deployment does not rest
on denying them — it rests on the scenario failure, which no amount of historical fidelity
repairs.

**§3.4.3's quality control did not exist.** EDCM was driving cloud fraction to −5.93% across
751 of 957 cells. Now enforced, with per-cell flag counts. Regenerating everything changed no
conclusion — and the reason is itself evidence: the tree models came back bit-identical
because the out-of-bound values were already outside the training range.

**RQ2 was answered against the wrong comparator.** Measured against *raw GCM output*, which is
what RQ2 actually asks, the chain cuts climatological RMSE 11.29 → 4.82 and lifts spatial
correlation 0.77 → 0.99. Most of the spatial gain is the bias correction, not the ML.

---

## Chapter edits — re-read these, they are not what your notes say

| Chapter | Section | What changed |
|---|---|---|
| 3 | **§3.4.4** | **New:** names the perfect-prognosis design and why MOS was not viable. The old opening implied CMIP6 was a training input. |
| 3 | §3.4.3 | Rewritten: the QC that now exists, with counts |
| 3 | §3.8.4 | Rewritten three times — deployment, then BCa, then both legs |
| 3 | §3.6.6, §3.8.1 | Deployed model corrected to XGBoost |
| 2 | ¶69, ¶71 | SARAH-2/NSRDB now read as planned, not performed |
| 2 | ¶128 | "confirm" → "report"; forward reference to §3.8.4 |
| 2 | gap claim | "closes all four gaps" → three closed, fourth partly |
| 1 | §1.9 | Calibration period corrected to 1985–2010, not 1985–2024 |

Backups sit beside each file with `_BACKUP_pre_*` names.

---

## Still open

- **Dozier & Frew (1990)** is not in Zotero, so it will not reach the reference list.
- **§3.9 suitability analysis and the maps** — next dissertation phase, deliberately not interview scope.
- **Chapters 4–5**, which follow the suitability analysis.
- **SARAH-2 / NSRDB validation** — committed, deferred.

---

## Where things live

- `brief/viva-brief.html` — **the single source for the interview brief.** Edit this, never a generated copy.
- `PROJECT_STATUS.md` — the full work record.
- `brief/build_brief_formats.py --pdf --combined` — rebuilds the PDF, the markdown and the combined reference.
- `check_brief_consistency.py`, `check_status_consistency.py` — guard both documents against stale numbers *and* superseded sentences. Both run under `pytest tests/` (19 tests).
