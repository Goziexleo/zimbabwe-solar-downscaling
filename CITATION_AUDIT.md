# Citation audit

Every source below was opened on 1 October 2026 and checked against what the
dissertation attributes to it. Verdicts are evidence-based; where a paper was
behind a paywall the publisher's own abstract or record page was used and that is
stated.

This matters more than an ordinary proofreading pass because Appendix B declares
AI assistance. An examiner who sees that declaration will check citations first,
and several of the summaries below have the signature of unverified paraphrase.

## Confirmed wrong

### 1. Buster et al. (2024) did not use ERA5 as a training target

Sup3rCC trained a GAN with **ERA5 as the low-resolution input** and the **NSRDB
(solar) and WTK (wind) as the high-resolution targets**. Input is ~100 km daily
GCM data; output is 4 km hourly. The thesis has this backwards.

Affected: Section 2.6.1, 3.3.2 and 3.6.1 all cite Buster for "the ERA5-as-reference
approach". Section 3.3.1 cites it for *monthly* training, but Sup3rCC is daily in,
hourly out.

Fix: drop Buster as support for using ERA5 as a target, and state the choice as
the study's own with its justification. The thesis has an independent reason that
stands on its own: ERA5 is the only candidate with the predictor fields required
for perfect prognosis over the full period.

### 2. Rampal et al. (2024) is a review, not a benchmarking study

Section 1.4.3 says it "conducted a systematic benchmarking of ML downscaling
approaches against CMIP6 baselines, demonstrating that deep learning methods
consistently outperform classical statistical baselines". It is a review of the
field, and one of its themes is that out-of-distribution generalisation is the
open problem — closer to the opposite of the claim made.

Affected: Sections 1.4.3 and 2.6.1.

### 3. Lin et al. (2023) says nothing about Zimbabwe, and did not use ERA5

The dataset covers **East Asia**, carries **nine** variables (the thesis says six
in Section 2.2.2), and its high-resolution target is **MSWX**, not ERA5.

Affected: Section 1.1 cites it for Zimbabwe annual GHI above 2,000 kWh/m²;
Section 3.3.2 cites it for ERA5 as the training reference; Section 2.6.1 pairs it
with Buster for the same false claim.

Fix: the Zimbabwe irradiance figure needs a real source — the Global Solar Atlas,
or SARAH-3 (Pfeifroth et al. 2024), which this study already holds.

### 4. Vandal et al. (2017) did not introduce the U-Net

Section 2.2.2 describes the U-Net's symmetric encoder-decoder with skip
connections and cites Vandal. DeepSD is a **stacked SRCNN** and has no such
structure. The U-Net is Ronneberger, Fischer and Brox (2015), which the thesis
does not cite.

### 5. EDCM is misattributed

Section 3.4.4 attributes equidistant CDF matching to Lanzante et al. (2020). The
method is **Li, Sheffield and Wood (2010)**, "Bias correction of monthly
precipitation and temperature fields from IPCC AR4 models using equidistant
quantile matching", JGR Atmospheres 115, D10. Lanzante et al. evaluate
distributional methods; they did not introduce this one.

### 6. Harilal et al. (2022) is not a CORDEX reference

Section 1.4.1 cites it for the CORDEX-Africa archive at 0.22 and 0.44 degrees. The
paper is EnhancedSD, a deep-learning solar-irradiance downscaling workshop paper,
and contains no CORDEX archive content. The CORDEX fact is true; the citation is
not its source.

### 7. Buster et al. page numbers are wrong

The reference list gives "Nature Energy 9, 1–13". The article is **9, 894–906**.

## Overextended rather than wrong

### 8. Polasky et al. (2023)

A **self-organising-map** downscaling of **precipitation** in West Africa. Its
finding is that skill differs between the coast and the interior. The thesis uses
it for ML skill and for ITCZ-driven effects on *solar radiation variability*,
which the paper does not address. Keep the citation only where the claim is about
precipitation downscaling skill in West Africa.

### 9. Citations attached to claims they do not support

Buster et al. is cited in Section 1.1 for the case for solar deployment **across
Africa** (it is a United States study), in Section 2.5.3 for whether suitability
conclusions agree across scenarios, and in Section 3.8.1 for deriving annual and
dry-season means. None of these are in that paper.

## Checked and sound

- Damiani et al. (2024): Japan, JRA-55 perfect predictors, NARO observations, CNN,
  and it does evaluate PV power output. The thesis's PV-yield claim stands. The
  separate claim that it motivates a dual-loss with a gradient penalty was not
  verifiable from the abstract and should be checked against the full text.
- Lin et al. (2023) as a U-Net applied to CMIP6 at 0.1 degrees over East Asia:
  correct where stated that way.
- Buster et al. (2024) downscaling GCM output to 4 km for wind and solar at a
  fraction of dynamical cost: correct.

## Not yet verified

Eyring et al. (2016) for CMIP6 rsds biases over southern Africa, Hersbach et al.
(2020) for ITCZ and ENSO cloud failures over southern Africa, Najafi et al. (2026)
on CNN and U-Net status, and Maposa et al. (2024) on the top predictors. The first
two are design and description papers and are unlikely to contain regional
evaluation results of that kind; check them before submission.

## What has to happen in Zotero

Prose claims have been corrected in the document. The citation fields themselves
must be changed in Word, because deleting a field is a Zotero operation:

1. Delete the Buster field in Section 1.1 (the Africa deployment sentence),
   Section 2.5.3 (scenario agreement) and Section 3.8.1 (annual/dry-season means).
2. Delete the Lin field in Section 1.1 and replace it with a real source for
   Zimbabwe irradiance.
3. Delete the Harilal field in Section 1.4.1 and cite CORDEX directly.
4. Replace the Lanzante field in Section 3.4.4 with Li, Sheffield and Wood (2010).
5. Replace the Vandal field in Section 2.2.2 with Ronneberger et al. (2015).
6. Correct the Buster entry's pages to 894–906 in the Zotero item.
7. Add to the library: Ronneberger et al. (2015); Li, Sheffield and Wood (2010);
   a Zimbabwe irradiance source; a CORDEX reference; and the three items the
   declaration names but the list omits, Cohen (1960), Dozier and Frew (1990) and
   the AI tool.
