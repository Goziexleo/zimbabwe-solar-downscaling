"""State the selection rule the thesis actually applies, and update Section 3.8.4.

The fourth critique's first and most serious finding. Chapter 3 still said
XGBoost was deployed because it achieved the lowest validation RMSE. After the
U-Net was retrained without dropout that is false: the U-Net holds the lowest
RMSE and XGBoost is deployed on the spatial axes instead. Three statements had
to be reconciled, and one of them had to be written for the first time:

  3.8.1  "It achieved the lowest validation-period RMSE ... and therefore
         satisfies the selection criterion stated above" - false
  3.8.4  "under the criterion Section 3.8.1 originally specified: the lowest
         validation-period RMSE" - false, and in tension with the same
         section's criticism of a criterion adopted after the ordering was known
  Table 3.3  RMSE described as "one of two axes in the Section 3.8.4 selection
         criterion", which contradicts both of the above

The rule is now written out in three ordered steps, and the honest account of
where the third step came from is given rather than implied. The distinction
from the rejected Random Forest composite is empirical and is quoted: that
composite rested on a structural advantage resampling does not establish,
while XGBoost's over the U-Net is established on both axes.

Section 3.8.4's two-axis paragraph also still carried the superseded U-Net
centred error and an interval that has since changed sign.

    python fix_selection_rule.py
"""

import os
import shutil
import sys

import docx
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
CH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal(2)/UNI ZIM/PROJECT CHAPTERS")
DOC = os.path.join(CH, "CR_Madukwe_Dissertation.docx")
LOCK = os.path.join(CH, "~$_Madukwe_Dissertation.docx")

bca = pd.read_csv(os.path.join(EVAL, "bca_intervals.csv"))
tay = pd.read_csv(os.path.join(EVAL, "taylor_diagram_stats.csv")).set_index("model")
t33 = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")


def pair(a, b, metric):
    """Point estimate and BCa interval for a minus b, ordered low to high."""
    r = bca[(bca.model_a == a) & (bca.model_b == b) & (bca.metric == metric)]
    sgn = 1.0
    if r.empty:
        r = bca[(bca.model_a == b) & (bca.model_b == a) & (bca.metric == metric)]
        sgn = -1.0
    r = r.iloc[0]
    lo, hi = sorted([sgn * r.bca_lo, sgn * r.bca_hi])
    return sgn * r.plug_in, lo, hi


cr = tay["centered_rmse_zw"]
# Pixel-wise against shared-weight, as the section frames it.
_factor = ((cr["CNN"] + cr["U-Net"]) / 2.0) / ((cr["Random Forest"] + cr["XGBoost"]) / 2.0)

ru_r, ru_lo, ru_hi = pair("Random Forest", "U-Net", "RMSE")
rc_r, rc_lo, rc_hi = pair("Random Forest", "CNN", "RMSE")
xu_c, xu_clo, xu_chi = pair("XGBoost", "U-Net", "centred RMSE")
xu_s, xu_slo, xu_shi = pair("XGBoost", "U-Net", "spatial R")
rx_c, rx_clo, rx_chi = pair("Random Forest", "XGBoost", "centred RMSE")
rx_s, rx_slo, rx_shi = pair("Random Forest", "XGBoost", "spatial R")

RULE = (
    "The pixel-wise XGBoost model is deployed for the projection stage. The selection "
    "rule is applied in three ordered steps, and the order carries the argument. First, "
    "a model must pass the scenario-discrimination screen: the separation between the two "
    "pathways must be positive and must grow with lead time, which is a necessary "
    "condition for a projection product and is tested in Section 4.4. Second, among the "
    "models that pass, the lowest validation-period RMSE over Zimbabwe is preferred, but "
    "only where that margin is established by the paired year-block bootstrap. Third, "
    "where the aggregate margin is not established, the model established better on the "
    "spatial structure of the error is preferred, because the product is a map and its "
    "spatial pattern is what a siting decision reads. XGBoost is deployed under the third "
    "step rather than the second: Section 4.2 reports that the retrained U-Net returns "
    "the lower aggregate error, %.2f W/m² against %.2f, and Section 4.4 that the "
    "margin does not survive resampling, while XGBoost's advantage on centred error and "
    "spatial correlation does. Section 3.8.4 gives the route to this rule, including why "
    "an earlier composite criterion that favoured the Random Forest was not sustained."
    % (t33.loc["U-Net", "RMSE_zw"], t33.loc["XGBoost", "RMSE_zw"]))

EDITS = [
    # 3.8.1: the rule, in place of the false claim.
    ("The pixel-wise XGBoost model is deployed for the projection stage. It achieved "
     "the lowest validation-period RMSE across the Zimbabwe domain and therefore "
     "satisfies the selection criterion stated above; the fuller justification, "
     "including why an earlier composite criterion that favoured the Random Forest was "
     "not sustained, is set out in Section 3.8.4.",
     RULE),

    # 3.8.4 opening: name the rule and date the third step honestly.
    ("Four architectures were benchmarked and the pixel-wise XGBoost model is deployed, "
     "under the criterion Section 3.8.1 originally specified: the lowest "
     "validation-period RMSE.",
     "Four architectures were benchmarked and the pixel-wise XGBoost model is deployed, "
     "under the three-step rule Section 3.8.1 states. The third step of that rule, which "
     "prefers the model established better on spatial structure where the aggregate "
     "margin is not established, was formalised after the retrained U-Net of Section 4.5 "
     "returned the lower RMSE. That origin is stated rather than concealed, because the "
     "paragraphs below reject a different criterion partly for having been adopted once "
     "an ordering was known."),

    # 3.8.4: the empirical distinction from the rejected composite.
    ("That criterion has since been tested against sampling uncertainty, and it does not "
     "survive the test.",
     "That criterion has since been tested against sampling uncertainty, and it does not "
     "survive the test. The difference between it and the third step above is empirical "
     "rather than rhetorical, and it is the whole of the defence. The Random Forest's "
     "composite case rested on a structural advantage that resampling does not establish: "
     "its centred-error difference from XGBoost is %+.3f W/m² with a bias-corrected "
     "interval of %+.3f to %+.3f and its spatial-correlation difference %+.4f at %+.4f to "
     "%+.4f, both spanning zero. XGBoost's structural advantage over the U-Net does "
     "survive, at %+.3f (%+.3f to %+.3f) and %+.4f (%+.4f to %+.4f). A rule that promotes "
     "spatial structure above an unestablished aggregate margin therefore selects XGBoost "
     "over the U-Net and would not have selected the Random Forest over XGBoost."
     % (rx_c, rx_clo, rx_chi, rx_s, rx_slo, rx_shi,
        xu_c, xu_clo, xu_chi, xu_s, xu_slo, xu_shi)),

    # Table 3.3's note on the RMSE row.
    ("Penalises large errors; aggregate error; one of two axes in the Section 3.8.4 "
     "selection criterion, not the criterion in itself",
     "Penalises large errors; aggregate error; the second step of the Section 3.8.1 "
     "selection rule, and binding only where its margin is established"),

    # 3.8.4's two-axis paragraph: the superseded U-Net centred error.
    ("at 0.405 W/m² for the Random Forest, 0.531 for XGBoost, 2.682 for the CNN and "
     "3.793 for the U-Net.",
     "at %.3f W/m² for the Random Forest, %.3f for XGBoost, %.3f for the CNN and "
     "%.3f for the U-Net."
     % (cr["Random Forest"], cr["XGBoost"], cr["CNN"], cr["U-Net"])),

    ("where the centred errors differ by a factor of about six",
     "where the centred errors differ by a factor of about %.0f" % _factor),

    # 3.8.4: the Random Forest is no longer unseparated from both networks.
    ("the CNN returned the lower point estimate of the two, and the Random Forest is "
     "separated from neither: +0.577 W/m² against the CNN with a bias-corrected interval "
     "of [-0.368, +1.105], and -0.413 against the U-Net at [-1.052, +0.394], both "
     "containing zero. What is established is the deployed model's margin over both "
     "networks, and the separation within the pixel-wise pair is what fails.",
     "the Random Forest is separated from one network and not the other: %+.3f W/m² "
     "against the CNN with a bias-corrected interval of %+.3f to %+.3f, which contains "
     "zero, and %+.3f against the U-Net at %+.3f to %+.3f, which does not, so the U-Net "
     "is established the better of that pair. What fails is the separation within the "
     "pixel-wise pair and, since the U-Net was retrained, the deployed model's margin "
     "over it as well."
     % (rc_r, rc_lo, rc_hi, ru_r, ru_lo, ru_hi)),
]


def is_field(r):
    x = r._r.xml
    return ("fldChar" in x) or ("instrText" in x)


def apply(par, old, new):
    for r in par.runs:
        if old in r.text:
            r.text = r.text.replace(old, new, 1)
            return True
    runs = par.runs
    full = "".join(r.text for r in runs)
    at = full.find(old)
    if at < 0:
        return False
    starts, pos = [], 0
    for r in runs:
        starts.append(pos)
        pos += len(r.text)
    f = max(i for i, st in enumerate(starts) if st <= at)
    l = max(i for i, st in enumerate(starts) if st < at + len(old))
    span = runs[f:l + 1]
    if any(is_field(r) for r in span):
        return False
    acc = "".join(r.text for r in span)
    k = acc.find(old)
    span[0].text = acc[:k] + new + acc[k + len(old):]
    for r in span[1:]:
        r.text = ""
    return True


def cells(d):
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    yield p


def main():
    if os.path.exists(LOCK):
        print("The dissertation is open in Word. Close it first.")
        return 2
    d = docx.Document(DOC)
    before = (d.element.xml.count("ZOTERO_ITEM"),
              d.element.xml.count('w:fldCharType="begin"'))
    done = left = 0
    for old, new in EDITS:
        if any(new[-70:] in p.text for p in list(d.paragraphs) + list(cells(d))):
            print("  already applied: %s" % old[:50])
            continue
        par = next((p for p in list(d.paragraphs) + list(cells(d)) if old in p.text), None)
        if par is None:
            print("  NOT FOUND: %s" % old[:50])
            left += 1
            continue
        if apply(par, old, new):
            done += 1
            print("  fixed: %s" % old[:50])
        else:
            left += 1
            print("  SPANS A FIELD, skipped: %s" % old[:50])
    if not done:
        return 0 if left == 0 else 1
    shutil.copy(DOC, DOC.replace(".docx", "_BACKUP_pre_rule.docx"))
    d.save(DOC)
    ax = docx.Document(DOC).element.xml
    after = (ax.count("ZOTERO_ITEM"), ax.count('w:fldCharType="begin"'))
    if after != before:
        shutil.copy(DOC.replace(".docx", "_BACKUP_pre_rule.docx"), DOC)
        print("CITATION FIELDS CHANGED %s -> %s, restored" % (before, after))
        return 1
    print("%d fixed, %d outstanding; citation fields intact %s" % (done, left, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
