"""Rewrite the Table 3.3 rows in PROJECT_STATUS.md and the viva brief from the CSVs.

Both documents carry the validation table as literal rows, and both went stale
when the clear-sky correction and the U-Net reseed changed every figure in it.

Whole rows are rebuilt rather than individual numbers replaced. A bare search for
"9.24" would have hit nineteen places in PROJECT_STATUS alone, most of them
deliberate history - the sigma_DS narrative, the leakage before-and-after - and
rewriting those would destroy the record of what was corrected and when.
"""

import os
import re
import shutil
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
STATUS = os.path.join(ROOT, "PROJECT_STATUS.md")
BRIEF = os.path.join(ROOT, "brief/viva-brief.html")


def load():
    t = pd.read_csv(os.path.join(EVAL, "table_3_3.csv")).set_index("model")
    ci = pd.read_csv(os.path.join(EVAL, "bootstrap_ci.csv"))
    u = pd.read_csv(os.path.join(EVAL, "uncertainty_decomposition_summary.csv"))
    b = pd.read_csv(os.path.join(EVAL, "bca_intervals.csv"))
    roll = pd.read_csv(os.path.join(EVAL, "rolling_origin.csv"))
    return t, ci, u, b, roll


def ci_for(ci, model):
    """The resampled RMSE interval, whatever the column names happen to be."""
    m = ci[ci.iloc[:, 0].astype(str).str.strip() == model]
    if m.empty:
        return None
    row = m.iloc[0]
    lo = next((row[c] for c in ci.columns if re.search(r"lo|lower|2\.5", str(c), re.I)), None)
    hi = next((row[c] for c in ci.columns if re.search(r"hi|upper|97\.5", str(c), re.I)), None)
    return (lo, hi)


def main():
    t, ci, u, b, roll = load()
    changed = []

    # ---- the viva brief's Table 3.3 ----
    s = open(BRIEF).read()
    orig = s
    for model, label in (("XGBoost", "XGBoost — deployed"), ("U-Net", "U-Net"),
                         ("CNN", "CNN"), ("Random Forest", "Random Forest")):
        r = t.loc[model]
        pat = re.compile(
            r'(<tr[^>]*><td class="name">' + re.escape(label) +
            r'</td>)<td>[^<]*</td><td>([^<]*)</td><td>[^<]*</td>'
            r'<td>[^<]*</td><td>[^<]*</td><td>[^<]*</td><td>[^<]*</td>')

        def rep(m, r=r):
            return ('%s<td>%.2f</td><td>%s</td><td>%.2f</td><td>%.4f</td>'
                    '<td>%+.2f</td><td>%.4f</td><td>%.4f</td>'
                    % (m.group(1), r["RMSE"], m.group(2), r["MAE"], r["Pearson R"],
                       r["MBE"], r["SS vs climatology"], r["R2"]))
        s, n = pat.subn(rep, s)
        if n:
            changed.append("brief Table 3.3 %s" % model)
    if s != orig:
        shutil.copy(BRIEF, BRIEF + ".bak")
        open(BRIEF, "w").write(s)

    # ---- PROJECT_STATUS Table 3.3 ----
    p = open(STATUS).read()
    origp = p
    # the canonical table: | model | RMSE | MAE | R | MBE | SS | R2 |
    for model in ("XGBoost", "Random Forest", "CNN", "U-Net"):
        r = t.loc[model]
        pat = re.compile(
            r"^\|\s*(\*\*)?" + re.escape(model) +
            r"(\*\*)?\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|"
            r"\s*[+\-−]?[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|$", re.M)

        def rep(m, r=r, model=model):
            star = m.group(1) or ""
            return ("| %s%s%s | %.2f | %.2f | %.4f | %+.2f | %.4f | %.4f |"
                    % (star, model, star, r["RMSE"], r["MAE"], r["Pearson R"],
                       r["MBE"], r["SS vs climatology"], r["R2"]))
        p, n = pat.subn(rep, p)
        if n:
            changed.append("status Table 3.3 %s (%d row)" % (model, n))

    # sigma_arch long-term share and the RF-XGB paired difference
    ux = u[(u.model == "xgb") & (u.period == "long_term_2076_2100")].iloc[0]
    p = p.replace("**σ_arch is 68.04%** of long-term variance",
                  "**σ_arch is %.2f%%** of long-term variance" % ux.pct_var_arch)
    p = p.replace("σ_DS at 7.11%.", "σ_DS at %.2f%%." % ux.pct_var_ds)
    rb = b[(b.model_a == "Random Forest") & (b.model_b == "XGBoost")
           & (b.metric == "RMSE")].iloc[0]
    p = re.sub(r"\+1\.058(?= W)", "+%.3f" % rb.plug_in, p)

    if p != origp:
        shutil.copy(STATUS, STATUS + ".bak")
        open(STATUS, "w").write(p)

    print("rewrote:")
    for c in changed:
        print("  " + c)
    if not changed:
        print("  nothing matched; check the table formats by hand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
