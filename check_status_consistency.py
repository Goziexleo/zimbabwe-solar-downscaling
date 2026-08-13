"""Guard against stale numbers in PROJECT_STATUS.md.

Four audit rounds have now found the same failure: a number is corrected in a
table and the sentence underneath keeps the old value. Round two it was the
sigma_arch shares, round three the ablation skill column and the delta-mapping
reference, round four the decomposition percentages, the member count and
XGBoost's mean bias. Every instance was caught by an external reader.

The tables already have a defence - compute_table33.py is the single canonical
source, so figures cannot drift between scripts. The prose has none. This is it.

Two checks, deliberately different in character:

  ANCHORS   values that must be present and correct. Read from the canonical
            CSVs at the precision the document uses, so a rerun that changes a
            metric fails here until the prose is updated to match.

  RETIRED   values known to be superseded, which must NOT reappear. This is the
            part that catches the actual failure mode: an old number surviving
            in a sentence beneath a corrected table. Add to this list whenever a
            figure is replaced.

Run directly for a report, or via pytest as part of the suite:

    python check_status_consistency.py
"""

import os
import re
import sys

import pandas as pd

# How close an explanatory phrase must sit to a retired value for the mention to
# count as "discussing the correction" rather than "still asserting the number".
EXPLANATION_WINDOW = 90
EXPLANATORY = (r"was |were |before |leak|stale|superseded|earlier|previously|retired|"
               r"instead of|rather than|no longer (?:correct|valid|used|reported)|"
               r"had been|used to|old |former|corrected|revised|replaced|"
               r"changed from|moved from|->|→")

ROOT = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(ROOT, "PROJECT_STATUS.md")
EVAL = os.path.join(ROOT, "data/processed/evaluation")


def _fmt(x, dp):
    return f"{x:.{dp}f}"


def canonical_anchors():
    """(label, string that must appear in the document) from the canonical CSVs."""
    anchors = []

    t33 = os.path.join(EVAL, "table_3_3.csv")
    if os.path.exists(t33):
        df = pd.read_csv(t33)
        for _, r in df.iterrows():
            m = r["model"]
            anchors.append((f"Table 3.3 {m} RMSE", _fmt(r["RMSE"], 2)))
            anchors.append((f"Table 3.3 {m} R", _fmt(r["Pearson R"], 4)))

    unc = os.path.join(EVAL, "uncertainty_decomposition_summary.csv")
    if os.path.exists(unc):
        df = pd.read_csv(unc)
        rf_long = df[(df["model"] == "rf") & (df["period"] == "long_term_2076_2100")]
        if not rf_long.empty:
            r = rf_long.iloc[0]
            anchors.append(("sigma_arch long-term variance share",
                            _fmt(r["pct_var_arch"], 2)))
            anchors.append(("sigma_DS long-term variance share",
                            _fmt(r["pct_var_ds"], 2)))
            anchors.append(("n_arch_members", f"n = {int(r['n_arch_members'])}"))

    boot = os.path.join(EVAL, "bootstrap_differences.csv")
    if os.path.exists(boot):
        df = pd.read_csv(boot)
        key = df[(df["metric"] == "RMSE")
                 & (df[["model_a", "model_b"]].apply(
                     lambda x: {x.iloc[0], x.iloc[1]} == {"Random Forest", "XGBoost"}, axis=1))]
        if not key.empty:
            anchors.append(("RF-XGB bootstrap difference",
                            _fmt(abs(key.iloc[0]["mean_difference"]), 3)))
    return anchors


# Values replaced by a correction. Each must never reappear in the document.
# Format: (superseded value as written, what replaced it and why)
RETIRED = [
    ("0.6922", "ablation config B skill, computed against the leaked 17.98 climatology"),
    ("0.4932", "ablation config C skill, same leaked reference"),
    ("17.98", "climatology reference built from the validation period (A7 leakage)"),
    ("ref RMSE 0.131", "delta-mapping reference before the A7 fix"),
    ("14.2% vs 2.0%", "sigma_arch share at n = 2"),
    ("72–95%", "sigma_DS share range at n = 2"),
    ("n = 2 architectures", "sigma_arch member count before CNN and XGBoost were added"),
    ("9.11", "XGBoost RMSE while early-stopped on the validation set"),
    ("+1.26", "XGBoost mean bias before the early-stopping fix"),
    ("~14 GB", "RF model footprint estimated from the daily experiment"),
    ("6,375", "grid cell count before the fine grid was aligned to coarse coverage"),
    ("75 latitude", "grid dimensions before the alignment fix"),
]


def check():
    text = open(STATUS).read()
    # ignore fenced code blocks: paths and commands legitimately contain digits
    prose = re.sub(r"```.*?```", "", text, flags=re.S)

    missing, resurrected = [], []

    for label, value in canonical_anchors():
        if value not in prose:
            missing.append((label, value))

    for value, why in RETIRED:
        for m in re.finditer(re.escape(value), prose):
            line_no = prose[:m.start()].count("\n") + 1
            line = prose.splitlines()[line_no - 1]
            # A retired value may legitimately appear where the document is
            # explaining that it was superseded. Look for that marker only in a
            # WINDOW AROUND THE VALUE, not anywhere on the line: these are long
            # paragraphs, and a line-wide search let a genuinely stale figure
            # through because an unrelated clause elsewhere in the same
            # paragraph happened to contain "no longer".
            lo = max(0, m.start() - EXPLANATION_WINDOW)
            hi = min(len(prose), m.end() + EXPLANATION_WINDOW)
            if re.search(EXPLANATORY, prose[lo:hi], re.I):
                continue
            resurrected.append((value, why, line_no, line.strip()[:90]))

    return missing, resurrected


def main():
    missing, resurrected = check()

    print("=" * 84)
    print(" PROJECT_STATUS.md consistency check")
    print("=" * 84)

    if missing:
        print(f"\nCANONICAL VALUES ABSENT FROM THE DOCUMENT ({len(missing)}):")
        for label, value in missing:
            print(f"  {label:<44} expected to find '{value}'")
    else:
        print("\nAll canonical values from the evaluation CSVs appear in the document.")

    if resurrected:
        print(f"\nSUPERSEDED VALUES PRESENT ({len(resurrected)}):")
        for value, why, line_no, line in resurrected:
            print(f"  line {line_no}: '{value}' — {why}")
            print(f"      {line}")
    else:
        print("No superseded value appears outside an explanatory context.")

    print()
    return 1 if (missing or resurrected) else 0


if __name__ == "__main__":
    sys.exit(main())
