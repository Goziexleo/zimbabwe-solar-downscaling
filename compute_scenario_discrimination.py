"""Section 4.4's scenario-discrimination screen, written to disk.

The screen asks the one question the 2011-2024 validation cannot: can the model
tell one emission pathway from another? The separation between the SSP5-8.5 and
SSP2-4.5 ensemble means should be positive and should grow with lead time.

This existed only as twelve literals typed into Table 4.5 and repeated in
PROJECT_STATUS, the viva brief and the interview deck. The Random Forest and
XGBoost figures were right, because neither was ever retrained. Every CNN and
U-Net figure was wrong: the honest retrain regenerated their projections and
nothing regenerated the table. The errors were small - +9.643 against a true
+9.478, and a CNN ordering score of 100.0% that is really 99.9% - but the table
is the evidence for a deployment decision, and a hand-typed number cannot be
re-derived by a reader.

    python compute_scenario_discrimination.py

Writes data/processed/evaluation/scenario_discrimination.csv.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data/processed")
OUT = os.path.join(PROC, "evaluation/scenario_discrimination.csv")

# The bare directory is the U-Net's: it was the deployed model when the
# aggregation script was written and never took a suffix.
MME = {
    "Random Forest": "mme_aggregations_rf",
    "XGBoost": "mme_aggregations_xgb",
    "CNN": "mme_aggregations_cnn",
    "U-Net": "mme_aggregations",
}

HORIZONS = [
    ("near_term", "near_term_2026_2050"),
    ("mid_term", "mid_term_2051_2075"),
    ("long_term", "long_term_2076_2100"),
]


def field(model_dir, ssp, tag):
    path = os.path.join(PROC, model_dir, "mme_suitability_%s_%s.nc" % (ssp, tag))
    return xr.open_dataset(path).ghi_mean.values


def main():
    rows = []
    for model, d in MME.items():
        row = {"model": model}
        for short, tag in HORIZONS:
            diff = field(d, "ssp585", tag) - field(d, "ssp245", tag)
            row["sep_" + short] = float(np.nanmean(diff))
            # Share of cells placing the higher pathway above the lower. Taken at
            # each horizon rather than only the last, so a model that orders well
            # early and badly late cannot hide behind one number.
            row["pct_ordered_" + short] = float(100 * np.nanmean(diff > 0))
        # "Grows" is the screen's actual criterion, evaluated rather than asserted.
        seps = [row["sep_" + s] for s, _ in HORIZONS]
        row["grows"] = bool(seps[-1] > seps[0])
        row["monotonic"] = bool(seps[0] <= seps[1] <= seps[2])
        row["passes_screen"] = bool(all(v > 0 for v in seps) and row["grows"])
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("scenario-discrimination screen, SSP5-8.5 minus SSP2-4.5 (W m-2)\n")
    print("%-14s %8s %8s %8s  %-6s %-10s %s"
          % ("model", "near", "mid", "long", "grows", "monotonic", "cells ordered (long)"))
    for r in rows:
        print("%-14s %+8.3f %+8.3f %+8.3f  %-6s %-10s %.1f%%"
              % (r["model"], r["sep_near_term"], r["sep_mid_term"], r["sep_long_term"],
                 "yes" if r["grows"] else "NO", "yes" if r["monotonic"] else "no",
                 r["pct_ordered_long_term"]))
    print("\nSaved %s" % OUT)


if __name__ == "__main__":
    main()
