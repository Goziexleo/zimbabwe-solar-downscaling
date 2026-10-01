"""Canonical Table 3.3 for all four models, from the saved validation fields.

Single source of truth. Every model is scored against identical references on
identical data, so the table cannot drift between scripts, and no retraining is
needed to regenerate it.

Two of the three references are CIRCULARITY DIAGNOSTICS, not skill references
(Section 3.7.3):

  interpolation   bilinear interpolation of the coarse rsds field. rsds is a
                  reuse of the same ssrd the target is derived from, so this is
                  very nearly an identity (RMSE ~0.23 against a target standard
                  deviation near 19). A skill score against an identity measures
                  the circularity of the task, not the quality of a model.
  delta-mapping   the same field rescaled to the training-period climatological
                  mean. Still built from rsds, so still a diagnostic.

  climatology     per-cell, per-calendar-month mean of the TRAINING record
                  (1985-2010) applied to the validation period. This is the only
                  admissible skill reference: it is constructible without seeing
                  the evaluation period and is independent of the target's
                  source field.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from ml_dataset_common import evaluation_baselines, skill_score
from zimbabwe_mask import describe as mask_describe
from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
FIELDS = os.path.join(ROOT, "data/processed/evaluation/validation_spatial_fields.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
OUT_CSV = os.path.join(ROOT, "data/processed/evaluation/table_3_3.csv")

MODELS = [
    ("Random Forest", "ghi_rf"),
    ("XGBoost", "ghi_xgb"),
    ("CNN", "ghi_cnn"),
    ("U-Net", "ghi_unet"),
]


def metrics(pred, truth, clim_rmse, mask=None):
    """Metrics over the full analysis box, or over Zimbabwe when a mask is given.

    42.8 per cent of the box lies in Zambia, Botswana, Mozambique and South
    Africa, so an unmasked figure is not a result about Zimbabwe however it is
    labelled. Both are computed; the masked ones carry a _zw suffix.
    """
    if mask is not None:
        pred, truth = pred[:, mask], truth[:, mask]
    err = pred - truth
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    mbe = float(err.mean())
    r = float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1])
    ss_res = float((err ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return {
        "RMSE": rmse, "MAE": mae, "Pearson R": r, "MBE": mbe,
        "R2": r2, "SS vs climatology": skill_score(rmse, clim_rmse),
    }


def main():
    ds_f = xr.open_dataset(FIELDS)
    ds_val = xr.open_dataset(VAL)
    ds_cs = xr.open_dataset(CLEARSKY)
    truth = ds_f["ghi_true"].values
    times = ds_val.time.values

    interp, clim, delta = evaluation_baselines(ds_val, truth, times, ds_cs)
    rmse_of = lambda a: float(np.sqrt(((a - truth) ** 2).mean()))
    clim_rmse, interp_rmse, delta_rmse = rmse_of(clim), rmse_of(interp), rmse_of(delta)

    mask = zimbabwe_mask(truth.shape)
    clim_rmse_zw = float(np.sqrt(((clim[:, mask] - truth[:, mask]) ** 2).mean()))

    rows = []
    for label, var in MODELS:
        pred = ds_f[var].values
        m = metrics(pred, truth, clim_rmse)
        m.update({k + "_zw": v for k, v in
                  metrics(pred, truth, clim_rmse_zw, mask=mask).items()})
        m["model"] = label
        rows.append(m)

    cols = ["RMSE", "MAE", "Pearson R", "MBE", "SS vs climatology", "R2"]
    df = pd.DataFrame(rows)
    # The skill reference belongs in the file, not in a chapter as a literal.
    # Chapter 4 previously carried "19.08 W m-2" typed into the caption.
    df["climatology_rmse"] = clim_rmse
    df["climatology_rmse_zw"] = clim_rmse_zw
    df = df[["model"] + cols + [c + "_zw" for c in cols]
            + ["climatology_rmse", "climatology_rmse_zw"]].sort_values("RMSE_zw")
    df.to_csv(OUT_CSV, index=False)

    print("=" * 92)
    print(" TABLE 3.3 - validation 2011-2024 (W m-2)")
    print(" mask: " + mask_describe())
    print("=" * 92)
    print(" full analysis box:")
    print(df[["model"] + cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n over Zimbabwe (the figures the prose should quote):")
    print(df[["model"] + [c + "_zw" for c in cols]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n change from masking:")
    for r in df.itertuples():
        print("   %-14s RMSE %7.4f -> %7.4f (%+.4f)   MBE %+7.4f -> %+7.4f"
              % (r.model, r.RMSE, r.RMSE_zw, r.RMSE_zw - r.RMSE, r.MBE, r.MBE_zw))
    print("=" * 92)
    print(f"\nSkill reference (admissible):")
    print(f"  climatology, 1985-2010 training record  RMSE {clim_rmse:8.4f}")
    print(f"\nCircularity diagnostics (NOT skill references — both built from rsds,")
    print(f"a reuse of the target's own source field):")
    print(f"  bilinear interpolation of rsds          RMSE {interp_rmse:8.4f}")
    print(f"  delta-mapped interpolation              RMSE {delta_rmse:8.4f}")
    for label, var in MODELS:
        p = ds_f[var].values
        print(f"    {label:<14} ratio to interpolation {rmse_of(p) / interp_rmse:7.1f}x")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
