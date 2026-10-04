"""Score the dropout setting of the deployed U-Net against the alternative.

The roles here have swapped once. The comparison originally set a dropout-free
variant against a deployed U-Net carrying dropout 0.3, and found the variant
better by 3.03 W/m2 on a clean like-for-like basis. The dropout-free
configuration is now the deployed one, so what this script measures is what
dropout 0.3 costs: the 0.3 configuration is the variant, and its fields come
from the run that reproduced the former deployed model bit for bit.

Scored from the same validation fields the rest of Chapter 4 uses, on one basis,
with the target checked for identity across the files first.

    python compute_dropout_variant_comparison.py
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
MAIN = os.path.join(EVAL, "validation_spatial_fields.nc")
# The deployed U-Net is dropout-free, so its fields are the main ones. The
# dropout 0.3 configuration is the variant, held in the reproduction run.
DROP03 = os.path.join(EVAL, "validation_fields_unet_seed42.nc")
OUT = os.path.join(EVAL, "unet_dropout_variant.csv")

# The labels Chapter 4 and Chapter 5 index this CSV by; they must not drift.
# "deployed" is reserved for XGBoost, which drives the product; the U-Net's
# dropout setting is "adopted", so one table does not carry two deployed models.
LABELS = {
    "ghi_xgb": "XGBoost (deployed model)",
    "ghi_cnn": "CNN",
    "ghi_rf": "Random Forest",
    "ghi_unet": "U-Net, dropout 0 (adopted)",
}
VARIANT = "U-Net, dropout 0.3 (variant)"


def scores(pred, truth, mask):
    """RMSE and mean bias over Zimbabwe, and RMSE over the whole analysis box."""
    err = pred - truth
    zw = np.broadcast_to(mask, err.shape)
    inzw = err[zw & np.isfinite(err)]
    allbox = err[np.isfinite(err)]
    return (float(np.sqrt(np.mean(inzw ** 2))), float(np.mean(inzw)),
            float(np.sqrt(np.mean(allbox ** 2))))


def main():
    main_ds = xr.open_dataset(MAIN)
    var_ds = xr.open_dataset(DROP03)

    truth = main_ds["ghi_true"].values
    assert truth.shape == var_ds["ghi_true"].values.shape, "validation grids differ"
    assert np.allclose(truth, var_ds["ghi_true"].values, equal_nan=True), (
        "the two files carry different targets, so the comparison is void")

    mask = zimbabwe_mask(shape=truth.shape[-2:])

    rows = []
    for var, label in LABELS.items():
        r, b, f = scores(main_ds[var].values, truth, mask)
        rows.append(dict(model=label, RMSE_zw=r, MBE_zw=b, RMSE_fullbox=f))
    r, b, f = scores(var_ds["ghi_unet"].values, truth, mask)
    rows.append(dict(model=VARIANT, RMSE_zw=r, MBE_zw=b, RMSE_fullbox=f))

    df = pd.DataFrame(rows).sort_values("RMSE_zw").reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print("Validation-period error on the current target (W/m2)\n")
    print("%-32s %9s %9s %9s" % ("model", "RMSE_zw", "MBE_zw", "RMSE_box"))
    for _, w in df.iterrows():
        print("%-32s %9.3f %9.3f %9.3f"
              % (w["model"], w["RMSE_zw"], w["MBE_zw"], w["RMSE_fullbox"]))

    dep = float(df[df.model == LABELS["ghi_unet"]].RMSE_zw.iloc[0])
    var = float(df[df.model == VARIANT].RMSE_zw.iloc[0])
    print("\nDropout 0.3 %s the U-Net over Zimbabwe: %.3f against the adopted "
          "%.3f W/m2, a difference of %.3f."
          % ("costs" if var > dep else "gains", var, dep, abs(var - dep)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
