"""Score the dropout-free U-Net against the deployed models on the current target.

Sections 4.5 and 5.6 quote this comparison, but the CSV behind it had no
generator: it was assembled by hand before the clear-sky rebuild, so it set a
variant trained on the superseded target against models trained on the new one.
That is not a comparison, and the deployed U-Net figure it carried (9.95 W/m2)
had already been overtaken by the retrain.

The variant is re-measured by rerun_dropout_variant.sh and scored here from the
same validation fields the rest of Chapter 4 uses, on one basis, with the target
checked for identity across the two files first.

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
DROP0 = os.path.join(EVAL, "validation_fields_unet_drop0.nc")
SEEDED = os.path.join(EVAL, "validation_fields_unet_seed42.nc")
OUT = os.path.join(EVAL, "unet_dropout_variant.csv")

# The labels Chapter 4 and Chapter 5 index this CSV by; they must not drift.
LABELS = {
    "ghi_xgb": "XGBoost (deployed model)",
    "ghi_cnn": "CNN",
    "ghi_rf": "Random Forest",
    "ghi_unet": "U-Net, dropout 0.3 (deployed)",
}
VARIANT = "U-Net, dropout 0 (variant)"
CONTROL = "U-Net, dropout 0.3 (reproduction check)"


def scores(pred, truth, mask):
    """RMSE and mean bias over Zimbabwe, and RMSE over the whole analysis box."""
    err = pred - truth
    zw = np.broadcast_to(mask, err.shape)
    inzw = err[zw & np.isfinite(err)]
    allbox = err[np.isfinite(err)]
    return (float(np.sqrt(np.mean(inzw ** 2))), float(np.mean(inzw)),
            float(np.sqrt(np.mean(allbox ** 2))))


def main():
    main_ds, d0 = xr.open_dataset(MAIN), xr.open_dataset(DROP0)

    truth = main_ds["ghi_true"].values
    truth0 = d0["ghi_true"].values
    # The point of the rerun: both files must describe the same target. If they
    # do not, the variant was trained or scored against a different field and the
    # comparison is void.
    assert truth.shape == truth0.shape, "validation grids differ"
    assert np.allclose(truth, truth0, equal_nan=True), (
        "the two files carry different targets, so the variant is still measured "
        "on a superseded clear-sky basis; rerun rerun_dropout_variant.sh")

    mask = zimbabwe_mask(shape=truth.shape[-2:])

    rows = []
    for var, label in LABELS.items():
        r, b, f = scores(main_ds[var].values, truth, mask)
        rows.append(dict(model=label, RMSE_zw=r, MBE_zw=b, RMSE_fullbox=f))
    r, b, f = scores(d0["ghi_unet"].values, truth0, mask)
    rows.append(dict(model=VARIANT, RMSE_zw=r, MBE_zw=b, RMSE_fullbox=f))

    # Retraining the deployed configuration from the committed script reproduces
    # it bit for bit: identical weights, the same epoch 7 and the same inner MSE
    # of 0.12041. That settles two things at once. The deployed model is seeded
    # despite predating the commit that recorded the seed, so the comparison
    # against the dropout-free variant was always like for like; and the
    # reproducibility Appendix A claims holds for the one model whose training is
    # stochastic. The row is kept as that check, not as a second model.
    control = None
    if os.path.exists(SEEDED):
        sd = xr.open_dataset(SEEDED)
        assert np.allclose(sd["ghi_true"].values, truth, equal_nan=True), (
            "the seeded control was scored against a different target")
        r, b, f = scores(sd["ghi_unet"].values, truth, mask)
        rows.append(dict(model=CONTROL, RMSE_zw=r, MBE_zw=b, RMSE_fullbox=f))
        control = r

    df = pd.DataFrame(rows).sort_values("RMSE_zw").reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print("Validation-period error on the current target (W/m2)\n")
    print("%-32s %9s %9s %9s" % ("model", "RMSE_zw", "MBE_zw", "RMSE_box"))
    for _, w in df.iterrows():
        print("%-32s %9.3f %9.3f %9.3f"
              % (w["model"], w["RMSE_zw"], w["MBE_zw"], w["RMSE_fullbox"]))

    dep = float(df[df.model == LABELS["ghi_unet"]].RMSE_zw.iloc[0])
    var = float(df[df.model == VARIANT].RMSE_zw.iloc[0])
    print("\nRemoving dropout %s the U-Net over Zimbabwe: %.3f against %.3f W/m2, "
          "a difference of %.3f."
          % ("improves" if var < dep else "worsens", var, dep, abs(var - dep)))
    if control is not None:
        drift = abs(control - dep)
        print("Retraining the deployed configuration reproduces it to %.4f W/m2, "
              "so the difference above is the dropout setting and not the draw."
              % drift)
        assert drift < 1e-6, (
            "the deployed configuration did not reproduce (%.4f W/m2 apart), so the "
            "dropout comparison is confounded with the training draw" % drift)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
