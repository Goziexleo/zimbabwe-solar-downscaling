"""Per-predictor importance for the two pixel-wise models (Sections 3.6.3, 3.6.4).

Random Forest: mean decrease in impurity, read from the persisted per-cell
forests, plus permutation importance on a random subsample of cells (permuting
every cell is prohibitively slow and adds nothing to the ranking).

XGBoost: gain and cover, from models refitted on the fly since they are not
persisted.

Importances are averaged across cells, so the output answers "which predictors
drive the fit across Zimbabwe", not "which drive it at one location".
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance

from pixelwise_common import build_fine_pixel_dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(ROOT, "data/processed/ml_ready/ml_training_dataset.nc")
VAL_PATH = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
TOPO_PATH = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
RF_DIR = os.path.join(ROOT, "data/processed/models/pixelwise_rf")
OUT_CSV = os.path.join(ROOT, "data/processed/evaluation/feature_importance.csv")

N_PERM_CELLS = int(os.environ.get("N_PERM_CELLS", "200"))
SEED = 42


def main():
    ds_train = xr.open_dataset(TRAIN_PATH)
    ds_val = xr.open_dataset(VAL_PATH)
    X_train, y_train, feature_vars = build_fine_pixel_dataset(ds_train, TOPO_PATH)
    X_val, y_val, _ = build_fine_pixel_dataset(ds_val, TOPO_PATH)
    n_lat, n_lon = X_train.shape[1], X_train.shape[2]
    print(f"{len(feature_vars)} features across {n_lat * n_lon} cells: {feature_vars}")

    rng = np.random.default_rng(SEED)
    cells = [(i, j) for i in range(n_lat) for j in range(n_lon)]
    subset = [cells[k] for k in rng.choice(len(cells), size=N_PERM_CELLS, replace=False)]

    # ---------------------------------------------------------- RF: MDI ----
    with open(os.path.join(RF_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["feature_vars"] == feature_vars, "RF manifest feature order differs"

    print(f"Reading mean decrease in impurity from {len(cells)} persisted forests...")

    def mdi(i, j):
        p = os.path.join(RF_DIR, f"rf_cell_{i:03d}_{j:03d}.joblib")
        if not os.path.exists(p):
            return None
        return joblib.load(p).feature_importances_

    mdi_vals = Parallel(n_jobs=-1, verbose=1)(delayed(mdi)(i, j) for i, j in cells)
    mdi_arr = np.array([v for v in mdi_vals if v is not None])
    rf_mdi = mdi_arr.mean(axis=0)

    # ------------------------------------------- RF: permutation importance --
    print(f"Permutation importance on {N_PERM_CELLS} random cells...")

    def perm(i, j):
        p = os.path.join(RF_DIR, f"rf_cell_{i:03d}_{j:03d}.joblib")
        if not os.path.exists(p):
            return None
        r = permutation_importance(
            joblib.load(p), X_val[:, i, j, :], y_val[:, i, j],
            n_repeats=5, random_state=SEED, n_jobs=1,
        )
        return r.importances_mean

    perm_vals = Parallel(n_jobs=-1, verbose=1)(delayed(perm)(i, j) for i, j in subset)
    rf_perm = np.array([v for v in perm_vals if v is not None]).mean(axis=0)

    # ------------------------------------------------ XGBoost: gain, cover --
    print(f"XGBoost gain/cover on {N_PERM_CELLS} random cells (models are not persisted)...")

    def xgb_imp(i, j):
        m = xgb.XGBRegressor(
            max_depth=6, learning_rate=0.05, subsample=0.8, min_child_weight=3,
            n_estimators=200, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=1,
        )
        m.fit(X_train[:, i, j, :], y_train[:, i, j])
        b = m.get_booster()
        out = []
        for kind in ("gain", "cover"):
            d = b.get_score(importance_type=kind)
            out.append(np.array([d.get(f"f{k}", 0.0) for k in range(len(feature_vars))]))
        return out

    xgb_vals = Parallel(n_jobs=-1, verbose=1)(delayed(xgb_imp)(i, j) for i, j in subset)
    xgb_gain = np.array([v[0] for v in xgb_vals]).mean(axis=0)
    xgb_cover = np.array([v[1] for v in xgb_vals]).mean(axis=0)

    def norm(a):
        return a / a.sum() if a.sum() > 0 else a

    df = pd.DataFrame({
        "feature": feature_vars,
        "rf_mdi": norm(rf_mdi),
        "rf_permutation": rf_perm,
        "xgb_gain": norm(xgb_gain),
        "xgb_cover": norm(xgb_cover),
    }).sort_values("rf_mdi", ascending=False)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 88)
    print(" PER-PREDICTOR IMPORTANCE (averaged across cells; MDI/gain/cover normalised)")
    print("=" * 88)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("=" * 88)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
