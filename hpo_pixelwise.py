"""Coarse grid search via 5-fold temporal cross-validation for the pixel-wise
RF and XGBoost models (Section 3.6.7, Table 3.2).

Running the full grid independently for all 5,751 output cells would take
many hours per model. Instead, hyperparameters are selected on a
representative random sample of cells (domain-mean CV RMSE), then applied
uniformly when the full model is retrained - consistent with how these
models already share one fixed hyperparameter set across all cells.

The deployed configuration is evaluated as an extra point on the same criterion.
For XGBoost it lies outside the grid - Section 3.6.4 deploys eta=0.05,
subsample=0.8 and min_child_weight=3, none of which the grid contains - so
without it the search cannot say whether the configuration actually in use is
better or worse than the one the search would pick. Results are written to CSV
so the comparison is in the archive rather than only in a console log.
"""

import os
import sys
import time
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from pixelwise_common import build_fine_pixel_dataset

MODEL = sys.argv[1] if len(sys.argv) > 1 else "rf"
assert MODEL in ("rf", "xgb")

OUT = os.path.join("data/processed/evaluation", "hpo_pixelwise_%s.csv" % MODEL)
N_JOBS = int(os.environ.get("HPO_N_JOBS", "-1"))

# What Section 3.6.3 and Section 3.6.4 actually deploy.
DEPLOYED = {
    "rf": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 5},
    "xgb": {"max_depth": 6, "eta": 0.05, "subsample": 0.8, "min_child_weight": 3},
}[MODEL]

N_SUBSET_CELLS = 150
N_FOLDS = 5
SEED = 42

train_path = "./data/processed/ml_ready/ml_training_dataset.nc"
topo_path = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")

print(f"Loading training data for {MODEL.upper()} hyperparameter search...")
ds_train = xr.open_dataset(train_path)
X_train, y_train, feature_vars = build_fine_pixel_dataset(ds_train, topo_path)
n_time, n_lat, n_lon, n_features = X_train.shape
print(f"Full grid: {n_lat} x {n_lon} = {n_lat * n_lon} cells, {n_time} training months")

# --- 5-fold temporal CV: 26 years (1985-2010) split into 5 ~5-year blocks
#     (Section 3.6.7: "each fold comprising one 5-year block") ---
fold_bounds = np.linspace(0, n_time, N_FOLDS + 1).astype(int)
folds = [(fold_bounds[k], fold_bounds[k + 1]) for k in range(N_FOLDS)]
print(f"Temporal folds (month index ranges): {folds}")

rng = np.random.RandomState(SEED)
flat_cells = [(i, j) for i in range(n_lat) for j in range(n_lon)]
subset_cells = [flat_cells[k] for k in rng.choice(len(flat_cells), size=N_SUBSET_CELLS, replace=False)]
print(f"Evaluating on a random subset of {N_SUBSET_CELLS}/{len(flat_cells)} cells.")


def cv_rmse_rf(i, j, n_estimators, max_features, min_samples_leaf):
    x, y = X_train[:, i, j, :], y_train[:, i, j]
    errs = []
    for start, end in folds:
        test_idx = np.arange(start, end)
        train_idx = np.setdiff1d(np.arange(n_time), test_idx)
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_features=max_features,
            min_samples_leaf=min_samples_leaf, random_state=SEED, n_jobs=1,
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        errs.append(mean_squared_error(y[test_idx], pred))
    return np.sqrt(np.mean(errs))


def cv_rmse_xgb(i, j, max_depth, eta, subsample, min_child_weight):
    x, y = X_train[:, i, j, :], y_train[:, i, j]
    errs = []
    for start, end in folds:
        test_idx = np.arange(start, end)
        train_idx = np.setdiff1d(np.arange(n_time), test_idx)
        # reg_alpha and reg_lambda are the deployed values from
        # train_pixelwise_xgb.py. Without them the search was scoring a
        # different estimator from the one deployed, which made the "deployed
        # configuration ranks Nth" comparison incoherent.
        model = xgb.XGBRegressor(
            max_depth=max_depth, learning_rate=eta, subsample=subsample,
            min_child_weight=min_child_weight, n_estimators=200,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=1,
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        errs.append(mean_squared_error(y[test_idx], pred))
    return np.sqrt(np.mean(errs))


if MODEL == "rf":
    grid = [
        {"n_estimators": n, "max_features": mf, "min_samples_leaf": msl}
        for n in [300, 500, 800]
        for mf in ["sqrt", "log2"]
        for msl in [3, 5, 10]
    ]
else:
    grid = [
        {"max_depth": d, "eta": e, "subsample": s, "min_child_weight": mcw}
        for d in [4, 6, 8]
        for e in [0.03, 0.1]
        for s in [0.7, 1.0]
        for mcw in [1, 5]
    ]

if DEPLOYED not in grid:
    grid.append(DEPLOYED)
    print(f"Deployed configuration {DEPLOYED} is not in the grid; appended as an extra point.")
else:
    print(f"Deployed configuration {DEPLOYED} is in the grid.")

print(f"\nGrid size: {len(grid)} combinations x {N_FOLDS} folds x {N_SUBSET_CELLS} cells "
      f"= {len(grid) * N_FOLDS * N_SUBSET_CELLS} total fits")

start_time = time.time()
results = []
cv_fn = cv_rmse_rf if MODEL == "rf" else cv_rmse_xgb
for combo_idx, params in enumerate(grid):
    cell_rmses = Parallel(n_jobs=N_JOBS)(delayed(cv_fn)(i, j, **params) for i, j in subset_cells)
    mean_rmse = float(np.mean(cell_rmses))
    results.append((params, mean_rmse))
    print(f"  [{combo_idx + 1}/{len(grid)}] {params} -> mean CV RMSE (CSI units): {mean_rmse:.5f} "
          f"({time.time() - start_time:.0f}s elapsed)")

results.sort(key=lambda r: r[1])
print("\n" + "=" * 70)
print(f" BEST {MODEL.upper()} HYPERPARAMETERS (Section 3.6.7)")
print("=" * 70)
for params, rmse in results[:5]:
    print(f"  {params} -> CV RMSE: {rmse:.5f}")

best_params, best_rmse = results[0]
print(f"\nSelected: {best_params}")

import pandas as pd

rows = []
for params, rmse in results:
    rows.append(dict(params, cv_rmse_csi=rmse, is_deployed=(params == DEPLOYED),
                     in_grid=(params != DEPLOYED or DEPLOYED in grid[:-1])))
df = pd.DataFrame(rows).sort_values("cv_rmse_csi").reset_index(drop=True)
df["rank"] = df.index + 1
df.to_csv(OUT, index=False)

dep = df[df.is_deployed].iloc[0]
print("\nDeployed configuration ranks %d of %d on the search criterion: "
      "CV RMSE %.5f against the best %.5f, a difference of %.5f CSI units (%.2f per cent)."
      % (int(dep["rank"]), len(df), dep["cv_rmse_csi"], best_rmse,
         dep["cv_rmse_csi"] - best_rmse,
         100.0 * (dep["cv_rmse_csi"] - best_rmse) / best_rmse))
print("wrote %s" % OUT)
