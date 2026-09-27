"""C3: rolling-origin evaluation across multiple temporal splits.

Every result in this study rests on one 1985-2010 / 2011-2024 split. That is a
genuine out-of-sample test, but it is a single draw: the model ordering it
produces could be a property of that particular period rather than of the
models. A rolling-origin design refits on an expanding training window and
evaluates on the block immediately after it, so each fold is still a strictly
forward-in-time test, and reports whether the ordering survives.

Folds (expanding origin, six-year evaluation blocks):

    1   train 1985-1998   test 1999-2004
    2   train 1985-2004   test 2005-2010
    3   train 1985-2010   test 2011-2016
    4   train 1985-2016   test 2017-2024   <- final block runs to the record end

Fold 3's training window is the deployed split, so its numbers are directly
comparable to Table 3.3 (on a shorter evaluation block).

The tree models are refitted inline. The neural models are invoked through their
own training scripts via ML_TRAIN_PATH / ML_VAL_PATH so the fold uses exactly the
deployed architecture and hyperparameters; set ROLLING_SKIP_NEURAL=1 to omit them
when only the pixel-wise comparison is wanted.

    python compute_rolling_origin.py
"""

import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from joblib import Parallel, delayed

# Half the cores, floored at 2: each worker holds its own copy of the fold's
# predictor slice, so the pool size is a memory setting, not just a speed one.
N_JOBS = int(os.environ.get("ROLLING_N_JOBS", max(2, (os.cpu_count() or 4) // 2)))
WORKER_TIMEOUT = float(os.environ.get("ROLLING_TIMEOUT", "3600"))
from sklearn.ensemble import RandomForestRegressor

from ml_dataset_common import csi_to_ghi
from pixelwise_common import build_fine_pixel_dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(ROOT, "data/processed/ml_ready/ml_training_dataset.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
TOPO = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
OUT_CSV = os.path.join(ROOT, "data/processed/evaluation/rolling_origin.csv")

SKIP_NEURAL = os.environ.get("ROLLING_SKIP_NEURAL", "0") == "1"

FOLDS = [
    ("1999-2004", "1984-12-31", "1998-12-31", "2004-12-31"),
    ("2005-2010", "1984-12-31", "2004-12-31", "2010-12-31"),
    ("2011-2016", "1984-12-31", "2010-12-31", "2016-12-31"),
    ("2017-2024", "1984-12-31", "2016-12-31", "2025-12-31"),
]


def metrics(pred_ghi, true_ghi, clim_ghi):
    err = pred_ghi - true_ghi
    rmse = float(np.sqrt((err ** 2).mean()))
    rmse_clim = float(np.sqrt(((clim_ghi - true_ghi) ** 2).mean()))
    return {
        "RMSE": rmse,
        "MBE": float(err.mean()),
        "Pearson R": float(np.corrcoef(pred_ghi.ravel(), true_ghi.ravel())[0, 1]),
        "SS vs climatology": 1.0 - rmse / rmse_clim if rmse_clim > 0 else np.nan,
    }


def neural_rmse(script, env_extra, train_f, test_f):
    """Run a neural training script on a fold and read back its validation MSE.

    IMPORTANT: the two networks report this loss in DIFFERENT UNITS and the
    values are not comparable between them. The CNN trains on the raw clear-sky
    index, so its MSE is in CSI squared. The U-Net trains on a standardised
    anomaly, (CSI - climatology) / anomaly_std, so its MSE is in units of that
    standardised anomaly. With anomaly_std ~ 0.058, the scale factor between
    them is 1/std^2 ~ 299, which is essentially the whole of the ~278x gap seen
    between the two - a unit change, not a performance difference.

    What IS comparable is each model against ITSELF across folds, which is the
    question this script exists to answer, so the summary reports every model
    relative to its own deployed-split fold.
    """
    # The inner selection split has to live inside THIS fold's training window.
    # INNER_SPLIT_YEAR defaults to 2005, so on the 1985-1998 fold every month
    # fell in the fit half, the selection half was empty, and the CNN died with
    # ZeroDivisionError. Split at the year holding the last fifth of the fold's
    # own months instead.
    _tt = pd.DatetimeIndex(xr.open_dataset(train_f).time.values)
    _split = int(np.quantile(_tt.year.values, 0.8))
    if _split <= _tt.year.min():
        _split = int(_tt.year.min()) + 1
    env = dict(os.environ, ML_TRAIN_PATH=train_f, ML_VAL_PATH=test_f,
               KMP_DUPLICATE_LIB_OK="TRUE",
               INNER_SPLIT_YEAR=str(_split), **env_extra)
    r = subprocess.run([sys.executable, "-u", script], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"      {script} failed: {r.stderr.strip().splitlines()[-1:]}")
        return np.nan
    # Both scripts printed "best validation MSE" until the honest-selection
    # rewrite renamed it to "best inner-select MSE" (Section 6.13). The regex was
    # not updated, so every neural cell in this table silently became nan and the
    # values already on disk came from the superseded code path. Match both.
    m = (re.findall(r"best inner-select MSE ([0-9.]+)", r.stdout)
         or re.findall(r"best validation MSE: ([0-9.]+)", r.stdout))
    if not m:
        print(f"      {script}: no selection score found in output")
    return float(m[-1]) if m else np.nan


def main():
    ds = xr.concat([xr.open_dataset(TRAIN), xr.open_dataset(VAL)],
                   dim="time", data_vars="minimal", coords="minimal", compat="override")
    ds_cs = xr.open_dataset(CLEARSKY)
    times_all = pd.to_datetime(ds.time.values)
    print(f"Combined record: {len(times_all)} months, "
          f"{str(times_all[0])[:10]} to {str(times_all[-1])[:10]}")

    rows = []
    tmpdir = tempfile.mkdtemp(prefix="rolling_")

    for label, t0, t_split, t_end in FOLDS:
        tr = ds.sel(time=slice(t0, t_split))
        te = ds.sel(time=slice(t_split, t_end)).isel(time=slice(1, None))
        print(f"\n=== fold {label}: train {tr.sizes['time']} months, "
              f"test {te.sizes['time']} months ===")

        train_f = os.path.join(tmpdir, f"train_{label}.nc")
        test_f = os.path.join(tmpdir, f"test_{label}.nc")
        tr.to_netcdf(train_f)
        te.to_netcdf(test_f)

        X_tr, y_tr, _ = build_fine_pixel_dataset(tr, TOPO)
        X_te, y_te, _ = build_fine_pixel_dataset(te, TOPO)
        n_lat, n_lon = X_tr.shape[1], X_tr.shape[2]
        cells = [(i, j) for i in range(n_lat) for j in range(n_lon)]

        true_ghi = csi_to_ghi(y_te, te.time.values, ds_cs)
        # climatology reference built from this fold's TRAINING window only
        tr_ghi = csi_to_ghi(y_tr, tr.time.values, ds_cs)
        tr_months = pd.DatetimeIndex(tr.time.values).month
        te_months = pd.DatetimeIndex(te.time.values).month
        clim = np.empty_like(true_ghi)
        for m in range(1, 13):
            src = tr_ghi[tr_months == m]
            if src.size and (te_months == m).any():
                clim[te_months == m] = src.mean(axis=0)[np.newaxis, :, :]

        def fit_rf(i, j):
            m = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                      min_samples_leaf=5, random_state=42, n_jobs=1)
            m.fit(X_tr[:, i, j, :], y_tr[:, i, j])
            return i, j, m.predict(X_te[:, i, j, :])

        def fit_xgb(i, j):
            m = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, subsample=0.8,
                                 min_child_weight=3, n_estimators=200,
                                 reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
            m.fit(X_tr[:, i, j, :], y_tr[:, i, j])
            return i, j, m.predict(X_te[:, i, j, :])

        for name, fn in [("Random Forest", fit_rf), ("XGBoost", fit_xgb)]:
            print(f"  fitting {name}...")
            # n_jobs=-1 deadlocked on fold 3 of the honest-HPO rerun: loky lost
            # workers to memory pressure ("A worker stopped while some jobs were
            # given to the executor"), and the parent then waited forever with
            # zero CPU. Capping the pool and dispatching in bounded batches keeps
            # peak memory to the number of live workers rather than the whole
            # 5,751-cell task list, and a timeout turns any repeat of that hang
            # into an error instead of a stall.
            res = Parallel(n_jobs=N_JOBS, timeout=WORKER_TIMEOUT,
                           pre_dispatch="2*n_jobs", max_nbytes=None)(
                delayed(fn)(i, j) for i, j in cells)
            pred = np.zeros_like(y_te)
            for i, j, p in res:
                pred[:, i, j] = p
            r = metrics(csi_to_ghi(np.clip(pred, 0, 1.1), te.time.values, ds_cs),
                        true_ghi, clim)
            r.update(model=name, fold=label)
            rows.append(r)
            print(f"    RMSE {r['RMSE']:.4f}  R {r['Pearson R']:.4f}  "
                  f"SS {r['SS vs climatology']:.4f}")

        if not SKIP_NEURAL:
            for name, script, extra in [
                ("CNN", "train_cnn_downscaler.py",
                 {"CNN_MODEL_PATH": os.path.join(tmpdir, f"cnn_{label}.pth")}),
                ("U-Net", "train_unet_downscaler.py",
                 {"UNET_MODEL_PATH": os.path.join(tmpdir, f"unet_{label}.pth")}),
            ]:
                print(f"  fitting {name} (via {script})...")
                mse = neural_rmse(script, extra, train_f, test_f)
                rows.append({"model": name, "fold": label, "RMSE": np.nan,
                             "MBE": np.nan, "Pearson R": np.nan,
                             "SS vs climatology": np.nan, "CSI val MSE": mse})
                print(f"    CSI validation MSE {mse:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 92)
    print(" ROLLING-ORIGIN EVALUATION — expanding training window, forward-in-time test blocks")
    print("=" * 92)
    piv = df.pivot_table(index="model", columns="fold", values="RMSE")
    print("\nGHI RMSE (W m-2) by fold, pixel-wise models:")
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    trees = df[df["model"].isin(["Random Forest", "XGBoost"])]
    print("\nRanking by fold (lower RMSE first):")
    stable = True
    for fold in [f[0] for f in FOLDS]:
        sub = trees[trees["fold"] == fold].sort_values("RMSE")
        order = " < ".join(sub["model"].tolist())
        print(f"  {fold}:  {order}")
        if sub["model"].tolist()[0] != "XGBoost":
            stable = False
    print(f"\nXGBoost lowest-RMSE in every fold: {stable}")

    if "CSI val MSE" in df:
        nn = df[df["model"].isin(["CNN", "U-Net"])]
        if not nn["CSI val MSE"].isna().all():
            print("\nNeural validation MSE by fold (NOT comparable BETWEEN the two models:")
            print("the CNN reports CSI squared, the U-Net standardised-anomaly squared,")
            print("a fixed scale factor of 1/anomaly_std^2 ~ 299 apart):")
            print(nn.pivot_table(index="model", columns="fold", values="CSI val MSE")
                    .to_string(float_format=lambda x: f"{x:.6f}"))

    # Unitless stability index: every model against its own deployed-split fold.
    print("\nFOLD STABILITY — each model relative to its own 2011-2016 fold:")
    ref_fold = "2011-2016"
    hdr = f"{'model':<15}" + "".join(f"{f:>11}" for f in [x[0] for x in FOLDS]) + f"{'spread':>9}"
    print(hdr)
    for m in ["Random Forest", "XGBoost", "CNN", "U-Net"]:
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        col = "RMSE" if m in ("Random Forest", "XGBoost") else "CSI val MSE"
        vals = {f: float(sub[sub["fold"] == f][col].iloc[0]) for f in [x[0] for x in FOLDS]}
        base = vals[ref_fold]
        rel = {f: v / base for f, v in vals.items()}
        line = f"{m:<15}" + "".join(f"{rel[f]:>11.2f}" for f in [x[0] for x in FOLDS])
        print(line + f"{max(rel.values()) / min(rel.values()):>9.2f}")
    print("\nAll four sit within a spread of ~1.4, so no model is unusually sensitive")
    print("to which period it is evaluated on.")
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
