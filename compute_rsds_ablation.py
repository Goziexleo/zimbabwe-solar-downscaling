"""Re-measure Section 3.7.4's rsds ablation under the deployed configuration.

Table 3.4 separates two effects that an earlier draft had conflated: fixing the
one-month predictor misalignment, and dropping rsds from the predictor set. Its
configurations were one-off refits that no script re-executed, so when the
pixel-wise hyperparameters changed in Section 3.6.7 the table went stale and a
limitations sentence ended up comparing 5.54 W/m2 from the old configuration
against 8.88 from the new one, on two different spatial bases.

This makes the reproducible part reproducible. Configuration C is the deployed
model and B is the same model with rsds restored, both at the current
hyperparameters, both scored on the full analysis box and over Zimbabwe so the
basis is never in doubt. Configuration A is not reproducible: it required the
one-month lag, and that was a defect in the feature builder rather than a switch,
so it is reported as a historical measurement and labelled as one.

MODEL_PREDICTOR_VARS is read at import, so each configuration needs its own
process. Run it twice:

    MODEL_PREDICTORS="clt,tas,ps,huss,od550aer"       python compute_rsds_ablation.py C
    MODEL_PREDICTORS="clt,tas,ps,huss,rsds,od550aer"  python compute_rsds_ablation.py B
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
import xgboost as xgb

from ml_dataset_common import csi_to_ghi, MODEL_PREDICTOR_VARS
from pixelwise_common import build_fine_pixel_dataset
from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
OUT = os.path.join(EVAL, "rsds_ablation.csv")
TRAIN = os.path.join(ROOT, "data/processed/ml_ready/ml_training_dataset.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
TOPO = os.path.join(ROOT, "data/processed/topography",
                    "zimbabwe_topographic_features_0.1deg.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid",
                        "clearsky_ghi_finegrid_climatology.nc")
SEED = 42
N_JOBS = int(os.environ.get("ABL_N_JOBS", "6"))

LABEL = {"B": "B. Lag fixed, rsds kept", "C": "C. Lag fixed, rsds dropped (deployed)"}


def default_of(name):
    src = open(os.path.join(ROOT, "train_pixelwise_xgb.py")).read()
    m = re.search(r'os\.environ\.get\(\s*"%s"\s*,\s*"([^"]+)"' % name, src)
    assert m is not None, name
    return m.group(1)


def literal_of(name):
    src = open(os.path.join(ROOT, "train_pixelwise_xgb.py")).read()
    m = re.search(r"%s\s*=\s*([0-9.]+)" % name, src)
    assert m is not None, name
    return float(m.group(1))


CFG = dict(max_depth=int(default_of("XGB_MAX_DEPTH")),
           learning_rate=float(default_of("XGB_ETA")),
           subsample=float(default_of("XGB_SUBSAMPLE")),
           min_child_weight=int(default_of("XGB_MIN_CHILD_WEIGHT")),
           n_estimators=int(default_of("XGB_N_ESTIMATORS")),
           reg_alpha=literal_of("reg_alpha"), reg_lambda=literal_of("reg_lambda"))


def fit_cell(xtr, ytr, xva):
    m = xgb.XGBRegressor(random_state=SEED, n_jobs=1, **CFG)
    m.fit(xtr, ytr)
    return m.predict(xva)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "C"
    assert which in LABEL, "configuration must be B or C"
    want_rsds = which == "B"
    assert ("rsds" in MODEL_PREDICTOR_VARS) == want_rsds, (
        "configuration %s needs MODEL_PREDICTORS %s rsds; it currently reads %s"
        % (which, "with" if want_rsds else "without", MODEL_PREDICTOR_VARS))

    dtr, dva = xr.open_dataset(TRAIN), xr.open_dataset(VAL)
    cs = xr.open_dataset(CLEARSKY)
    Xtr, ytr, feats = build_fine_pixel_dataset(dtr, TOPO)
    Xva, yva, _ = build_fine_pixel_dataset(dva, TOPO)
    nt, nlat, nlon, _ = Xva.shape
    print("%s: %d features %s" % (which, len(feats), feats))

    cells = [(i, j) for i in range(nlat) for j in range(nlon)]
    preds = Parallel(n_jobs=N_JOBS)(
        delayed(fit_cell)(Xtr[:, i, j, :], ytr[:, i, j], Xva[:, i, j, :])
        for i, j in cells)
    csi = np.zeros((nt, nlat, nlon), dtype=np.float32)
    for (i, j), p in zip(cells, preds):
        csi[:, i, j] = p

    times = dva.time.values
    ghi, truth = csi_to_ghi(csi, times, cs), csi_to_ghi(yva, times, cs)
    err = ghi - truth
    mask = zimbabwe_mask(shape=(nlat, nlon))
    zw = np.broadcast_to(mask, err.shape)
    fin = np.isfinite(err)
    row = dict(configuration=LABEL[which], predictors=",".join(MODEL_PREDICTOR_VARS),
               n_features=len(feats),
               RMSE_fullbox=float(np.sqrt(np.mean(err[fin] ** 2))),
               RMSE_zw=float(np.sqrt(np.mean(err[zw & fin] ** 2))),
               pearson_r=float(np.corrcoef(ghi[fin].ravel(), truth[fin].ravel())[0, 1]),
               config=json.dumps(CFG, sort_keys=True))

    df = pd.DataFrame([row])
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        df = pd.concat([old[old.configuration != row["configuration"]], df],
                       ignore_index=True)
    df = df.sort_values("configuration").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print("  full box %.4f | Zimbabwe %.4f | r %.4f"
          % (row["RMSE_fullbox"], row["RMSE_zw"], row["pearson_r"]))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
