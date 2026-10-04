"""Score the cross-validated pixel-wise configurations on the validation record.

Section 3.6.7 says the defaults were retained because the cross-validated
configurations "performed marginally worse on the full 5,751-cell validation
set". That is selection on the withheld record, and it is the defect the thesis
corrects at length for the neural models' checkpoints and hyperparameter grids.
It cannot be argued away, but it can at least be quantified: the comparison the
sentence rests on was never reported, so a reader has no way to see how much
turns on it.

Both configurations are fitted here and scored on the same validation fields the
rest of Chapter 4 uses, on the Zimbabwe basis. Training and scoring happen in
one pass per cell, so nothing is persisted: fitting 5,751 cells twice for each
family is cheaper than writing two more model directories to disk.

The deployed configurations come from the training scripts and the
cross-validated ones from hpo_pixelwise_*.csv, so neither is typed in here.

    python compute_pixelwise_cv_config_check.py
"""

import json
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from ml_dataset_common import csi_to_ghi
from pixelwise_common import build_fine_pixel_dataset
from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
OUT = os.path.join(EVAL, "pixelwise_cv_config_check.csv")
TRAIN = os.path.join(ROOT, "data/processed/ml_ready/ml_training_dataset.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
TOPO = os.path.join(ROOT, "data/processed/topography",
                    "zimbabwe_topographic_features_0.1deg.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid",
                        "clearsky_ghi_finegrid_climatology.nc")
SEED = 42
N_JOBS = int(os.environ.get("CVCHK_N_JOBS", "6"))


def default_of(script, name):
    src = open(os.path.join(ROOT, script)).read()
    m = re.search(r'os\.environ\.get\(\s*"%s"\s*,\s*"([^"]+)"' % name, src)
    assert m is not None, "could not read %s from %s" % (name, script)
    return m.group(1)


def deployed():
    return {
        "XGBoost": dict(max_depth=int(default_of("train_pixelwise_xgb.py", "XGB_MAX_DEPTH")),
                        eta=float(default_of("train_pixelwise_xgb.py", "XGB_ETA")),
                        subsample=float(default_of("train_pixelwise_xgb.py", "XGB_SUBSAMPLE")),
                        min_child_weight=int(default_of("train_pixelwise_xgb.py",
                                                        "XGB_MIN_CHILD_WEIGHT"))),
        "Random Forest": dict(
            n_estimators=int(default_of("train_pixelwise_rf.py", "RF_N_ESTIMATORS")),
            max_features=default_of("train_pixelwise_rf.py", "RF_MAX_FEATURES"),
            min_samples_leaf=int(default_of("train_pixelwise_rf.py",
                                            "RF_MIN_SAMPLES_LEAF"))),
    }


def cv_winner(tag):
    d = pd.read_csv(os.path.join(EVAL, "hpo_pixelwise_%s.csv" % tag))
    r = d.sort_values("cv_rmse_csi").iloc[0]
    if tag == "xgb":
        return dict(max_depth=int(r.max_depth), eta=float(r.eta),
                    subsample=float(r.subsample),
                    min_child_weight=int(r.min_child_weight))
    return dict(n_estimators=int(r.n_estimators), max_features=str(r.max_features),
                min_samples_leaf=int(r.min_samples_leaf))


N_ROUNDS = int(default_of("train_pixelwise_xgb.py", "XGB_N_ESTIMATORS"))


def literal_of(script, name):
    """A hardcoded keyword argument in a training script, so the comparison uses
    the deployed functional form rather than a nearby one. The deployed XGBoost
    carries reg_alpha and reg_lambda that are not env-overridable and were absent
    from both this check and the grid search until they were found to account for
    a 0.1 W/m2 discrepancy against Table 4.1."""
    src = open(os.path.join(ROOT, script)).read()
    m = re.search(r"%s\s*=\s*([0-9.]+)" % name, src)
    assert m is not None, "could not read %s from %s" % (name, script)
    return float(m.group(1))


XGB_REG = dict(reg_alpha=literal_of("train_pixelwise_xgb.py", "reg_alpha"),
               reg_lambda=literal_of("train_pixelwise_xgb.py", "reg_lambda"))


def fit_cell(family, cfg, xtr, ytr, xva):
    if family == "XGBoost":
        m = xgb.XGBRegressor(n_estimators=N_ROUNDS, random_state=SEED, n_jobs=1,
                             **XGB_REG, **cfg)
    else:
        m = RandomForestRegressor(random_state=SEED, n_jobs=1, **cfg)
    m.fit(xtr, ytr)
    return m.predict(xva)


def main():
    dtr, dva = xr.open_dataset(TRAIN), xr.open_dataset(VAL)
    cs = xr.open_dataset(CLEARSKY)
    Xtr, ytr, feats = build_fine_pixel_dataset(dtr, TOPO)
    Xva, yva, feats_v = build_fine_pixel_dataset(dva, TOPO)
    assert feats == feats_v, "feature order differs between train and validation"
    with open(os.path.join(ROOT, "data/processed/models/pixelwise_xgb/manifest.json")) as f:
        assert json.load(f)["feature_vars"] == feats, "manifest feature order mismatch"

    nt, nlat, nlon, _ = Xva.shape
    times = dva.time.values
    mask = zimbabwe_mask(shape=(nlat, nlon))
    truth = csi_to_ghi(yva, times, cs)
    cells = [(i, j) for i in range(nlat) for j in range(nlon)]

    rows = []
    for family, tag in (("XGBoost", "xgb"), ("Random Forest", "rf")):
        for label, cfg in (("deployed", deployed()[family]),
                           ("cross-validated", cv_winner(tag))):
            print("fitting %s, %s configuration: %s" % (family, label, cfg))
            preds = Parallel(n_jobs=N_JOBS, verbose=0)(
                delayed(fit_cell)(family, cfg, Xtr[:, i, j, :], ytr[:, i, j],
                                  Xva[:, i, j, :]) for i, j in cells)
            csi = np.zeros((nt, nlat, nlon), dtype=np.float32)
            for (i, j), p in zip(cells, preds):
                csi[:, i, j] = p
            ghi = csi_to_ghi(csi, times, cs)
            err = ghi - truth
            zw = np.broadcast_to(mask, err.shape)
            sel = zw & np.isfinite(err)
            rows.append(dict(model=family, configuration=label,
                             RMSE_zw=float(np.sqrt(np.mean(err[sel] ** 2))),
                             MBE_zw=float(np.mean(err[sel])),
                             RMSE_fullbox=float(np.sqrt(np.mean(err[np.isfinite(err)] ** 2))),
                             config=json.dumps(cfg, sort_keys=True)))
            print("   RMSE over Zimbabwe %.4f W/m2" % rows[-1]["RMSE_zw"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\nValidation error by configuration (W/m2, Zimbabwe basis)\n")
    print("%-14s %-17s %10s %10s" % ("model", "configuration", "RMSE_zw", "MBE_zw"))
    for _, w in df.iterrows():
        print("%-14s %-17s %10.3f %10.3f"
              % (w["model"], w["configuration"], w["RMSE_zw"], w["MBE_zw"]))

    print()
    for family in ("XGBoost", "Random Forest"):
        g = df[df.model == family].set_index("configuration")
        dep, cv = g.loc["deployed", "RMSE_zw"], g.loc["cross-validated", "RMSE_zw"]
        print("%-14s deployed %.3f against cross-validated %.3f: the default is %s "
              "by %.3f W/m2" % (family, dep, cv,
                                "better" if dep < cv else "worse", abs(dep - cv)))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
