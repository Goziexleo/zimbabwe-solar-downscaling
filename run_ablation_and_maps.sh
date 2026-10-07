#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD"
stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "1  suitability maps, so the figure guard passes honestly"
$PY make_suitability_maps.py

stage "2  rsds ablation, configuration C (deployed, rsds dropped)"
MODEL_PREDICTORS="clt,tas,ps,huss,od550aer" ABL_N_JOBS=6 $PY -u compute_rsds_ablation.py C

stage "3  rsds ablation, configuration B (rsds kept)"
MODEL_PREDICTORS="clt,tas,ps,huss,rsds,od550aer" ABL_N_JOBS=6 $PY -u compute_rsds_ablation.py B

stage "ABLATION COMPLETE"
$PY -c "import pandas as pd; print(pd.read_csv('data/processed/evaluation/rsds_ablation.csv')[['configuration','n_features','RMSE_fullbox','RMSE_zw','pearson_r']].round(4).to_string(index=False))"
