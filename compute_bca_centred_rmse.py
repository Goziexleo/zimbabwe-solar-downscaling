"""BCa intervals for the centred-RMSE differences that Section 3.8.4's abandoned
composite criterion rested on.

Why this exists. compute_bootstrap_ci.py reports PERCENTILE intervals. For
centred RMSE the bootstrap distribution is known to be biased: every replicate
recomputes the time-mean map from resampled years, so its centred RMSE carries
sampling noise added in quadrature, and differences compress toward zero. The
percentile method assumes an approximately unbiased, symmetric bootstrap
distribution, so it is the wrong interval here - and naming the bias in the
documentation invites the obvious response, which is to correct for it.

Correcting for it can move the verdict. The percentile interval on
RF - XGBoost centred RMSE spans zero; a bias-shifted or basic (reverse
percentile) interval does not. BCa corrects for both bias and skewness and is
the standard resolution, so it is what decides the question rather than a choice
between two conventions that happen to disagree.

The resampling unit is the calendar year, matching compute_bootstrap_ci.py, and
the jackknife for the acceleration constant leaves out one year at a time.

    python compute_bca_centred_rmse.py
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

ROOT = os.path.dirname(os.path.abspath(__file__))
FIELDS = os.path.join(ROOT, "data/processed/evaluation/validation_spatial_fields.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
OUT = os.path.join(ROOT, "data/processed/evaluation/bca_intervals.csv")

N_BOOT = int(os.environ.get("N_BOOT", "2000"))
SEED = 42
ALPHA = 0.05

MODELS = {"Random Forest": "ghi_rf", "XGBoost": "ghi_xgb",
          "CNN": "ghi_cnn", "U-Net": "ghi_unet"}
PAIRS = [("Random Forest", "XGBoost"), ("CNN", "XGBoost"), ("U-Net", "XGBoost")]
METRICS = ["centred RMSE", "spatial R", "RMSE"]


def statistic(pred_a, pred_b, truth, idx, metric):
    """Difference a - b on the subset of time indices idx."""
    t = truth[idx]
    a, b = pred_a[idx], pred_b[idx]
    if metric == "RMSE":
        return float(np.sqrt(((a - t) ** 2).mean()) - np.sqrt(((b - t) ** 2).mean()))
    t_map = t.mean(axis=0)
    a_map, b_map = a.mean(axis=0), b.mean(axis=0)
    if metric == "centred RMSE":
        return float((a_map - t_map).std() - (b_map - t_map).std())
    return float(np.corrcoef(a_map.ravel(), t_map.ravel())[0, 1]
                 - np.corrcoef(b_map.ravel(), t_map.ravel())[0, 1])


def main():
    ds = xr.open_dataset(FIELDS)
    times = pd.DatetimeIndex(xr.open_dataset(VAL).time.values)
    years = times.year.values
    uniq = np.unique(years)
    year_idx = {y: np.where(years == y)[0] for y in uniq}
    full = np.arange(len(times))
    truth = ds["ghi_true"].values
    preds = {k: ds[v].values for k, v in MODELS.items()}

    print(f"{len(uniq)} years, {len(times)} months, {N_BOOT} replicates, "
          f"jackknife n = {len(uniq)}")

    rng = np.random.default_rng(SEED)
    draws = [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(N_BOOT)]
    boot_idx = [np.concatenate([year_idx[y] for y in d]) for d in draws]
    jack_idx = [np.concatenate([year_idx[y] for y in uniq if y != leave]) for leave in uniq]

    rows = []
    for a, b in PAIRS:
        for metric in METRICS:
            pa, pb = preds[a], preds[b]
            theta_hat = statistic(pa, pb, truth, full, metric)
            boot = np.array([statistic(pa, pb, truth, i, metric) for i in boot_idx])
            jack = np.array([statistic(pa, pb, truth, i, metric) for i in jack_idx])

            # bias-correction constant
            prop = float(np.mean(boot < theta_hat))
            prop = min(max(prop, 1.0 / (2 * N_BOOT)), 1 - 1.0 / (2 * N_BOOT))
            z0 = norm.ppf(prop)

            # acceleration from the jackknife
            dev = jack.mean() - jack
            denom = 6.0 * (np.sum(dev ** 2) ** 1.5)
            accel = float(np.sum(dev ** 3) / denom) if denom != 0 else 0.0

            out = {}
            for tag, alpha in [("lo", ALPHA / 2), ("hi", 1 - ALPHA / 2)]:
                z = norm.ppf(alpha)
                adj = z0 + (z0 + z) / (1 - accel * (z0 + z))
                out[tag] = float(np.percentile(boot, 100 * norm.cdf(adj)))

            pct_lo, pct_hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
            basic_lo, basic_hi = 2 * theta_hat - pct_hi, 2 * theta_hat - pct_lo

            spans = lambda lo, hi: (lo <= 0 <= hi)
            rows.append({
                "model_a": a, "model_b": b, "metric": metric,
                "plug_in": theta_hat, "resample_mean": float(boot.mean()),
                "bias": float(boot.mean() - theta_hat),
                "pct_lo": pct_lo, "pct_hi": pct_hi, "pct_spans_zero": spans(pct_lo, pct_hi),
                "basic_lo": basic_lo, "basic_hi": basic_hi,
                "basic_spans_zero": spans(basic_lo, basic_hi),
                "bca_lo": out["lo"], "bca_hi": out["hi"],
                "bca_spans_zero": spans(out["lo"], out["hi"]),
                "z0": z0, "acceleration": accel,
            })
            print(f"  {a[:12]:<13} - {b:<8} {metric:<13} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 104)
    print(" BCa vs PERCENTILE vs BASIC - does the difference survive?")
    print("=" * 104)
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        for _, r in df[df.metric == metric].iterrows():
            v = lambda s: "spans zero" if r[s] else "EXCLUDES zero"
            print(f"  {r.model_a[:13]:<14} - {r.model_b:<8} "
                  f"plug-in {r.plug_in:+8.4f}  resample {r.resample_mean:+8.4f}  "
                  f"bias {r.bias:+7.4f}")
            print(f"  {'':<24} percentile [{r.pct_lo:+8.4f}, {r.pct_hi:+8.4f}]  {v('pct_spans_zero')}")
            print(f"  {'':<24} basic      [{r.basic_lo:+8.4f}, {r.basic_hi:+8.4f}]  {v('basic_spans_zero')}")
            print(f"  {'':<24} BCa        [{r.bca_lo:+8.4f}, {r.bca_hi:+8.4f}]  {v('bca_spans_zero')}"
                  f"   (z0 {r.z0:+.3f}, a {r.acceleration:+.4f})")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
