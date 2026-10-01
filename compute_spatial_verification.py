"""Section 3.7.2 spatial verification: Taylor diagram statistics (spatial
correlation, normalised std ratio, centered RMSE of the time-mean
climatological pattern) plus per-pixel time-mean bias and RMSE maps, for
each of the 4 finalized models against the true reference field.

Every statistic is computed twice: over the full 71 x 81 analysis box, and over
the 3,291 cells inside Zimbabwe. The box extends well into Zambia, Botswana,
Mozambique and South Africa, so a spatial correlation computed across it is
partly a correlation over other countries, which is not what Section 3.7.2
claims to report. The masked columns carry a _zw suffix and are the ones the
prose should quote.

The maps written to spatial_verification_maps.nc are left unmasked, because a
map should show what the model did everywhere it was run; the national boundary
belongs on the figure as a drawn border, not as missing data.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd

from zimbabwe_mask import describe as mask_describe
from zimbabwe_mask import zimbabwe_mask

INPUT_PATH = os.path.abspath("./data/processed/evaluation/validation_spatial_fields.nc")
OUTPUT_PATH = os.path.abspath("./data/processed/evaluation/spatial_verification_maps.nc")
TAYLOR_CSV_PATH = os.path.abspath("./data/processed/evaluation/taylor_diagram_stats.csv")

ds = xr.open_dataset(INPUT_PATH)
ref = ds["ghi_true"].values  # (time, fine_lat, fine_lon)

models = {
    "Baseline (bilinear)": ds["ghi_baseline"].values,
    "Random Forest": ds["ghi_rf"].values,
    "XGBoost": ds["ghi_xgb"].values,
    "CNN": ds["ghi_cnn"].values,
    "U-Net": ds["ghi_unet"].values,
}

ref_mean_map = ref.mean(axis=0)
ref_std_spatial = ref_mean_map.std()

MASK = zimbabwe_mask(ref.shape)
ref_std_zw = ref_mean_map[MASK].std()


def taylor(model_mean_map, ref_map, ref_std, sel=None):
    """Spatial-pattern statistics, over every cell or over a selection."""
    m = model_mean_map[sel] if sel is not None else model_mean_map.ravel()
    r = ref_map[sel] if sel is not None else ref_map.ravel()
    corr = float(np.corrcoef(m, r)[0, 1])
    sd = float(m.std())
    ratio = sd / ref_std if ref_std > 0 else np.nan
    crmse = float(np.sqrt(sd ** 2 + ref_std ** 2 - 2 * sd * ref_std * corr))
    return corr, ratio, crmse

rows = []
bias_maps = {}
rmse_maps = {}

for name, arr in models.items():
    model_mean_map = arr.mean(axis=0)

    # --- Taylor diagram stats, computed on the spatial pattern of the
    #     time-mean climatology (Section 3.7.2) ---
    spatial_corr, std_ratio, crmse = taylor(
        model_mean_map, ref_mean_map, ref_std_spatial)
    corr_zw, ratio_zw, crmse_zw = taylor(
        model_mean_map, ref_mean_map, ref_std_zw, sel=MASK)

    # --- Bias & RMSE maps: full temporal comparison at every pixel ---
    bias_map = model_mean_map - ref_mean_map
    rmse_map = np.sqrt(((arr - ref) ** 2).mean(axis=0))

    bias_maps[name] = bias_map
    rmse_maps[name] = rmse_map

    rows.append({
        "model": name,
        "spatial_correlation": spatial_corr,
        "std_ratio": std_ratio,
        "centered_rmse": crmse,
        "domain_mean_bias": bias_map.mean(),
        "domain_mean_rmse": rmse_map.mean(),
        "spatial_correlation_zw": corr_zw,
        "std_ratio_zw": ratio_zw,
        "centered_rmse_zw": crmse_zw,
        "domain_mean_bias_zw": bias_map[MASK].mean(),
        "domain_mean_rmse_zw": rmse_map[MASK].mean(),
    })

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(TAYLOR_CSV_PATH), exist_ok=True)
df.to_csv(TAYLOR_CSV_PATH, index=False)

print("=" * 100)
print(" TAYLOR DIAGRAM STATISTICS (Section 3.7.2) - time-mean climatological pattern vs. reference")
print(" Reference spatial std (true GHI time-mean map): {:.4f} W m-2 full box,"
      " {:.4f} over Zimbabwe".format(ref_std_spatial, ref_std_zw))
print(" mask: " + mask_describe())
print("=" * 100)
_full = ["model", "spatial_correlation", "std_ratio", "centered_rmse",
         "domain_mean_bias", "domain_mean_rmse"]
print(" full analysis box:")
print(df[_full].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n over Zimbabwe (the figures the prose should quote):")
print(df[["model"] + [c + "_zw" for c in _full[1:]]].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))
print("=" * 100)

safe_names = {
    "Baseline (bilinear)": "baseline",
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "CNN": "cnn",
    "U-Net": "unet",
}

data_vars = {"ghi_true_mean_map": (["fine_lat", "fine_lon"], ref_mean_map)}
for name, key in safe_names.items():
    data_vars[f"bias_map_{key}"] = (["fine_lat", "fine_lon"], bias_maps[name])
    data_vars[f"rmse_map_{key}"] = (["fine_lat", "fine_lon"], rmse_maps[name])

ds_out = xr.Dataset(
    data_vars,
    coords={"fine_lat": ds["fine_lat"].values, "fine_lon": ds["fine_lon"].values},
)
for v in ds_out.data_vars:
    ds_out[v].attrs["units"] = "W m-2"
ds_out.to_netcdf(OUTPUT_PATH)
print(f"\nSaved bias/RMSE maps: {OUTPUT_PATH}")
print(f"Saved Taylor diagram stats table: {TAYLOR_CSV_PATH}")
