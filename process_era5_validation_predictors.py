"""Aggregate ERA5 validation predictors to monthly clt, tas, ps, and huss fields."""

from pathlib import Path

import numpy as np
import xarray as xr


START_YEAR = 2011
END_YEAR = 2024
INPUT_DIR = Path("./data/raw/era5/predictors/validation")
OUTPUT_PATH = Path("./data/processed/era5/predictors/era5_monthly_predictors_2011_2024.nc")


def open_clean(path: Path) -> xr.Dataset:
    dataset = xr.open_dataset(path)
    rename_map = {}
    if "valid_time" in dataset.coords:
        rename_map["valid_time"] = "time"
    if "latitude" in dataset.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in dataset.coords:
        rename_map["longitude"] = "lon"
    return dataset.rename(rename_map).drop_vars(["expver", "number"], errors="ignore")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly = []
    for year in range(START_YEAR, END_YEAR + 1):
        paths = {
            "tcc": INPUT_DIR / f"era5_total_cloud_cover_{year}.nc",
            "t2m": INPUT_DIR / f"era5_2m_temperature_{year}.nc",
            "sp": INPUT_DIR / f"era5_surface_pressure_{year}.nc",
            "d2m": INPUT_DIR / f"era5_2m_dewpoint_temperature_{year}.nc",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing validation predictor files:\n" + "\n".join(missing))

        datasets = {name: open_clean(path) for name, path in paths.items()}
        d2m, surface_pressure = datasets["d2m"]["d2m"], datasets["sp"]["sp"]
        vapour_pressure = 611.21 * np.exp(17.502 * (d2m - 273.16) / (d2m - 32.19))
        huss = (287.0597 / 461.5250) * vapour_pressure / (
            surface_pressure - (1 - 287.0597 / 461.5250) * vapour_pressure
        )
        year_data = xr.merge([
            (datasets["tcc"]["tcc"] * 100).rename("clt"),
            datasets["t2m"]["t2m"].rename("tas"),
            surface_pressure.rename("ps"),
            huss.rename("huss"),
        ]).resample(time="1MS").mean()
        monthly.append(year_data.load())
        for dataset in datasets.values():
            dataset.close()
        print(f"Processed {year}.")

    xr.concat(monthly, dim="time").to_netcdf(OUTPUT_PATH)
    print(f"Saved monthly validation predictors to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
