"""Download the four ERA5 predictors needed for the 2011--2024 validation split."""

from pathlib import Path
import time

import cdsapi


START_YEAR = 2011
END_YEAR = 2024
OUTPUT_DIR = Path("./data/raw/era5/predictors/validation")
VARIABLES = (
    "total_cloud_cover",
    "2m_temperature",
    "surface_pressure",
    "2m_dewpoint_temperature",
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for variable in VARIABLES:
        for year in range(START_YEAR, END_YEAR + 1):
            output_path = OUTPUT_DIR / f"era5_{variable}_{year}.nc"
            if output_path.exists():
                print(f"{output_path.name}: already present; skipping.")
                continue

            print(f"Downloading {variable} for {year}...")
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": variable,
                    "year": str(year),
                    "month": [f"{month:02d}" for month in range(1, 13)],
                    "day": [f"{day:02d}" for day in range(1, 32)],
                    "time": [f"{hour:02d}:00" for hour in range(24)],
                    "area": [-15, 25, -22, 33],
                },
                str(output_path),
            )
            time.sleep(2)


if __name__ == "__main__":
    main()
