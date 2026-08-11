"""Parse the NOAA CPC Oceanic Nino Index (ONI) table for ENSO phase
classification, used by the stratified mini-batch sampling of Section 3.6.8.

Source: NOAA CPC oni.ascii.txt (public climate dataset), saved locally at
data/raw/oni/oni_ascii.txt. Each row is a 3-month running-mean season (e.g.
"DJF", "NDJ") attributed to its middle calendar month, following NOAA's
standard convention.
"""

import os
import pandas as pd

_SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

EL_NINO_THRESHOLD = 0.5
LA_NINA_THRESHOLD = -0.5

_ONI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "oni", "oni_ascii.txt")


def load_oni_by_year_month(path=_ONI_PATH):
    """Returns a dict {(year, month): oni_anomaly}."""
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.strip().upper() for c in df.columns]
    month = df["SEAS"].map(_SEASON_TO_MONTH)
    return {(int(yr), int(mo)): float(anom) for yr, mo, anom in zip(df["YR"], month, df["ANOM"])}


def classify_enso_phase(times, oni_path=_ONI_PATH):
    """times: iterable of datetime-like. Returns a list of phase strings
    ('el_nino', 'la_nina', 'neutral') aligned with `times`."""
    oni_lookup = load_oni_by_year_month(oni_path)
    phases = []
    for t in pd.DatetimeIndex(pd.to_datetime(list(times))):
        oni = oni_lookup.get((t.year, t.month))
        if oni is None:
            phases.append("neutral")
        elif oni > EL_NINO_THRESHOLD:
            phases.append("el_nino")
        elif oni < LA_NINA_THRESHOLD:
            phases.append("la_nina")
        else:
            phases.append("neutral")
    return phases
