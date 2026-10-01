"""The national mask, shared by every script that reports a metric "over Zimbabwe".

The analysis box spans 25.0 to 33.0 east and 22.0 to 15.0 south on a 0.1 degree
mesh, which is 5,751 cells. Only 3,291 of those are inside Zimbabwe. The other
2,460, or 42.8 per cent, lie in Zambia, Botswana, Mozambique and South Africa.
Every validation metric in Chapter 4 was computed over the whole box while being
described as a result over Zimbabwe, which an examiner identified.

This module is the single definition of the mask, so that the masked metrics
cannot disagree between scripts. The threshold is areal: a cell counts as inside
when more than half of it falls within the national boundary, which is the same
rule compute_suitability.py already applies through AREAL_EXCLUSION.

Metrics that are a spatial transform of a complete field - the radially averaged
power spectra, the effective-resolution ratios and the round-trip variance shares
- CANNOT take this mask. A discrete Fourier transform needs a filled rectangle,
and zeroing two fifths of one would put a coastline-shaped step into the spectrum
and dominate the high wavenumbers. Those analyses stay on the full box and must
be reported as covering the analysis domain, not Zimbabwe.
"""

import os

import numpy as np
import xarray as xr

CRITERIA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data/processed/suitability/criterion_layers.nc")
AREAL_THRESHOLD = 0.5

_CACHE = {}


def zimbabwe_mask(shape=None):
    """Boolean (lat, lon) mask, True inside Zimbabwe.

    The grid is asserted against the caller's field shape rather than assumed,
    because a silent shape mismatch here would mask the wrong cells and every
    downstream number would be quietly wrong rather than visibly broken.
    """
    if "mask" not in _CACHE:
        ds = xr.open_dataset(CRITERIA_PATH)
        m = ds["in_zimbabwe"].values > AREAL_THRESHOLD
        _CACHE["mask"] = m
        _CACHE["lat"] = ds["lat"].values
        _CACHE["lon"] = ds["lon"].values
    m = _CACHE["mask"]
    if shape is not None:
        assert m.shape == tuple(shape[-2:]), (
            f"mask grid {m.shape} does not match field grid {tuple(shape[-2:])}")
    return m


def apply_mask(field, mask=None):
    """Flatten a (time, lat, lon) or (lat, lon) field to its inside-Zimbabwe cells."""
    m = zimbabwe_mask(field.shape) if mask is None else mask
    return field[..., m] if field.ndim == 2 else field[:, m]


def describe():
    m = zimbabwe_mask()
    n, tot = int(m.sum()), m.size
    return (f"{n} of {tot} cells inside Zimbabwe "
            f"({tot - n} outside, {100 * (tot - n) / tot:.1f} per cent)")


if __name__ == "__main__":
    print(describe())
    m = zimbabwe_mask()
    lat, lon = _CACHE["lat"], _CACHE["lon"]
    rows = m.sum(axis=1)
    print(f"southernmost row with land: {lat[rows > 0].min():.1f}, "
          f"northernmost {lat[rows > 0].max():.1f}")
    print(f"cells on the southern edge row ({lat[0]:.1f}): {int(m[0].sum())}"
          "  <- non-zero means the domain truncates Zimbabwe")
