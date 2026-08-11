"""C2: radially averaged power spectra and effective resolution.

Ullrich et al. recommend power spectra because ML emulators typically damp high
wavenumbers relative to physics-based models, making their effective resolution
coarser than their stated grid spacing. This restates the round-trip information
test (compute_information_content.py) wavenumber by wavenumber.

The result here runs the OTHER WAY, and the reason matters. The reference field
is itself a bicubic interpolation of 0.25 deg ERA5, so it carries almost no
genuine signal below about 25 km - 0.02 per cent of its power sits beyond
wavenumber 10. The models therefore hold MORE short-wavelength power than the
target, not less: they inject small-scale structure rather than damping it, and
because the target has nothing there to predict, that structure is spurious.

The usual "power falls to half the reference" definition of effective resolution
is one-sided and returns nothing here. Effective resolution is instead reported
as the shortest wavelength still inside the band carrying 99 per cent of a
field's own power, which is well defined under damping and injection alike.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
FIELDS = os.path.join(ROOT, "data/processed/evaluation/validation_spatial_fields.nc")
OUT_CSV = os.path.join(ROOT, "data/processed/evaluation/power_spectra.csv")
OUT_SUMMARY = os.path.join(ROOT, "data/processed/evaluation/effective_resolution.csv")

FINE_RES_DEG = 0.1
KM_PER_DEG = 111.0
CUMULATIVE_THRESHOLD = 0.99  # effective resolution: shortest wavelength still carrying signal

SERIES = [
    ("Truth (ERA5)", "ghi_true"),
    ("Baseline (bilinear)", "ghi_baseline"),
    ("Random Forest", "ghi_rf"),
    ("XGBoost", "ghi_xgb"),
    ("CNN", "ghi_cnn"),
    ("U-Net", "ghi_unet"),
]


def radial_spectrum(field):
    """Radially averaged 2-D power spectrum of a single 2-D field.

    Returns (wavenumber_index, power). The field is detrended by removing its
    mean and windowed with a Hann taper in both directions, since the domain is
    not periodic and an untapered FFT would leak edge discontinuities across all
    wavenumbers.
    """
    f = field - field.mean()
    ny, nx = f.shape
    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    f = f * (wy * wx)

    p = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)

    nbins = min(cy, cx)
    power = np.array([p[r == k].mean() if (r == k).any() else np.nan
                      for k in range(1, nbins)])
    k = np.arange(1, nbins)
    return k, power


def main():
    ds = xr.open_dataset(FIELDS)
    ny = ds.sizes["fine_lat"]

    spectra = {}
    for label, var in SERIES:
        mean_map = ds[var].values.mean(axis=0)
        k, p = radial_spectrum(mean_map)
        spectra[label] = p
    k = np.arange(1, len(next(iter(spectra.values()))) + 1)

    # wavelength in km for each radial wavenumber index
    domain_km = ny * FINE_RES_DEG * KM_PER_DEG
    wavelength_km = domain_km / k

    df = pd.DataFrame({"wavenumber": k, "wavelength_km": wavelength_km})
    for label in spectra:
        df[label] = spectra[label]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    ref = spectra["Truth (ERA5)"]
    rows = []
    for label, _ in SERIES:
        p = spectra[label]
        # Effective resolution: the shortest wavelength still inside the band
        # holding CUMULATIVE_THRESHOLD of the field's total power. Defined this
        # way it is meaningful whether a field damps the high wavenumbers or
        # injects into them, unlike a one-sided "power falls below X" rule.
        cum = np.nancumsum(p) / np.nansum(p)
        idx = int(np.searchsorted(cum, CUMULATIVE_THRESHOLD))
        idx = min(idx, len(wavelength_km) - 1)
        tail = np.nansum(p[10:]) / np.nansum(p)
        rows.append({
            "field": label,
            f"effective resolution (km, {int(CUMULATIVE_THRESHOLD*100)}% of power)":
                float(wavelength_km[idx]),
            "power fraction beyond wavenumber 10": float(tail),
            "vs truth": ("reference" if label == "Truth (ERA5)"
                         else f"{tail / (np.nansum(ref[10:]) / np.nansum(ref)):.1f}x"),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("=" * 96)
    print(" RADIALLY AVERAGED POWER SPECTRA — time-mean GHI field")
    print(f" Domain {domain_km:.0f} km across; grid spacing {FINE_RES_DEG}deg "
          f"(~{FINE_RES_DEG * KM_PER_DEG:.0f} km); Nyquist ~{2 * FINE_RES_DEG * KM_PER_DEG:.0f} km")
    print("=" * 96)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 96)
    print("\nNOTE: every model holds MORE short-wavelength power than the truth field,")
    print("not less. The truth is a bicubic interpolation of 0.25 deg ERA5 and carries")
    print("almost no genuine sub-25 km signal, so the models are INJECTING small-scale")
    print("structure rather than damping it. The ratios below are therefore large")
    print("because the denominator is near zero, and they measure spurious detail, not")
    print("recovered detail. This is the spectral statement of the round-trip result.")
    print("\nPower ratio to truth, by wavelength band:")
    hdr = f"{'wavelength':>12} " + "".join(f"{l.split(' (')[0][:12]:>14}" for l, _ in SERIES[1:])
    print(hdr)
    for idx in np.linspace(0, len(k) - 1, 8).astype(int):
        line = f"{wavelength_km[idx]:>10.0f}km "
        for label, _ in SERIES[1:]:
            line += f"{spectra[label][idx] / ref[idx]:>14.3f}"
        print(line)
    print(f"\nSaved: {OUT_CSV}\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
