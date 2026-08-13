"""C2: radially averaged power spectra and effective resolution.

Ullrich et al. recommend power spectra because ML emulators typically damp high
wavenumbers relative to physics-based models, making their effective resolution
coarser than their stated grid spacing. This restates the round-trip information
test (compute_information_content.py) wavenumber by wavenumber.

SPECTRA ARE COMPUTED IN BOTH SPACES, AND THE DISTINCTION IS THE WHOLE POINT.

The models predict the CLEAR-SKY INDEX. The delivered product is GHI, recovered
as CSI x clearsky. Because the CSI target was built by DIVIDING irradiance by
that same clear-sky field, the target's fine-scale structure is very nearly the
inverse of the clear-sky field's, and multiplying back cancels it - which is why
the GHI target is smooth (0.02 per cent of its power beyond wavenumber 10) while
the CSI target is not (0.25 per cent).

A model is therefore judged in CSI space. A model that reproduces CSI's
fine-scale structure recovers the cancellation and yields a smooth GHI; a model
that smooths CSI destroys the structure that would have cancelled, and the
clear-sky field's own structure then survives into its GHI. GHI-space excess is
consequently a symptom of getting CSI wrong, NOT evidence that a model invented
detail.

Measured in CSI space, the U-Net damps power beyond wavenumber 10 by a factor of
about twelve relative to the target - the damping Ullrich et al. describe, tied
to the loss function and architecture. The CNN, which carries an explicit
spatial-gradient penalty, sits within 4 per cent of the target and does not damp.

NO EFFECTIVE-RESOLUTION COLUMN IS REPORTED, deliberately. The conventional
definition - the wavelength at which power falls to half the reference - is
one-sided and degenerates on this GHI field, whose reference has almost no
short-wavelength power for a ratio to be taken against. A cumulative-power
substitute was tried and discarded: on a 71 x 81 grid the available wavelengths
are the discrete set 788/k km (788, 394, 263, 197, 158, 131, 113, ...), so the
statistic is quantised to a handful of bins. Across all six fields it took only
two distinct values, and it placed the CNN and the U-Net in the same bin while
their power ratios were 1.04x and 0.08x - opposite verdicts, identical number.
It could not discriminate, and read cold it appeared to say the damped model
resolved finer scales than the truth. The power ratio alone carries the result
without that risk.
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

    # Recover CSI = GHI / clearsky so the models can be judged in the space
    # their loss actually operates in.
    import xarray as _xr
    from ml_dataset_common import _clearsky_index_and_stack
    ds_val = _xr.open_dataset(os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc"))
    ds_cs = _xr.open_dataset(os.path.join(
        ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc"))
    stack = _clearsky_index_and_stack(ds_val.time.values, ds_cs)

    spectra, spectra_csi = {}, {}
    for label, var in SERIES:
        field = ds[var].values
        k, p = radial_spectrum(field.mean(axis=0))
        spectra[label] = p
        k, pc = radial_spectrum((field / stack).mean(axis=0))
        spectra_csi[label] = pc
    k = np.arange(1, len(next(iter(spectra.values()))) + 1)

    # wavelength in km for each radial wavenumber index
    domain_km = ny * FINE_RES_DEG * KM_PER_DEG
    wavelength_km = domain_km / k

    df = pd.DataFrame({"wavenumber": k, "wavelength_km": wavelength_km})
    for label in spectra:
        df[label] = spectra[label]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    ref, ref_csi = spectra["Truth (ERA5)"], spectra_csi["Truth (ERA5)"]
    tail_ref_csi = np.nansum(ref_csi[10:]) / np.nansum(ref_csi)
    rows = []
    for label, _ in SERIES:
        p = spectra[label]
        pc = spectra_csi[label]
        tail_csi = np.nansum(pc[10:]) / np.nansum(pc)
        ratio_csi = tail_csi / tail_ref_csi
        rows.append({
            "field": label,
            "CSI power beyond k=10": float(tail_csi),
            "CSI vs truth": ("reference" if label == "Truth (ERA5)"
                             else f"{ratio_csi:.2f}x"),
            "CSI verdict": ("reference" if label == "Truth (ERA5)"
                            else "DAMPED" if ratio_csi < 0.7
                            else "excess" if ratio_csi > 1.4 else "matches truth"),
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
    print("""
READ THE CSI COLUMNS, NOT THE GHI ONE. The models predict CSI; GHI is recovered
as CSI x clearsky. The CSI target's fine structure is nearly the inverse of the
clear-sky field's (it was made by dividing by it), so a model that reproduces
CSI correctly recovers the cancellation and gets a smooth GHI. GHI-space excess
therefore means a model got CSI WRONG - not that it invented detail.

  U-Net  damps CSI power beyond k=10 by ~12x: the Ullrich failure mode.
  CNN    sits within a few per cent of the target: its spatial-gradient
         penalty, which acts on CSI, is doing its job.

No effective-resolution figure is quoted. On this grid the available wavelengths
are the discrete set 788/k km, so any such statistic is quantised into a few
bins - it put the CNN and U-Net in the same bin despite opposite verdicts, and
read backwards besides. The power ratio says the same thing unambiguously.
""")
    print("\nGHI-space power ratio to truth, by wavelength band (symptom, not cause):")
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
