"""Generate every presentation figure into figures/, from committed code.

Run after any retrain so the figures cannot drift from the numbers.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})
C = {"Random Forest": "#1f4e79", "XGBoost": "#c07a1e", "CNN": "#3d8b7d",
     "U-Net": "#9b3b3b", "Baseline (bilinear)": "#8a8f98", "Truth (ERA5)": "#111111"}
MODELS = [("Random Forest", "ghi_rf"), ("XGBoost", "ghi_xgb"),
          ("CNN", "ghi_cnn"), ("U-Net", "ghi_unet")]


def fields():
    return xr.open_dataset(os.path.join(EVAL, "validation_spatial_fields.nc"))


# ---------------------------------------------------- 1. per-cell bias ------
def fig_bias_distribution():
    ds = fields()
    truth = ds["ghi_true"].values
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                  gridspec_kw={"width_ratios": [1.4, 1]})
    for label, var in MODELS:
        b = (ds[var].values - truth).mean(axis=0).ravel()
        ax.hist(b, bins=70, histtype="step", lw=1.8, label=label, color=C[label])
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Per-cell bias (W m$^{-2}$)")
    ax.set_ylabel("Grid cells")
    ax.set_title("Per-cell bias distribution\nU-Net's zero mean is cancellation, not accuracy")
    ax.legend(frameon=False, fontsize=9)

    stats = []
    for label, var in MODELS:
        b = (ds[var].values - truth).mean(axis=0)
        stats.append((label, b.mean(), b.std(), b.min(), b.max()))
    y = np.arange(len(stats))
    ax2.barh(y, [s[3] for s in stats], height=0.02, color="none")
    for i, (label, mn, sd, lo, hi) in enumerate(stats):
        ax2.plot([lo, hi], [i, i], lw=3, color=C[label], solid_capstyle="round", alpha=.55)
        ax2.plot(mn, i, "o", color=C[label], ms=8, zorder=3)
        ax2.text(hi + 0.6, i, f"mean {mn:+.2f}", va="center", fontsize=8.5, color=C[label])
    ax2.axvline(0, color="k", lw=0.8, ls="--")
    ax2.set_yticks(y); ax2.set_yticklabels([s[0] for s in stats])
    ax2.set_xlabel("Per-cell bias range (W m$^{-2}$)")
    ax2.set_title("Range of per-cell bias")
    fig.tight_layout(); fig.savefig(f"{FIG}/01_per_cell_bias.png"); plt.close(fig)


# ---------------------------------------------------------- 2. Taylor ------
def fig_taylor():
    df = pd.read_csv(os.path.join(EVAL, "taylor_diagram_stats.csv"))
    fig = plt.figure(figsize=(6.4, 6.2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetalim(0, np.pi / 2)
    ax.set_rlim(0, 1.35)
    for r in [0.5, 1.0]:
        ax.plot(np.linspace(0, np.pi / 2, 100), [r] * 100, color="0.85", lw=0.8, zorder=0)
    for label, sr, corr in zip(df["model"], df["std_ratio"], df["spatial_correlation"]):
        ax.plot(np.arccos(np.clip(corr, -1, 1)), sr, "o", ms=11,
                color=C.get(label, "0.4"), label=f"{label}  (R={corr:.4f})", zorder=3)
    ax.plot(0, 1.0, "*", ms=18, color="k", label="Reference", zorder=4)
    ticks = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
    ax.set_thetagrids(np.degrees(np.arccos(ticks)), [str(t) for t in ticks])
    ax.set_xlabel("Standard-deviation ratio")
    ax.set_title("Taylor diagram — time-mean spatial pattern\n"
                 "angle = spatial correlation, radius = amplitude ratio", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.13), frameon=False, fontsize=8.4)
    fig.tight_layout(); fig.savefig(f"{FIG}/02_taylor.png"); plt.close(fig)


# ------------------------------------------------ 3. feature importance ----
def fig_feature_importance():
    df = pd.read_csv(os.path.join(EVAL, "feature_importance.csv")).sort_values("rf_mdi")
    fig, ax = plt.subplots(figsize=(7.6, 5))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["rf_mdi"], height=0.38, color=C["Random Forest"], label="RF (impurity)")
    ax.barh(y + 0.2, df["xgb_gain"], height=0.38, color=C["XGBoost"], label="XGBoost (gain)")
    ax.set_yticks(y); ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Normalised importance")
    ax.set_title("Predictor importance, pixel-wise models\n"
                 "cloud fraction dominates; topography is exactly zero")
    for i, f in enumerate(df["feature"]):
        if f in ("elevation", "slope", "svf"):
            ax.text(0.004, i, "0.00000 — constant within each cell",
                    va="center", fontsize=8, color="0.35")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/03_feature_importance.png"); plt.close(fig)


# --------------------------------------------------- 4. uncertainty --------
def fig_uncertainty():
    df = pd.read_csv(os.path.join(EVAL, "uncertainty_decomposition_summary.csv"))
    df = df[df["model"] == "xgb"]          # the deployed model (Section 3.8.4)
    per = ["near_term_2026_2050", "mid_term_2051_2075", "long_term_2076_2100"]
    lbl = ["Near\n2026-50", "Mid\n2051-75", "Long\n2076-2100"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    keys = [("pct_var_ds", "$\\sigma_{DS}$ downscaling", "#1f4e79"),
            ("pct_var_arch", "$\\sigma_{arch}$ architecture", "#c07a1e"),
            ("pct_var_gcm", "$\\sigma_{GCM}$ GCM", "#3d8b7d"),
            ("pct_var_ssp", "$\\sigma_{SSP}$ scenario", "#9b3b3b")]
    x, bottom = np.arange(3), np.zeros(3)
    for k, name, col in keys:
        v = np.array([df[df["period"] == p][k].iloc[0] for p in per])
        a1.bar(x, v, bottom=bottom, label=name, color=col, width=0.62)
        bottom += v
    a1.set_xticks(x); a1.set_xticklabels(lbl)
    a1.set_ylabel("% of total variance"); a1.set_ylim(0, 100)
    a1.set_title("Uncertainty decomposition (XGBoost, deployed)")
    a1.legend(frameon=False, fontsize=8.6, loc="center right")

    for k, name, col in keys[1:3]:
        v = [df[df["period"] == p][k].iloc[0] for p in per]
        a2.plot(x, v, "o-", color=col, lw=2, ms=8, label=name)
    a2.set_xticks(x); a2.set_xticklabels(lbl)
    a2.set_ylabel("% of total variance")
    a2.set_title("Architecture choice overtakes GCM choice\nby the long-term horizon")
    a2.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/04_uncertainty.png"); plt.close(fig)


# ------------------------------------------------------ 5. spectra ---------
def fig_spectra():
    """Judge the models in CSI space, which is where their loss operates.

    An earlier version of this figure plotted GHI only and titled the ratio
    panel "Models INJECT small-scale power the target does not contain". That is
    the interpretation Section 7.8 overturned: the CSI target was built by
    DIVIDING irradiance by the clear-sky field, so multiplying back cancels its
    fine structure. A model that reproduces CSI correctly recovers the
    cancellation and yields a smooth GHI; one that smooths CSI destroys the
    structure that would have cancelled, and the clear-sky field's own structure
    then survives into its GHI. GHI-space excess is therefore a SYMPTOM of
    getting CSI wrong, not evidence that a model invented detail.
    """
    df = pd.read_csv(os.path.join(EVAL, "power_spectra.csv"))
    summ = pd.read_csv(os.path.join(EVAL, "effective_resolution.csv")).set_index("field")
    cols = ["Truth (ERA5)", "Baseline (bilinear)", "Random Forest", "XGBoost", "CNN", "U-Net"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.6, 4.5))

    wl = df["wavelength_km"]
    TICKS = [t for t in (800, 400, 200, 100, 50, 25) if wl.min() <= t <= wl.max()]

    def wavelength_axis(ax):
        """Log wavelength axis with readable labels. Matplotlib's minor tick
        labels overlap into an unreadable smear on an inverted log axis."""
        ax.invert_xaxis()
        ax.set_xticks(TICKS)
        ax.set_xticklabels([str(t) for t in TICKS])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Wavelength (km)")

    # --- CSI spectra: the space the models are actually judged in
    for c in cols:
        a1.loglog(df["wavelength_km"], df["CSI " + c], lw=1.9, color=C[c],
                  label=c, ls="--" if c == "Truth (ERA5)" else "-")
    wavelength_axis(a1)
    a1.set_ylabel("Power")
    a1.set_title("CSI power spectrum — the space judged in\nthe models predict CSI, not GHI")
    a1.legend(frameon=False, fontsize=8.2)

    # --- CSI ratio: this is where 1.04x and 0.08x live
    for c in cols[1:]:
        a2.loglog(df["wavelength_km"], df["CSI " + c] / df["CSI Truth (ERA5)"],
                  lw=1.9, color=C[c], label=c)
    a2.axhline(1, color="k", lw=0.9, ls="--")
    wavelength_axis(a2)
    a2.set_ylabel("CSI power ratio to truth")
    a2.set_title("U-Net DAMPS fine scales; the CNN does not\n"
                 "beyond k=10: CNN 1.04x, U-Net 0.08x")
    a2.legend(frameon=False, fontsize=8.2)
    for c, dy in [("CNN", 1.5), ("U-Net", 0.55)]:
        if c in summ.index:
            a2.text(a2.get_xlim()[1] * 1.05, dy, summ.loc[c, "CSI vs truth"],
                    fontsize=9, color=C[c], va="center", fontweight="bold")

    # --- GHI ratio, correctly labelled as a symptom rather than as invention
    for c in cols[1:]:
        a3.loglog(df["wavelength_km"], df[c] / df["Truth (ERA5)"], lw=1.9, color=C[c], label=c)
    a3.axhline(1, color="k", lw=0.9, ls="--")
    wavelength_axis(a3)
    a3.set_ylabel("GHI power ratio to truth")
    a3.set_title("GHI-space excess is a SYMPTOM of getting CSI wrong\n"
                 "not evidence a model invented detail")
    a3.legend(frameon=False, fontsize=8.2)

    fig.suptitle("Power spectra — read the CSI panels, not the GHI one", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/05_power_spectra.png", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------- 6. deployed error maps ------
def fig_error_maps():
    ds = xr.open_dataset(os.path.join(EVAL, "spatial_verification_maps.nc"))
    lat, lon = ds["fine_lat"].values, ds["fine_lon"].values
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    ext = [lon.min(), lon.max(), lat.min(), lat.max()]
    panels = [("ghi_true_mean_map", "Mean GHI, ERA5 truth", "viridis", None),
              ("bias_map_xgb", "XGBoost bias (deployed)", "RdBu_r", 2.2),
              ("rmse_map_xgb", "XGBoost RMSE (deployed)", "magma", None)]
    for ax, (var, title, cmap, lim) in zip(axes, panels):
        d = ds[var].values
        kw = dict(vmin=-lim, vmax=lim) if lim else {}
        im = ax.imshow(d, origin="lower", extent=ext, cmap=cmap, aspect="auto", **kw)
        ax.set_title(title); ax.set_xlabel("Longitude")
        ax.grid(alpha=.15)
        fig.colorbar(im, ax=ax, label="W m$^{-2}$", fraction=.046)
    axes[0].set_ylabel("Latitude")
    fig.suptitle("Deployed model — spatial error structure (2011-2024)", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/06_error_maps.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for fn in [fig_bias_distribution, fig_taylor, fig_feature_importance,
               fig_uncertainty, fig_spectra, fig_error_maps]:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\nFigures written to {FIG}/")
