"""Section 3.9 map figures, from suitability_index.nc and criterion_layers.nc.

Four figures:
  07  the seven standardised criteria and the exclusion mask
  08  the primary suitability map with its five tiers
  09  the four weighting schemes side by side, and the robust set
  10  suitability by period, and the change against the ERA5-basis present

The palette follows the rest of the deck: petrol for the low end, gold for the
high, with excluded land in a flat grey that reads as "not assessed" rather than
as a low score - the commonest way a suitability map misleads is by putting
exclusions on the same colour ramp as poor sites.
"""

import os
import warnings

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
SUIT = os.path.join(ROOT, "data/processed/suitability")
EVAL = os.path.join(ROOT, "data/processed/evaluation")
FIG = os.path.join(ROOT, "figures")

DEEP, PETROL, GOLD = "#0F2E3A", "#1E4E5F", "#C07A1E"
GREY = "#D9D9D6"
TIER_COLOURS = ["#7A3E00", "#C07A1E", "#D9C27A", "#9FB3BC", GREY]
TIER_NAMES = ["very high", "high", "moderate", "low", "unsuitable / excluded"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.titlesize": 9.5, "figure.dpi": 130})


def _frame(ax, lat, lon, title):
    ax.set_title(title, pad=5)
    ax.set_xticks([26, 28, 30, 32]); ax.set_yticks([-21, -19, -17])
    ax.set_xticklabels(["26°E", "28°E", "30°E", "32°E"])
    ax.set_yticklabels(["21°S", "19°S", "17°S"])
    ax.tick_params(length=2, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#B8B8B4")


def show(ax, da, lat, lon, mask=None, **kw):
    a = np.asarray(da, dtype=float).copy()
    if mask is not None:
        a[mask] = np.nan
    ext = [lon.min() - .05, lon.max() + .05, lat.min() - .05, lat.max() + .05]
    im = ax.imshow(a, origin="lower", extent=ext, aspect="auto", **kw)
    return im


def fig_criteria(lay, suit):
    lat, lon = lay.lat.values, lay.lon.values
    ex = suit.tier_primary.values == 4
    panels = [
        ("Irradiance (SARAH)", lay.ghi_present_sarah.values, "W m$^{-2}$"),
        ("Slope", lay.slope.values, "degrees"),
        ("Land cover score", lay.landcover_score.values, "0–1"),
        ("Distance to roads", np.minimum(lay.dist_roads.values, 80), "km"),
        ("Distance to grid", np.minimum(lay.dist_grid.values, 80), "km"),
        ("Population", np.log10(lay.population.values + 1), "log$_{10}$(people+1)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.0))
    for ax, (t, a, unit) in zip(axes.ravel(), panels):
        im = show(ax, a, lat, lon, mask=ex, cmap="cividis")
        _frame(ax, lat, lon, t)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.02).set_label(unit, size=7)
    # exclusion mask
    ax = axes.ravel()[6]
    comp = np.zeros_like(lay.slope.values)
    comp[lay.in_zimbabwe.values <= .5] = 1
    comp[lay.protected_fraction.values > .5] = 2
    comp[lay.urban_fraction.values > .5] = 3
    wet = lay.water_fraction.values + (lay.river_fraction.values
                                       if "river_fraction" in lay else 0)
    comp[wet > .5] = 4
    cmap = ListedColormap(["#F2F2EF", GREY, "#3D8B7D", "#94362A", "#1F4E79"])
    show(ax, comp, lat, lon, cmap=cmap, norm=BoundaryNorm(range(6), 5))
    _frame(ax, lat, lon, "Exclusion mask")
    from matplotlib.patches import Patch
    axes.ravel()[7].axis("off")
    axes.ravel()[7].legend(handles=[
        Patch(fc="#F2F2EF", ec="#999", label="retained"),
        Patch(fc=GREY, label="outside Zimbabwe"),
        Patch(fc="#3D8B7D", label="protected area"),
        Patch(fc="#94362A", label="urban + 1 km buffer"),
        Patch(fc="#1F4E79", label="water / riparian")],
        loc="center", frameon=False, fontsize=8.5)
    fig.suptitle("Section 3.9 criterion layers, excluded cells left blank",
                 y=.99, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(os.path.join(FIG, "07_suitability_criteria.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_primary(suit):
    lat, lon = suit.lat.values, suit.lon.values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ex = suit.tier_primary.values == 4
    im = show(a1, suit.si_primary.values, lat, lon, mask=ex, cmap="cividis",
              vmin=.3, vmax=.8)
    _frame(a1, lat, lon, "Suitability index — primary weights")
    fig.colorbar(im, ax=a1, fraction=.046).set_label("SI", size=8)
    cmap = ListedColormap(TIER_COLOURS)
    show(a2, suit.tier_primary.values, lat, lon, cmap=cmap,
         norm=BoundaryNorm(range(6), 5))
    _frame(a2, lat, lon, "Five-tier classification")
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(fc=c, label=n) for c, n in zip(TIER_COLOURS, TIER_NAMES)],
              loc="lower left", frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "08_suitability_primary.png"), bbox_inches="tight")
    plt.close(fig)


def fig_schemes(suit):
    lat, lon = suit.lat.values, suit.lon.values
    keys = ["primary", "irradiance-dominant", "infrastructure-dominant", "equal"]
    names = ["primary", "irradiance-dominant", "infrastructure-dominant",
             "equal weights"]
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.8))
    cmap = ListedColormap(TIER_COLOURS)
    for ax, k, n in zip(axes[:4], keys, names):
        show(ax, suit["tier_" + k].values, lat, lon, cmap=cmap,
             norm=BoundaryNorm(range(6), 5))
        _frame(ax, lat, lon, n)
    rob = suit.robustly_suitable.values.astype(float)
    rob[suit.tier_primary.values == 4] = np.nan
    show(axes[4], rob, lat, lon, cmap=ListedColormap(["#EDEDEA", "#7A3E00"]),
         vmin=0, vmax=1)
    # Read, never hardcoded. These titles said "42 cells" and "90.8%" - correct
    # for the first run, wrong from the decay resolution onward, and embedded in
    # a figure that appears in both Chapter 4 and Chapter 5 while the generated
    # captions around it said 145 and 87.8%.
    import pandas as pd
    r = pd.read_csv(os.path.join(EVAL, "suitability_robustness.csv")).iloc[0]
    _frame(axes[4], lat, lon,
           "Robust: high or better\nunder ALL four schemes (%d cells)" % r.robust)
    fig.suptitle("Weighting sensitivity — %.1f%% of retained cells change tier "
                 "under at least one scheme"
                 % (100 * r.weight_sensitive / r.assessed), y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "09_suitability_schemes.png"), bbox_inches="tight")
    plt.close(fig)


def fig_periods(suit):
    lat, lon = suit.lat.values, suit.lon.values
    per = [("present", "Present (SARAH)"),
           ("ssp245_near_term_2026_2050", "SSP2-4.5 near"),
           ("ssp245_long_term_2076_2100", "SSP2-4.5 long"),
           ("ssp585_near_term_2026_2050", "SSP5-8.5 near"),
           ("ssp585_long_term_2076_2100", "SSP5-8.5 long")]
    ex = suit.tier_primary.values == 4
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 7.0))
    for ax, (k, n) in zip(axes[0], per):
        im = show(ax, suit["si_" + k].values, lat, lon, mask=ex, cmap="cividis",
                  vmin=.3, vmax=.8)
        _frame(ax, lat, lon, n)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=.02).set_label("SI", size=8)
    axes[1][0].axis("off")
    axes[1][0].text(.5, .5, "Change is taken against the\nERA5-basis present, not\n"
                    "against SARAH — so the\ndifference stays inside one\n"
                    "measurement system.\n\nOnly the GHI layer varies by\n"
                    "period: infrastructure and\npopulation are frozen at\n"
                    "present values.",
                    ha="center", va="center", fontsize=8.2, color=DEEP)
    for ax, (k, n) in zip(axes[1][1:], per[1:]):
        im = show(ax, suit["dsi_" + k].values, lat, lon, mask=ex, cmap="RdYlBu_r",
                  vmin=-.15, vmax=.15)
        _frame(ax, lat, lon, "Δ SI — " + n)
    fig.colorbar(im, ax=axes[1][1:].tolist(), fraction=.02).set_label("Δ SI", size=8)
    fig.savefig(os.path.join(FIG, "10_suitability_periods.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    lay = xr.open_dataset(os.path.join(SUIT, "criterion_layers.nc"))
    suit = xr.open_dataset(os.path.join(SUIT, "suitability_index.nc"))
    os.makedirs(FIG, exist_ok=True)
    for fn, name in ((fig_criteria, "07_suitability_criteria"),
                     (fig_primary, "08_suitability_primary"),
                     (fig_schemes, "09_suitability_schemes"),
                     (fig_periods, "10_suitability_periods")):
        try:
            fn(lay, suit) if fn is fig_criteria else fn(suit)
            print("  ok  %s" % name)
        except Exception as e:
            print("  FAIL %s: %s: %s" % (name, type(e).__name__, e))
    print("\nFigures in %s" % FIG)


if __name__ == "__main__":
    main()
