"""Figure 1.1: the study area, which the dissertation did not have.

Chapters 1 to 3 contained no figure at all - no study-area map, no workflow, no
architecture diagram - which an examiner noted first among the presentation
faults. A reader could not see where Zimbabwe is, what the terrain does, or how
the analysis box relates to the national boundary.

The map carries the three things the text relies on and never showed:

  relief          SRTM elevation, which the slope and sky-view criteria derive
                  from and which explains the Eastern Highlands
  provinces       GADM level 1, so the robust-set counts in Section 4.8.1 can be
                  read against named regions
  the box         the 25-33 E, 15-22 S analysis domain, drawn against the border
                  so the truncation at 22 S is visible rather than buried in a
                  coordinate list

Writes figures/00_study_area.png.
"""

import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.abspath(__file__))
GADM = os.path.join(ROOT, "data/raw/gadm/gadm41_ZWE.gpkg")
TOPO = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
FIG = os.path.join(ROOT, "figures")

BOX = dict(lon0=25.0, lon1=33.0, lat0=-22.0, lat1=-15.0)
TOWNS = [("Harare", 31.05, -17.83), ("Bulawayo", 28.58, -20.15),
         ("Mutare", 32.65, -18.97), ("Gweru", 29.82, -19.45),
         ("Kwekwe", 29.82, -18.92), ("Beitbridge", 30.00, -22.22)]


def main():
    os.makedirs(FIG, exist_ok=True)
    prov = gpd.read_file(GADM, layer="ADM_ADM_1")
    nat = gpd.read_file(GADM, layer="ADM_ADM_0")

    t = xr.open_dataset(TOPO)
    ln = "lat" if "lat" in t.coords else "y"
    gn = "lon" if "lon" in t.coords else "x"
    t = t.rename({ln: "lat", gn: "lon"}).sortby("lat")
    elev = t["elevation"]

    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    im = ax.pcolormesh(elev.lon.values, elev.lat.values, elev.values,
                       cmap="terrain", shading="auto", vmin=0, vmax=2000, zorder=1)
    prov.boundary.plot(ax=ax, linewidth=0.7, edgecolor="0.25", zorder=3)
    nat.boundary.plot(ax=ax, linewidth=1.8, edgecolor="black", zorder=4)

    ax.add_patch(Rectangle((BOX["lon0"], BOX["lat0"]),
                           BOX["lon1"] - BOX["lon0"], BOX["lat1"] - BOX["lat0"],
                           fill=False, edgecolor="crimson", linewidth=2.0,
                           linestyle="--", zorder=5))
    ax.annotate("analysis domain\n25–33°E, 15–22°S",
                xy=(25.15, -21.82), fontsize=9, color="crimson", zorder=6,
                va="bottom")

    for name, lon, lat in TOWNS:
        outside = lat < BOX["lat0"]
        ax.plot(lon, lat, "o", ms=4.5,
                color="crimson" if outside else "black", zorder=6)
        ax.annotate(name + (" (outside the domain)" if outside else ""),
                    xy=(lon, lat), xytext=(4, 4), textcoords="offset points",
                    fontsize=8.4, color="crimson" if outside else "black", zorder=6)

    # Harare and Bulawayo are both city-provinces and towns; labelling them twice
    # overprints the marker.
    for r in prov.itertuples():
        if r.NAME_1 in ("Harare", "Bulawayo"):
            continue
        c = r.geometry.representative_point()
        ax.annotate(r.NAME_1, xy=(c.x, c.y), ha="center", fontsize=7.6,
                    color="0.2", zorder=5)

    ax.set_xlim(24.6, 33.9)
    ax.set_ylim(-22.9, -14.8)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°S)")
    ax.set_title("Zimbabwe: relief, provinces and the analysis domain")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Elevation (m)")

    # A scale bar, since the map is in degrees. One degree of longitude at 19 S
    # is about 105 km.
    km_per_deg = 111.32 * np.cos(np.deg2rad(19.0))
    L = 200.0 / km_per_deg
    x0, y0 = 25.2, -22.55
    ax.plot([x0, x0 + L], [y0, y0], color="black", lw=2.6, zorder=7)
    ax.annotate("200 km", xy=(x0 + L / 2, y0), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=8.4, zorder=7)
    ax.annotate("N", xy=(33.5, -15.3), ha="center", fontsize=12,
                fontweight="bold", zorder=7)
    ax.annotate("↑", xy=(33.5, -15.9), ha="center", fontsize=15, zorder=7)

    ax.set_aspect(1.0 / np.cos(np.deg2rad(19.0)))
    fig.tight_layout()
    out = os.path.join(FIG, "00_study_area.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    inside = nat.geometry.iloc[0].bounds
    print("national bounds: lon %.2f to %.2f, lat %.2f to %.2f"
          % (inside[0], inside[2], inside[1], inside[3]))
    print("domain southern edge is %.2f, country reaches %.2f -> %.2f degrees cut off"
          % (BOX["lat0"], inside[1], BOX["lat0"] - inside[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
