"""Acquire the four auxiliary datasets Section 3.9's suitability analysis needs.

None of these feeds the ML downscaling; they are consumed only by the
multi-criteria evaluation, which is why they were left until now. Section 3.3.4
of Chapter 3 names them and records that only SRTM had been acquired.

    ESA CCI Land Cover 300 m   CDS, area-subset to the domain (a few MB, not the
                               multi-GB global file - the dataset exposes an
                               `area` widget, so ask for Zimbabwe only)
    OpenStreetMap Zimbabwe     Geofabrik country extract, roads + power lines
    WorldPop 100 m             constrained 2020 population count
    GADM 4.1 boundaries        national outline and admin levels, as a GeoPackage
    WDPA protected areas       protectedplanet.net, which requires accepting its
                               terms in a browser - not scriptable here

CM SAF SARAH is deliberately NOT here. It is not a suitability layer, and its
acquisition is not a plain GET: you place an order on the CM SAF Web User
Interface, they extract server-side, and a per-order URL and password arrive by
email and expire after seven days. See fetch_sarah_order.py, which takes the
order number from that email.

GADM earns its place twice over. Table 3.5 names it for the proximity-to-
settlements criterion, and the national outline is separately needed as an
exclusion mask: WorldPop is clipped to Zimbabwe's border while the target grid
is a rectangle, so the north-west of the box falls in Zambia and Mozambique.
Without the mask those cells enter the weighted linear combination as nodata
holes instead of clean exclusions.

Downloads resume rather than restart, so an interrupted run on a slow link picks
up where it stopped.

    python download_suitability_data.py             # everything available
    python download_suitability_data.py --only osm worldpop
"""

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data/raw")

# The fine target grid, from ml_validation_dataset.nc, with half a degree of
# margin so no suitability layer needs edge extrapolation.
NORTH, WEST, SOUTH, EAST = -14.5, 24.5, -22.5, 33.5

GEOFABRIK = "https://download.geofabrik.de/africa/zimbabwe-latest.osm.pbf"
WORLDPOP = ("https://data.worldpop.org/GIS/Population/"
            "Global_2000_2020_Constrained/2020/maxar_v1/ZWE/"
            "zwe_ppp_2020_constrained.tif")
# GeoPackage rather than the smaller shapefile zip: one file carries every admin
# level, and nothing gets truncated to the shapefile's 10-character field names.
GADM = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_ZWE.gpkg"

# Both must be accepted once, on the dataset page, before the API will serve it.
LANDCOVER_LICENCES = ["satellite-land-cover", "vito-proba-v"]
LANDCOVER_PAGE = ("https://cds.climate.copernicus.eu/datasets/"
                  "satellite-land-cover?tab=download")


def human(n):
    return "%.1f MB" % (n / 1048576.0) if n else "unknown size"


def fetch(url, dest, label):
    """Resumable GET. Returns True when dest ends up complete."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0

    head = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(head, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
    except Exception as exc:
        print("  ! could not reach %s: %s" % (url, exc))
        return False

    if total and have == total:
        print("  %s already complete (%s)" % (label, human(total)))
        return True
    if have > total > 0:
        print("  %s local copy is larger than the source; re-fetching" % label)
        os.remove(dest)
        have = 0

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", "bytes=%d-" % have)
        print("  resuming %s at %s of %s" % (label, human(have), human(total)))
    else:
        print("  downloading %s (%s)" % (label, human(total)))

    try:
        with urllib.request.urlopen(req, timeout=120) as r, \
                open(dest, "ab" if have else "wb") as fh:
            got = have
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if total:
                    pct = 100.0 * got / total
                    sys.stdout.write("\r    %5.1f%%  %s" % (pct, human(got)))
                    sys.stdout.flush()
    except Exception as exc:
        print("\n  ! %s failed: %s (re-run to resume)" % (label, exc))
        return False

    print("\r    done: %s%s" % (human(os.path.getsize(dest)), " " * 12))
    return True


def get_osm():
    print("\nOpenStreetMap Zimbabwe (Geofabrik) - roads and power lines")
    return fetch(GEOFABRIK, os.path.join(RAW, "osm/zimbabwe-latest.osm.pbf"),
                 "zimbabwe-latest.osm.pbf")


def get_worldpop():
    print("\nWorldPop 100 m constrained population, 2020")
    return fetch(WORLDPOP, os.path.join(RAW, "worldpop/zwe_ppp_2020_constrained.tif"),
                 "zwe_ppp_2020_constrained.tif")


def get_gadm():
    print("\nGADM 4.1 administrative boundaries (GeoPackage)")
    return fetch(GADM, os.path.join(RAW, "gadm/gadm41_ZWE.gpkg"), "gadm41_ZWE.gpkg")


def get_landcover(year="2022", version="v2_1_1"):
    print("\nESA CCI Land Cover 300 m, %s (%s), subset to the domain" % (year, version))
    try:
        import cdsapi
    except ImportError:
        print("  ! cdsapi is not installed")
        return False

    dest = os.path.join(RAW, "landcover/esacci_lc_%s_zimbabwe.zip" % year)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("  already present: %s (%s)" % (os.path.basename(dest),
                                              human(os.path.getsize(dest))))
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        cdsapi.Client().retrieve(
            "satellite-land-cover",
            {"variable": "all", "year": [year], "version": [version],
             "area": [NORTH, WEST, SOUTH, EAST], "format": "zip"},
            dest)
    except Exception as exc:
        msg = str(exc)
        print("  ! request failed: %s" % msg[:400])
        if "licence" in msg.lower() or "licen" in msg.lower() or "403" in msg:
            print("\n  This dataset needs two licences accepted once, in a browser,")
            print("  on your own CDS account - it is not something a script should")
            print("  click through for you:")
            for lic in LANDCOVER_LICENCES:
                print("      - %s" % lic)
            print("  Accept them at the bottom of:\n      %s" % LANDCOVER_PAGE)
            print("  then re-run:  python download_suitability_data.py --only landcover")
        return False
    print("  done: %s" % human(os.path.getsize(dest)))
    return True


def note_wdpa():
    print("\nWDPA / ZimParks protected areas")
    print("  Not scriptable: protectedplanet.net serves country downloads only")
    print("  after its terms are accepted in a browser, and creating accounts or")
    print("  accepting terms on your behalf is out of scope for this script.")
    print("  Download 'Zimbabwe' (shapefile or GeoPackage) from:")
    print("      https://www.protectedplanet.net/country/ZWE")
    print("  and unzip it into  data/raw/protected_areas/")
    print("  The OSM extract above also carries boundary=protected_area polygons,")
    print("  which are a usable cross-check but are not the statutory source")
    print("  Section 3.3.4 specifies.")
    return None


JOBS = {"osm": get_osm, "worldpop": get_worldpop, "gadm": get_gadm,
        "landcover": get_landcover, "wdpa": note_wdpa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(JOBS), default=sorted(JOBS))
    args = ap.parse_args()

    results = {name: JOBS[name]() for name in args.only}

    print("\n" + "=" * 68)
    for name, ok in results.items():
        state = {True: "ready", False: "FAILED", None: "needs your browser"}[ok]
        print("  %-10s %s" % (name, state))
    print("=" * 68)
    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
