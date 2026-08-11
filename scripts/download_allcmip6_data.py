import os
import sys
import re
import requests

# esgf-node.llnl.gov's legacy Solr-based esg-search API was shut down in
# July 2025 as part of ESGF's move to the new "ESGF-NG" infrastructure
# (rolling out around mid-2026). The European/AU nodes still speak the
# same esg-search REST schema this script uses, so DKRZ is used here
# instead. Run a small test query before the full loop -- if this node
# also stops responding, check https://esgf.github.io/nodes.html for the
# current list, or consider the actively maintained `esgpull` or
# `intake-esgf` clients instead of raw esg-search calls.
BASE_URL = "https://esgf-data.dkrz.de/esg-search/search"

# Chapter 3, Section 3.3.1 / Table 3.1
MODELS = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
VARIABLES = ["rsds", "clt", "huss", "tas", "ps", "od550aer"]
EXPERIMENTS = ["historical", "ssp245", "ssp585"]

# CNRM-CM6-1 was published with forcing_index=2 (variant r1i1p1f2), unlike
# the other two models, which use r1i1p1f1. A single hardcoded variant
# label silently returns zero files for every CNRM-CM6-1 query.
VARIANT_LABELS = {
    "CNRM-CM6-1": "r1i1p1f2",
    "MPI-ESM1-2-HR": "r1i1p1f1",
    "ACCESS-CM2": "r1i1p1f1",
}

def should_download(filename, exp):
    """
    Matches CMIP6 filename timestamps to the temporal windows in Chapter 3,
    Section 3.3.1: the historical run covers training (1985-2010), and its
    tail end (2011-2014, where CMIP6 "historical" runs end under the
    protocol) is kept too so predictor coverage is continuous into the
    validation period (Section 3.6.2: validation is 2011-2024). ssp245 and
    ssp585 pick up from 2015, covering the rest of validation (2015-2024)
    plus the future projections (2026-2100).

    There is no "day" frequency case here: Section 3.3.1 states CMIP6
    predictor extraction uses monthly mean fields only. The daily-
    resolution validation products in Section 3.3.3 (SARAH-2, NSRDB) are
    satellite products, not CMIP6 GCM output, and are not retrieved from
    ESGF at all.
    """
    match = re.search(r'_(\d+)-(\d+)\.nc$', filename)
    if not match:
        return True  # Fallback if filename structure deviates

    start_str, end_str = match.groups()
    start_year = int(start_str[:4])
    end_year = int(end_str[:4])

    if exp == "historical":
        return not (end_year < 1985 or start_year > 2014)
    elif exp in ["ssp245", "ssp585"]:
        return not (end_year < 2015 or start_year > 2100)

    return False

print("====================================================")
print("RUNNING: Monthly CMIP6 predictor dataset (1985-2100)")
print("====================================================")

for model in MODELS:
    variant = VARIANT_LABELS[model]
    for exp in EXPERIMENTS:
        for var in VARIABLES:

            params = {
                "type": "File",
                "project": "CMIP6",
                "source_id": model,
                "experiment_id": exp,
                "variable_id": var,
                "frequency": "mon",
                "variant_label": variant,
                "latest": "true",
                "distrib": "true",
                "format": "application/solr+json",
                "limit": 10000
            }

            try:
                response = requests.get(BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                docs = data.get("response", {}).get("docs", [])
                urls = []

                for doc in docs:
                    for url_str in doc.get("url", []):
                        parts = url_str.split("|")
                        if len(parts) >= 3 and parts[2] == "HTTPServer":
                            urls.append(parts[0])

                if not urls:
                    continue

                for url in urls:
                    filename = url.split("/")[-1]

                    if should_download(filename, exp):
                        # FIX 1: Deduplicate mirror downloads. If the file is already fully 
                        # downloaded, skip processing the remaining replica node urls.
                        if os.path.exists(filename):
                            print(f" -> File {filename} already exists locally. Skipping remaining mirrors.")
                            continue

                        print(f"Approved for download: {filename}")
                        
                        # FIX 2: Mitigate silent server drops and network freezes
                        # --speed-limit 10240 --speed-time 60: Abort if download falls below 10KB/s for 60s
                        # --retry 3: Attempt 3 standard reconnections before dropping out
                        # -L: Handle internal server redirects automatically
                        os.system(f'curl -L -C - --speed-limit 10240 --speed-time 60 --retry 3 -O "{url}"')

            except Exception as e:
                print(f" -> Error connecting to ESGF node for [{model}|{exp}|{var}]: {e}", file=sys.stderr)

print("\nMonthly CMIP6 predictor download loop matched to Chapter 3, Section 3.3.1.")