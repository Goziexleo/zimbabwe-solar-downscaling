import earthaccess

# Use 'netrc' strategy since credentials were already cached by download_merra2.py
# (persist=True wrote them to ~/.netrc); avoids requiring an interactive browser login.
auth = earthaccess.login(strategy="netrc", persist=True)

# Search: 2011-2024 validation period (Section 3.6.2)
results = earthaccess.search_data(
    short_name="M2TMNXAER",
    temporal=("2011-01-01", "2024-12-31"),
    bounding_box=(25.0, -22.5, 33.5, -15.0),
)

# Download
files = earthaccess.download(results, local_path="./data/raw/merra2_validation")
print(f"Downloaded {len(files)} files to ./data/raw/merra2_validation")
