import earthaccess

# Use 'interactive' strategy to open a browser window for login.
# 'persist=True' will save your session so you don't have to do this every time.
auth = earthaccess.login(strategy="interactive", persist=True)

# Search
results = earthaccess.search_data(
    short_name="M2TMNXAER",
    temporal=("1985-01-01", "2010-12-31"),
    bounding_box=(25.0, -22.5, 33.5, -15.0),
)

# Download
files = earthaccess.download(results, local_path="./data/raw/merra2")