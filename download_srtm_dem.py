import os
import elevation
import rioxarray
from shapely.geometry import box

# --- Configuration ---
output_dir = os.path.abspath("./data/raw/srtm")
os.makedirs(output_dir, exist_ok=True)
output_dem = os.path.join(output_dir, "zimbabwe_dem_90m.tif")

# Zimbabwe study domain bounding box with 0.25° buffer (Section 3.2.1)[cite: 1]
min_lon, min_lat, max_lon, max_lat = 24.75, -22.75, 33.75, -14.75
bounds = (min_lon, min_lat, max_lon, max_lat)

print("Ensuring SRTM tiles are seeded in cache...")
gen_bounds = elevation.datasource.build_bounds(bounds, margin='0')
elevation.datasource.seed(bounds=gen_bounds, product='SRTM3', max_download_tiles=300)

# Locate the generated VRT file inside the local elevation cache
cache_dir = os.path.expanduser('~/Library/Caches/elevation/SRTM3')
vrt_path = os.path.join(cache_dir, 'SRTM3.vrt')

print(f"Clipping from cache VRT to target path...")
# Open the VRT with rioxarray, clip to the Zimbabwe bounding box, and save
with rioxarray.open_rasterio(vrt_path, masked=True) as src:
    geoms = [box(min_lon, min_lat, max_lon, max_lat)]
    clipped = src.rio.clip(geoms, crs="EPSG:4326", from_disk=True)
    clipped.rio.to_raster(output_dem)

print(f"Success! DEM saved to: {output_dem}")