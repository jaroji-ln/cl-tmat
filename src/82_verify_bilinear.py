"""Extract the four CHIRPS cells around each station for one day, to verify bilinear sampling.

Downloads the CHIRPS v2.0 daily raster (if absent) into data/raw/chirps_tifs/, which is not
committed, and writes the neighbouring cell values to a small committed table that
tests/test_bilinear.py checks against the Earth Engine export.

Output: data/external/chirps_bilinear_check.csv
"""
import gzip
import math
import shutil
import sys
import urllib.request

import pandas as pd
import tifffile

from common import COORDS_CSV, ROOT

CHECK_DATE = "2018-06-24"  # rain above 5 mm at all eight stations, so methods differ visibly
URL = "https://data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/tifs/p05/{y}/chirps-v2.0.{y}.{m}.{d}.tif.gz"
RASTER_DIR = ROOT / "data" / "raw" / "chirps_tifs"
CHECK_CSV = ROOT / "data" / "external" / "chirps_bilinear_check.csv"


def raster_path():
    y, m, d = CHECK_DATE.split("-")
    tif = RASTER_DIR / f"chirps-v2.0.{y}.{m}.{d}.tif"
    if not tif.exists():
        RASTER_DIR.mkdir(parents=True, exist_ok=True)
        gz = tif.with_suffix(".tif.gz")
        urllib.request.urlretrieve(URL.format(y=y, m=m, d=d), gz)
        with gzip.open(gz) as source, open(tif, "wb") as target:
            shutil.copyfileobj(source, target)
    return tif


def main() -> int:
    page = tifffile.TiffFile(raster_path()).pages[0]
    tags = {tag.name: tag.value for tag in page.tags.values()}
    size_x, size_y = tags["ModelPixelScaleTag"][:2]
    origin_x, origin_y = tags["ModelTiepointTag"][3:5]
    grid = page.asarray()
    records = []
    for station in pd.read_csv(COORDS_CSV).itertuples():
        col = math.floor((station.longitude - origin_x) / size_x - 0.5)
        row = math.floor((origin_y - station.latitude) / size_y - 0.5)
        cells = grid[row : row + 2, col : col + 2]
        records.append({
            "station": station.station, "date": CHECK_DATE, "row": row, "col": col,
            "origin_x": origin_x, "origin_y": origin_y, "size_x": size_x, "size_y": size_y,
            "v00": cells[0, 0], "v01": cells[0, 1], "v10": cells[1, 0], "v11": cells[1, 1],
        })
    pd.DataFrame(records).to_csv(CHECK_CSV, index=False, float_format="%.8g")
    print(f"Ditulis: {CHECK_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
