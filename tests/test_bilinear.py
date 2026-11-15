import math

import pandas as pd
import pytest

from common import CHIRPS_CSV, COORDS_CSV, ROOT

CHECK_CSV = ROOT / "data" / "external" / "chirps_bilinear_check.csv"
# Earth Engine evaluates the bilinear surface at the centre of the 100 m sampling pixel
# (scale = 100 m in EPSG:4326, grid anchored at the raster origin), not at the exact point.
SAMPLE_DEG = 100 / (2 * math.pi * 6378137 / 360)
TOLERANCE_MM = 0.002  # float32 rounding


def bilinear(cell, lon, lat):
    fx = (lon - cell.origin_x) / cell.size_x - 0.5 - cell.col
    fy = (cell.origin_y - lat) / cell.size_y - 0.5 - cell.row
    assert 0 <= fx <= 1 and 0 <= fy <= 1
    return (cell.v00 * (1 - fx) * (1 - fy) + cell.v01 * fx * (1 - fy)
            + cell.v10 * (1 - fx) * fy + cell.v11 * fx * fy)


@pytest.fixture(scope="module")
def cases():
    check = pd.read_csv(CHECK_CSV)
    coords = pd.read_csv(COORDS_CSV).set_index("station")
    export = pd.read_csv(CHIRPS_CSV)
    export = export[export["date"] == check["date"].iloc[0]].set_index("station")["precip_mm"]
    return check, coords, export


def test_export_equals_manual_bilinear_from_four_neighbours(cases):
    check, coords, export = cases
    for cell in check.itertuples():
        lon, lat = coords.loc[cell.station, ["longitude", "latitude"]]
        centre_lon = cell.origin_x + (math.floor((lon - cell.origin_x) / SAMPLE_DEG) + 0.5) * SAMPLE_DEG
        centre_lat = cell.origin_y - (math.floor((cell.origin_y - lat) / SAMPLE_DEG) + 0.5) * SAMPLE_DEG
        assert export[cell.station] == pytest.approx(bilinear(cell, centre_lon, centre_lat), abs=TOLERANCE_MM), cell.station


def test_export_is_not_nearest_cell(cases):
    check, coords, export = cases
    misses = []
    for cell in check.itertuples():
        lon, lat = coords.loc[cell.station, ["longitude", "latitude"]]
        fx = (lon - cell.origin_x) / cell.size_x - 0.5 - cell.col
        fy = (cell.origin_y - lat) / cell.size_y - 0.5 - cell.row
        nearest = [[cell.v00, cell.v01], [cell.v10, cell.v11]][round(fy)][round(fx)]
        misses.append(abs(export[cell.station] - nearest))
    assert min(misses) > 10 * TOLERANCE_MM
