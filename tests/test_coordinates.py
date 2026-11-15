import re
from pathlib import Path

import pandas as pd
import pytest

from common import COORDS_CSV, ROOT

VALIDATION_CSV = ROOT / "results" / "coord_validation.csv"
FIGURE = ROOT / "data" / "external" / "taufik2022_fig1.jpg"


def test_earth_engine_script_uses_the_coordinate_file():
    script = (ROOT / "src" / "00_fetch_chirps.js").read_text()
    points = {s: (float(lon), float(lat)) for lon, lat, s in re.findall(r"Point\(\[([0-9.]+), (-?[0-9.]+)\]\), \{station: '(\w+)'\}", script)}
    coords = pd.read_csv(COORDS_CSV).set_index("station")
    assert set(points) == set(coords.index)
    for station, (lon, lat) in points.items():
        assert (lon, lat) == pytest.approx((coords.loc[station, "longitude"], coords.loc[station, "latitude"]), abs=1e-9)


def test_published_stations_keep_their_published_coordinates():
    coords = pd.read_csv(COORDS_CSV).set_index("station")
    assert coords.loc["BRG6", ["longitude", "latitude"]].tolist() == [103.9693, -1.44281]
    assert coords.loc["BRG18", ["longitude", "latitude"]].tolist() == [109.51923, -0.27979]
    assert (coords["source"] == "estimated").sum() == 6


def test_validation_errors_come_from_the_script():
    if not FIGURE.exists():
        pytest.skip("gambar masukan belum ada")
    errors = pd.read_csv(VALIDATION_CSV).set_index("station")["error_m"]
    assert errors.to_dict() == {"BRG6": 214.3, "BRG18": 208.1}
