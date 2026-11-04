import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import RAW_XLSX, STATIONS  # noqa: E402


def load_script(filename: str):
    """Scripts start with a digit, so they cannot be imported by name."""
    spec = importlib.util.spec_from_file_location(filename.replace(".py", "").lstrip("0123456789_"), SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def prepare():
    return load_script("01_prepare_gwt.py")


@pytest.fixture(scope="session")
def features_module():
    return load_script("02_build_features.py")


@pytest.fixture(scope="session")
def raw(prepare):
    if not RAW_XLSX.exists():
        pytest.skip("data/raw/mmc1.xlsx belum ada")
    return prepare.load_raw()


@pytest.fixture(scope="session")
def gwt(prepare, raw):
    return prepare.clean(raw)


@pytest.fixture(scope="session")
def rain_stub():
    """In-memory rain for testing feature logic only. Never written to disk or used for results."""
    dates = pd.date_range("2018-01-01", "2020-01-31")
    pattern = (np.sin(np.arange(len(dates)) / 3.0) > 0.2) * 5.0
    chirps = pd.concat([pd.DataFrame({"station": s, "date": dates, "precip_mm": pattern}) for s in STATIONS], ignore_index=True)
    climatology = pd.concat([pd.DataFrame({"station": s, "doy": range(1, 367), "precip_clim_mm": 2.0}) for s in STATIONS], ignore_index=True)
    return chirps, climatology
