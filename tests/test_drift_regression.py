import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def drift(gwt):
    module = load_script("80_drift_regression.py")
    return module, module.evaluate(gwt)


def test_drift_table_is_reproduced_within_tolerance(drift):
    module, result = drift
    worst = result.groupby("mode")["max_abs_diff_m"].max()
    assert (worst <= module.MAE_TOLERANCE_M).any(), (
        f"Tabel drift tidak tereproduksi; selisih terbesar per konfigurasi {worst.round(3).to_dict()} m, "
        f"toleransi {module.MAE_TOLERANCE_M} m. Laporkan, jangan menyetel konfigurasi."
    )


@pytest.mark.parametrize("mode", ["per_station", "panel"])
def test_drought_degrades_static_model(drift, mode):
    _, result = drift
    rows = result[result["mode"] == mode].set_index("station")
    assert (rows["ratio"] > 1).all()
    assert rows["ratio"].idxmax() == "BRG3"
