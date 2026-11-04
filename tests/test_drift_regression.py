import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def drift(gwt):
    module = load_script("80_drift_regression.py")
    return module, module.evaluate(gwt)


def test_drift_table_is_reproduced_within_tolerance(drift):
    module, result = drift
    worst = result[result["data"] == "with_suspect"].groupby("mode")["max_abs_diff_m"].max()
    assert (worst <= module.MAE_TOLERANCE_M).any(), (
        f"Tabel drift tidak tereproduksi; selisih terbesar per konfigurasi {worst.round(3).to_dict()} m, "
        f"toleransi {module.MAE_TOLERANCE_M} m. Laporkan, jangan menyetel konfigurasi."
    )


@pytest.mark.parametrize("data", ["with_suspect", "suspect_excluded"])
@pytest.mark.parametrize("mode", ["per_station", "panel"])
def test_drought_degrades_static_model(drift, data, mode):
    _, result = drift
    rows = result[(result["data"] == data) & (result["mode"] == mode)]
    assert (rows["ratio"] > 1).all()


def test_suspect_exclusion_changes_only_brg3(drift):
    _, result = drift
    keys = ["mode", "station"]
    kept = result[result["data"] == "with_suspect"].set_index(keys)["mae_drought_m"]
    clean = result[result["data"] == "suspect_excluded"].set_index(keys)["mae_drought_m"]
    changed = {station for (_, station), differs in (kept != clean).items() if differs}
    assert changed == {"BRG3"}
    brg3 = result[(result["data"] == "suspect_excluded") & (result["station"] == "BRG3")]
    assert (brg3["n_drought"] == 9).all()
