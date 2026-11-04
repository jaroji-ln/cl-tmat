import pytest

from conftest import load_script

# Protocol 2.3, panel Ridge with the suspect segment excluded: pinned from this repository's own code.
PANEL_RATIOS = {"BRG6": 2.94, "BRG5": 2.71, "BRG3": 2.16, "BRG19": 1.31, "BRG20": 1.08}


@pytest.fixture(scope="module")
def drift(gwt):
    module = load_script("80_drift_regression.py")
    return module, module.evaluate(gwt)


@pytest.mark.parametrize("station,expected", PANEL_RATIOS.items())
def test_panel_ratio_matches_protocol(drift, station, expected):
    _, result = drift
    row = result[(result["data"] == "suspect_excluded") & (result["mode"] == "panel") & (result["station"] == station)]
    assert row["ratio"].item() == pytest.approx(expected, abs=0.005)


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
