import pytest

from conftest import load_script

# Panel Ridge with the suspect segment excluded, information cutoff t: pinned from this repository's own code.
# BRG20 does not degrade (ratio below 1); protocol 2.3 still says every ratio exceeds 1.
PANEL_RATIOS = {"BRG6": 2.68, "BRG5": 2.43, "BRG3": 1.95, "BRG19": 1.17, "BRG20": 0.99}


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
def test_drought_degrades_every_per_station_model(drift, data):
    _, result = drift
    rows = result[(result["data"] == data) & (result["mode"] == "per_station")]
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
