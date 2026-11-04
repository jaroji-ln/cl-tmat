import pandas as pd
import pytest

from common import AUDIT_CSV, CHIRPS_CSV, FEATURES_CSV
from conftest import load_script

# Numbers quoted in docs/protokol-riset-cl-tmat.md; they must come from src/81_data_audit.py.
PROTOCOL = {
    "brg3_line_days": (45, 0),
    "brg3_line_residual_mm": (0.28, 0.005),
    "brg3_outside_over_line_sd": (115, 1),
    "same_window_smoothest_ratio_min": (11, 1),
    "same_window_smoothest_ratio_max": (56, 1),
    "brg3_rain_during_line_mm": (58, 0.5),
    "brg3_jump_m": (-0.254, 0.0005),
    "T4_kemarau_2019_rows": (216, 0),
}


@pytest.fixture(scope="module")
def audit(gwt):
    if not (CHIRPS_CSV.exists() and FEATURES_CSV.exists()):
        pytest.skip("CHIRPS atau features.csv belum ada")
    module = load_script("81_data_audit.py")
    assert module.main() == 0
    return pd.read_csv(AUDIT_CSV).set_index("metric")["value"]


@pytest.mark.parametrize("metric,expected", PROTOCOL.items())
def test_protocol_number_is_rebuilt_by_script(audit, metric, expected):
    value, tolerance = expected
    assert audit[metric] == pytest.approx(value, abs=tolerance)


def test_kubu_raya_t2_deficit_is_negative_as_protocol_states(audit):
    values = audit[[f"T2_deficit_mean_{s}_mm" for s in ["BRG17", "BRG18", "BRG19", "BRG20"]]]
    assert values.min() == pytest.approx(-67, abs=0.5) and values.max() == pytest.approx(-44, abs=0.5)
