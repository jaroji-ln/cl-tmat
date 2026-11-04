import pandas as pd
import pytest

from splits import TASK_ORDER, split_by_target_date, split_by_task


@pytest.fixture(scope="module")
def table(features_module, gwt, rain_stub):
    return features_module.build_features(gwt, *rain_stub)


@pytest.mark.parametrize("position", range(1, len(TASK_ORDER)))
def test_task_split_has_no_temporal_leak(table, position):
    train, test = split_by_task(table, TASK_ORDER[:position], TASK_ORDER[position])
    assert len(train) > 0
    if len(test):
        assert train["target_date"].max() < test["target_date"].min()
        assert not set(train.index) & set(test.index)


def test_cutoff_split_has_no_temporal_leak(table):
    train, test = split_by_target_date(table, "2019-01-01")
    assert train["target_date"].max() < pd.Timestamp("2019-01-01") <= test["target_date"].min()
    assert len(train) + len(test) == len(table)


def test_training_on_a_later_task_is_refused(table):
    with pytest.raises(ValueError):
        split_by_task(table, ["T3_basah_201819"], "T2_kemarau_2018")
    with pytest.raises(ValueError):
        split_by_task(table, ["T2_kemarau_2018"], "T2_kemarau_2018")


def test_splits_are_deterministic(table):
    first, _ = split_by_task(table, TASK_ORDER[:2], TASK_ORDER[2])
    second, _ = split_by_task(table, TASK_ORDER[:2], TASK_ORDER[2])
    pd.testing.assert_frame_equal(first, second)
