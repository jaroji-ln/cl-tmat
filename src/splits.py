"""Temporal data splits. Rows are ordered by target date; nothing here is random."""
import pandas as pd

from common import REGIMES

TASK_ORDER = [label for label, *_ in REGIMES]


def split_by_target_date(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on targets before cutoff, test on targets on or after it."""
    cut = pd.Timestamp(cutoff)
    return df[df["target_date"] < cut], df[df["target_date"] >= cut]


def split_by_task(df: pd.DataFrame, train_tasks: list[str], test_task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier tasks and test on a later one; any other order raises."""
    test_rank = TASK_ORDER.index(test_task)
    if not train_tasks or max(TASK_ORDER.index(t) for t in train_tasks) >= test_rank:
        raise ValueError(f"Task latih {train_tasks} harus seluruhnya sebelum task uji {test_task}")
    return df[df["task"].isin(train_tasks)], df[df["task"] == test_task]
