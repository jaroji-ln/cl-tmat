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


def add_folds(df: pd.DataFrame, train_fraction: float = 0.7) -> pd.DataFrame:
    """Split each task once by target date; one date never falls in both folds."""
    out = df.copy()
    out["fold"] = "train"
    for task in TASK_ORDER:
        dates = out.loc[out["task"] == task, "target_date"].sort_values()
        cut = dates.iloc[int(len(dates) * train_fraction)]
        out.loc[(out["task"] == task) & (out["target_date"] >= cut), "fold"] = "holdout"
    return out


def available_for_training(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Training rows whose target had been observed by as_of. Holdouts stay untouched."""
    return df[(df["fold"] == "train") & (df["target_date"] <= as_of)]
