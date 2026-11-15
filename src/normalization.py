"""Normalization frozen from the training fold of T1 (rule 2)."""
import json

import numpy as np
import pandas as pd

T1 = "T1_transisi_2018"


class FrozenScaler:
    def __init__(self, means: dict, stds: dict):
        self.means, self.stds = means, stds

    @classmethod
    def fit(cls, rows: pd.DataFrame, columns: list[str]) -> "FrozenScaler":
        source = rows[(rows["task"] == T1) & (rows["fold"] == "train")]
        stds = source[columns].std(ddof=0)
        if (stds == 0).any():
            raise ValueError(f"Simpangan baku nol di T1: {list(stds[stds == 0].index)}")
        return cls(source[columns].mean().to_dict(), stds.to_dict())

    def transform(self, values, column: str) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.means[column]) / self.stds[column]

    def inverse(self, values, column: str) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.stds[column] + self.means[column]

    def save(self, path) -> None:
        path.write_text(json.dumps({"means": self.means, "stds": self.stds}, indent=1, sort_keys=True) + "\n")


class ExpandingScaler:
    """Diagnostic D2 (exploratory): row-specific statistics from rows issued on or before
    max(t, T1_end), where T1_end is the last issue date of the T1 training fold. Rows before
    T1_end get exactly the frozen T1 statistics; later rows only ever see the past."""

    def __init__(self, means: dict, stds: dict):
        self.means, self.stds = means, stds  # column -> array aligned with the rows it was fit on

    @classmethod
    def fit(cls, rows: pd.DataFrame, columns: list[str]) -> "ExpandingScaler":
        t1 = rows[(rows["task"] == T1) & (rows["fold"] == "train")]
        t1_end = t1["date"].max()
        frozen = FrozenScaler.fit(rows, columns)
        by_date = rows.groupby("date")[columns]
        counts = by_date.size().cumsum()
        sums = by_date.sum().cumsum()
        squares = rows[columns].pow(2).groupby(rows["date"]).sum().cumsum()
        mean_at = sums.div(counts, axis=0)
        std_at = np.sqrt((squares.div(counts, axis=0) - mean_at**2).clip(lower=0))
        means, stds = {}, {}
        late = (rows["date"] > t1_end).to_numpy()
        for column in columns:
            mean = np.full(len(rows), frozen.means[column])
            std = np.full(len(rows), frozen.stds[column])
            mean[late] = mean_at[column].reindex(rows.loc[late, "date"]).to_numpy()
            std[late] = std_at[column].reindex(rows.loc[late, "date"]).to_numpy()
            means[column], stds[column] = mean, std
        return cls(means, stds)

    def _stats(self, values: np.ndarray, column: str):
        mean, std = self.means[column], self.stds[column]
        if values.ndim == 2:
            return mean[:, None], std[:, None]
        return mean, std

    def transform(self, values, column: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        mean, std = self._stats(values, column)
        return (values - mean) / std

    def inverse(self, values, column: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        mean, std = self._stats(values, column)
        return values * std + mean

    def save(self, path) -> None:
        raise NotImplementedError("Statistik per baris tidak disimpan")
