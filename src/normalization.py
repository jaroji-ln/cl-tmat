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
