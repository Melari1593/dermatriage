"""Particiones train/val/test sin fuga entre pacientes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Agrega la columna `split` (train/val/test), aprox. 70/15/15.

    Un paciente (o caso SCIN) cae en una sola partición.
    La partición conserva la proporción de clases de riesgo.
    """
    df = df.reset_index(drop=True).copy()
    idx = np.arange(len(df))
    outer = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    trainval, test = next(outer.split(idx, df["risk"], groups=df["group_key"]))
    sub = df.iloc[trainval]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = next(inner.split(np.arange(len(sub)), sub["risk"], groups=sub["group_key"]))
    df["split"] = ""
    df.loc[trainval[tr], "split"] = "train"
    df.loc[trainval[va], "split"] = "val"
    df.loc[test, "split"] = "test"
    assert_no_leakage(df)
    return df


def assert_no_leakage(df: pd.DataFrame) -> None:
    groups = df.groupby("group_key")["split"].nunique()
    leaked = groups[groups > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} pacientes aparecen en más de una partición.")
