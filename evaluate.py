"""Shared evaluation rules, so every model (baseline or ML) is judged the same honest way.

Time split (never shuffled: the model must not see the future):
    train       up to 2021-12-31   fit anything here
    validation  2022 and 2023      choose between models / settings here
    test        2024 onwards       look at this ONCE, at the end

A forecast "at horizon h" for a target day D is made using only data up to day D - h.
"""
import numpy as np
import pandas as pd

TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"
HORIZONS = list(range(1, 11))


def load_daily(path=None) -> pd.Series:
    from config import DATA
    df = pd.read_csv(path or DATA / "energy_daily.csv", index_col="date", parse_dates=["date"])
    s = df["energy_kwh"].asfreq("D")
    if s.isna().any():
        raise ValueError(f"{int(s.isna().sum())} days missing from the daily series; rebuild the dataset")
    return s


def periods(s: pd.Series) -> dict[str, pd.DatetimeIndex]:
    idx = s.index
    return {"train": idx[idx <= TRAIN_END],
            "validation": idx[(idx > TRAIN_END) & (idx <= VAL_END)],
            "test": idx[idx > VAL_END]}


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    e = (y_pred - y_true).to_numpy()
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e))}


def horizon_table(s: pd.Series, forecasters: dict, split: str, horizons=HORIZONS) -> pd.DataFrame:
    """MAE (kWh/day) per model and horizon. `forecasters[name](s, target_dates, h)` returns a Series
    indexed by target_dates and may only use s up to (target - h)."""
    targets = periods(s)[split]
    rows = {}
    for name, f in forecasters.items():
        rows[name] = {h: metrics(s.loc[targets], f(s, targets, h))["mae"] for h in horizons}
    return pd.DataFrame(rows).T.rename_axis("model")
