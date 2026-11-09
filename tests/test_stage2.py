import numpy as np
import pandas as pd
import pytest

import baselines
from evaluate import horizon_table, metrics, periods


def make_series(seed=0):
    idx = pd.date_range("2015-01-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(seed)
    season = 17 + 11 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365.25)
    noise = pd.Series(rng.normal(0, 4, len(idx))).rolling(3, min_periods=1).mean().to_numpy()
    return pd.Series(np.clip(season + noise, 0.5, None), index=idx)


def test_split_is_chronological_and_disjoint():
    p = periods(make_series())
    assert p["train"].max() < p["validation"].min() <= p["validation"].max() < p["test"].min()


@pytest.mark.parametrize("name", list(baselines.ALL))
def test_baselines_cannot_see_the_future(name):
    """Scramble everything AFTER the forecast origin; the prediction must not move."""
    s = make_series()
    target = pd.DatetimeIndex(["2024-07-15"])
    h = 5
    origin = target[0] - pd.Timedelta(days=h)
    a = baselines.ALL[name](s, target, h)
    s2 = s.copy()
    s2.loc[origin + pd.Timedelta(days=1):] = 999.0
    b = baselines.ALL[name](s2, target, h)
    assert a.iloc[0] == pytest.approx(b.iloc[0])


def test_climatology_ignores_test_years():
    s = make_series()
    s2 = s.copy()
    s2.loc["2022-01-01":] = 0.0          # destroy everything after the training period
    t = pd.DatetimeIndex(["2024-06-01"])
    assert baselines.climatology(s, t, 3).iloc[0] == pytest.approx(baselines.climatology(s2, t, 3).iloc[0])


def test_climatology_beats_persistence_far_out_and_loses_at_one_day():
    t = horizon_table(make_series(), baselines.ALL, "validation", horizons=[1, 10])
    assert t.loc["persistence", 1] < t.loc["climatology", 1] * 1.2       # yesterday is informative
    assert t.loc["climatology", 10] < t.loc["persistence", 10]            # ten days ago is not


def test_metrics_sign_convention():
    y = pd.Series([1.0, 2.0]); p = pd.Series([2.0, 3.0])
    m = metrics(y, p)
    assert m["bias"] == 1.0 and m["mae"] == 1.0 and m["rmse"] == 1.0
