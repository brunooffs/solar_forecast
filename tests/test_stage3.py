import numpy as np
import pandas as pd
import pytest

from evaluate import TRAIN_END, periods
from ml_models import MLForecaster, feature_frame
from baselines import climatology_table, _doy
from tests.test_stage2 import make_series


def test_features_use_only_the_past():
    s = make_series()
    clim = pd.Series(climatology_table(s)[_doy(s.index)], index=s.index)
    t = pd.DatetimeIndex(["2024-07-15"])
    h = 5
    f1 = feature_frame(s, clim, t - pd.Timedelta(days=h), t)
    s2 = s.copy()
    s2.loc["2024-07-11":] = 999.0           # everything after the origin (2024-07-10)
    f2 = feature_frame(s2, clim, t - pd.Timedelta(days=h), t)
    pd.testing.assert_frame_equal(f1, f2)


@pytest.mark.parametrize("name", ["ridge", "gradient_boosting"])
def test_training_never_touches_data_after_fit_end(name):
    s = make_series()
    s2 = s.copy()
    s2.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):] = 0.0     # ruin everything after training
    targets = periods(s)["validation"][:20]
    # the ruined copy is only used to FIT; predictions use the real, un-ruined series as history
    a = MLForecaster(name)
    a._fit(s2, 3)
    b = MLForecaster(name)
    b._fit(s, 3)
    pa = a(s, targets, 3)
    pb = b(s, targets, 3)
    assert np.allclose(pa, pb)


def test_ml_beats_climatology_on_data_with_weather_memory():
    s = make_series()
    targets = periods(s)["validation"]
    clim = pd.Series(climatology_table(s)[_doy(targets)], index=targets)
    f = MLForecaster("ridge")
    err_ml = (f(s, targets, 1) - s.loc[targets]).abs().mean()
    err_clim = (clim - s.loc[targets]).abs().mean()
    assert err_ml < err_clim            # the synthetic noise is smoothed, so yesterday really informs today


def test_predictions_are_not_negative_and_cover_all_targets():
    s = make_series()
    targets = periods(s)["validation"]
    p = MLForecaster("random_forest")(s, targets, 7)
    assert len(p) == len(targets) and (p >= 0).all() and p.notna().all()
