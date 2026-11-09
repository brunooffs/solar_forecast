"""Stage 3: machine-learning forecasters that use only the past energy history.

Idea: do not predict kWh directly. Predict how far the day will be ABOVE or BELOW the typical
value for that time of year (the "anomaly"), then add the typical value back. The season is
then handled by the climatology, and the model only has to learn the weather part.

For a forecast at horizon h of target day D, the features come from the origin day D - h:
  - anomalies of the last 14 days (lag 0..13) and their 3/7/14/30-day averages
  - the typical value for the target day, and the day of year as a sin/cos pair
One separate model is trained per horizon.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import _doy, climatology_table
from evaluate import TRAIN_END

LAGS = range(14)
WINDOWS = (3, 7, 14, 30)

MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                                                   max_features=0.5, n_jobs=-1, random_state=0),
    "gradient_boosting": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                               max_iter=200, min_samples_leaf=30,
                                                               random_state=0),
}


def feature_frame(s: pd.Series, clim: pd.Series, origins: pd.DatetimeIndex, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """Everything here uses data up to and including the origin day, never later."""
    a = s - clim                                    # anomaly: above / below typical
    cols = {f"lag{k}": a.shift(k) for k in LAGS}
    for w in WINDOWS:
        cols[f"mean{w}"] = a.rolling(w).mean()
    f = pd.DataFrame(cols).reindex(origins)
    doy = targets.dayofyear.to_numpy()
    f["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    f.index = targets                               # row = the day being forecast
    return f


class MLForecaster:
    """Callable (s, target_dates, h) -> predictions, like the baselines in baselines.py.

    The model for each horizon is trained on the first call, using only days up to `fit_end`."""

    def __init__(self, name: str, fit_end: str = TRAIN_END):
        self.name, self.fit_end = name, pd.Timestamp(fit_end)
        self._models: dict[int, object] = {}
        self._columns: list[str] = []

    def _clim(self, s):
        return pd.Series(climatology_table(s)[_doy(s.index)], index=s.index)

    def _fit(self, s: pd.Series, h: int):
        clim = self._clim(s)
        targets = s.index[s.index <= self.fit_end][h:]
        x = feature_frame(s, clim, targets - pd.Timedelta(days=h), targets)
        y = (s - clim).reindex(targets)
        ok = x.notna().all(axis=1)
        self._columns = list(x.columns)
        m = MODELS[self.name]()
        m.fit(x[ok].to_numpy(), y[ok].to_numpy())
        self._models[h] = m

    def __call__(self, s, target_dates, h):
        if h not in self._models:
            self._fit(s, h)
        clim = self._clim(s)
        x = feature_frame(s, clim, target_dates - pd.Timedelta(days=h), target_dates)
        ok = x.notna().all(axis=1)
        anomaly = pd.Series(np.nan, index=target_dates)
        anomaly[ok] = self._models[h].predict(x[ok].to_numpy())
        pred = clim.reindex(target_dates) + anomaly.fillna(0.0)    # no features yet -> fall back to typical
        return pred.clip(lower=0.0)

    def importances(self, h: int) -> pd.Series | None:
        m = self._models.get(h)
        imp = getattr(m, "feature_importances_", None)
        if imp is None and hasattr(m, "named_steps"):
            imp = np.abs(m.named_steps["ridge"].coef_)
        return None if imp is None else pd.Series(imp, index=self._columns).sort_values(ascending=False)
