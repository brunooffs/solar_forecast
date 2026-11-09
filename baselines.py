"""Stage 2: simple forecasters any real model has to beat.

If a fancy model cannot beat these, it has learned nothing useful.
Each takes (s, target_dates, h) and may use only data up to day (target - h).
"""
import numpy as np
import pandas as pd

from evaluate import TRAIN_END


def _doy(index: pd.DatetimeIndex) -> np.ndarray:
    return np.minimum(index.dayofyear.to_numpy(), 365) - 1      # 0..364 (Dec 31 of leap years shares 364)


def climatology_table(s: pd.Series, train_end=TRAIN_END, half_window=7) -> np.ndarray:
    """Average energy for each day of the year over the TRAIN years, smoothed over +-7 days (wrapping at new year)."""
    tr = s.loc[:train_end]
    by_doy = np.array([tr.to_numpy()[_doy(tr.index) == d].mean() for d in range(365)])
    padded = np.concatenate([by_doy[-half_window:], by_doy, by_doy[:half_window]])
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    return np.convolve(padded, kernel, mode="valid")


def climatology(s, target_dates, h):
    """'It will be the typical energy for this time of year.' Ignores h: it knows no weather."""
    table = climatology_table(s)
    return pd.Series(table[_doy(target_dates)], index=target_dates)


def persistence(s, target_dates, h):
    """'It will be the same as the last day I know.' The day h days before the target."""
    return pd.Series(s.shift(h).reindex(target_dates).to_numpy(), index=target_dates)


def recent_week_mean(s, target_dates, h):
    """'It will be like the average of the last 7 days I know.'"""
    return pd.Series(s.rolling(7).mean().shift(h).reindex(target_dates).to_numpy(), index=target_dates)


def anomaly_persistence(s, target_dates, h, decay_days=3.0):
    """Typical value for the season, plus how far the last week was above or below typical,
    fading with distance: 'a cloudy spell tends to last a few more days'."""
    table = climatology_table(s)
    clim_all = pd.Series(table[_doy(s.index)], index=s.index)
    anomaly = (s - clim_all).rolling(3).mean().shift(h).reindex(target_dates)
    pred = pd.Series(table[_doy(target_dates)], index=target_dates)
    return pred + anomaly.fillna(0.0) * np.exp(-h / decay_days)


ALL = {"climatology": climatology, "persistence": persistence,
       "recent_week_mean": recent_week_mean, "anomaly_persistence": anomaly_persistence}
