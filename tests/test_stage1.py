import copy
import json

import pandas as pd
import pytest
from pvlib.location import Location

import fetch_weather
from config import load_system, peak_kw
from energy_model import daily_energy, hourly_power_kw


@pytest.fixture
def system():
    return load_system()


def clear_sky_weather(system, start, days):
    loc = system["location"]
    site = Location(loc["latitude"], loc["longitude"], tz="UTC")
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    # Open-Meteo stamps the END of the hour, the model uses the middle: build clear sky at the middle
    cs = site.get_clearsky(idx - pd.Timedelta(minutes=30))
    return pd.DataFrame({
        "temperature_2m": 20.0, "wind_speed_10m": 2.0,
        "shortwave_radiation": cs["ghi"].to_numpy(),
        "direct_normal_irradiance": cs["dni"].to_numpy(),
        "diffuse_radiation": cs["dhi"].to_numpy(),
    }, index=idx)


def test_summer_clear_day_is_plausible_and_nights_are_zero(system):
    w = clear_sky_weather(system, "2024-06-20", 2)
    h = hourly_power_kw(w, system)
    assert h.loc["2024-06-20 00:00":"2024-06-20 03:00", "power_kw"].max() == 0
    d = daily_energy(w, system)
    kwp = peak_kw(system)
    # ~4 to 6.5 kWh per kWp on a clear June day at 41 N
    assert 4.0 * kwp < d["energy_kwh"].iloc[0] < 6.5 * kwp


def test_summer_beats_winter(system):
    summer = daily_energy(clear_sky_weather(system, "2024-06-20", 2), system)["energy_kwh"].iloc[0]
    winter = daily_energy(clear_sky_weather(system, "2024-12-20", 2), system)["energy_kwh"].iloc[0]
    # tilted south-facing panels flatten the seasons on CLEAR days; the big winter drop comes from clouds
    assert 1.1 * winter < summer < 2.5 * winter
    assert winter > 0


def test_double_the_panels_double_the_energy(system):
    w = clear_sky_weather(system, "2024-06-20", 2)
    big = copy.deepcopy(system)
    big["panels"]["count"] *= 2
    a = daily_energy(w, system)["energy_kwh"].iloc[0]
    b = daily_energy(w, big)["energy_kwh"].iloc[0]
    assert b == pytest.approx(2 * a, rel=1e-6)


def test_facing_away_from_the_sun_loses_energy(system):
    w = clear_sky_weather(system, "2024-03-20", 2)
    north = copy.deepcopy(system)
    north["orientation"]["azimuth_deg"] = 0
    assert daily_energy(w, north)["energy_kwh"].iloc[0] < 0.7 * daily_energy(w, system)["energy_kwh"].iloc[0]


def test_inverter_cap(system):
    w = clear_sky_weather(system, "2024-06-20", 2)
    capped = copy.deepcopy(system)
    capped["inverter_kw"] = 1.0
    assert hourly_power_kw(w, capped)["power_kw"].max() <= 1.0 + 1e-9


def test_partial_days_are_dropped(system):
    w = clear_sky_weather(system, "2024-06-20", 3).iloc[5:-7]
    assert len(daily_energy(w, system)) == 1     # only the middle full local day survives


def test_bad_config_fails_clearly(tmp_path):
    import yaml
    s = yaml.safe_load(open("system.yaml"))
    s["orientation"]["tilt_deg"] = 120
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(s))
    with pytest.raises(ValueError, match="tilt"):
        load_system(p)


# ---- the download code, against a fake server (the real API is not reachable from the test box) ----

def fake_payload(n=48):
    t = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {"hourly": {"time": t, **{v: [1.0] * n for v in fetch_weather.VARIABLES}}}


class Resp:
    def __init__(self, code, body):
        self.status_code, self._b, self.text = code, body, json.dumps(body)

    def json(self):
        return self._b


class Session:
    def __init__(self, *codes):
        self.codes, self.calls, self.params = list(codes), 0, None

    def get(self, url, params, timeout):
        self.calls += 1
        self.params = params
        code = self.codes.pop(0) if self.codes else 200
        return Resp(code, fake_payload() if code == 200 else {"error": True, "reason": "bad thing"})


def test_download_parses_and_asks_for_ms_and_gmt():
    from datetime import date
    s = Session()
    df = fetch_weather.request_year_chunk(1, 2, date(2024, 1, 1), date(2024, 1, 2), session=s)
    assert len(df) == 48 and str(df.index.tz) == "UTC"
    assert s.params["wind_speed_unit"] == "ms" and s.params["timezone"] == "GMT"


def test_download_retries_rate_limit(monkeypatch):
    from datetime import date
    monkeypatch.setattr(fetch_weather.time, "sleep", lambda s: None)
    s = Session(429, 503, 200)
    assert len(fetch_weather.request_year_chunk(1, 2, date(2024, 1, 1), date(2024, 1, 2), session=s)) == 48
    assert s.calls == 3


def test_download_reports_api_reason():
    from datetime import date
    with pytest.raises(fetch_weather.WeatherError, match="bad thing"):
        fetch_weather.request_year_chunk(1, 2, date(2024, 1, 1), date(2024, 1, 2), session=Session(400))


def test_parse_rejects_missing_variable():
    p = fake_payload()
    del p["hourly"]["diffuse_radiation"]
    with pytest.raises(fetch_weather.WeatherError, match="diffuse_radiation"):
        fetch_weather.parse_hourly(p)
