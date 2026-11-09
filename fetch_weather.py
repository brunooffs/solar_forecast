"""Stage 1a: download hourly historical weather from Open-Meteo and store it as CSV.

    python fetch_weather.py --start 2015-01-01

No API key needed (free for non-commercial use). One request per calendar year;
finished years are cached in data/raw/ so a rerun only downloads what is new.
Data: Open-Meteo.com (ERA5 reanalysis, Copernicus Climate Change Service).
"""
import argparse
import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests

from config import DATA, load_system

URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["temperature_2m", "shortwave_radiation", "direct_normal_irradiance",
             "diffuse_radiation", "wind_speed_10m"]
LATENCY_DAYS = 7   # the archive trails real time by about 5 days; stay clear of it


class WeatherError(RuntimeError):
    pass


def request_year_chunk(lat, lon, start: date, end: date, model=None, session=None, retries=5) -> pd.DataFrame:
    """One API call -> DataFrame indexed by UTC time. Retries rate limits and server errors."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(VARIABLES),
        "wind_speed_unit": "ms",     # default is km/h; the energy model wants m/s
        "timezone": "GMT",           # keep everything in UTC; convert to local time later
    }
    if model:
        params["models"] = model
    get = (session or requests).get
    for attempt in range(retries):
        try:
            r = get(URL, params=params, timeout=60)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise WeatherError(f"could not reach Open-Meteo: {e}") from e
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == retries - 1:
                raise WeatherError(f"Open-Meteo kept answering HTTP {r.status_code}. Try again later.")
            time.sleep(5 * 2 ** attempt)    # rate limited: wait longer each time
            continue
        if r.status_code != 200:
            try:
                reason = r.json().get("reason", r.text[:200])
            except ValueError:
                reason = r.text[:200]
            raise WeatherError(f"Open-Meteo refused the request (HTTP {r.status_code}): {reason}")
        return parse_hourly(r.json())
    raise WeatherError("unreachable")


def parse_hourly(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherError("unexpected response: no 'hourly' block")
    missing = [v for v in VARIABLES if v not in hourly]
    if missing:
        raise WeatherError(f"response lacks variables: {missing}")
    df = pd.DataFrame({v: hourly[v] for v in VARIABLES})
    df.index = pd.to_datetime(hourly["time"], utc=True)
    df.index.name = "time"
    return df.astype(float)


def fetch(start: date, end: date, system: dict, session=None) -> pd.DataFrame:
    loc = system["location"]
    model = (system.get("weather") or {}).get("model")
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    key = f"{loc['latitude']}_{loc['longitude']}_{model or 'auto'}"
    parts = []
    for year in range(start.year, end.year + 1):
        y0, y1 = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        cache = raw / f"{key}_{year}.csv"
        complete = y1 == date(year, 12, 31) and y0 == date(year, 1, 1)
        if cache.exists() and complete:
            print(f"  {year}: cached")
            parts.append(pd.read_csv(cache, index_col="time", parse_dates=["time"]))
            continue
        print(f"  {year}: downloading {y0} .. {y1}")
        df = request_year_chunk(loc["latitude"], loc["longitude"], y0, y1, model, session)
        if complete:
            df.to_csv(cache)
        parts.append(df)
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2015-01-01", help="first day, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="last day (default: a week ago)")
    ap.add_argument("--system", default=None, help="path to a different system.yaml")
    a = ap.parse_args()
    system = load_system(a.system)
    start = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=LATENCY_DAYS)
    if start > end:
        raise SystemExit("start is after end")
    df = fetch(start, end, system)
    DATA.mkdir(exist_ok=True)
    path = DATA / "weather_hourly.csv"
    df.to_csv(path)
    print(f"wrote {path}: {len(df)} hourly rows, {df.index.min()} .. {df.index.max()}")
    gaps = df.isna().sum()
    if gaps.any():
        print("missing values per column:\n" + gaps[gaps > 0].to_string())


if __name__ == "__main__":
    try:
        main()
    except WeatherError as e:
        raise SystemExit(f"error: {e}")
