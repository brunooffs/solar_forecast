"""Checks what Open-Meteo's forecast APIs really return, so stage 4 is built on facts, not guesses.

    python probe_forecast_api.py

Makes 3 small requests and prints a short report. Paste the whole output back.
"""
import requests

from config import load_system

BASE = ["temperature_2m", "shortwave_radiation", "direct_normal_irradiance",
        "diffuse_radiation", "wind_speed_10m", "cloud_cover"]


def show(title, url, params):
    print(f"\n=== {title}")
    try:
        r = requests.get(url, params=params, timeout=60)
    except requests.RequestException as e:
        print("request failed:", e)
        return
    print("HTTP", r.status_code)
    try:
        j = r.json()
    except ValueError:
        print("not JSON:", r.text[:300])
        return
    if r.status_code != 200:
        print("reason:", j.get("reason", j))
        return
    h = j.get("hourly", {})
    t = h.get("time", [])
    print(f"rows: {len(t)}   first: {t[0] if t else '-'}   last: {t[-1] if t else '-'}")
    for k, v in h.items():
        if k == "time":
            continue
        nulls = sum(x is None for x in v)
        sample = next((x for x in v if x is not None), None)
        print(f"  {k:<45} nulls {nulls:>4}/{len(v)}   first value {sample}")
    print("units:", {k: u for k, u in j.get("hourly_units", {}).items() if k != "time"})


def main():
    loc = load_system()["location"]
    common = {"latitude": loc["latitude"], "longitude": loc["longitude"],
              "wind_speed_unit": "ms", "timezone": "GMT"}

    show("1. Forecast API, next 16 days", "https://api.open-meteo.com/v1/forecast",
         {**common, "hourly": ",".join(BASE), "forecast_days": 16})

    prev = [f"{v}_previous_day{n}" for v in BASE for n in (1, 7)]
    show("2. Previous Runs API, forecasts made 1 and 7 days earlier (June 2024)",
         "https://previous-runs-api.open-meteo.com/v1/forecast",
         {**common, "hourly": ",".join(prev), "start_date": "2024-06-01", "end_date": "2024-06-03"})

    show("3. Previous Runs API, same but January 2024 (earliest archive start?)",
         "https://previous-runs-api.open-meteo.com/v1/forecast",
         {**common, "hourly": "shortwave_radiation_previous_day1,shortwave_radiation_previous_day7",
          "start_date": "2024-01-01", "end_date": "2024-01-02"})


if __name__ == "__main__":
    main()
