# solar_forecast

Forecast how much energy a solar installation will produce over the next 10 days.

The installation is hypothetical and described in `system.yaml` (number of panels, panel size,
efficiency, tilt, direction, location anywhere on Earth). Its output is **modelled with physics
(pvlib), not measured**. Numbers are "typical for this setup", never exact.

This folder is independent of the older notebook in the repo root and has its own requirements.

## Stages

| Stage | What | Status |
|---|---|---|
| 1 | Download history, model the panels, build the daily dataset | this folder |
| 2 | Baselines to beat: "same as last year's average" and "same as 10 days ago" | next |
| 3 | Machine-learning model, honest time-based evaluation | next |
| 4 | Real forecast: feed in tomorrow's to day-10 weather forecast | next |

## Setup

```bash
cd solar_forecast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                       # should all pass
```

## Stage 1: build the dataset

```bash
python fetch_weather.py --start 2015-01-01   # hourly weather -> data/weather_hourly.csv (no API key)
python build_dataset.py                      # panels model   -> data/energy_daily.csv
```

`build_dataset.py` prints a sanity number: kWh per kWp per year. Most inhabited places land
between about 900 and 1800. Far outside that range means something in `system.yaml` is off
(for example an azimuth that makes the panels face away from the sun).

Weather data: [Open-Meteo](https://open-meteo.com) (ERA5 reanalysis, Copernicus Climate Change
Service). Free for non-commercial use; please credit them if you publish results.
