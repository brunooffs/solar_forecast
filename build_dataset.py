"""Stage 1c: weather_hourly.csv -> energy_daily.csv (the dataset the models will use).

    python build_dataset.py
"""
import pandas as pd

from config import DATA, load_system, peak_kw
from energy_model import daily_energy, yearly_yield_kwh_per_kwp


def main():
    system = load_system()
    src = DATA / "weather_hourly.csv"
    if not src.exists():
        raise SystemExit("data/weather_hourly.csv not found. Run: python fetch_weather.py")
    weather = pd.read_csv(src, index_col="time", parse_dates=["time"])
    bad = weather.isna().any(axis=1).sum()
    if bad:
        print(f"note: {bad} hourly rows have missing values; those hours count as zero")
        weather = weather.fillna(0.0)
    daily = daily_energy(weather, system)
    out = DATA / "energy_daily.csv"
    daily.round(3).to_csv(out)
    print(f"system: {peak_kw(system):.2f} kWp, tilt {system['orientation']['tilt_deg']}, "
          f"azimuth {system['orientation']['azimuth_deg']}")
    print(f"wrote {out}: {len(daily)} days, {daily.index.min().date()} .. {daily.index.max().date()}")
    print(f"mean {daily['energy_kwh'].mean():.1f} kWh/day, best {daily['energy_kwh'].max():.1f}, "
          f"worst {daily['energy_kwh'].min():.1f}")
    print(f"yield {yearly_yield_kwh_per_kwp(daily, system):.0f} kWh per kWp per year "
          f"(sanity range ~900 to 1800)")


if __name__ == "__main__":
    main()
