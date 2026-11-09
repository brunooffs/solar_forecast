"""Stage 1b: turn hourly weather into the energy a given solar installation would produce.

Steps (each is one block below):
  1. where is the sun at each hour               (pvlib.solarposition)
  2. how much sunlight lands on the TILTED panel (pvlib.irradiance, 'plane of array')
  3. how hot the panels get                      (pvlib.temperature.faiman)
  4. electrical power = sunlight x area x efficiency, minus heat and system losses
  5. add up the hours into local days (kWh/day)

This is a physics MODEL, not a measurement. Treat its numbers as "typical", not exact.
"""
import numpy as np
import pandas as pd
from pvlib import irradiance, solarposition, temperature

from config import peak_kw


def hourly_power_kw(weather: pd.DataFrame, system: dict) -> pd.DataFrame:
    """Hourly AC power estimate. `weather` is indexed by UTC time (hour END, as Open-Meteo gives it)."""
    loc, pan, ori = system["location"], system["panels"], system["orientation"]

    # Open-Meteo radiation is the mean over the PRECEDING hour, so the sun position that
    # matches it is the middle of that hour: timestamp minus 30 minutes.
    mid = weather.index - pd.Timedelta(minutes=30)

    sun = solarposition.get_solarposition(mid, loc["latitude"], loc["longitude"])
    dni_extra = irradiance.get_extra_radiation(mid)

    poa = irradiance.get_total_irradiance(
        surface_tilt=ori["tilt_deg"], surface_azimuth=ori["azimuth_deg"],
        solar_zenith=sun["apparent_zenith"].to_numpy(), solar_azimuth=sun["azimuth"].to_numpy(),
        dni=weather["direct_normal_irradiance"].to_numpy(),
        ghi=weather["shortwave_radiation"].to_numpy(),
        dhi=weather["diffuse_radiation"].to_numpy(),
        dni_extra=dni_extra.to_numpy(), model="haydavies", albedo=0.2,
    )
    poa_global = np.nan_to_num(np.asarray(poa["poa_global"]), nan=0.0).clip(min=0)

    cell_temp = temperature.faiman(poa_global, weather["temperature_2m"].to_numpy(),
                                   weather["wind_speed_10m"].to_numpy())
    area = pan["count"] * pan["area_m2"]
    dc_w = poa_global * area * pan["efficiency"] * (1 + pan["temp_coefficient"] * (cell_temp - 25.0))
    ac_kw = (dc_w * (1 - system["losses"])).clip(min=0) / 1000.0
    cap = system.get("inverter_kw")
    if cap:
        ac_kw = ac_kw.clip(max=cap)

    return pd.DataFrame({"poa_w_m2": poa_global, "power_kw": ac_kw,
                         "ghi_w_m2": weather["shortwave_radiation"].to_numpy(),
                         "temp_c": weather["temperature_2m"].to_numpy()}, index=weather.index)


def daily_energy(weather: pd.DataFrame, system: dict) -> pd.DataFrame:
    """Local-day totals. One row per day that has all its hours of data."""
    hourly = hourly_power_kw(weather, system)
    local = hourly.tz_convert(system["location"]["timezone"])
    # each hourly row is the hour ENDING at its timestamp: shift back so it falls in the day it belongs to
    local.index = local.index - pd.Timedelta(minutes=30)
    g = local.resample("D")
    daily = pd.DataFrame({
        "energy_kwh": g["power_kw"].sum(),          # kW x 1 h = kWh
        "poa_kwh_m2": g["poa_w_m2"].sum() / 1000.0,
        "ghi_kwh_m2": g["ghi_w_m2"].sum() / 1000.0,
        "temp_mean_c": g["temp_c"].mean(),
        "hours": g["power_kw"].count(),
    })
    # drop partial days at the ends (23 and 25 hour DST days are fine)
    daily = daily[daily["hours"] >= 23].drop(columns="hours")
    daily.index = daily.index.tz_localize(None)
    daily.index.name = "date"
    return daily


def yearly_yield_kwh_per_kwp(daily: pd.DataFrame, system: dict) -> float:
    """Sanity number: kWh per kW of panels per year. Roughly 900 to 1800 for most inhabited places."""
    years = len(daily) / 365.25
    return float(daily["energy_kwh"].sum() / years / peak_kw(system))
