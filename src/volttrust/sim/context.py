"""Calendar, real weather and solar context for a neighborhood near Tunis (2025)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAT, LON = 36.80, 10.18          # Tunis

# Tunisian public holidays 2025 (official dates as observed).
HOLIDAYS_2025 = [
    "2025-01-01", "2025-01-14", "2025-03-20", "2025-03-30", "2025-03-31", "2025-04-01",  # Eid al-Fitr
    "2025-04-09", "2025-05-01", "2025-06-06", "2025-06-07",                               # Eid al-Adha
    "2025-06-26", "2025-07-25", "2025-08-13", "2025-09-04",                               # Hijri new year, Mawlid
    "2025-10-15", "2025-12-17",
]
RAMADAN_2025 = ("2025-03-01", "2025-03-29")

# Real hourly weather for Tunis (Open-Meteo historical API, 36.80N 10.17E, 2025, CC BY 4.0).
WEATHER_CSV = Path(__file__).resolve().parents[3] / "inputs" / "open_meteo_tunis_2025_hourly.csv"
# Heatwave days: a 3-day window whose mean temperature is at least this (deg C).
HEATWAVE_3DAY_MEAN_C = 30.5


def time_index(start: str, end: str, freq_min: int) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=f"{freq_min}min")


def calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    day = idx.normalize()
    hol = day.isin(pd.to_datetime(HOLIDAYS_2025))
    ram = (day >= RAMADAN_2025[0]) & (day <= RAMADAN_2025[1])
    weekend = idx.dayofweek >= 5
    return pd.DataFrame({"is_weekend": weekend, "is_holiday": hol, "is_ramadan": ram}, index=idx)


def solar_elevation(idx: pd.DatetimeIndex, tz_offset_h: int) -> np.ndarray:
    """Solar elevation angle (rad) at the midpoint of each interval, local standard time."""
    t = idx + pd.Timedelta(minutes=7.5)
    doy = t.dayofyear.values
    hour_utc = t.hour.values + t.minute.values / 60 - tz_offset_h
    g = 2 * np.pi / 365 * (doy - 1 + (hour_utc - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    solar_time = hour_utc * 60 + eot + 4 * LON
    ha = np.deg2rad(solar_time / 4 - 180)
    lat = np.deg2rad(LAT)
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return np.arcsin(np.clip(sin_el, -1, 1))


def read_weather(path: str | Path = WEATHER_CSV) -> pd.DataFrame:
    """Open-Meteo hourly CSV -> DataFrame indexed by local time (temp_c, ghi_wm2, cloud)."""
    w = pd.read_csv(path, skiprows=3)
    w.columns = ["time", "temp_c", "ghi_wm2", "cloud_pct"]
    w["time"] = pd.to_datetime(w["time"])
    return w.set_index("time")


def _at(hourly: pd.Series, t: pd.DatetimeIndex) -> np.ndarray:
    x = hourly.index.asi8.astype(float)
    return np.interp(t.asi8.astype(float), x, hourly.values.astype(float))


def weather(idx: pd.DatetimeIndex, tz_offset_h: int, rng: np.random.Generator,
            path: str | Path = WEATHER_CSV) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temperature (deg C), GHI (W/m2) and heatwave flag at each 15-minute interval.

    Temperature is the instantaneous hourly value, interpolated to the interval midpoint.
    Open-Meteo radiation is the mean over the preceding hour, so it is placed at the middle of
    that hour before interpolating, then given intra-hour cloud ramps whose size grows with
    cloud cover; the hourly mean is kept.
    """
    w = read_weather(path)
    mid = idx + pd.Timedelta(minutes=7.5)
    temp = _at(w["temp_c"], mid)
    ghi_h = w["ghi_wm2"].copy()
    ghi_h.index = ghi_h.index - pd.Timedelta(minutes=30)
    ghi = np.clip(_at(ghi_h, mid), 0, None)
    cloud = _at(w["cloud_pct"], mid) / 100

    el = solar_elevation(idx, tz_offset_h)
    cosz = np.clip(np.sin(el), 0, None)
    clear = np.where(cosz > 0.01, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 0.01)), 0.0)
    fl = np.zeros(len(idx))
    e = rng.normal(0, 0.12, len(idx))
    for i in range(1, len(idx)):
        fl[i] = 0.6 * fl[i - 1] + e[i]
    amp = 1.6 * np.sqrt(cloud * (1 - cloud))
    hour = idx.floor("h")
    f = 1 + amp * fl
    f = f / pd.Series(f).groupby(hour).transform("mean").values      # keep the hourly mean
    ghi = np.clip(ghi * f, 0, np.maximum(clear * 1.1, ghi))

    daily = w["temp_c"].resample("D").mean()
    hot = daily.rolling(3, center=True).mean() >= HEATWAVE_3DAY_MEAN_C
    hot = hot | hot.shift(1, fill_value=False) | hot.shift(-1, fill_value=False)
    heat_days = daily.index[hot.values]
    heatwave = idx.normalize().isin(heat_days)
    return temp, ghi, np.asarray(heatwave)


def build_context(idx: pd.DatetimeIndex, tz_offset_h: int, rng: np.random.Generator) -> pd.DataFrame:
    ctx = calendar(idx)
    temp, ghi, heatwave = weather(idx, tz_offset_h, rng)
    ctx["ambient_temp_c"] = temp
    ctx["ghi_wm2"] = ghi
    ctx["heatwave"] = heatwave
    return ctx
