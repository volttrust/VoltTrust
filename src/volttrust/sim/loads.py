"""Household loads and rooftop PV for Tunisian homes.

Each home = always-on base (fridge, standby) + lighting after dark + daily routine (cooking, TV,
washing) + appliance events + air conditioning (driven by a lagged, indoor-like temperature)
+ electric heating in winter + optional electric water heater - rooftop PV.

Calibrated against public data (see inputs/README.md):
- median home about 1,450 kWh a year (Tunisian household benchmark, UNESCWA 2017);
- months: summer about 1.2x the yearly mean, as in STEG's monthly low/medium-voltage sales;
- daily curve of the feeder: quiet but not empty nights and an evening peak, between the
  Moroccan district feeders (thousands of homes) and what 48 homes can smooth out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Hourly activity shapes (0..1). Ramadan moves activity to suhoor (~03:00) and after iftar.
WEEKDAY = [.15, .10, .08, .08, .08, .12, .30, .45, .40, .35, .38, .48,
           .62, .58, .45, .38, .40, .50, .65, .82, .92, .85, .65, .35]
WEEKEND = [.18, .12, .08, .06, .06, .08, .12, .22, .38, .50, .55, .62,
           .72, .66, .50, .42, .42, .50, .62, .85, 1.0, .95, .80, .48]
RAMADAN = [.70, .60, .55, .80, .60, .25, .18, .18, .20, .20, .22, .25,
           .30, .30, .32, .40, .55, .85, .70, .55, .75, .95, 1.0, .90]
# Fraction of the household at home (drives AC, heating and lighting).
OCC_WORKER = [.80, .80, .80, .80, .80, .80, .70, .45, .25, .20, .20, .25,
              .35, .35, .25, .30, .40, .65, .85, 1.0, 1.0, 1.0, .95, .85]
OCC_HOME = [.80, .80, .80, .80, .80, .80, .70, .65, .70, .75, .80, .85,
            .90, .90, .90, .85, .85, .90, .95, 1.0, 1.0, 1.0, .95, .85]
# Awake share for lighting.
AWAKE = [.25, .10, .05, .05, .05, .15, .55, .75, .60, .50, .50, .50,
         .50, .50, .50, .50, .55, .70, .90, 1.0, 1.0, .95, .80, .55]


@dataclass
class Home:
    meter_id: str
    stay_home: bool
    size: float              # household size / affluence multiplier on everything except base
    base_kw: float
    light_kw: float
    activity_kw: float
    shift_h: float
    has_ac: bool
    ac_kw: float
    ac_setpoint_c: float
    has_heater: bool
    heater_kw: float
    has_water_heater: bool
    has_pv: bool
    pv_kwp: float
    events_per_day: float


def make_homes(n: int, cfg, rng: np.random.Generator) -> list[Home]:
    homes = []
    for i in range(n):
        has_pv = rng.random() < cfg.p_pv
        size = float(np.exp(rng.normal(0, 0.30)))
        homes.append(Home(
            meter_id=f"M{i + 1:03d}",
            stay_home=rng.random() < cfg.p_stay_home,
            size=size,
            base_kw=rng.uniform(0.042, 0.085) * size ** 0.5,
            light_kw=rng.uniform(0.03, 0.08) * size,
            activity_kw=rng.uniform(0.04, 0.09) * size,
            shift_h=rng.normal(0, 0.6),
            has_ac=rng.random() < cfg.p_ac,
            ac_kw=rng.uniform(0.45, 0.85) * size ** 0.5,
            ac_setpoint_c=rng.uniform(27.0, 30.0),
            has_heater=rng.random() < cfg.p_heater,
            heater_kw=rng.uniform(0.8, 1.6),
            has_water_heater=rng.random() < cfg.p_water_heater,
            has_pv=has_pv,
            pv_kwp=float(rng.choice([1.5, 2.0, 2.5, 3.0])) if has_pv else 0.0,
            events_per_day=rng.uniform(0.4, 1.0) * size,
        ))
    return homes


def _shape(table: list[float], hour: np.ndarray, shift: float) -> np.ndarray:
    h = (hour - shift) % 24
    xs = np.arange(25)
    return np.interp(h, xs, table + table[:1])


def _lagged(x: np.ndarray, tau_steps: float) -> np.ndarray:
    """First-order lag (building thermal mass), tau in 15-minute steps."""
    a = 1 / tau_steps
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = y[t - 1] + a * (x[t] - y[t - 1])
    return y


def home_load(home: Home, idx: pd.DatetimeIndex, ctx: pd.DataFrame, rng: np.random.Generator
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (P_load_kw, Q_load_kvar, P_pv_kw, P_ac_kw) for one home."""
    T = len(idx)
    hour = idx.hour.values + idx.minute.values / 60 + 7.5 / 60
    off = (ctx["is_weekend"] | ctx["is_holiday"]).values
    ram = ctx["is_ramadan"].values
    temp = ctx["ambient_temp_c"].values
    ghi = ctx["ghi_wm2"].values

    act = np.where(off, _shape(WEEKEND, hour, home.shift_h), _shape(WEEKDAY, hour, home.shift_h))
    act = np.where(ram, _shape(RAMADAN, hour, home.shift_h), act)
    day_scale = np.repeat(rng.lognormal(0, 0.15, T // 96 + 1), 96)[:T]
    noise = rng.lognormal(0, 0.18, T)
    p_routine = home.activity_kw * act * day_scale * noise

    # Always on: fridge and freezer work harder in summer, plus standby and the router.
    fridge = 1 + 0.02 * (temp - 20)
    p_base = home.base_kw * fridge * (1 + 0.06 * np.sin(2 * np.pi * (rng.random() + hour / 24)))

    # Lighting: on when it is dark (low irradiance) and people are awake.
    dark = np.clip(1 - ghi / 120, 0, 1)
    awake = _shape(AWAKE, hour, home.shift_h)
    p_light = home.light_kw * dark * awake * rng.uniform(0.8, 1.2, T)

    # Appliance events (kettle, oven, washing machine, iron): Poisson, timed by activity.
    p_app = np.zeros(T)
    n_ev = rng.poisson(home.events_per_day * T / 96)
    w = act / act.sum()
    starts = rng.choice(T, size=n_ev, p=w)
    for s, kw, dur in zip(starts, rng.uniform(0.6, 2.0, n_ev), rng.integers(1, 5, n_ev)):
        p_app[s:s + dur] += kw * rng.uniform(0.4, 1.0)

    occ = _shape(OCC_HOME if home.stay_home else OCC_WORKER, hour, home.shift_h)
    # Indoor-like temperature: outdoor lagged by the building (about 5 h), plus solar gain.
    t_in = _lagged(temp, 20) + 2.0 * _lagged(ghi / 1000, 12)
    p_ac = np.zeros(T)
    if home.has_ac:
        need = np.clip((t_in - home.ac_setpoint_c) / 10.0, 0, 0.5)
        duty = np.clip(need * occ * rng.uniform(0.75, 1.15, T), 0, 1)
        p_ac = home.ac_kw * duty
    p_heat = np.zeros(T)
    if home.has_heater:
        need = np.clip((15.0 - _lagged(temp, 12)) / 7.0, 0, 1)
        evening = (hour >= 18) & (hour < 23.5)
        p_heat = home.heater_kw * need * evening * occ * rng.uniform(0.5, 1.0, T)
    p_wh = np.zeros(T)
    if home.has_water_heater:
        # Thermostat cycles, more often after showers (morning, evening) and in winter.
        winter = 1 + 0.5 * np.clip((20 - temp) / 10, 0, 1)
        use = 0.05 + 0.05 * ((hour >= 6) & (hour < 9)) + 0.08 * ((hour >= 19) & (hour < 23.5))
        p_wh = 1.5 * (rng.random(T) < use * winter)

    p_load = p_base + p_light + p_routine + p_app + p_ac + p_heat + p_wh
    tan = lambda pf: np.tan(np.arccos(pf))  # noqa: E731
    q_load = (p_base + p_routine) * tan(0.94) + p_light * tan(0.90) + p_app * tan(0.98) + p_ac * tan(0.86)

    p_pv = np.zeros(T)
    if home.has_pv:
        t_cell = temp + 0.03 * ghi
        p_pv = home.pv_kwp * ghi / 1000 * 0.82 * (1 - 0.004 * (t_cell - 25))
        p_pv = np.clip(p_pv, 0, None)
    return p_load, q_load, p_pv, p_ac
