"""Anomaly scheduling and injection.

Two layers (see docs/anomalies.md):
- PHYSICAL events change the inputs of the power flow (source voltage, loads, line resistance,
  energized phases, frequency, transformer cooling). Every sensor then sees them consistently.
- MEASUREMENT events change only what one meter reports. True values are kept separately.

Each event is a dict that ends up as one row of events.csv.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .grid import PHASES

# type -> (family, verdict, layer)
CATALOG = {
    # genuine grid events
    "voltage_sag":            ("grid_event", "grid_event", "physical"),
    "mv_undervoltage":        ("grid_event", "grid_event", "physical"),
    "tap_swell":              ("grid_event", "grid_event", "physical"),
    "outage":                 ("grid_event", "grid_event", "physical"),
    "frequency_excursion":    ("grid_event", "grid_event", "physical"),
    # equipment faults (real, physical)
    "phase_loss":             ("equipment_fault", "equipment_fault", "physical"),
    "high_resistance_joint":  ("equipment_fault", "equipment_fault", "physical"),
    "leakage":                ("equipment_fault", "equipment_fault", "physical"),
    "transformer_cooling":    ("equipment_fault", "equipment_fault", "physical"),
    # degraded sensors
    "drift_voltage":          ("degraded_sensor", "data_quality", "measurement"),
    "drift_current":          ("degraded_sensor", "data_quality", "measurement"),
    "noise_growth":           ("degraded_sensor", "data_quality", "measurement"),
    "stuck":                  ("degraded_sensor", "data_quality", "measurement"),
    # biased readings
    "bias_voltage":           ("biased_reading", "data_quality", "measurement"),
    "bias_ct_gain":           ("biased_reading", "data_quality", "measurement"),
    "clock_skew":             ("biased_reading", "data_quality", "measurement"),
    # corrupted readings
    "spike":                  ("corrupted_reading", "data_quality", "measurement"),
    "dropout":                ("corrupted_reading", "data_quality", "measurement"),
    # falsified readings
    "theft_bypass":           ("falsified_reading", "data_quality", "measurement"),
    "tamper_report":          ("falsified_reading", "data_quality", "measurement"),
    "replay":                 ("falsified_reading", "data_quality", "measurement"),
    "coordinated_fdi":        ("falsified_reading", "data_quality", "measurement"),
    # natural failure of the meter itself
    "meter_end_of_life":      ("natural_failure", "data_quality", "measurement"),
    # rooftop PV: real faults on the panels or inverter, and problems with the inverter's reading
    "pv_soiling":             ("equipment_fault", "equipment_fault", "physical"),
    "pv_inverter_trip":       ("equipment_fault", "equipment_fault", "physical"),
    "pv_degradation":         ("equipment_fault", "equipment_fault", "physical"),
    "pv_reading_stuck":       ("degraded_sensor", "data_quality", "measurement"),
    "pv_inflated_report":     ("falsified_reading", "data_quality", "measurement"),
}
PV_READING = ("pv_reading_stuck", "pv_inflated_report")

DAY = 96  # intervals per day at 15 min


@dataclass
class PhysicalInputs:
    """Arrays the physical events write into, consumed by the power flow."""
    T: int
    H: int
    F: int
    K: int
    v_src_scale: np.ndarray = None      # (T,)
    load_scale: np.ndarray = None       # (T, H) multiplies consumption (cold-load pickup)
    extra_p: np.ndarray = None          # (T, H) kW added (leakage)
    extra_q: np.ndarray = None
    extra_r: np.ndarray = None          # (T, F, K, 3)
    energized: np.ndarray = None        # (T, F, 3)
    freq_offset: np.ndarray = None      # (T,)
    sag_depth: np.ndarray = None        # (T,) relative dip of the in-interval minimum voltage
    cooling_factor: np.ndarray = None   # (T,) multiplies rated top-oil rise
    pv_scale: np.ndarray = None         # (T, H) multiplies rooftop PV output (soiling, trips, ageing)

    def __post_init__(self):
        T, H, F, K = self.T, self.H, self.F, self.K
        self.v_src_scale = np.ones(T)
        self.load_scale = np.ones((T, H))
        self.extra_p = np.zeros((T, H))
        self.extra_q = np.zeros((T, H))
        self.extra_r = np.zeros((T, F, K, 3))
        self.energized = np.ones((T, F, 3), bool)
        self.freq_offset = np.zeros(T)
        self.sag_depth = np.zeros(T)
        self.pv_scale = np.ones((T, H))
        self.cooling_factor = np.ones(T)


@dataclass
class Labels:
    T: int
    H: int
    F: int
    meter: np.ndarray = None    # (T, H) event index or -1
    head: np.ndarray = None     # (T, F, 3)
    tx: np.ndarray = None       # (T,)
    events: list = field(default_factory=list)

    def __post_init__(self):
        self.meter = np.full((self.T, self.H), -1, np.int32)
        self.head = np.full((self.T, self.F, 3), -1, np.int32)
        self.tx = np.full(self.T, -1, np.int32)


class Scheduler:
    """Places events in time without overlapping on the same meter."""

    def __init__(self, idx: pd.DatetimeIndex, topo, cfg, rng: np.random.Generator):
        self.idx, self.topo, self.cfg, self.rng = idx, topo, cfg, rng
        self.T, self.H = len(idx), len(topo.feeder)
        self.busy = np.zeros((self.T, self.H), bool)        # any event on the meter
        self.busy_local = np.zeros((self.T, self.H), bool)  # meter-level events only
        self.head_busy = np.zeros((self.T, topo.n_feeders, 3), bool)
        self.tx_busy = np.zeros(self.T, bool)
        self.val_start = int(np.searchsorted(idx, pd.Timestamp(cfg.train_end), side="right"))
        self.test_start = int(np.searchsorted(idx, pd.Timestamp(cfg.val_end), side="right"))
        self.labels = Labels(self.T, self.H, topo.n_feeders)

    def window(self, dur: int, meters, lo: int = 0, hi: int | None = None, pad: int = 4,
               heads=None, tx=False, tries: int = 400, hours=None, local_only=False) -> int | None:
        """Random start with no overlap. local_only=True lets a meter-level event overlap
        feeder- or transformer-wide events (a drifting sensor keeps drifting during a sag)."""
        hi = self.T if hi is None else hi
        busy = self.busy_local if local_only else self.busy
        for _ in range(tries):
            # `hi` bounds the start only: an event may run past the end of its split
            a, b = max(lo, pad + 7 * DAY), min(hi, self.T - dur - pad)
            if b <= a:
                return None
            s = int(self.rng.integers(a, b))
            if hours is not None and self.idx[s].hour not in hours:
                continue
            a, b = s - pad, s + dur + pad
            if len(meters) and busy[a:b, meters].any():
                continue
            if heads is not None and any(self.head_busy[a:b, f, p].any() for f, p in heads):
                continue
            if tx and self.tx_busy[a:b].any():
                continue
            return s
        return None

    def add(self, etype: str, s: int, dur: int, meters, heads=(), tx=False, params=None,
            channels=(), scope="") -> int:
        family, verdict, layer = CATALOG[etype]
        e = s + dur
        k = len(self.labels.events)
        meters = np.asarray(meters, int)
        if len(meters):
            # meter-level labels win over feeder-wide ones where they overlap
            self.busy[s:e, meters] = True
            if len(meters) <= 12:
                self.busy_local[s:e, meters] = True
            self.labels.meter[s:e, meters] = k
        for f, p in heads:
            self.head_busy[s:e, f, p] = True
            self.labels.head[s:e, f, p] = k
        if tx:
            self.tx_busy[s:e] = True
            self.labels.tx[s:e] = k
        split = "train" if s < self.val_start else ("val" if s < self.test_start else "test")
        self.labels.events.append({
            "event_id": f"EV{k + 1:04d}", "anomaly_type": etype, "family": family,
            "verdict": verdict, "layer": layer, "scope": scope,
            "meters": ";".join(f"M{m + 1:03d}" for m in meters),
            "n_meters": len(meters),
            "heads": ";".join(f"F{f + 1}-{PHASES[p]}" for f, p in heads),
            "transformer": bool(tx),
            "channels": ";".join(channels),
            "start": self.idx[s], "end": self.idx[min(e, self.T) - 1] + pd.Timedelta(minutes=15),
            "start_idx": s, "end_idx": e, "duration_intervals": dur, "split": split,
            "held_out": etype in self.cfg.held_out_types,
            "params": json.dumps(params or {}, default=float),
        })
        return k


# ----------------------------------------------------------------------------- physical layer
def schedule_physical(sch: Scheduler, phys: PhysicalInputs) -> None:
    cfg, rng, topo = sch.cfg, sch.rng, sch.topo
    all_m = np.arange(sch.H)
    all_heads = [(f, p) for f in range(topo.n_feeders) for p in range(3)]

    def feeder_meters(f):
        return np.where(topo.feeder == f)[0]

    for _ in range(cfg.n_voltage_sag):
        s = sch.window(1, all_m, heads=all_heads)
        if s is None:
            continue
        depth = rng.uniform(0.15, 0.5)
        phys.sag_depth[s] = depth
        phys.v_src_scale[s] *= 1 - rng.uniform(0.002, 0.01)   # a ~1 s dip barely moves the 15-min mean
        sch.add("voltage_sag", s, 1, all_m, all_heads, scope="transformer",
                params={"depth_of_min_voltage": depth}, channels=["voltage_min_v"])

    for _ in range(cfg.n_mv_undervoltage):
        dur = int(rng.integers(2, 13))
        s = sch.window(dur, all_m, heads=all_heads)
        if s is None:
            continue
        drop = rng.uniform(0.04, 0.09)
        phys.v_src_scale[s:s + dur] *= 1 - drop
        sch.add("mv_undervoltage", s, dur, all_m, all_heads, scope="transformer",
                params={"mv_drop_pu": drop}, channels=["voltage_v"])

    for _ in range(cfg.n_tap_swell):
        dur = int(rng.integers(4, 33))
        s = sch.window(dur, all_m, heads=all_heads, hours=range(9, 15))
        if s is None:
            continue
        rise = rng.uniform(0.04, 0.07)
        phys.v_src_scale[s:s + dur] *= 1 + rise
        sch.add("tap_swell", s, dur, all_m, all_heads, scope="transformer",
                params={"mv_rise_pu": rise}, channels=["voltage_v"])

    for _ in range(cfg.n_outage):
        dur, pickup = int(rng.integers(2, 17)), 8
        whole = rng.random() < 0.3
        f = int(rng.integers(topo.n_feeders))
        meters = all_m if whole else feeder_meters(f)
        heads = all_heads if whole else [(f, p) for p in range(3)]
        s = sch.window(dur + pickup, meters, heads=heads, tx=whole)
        if s is None:
            continue
        if whole:
            phys.energized[s:s + dur] = False
        else:
            phys.energized[s:s + dur, f] = False
        k = np.arange(pickup)
        phys.load_scale[s + dur:s + dur + pickup][:, meters] *= (1 + 0.6 * np.exp(-k / 3))[:, None]
        sch.add("outage", s, dur + pickup, meters, heads, tx=whole,
                scope="transformer" if whole else f"F{f + 1}",
                params={"outage_intervals": dur, "cold_load_pickup_intervals": pickup},
                channels=["voltage_v", "current_a", "active_power_kw"])

    for _ in range(cfg.n_frequency_excursion):
        dur = int(rng.integers(1, 3))
        s = sch.window(dur, [], heads=all_heads)
        if s is None:
            continue
        df = -rng.uniform(0.3, 0.7)
        phys.freq_offset[s:s + dur] = df
        sch.add("frequency_excursion", s, dur, [], all_heads, scope="system",
                params={"delta_hz": df}, channels=["frequency_hz"])

    for _ in range(cfg.n_phase_loss):
        dur = int(rng.integers(2, 13))
        f, p = int(rng.integers(topo.n_feeders)), int(rng.integers(3))
        meters = np.where((topo.feeder == f) & (topo.phase == p))[0]
        s = sch.window(dur, meters, heads=[(f, p)])
        if s is None:
            continue
        phys.energized[s:s + dur, f, p] = False
        sch.add("phase_loss", s, dur, meters, [(f, p)], scope=f"F{f + 1}-{PHASES[p]}",
                params={"cause": "blown fuse"}, channels=["voltage_v", "current_a", "active_power_kw"])

    for _ in range(cfg.n_high_resistance_joint):
        dur = int(rng.integers(5, 21)) * DAY
        f, p = int(rng.integers(topo.n_feeders)), int(rng.integers(3))
        k = int(rng.integers(2, topo.n_poles - 2))
        meters = np.where((topo.feeder == f) & (topo.phase == p) & (topo.pole >= k))[0]
        s = sch.window(dur, meters, heads=[(f, p)])
        if s is None:
            continue
        r_max = rng.uniform(0.15, 0.5)
        ramp = np.linspace(0, 1, dur) ** 2 * r_max     # corrosion: slow, then accelerating
        phys.extra_r[s:s + dur, f, k - 1, p] += ramp
        sch.add("high_resistance_joint", s, dur, meters, [(f, p)],
                scope=f"F{f + 1}-pole{k}-{PHASES[p]}",
                params={"pole": k, "r_max_ohm": r_max}, channels=["voltage_v"])

    for _ in range(cfg.n_leakage):
        dur = int(rng.integers(3, 16)) * DAY
        m = int(rng.integers(sch.H))
        s = sch.window(dur, [m])
        if s is None:
            continue
        kw = rng.uniform(0.2, 0.8)
        grow = np.clip(np.linspace(0.3, 1.3, dur), 0, 1)
        phys.extra_p[s:s + dur, m] += kw * grow
        phys.extra_q[s:s + dur, m] += kw * grow * np.tan(np.arccos(0.5))
        sch.add("leakage", s, dur, [m], scope="home",
                params={"extra_kw": kw, "pf": 0.5}, channels=["current_a", "active_power_kw", "power_factor"])

    # rooftop PV faults: the panels really produce less, so the meter's net flow changes too
    # They start in val or test, alternating, so train stays clean for learning each home's output.
    pv_homes = np.where(sch.has_pv)[0]

    def pv_event(etype, n, dur_fn, apply, hours=None):
        for i in range(n if len(pv_homes) else 0):
            lo, hi = (sch.test_start, sch.T) if i % 2 else (sch.val_start, sch.test_start)
            m = int(rng.choice(pv_homes))
            dur = int(dur_fn())
            s = sch.window(dur, [m], lo=lo, hi=hi, hours=hours, local_only=True)
            if s is None:
                continue
            params = apply(m, s, dur)
            sch.add(etype, s, dur, [m], scope="home", params=params,
                    channels=["pv_power_kw", "active_power_kw", "energy_export_kwh"])

    def soiling(m, s, dur):
        loss = rng.uniform(0.10, 0.30)          # dust and sand build up, then cleaning or rain
        phys.pv_scale[s:s + dur, m] *= 1 - loss * np.linspace(0, 1, dur)
        return {"max_loss": loss, "recovery": "abrupt (cleaned or rain)"}

    def trip(m, s, dur):
        phys.pv_scale[s:s + dur, m] = 0.0
        return {"output": 0.0}

    def degradation(m, s, dur):
        loss = rng.uniform(0.15, 0.25)          # failed string or bypass diode: a step down
        phys.pv_scale[s:s + dur, m] *= 1 - loss
        return {"loss": loss, "until": "repaired at the end of the event"}

    pv_event("pv_soiling", cfg.n_pv_soiling, lambda: rng.integers(14, 41) * DAY, soiling)
    pv_event("pv_inverter_trip", cfg.n_pv_inverter_trip, lambda: rng.integers(8, 193), trip,
             hours=range(8, 14))
    pv_event("pv_degradation", cfg.n_pv_degradation, lambda: rng.integers(30, 61) * DAY, degradation)

    for _ in range(cfg.n_transformer_cooling):
        dur = int(rng.integers(5, 16)) * DAY
        s = sch.window(dur, [], lo=sch.val_start, tx=True)
        if s is None:
            continue
        fac = rng.uniform(1.3, 1.6)
        phys.cooling_factor[s:s + dur] = fac
        sch.add("transformer_cooling", s, dur, [], tx=True, scope="transformer",
                params={"rated_rise_factor": fac}, channels=["top_oil_temp_c"])


# -------------------------------------------------------------------------- measurement layer
SENSE = ["voltage_v", "voltage_min_v", "voltage_max_v", "current_a", "active_power_kw",
         "reactive_power_kvar"]
REPORT = SENSE + ["power_factor", "energy_import_kwh", "energy_export_kwh"]


def schedule_measurement(sch: Scheduler) -> list[tuple]:
    """Place measurement events; return (event_index, meter, s, e, type, params) to apply later."""
    cfg, rng, topo = sch.cfg, sch.rng, sch.topo
    out = []
    models = sch.meter_models

    def pick_meter(etype):
        if etype in ("drift_voltage", "drift_current") and rng.random() < cfg.drift_prone_share:
            cands = np.where(models == cfg.drift_prone_model)[0]
        else:
            cands = np.arange(sch.H)
        return int(rng.choice(cands))

    def place(etype, n, dur_fn, param_fn, channels, held_out=False, hours=None):
        for i in range(n):
            # alternate val / test so every non-held-out type is seen in both
            if held_out or i % 2:
                lo, hi = sch.test_start, sch.T
            else:
                lo, hi = sch.val_start, sch.test_start
            dur = int(dur_fn())
            m = pick_meter(etype)
            s = sch.window(dur, [m], lo=lo, hi=hi, hours=hours, local_only=True)
            if s is None:
                continue
            p = param_fn()
            k = sch.add(etype, s, dur, [m], scope="meter", params=p, channels=channels)
            out.append((k, m, s, s + dur, etype, p))

    sgn = lambda: float(rng.choice([-1, 1]))  # noqa: E731
    place("drift_voltage", cfg.n_drift_voltage, lambda: rng.integers(10, 46) * DAY,
          lambda: {"rate_pct_per_day": sgn() * rng.uniform(0.05, 0.5)},
          ["voltage_v", "active_power_kw"])
    place("drift_current", cfg.n_drift_current, lambda: rng.integers(10, 46) * DAY,
          lambda: {"rate_pct_per_day": sgn() * rng.uniform(0.05, 0.5)},
          ["current_a", "active_power_kw", "energy_import_kwh"])
    place("bias_voltage", cfg.n_bias_voltage, lambda: rng.integers(20, 61) * DAY,
          lambda: {"offset_v": sgn() * rng.uniform(2, 8)}, ["voltage_v"])
    place("bias_ct_gain", cfg.n_bias_ct_gain, lambda: rng.integers(20, 61) * DAY,
          lambda: {"gain": float(rng.choice([rng.uniform(0.88, 0.97), rng.uniform(1.03, 1.12)]))},
          ["current_a", "active_power_kw", "energy_import_kwh"])
    place("noise_growth", cfg.n_noise_growth, lambda: rng.integers(3, 15) * DAY,
          lambda: {"noise_multiplier": rng.uniform(3, 10)}, ["voltage_v", "current_a", "active_power_kw"])
    place("stuck", cfg.n_stuck, lambda: rng.integers(4, 193), lambda: {}, REPORT)
    place("spike", cfg.n_spike, lambda: 1,
          lambda: {"channel": str(rng.choice(["voltage_v", "current_a", "active_power_kw",
                                              "energy_import_kwh"]))}, [])
    place("dropout", cfg.n_dropout, lambda: rng.integers(1, 97), lambda: {}, REPORT)
    place("clock_skew", cfg.n_clock_skew, lambda: rng.integers(3, 21) * DAY,
          lambda: {"lag_intervals": int(rng.integers(1, 5))}, REPORT)
    place("theft_bypass", cfg.n_theft_bypass, lambda: rng.integers(3, 31) * DAY,
          lambda: {"fraction_metered": rng.uniform(0.3, 0.7),
                   "mode": str(rng.choice(["continuous", "evening"]))},
          ["current_a", "active_power_kw", "energy_import_kwh"])
    place("tamper_report", cfg.n_tamper_report, lambda: rng.integers(3, 31) * DAY,
          lambda: {"fraction_reported": rng.uniform(0.3, 0.7)},
          ["active_power_kw", "reactive_power_kvar", "energy_import_kwh"])
    place("meter_end_of_life", cfg.n_meter_end_of_life, lambda: rng.integers(10, 31) * DAY,
          lambda: {}, REPORT)
    place("replay", cfg.n_replay, lambda: rng.integers(1, 4) * DAY,
          lambda: {"replayed_from_days_before": 7}, REPORT, held_out=True)

    # inverter reading problems: the inverter reports something the panels did not do.
    # Only homes that share their inverter data have a reading that can go wrong.
    pv_homes = np.asarray(getattr(sch, "pv_optin", np.where(sch.has_pv)[0]), int)
    for etype, n, dur_fn, param_fn in [
        ("pv_reading_stuck", cfg.n_pv_reading_stuck, lambda: rng.integers(8, 97), lambda: {}),
        ("pv_inflated_report", cfg.n_pv_inflated_report, lambda: rng.integers(3, 21) * DAY,
         lambda: {"inflation": rng.uniform(1.2, 1.5)}),
    ]:
        for i in range(n if len(pv_homes) else 0):
            lo, hi = (sch.test_start, sch.T) if i % 2 else (sch.val_start, sch.test_start)
            m = int(rng.choice(pv_homes))
            dur = int(dur_fn())
            s = sch.window(dur, [m], lo=lo, hi=hi, local_only=True,
                           hours=range(8, 15) if etype == "pv_reading_stuck" else None)
            if s is None:
                continue
            p = param_fn()
            k = sch.add(etype, s, dur, [m], scope="inverter", params=p, channels=["pv_power_kw"])
            out.append((k, m, s, s + dur, etype, p))

    # coordinated false data injection: several meters on one feeder fake the same voltage rise
    for _ in range(cfg.n_coordinated_fdi):
        f = int(rng.integers(topo.n_feeders))
        cands = np.where(topo.feeder == f)[0]
        meters = rng.choice(cands, size=int(rng.integers(3, 9)), replace=False)
        dur = int(rng.integers(8, 49))
        s = sch.window(dur, meters, lo=sch.test_start, local_only=True)
        if s is None:
            continue
        p = {"offset_v": rng.uniform(8, 14), "current_adjusted_to_keep_power_identity": True}
        k = sch.add("coordinated_fdi", s, dur, meters, scope=f"F{f + 1}", params=p,
                    channels=["voltage_v", "current_a"])
        for m in meters:
            out.append((k, int(m), s, s + dur, "coordinated_fdi", p))
    return out


def apply_sensing(ev: list[tuple], meas: dict, rng: np.random.Generator) -> None:
    """Faults inside the meter's measurement chain: they also feed the energy registers."""
    for k, m, s, e, etype, p in ev:
        n = e - s
        V, I, P, Q = (meas[c][s:e, m] for c in ("voltage_v", "current_a", "active_power_kw",
                                                "reactive_power_kvar"))
        if etype == "drift_voltage":
            g = 1 + p["rate_pct_per_day"] / 100 * np.arange(n) / DAY
            for c in ("voltage_v", "voltage_min_v", "voltage_max_v", "active_power_kw",
                      "reactive_power_kvar"):
                meas[c][s:e, m] *= g
        elif etype == "drift_current":
            g = 1 + p["rate_pct_per_day"] / 100 * np.arange(n) / DAY
            for c in ("current_a", "active_power_kw", "reactive_power_kvar"):
                meas[c][s:e, m] *= g
        elif etype == "bias_voltage":
            g = np.where(V > 1, (V + p["offset_v"]) / np.maximum(V, 1), 1.0)
            for c in ("voltage_v", "voltage_min_v", "voltage_max_v"):
                meas[c][s:e, m] = np.where(V > 1, meas[c][s:e, m] + p["offset_v"], 0)
            meas["active_power_kw"][s:e, m] *= g
            meas["reactive_power_kvar"][s:e, m] *= g
        elif etype == "bias_ct_gain":
            for c in ("current_a", "active_power_kw", "reactive_power_kvar"):
                meas[c][s:e, m] *= p["gain"]
        elif etype == "noise_growth":
            sd = p["noise_multiplier"]
            meas["voltage_v"][s:e, m] *= 1 + rng.normal(0, 0.001 * sd, n)
            meas["current_a"][s:e, m] *= 1 + rng.normal(0, 0.003 * sd, n)
            meas["active_power_kw"][s:e, m] *= 1 + rng.normal(0, 0.003 * sd, n)
        elif etype == "theft_bypass":
            a = np.full(n, p["fraction_metered"])
            if p["mode"] == "evening":
                hrs = meas["_hour"][s:e]
                a = np.where((hrs >= 17) & (hrs < 23), a, 1.0)
            pos = P > 0                                   # bypass only hides imported energy
            for c in ("current_a", "active_power_kw", "reactive_power_kvar"):
                meas[c][s:e, m] = np.where(pos, meas[c][s:e, m] * a, meas[c][s:e, m])
        elif etype == "tamper_report":
            for c in ("active_power_kw", "reactive_power_kvar"):
                meas[c][s:e, m] *= p["fraction_reported"]
            meas["_pf_frozen"][s:e, m] = True             # PF keeps reporting the true value
        elif etype == "meter_end_of_life":
            sd = np.linspace(1, 8, n)
            meas["voltage_v"][s:e, m] *= 1 + rng.normal(0, 0.001, n) * sd
            meas["current_a"][s:e, m] *= 1 + rng.normal(0, 0.003, n) * sd
            meas["active_power_kw"][s:e, m] *= 1 + rng.normal(0, 0.003, n) * sd


def apply_reporting(ev: list[tuple], rep: dict, rng: np.random.Generator) -> None:
    """Faults between the meter and the head-end system: they change only the reported rows."""
    for k, m, s, e, etype, p in ev:
        n = e - s
        if etype == "stuck":
            for c in REPORT:
                rep[c][s:e, m] = rep[c][s - 1, m]
        elif etype == "dropout":
            for c in REPORT:
                rep[c][s:e, m] = np.nan
        elif etype == "spike":
            c = p["channel"]
            if c == "voltage_v":
                rep[c][s, m] *= rng.choice([rng.uniform(1.25, 1.6), rng.uniform(0.3, 0.6)])
            elif c in ("current_a", "active_power_kw"):
                rep[c][s, m] = rep[c][s, m] * rng.uniform(3, 10) + rng.uniform(5, 30) * (c == "current_a")
            else:
                rep[c][s, m] += 2.0 ** int(rng.integers(10, 17))   # register bit flip
        elif etype == "clock_skew":
            lag = p["lag_intervals"]
            for c in REPORT:
                rep[c][s:e, m] = rep[c][s - lag:e - lag, m].copy()
        elif etype == "replay":
            back = p["replayed_from_days_before"] * DAY
            for c in REPORT:
                rep[c][s:e, m] = rep[c][s - back:e - back, m].copy()
            base = rep["energy_import_kwh"][s - 1, m]
            rep["energy_import_kwh"][s:e, m] = base + np.cumsum(np.clip(rep["active_power_kw"][s:e, m], 0, None)) * 0.25
            base = rep["energy_export_kwh"][s - 1, m]
            rep["energy_export_kwh"][s:e, m] = base + np.cumsum(np.clip(-rep["active_power_kw"][s:e, m], 0, None)) * 0.25
        elif etype == "coordinated_fdi":
            ramp = np.minimum(1, np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)) / 4)
            d = p["offset_v"] * ramp
            V = rep["voltage_v"][s:e, m]
            for c in ("voltage_v", "voltage_min_v", "voltage_max_v"):
                rep[c][s:e, m] += d
            rep["current_a"][s:e, m] *= V / (V + d)
        elif etype == "meter_end_of_life":
            # last 60%: intermittent dropouts becoming more frequent; last 20%: dead
            a, b = s + int(0.4 * n), s + int(0.8 * n)
            prob = np.linspace(0.05, 0.6, b - a)
            gone = rng.random(b - a) < prob
            for c in REPORT:
                rep[c][a:b, m] = np.where(gone, np.nan, rep[c][a:b, m])
                rep[c][b:e, m] = np.nan


def apply_inverter(ev: list[tuple], rep: dict) -> None:
    """Problems with the inverter's production reading: only `pv_power_kw` changes."""
    pv = rep["pv_power_kw"]
    for k, m, s, e, etype, p in ev:
        if etype == "pv_reading_stuck":
            pv[s:e, m] = pv[s - 1, m]
        elif etype == "pv_inflated_report":
            pv[s:e, m] *= p["inflation"]
