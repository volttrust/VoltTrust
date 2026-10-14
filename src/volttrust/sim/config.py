"""Simulation parameters. Every number that shapes the dataset lives here."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SimConfig:
    seed: int = 42
    start: str = "2025-01-01 00:00"
    end: str = "2025-12-31 23:45"
    freq_min: int = 15
    tz_offset_h: int = 1                  # Tunisia: UTC+1 all year, no DST

    # Neighborhood
    n_feeders: int = 2
    poles_per_feeder: int = 12
    homes_per_pole: int = 2
    nominal_v: float = 230.0
    transformer_kva: float = 100.0
    transformer_z_pct: float = 4.0
    transformer_xr: float = 2.5
    tap_pu: float = 1.035                 # secondary set slightly high, common STEG practice
    span_km: float = 0.035
    r_phase_ohm_km: float = 0.443         # 70 mm2 Al twisted cable
    r_neutral_ohm_km: float = 0.630       # 54.6 mm2 Almelec neutral
    neutral_return_share: float = 0.5     # per-phase equivalent: part of the current returns via neutral
    x_ohm_km: float = 0.08
    drop_km: float = 0.025
    r_drop_ohm_km: float = 1.91           # 16 mm2 Al service drop, per conductor (phase + neutral)

    # Households
    p_ac: float = 0.65
    p_heater: float = 0.35
    p_water_heater: float = 0.15
    p_pv: float = 0.30
    p_stay_home: float = 0.35

    # Splits (by timestamp)
    train_end: str = "2025-04-30 23:45"
    val_end: str = "2025-07-31 23:45"

    # Events per year (physical layer, any split)
    n_voltage_sag: int = 12
    n_mv_undervoltage: int = 6
    n_tap_swell: int = 5
    n_outage: int = 6
    n_frequency_excursion: int = 4
    n_phase_loss: int = 3
    n_high_resistance_joint: int = 3
    n_leakage: int = 5
    n_transformer_cooling: int = 2
    # rooftop PV faults (physical, solar homes only, after the first month)
    n_pv_soiling: int = 3
    n_pv_inverter_trip: int = 4
    n_pv_degradation: int = 3

    # Measurement-layer events (val + test only; train holds no data problems)
    n_drift_voltage: int = 8
    n_drift_current: int = 8
    n_bias_voltage: int = 6
    n_bias_ct_gain: int = 6
    n_noise_growth: int = 6
    n_stuck: int = 15
    n_spike: int = 40
    n_dropout: int = 25
    n_clock_skew: int = 6
    n_theft_bypass: int = 6
    n_tamper_report: int = 6
    n_meter_end_of_life: int = 4
    # inverter reading problems (solar homes only)
    n_pv_reading_stuck: int = 4
    n_pv_inflated_report: int = 4
    # Held out: only scheduled in the test split
    n_replay: int = 4
    n_coordinated_fdi: int = 4
    held_out_types: list = field(default_factory=lambda: ["replay", "coordinated_fdi"])

    # What STEG's systems really provide. New randomness uses its own generator (seed + 7919)
    # so the events above keep their places.
    n_pv_optin: int = 3                   # solar homes sharing their inverter data (others: net meter only)
    reference_panel_kwp: float = 1.0      # one measured reference panel per neighborhood (0 = none)
    batch_hours: int = 12                 # concentrator reads the load profiles every 12 h
    collection_loss: tuple = (0.05, 0.15) # per-meter chance that a batch read fails (drawn per meter)
    n_concentrator_offline: int = 2       # whole batches lost for every meter
    p_theft_tamper_alarm: float = 0.5     # a bypass opens the terminal cover: tamper alarm at the start
    n_nuisance_tamper_alarms: int = 4     # cover or magnet alarms with no fraud behind them

    # Meter fleet: model B is a batch that drifts more (a "calibration insight" for Goal 4)
    meter_models: dict = field(default_factory=lambda: {"MA-100": 0.4, "MB-200": 0.35, "MC-300": 0.25})
    drift_prone_model: str = "MB-200"
    drift_prone_share: float = 0.7

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "SimConfig":
        return cls(**json.loads(Path(path).read_text()))
