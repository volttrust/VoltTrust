"""What the decision log records about each decision.

Versions of the detector code and of the confidence table, the data window a verdict was
made on, and a snapshot of the raw readings in that window. The snapshot is stored outside
the hash-chained log (see escalation/router.py SnapshotStore) so that deleting it after
1 year never breaks the chain. Households appear by meter number only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from volttrust import __version__

DETECT_DIR = Path(__file__).with_name("detect")
# columns a replay needs; ground-truth labels from the simulator are never copied
METER_COLS = ["timestamp", "meter_id", "feeder", "phase", "voltage_v", "voltage_min_v", "voltage_max_v",
              "current_a", "active_power_kw", "reactive_power_kvar", "power_factor",
              "energy_import_kwh", "energy_export_kwh", "pv_power_kw"]
HEAD_COLS = ["timestamp", "feeder", "phase", "voltage_v", "voltage_min_v", "current_a", "active_power_kw",
             "reactive_power_kvar", "power_factor", "frequency_hz", "thd_v_pct"]
HISTORY = pd.Timedelta(hours=24)   # readings kept before the alert starts


def detector_version() -> str:
    """Package version plus a hash of the detection code, so two runs with different code
    never share a version string."""
    h = hashlib.sha256()
    for p in sorted(DETECT_DIR.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return f"volttrust {__version__} detect:{h.hexdigest()[:12]}"


def calibration_version(fitted_models: dict | str | Path) -> str | None:
    """Fit date of the per-rule confidence table plus a short hash of the table itself."""
    if not isinstance(fitted_models, dict):
        fitted_models = json.loads(Path(fitted_models).read_text())
    cal = fitted_models.get("confidence_by_rule")
    if not cal:
        return None
    digest = hashlib.sha256(json.dumps(cal["table"], sort_keys=True).encode()).hexdigest()[:8]
    return f"{cal['fit_date']} table:{digest}"


def data_window(alert: dict | pd.Series, meters: pd.DataFrame) -> dict:
    """Assets and time span a verdict was made on: the alert's asset and its same-phase
    neighbours (what the checks compare against), from 24 h before the alert to the moment
    the operator could see it. Nothing after that moment is included."""
    a = dict(alert)
    feeder, phase = a["feeder"], a["phase"]
    same = meters[(meters.feeder == feeder) & (meters.phase == phase)] if isinstance(phase, str) \
        else meters[meters.feeder == feeder]
    assets = sorted(set(same.meter_id)) if len(same) else []
    if str(a["asset"]).startswith("M") and a["asset"] not in assets:
        assets = sorted(assets + [a["asset"]])
    end = pd.Timestamp(a.get("detected_at") or a["t_end"])
    end = max(end, pd.Timestamp(a["t_end"]))
    return {"alert_asset": a["asset"], "feeder": feeder, "phase": phase if isinstance(phase, str) else None,
            "assets": assets, "start": str(pd.Timestamp(a["t_start"]) - HISTORY), "end": str(end)}


def snapshot_frame(win: dict, readings: pd.DataFrame, head: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw readings in the data window: the listed meters plus the substation head on that
    feeder (and phase). One table, `source` says which."""
    t0, t1 = pd.Timestamp(win["start"]), pd.Timestamp(win["end"])
    m = readings[readings.meter_id.isin(win["assets"])
                 & (readings.timestamp >= t0) & (readings.timestamp <= t1)]
    parts = [m[[c for c in METER_COLS if c in m.columns]].assign(source="meter")]
    if head is not None:
        h = head[(head.feeder == win["feeder"]) & (head.timestamp >= t0) & (head.timestamp <= t1)]
        if win.get("phase"):
            h = h[h.phase == win["phase"]]
        parts.append(h[[c for c in HEAD_COLS if c in h.columns]].assign(source="head"))
    return pd.concat(parts, ignore_index=True).sort_values(["source", "timestamp"], kind="stable")


def snapshot_bytes(frame: pd.DataFrame) -> bytes:
    """CSV bytes, the same every time for the same readings."""
    return frame.to_csv(index=False, float_format="%.6g").encode()
