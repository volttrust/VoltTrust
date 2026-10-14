"""Versions, data window and raw-readings snapshot recorded with each decision."""
import sys
from pathlib import Path

import pandas as pd

from volttrust import provenance as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "escalation"))
from router import Event, Router, SnapshotStore  # noqa: E402

POLICY = Path(__file__).resolve().parents[1] / "escalation" / "policy.yaml"


def _readings():
    ts = pd.date_range("2025-05-08", "2025-05-11", freq="15min")
    rows = [{"timestamp": t, "meter_id": m, "feeder": "F2", "phase": ph, "voltage_v": 230.0,
             "active_power_kw": 0.5, "is_anomaly": 0, "anomaly_type": "normal"}
            for t in ts for m, ph in (("M036", "A"), ("M037", "A"), ("M038", "B"))]
    return pd.DataFrame(rows)


def test_window_snapshot_and_log_round_trip(tmp_path):
    D = _readings()
    alert = {"asset": "F2-A", "feeder": "F2", "phase": "A", "t_start": "2025-05-10 06:00",
             "t_end": "2025-05-10 06:15", "detected_at": "2025-05-10 12:00"}
    w = P.data_window(alert, D)
    assert w["assets"] == ["M036", "M037"]                    # same phase only
    assert w["start"] == "2025-05-09 06:00:00" and w["end"] == "2025-05-10 12:00:00"
    f = P.snapshot_frame(w, D)
    assert f.timestamp.max() <= pd.Timestamp(w["end"])        # nothing the operator could not see yet
    assert "is_anomaly" not in f and "anomaly_type" not in f  # simulator labels never stored
    b = P.snapshot_bytes(f)
    assert b == P.snapshot_bytes(P.snapshot_frame(w, D))      # same readings, same hash

    store = SnapshotStore(tmp_path)
    ref = store.save("AL1", b)
    r = Router(POLICY, versions={"detector_version": P.detector_version(),
                                 "calibration_version": "2026-10-13 table:0"})
    r.route(Event("AL1", "data_quality", "energy_gap", 0.8), data_window=w, snapshot_ref=ref)
    assert store.load(ref) == b
    assert r.last_decision("AL1")["payload"]["detector_version"].startswith("volttrust ")
    assert r.verify_log()
