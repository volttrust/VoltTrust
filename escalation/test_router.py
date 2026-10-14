from pathlib import Path

import pytest

from router import Event, Router

POLICY = Path(__file__).with_name("policy.yaml")


@pytest.fixture
def r():
    return Router(POLICY)


@pytest.mark.parametrize("event,tier", [
    (Event("a", "data_quality", "sensor_drift", 0.95), "T1"),
    (Event("b", "data_quality", "sensor_drift", 0.75), "T2"),
    (Event("c", "data_quality", "sensor_drift", 0.50), "T3"),
    (Event("d", "data_quality", "sensor_drift", 0.20), "T0"),
    (Event("e", "grid_event", "surge", 0.95), "T2"),
    (Event("f", "grid_event", "surge", 0.60), "T3"),
    (Event("f2", "grid_event", "sag", 0.10), "T3"),
    (Event("f3", "equipment_fault", "transformer_overheat", 0.90), "T2"),
    (Event("f4", "equipment_fault", "phase_loss", 0.70), "T3"),
    (Event("f5", "data_quality", "bias", 0.99, uncertain=True), "T3"),
    (Event("f6", "data_quality", "theft", 0.99), "T3"),
    (Event("f7", "data_quality", "under_reporting", 0.99), "T3"),
    (Event("f8", "data_quality", "over_reporting", 0.95), "T1"),
    (Event("g", "data_quality", "falsified", 0.99), "T3"),
    (Event("h", "data_quality", "x", 0.99, novelty=0.9), "T3"),
    (Event("i", "data_quality", "x", 0.99, detectors_agree=False), "T3"),
    (Event("j", "grid_event", "outage", 0.99, proposed_action="load_shedding"), "T3"),
    (Event("k", "data_quality", "x", 0.99, customers_affected=50), "T3"),
    (Event("l", "grid_event", "overvoltage_trip", 0.5), "T4"),
    (Event("g1", "data_quality", "energy_gap", 0.97, correlated_meters=1), "T3"),
    (Event("g2", "equipment_fault", "pv_aging_inspect", 0.97), "T3"),
    (Event("u1", "data_quality", "unusual_pattern", 0.91), "T2"),    # forest alerts never act silently
    (Event("d1", "equipment_fault", "pv_soiling_likely", 0.55), "T2"),   # dust notifies the owner
    (Event("d2", "equipment_fault", "pv_soiling_likely", 0.55, correlated_meters=6), "T3"),   # ... unless a hard rule applies
])
def test_routing(r, event, tier):
    assert r.route(event).tier == tier


def test_safe_mode_sends_everything_to_human(r):
    r.policy["safe_mode"] = True
    assert r.route(Event("a", "data_quality", "sensor_drift", 0.99)).tier == "T3"


def test_circuit_breaker(r):
    limit = r.policy["circuit_breaker"]["max_auto_actions"]
    for i in range(limit):
        assert r.route(Event(f"e{i}", "data_quality", "drift", 0.95)).tier == "T1"
    assert r.route(Event("over", "data_quality", "drift", 0.95)).tier == "T3"


def test_override_needs_reason(r):
    with pytest.raises(ValueError):
        r.operator_decision("a", "op", "override")


def test_log_tamper_detected(r):
    r.route(Event("a", "data_quality", "drift", 0.95))
    r.route(Event("b", "data_quality", "drift", 0.95))
    assert r.verify_log()
    r.db.execute("UPDATE decision_log SET tier='T0' WHERE event_id='a'")
    assert not r.verify_log()


def test_latency_under_budget(r):
    d = r.route(Event("a", "data_quality", "drift", 0.95))
    assert d.latency_ms < 1000


def test_from_goal1_verdict(r):
    rec = {"event_id": "x", "assets": ["meter_017"], "verdict": "data_quality",
           "subtype": "bias", "confidence": 0.93,
           "evidence": {"rules_fired": ["voltage_peer_offset"], "neighbors_affected": 0.02}}
    assert r.route(Event.from_verdict(rec)).tier == "T1"
    rec["evidence"]["neighbors_affected"] = 0.6        # 30 of 50 meters -> human
    assert r.route(Event.from_verdict(rec)).tier == "T3"


# ---- what each decision records ----------------------------------------------
VERSIONS = {"calibration_version": "2026-10-13 table:abcd1234", "detector_version": "volttrust 0.1.0 detect:x"}
WINDOW = {"alert_asset": "M037", "feeder": "F2", "phase": "A", "assets": ["M036", "M037"],
          "start": "2025-05-09 06:00:00", "end": "2025-05-10 12:00:00"}


def test_every_row_records_versions_window_and_snapshot(tmp_path):
    from router import SnapshotStore
    r = Router(POLICY, versions=VERSIONS)
    ref = SnapshotStore(tmp_path).save("a", b"timestamp,meter_id,voltage_v\n2025-05-10,M037,231.0\n")
    r.route(Event("a", "data_quality", "drift", 0.95), data_window=WINDOW, snapshot_ref=ref)
    r.operator_decision("a", "op1", "approve")
    rows = r.history("a")
    assert [x["actor"] for x in rows] == ["ai", "operator:op1"]
    for x in rows:   # the operator row points at the same readings the AI routed on
        p = x["payload"]
        assert p["calibration_version"] == VERSIONS["calibration_version"]
        assert p["detector_version"] == VERSIONS["detector_version"]
        assert p["data_window"] == WINDOW
        assert p["snapshot_ref"]["sha256"] == ref["sha256"]
    assert r.verify_log()


def test_deleting_a_snapshot_keeps_the_chain_and_tampering_is_caught(tmp_path):
    from router import SnapshotStore
    store = SnapshotStore(tmp_path)
    r = Router(POLICY, versions=VERSIONS)
    ref = store.save("a", b"x,y\n1,2\n")
    r.route(Event("a", "data_quality", "drift", 0.95), data_window=WINDOW, snapshot_ref=ref)
    assert store.load(ref) == b"x,y\n1,2\n"
    Path(ref["path"]).write_bytes(Path(store.save("b", b"x,y\n1,3\n")["path"]).read_bytes())
    with pytest.raises(ValueError):          # edited readings no longer match the log
        store.load(ref)
    Path(ref["path"]).unlink()               # 1-year retention: raw readings go, the chain stays
    assert r.verify_log()


def test_log_refuses_anything_but_meter_numbers():
    r = Router(POLICY)
    with pytest.raises(ValueError):
        r.route(Event("a", "data_quality", "drift", 0.95),
                data_window={**WINDOW, "assets": ["Mohamed Ben Ali, Rue 12"]})
