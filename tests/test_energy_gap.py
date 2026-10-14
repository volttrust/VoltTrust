"""Energy-balance gaps raise an alert on the feeder-phase with homes to check, never on a home."""
import pandas as pd

from volttrust.detect import detectors, evaluate


def _phase_alert(asset="F2-A", start=10, end=30, suspects=None):
    return {"alert_id": "AL1", "asset": asset, "feeder": asset[:2], "phase": asset[-1],
            "start_idx": start, "end_idx": end,
            "t_start": pd.Timestamp("2025-08-01") + pd.Timedelta(minutes=15 * start),
            "t_end": pd.Timestamp("2025-08-01") + pd.Timedelta(minutes=15 * end),
            "verdict": "data_quality", "subtype": "energy_gap", "rule": "balance_gap_phase",
            "rules_fired": ["balance_gap_phase"], "confidence": 0.7, "neighbors_affected": 1.0,
            "gap_kw": 0.2, "suspects": suspects if suspects is not None else
            [{"meter": "M037", "beta": 0.3, "t": 9.1, "evidence": "strong"},
             {"meter": "M026", "beta": 0.05, "t": 2.0, "evidence": "weak"}]}


def _event(meter, layer="measurement", channels="current_a;active_power_kw;energy_import_kwh"):
    return {"event_id": "E1", "anomaly_type": "theft_bypass", "family": "falsified_reading",
            "verdict": "data_quality", "layer": layer, "meters": meter, "heads": None, "transformer": False,
            "channels": channels, "start": "2025-08-01 02:30", "duration_intervals": 40, "split": "test",
            "held_out": False}


IDX = pd.date_range("2025-08-01", periods=200, freq="15min")
PHASE_OF = {"M037": "F2-A", "M026": "F2-A", "M001": "F1-B"}


def test_no_home_level_balance_rule():
    names = [r[0] for r in detectors.RULES]
    assert "balance_attribution" not in names and "balance_slow" not in names
    for r in detectors.PHASE_RULES:
        assert detectors.RULES[detectors.RULE_IDX[r]][2] == "energy_gap"


def test_handoff_lists_suspects_on_the_phase():
    rec = detectors.to_verdict_records(pd.DataFrame([_phase_alert()]))[0]
    assert rec["assets"] == ["F2-A"]
    assert [s["meter"] for s in rec["evidence"]["suspects"]] == ["M037", "M026"]


def test_phase_alert_matches_only_events_on_its_phase():
    al = pd.DataFrame([_phase_alert()])
    res, _ = evaluate.match(al, pd.DataFrame([_event("M037")]), IDX, PHASE_OF)
    assert res.detected.iloc[0] and res.suspect_hit.iloc[0]
    res, _ = evaluate.match(al, pd.DataFrame([_event("M001")]), IDX, PHASE_OF)
    assert not res.detected.iloc[0]


def test_physical_events_are_not_credited_to_an_energy_gap():
    al = pd.DataFrame([_phase_alert()])
    ev = _event("M037", layer="physical", channels="pv_power_kw;active_power_kw;energy_export_kwh")
    res, _ = evaluate.match(al, pd.DataFrame([ev]), IDX, PHASE_OF)
    assert not res.detected.iloc[0]


def test_measured_confidence_per_rule():
    """(correct + 10 * hand-set) / (n + 10), capped at 0.97; agreeing alerts get their own cell."""
    import numpy as np
    from volttrust.detect import calibrate
    n = 30
    al = pd.DataFrame({"alert_id": [f"A{i}" for i in range(n)], "rule": "voltage_offset", "verdict": "data_quality",
                       "rules_fired": [["voltage_offset"]] * 20 + [["voltage_offset", "iforest"]] * 10,
                       "start_idx": np.arange(n), "matched_event": ["E1"] * 15 + [None] * 5 + ["E1"] * 10,
                       "confidence": 0.7, "confidence_hand_set": 0.7})
    res = pd.DataFrame({"event_id": ["E1"], "verdict": ["data_quality"]})
    cal = calibrate.fit(al, res, np.array(["val"] * n))
    row = cal["table"]["voltage_offset"]
    assert row["single"]["confidence"] == round((15 + 10 * 0.70) / 30, 3)
    assert row["agree"]["confidence"] == round(min(0.97, (10 + 10 * 0.74) / 20), 3)
    out = calibrate.apply(al.assign(rule=["voltage_offset"] * 29 + ["tamper_alarm"]), cal)
    assert out.confidence.iloc[-1] == 0.84          # a rule with no validation alerts keeps its prior (0.80 + 0.04)
    assert calibrate.ece([0.9] * 10, [1] * 9 + [0]) == 0.0


def test_shap_names_the_signal_behind_a_forest_alert():
    """Positive SHAP = pushed toward unusual; the top signal is the one that was off."""
    import numpy as np
    from sklearn.ensemble import IsolationForest
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    model = IsolationForest(n_estimators=100, random_state=0).fit(X)
    feats = np.zeros((10, 1, 4))
    feats[5, 0, 2] = 6.0                                # power-identity residual far off at row 5
    score = np.zeros((10, 1))
    score[5, 0] = 1.0
    al = pd.DataFrame({"rule": ["iforest"], "asset": ["M001"], "start_idx": [3], "end_idx": [8]})
    out, timing = detectors.explain_forest(al, {"model": model, "features": feats, "background": X[:200]},
                                           score, ["M001"])
    assert out.shap_top[0]["feature"] == "r_id" and out.shap_top[0]["value"] == 6.0
    assert out.shap[0]["r_id"] > 0 and timing["forest_alerts"] == 1


def test_explanation_quality_table():
    """Four measures with targets, both windows, SHAP match also per fault type."""
    import pandas as pd
    from pathlib import Path

    from volttrust.detect import quality as Q
    out = Path(__file__).resolve().parents[1] / "results"
    if not (out / "alerts.csv").exists():
        import pytest
        pytest.skip("no results")
    q = Q.quality(*Q.load_run(out), seed=42)
    top = q[q.fault_type == "all"]
    assert set(top.measure) == set(Q.TARGETS) and set(top.window) == {"val", "test"}
    assert top.value.between(0, 1).all()
    shap_rows = q[(q.measure == "shap_main_signal_match") & (q.fault_type != "all")]
    assert len(shap_rows) and shap_rows.note.str.len().gt(0).all()
    assert Q.expected_signal("spike", '{"channel": "energy_import_kwh"}') == "r_reg"
    assert Q.expected_signal("spike", '{"channel": "active_power_kw"}') is None
    saved = pd.read_csv(out / "explanation_quality.csv")
    s42 = saved[(saved.seed == 42) & (saved.fault_type == "all")].set_index(["window", "measure"]).value
    assert (s42 - top.set_index(["window", "measure"]).value).abs().max() < 1e-9
