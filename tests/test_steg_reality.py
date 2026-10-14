"""Running on the data STEG really has: fallbacks, phase inference, batch timing,
overload estimate, estimated solar, dust vs aging."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from volttrust.detect import detectors, features, phase
from volttrust.detect.site import Site
from volttrust.sim.simulate import write


def test_registers_rebuild_missing_power():
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.5, 2.0, (200, 2))
    e_imp = np.cumsum(np.clip(P, 0, None), 0) * 0.25
    e_exp = np.cumsum(np.clip(-P, 0, None), 0) * 0.25
    Pm, Ei, Ee = P.copy(), e_imp.copy(), e_exp.copy()
    for a in (Pm, Ei, Ee):
        a[50:98, 0] = np.nan                             # one lost 12 h batch
    out = features.fill_power_from_registers(Pm, Ei, Ee)
    assert np.isfinite(out[50:98, 0]).all()
    assert out[50:98, 0].sum() == pytest.approx(P[50:98, 0].sum(), rel=1e-9)
    assert np.array_equal(out[:, 1], P[:, 1])


def test_stand_in_head_from_homes():
    T = 4
    m = {c: np.ones((T, 4)) for c in features.HEAD_CHANNELS + ["voltage_min_v"]}
    m["voltage_v"] = np.array([[230, 232, 240, 228.0]] * T)
    m["active_power_kw"] = np.array([[1, 2, 3, 4.0]] * T)
    h = features.pseudo_head(m, np.zeros(4, int), np.array([0, 0, 1, -1]), 1)
    assert h["active_power_kw"][0, 0, 0] == 3 and h["active_power_kw"][0, 0, 1] == 3
    assert h["voltage_v"][0, 0, 0] == 231
    assert np.isnan(h["voltage_v"][0, 0, 2]) and np.isnan(h["frequency_hz"]).all()


def _fake_tx(load_pu, amb=30.0):
    T = len(load_pu)
    idx = pd.date_range("2025-07-01", periods=T, freq="15min")
    P = np.zeros((T, 2, 3))
    P[:, 0, 0] = load_pu * 100.0
    return SimpleNamespace(h={"active_power_kw": P, "reactive_power_kvar": np.zeros_like(P)}, kva=100.0,
                           idx=idx, ctx=pd.DataFrame({"ambient_temp_c": amb}, index=idx))


def test_overload_notice_needs_more_than_one_hour():
    load = np.full(96, 0.5)
    load[10:14] = 1.2                                   # 1 h exactly: no notice
    load[40:46] = 1.2                                   # 1.5 h: notice
    notice, alarm, _, oil = detectors.overload_flags(_fake_tx(load), detectors.Params())
    assert not notice[:40].any() and notice[45]
    assert not alarm.any() and oil.max() < 105


def test_overload_alarm_on_estimated_top_oil():
    notice, alarm, _, oil = detectors.overload_flags(_fake_tx(np.full(96 * 2, 1.6), amb=40.0), detectors.Params())
    assert notice.any() and alarm.any() and oil.max() > 105


def test_alarms_arrive_at_once_profiles_at_the_next_batch():
    assert detectors.detected_index("tamper_alarm", 10, 1000, 48) == 11
    assert detectors.detected_index("head_outage", 10, 1000, 48) == 11
    assert detectors.detected_index("head_outage", 10, 1000, 48, fast_ok=False) == 48
    assert detectors.detected_index("voltage_offset", 47, 1000, 48) == 48
    assert detectors.detected_index("voltage_offset", 48, 1000, 48) == 96


def test_phase_inferred_from_voltage_and_power():
    rng = np.random.default_rng(1)
    T, H = 96 * 10, 9
    true = np.repeat([0, 1, 2], 3)
    P = np.abs(rng.normal(0.5, 0.4, (T, H)))
    head_p = np.stack([P[:, true == p].sum(1) for p in range(3)], 1)[:, None, :]
    common = np.cumsum(rng.normal(0, 0.3, T))[:, None]
    head_v = 230 + common + np.cumsum(rng.normal(0, 0.3, (T, 3)), 0) - 0.5 * head_p[:, 0, :]
    V = head_v[:, true] - 0.3 * P + rng.normal(0, 0.1, (T, H))
    ph, margin, _ = phase.infer(V, P, np.zeros(H, int), head_v[:, None, :], head_p, slice(0, T))
    assert (ph == true).all() and (margin > 0.1).all()


def test_dust_confirmed_by_recovery_else_aging():
    T = 96 * 80
    ch = np.zeros((T, 2))
    ch[96 * 20, 0] = 0.15                               # home 0 jumps back up after 20 days
    al = pd.DataFrame({"rule": "pv_underperform", "asset": ["M001", "M002"], "start_idx": [0, 0],
                       "end_idx": [96 * 20, 96 * 70], "subtype": "pv_soiling", "pv_ratio_change_1d": [-0.01, -0.01]})
    D = SimpleNamespace(meters=["M001", "M002"], idx=pd.date_range("2025-01-01", periods=T, freq="15min"), T=T)
    out = detectors._panel_dust_or_aging(al, {"pv_ratio_change_1d": ch}, D, detectors.Params())
    assert list(out.subtype) == ["pv_soiling", "pv_aging_inspect"]
    assert out.pv_resolved_at[0] == D.idx[96 * 20]


@pytest.fixture(scope="module")
def small_dir(small_sim, tmp_path_factory):
    d = tmp_path_factory.mktemp("small")
    write(small_sim, d)
    return d


@pytest.mark.parametrize("site", [Site(), Site(head_meter="none"), Site(phase_source="inferred"),
                                  Site(pv_source="estimated", use_reference_panel=False)],
                         ids=["concentrator", "no_head", "inferred_phase", "estimated_solar"])
def test_detector_runs_with_any_equipment(small_dir, site):
    D = features.load(small_dir, site)
    R = features.residuals(D)
    det = detectors.detect(D, R, use_iforest=False)
    al = det.alerts
    assert len(al) and (al.detected_idx >= al.start_idx).all()
    gap = al.rule.isin(detectors.PHASE_RULES)
    if not site.has_head:
        assert not gap.any()
    # collection gaps the head-end logged never raise a meter fault
    assert not (det.flags["dropout"] & ~D.collected).any()
    if site.phase_source == "inferred":
        topo = pd.DataFrame(json.loads((small_dir / "topology.json").read_text())["meters"]).set_index("meter_id")
        true = topo.loc[D.meters, "phase"].map({"A": 0, "B": 1, "C": 2}).to_numpy()
        assert (D.phase[D.phase_known] == true[D.phase_known]).mean() >= 0.95
