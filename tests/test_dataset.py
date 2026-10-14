"""Physics and label sanity checks on a short simulation."""
import numpy as np


def test_clean_power_identity(small_sim):
    t = small_sim["true"]
    s = t["voltage_v"] * t["current_a"] / 1000
    ok = s > 0.05
    lhs = np.hypot(t["active_power_kw"], t["reactive_power_kvar"])[ok]
    assert np.allclose(lhs, s[ok], rtol=1e-6)


def test_feeder_energy_balance(small_sim):
    head_p = small_sim["head"]["active_power_kw"].sum(axis=(1, 2))
    homes_p = small_sim["true"]["active_power_kw"].sum(axis=1)
    loss = small_sim["tx"]["losses_kw"].values
    resid = head_p - homes_p - loss
    assert np.percentile(np.abs(resid), 99) < 0.25   # kW, within head-meter noise


def test_voltage_within_en50160_when_normal(small_sim):
    v = small_sim["true"]["voltage_v"]
    normal = small_sim["labels"].meter < 0
    vals = v[normal & (v > 1)]
    assert np.mean((vals > 207) & (vals < 253)) > 0.99


def test_every_event_is_labelled(small_sim):
    lab = small_sim["labels"]
    for k, e in enumerate(lab.events):
        hit = (lab.meter == k).any() or (lab.head == k).any() or (lab.tx == k).any()
        assert hit, e["event_id"]


def test_train_has_no_data_problems(small_sim):
    for e in small_sim["labels"].events:
        if e["split"] == "train":
            assert e["layer"] == "physical"


def test_held_out_only_in_test(small_sim):
    for e in small_sim["labels"].events:
        if e["held_out"]:
            assert e["split"] == "test"


def test_pv_reading_only_for_opt_in_homes(small_sim):
    """STEG installs only a net meter: the inverter reading exists only where the owner shares it."""
    pv = small_sim["rep"]["pv_power_kw"]
    has = np.array([h.has_pv for h in small_sim["homes"]])
    optin = np.isin(np.arange(len(has)), small_sim["pv_optin"])
    assert optin.sum() == small_sim["cfg"].n_pv_optin and has[optin].all()
    assert np.isnan(pv[:, ~optin]).all()
    assert np.isfinite(pv[:, optin]).all()


def test_collection_loss_is_logged(small_sim):
    lost, log = small_sim["collection_lost"], small_sim["collection_log"]
    assert 0.03 < lost.mean() < 0.2
    assert set(log.reason) == {"read_failed", "concentrator_offline"}
    assert np.isnan(small_sim["rep"]["voltage_v"][lost]).all()


def test_alarms_follow_power_cuts(small_sim):
    al = small_sim["alarms"]
    assert (al.alarm == "power_cut").sum() == (al.alarm == "power_restore").sum() > 0


def test_pv_stops_when_phase_is_off(small_sim):
    t = small_sim["true"]
    has = np.array([h.has_pv for h in small_sim["homes"]])
    off = t["voltage_v"][:, has] < 1
    assert np.all(t["pv_power_kw"][:, has][off] == 0)


def test_pv_events_only_on_solar_homes(small_sim):
    has = {h.meter_id for h in small_sim["homes"] if h.has_pv}
    for e in small_sim["labels"].events:
        if e["anomaly_type"].startswith("pv_"):
            assert set(e["meters"].split(";")) <= has
            assert e["split"] != "train"
