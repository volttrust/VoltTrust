"""Operator dashboard wording (EN + FR) and its link to the decision log."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "escalation")]
import explain as X  # noqa: E402

T = X.load_templates()
LANGS = ("en", "fr")


def test_every_template_has_both_languages():
    for key, tpl in T["explain"].items():
        for line in ("headline", "saw", "why", "check"):
            assert set(tpl[line]) == set(LANGS), (key, line)
            en, fr = (set(re.findall(r"{(\w+)}", tpl[line][g])) for g in LANGS)
            assert en == fr, (key, line, en ^ fr)          # same facts in both languages
    for section in ("ui", "verdicts", "tiers", "bands", "top_signal_text", "notes", "rules", "choices"):
        for k, v in T[section].items():
            if isinstance(v, dict) and "en" in v:
                assert v.get("fr"), (section, k)


@pytest.fixture(scope="module")
def real():
    if not (ROOT / "results" / "alerts.csv").exists():
        pytest.skip("no results")
    al = X.read_alerts(ROOT / "results" / "alerts.csv")
    cols = ["timestamp", "meter_id", "feeder", "phase", "voltage_v", "voltage_min_v", "current_a",
            "active_power_kw", "reactive_power_kvar", "energy_import_kwh", "pv_power_kw"]
    d = X.Data(pd.read_parquet(ROOT / "data" / "meters.parquet", columns=cols),
               pd.read_parquet(ROOT / "data" / "feeder_head.parquet"), pd.read_csv(ROOT / "data" / "meters.csv"),
               pd.read_csv(ROOT / "data" / "alarms.csv", parse_dates=["timestamp"]), {"batch_hours": 12})
    return al, d


def test_every_rule_explains_in_both_languages(real):
    al, d = real
    for _, a in al.groupby("rule").head(1).iterrows():
        for lang in LANGS:
            e = X.explain(a, d, lang, T, al)
            text = " ".join([e["headline"], e["saw"], e["why"], e["check"], *e["notes"]])
            assert "{" not in text and "}" not in text, (a.rule, lang, text)
            assert X.evidence(a, d, lang, T), a.rule
            sig, _ = X.signal_frame(a, d, T, lang)
            assert sig.value.notna().any(), a.rule


def test_energy_gap_names_homes_to_check_never_a_culprit(real):
    al, d = real
    for _, a in al[al.subtype == "energy_gap"].head(10).iterrows():
        e = X.explain(a, d, "en", T, al)
        assert not re.search(r"M\d{3}", e["headline"] + e["saw"])   # no single home in the verdict
        assert "Homes to check first" in e["why"] and "not proof" in e["why"]
        f = X.explain(a, d, "fr", T, al)
        assert "Foyers à vérifier en priorité" in f["why"] and "pas une preuve" in f["why"]


def test_buttons_write_to_the_hash_chained_log(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from router import Router
    monkeypatch.setenv("VOLTTRUST_STATE", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.query_params["alert"] = "AL00221"
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    at.button[labels.index("Override")].click().run()          # no reason: refused
    assert at.warning and not at.success
    at.text_input(key="reason_AL00221").input("meter replaced yesterday").run()
    at.button[labels.index("Override")].click().run()
    assert at.success
    r = Router(ROOT / "escalation" / "policy.yaml", tmp_path / "decision_log.sqlite")
    h = r.history("AL00221")
    assert [x["actor"] for x in h] == ["ai", "operator:op-demo"]
    assert h[1]["rule"] == "override" and h[1]["payload"]["reason"] == "meter replaced yesterday"
    assert h[1]["payload"]["snapshot_ref"]["sha256"] == h[0]["payload"]["snapshot_ref"]["sha256"]
    assert (tmp_path / "snapshots" / "AL00221.csv.gz").exists()
    assert r.verify_log()


def test_pilot_mode_records_answers_outside_the_log(tmp_path, monkeypatch):
    """12 fixed items, explanations hidden for half by group, answers to their own file."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    items = pd.read_csv(ROOT / "app" / "pilot_alerts.csv")
    assert list(items.columns) == ["item", "set", "alert_id"] and len(items) == 12   # no answer key
    answers = tmp_path / "answers.csv"
    monkeypatch.setenv("VOLTTRUST_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("VOLTTRUST_PILOT_ANSWERS", str(answers))
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.query_params.update({"pilot": "1", "group": "2", "participant": "P1"})
    at.run()
    assert not at.exception
    # item 1 is set A: group 2 sees it without explanation (no "What we saw", no evidence table)
    assert not any("What we saw" in m.value for m in at.markdown)
    at.button[0].click().run()                          # submit with nothing picked: refused
    assert at.warning and not answers.exists()
    r = [x for x in at.radio if x.label.startswith("What really")][0]
    r.set_value("data_quality")
    [x for x in at.radio if x.label.startswith("Your decision")][0].set_value("escalate")
    [x for x in at.radio if x.label.startswith("I understood")][0].set_value(4)
    at.button[0].click().run()
    assert not at.exception
    got = pd.read_csv(answers)
    assert got.iloc[0][["participant", "group", "item", "explanation_shown", "call", "action", "clarity"]].tolist() \
        == ["P1", 2, 1, "N", "data_quality", "escalate", 4]
    assert got.seconds.iloc[0] >= 0
    # item 2 is set B: group 2 sees its explanation
    assert any("What we saw" in m.value for m in at.markdown)
    assert not (tmp_path / "state" / "decision_log.sqlite").exists()   # pilot never touches the log
