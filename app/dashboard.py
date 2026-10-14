"""VoltTrust operator dashboard, one screen, English and French.

    streamlit run app/dashboard.py

Reads results/alerts.csv and the data files. Left: the alert queue. Right: the selected
alert's explanation, its raw signal, the evidence and the escalation tier. Approve /
Override / Escalate write a row to the router's hash-chained decision log; nothing is sent
to the network. Query parameters ?lang=fr&alert=AL00221 open a given view (screenshots).
"""
from __future__ import annotations

import json
import os
import re
import time
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "escalation", ROOT / "app"):
    sys.path.insert(0, str(p))

from volttrust import provenance  # noqa: E402
from explain import (Data, band, evidence, explain, headline_only, load_templates,  # noqa: E402
                     owner_message, read_alerts, signal_frame)
from router import Event, Router, SnapshotStore  # noqa: E402

RESULTS, DATA = ROOT / "results", ROOT / "data"
STATE = Path(os.environ.get("VOLTTRUST_STATE", ROOT / "app" / "state"))   # decision log + snapshots (not committed)
POLICY = ROOT / "escalation" / "policy.yaml"
PILOT_ITEMS = ROOT / "app" / "pilot_alerts.csv"   # item, set, alert_id only (never the answer key)
PILOT_ANSWERS = Path(os.environ.get("VOLTTRUST_PILOT_ANSWERS", STATE / "pilot_answers.csv"))
PILOT_COLUMNS = ["participant", "group", "lang", "item", "explanation_shown", "call", "action", "seconds", "clarity"]
READ_COLS = ["timestamp", "meter_id", "feeder", "phase", "voltage_v", "voltage_min_v", "voltage_max_v",
             "current_a", "active_power_kw", "reactive_power_kvar", "power_factor",
             "energy_import_kwh", "energy_export_kwh", "pv_power_kw"]

st.set_page_config(page_title="VoltTrust", page_icon="⚡", layout="wide")


@st.cache_data
def load_all():
    al = read_alerts(RESULTS / "alerts.csv")
    d = Data(pd.read_parquet(DATA / "meters.parquet", columns=READ_COLS),
             pd.read_parquet(DATA / "feeder_head.parquet"),
             pd.read_csv(DATA / "meters.csv"),
             pd.read_csv(DATA / "alarms.csv", parse_dates=["timestamp"]),
             json.loads((RESULTS / "summary.json").read_text())["site"])
    versions = {"detector_version": provenance.detector_version(),
                "calibration_version": provenance.calibration_version(RESULTS / "fitted_models.json")}
    return al, d, versions


def router(versions) -> Router:
    # one connection per rerun: Streamlit reruns on other threads, SQLite connections can't follow
    STATE.mkdir(parents=True, exist_ok=True)
    return Router(POLICY, STATE / "decision_log.sqlite", versions=versions)


def route_once(r: Router, a: pd.Series, d: Data, reason: str) -> dict:
    """The AI decision for this alert, routed and logged the first time it is opened, with its
    data window and a snapshot of the raw readings."""
    last = r.last_decision(a.alert_id, "ai")
    if last:
        return last
    win = provenance.data_window(a, d.meters)
    snap = provenance.snapshot_bytes(provenance.snapshot_frame(win, d.readings, d.head))
    ref = SnapshotStore(STATE / "snapshots").save(a.alert_id, snap)
    n_feeder = int((d.meters.feeder == a.feeder).sum())
    rec = {"event_id": a.alert_id, "verdict": a.verdict, "subtype": a.subtype,
           "confidence": float(a.confidence), "assets": [a.asset],
           "evidence": {"neighbors_affected": float(a.neighbors_affected)}}
    r.route(Event.from_verdict(rec, meters_on_feeder=n_feeder, reasons=[reason]),
            data_window=win, snapshot_ref=ref)
    return r.last_decision(a.alert_id, "ai")


def tier_reason(rule: str, T: dict, lang: str) -> str:
    rr = T["router_reasons"]
    if rule.endswith("; never silent"):
        return tier_reason(rule.removesuffix("; never silent"), T, lang) + (" ; " if lang == "fr" else "; ") + rr["never silent"][lang]
    if rule in rr:
        return rr[rule][lang]
    m = re.match(r"confidence ([\d.]+) >= ([\d.]+)", rule)
    if m:
        c, b = m.groups()
        s = rr["confidence_at_least"][lang].format(c=c, b=b)
        return s.replace(".", ",") if lang == "fr" else s
    m = re.match(r"confidence ([\d.]+) below", rule)
    if m:
        s = rr["confidence_below"][lang].format(c=m.group(1))
        return s.replace(".", ",") if lang == "fr" else s
    return rule


# ---- sidebar ------------------------------------------------------------------
T = load_templates()
qp = st.query_params
lang = st.sidebar.radio("Language / Langue", ["en", "fr"], horizontal=True,
                        index=1 if qp.get("lang") == "fr" else 0,
                        format_func=lambda x: {"en": "English", "fr": "Français"}[x])
ui = {k: (v[lang] if isinstance(v, dict) and lang in v else v) for k, v in T["ui"].items()}
al, d, versions = load_all()
pilot = st.sidebar.toggle(ui["pilot_mode"], value=qp.get("pilot") == "1")
if pilot:   # pilot mode: a small user study of the explanations
    group = st.sidebar.radio(ui["pilot_group"], [1, 2], horizontal=True, index=int(qp.get("group", 1)) - 1)
    participant = st.sidebar.text_input(ui["pilot_participant"], value=qp.get("participant", ""))
else:
    operator = st.sidebar.text_input(ui["operator"], value=qp.get("op", "op-demo"))
    verdicts = list(T["verdicts"])
    show = st.sidebar.multiselect(ui["filter_verdict"], verdicts, default=verdicts,
                                  format_func=lambda v: T["verdicts"][v][lang])
    bands = ["high", "medium", "low"]
    show_b = st.sidebar.multiselect(ui["filter_band"], bands, default=bands,
                                    format_func=lambda b: T["bands"][b][lang].capitalize())
    st.sidebar.caption(ui["no_action"])

# ---- pieces of the detail view (also used by pilot mode) -------------------------
def show_explanation(ex):
    st.markdown(f"**{ui['saw']}.** {ex['saw']}  \n**{ui['why']}.** {ex['why']}  \n**{ui['check']}.** {ex['check']}")
    for n in ex["notes"]:
        st.caption(f"ⓘ {n}")


def show_chart(a):
    """Raw signal around the alert."""
    sig, ch = signal_frame(a, d, T, lang)
    end = min(max(a.t_end, a.t_start + pd.Timedelta(minutes=15)), sig.timestamp.max())
    span = pd.DataFrame({"start": [a.t_start], "end": [end]})
    shade = alt.Chart(span).mark_rect(opacity=0.15, color="#e4572e").encode(x="start:T", x2="end:T")
    lines = alt.Chart(sig.dropna()).mark_line(strokeWidth=1.6).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d/%m %H:%M", labelOverlap=True)),
        y=alt.Y("value:Q", title=T["ui"]["units"][ch][lang], scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")))
    st.markdown(f"**{ui['chart']}**")
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 3], color="#888").encode(y="y:Q")
    st.altair_chart((shade + lines + zero if ch == "balance" else shade + lines).properties(height=210),
                    width="stretch")


def show_evidence(a):
    """Rules fired with measured vs allowed."""
    st.markdown(f"**{ui['evidence']}**")
    ev = pd.DataFrame(evidence(a, d, lang, T)).rename(columns={
        "rule": ui["ev_rule"], "measured": ui["ev_measured"], "allowed": ui["ev_allowed"]})
    st.dataframe(ev, hide_index=True, width="stretch")


def show_shap_and_homes(a):
    if isinstance(a.get("shap"), dict):
        sh = pd.DataFrame({"signal": [T["shap_features"][k][lang] for k in a.shap],
                           "shap": list(a.shap.values())})
        st.caption(ui["shap_title"])
        st.altair_chart(alt.Chart(sh).mark_bar().encode(
            x=alt.X("shap:Q", title=None), y=alt.Y("signal:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.condition("datum.shap > 0", alt.value("#e4572e"), alt.value("#8a9bb0"))
        ).properties(height=160), width="stretch")
    if isinstance(a.get("suspects"), list):
        st.markdown(f"**{ui['homes_title']}**")
        if a.suspects:
            st.markdown("\n".join(
                f"{i + 1}. {s['meter']}" + (f" ({ui['homes_weak']})" if s.get("evidence") == "weak" else "")
                for i, s in enumerate(a.suspects)))
        else:
            st.caption(ui["homes_none"])


# ---- pilot mode ---------------------------------------------------------
CALLS = ["grid_event", "equipment_fault", "data_quality", "nothing"]
ACTIONS = ["approve", "override", "escalate"]


def save_answer(row: dict) -> None:
    """One row per answer, outside the decision log."""
    PILOT_ANSWERS.parent.mkdir(parents=True, exist_ok=True)
    new = not PILOT_ANSWERS.exists()
    pd.DataFrame([row], columns=PILOT_COLUMNS).to_csv(PILOT_ANSWERS, mode="a", header=new, index=False)


def pilot_page():
    st.markdown(f"### {ui['pilot_title']}")
    if not participant.strip():
        st.info(ui["pilot_need_id"])
        st.stop()
    items = pd.read_csv(PILOT_ITEMS)[["item", "set", "alert_id"]].sort_values("item")
    key = f"pilot_{participant}_{group}"
    k = st.session_state.setdefault(key, 0)
    if k >= len(items):
        st.success(ui["pilot_done"])
        st.stop()
    it = items.iloc[k]
    shown = (it.set == "A") == (group == 1)          # group 1 sees A explained, group 2 sees B
    opened = st.session_state.setdefault(f"{key}_open_{k}", time.time())
    a = al.set_index("alert_id").loc[it.alert_id]
    a = pd.concat([pd.Series({"alert_id": it.alert_id}), a])
    st.markdown(f"**{ui['pilot_item'].format(i=int(it['item']), n=len(items))}** · {a.asset} · "
                f"{a.t_start:%Y-%m-%d %H:%M}")
    st.markdown(f"{ui['pilot_system_says']}: **{T['verdicts'][a.verdict][lang]}**, "
                f"{T['bands'][band(a.confidence)][lang]} ({a.confidence * 100:.0f}{' %' if lang == 'fr' else '%'})")
    left, right = st.columns([3, 2], gap="large")
    with left:
        if shown:
            ex = explain(a, d, lang, T, al)
            st.markdown(f"#### {ex['headline']}")
            show_explanation(ex)
        show_chart(a)
        if shown:
            show_evidence(a)
            show_shap_and_homes(a)
    with right:
        with st.form(f"{key}_form_{k}", clear_on_submit=True):
            call = st.radio(ui["pilot_call"], CALLS, index=None, format_func=lambda c: T["pilot_calls"][c][lang])
            action = st.radio(ui["pilot_action"], ACTIONS, index=None, horizontal=True,
                              format_func=lambda c: ui[c])
            clarity = st.radio(ui["pilot_clarity"], [1, 2, 3, 4, 5], index=None, horizontal=True)
            sent = st.form_submit_button(ui["pilot_submit"], type="primary")
        if sent:
            if call is None or action is None or clarity is None:
                st.warning(ui["pilot_answer_all"])
            else:
                save_answer({"participant": participant.strip(), "group": group, "lang": lang,
                             "item": int(it["item"]), "explanation_shown": "Y" if shown else "N",
                             "call": call, "action": action, "seconds": round(time.time() - opened, 1),
                             "clarity": clarity})
                st.session_state[key] = k + 1
                st.rerun()
    st.progress(k / len(items))


if pilot:
    pilot_page()
    st.stop()


# ---- queue (left) -------------------------------------------------------------
st.markdown(f"### {ui['title']}")
left, right = st.columns([5, 6], gap="large")
q = al[al.verdict.isin(show) & al.confidence.map(band).isin(show_b)].sort_values(
    ["detected_at", "alert_id"], ascending=[False, True])


# energy gaps carry their feeder-phase (F2-A) as the asset, never a single home
view = pd.DataFrame({
    ui["col_time"]: q.t_start.dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
    ui["col_asset"]: q.asset.to_numpy(),
    ui["col_verdict"]: q.verdict.map(lambda v: T["verdicts"][v][lang]).to_numpy(),
    ui["col_conf"]: [f"{T['bands'][band(c)][lang].capitalize()} {c * 100:.0f}%" for c in q.confidence],
    ui["col_summary"]: [headline_only(a, lang, T) for _, a in q.iterrows()],
}, index=q.alert_id)
with left:
    st.markdown(f"**{ui['queue']}** · {ui['queue_count'].format(n=len(q))}")
    sel = st.dataframe(view, height=640, on_select="rerun", selection_mode="single-row", key="queue",
                       hide_index=True, column_config={ui["col_summary"]: st.column_config.TextColumn(width="large")})
picked = sel.selection.rows if sel and sel.selection else []
alert_id = q.alert_id.iloc[picked[0]] if picked else qp.get("alert", q.alert_id.iloc[0] if len(q) else None)

# ---- detail (right) -----------------------------------------------------------
with right:
    if alert_id is None or alert_id not in set(al.alert_id):
        st.info(ui["pick"])
        st.stop()
    a = al.set_index("alert_id").loc[alert_id]
    a = pd.concat([pd.Series({"alert_id": alert_id}), a])
    ex = explain(a, d, lang, T, al)
    st.markdown(f"#### {ex['headline']}")
    st.caption(f"{a.alert_id} · {a.t_start:%Y-%m-%d %H:%M} → {a.t_end:%Y-%m-%d %H:%M} · "
               f"{ui['detected']} {a.detected_at:%Y-%m-%d %H:%M}")
    show_explanation(ex)

    show_chart(a)

    # evidence: rules fired with measured vs allowed
    show_evidence(a)
    c1, c2 = st.columns([1, 1])
    with c1:
        show_shap_and_homes(a)

    # escalation tier and the operator's decision
    r = router(versions)
    dec = route_once(r, a, d, explain(a, d, "en", T, al)["saw"])
    with c2:
        st.markdown(f"**{ui['tier']}**")
        st.markdown(f"`{dec['tier']}` {T['tiers'][dec['tier']][lang].split(': ', 1)[1]}  \n"
                    f"<small>{tier_reason(dec['rule'], T, lang)}</small>", unsafe_allow_html=True)
        notified = dec["payload"].get("action") == r.policy.get("auto_notify_action")
        if notified:   # the owner was sent a cleaning tip; the operator can undo it
            st.markdown(f"**{ui['owner_sent']}**")
            st.info(owner_message(a, d, lang, T))
        st.markdown(f"**{ui['decide']}**")
        reason = st.text_input(ui["reason"], key=f"reason_{alert_id}")
        cols = st.columns(4 if notified else 3)
        choice = ("approve" if cols[0].button(ui["approve"], type="primary", width="stretch") else
                  "override" if cols[1].button(ui["override"], width="stretch") else
                  "escalate" if cols[2].button(ui["escalate"], width="stretch") else
                  "undo" if notified and cols[3].button(ui["undo"], width="stretch") else None)
        if choice in ("override", "undo") and not reason.strip():
            st.warning(ui["need_reason"])
        elif choice:
            r.operator_decision(alert_id, operator, choice, reason.strip())
            st.success(ui["logged"].format(choice=T["choices"][choice][lang], op=operator))

    # what the log holds for this alert
    hist = r.history(alert_id)
    with st.expander(f"{ui['history']} ({len(hist)})", expanded=bool(qp.get("log"))):
        st.dataframe(pd.DataFrame([{
            ui["h_time"]: pd.Timestamp(h["ts"], unit="s").strftime("%Y-%m-%d %H:%M:%S"), ui["h_actor"]: h["actor"],
            ui["h_action"]: h["tier"] if h["actor"] == "ai" else T["choices"].get(h["rule"], {}).get(lang, h["rule"]),
            ui["h_reason"]: h["payload"].get("reason", tier_reason(h["rule"], T, lang))} for h in hist]),
            hide_index=True, width="stretch")
        p = dec["payload"]
        w, ref = p["data_window"], p["snapshot_ref"]
        colon = " :" if lang == "fr" else ":"
        st.caption(f"{ui['versions']}{colon} {p['detector_version']} · {p['calibration_version']} · "
                   f"policy {r.policy['version']}  \n{ui['window']}{colon} {len(w['assets'])} {ui['meters_word']}, "
                   f"{w['start']} → {w['end']}  \n{ui['snapshot']}{colon} {Path(ref['path']).name} · "
                   f"SHA-256 {ref['sha256'][:16]}…")
        ok = r.verify_log()
        st.markdown(("✅ " + ui["log_ok"]) if ok else ("❌ " + ui["log_bad"]))
