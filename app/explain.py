"""Explanations for the operator dashboard, in English and French.

Every sentence comes from app/templates.yaml; every number in it comes from the alert row
(results/alerts.csv) or the raw readings in its data window. No Streamlit here, so the
wording can be tested on every alert.
"""
from __future__ import annotations

import ast
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from volttrust.detect.detectors import FAST_RULES, PHASE_RULES, Params

TEMPLATES = Path(__file__).with_name("templates.yaml")
STEP = pd.Timedelta(minutes=15)
P = Params()


def load_templates(path: Path = TEMPLATES) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def band(conf: float) -> str:
    return "high" if conf >= 0.85 else "medium" if conf >= 0.60 else "low"


def _lit(x):
    """alerts.csv keeps lists and dicts as Python literals."""
    if isinstance(x, str) and x[:1] in "[{":
        return ast.literal_eval(x)
    return x


def read_alerts(path: str | Path) -> pd.DataFrame:
    al = pd.read_csv(path)
    for c in ("t_start", "t_end", "detected_at", "pv_resolved_at"):
        if c in al:
            al[c] = pd.to_datetime(al[c])
    for c in ("rules_fired", "suspects", "shap", "shap_top", "linked_rule_alert"):
        if c in al:
            al[c] = al[c].map(_lit)
    return al


def template_key(a: pd.Series) -> str:
    """Which explanation an alert gets."""
    if a.subtype == "energy_gap":
        return "energy_gap_over" if (a.get("gap_kw") or 0) < 0 else "energy_gap"
    if a.subtype == "overload_risk":
        return "overload_risk_alarm" if a.rule == "transformer_overload_alarm" else "overload_risk_notice"
    return a.subtype


@dataclass
class Data:
    """Readings the dashboard has: home meters, substation heads, meter list, alarms, site."""
    readings: pd.DataFrame
    head: pd.DataFrame
    meters: pd.DataFrame
    alarms: pd.DataFrame | None = None
    site: dict | None = None


def neighbours(a: pd.Series, d: Data) -> list[str]:
    """Same feeder and phase (whole feeder for a feeder-level alert), without the asset itself."""
    m = d.meters[d.meters.feeder == a.feeder]
    if isinstance(a.phase, str):
        m = m[m.phase == a.phase]
    return [x for x in m.meter_id if x != a.asset]


def _slice(df, t0, t1, **eq):
    sel = (df.timestamp >= t0) & (df.timestamp <= t1)
    for k, v in eq.items():
        sel &= df[k].isin(v) if isinstance(v, (list, tuple, set)) else df[k] == v
    return df[sel]


def _dur(t0, t1, lang) -> str:
    mins = int(((t1 - t0) + STEP) / pd.Timedelta(minutes=1))
    d, rem = divmod(mins, 24 * 60)
    h, m = divmod(rem, 60)
    parts = ([f"{d} {'d' if lang == 'en' else 'j'}"] if d else []) + ([f"{h} h"] if h else []) \
        + ([f"{m} min"] if m and not d else [])
    return " ".join(parts) or "15 min"


def _num(x, nd=1, lang="en"):
    if x is None or not np.isfinite(x):
        return None
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if lang == "fr" else s


def facts(a: pd.Series, d: Data, lang: str, T: dict, linked: pd.Series | None = None) -> dict:
    """Placeholder values for one alert, as display strings."""
    t0, t1 = a.t_start, a.t_end
    na = T["na"][lang]
    nb = neighbours(a, d)
    is_meter = str(a.asset).startswith("M")
    own = _slice(d.readings, t0, t1, meter_id=a.asset) if is_meter else d.readings.iloc[:0]
    nbr = _slice(d.readings, t0, t1, meter_id=nb)
    nbm = nbr.groupby("timestamp").voltage_v.median()
    dv = None
    if len(own):
        diff = own.set_index("timestamp").voltage_v - nbm
        dv = float(np.nanmean(np.abs(diff))) if diff.notna().any() else None
    if not dv and pd.notna(a.get("max_abs_voltage_resid_v")) and a.max_abs_voltage_resid_v > 0:
        dv = float(a.max_abs_voltage_resid_v)
    feeder_rows = _slice(d.readings, t0, t1, feeder=a.feeder)
    hd = _slice(d.head, t0, t1, feeder=a.feeder)
    hd_ph = hd[hd.phase == a.phase] if isinstance(a.phase, str) else hd
    prev = _slice(d.head, t0 - pd.Timedelta(hours=24), t0 - STEP, feeder=a.feeder)
    step = None
    if len(prev) and len(hd):
        step = (hd.voltage_v[hd.voltage_v > 1].mean() / prev.voltage_v[prev.voltage_v > 1].median() - 1) * 100
    f = None
    if len(hd):
        k = (hd.frequency_hz - 50).abs().idxmax()
        f = hd.frequency_hz[k]
    q_ratio = None
    if len(own):
        night = own[own.timestamp.dt.hour.between(1, 4)]
        ok = night.active_power_kw.abs() > 1e-3
        if ok.any():
            q_ratio = float((night.reactive_power_kvar[ok] / night.active_power_kw[ok]).median())
    value = na
    if len(own):
        r0 = own.iloc[0]
        value = (f"{_num(r0.active_power_kw, 2, lang)} kW, {_num(r0.voltage_v, 1, lang)} V, "
                 f"{_num(r0.current_a, 2, lang)} A")
    id_err = a.get("max_abs_identity_resid")
    pct = lambda x: _num(100 * x, 0, lang) if x is not None and pd.notna(x) else None  # noqa: E731

    out = {
        "asset": a.asset, "feeder": a.feeder, "phase": a.phase if isinstance(a.phase, str) else na,
        "subtype": str(a.subtype).replace("_", " ").capitalize(),
        "conf": _num(100 * a.confidence, 0, lang), "band": T["bands"][band(a.confidence)][lang],
        "t_start": t0.strftime("%Y-%m-%d %H:%M"), "dur": _dur(t0, t1, lang),
        "share": pct(a.get("neighbors_affected")),
        "dv": _num(dv, 1, lang),
        "slope": _num(abs(a.voltage_slope_v_per_day), 2, lang) if pd.notna(a.get("voltage_slope_v_per_day")) else None,
        "id_err": pct(id_err),
        "gap_kw": _num(abs(a.gap_kw), 2, lang) if pd.notna(a.get("gap_kw")) else None,
        "ratio": pct(max(a.pv_ratio_to_peers, 0.0)) if pd.notna(a.get("pv_ratio_to_peers")) else None,
        "v_min": _num(feeder_rows.voltage_min_v[feeder_rows.voltage_min_v > 1].min(), 0, lang)
        if len(feeder_rows) else None,
        "step": _num(abs(step), 1, lang) if step is not None else None,
        "f": _num(f, 2, lang) if f is not None else None,
        "v": _num(hd_ph.voltage_v.min(), 0, lang) if len(hd_ph) else None,
        "q_ratio": _num(q_ratio, 1, lang),
        "value": value,
        "oil": _num(a.get("oil_resid_c"), 1, lang) if pd.notna(a.get("oil_resid_c", np.nan)) else None,
        "batch_time": a.detected_at.strftime("%Y-%m-%d %H:%M") if pd.notna(a.get("detected_at")) else na,
        "rules": ", ".join(a.rules_fired) if isinstance(a.rules_fired, list) else str(a.rule),
    }
    out = {k: (na if v is None else v) for k, v in out.items()}
    # homes to check (energy gaps): ranked, never a single culprit
    sus = a.get("suspects")
    if isinstance(sus, list) and sus:
        weak = T["ui"]["homes_weak"][lang]
        out["homes"] = ", ".join(f"{i + 1}. {s['meter']}" + (f" ({weak})" if s.get("evidence") == "weak" else "")
                                 for i, s in enumerate(sus))
    else:
        out["homes"] = T["ui"]["homes_none"][lang].rstrip(".")
    # main SHAP signal for forest alerts
    top = a.get("shap_top")
    if isinstance(top, dict):
        # the SHAP record keeps the signal's value at the forest's peak row
        peak = dict(out)
        if top["feature"] == "r_v":
            peak["dv"] = _num(abs(top["value"]), 1, lang)
        elif top["feature"] == "r_id":
            peak["id_err"] = _num(100 * abs(top["value"]), 0, lang)
        out["top_signal_text"] = _fill(T["top_signal_text"][top["feature"]][lang], peak)
    else:
        out["top_signal_text"] = na
    # tamper reason
    if a.rule in T["tamper_reason"]:
        alarm_time, kind = t0.strftime("%Y-%m-%d %H:%M"), na
        if d.alarms is not None and a.rule == "tamper_alarm":
            al = d.alarms[(d.alarms.meter_id == a.asset) & d.alarms.alarm.str.startswith("tamper")
                          & (d.alarms.timestamp >= t0 - 4 * STEP) & (d.alarms.timestamp <= t1)]
            if len(al):
                alarm_time = al.timestamp.min().strftime("%Y-%m-%d %H:%M")
                kind = T["alarm_kinds"].get(al.alarm.iloc[0], {}).get(lang, al.alarm.iloc[0])
        out["tamper_reason"] = _fill(T["tamper_reason"][a.rule][lang],
                                     {**out, "alarm_time": alarm_time, "alarm_kind": kind})
    else:
        out["tamper_reason"] = na
    if linked is not None:
        out["linked"] = _fill(T["explain"].get(template_key(linked), T["explain"]["_default"])["saw"][lang],
                              facts(linked, d, lang, T)).rstrip(".")
        if len(out["linked"]) > 1 and out["linked"][1].islower():
            out["linked"] = out["linked"][0].lower() + out["linked"][1:]
        out["linked_rule"] = T["rules"].get(linked.rule, {}).get(lang, linked.rule)
    return out


def _fill(tpl: str, vals: dict) -> str:
    keys = [k for _, k, _, _ in string.Formatter().parse(tpl) if k]
    return tpl.format(**{k: vals.get(k, "n/a") for k in keys})


def linked_rule_alert(a: pd.Series, alerts: pd.DataFrame) -> pd.Series | None:
    """The nearby voltage-rule alert a forest alert leads with."""
    lk = a.get("linked_rule_alert")
    if a.rule != "iforest" or not isinstance(lk, dict):
        return None
    o = alerts[alerts.alert_id == lk["alert_id"]]
    return o.iloc[0] if len(o) else None


def explain(a: pd.Series, d: Data, lang: str, T: dict, alerts: pd.DataFrame | None = None) -> dict:
    """Four lines (headline, saw, why, check) plus any data-limit notes."""
    link = linked_rule_alert(a, alerts) if alerts is not None else None
    v = facts(a, d, lang, T, link)
    tpl = T["explain"].get(template_key(a), T["explain"]["_default"])
    out = {k: _fill(tpl[k][lang], v) for k in ("headline", "saw", "why", "check")}
    if link is not None:
        out["saw"] = _fill(T["linked"][lang], v)
    notes = []
    site = d.site or {}
    if a.rule not in FAST_RULES and site.get("batch_hours"):
        notes.append(_fill(T["notes"]["batch"][lang], v))
    if str(a.rule).startswith("pv_"):
        src = d.meters.set_index("meter_id").get("pv_source")
        if src is not None and src.get(a.asset) == "estimated":
            notes.append(T["notes"]["estimated_solar"][lang])
    if site.get("head_meter") == "none" and a.rule in PHASE_RULES:
        notes.append(T["notes"]["no_head"][lang])
    if site.get("phase_source") == "inferred" and str(a.asset).startswith("M"):
        notes.append(T["notes"]["phase_unknown"][lang])
    out["notes"] = notes
    return out


def evidence(a: pd.Series, d: Data, lang: str, T: dict) -> list[dict]:
    """One line per rule that fired: what was measured and what is allowed."""
    v = facts(a, d, lang, T)
    na = T["na"][lang]
    n = lambda x, nd=2: _num(x, nd, lang) if x is not None and pd.notna(x) else na  # noqa: E731
    pv = a.get("pv_ratio_to_peers")
    pc = " %" if lang == "fr" else "%"
    measured_allowed = {
        "head_outage": (f"{v['v']} V", "> 1 V"),
        "head_phase_loss": (f"{v['v']} V", "> 1 V"),
        "head_sag": (f"{v['v_min']} V", f"≥ 90{pc} × V"),
        "head_mv_step": (f"{v['step']}{pc}", f"< {n(100 * P.head_step, 1)}{pc}"),
        "head_frequency": (f"{v['f']} Hz", f"50 ± {n(P.freq_tol, 1)} Hz"),
        "group_voltage_drop": (f"{v['share']}{pc}", f"< {P.group_min_drop} {T['ui']['meters_word'][lang]}"),
        "group_voltage_rise": (f"{v['share']}{pc}", f"< {P.group_min_rise} {T['ui']['meters_word'][lang]}"),
        "night_leakage": (v["q_ratio"], f"≤ {n(P.leak_qp, 1)}"),
        "dropout": (v["dur"], "0"),
        "stuck": (v["dur"], f"< {P.stuck_run} × 15 min"),
        "tamper_alarm": ("1", "0"),
        "power_identity_persist": (f"{v['id_err']}{pc}", f"≤ {n(100 * P.id_tol, 0)}{pc}"),
        "power_identity": (f"{v['id_err']}{pc}", f"≤ {n(100 * P.id_tol, 0)}{pc}"),
        "voltage_spike": (f"{v['dv']} V", f"≤ {n(P.v_spike, 0)} V"),
        "register_jump": (f"{n(_register_step(a, d))} kWh", f"≤ {n(P.reg_tol, 2)} kWh"),
        "pv_reading_stuck": (v["dur"], f"< {P.pv_stuck_run} × 15 min"),
        "pv_inverter_trip": (f"{v['ratio']}{pc}", f"≥ {n(100 * P.pv_trip, 0)}{pc}"),
        "pv_over_report": (f"{v['ratio']}{pc}", f"≤ {n(100 * P.pv_high, 0)}{pc}"),
        "pv_underperform": (f"{v['ratio']}{pc}", f"≥ {n(100 * P.pv_low, 0)}{pc}"),
        "balance_gap_phase": (f"{v['gap_kw']} kW", f"≤ {n(P.bal_min_kw)} kW + {n(100 * P.bal_rel, 0)}{pc}"),
        "balance_gap_phase_slow": (f"{v['gap_kw']} kW", f"≤ {n(P.bal_slow_kw)} kW"),
        "voltage_offset": (f"{v['dv']} V", f"≤ {n(P.v_offset_min, 1)} V"),
        "iforest": (n(a.get("iforest_score"), 3), T["ui"]["if_allowed"][lang]),
    }
    if pd.isna(pv):
        for k in ("pv_inverter_trip", "pv_over_report", "pv_underperform"):
            measured_allowed[k] = (na, measured_allowed[k][1])
    rows = []
    unit = lambda x: na if x.startswith(na) else x  # noqa: E731  (no unit after "n/a")
    for r in (a.rules_fired if isinstance(a.rules_fired, list) else [a.rule]):
        m, al = measured_allowed.get(r, (na, na))
        rows.append({"rule": T["rules"].get(r, {}).get(lang, r), "measured": unit(m), "allowed": al})
    return rows


def _register_step(a: pd.Series, d: Data) -> float | None:
    """Largest gap between the import register's step and power × 15 min, in the alert."""
    own = _slice(d.readings, a.t_start - STEP, a.t_end, meter_id=a.asset).sort_values("timestamp")
    if len(own) < 2:
        return None
    gap = (own.energy_import_kwh.diff() - own.active_power_kw.clip(lower=0) * 0.25).abs()
    return float(gap.max()) if gap.notna().any() else None


def headline_only(a: pd.Series, lang: str, T: dict) -> str:
    """The queue's one-liner: headline only, from the alert row (no raw readings needed)."""
    tpl = T["explain"].get(template_key(a), T["explain"]["_default"])["headline"][lang]
    vals = {"asset": a.asset, "feeder": a.feeder, "phase": a.phase if isinstance(a.phase, str) else T["na"][lang],
            "subtype": str(a.subtype).replace("_", " ").capitalize(),
            "conf": _num(100 * a.confidence, 0, lang), "band": T["bands"][band(a.confidence)][lang]}
    return _fill(tpl, vals)


# ---- raw signal around the alert --------------------------------------------
VOLTAGE, POWER, SOLAR, FREQ, BALANCE = "voltage_v", "active_power_kw", "solar_kw", "frequency_hz", "balance"
POWER_RULES = ("power_identity", "power_identity_persist", "stuck", "register_jump", "night_leakage",
               "tamper_alarm")


def channel(a: pd.Series) -> str:
    if a.rule in PHASE_RULES:
        return BALANCE
    if str(a.rule).startswith("pv_"):
        return SOLAR
    if a.rule == "head_frequency":
        return FREQ
    if a.rule in POWER_RULES:
        return POWER
    return VOLTAGE


def signal_frame(a: pd.Series, d: Data, T: dict, lang: str) -> tuple[pd.DataFrame, str]:
    """Long table (timestamp, series, value) around the alert and the channel shown:
    the asset's channel and its neighbours' median, or substation vs homes for energy gaps.
    Long alerts are shown from 12 h before to 7 days in, hourly."""
    ui = T["ui"]
    t0 = a.t_start - pd.Timedelta(hours=12)
    t1 = min(a.t_end + pd.Timedelta(hours=6), a.t_start + pd.Timedelta(days=7))
    ch = channel(a)
    nb = neighbours(a, d)
    out = []
    if ch == BALANCE:
        hd = _slice(d.head, t0, t1, feeder=a.feeder)
        if isinstance(a.phase, str):
            hd = hd[hd.phase == a.phase]
        homes = _slice(d.readings, t0, t1, meter_id=nb).groupby("timestamp").active_power_kw.sum(min_count=1)
        # the gap itself: what leaves the substation minus what the homes record
        out.append((hd.groupby("timestamp").active_power_kw.sum() - homes).rename(ui["chart_gap"][lang]))
    elif ch == FREQ or not str(a.asset).startswith("M"):
        col = FREQ if ch == FREQ else VOLTAGE
        hd = _slice(d.head, t0, t1, feeder=a.feeder)
        own = hd[hd.phase == a.phase] if isinstance(a.phase, str) else hd
        out.append(own.groupby("timestamp")[col].median().rename(ui["chart_head"][lang]))
        if col == VOLTAGE:
            out.append(_slice(d.readings, t0, t1, meter_id=nb).groupby("timestamp")[col].median()
                       .rename(ui["chart_neigh"][lang]))
    else:
        rows = _slice(d.readings, t0, t1, meter_id=[a.asset] + nb)
        if ch == SOLAR:
            pv = d.meters.set_index("meter_id").has_pv.astype(bool)
            rows = rows[rows.meter_id.map(pv).fillna(False)]
            rows = rows.assign(value=rows.pv_power_kw.where(rows.pv_power_kw.notna(), -rows.active_power_kw))
        else:
            rows = rows.assign(value=rows[ch])
        out.append(rows[rows.meter_id == a.asset].set_index("timestamp").value.rename(f"{a.asset} ({ui['chart_this'][lang]})"))
        out.append(rows[rows.meter_id != a.asset].groupby("timestamp").value.median().rename(ui["chart_neigh"][lang]))
    df = pd.concat(out, axis=1)
    if t1 - t0 > pd.Timedelta(days=3):
        df = df.resample("1h").mean()
    if ch == BALANCE:   # the gap is small next to the swings; its daily mean shows it
        df[ui["chart_gap_day"][lang]] = df.iloc[:, 0].rolling("24h", min_periods=4).mean()
    long = df.reset_index().melt("timestamp", var_name="series", value_name="value")
    return long, ch


def owner_message(a: pd.Series, d: Data, lang: str, T: dict) -> str:
    """The cleaning tip a dust alert sends the panel owner."""
    v = facts(a, d, lang, T)
    r = a.get("pv_ratio_to_peers")
    v["loss"] = _num(100 * (1 - r), 0, lang) if pd.notna(r) else T["na"][lang]
    return _fill(T["owner_message"]["dust"][lang], v)
