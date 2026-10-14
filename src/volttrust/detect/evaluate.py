"""Score alerts against the injected ground truth (spec III.e: precision/recall, false positives,
verdict accuracy, time-to-detect, latency)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TOL = 4  # intervals of tolerance (1 h) around an event window


def _event_assets(e) -> set:
    a = set(filter(None, str(e.meters).split(";"))) if pd.notna(e.meters) else set()
    a |= set(filter(None, str(e.heads).split(";"))) if pd.notna(e.heads) else set()
    if bool(e.transformer):
        a.add("TX")
    return a


PHASE_RULES = ("balance_gap_phase", "balance_gap_phase_slow")


def _opens_energy_gap(e) -> bool:
    """Only a reporting or sensing error on a home's power or energy reading can open a gap between
    the feeder head and the sum of the meters; physical events change both sides alike."""
    ch = str(e.channels)
    return e.layer == "measurement" and ("active_power_kw" in ch or "energy_import_kwh" in ch)


def match(alerts: pd.DataFrame, events: pd.DataFrame, idx: pd.DatetimeIndex,
          meter_phase: dict | None = None, energy_events_only: bool = True,
          collected: np.ndarray | None = None, meters: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """meter_phase maps meter id -> feeder-phase (e.g. "F2-A"). A feeder-phase energy-gap alert
    matches an event when one of the event's meters is on that feeder-phase
    and, with energy_events_only, the event is one that can open an energy gap. For the verdict vote,
    a phase alert competes with the alerts raised on the event's own meter."""
    meter_phase = meter_phase or {}
    ev = events.copy()
    ev["s"] = idx.searchsorted(pd.to_datetime(ev.start))
    ev["e"] = ev.s + ev.duration_intervals
    al = alerts.copy()
    al["matched_event"] = None
    is_phase = al.rule.isin(PHASE_RULES) if "rule" in al else pd.Series(False, index=al.index)
    by_asset = {a: g for a, g in al[~is_phase].groupby("asset")}
    by_phase = {a: g for a, g in al[is_phase].groupby("asset")}
    rows = []
    for e in ev.itertuples():
        hits = []
        assets = _event_assets(e)
        ev_meters = {a for a in assets if a in meter_phase}
        # an energy-gap alert sits on a feeder-phase ("F2-A"), or on the whole feeder ("F2") while
        # some home's phase is still unknown
        phases = ({meter_phase[m] for m in ev_meters} | {meter_phase[m].split("-")[0] for m in ev_meters}
                  if (_opens_energy_gap(e) or not energy_events_only) else set())
        groups = [by_asset.get(a) for a in assets] + [by_phase.get(ph) for ph in phases]
        for g in groups:
            if g is None:
                continue
            # alerts that start during the event (or just around it): an alert already running
            # before the event began was not caused by it
            ov = g[(g.start_idx < e.e + TOL) & (g.start_idx >= e.s - TOL)]
            hits.append(ov)
        hits = pd.concat(hits) if hits else al.iloc[:0]
        if len(hits):
            al.loc[hits.index, "matched_event"] = e.event_id
            # each asset votes with the alert covering most of the event window; majority wins
            # a phase alert competes on the event's own meter, like any alert raised there
            voters = hits.copy()
            on_phase = voters.rule.isin(PHASE_RULES)
            if on_phase.any() and ev_meters:
                voters.loc[on_phase, "asset"] = sorted(ev_meters)[0]
            ov = (np.minimum(voters.end_idx, e.e + TOL) - np.maximum(voters.start_idx, e.s)).clip(lower=1)
            per_asset = voters.assign(ov=ov).sort_values("ov").groupby("asset").tail(1)
            votes = per_asset.verdict.value_counts()
            best = per_asset.sort_values("ov").iloc[-1]
            first = hits.start_idx.min()
            # the real delay: until the first alert reaches a person, batch wait included
            reach = hits.detected_idx.min() if "detected_idx" in hits else first
            # is one of the event's meters among the homes to check of a matched phase alert?
            ph = hits[hits.rule.isin(PHASE_RULES)]
            hit = (any(s["meter"] in ev_meters for lst in ph.suspects for s in lst)
                   if len(ph) and ev_meters else None)
            rows.append({"event_id": e.event_id, "detected": True, "pred_verdict": votes.idxmax(),
                         "pred_subtype": best.subtype,
                         "time_to_detect_h": max(0, first - e.s) / 4,
                         "time_to_operator_h": max(0, reach - e.s) / 4, "suspect_hit": hit})
        else:
            rows.append({"event_id": e.event_id, "detected": False, "pred_verdict": None,
                         "pred_subtype": None, "time_to_detect_h": np.nan, "time_to_operator_h": np.nan,
                         "suspect_hit": None})
    res = ev.merge(pd.DataFrame(rows), on="event_id")
    # an event whose readings were all lost in collection never reached the detector
    res["hidden_by_collection_gap"] = False
    if collected is not None and meters is not None:
        col = {m: j for j, m in enumerate(meters)}
        for i, e in res.iterrows():
            ms = [col[m] for m in str(e.meters).split(";") if m in col] if pd.notna(e.meters) else []
            if ms and not pd.notna(e.heads) and not bool(e.transformer):
                res.at[i, "hidden_by_collection_gap"] = not collected[e.s:max(e.e, e.s + 1), ms].any()
    res["verdict_correct"] = res.detected & (res.pred_verdict == res.verdict)
    return res, al


def summarize(res: pd.DataFrame, al: pd.DataFrame, idx: pd.DatetimeIndex, split_of_t: np.ndarray,
              point: dict | None = None) -> dict:
    out = {}
    al = al.assign(split=split_of_t[al.start_idx.values])
    for sp in ("val", "test"):
        r = res[res.split == sp]
        a = al[al.split == sp]
        days = (split_of_t == sp).sum() / 96
        meas = r[r.layer == "measurement"]
        out[sp] = {
            "events": int(len(r)),
            "event_recall": round(float(r.detected.mean()), 3),
            "event_recall_physical": round(float(r[r.layer == "physical"].detected.mean()), 3),
            "event_recall_data_problems": round(float(meas.detected.mean()), 3),
            "verdict_accuracy_on_detected": round(float(r[r.detected].verdict_correct.mean()), 3),
            "events_hidden_by_collection_gap": int(r.hidden_by_collection_gap.sum()),
            "event_recall_observable": round(float(r[~r.hidden_by_collection_gap].detected.mean()), 3),
            "alerts": int(len(a)),
            "alert_precision": round(float(a.matched_event.notna().mean()), 3) if len(a) else None,
            "false_alerts_per_day": round(float(a.matched_event.isna().sum() / days), 2),
            "false_alerts_per_feeder_per_day": round(float(a.matched_event.isna().sum() / days / 2), 2),
            "median_time_to_detect_h": round(float(r.time_to_detect_h.median()), 2),
            "median_time_to_operator_h": round(float(r.time_to_operator_h.median()), 2),
            "median_time_to_operator_h_physical": round(float(r[r.layer == "physical"].time_to_operator_h.median()), 2),
        }
        sh = r.suspect_hit.dropna().astype(bool) if "suspect_hit" in r else pd.Series(dtype=bool)
        out[sp]["energy_gap_alerts"] = int(a.rule.isin(PHASE_RULES).sum()) if "rule" in a else 0
        out[sp]["events_with_energy_gap_alert"] = int(len(sh))
        out[sp]["suspect_top3_hit_rate"] = round(float(sh.mean()), 3) if len(sh) else None
        if point:
            out[sp].update(point.get(sp, {}))
    return out


def per_type(res: pd.DataFrame) -> pd.DataFrame:
    g = res[res.split != "train"].groupby(["family", "anomaly_type"])
    t = g.agg(events=("event_id", "size"), recall=("detected", "mean"),
              verdict_acc=("verdict_correct", "sum"), held_out=("held_out", "first"),
              median_ttd_h=("time_to_detect_h", "median"),
              median_to_operator_h=("time_to_operator_h", "median"))
    t["verdict_acc"] = t.verdict_acc / g.detected.sum().replace(0, np.nan)
    if "suspect_hit" in res:
        # share of events caught by a phase energy-gap alert whose top-3 homes to check include the culprit
        t["suspect_top3_hit_rate"] = g.suspect_hit.agg(lambda x: x.dropna().astype(bool).mean() if x.notna().any() else np.nan)
    return t.round(3).reset_index()


def phase_alias(D, true_phase: np.ndarray) -> dict:
    """Without STEG's map and without a substation reading, the detector's phase groups are arbitrary
    labels. For scoring only, each group is named after the true phase of most of its homes."""
    out = {}
    for f in np.unique(D.feeder):
        for g in range(3):
            sel = (D.feeder == f) & (D.phase == g) & D.phase_known
            if sel.any():
                tp = np.bincount(true_phase[sel], minlength=3).argmax()
                out[f"F{f + 1}-{'ABC'[g]}"] = f"F{f + 1}-{'ABC'[tp]}"
    return out


def by_pv_source(res: pd.DataFrame, al: pd.DataFrame, D) -> pd.DataFrame:
    """Panel results for inverter (opt-in) and estimated solar homes, reported separately."""
    src = dict(zip(D.meters, D.pv_source))
    r = res[(res.split != "train") & res.anomaly_type.str.startswith("pv_")].copy()
    r["pv_source"] = r.meters.astype(str).str.split(";").str[0].map(src)
    a = al[al.rule.isin(["pv_inverter_trip", "pv_underperform", "pv_over_report", "pv_reading_stuck"])].copy()
    a = a[D.split[a.start_idx.values] != "train"]
    a["pv_source"] = a.asset.map(src)
    days = (D.split != "train").sum() / 96
    rows = []
    for s in ("inverter_optin", "estimated"):
        rr, aa = r[r.pv_source == s], a[a.pv_source == s]
        n_homes = int((D.pv_source == s).sum())
        rows.append({"pv_source": s, "homes": n_homes, "events": len(rr),
                     "recall": round(float(rr.detected.mean()), 3) if len(rr) else None,
                     # reading faults (frozen or inflated inverter reading) need the inverter reading
                     "recall_panel_faults": round(float(rr[rr.layer == "physical"].detected.mean()), 3)
                     if (rr.layer == "physical").any() else None,
                     "verdict_acc": round(float(rr[rr.detected].verdict_correct.mean()), 3) if rr.detected.any() else None,
                     "panel_alerts": len(aa), "false_panel_alerts": int(aa.matched_event.isna().sum()),
                     "false_per_home_per_year": round(float(aa.matched_event.isna().sum()) / max(n_homes, 1) / days * 365, 2)})
    return pd.DataFrame(rows)


def confusion(res: pd.DataFrame) -> pd.DataFrame:
    r = res[(res.split != "train")].copy()
    r["pred_verdict"] = r.pred_verdict.fillna("missed")
    return pd.crosstab(r.verdict, r.pred_verdict)


def point_metrics(meter_rule: np.ndarray, labels_path: Path, meters: list, idx, split: np.ndarray) -> dict:
    lab = pd.read_parquet(labels_path, columns=["timestamp", "meter_id", "is_anomaly"])
    y = lab.pivot(index="timestamp", columns="meter_id", values="is_anomaly")[meters].to_numpy() > 0
    p = meter_rule >= 0
    out = {}
    for sp in ("val", "test"):
        rows = split == sp
        yt, pt = y[rows].ravel(), p[rows].ravel()
        tp, fp = (yt & pt).sum(), (~yt & pt).sum()
        fn, tn = (yt & ~pt).sum(), (~yt & ~pt).sum()
        out[sp] = {"row_precision": round(tp / max(tp + fp, 1), 3), "row_recall": round(tp / max(tp + fn, 1), 3),
                   "row_false_positive_rate": round(fp / max(fp + tn, 1), 5)}
    return out
