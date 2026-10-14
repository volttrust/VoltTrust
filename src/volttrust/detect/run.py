"""Run the Phase 1 baseline detector on a generated dataset and write alerts + metrics.

    python -m volttrust.detect.run --data data --out results
    python -m volttrust.detect.run --head-meter none --phase-source inferred --pv-source estimated

The site options say what the neighborhood's equipment provides.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibrate, detectors, evaluate, features
from .site import HEAD_METERS, Site


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-iforest", action="store_true")
    ap.add_argument("--head-meter", default="concentrator", choices=HEAD_METERS)
    ap.add_argument("--phase-source", default="steg_map", choices=("steg_map", "inferred"))
    ap.add_argument("--transformer-monitor", action="store_true")
    ap.add_argument("--pv-source", default="as_data", choices=("as_data", "estimated"))
    ap.add_argument("--no-reference-panel", action="store_true")
    ap.add_argument("--batch-hours", type=int, default=12)
    ap.add_argument("--confidence-table", default=None,
                    help="fitted_models.json whose confidence_by_rule to use (default: fit on this data's May to Jul)")
    ap.add_argument("--params", default="{}", help='detector parameter overrides as JSON, e.g. {"pv_est_trip_run": 2}')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    site = Site(head_meter=a.head_meter, phase_source=a.phase_source, transformer_monitor=a.transformer_monitor,
                pv_source=a.pv_source, use_reference_panel=not a.no_reference_panel, batch_hours=a.batch_hours)

    t0 = time.perf_counter()
    D = features.load(a.data, site)
    t1 = time.perf_counter()
    R = features.residuals(D)
    P = detectors.Params(**json.loads(a.params))
    det = detectors.detect(D, R, P, use_iforest=not a.no_iforest)
    t2 = time.perf_counter()

    events = pd.read_csv(Path(a.data) / "events.csv")
    topo = pd.DataFrame(json.loads((Path(a.data) / "topology.json").read_text())["meters"]).set_index("meter_id")
    true_phase = topo.loc[D.meters, "phase"].map({"A": 0, "B": 1, "C": 2}).to_numpy()
    meter_phase = {m: f"F{f + 1}-{'ABC'[p]}" for m, f, p in zip(D.meters, D.feeder, true_phase)}
    alerts = det.alerts
    phase_report = {"source": site.phase_source}
    if site.phase_source == "inferred":
        ok = D.phase_known
        phase_report.update({
            "homes": int(D.H), "unclear_technician_checks": int((~ok).sum()),
            "margin_threshold": site.phase_unclear_margin, "min_margin": round(float(D.phase_margin.min()), 3)})
        if site.has_head:
            phase_report["correct_all"] = int((D.phase == true_phase).sum())
            phase_report["wrong_among_known"] = int(((D.phase != true_phase) & ok).sum())
        else:
            # group labels are arbitrary: name each group after its majority true phase, for scoring only
            alias = evaluate.phase_alias(D, true_phase)
            alerts = alerts.assign(asset=alerts.asset.replace(alias))
            mapped = np.array([alias.get(f"F{f + 1}-{'ABC'[p]}", "?")[-1:] for f, p in zip(D.feeder, D.phase)])
            phase_report["correct_all"] = int((mapped == np.array(list("ABC"))[true_phase]).sum())
            phase_report["wrong_among_known"] = int(((mapped != np.array(list("ABC"))[true_phase]) & ok).sum())
        pd.DataFrame({"meter_id": np.array(D.meters)[~ok], "feeder": [f"F{f + 1}" for f in D.feeder[~ok]],
                      "margin": D.phase_margin[~ok].round(3),
                      "task": "phase unclear: technician check at the pole"}).to_csv(out / "maintenance_tasks.csv", index=False)
    res, al = evaluate.match(alerts, events, D.idx, meter_phase, collected=D.collected, meters=D.meters)
    point = evaluate.point_metrics(det.meter_rule, Path(a.data) / "meters.parquet", D.meters, D.idx, D.split)
    summary = evaluate.summarize(res, al, D.idx, D.split, point)
    # measured confidence per rule: fitted on May to Jul, or taken from another run
    if a.confidence_table:
        cal = json.loads(Path(a.confidence_table).read_text())["confidence_by_rule"]
    else:
        cal = calibrate.fit(al, res, D.split, "val", D.idx, source=str(a.data))
    al = calibrate.apply(al, cal)
    det.alerts["confidence"] = al.set_index("alert_id").confidence.reindex(det.alerts.alert_id).to_numpy()
    conf_rep = {"fitted_on": cal["data"], "window": cal["window"], "fit_date": cal["fit_date"]}
    per_rule = []
    for name, splits in (("test_months", ("test",)), ("val_and_test_months", ("val", "test"))):
        conf_rep[name], per = calibrate.report(al, res, D.split, splits)
        per_rule.append(per.assign(months=name))
    pd.concat(per_rule).to_csv(out / "confidence_by_rule.csv", index=False)
    summary["confidence"] = conf_rep
    summary["site"] = site.to_dict()
    summary["param_overrides"] = json.loads(a.params)
    summary["phase_map"] = phase_report
    pv = evaluate.by_pv_source(res, al, D)
    pv.to_csv(out / "panel_by_source.csv", index=False)
    summary["panels_by_source"] = pv.to_dict("records")
    summary["panel_reference"] = detectors.reference_pv(D)[1]
    low = al[al.rule == "pv_underperform"]
    if "pv_status" in low:
        # per real dust event: confirmed by a sudden recovery, and how long after the dust began
        ev_start = dict(zip(res.event_id, pd.to_datetime(res.start)))
        ev_type = dict(zip(res.event_id, res.anomaly_type))
        low = low.assign(true_type=low.matched_event.map(ev_type))
        dust = low[low.true_type == "pv_soiling"]
        conf = dust[dust.pv_status == "dust_confirmed_by_recovery"]
        days = (pd.to_datetime(conf.pv_resolved_at) - conf.matched_event.map(ev_start)).dt.total_seconds() / 86400
        false_low = low[low.matched_event.isna()]
        summary["panel_dust_vs_aging"] = {
            "dust_events_flagged": int(dust.matched_event.nunique()),
            "dust_events_confirmed_by_recovery": int(conf.matched_event.nunique()),
            "median_days_from_dust_start_to_confirmation": round(float(days.groupby(conf.matched_event).min().median()), 1)
            if len(conf) else None,
            "degradation_events_labelled_degradation": int(low[(low.true_type == "pv_degradation")
                                                               & (low.subtype == "pv_degradation")].matched_event.nunique()),
            "false_slow_slides": int(len(false_low)),
            "false_slides_confirmed_as_dust": int((false_low.pv_status == "dust_confirmed_by_recovery").sum()),
            "aging_inspect_alerts": int((low.pv_status == "no_recovery_in_60_days").sum())}
    summary["transformer"] = {
        "monitor": site.transformer_monitor,
        "peak_load_pu": round(float(np.nanmax(det.evidence["transformer_load_pu"])), 3),
        "peak_top_oil_estimate_c": round(float(np.nanmax(det.evidence["top_oil_estimate_c"])), 1),
        "alerts": {k: int((al.rule == k).sum()) for k in detectors.TX_RULES}}
    summary["latency"] = {
        # detection alone; the SHAP explanations of forest alerts are timed separately below
        "batch_seconds_for_full_year": round(t2 - t1 - (det.timing or {}).get("shap_seconds_total", 0), 2),
        "ms_per_15min_snapshot_all_assets": round((t2 - t1 - (det.timing or {}).get("shap_seconds_total", 0)) / D.T * 1000, 3),
        "load_seconds": round(t1 - t0, 2),
        **({"shap_" + k.removeprefix("shap_"): v for k, v in (det.timing or {}).items()}),
    }
    # which signal SHAP names for each forest alert, by the true anomaly type it matched
    fa = al[al.rule == "iforest"]
    if len(fa) and "shap_top" in fa:
        ev_type = dict(zip(res.event_id, res.anomaly_type))
        tops = pd.DataFrame({"true_type": fa.matched_event.map(ev_type).fillna("no event (false alert)"),
                             "shap_top": fa.shap_top.map(lambda x: x["feature"] if isinstance(x, dict) else None)})
        pd.crosstab(tops.true_type, tops.shap_top).to_csv(out / "shap_top_by_type.csv")

    al.drop(columns=["start_idx", "end_idx"]).to_csv(out / "alerts.csv", index=False)
    res.drop(columns=["s", "e"]).to_csv(out / "event_results.csv", index=False)
    evaluate.per_type(res).to_csv(out / "recall_by_type.csv", index=False)
    evaluate.confusion(res).to_csv(out / "verdict_confusion.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    recs = detectors.to_verdict_records(det.alerts[det.alerts.t_start >= "2025-08-01"].head(200))
    (out / "verdicts_sample.json").write_text(json.dumps(recs, indent=2, default=float))
    (out / "fitted_models.json").write_text(json.dumps({**R.models, "confidence_by_rule": cal}, indent=1, default=float))

    print(json.dumps(summary, indent=2, default=float))
    print(evaluate.per_type(res).to_string(index=False))
    print(evaluate.confusion(res))


if __name__ == "__main__":
    main()
