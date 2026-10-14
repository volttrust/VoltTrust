"""Explanation quality: the four measures and targets.

All four are scored against the injected ground truth, per seed and per window (val, test):
1. verdict accuracy: detected events whose verdict (grid event / equipment fault / data
   problem) is right;
2. right home in the top 3: energy-gap events whose true meter is among the homes to check;
3. calibration error (ECE, 10 bins) of the alert confidence against "the verdict was right";
4. SHAP match: forest alerts whose SHAP main signal is the signal the injected fault corrupts.
   Fault types with no expected signal are reported per type but left out of the rate. Reported raw
   (SHAP only) and as shown to the operator: a forest alert linked to a nearby voltage-rule alert
   leads with that rule's reason, so its shown main signal is the rule's.

    PYTHONPATH=src python -m volttrust.detect.quality --run 42=results --run 2026=results/fresh_seed_2026 \\
        --out results/explanation_quality.csv
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from volttrust.detect.calibrate import ece
from volttrust.detect.detectors import LINK_RULE_SIGNAL

TARGETS = {"verdict_accuracy": (">=", 0.85), "right_home_in_top3": (">=", 0.70),
           "calibration_error_ece": ("<=", 0.08), "shap_main_signal_match": (">=", 0.80),
           "shap_main_signal_match_as_shown": (">=", 0.80)}
# expected SHAP main signal per injected type
SHAP_EXPECTED = {
    "noise_growth": "rv_sd", "meter_end_of_life": "rv_sd",
    "high_resistance_joint": "r_v", "bias_voltage": "r_v", "drift_voltage": "r_v", "coordinated_fdi": "r_v",
    "drift_current": "r_id", "bias_ct_gain": "r_id", "theft_bypass": "r_id", "tamper_report": "r_id",
    "leakage": "r_id",
}
REGISTER_CHANNEL = "energy_import_kwh"   # a spike on the energy register is a register fault -> r_reg


def expected_signal(anomaly_type: str, params) -> str | None:
    if anomaly_type == "spike":
        p = json.loads(params) if isinstance(params, str) and params.startswith("{") else {}
        return "r_reg" if p.get("channel") == REGISTER_CHANNEL else None
    return SHAP_EXPECTED.get(anomaly_type)


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) and x[:1] in "[{" else x


def load_run(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out_dir = Path(out_dir)
    al = pd.read_csv(out_dir / "alerts.csv", parse_dates=["t_start"])
    for c in ("rules_fired", "shap_top", "linked_rule_alert"):
        if c not in al:
            al[c] = None
        al[c] = al[c].map(_lit)
    res = pd.read_csv(out_dir / "event_results.csv")
    summ = json.loads((out_dir / "summary.json").read_text())
    return al, res, summ


def _met(measure, value):
    op, t = TARGETS[measure]
    if value is None or not np.isfinite(value):
        return None
    return bool(value >= t) if op == ">=" else bool(value <= t)


def quality(al: pd.DataFrame, res: pd.DataFrame, summ: dict, seed, windows=("val", "test"),
            fit_seed=42) -> pd.DataFrame:
    """One row per (window, measure), plus one row per fault type for the SHAP match."""
    ev = res.set_index("event_id")
    # an alert belongs to the window of its start month (same split dates as the dataset)
    split_of = {"val": (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-08-01")),
                "test": (pd.Timestamp("2025-08-01"), pd.Timestamp("2026-01-01"))}
    ok = al.matched_event.notna() & (al.matched_event.map(ev.verdict) == al.verdict)
    rows = []
    for w in windows:
        s = summ[w]
        lo, hi = split_of[w]
        inw = (al.t_start >= lo) & (al.t_start < hi)
        fr = al[inw & (al.rule == "iforest") & al.matched_event.notna()].copy()
        fr["true_type"] = fr.matched_event.map(ev.anomaly_type)
        fr["expected"] = [expected_signal(t, ev.params.get(e)) for t, e in zip(fr.true_type, fr.matched_event)]
        fr["got"] = fr.shap_top.map(lambda x: x["feature"] if isinstance(x, dict) else None)
        fr["shown"] = [LINK_RULE_SIGNAL.get(lk["rule"], g) if isinstance(lk, dict) else g
                       for lk, g in zip(fr.linked_rule_alert, fr.got)]
        scored = fr[fr.expected.notna()]
        n_gap = int(s.get("events_with_energy_gap_alert", 0))
        vals = {
            "verdict_accuracy": (s["verdict_accuracy_on_detected"], int(round(s["event_recall"] * s["events"]))),
            "right_home_in_top3": (s.get("suspect_top3_hit_rate"), n_gap),
            "calibration_error_ece": (ece(al.confidence[inw].to_numpy(), ok[inw].to_numpy()), int(inw.sum())),
            "shap_main_signal_match": ((scored.expected == scored.got).mean() if len(scored) else np.nan,
                                       int(len(scored))),
            "shap_main_signal_match_as_shown": ((scored.expected == scored.shown).mean() if len(scored) else np.nan,
                                                int(len(scored))),
        }
        note = {"calibration_error_ece": "in-sample: the confidence table was fitted on these months"
                if w == "val" and str(seed) == str(fit_seed) else ""}
        for m, (v, n) in vals.items():
            v = None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 3)
            op, t = TARGETS[m]
            rows.append({"seed": seed, "window": w, "measure": m, "fault_type": "all", "value": v,
                         "target": f"{op} {t}", "met": _met(m, v), "n": n, "note": note.get(m, "")})
        for t, g in fr.groupby("true_type"):
            exp = g.expected.iloc[0]
            for m, col in (("shap_main_signal_match", "got"), ("shap_main_signal_match_as_shown", "shown")):
                rows.append({"seed": seed, "window": w, "measure": m, "fault_type": t,
                             "value": round(float((g.expected == g[col]).mean()), 3) if exp else None,
                             "target": f">= {TARGETS[m][1]}", "met": None, "n": int(len(g)),
                             "note": f"expected {exp}; got " + ", ".join(f"{k} {v}" for k, v in g[col].value_counts().items())
                             if exp else "no expected signal: reported, left out of the rate"})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="SEED=OUT_DIR (alerts.csv, event_results.csv, summary.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-seed", default="42", help="seed whose val months the confidence table was fitted on")
    a = ap.parse_args(argv)
    parts = []
    for spec in a.run:
        seed, d = spec.split("=", 1)
        parts.append(quality(*load_run(d), seed=seed, fit_seed=a.fit_seed))
    q = pd.concat(parts, ignore_index=True)
    q.to_csv(a.out, index=False)
    print(q[q.fault_type == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
