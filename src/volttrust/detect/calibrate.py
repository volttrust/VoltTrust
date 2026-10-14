"""Measured confidence per rule.

Each alert's confidence is how often alerts of its deciding rule had the right verdict in the
validation months (May to Jul), pulled toward the hand-set value and capped:

    conf = (correct + k * hand_set) / (n + k),  k = 10,  at most 0.97

"Right" means the alert matched an event and its verdict equals the event's verdict; a false alert
counts as wrong. Alerts where 2 or more rules agree get their own cell per rule, whose prior is the
hand-set value + 0.04; a cell with no validation alerts falls back to the rule's single-rule value.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .detectors import CONF_CAP, AGREE_PRIOR_BONUS, hand_set_confidence

K = 10


def _cell(rules_fired) -> str:
    return "agree" if len(rules_fired) >= 2 else "single"


def correct(al: pd.DataFrame, res: pd.DataFrame) -> pd.Series:
    """True where the alert matched an event and gave that event's verdict."""
    ev_verdict = dict(zip(res.event_id, res.verdict))
    return al.matched_event.notna() & (al.matched_event.map(ev_verdict) == al.verdict)


def fit(al: pd.DataFrame, res: pd.DataFrame, split_of_t: np.ndarray, window: str = "val",
        idx: pd.DatetimeIndex | None = None, source: str = "") -> dict:
    """Fit the table on the alerts that start in `window`. `al` is evaluate.match's output."""
    a = al[split_of_t[al.start_idx.values] == window].assign(ok=correct(al, res), cell=lambda x: x.rules_fired.map(_cell))
    table = {}
    for rule in sorted(set(al.rule)):
        g = a[a.rule == rule]
        base = hand_set_confidence(rule)
        single = g[g.cell == "single"]
        n1, c1 = len(single), int(single.ok.sum())
        p1 = min(CONF_CAP, base)
        v1 = min(CONF_CAP, (c1 + K * p1) / (n1 + K))
        agree = g[g.cell == "agree"]
        n2, c2 = len(agree), int(agree.ok.sum())
        p2 = min(CONF_CAP, base + AGREE_PRIOR_BONUS)
        v2 = min(CONF_CAP, (c2 + K * p2) / (n2 + K)) if n2 else v1
        table[rule] = {"hand_set": base,
                       "single": {"n": n1, "correct": c1, "prior": p1, "confidence": round(v1, 3)},
                       "agree": {"n": n2, "correct": c2, "prior": p2, "confidence": round(v2, 3),
                                 "fallback_to_single": n2 == 0}}
    win = ""
    if idx is not None:
        t = idx[split_of_t == window]
        win = f"{t[0].date()} to {t[-1].date()}"
    return {"method": "per-rule verdict hit rate, pulled toward the hand-set value",
            "k": K, "cap": CONF_CAP, "agree_prior_bonus": AGREE_PRIOR_BONUS,
            "fit_date": str(dt.date.today()), "data": source, "window": f"{window}: {win}", "table": table}


def apply(al: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Set each alert's confidence from the fitted table; rules the table never saw keep their prior."""
    tab = cal["table"]

    def conf(r):
        row = tab.get(r.rule)
        if row is None:
            base = hand_set_confidence(r.rule)
            return round(min(CONF_CAP, base + (AGREE_PRIOR_BONUS if len(r.rules_fired) >= 2 else 0)), 3)
        return row[_cell(r.rules_fired)]["confidence"]

    al = al.copy()
    if len(al):
        al["confidence"] = [conf(r) for r in al.itertuples()]
    return al


def ece(conf: np.ndarray, ok: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: share-weighted |accuracy - stated confidence| over 10 equal bins."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    if not len(conf):
        return float("nan")
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum((b == i).mean() * abs(ok[b == i].mean() - conf[b == i].mean())
                     for i in range(bins) if (b == i).any()))


def report(al: pd.DataFrame, res: pd.DataFrame, split_of_t: np.ndarray, splits=("test",)) -> tuple[dict, pd.DataFrame]:
    """ECE before (hand-set, +0.02 per extra rule) and after (measured), and stated vs observed per rule."""
    a = al[np.isin(split_of_t[al.start_idx.values], splits)].assign(ok=correct(al, res),
                                                                   cell=lambda x: x.rules_fired.map(_cell))
    summ = {"alerts": int(len(a)), "share_right": round(float(a.ok.mean()), 3) if len(a) else None,
            "ece_before": round(ece(a.confidence_hand_set, a.ok), 3),
            "ece_after": round(ece(a.confidence, a.ok), 3)}
    per = (a.groupby(["rule", "cell"])
            .agg(alerts=("ok", "size"), observed=("ok", "mean"), stated_before=("confidence_hand_set", "mean"),
                 stated_after=("confidence", "mean"))
            .round(3).reset_index())
    return summ, per
