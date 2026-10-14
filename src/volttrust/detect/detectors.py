"""Phase 1 baseline detector, following the Goal 1 strategy.

L1  physics and rule checks       -> per-interval flags on meters, feeder heads and transformer
L2  Isolation Forest + persistent-offset tests on the voltage-profile residual (drift / bias)
L3  rule-table fusion             -> one verdict per alert: grid_event | equipment_fault | data_quality

Every flag is causal (uses only the present and past). The leakage rule decides once per night,
at 05:00, and holds its decision until the next night.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import Data, Residuals

PH = "ABC"

# Verdict, subtype and base confidence per rule. Order = precedence when several fire.
RULES = [
    # rule id                 verdict             subtype                  base confidence
    ("head_outage",           "grid_event",       "outage",                0.95),
    ("head_phase_loss",       "equipment_fault",  "phase_loss",            0.90),
    ("head_sag",              "grid_event",       "sag",                   0.90),
    ("head_mv_step",          "grid_event",       "mv_voltage_step",       0.85),
    ("head_frequency",        "grid_event",       "frequency_excursion",   0.90),
    ("group_voltage_drop",    "equipment_fault",  "high_resistance_joint", 0.75),
    ("night_leakage",         "equipment_fault",  "leakage",               0.75),
    ("group_voltage_rise",    "data_quality",     "falsified",             0.80),
    ("dropout",               "data_quality",     "dropout",               0.99),
    ("stuck",                 "data_quality",     "stuck",                 0.95),
    ("tamper_alarm",          "data_quality",     "tampering",             0.80),
    ("power_identity_persist", "data_quality",    "tampering",             0.90),
    ("power_identity",        "data_quality",     "spike",                 0.85),
    ("voltage_spike",         "data_quality",     "spike",                 0.90),
    ("register_jump",         "data_quality",     "spike",                 0.90),
    ("pv_reading_stuck",      "data_quality",     "pv_reading_stuck",      0.90),
    ("pv_inverter_trip",      "equipment_fault",  "pv_inverter_trip",      0.85),
    ("pv_over_report",        "data_quality",     "pv_reading_high",       0.80),   # blames nobody
    ("pv_underperform",       "equipment_fault",  "pv_soiling",            0.75),
    ("balance_gap_phase",     "data_quality",     "energy_gap",            0.70),
    ("balance_gap_phase_slow", "data_quality",    "energy_gap",            0.60),
    ("voltage_offset",        "data_quality",     "bias",                  0.70),
    ("iforest",               "data_quality",     "unusual_pattern",       0.50),
]
RULE_IDX = {r[0]: i for i, r in enumerate(RULES)}
# Transformer rules, raised on "TX" (not part of the meter precedence above)
TX_RULES = {
    "transformer_overload":       ("equipment_fault", "overload_risk",        0.70),   # notice
    "transformer_overload_alarm": ("equipment_fault", "overload_risk",        0.90),   # alarm
    "oil_temperature_residual":   ("equipment_fault", "transformer_overheat", 0.80),   # needs the monitor
}
CONF_CAP = 0.97          # no measured confidence above this
AGREE_PRIOR_BONUS = 0.04  # prior for alerts where 2 or more rules agree


def hand_set_confidence(rule: str) -> float:
    if rule in TX_RULES:
        return TX_RULES[rule][2]
    return RULES[RULE_IDX[rule]][3]


def prior_confidence(rule: str, n_fired: int = 1) -> float:
    """The hand-set value, used until (and where) a measured one exists."""
    base = hand_set_confidence(rule)
    return min(CONF_CAP, base + AGREE_PRIOR_BONUS) if n_fired >= 2 else min(CONF_CAP, base)


# Pushed by the meter or the head at once; every other rule runs on the
# load-profile batches the concentrator collects every `Site.batch_hours`
FAST_RULES = ("head_outage", "head_phase_loss", "tamper_alarm")
# Raised on a feeder-phase, never on a home: the suspect list goes to a person
PHASE_RULES = ("balance_gap_phase", "balance_gap_phase_slow")
N_SUSPECTS = 3


@dataclass
class Params:
    v_spike: float = 15.0          # V away from the voltage-profile model
    v_offset_min: float = 1.5      # V, persistent offset (trailing 4 h median)
    v_offset_sigma: float = 5.0
    id_tol: float = 0.05           # relative power-identity tolerance
    reg_tol: float = 0.02          # kWh per interval
    stuck_run: int = 3             # identical consecutive rows
    bal_sigma: float = 8.0
    bal_min_kw: float = 0.03       # retuned on val after calibrating homes to ~1,450 kWh/yr
    bal_rel: float = 0.02          # share of the head-meter phase power
    bal_slow_kw: float = 0.01      # 7-day mean gap, for small CT errors and slow drifts
    bal_beta: float = 0.04         # attributed under/over-reporting share
    bal_t: float = 6.0             # t-statistic of the attribution
    head_step: float = 0.035       # pu change vs the head's last 24 h median, on every head
    freq_tol: float = 0.2          # Hz
    leak_qp: float = 0.9           # night Q/P above this -> leakage
    oil_resid_c: float = 5.0
    group_min_drop: int = 2
    group_min_rise: int = 3
    if_quantile: float = 0.9999    # Isolation Forest threshold on train scores
    pv_learn_days: int = 21        # each home's normal output is learned from the first days
    pv_ghi_min: float = 150.0      # W/m2: only judge panels in real daylight
    pv_low: float = 0.90           # output / peers, median of ~4 sunny hours, below -> underperforming
    pv_high: float = 1.10          # output / peers, 2-hour median, above -> over-reporting
    pv_trip: float = 0.05          # output / expected, below while peers produce -> tripped
    pv_stuck_run: int = 4          # identical consecutive daytime readings
    # estimated solar homes (net meter + reference panel) wait longer than the inverter homes
    # (2 low readings, 16 sunny readings), because the estimate is noisier
    pv_est_trip_run: int = 6       # consecutive low readings before an estimated trip
    pv_est_window: int = 48        # sunny readings in the estimated underperform window (12 sunny hours)
    pv_recovery: float = 0.08      # a 1-day jump back up by this much confirms dust
    pv_aging_days: int = 60        # no recovery within this many days -> aging, inspect
    tamper_hold: int = 4           # intervals a tamper alarm keeps its meter flagged
    collection_gap_share: float = 0.5  # this share of meters missing at once = collection gap, not dropout
    overload_pu: float = 1.0       # notice: load above 100% of the rating ...
    overload_hours: float = 1.0    # ... for more than 1 h
    overload_oil_c: float = 105.0  # alarm: estimated top oil above this
    suspects_fallback: bool = True  # fill the list of homes to check with weaker evidence, marked as such
    gap: int = 2                   # merge flags separated by <= gap intervals


def _forward_fill_run(x: np.ndarray, n: int) -> np.ndarray:
    """Mark the n rows ending at each True (the whole run that triggered it)."""
    return pd.DataFrame(x[::-1]).rolling(n + 1, min_periods=1).max().to_numpy()[::-1] > 0


def _attribute(r: np.ndarray, Pg: np.ndarray, window: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-window regression r ~ c + sum_j beta_j * P_j, refreshed every `step` intervals.
    Returns beta and t-statistics per meter, (T, n), held between refreshes."""
    T, n = Pg.shape
    beta = np.full((T, n), np.nan)
    tst = np.full((T, n), np.nan)
    for t in range(window, T + 1, step):
        y = r[t - window:t]
        X = np.column_stack([np.ones(window), Pg[t - window:t]])
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        if ok.sum() < window // 2:
            continue
        Xo, yo = X[ok], y[ok]
        coef, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
        res = yo - Xo @ coef
        dof = max(len(yo) - X.shape[1], 1)
        s2 = res @ res / dof
        try:
            cov = s2 * np.linalg.inv(Xo.T @ Xo)
        except np.linalg.LinAlgError:
            continue
        se = np.sqrt(np.clip(np.diag(cov)[1:], 1e-12, None))
        beta[t - 1:t - 1 + step] = coef[1:]
        tst[t - 1:t - 1 + step] = coef[1:] / se
    return beta, tst


def _attribution_score(beta, tst, Pg, r, window, P) -> np.ndarray:
    """Keep a meter only if its coefficient is significant AND it explains most of the gap:
    beta_j * mean(P_j) must be 50 to 150% of mean(gap) over the same window."""
    mp = pd.DataFrame(Pg).rolling(window, min_periods=window // 2).mean().to_numpy()
    mr = pd.Series(r).rolling(window, min_periods=window // 2).mean().to_numpy()[:, None]
    share = beta * mp / np.where(np.abs(mr) > 1e-6, mr, np.nan)
    ok = (np.abs(beta) > P.bal_beta) & (np.abs(tst) > P.bal_t) & (share > 0.5) & (share < 1.5)
    return np.where(ok, np.abs(beta), 0.0)


def meter_flags(D: Data, R: Residuals, P: Params) -> tuple[dict, dict]:
    """Return {rule: (T,H) bool} and {name: (T,H) float} evidence scores."""
    m = D.m
    V, I, Pk, Q = m["voltage_v"], m["current_a"], m["active_power_kw"], m["reactive_power_kvar"]
    f: dict[str, np.ndarray] = {}
    ev: dict[str, np.ndarray] = {}

    # a meter that sent nothing, not a read the head-end knows it failed, and not one of many meters
    # missing at once (a collection gap on the concentrator raises no meter fault)
    miss = np.isnan(V)
    collected = D.collected if D.collected is not None else np.ones_like(miss)
    gap_rows = miss.mean(1) >= P.collection_gap_share
    f["dropout"] = miss & collected & ~gap_rows[:, None]
    ev["collection_gap"] = (miss & ~collected) | (miss & gap_rows[:, None])
    live = V > 50

    # tamper alarms pushed by the meter (cover opened, magnet)
    tam = np.zeros_like(miss)
    if D.alarms is not None and len(D.alarms):
        a = D.alarms[D.alarms.alarm.str.startswith("tamper")]
        for t, j in zip(a.t.astype(int), a.meter.astype(int)):
            tam[t:t + P.tamper_hold, j] = True
    f["tamper_alarm"] = tam

    eq = np.ones_like(V, bool)
    for c in ("voltage_v", "voltage_min_v", "voltage_max_v", "current_a", "active_power_kw"):
        x = m[c]
        eq &= x == np.roll(x, 1, axis=0)
    run = pd.DataFrame(eq.astype(float)).rolling(P.stuck_run, min_periods=P.stuck_run).sum().to_numpy() >= P.stuck_run
    f["stuck"] = _forward_fill_run(run, P.stuck_run) & live

    rv = R.r_v
    f["voltage_spike"] = np.abs(rv) > P.v_spike
    s_kva = V * I / 1000
    bad_id = (np.abs(R.r_id) > P.id_tol) & (s_kva > 0.05)
    persist = pd.DataFrame(bad_id.astype(float)).rolling(8, min_periods=1).sum().to_numpy() >= 4
    f["power_identity_persist"] = bad_id & persist
    f["power_identity"] = bad_id & ~persist
    f["register_jump"] = np.abs(R.r_reg) > P.reg_tol
    ev["identity_residual"] = R.r_id

    # persistent offset vs the voltage-profile model (bias and drift)
    rv_clean = np.where(np.abs(rv) > P.v_spike, np.nan, rv)
    med = pd.DataFrame(rv_clean).rolling(16, min_periods=8).median().to_numpy()
    thr = np.maximum(P.v_offset_min, P.v_offset_sigma * R.sigma_v)[None, :]
    off = np.abs(med) > thr
    ev["voltage_offset_v"] = med
    # slope over the last 2 days separates drift from a constant bias
    tt = np.arange(D.T, dtype=float)
    df = pd.DataFrame(rv_clean)
    cov = df.rolling(192, min_periods=96).cov(pd.Series(tt)).to_numpy()
    var = pd.Series(tt).rolling(192, min_periods=96).var().to_numpy()[:, None]
    ev["voltage_slope_v_per_day"] = cov / var * 96

    # spatial groups: several meters with the same persistent offset
    drop = np.zeros_like(off)
    rise = np.zeros_like(off)
    known = D.phase_known if D.phase_known is not None else np.ones(D.H, bool)
    F = int(D.feeder.max()) + 1
    for fd in range(F):
        sel_f = D.feeder == fd
        up = off & (med > 0) & sel_f[None, :]
        n_up = up.sum(1)
        rise |= up & (n_up >= P.group_min_rise)[:, None]
        # a joint drops the voltage of the homes on one phase; homes of unknown phase form their own group
        for sel in [sel_f & known & (D.phase == p) for p in range(3)] + [sel_f & ~known]:
            dn = off & (med < 0) & sel[None, :]
            drop |= dn & (dn.sum(1) >= P.group_min_drop)[:, None]
    f["group_voltage_drop"] = drop
    f["group_voltage_rise"] = rise
    f["voltage_offset"] = off & ~drop & ~rise

    # EN 50160 limit violations with a normal residual are real grid conditions, not data problems:
    # reported as power-quality notices, not alerts
    ev["voltage_out_of_limits"] = live & ((V < 207) | (V > 253))


    # leakage: a night-time load with very low power factor, decided once per night
    hour = D.idx.hour.values
    night = (hour >= 1) & (hour < 5)
    qp = np.where(night[:, None] & (Pk > 0.02), Q / np.maximum(Pk, 1e-3), np.nan)
    day = D.idx.normalize()
    qp_day = pd.DataFrame(qp).groupby(day).median()
    # decided at 05:00 from that night's readings, then held until the next decision (causal)
    key = (D.idx - pd.Timedelta(hours=5)).normalize()
    leak_day = (qp_day > P.leak_qp).reindex(key, fill_value=False).to_numpy(bool)
    f["night_leakage"] = leak_day & live
    ev["night_q_over_p"] = qp_day.reindex(key).to_numpy()

    pf, pe = pv_flags(D, P, live)
    f.update(pf)
    ev.update(pe)
    return f, ev


@dataclass
class Balance:
    """Energy balance per feeder-phase (or per feeder while a home's phase is unknown). The gap raises
    an alert on that group itself; the attribution regression only ranks its homes as suspects for a
    person to check."""
    groups: list            # dicts: asset, feeder, phases, meters, fast (T,), slow (T,), gap_kw (T,)
    score: np.ndarray       # (T, H) suspect score: |beta| when significant and explaining the gap, else 0
    beta: np.ndarray        # (T, H)
    tstat: np.ndarray       # (T, H)


def balance_flags(D: Data, R: Residuals, P: Params) -> Balance:
    """The gap r(t) = head P - sum(meter P) - losses is regressed on every meter's reported power:
    a meter that reports a fraction a of its true power leaves r = (1/a - 1) * P_reported,
    so its coefficient stands out while honest meters stay near zero. Missing readings are
    rebuilt from the energy registers (R.P_fill)."""
    Pk = R.P_fill
    score = np.zeros((D.T, D.H))
    beta = np.full((D.T, D.H), np.nan)
    tstat = np.full((D.T, D.H), np.nan)
    groups = []
    for g in R.bal_groups:
        sel, fd, phs = g["meters"], g["feeder"], list(g["phases"])
        r0 = g["r"]
        r = pd.Series(r0)
        head_p = np.abs(D.h["active_power_kw"][:, fd, phs].sum(1))
        lim = P.bal_sigma * g["sigma"] + P.bal_min_kw + P.bal_rel * head_p
        rm = r.rolling(16, min_periods=8).mean().to_numpy()
        gap = np.abs(rm) > lim
        gap &= pd.Series(gap.astype(float)).rolling(16, min_periods=1).sum().to_numpy() >= 8
        b1, t1 = _attribute(r0, Pk[:, sel], window=96, step=4)
        s1 = _attribution_score(b1, t1, Pk[:, sel], r0, 96, P)

        # slow version: one-week window, catches gaps too small to see within a day
        w = 7 * 96
        rw = r.rolling(w, min_periods=w // 2).mean().to_numpy()
        rd = r.rolling(96, min_periods=48).mean().to_numpy()      # still present today
        b7, t7 = _attribute(r0, Pk[:, sel], window=w, step=16)
        s7 = _attribution_score(b7, t7, Pk[:, sel], r0, w, P)
        sl = (np.abs(rw) > P.bal_slow_kw) & (np.abs(rd) > P.bal_slow_kw) & (np.sign(rw) == np.sign(rd))

        groups.append({**{k: g[k] for k in ("asset", "feeder", "phases", "meters")},
                       "fast": gap, "slow": sl & ~gap, "gap_kw": np.where(gap, rm, rw)})
        # suspects: the fast regression while the fast gap is open, the weekly one otherwise
        use_fast = gap[:, None]
        score[:, sel] = np.where(use_fast, s1, s7)
        beta[:, sel] = np.where(use_fast, b1, b7)
        tstat[:, sel] = np.where(use_fast, t1, t7)
    return Balance(groups, score, beta, tstat)


def reference_pv(D: Data) -> tuple[np.ndarray, str]:
    """kW per installed kWp of a clean panel in this neighborhood: the measured reference panel when
    the site has one, otherwise a generic PV model on hourly city weather."""
    if D.site is not None and D.site.use_reference_panel and "ref_pv_kw_per_kwp" in D.ctx:
        return D.ctx["ref_pv_kw_per_kwp"].reindex(D.idx).to_numpy(), "reference_panel"
    hourly = D.ctx[["ghi_wm2", "ambient_temp_c"]].resample("1h").mean().reindex(D.idx, method="ffill")
    g, t = hourly.ghi_wm2.to_numpy(), hourly.ambient_temp_c.to_numpy()
    # 0.82 system derate, -0.4 %/degC above a 25 degC cell, cell ~ air + 0.03 degC per W/m2
    return np.clip(g / 1000 * 0.82 * (1 - 0.004 * (t + 0.03 * g - 25)), 0, None), "weather_model"


def _sunny_rolling(x: np.ndarray, lit: np.ndarray, window: int, q: float | None) -> np.ndarray:
    """Rolling median (q None) or quantile over the last `window` daylight readings, held through
    the night (nights carry no evidence)."""
    out = np.full_like(x, np.nan)
    r = pd.DataFrame(x[lit]).rolling(window, min_periods=window // 2)
    out[lit] = (r.median() if q is None else r.quantile(q)).to_numpy()
    return pd.DataFrame(out).ffill().to_numpy()


def pv_flags(D: Data, P: Params, live: np.ndarray) -> tuple[dict, dict]:
    """Rooftop PV checks.

    Inverter homes (opt-in): each home's output per unit of irradiance (kW per kW/m2) is learned from
    the first days of the year. Peer vs peer: all panels see the same sun, so a home's ratio is
    compared with the reference panel's ratio at the same time (or, without one, the median of the
    opt-in homes); clouds move every panel together and cancel out.

    Estimated homes: production is estimated as the export per kW of
    the reference panel, x = -net / ref, scaled by each home's own sunny export capacity learned over
    the same first days. Household load hides part of the panel, so the window uses the upper
    envelope (90th percentile) instead of the median. Same thresholds as inverter homes.
    """
    H = D.H
    pv = D.m["pv_power_kw"]
    ghi = D.ctx["ghi_wm2"].reindex(D.idx).to_numpy() / 1000
    src = D.pv_source if D.pv_source is not None else np.where(np.isfinite(pv).any(0), "inverter_optin", "")
    inv = (src == "inverter_optin") & np.isfinite(pv).any(axis=0)
    est = src == "estimated"
    sun = (ghi * 1000 > P.pv_ghi_min)[:, None]
    lit = sun[:, 0]
    learn = np.arange(D.T) < P.pv_learn_days * 96
    ref, ref_kind = reference_pv(D)
    f = {k: np.zeros((D.T, H), bool) for k in ("pv_inverter_trip", "pv_reading_stuck", "pv_over_report",
                                                  "pv_underperform")}
    q = np.full((D.T, H), np.nan)
    day_q = np.full((D.T, H), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        # ---- inverter homes, with the reference panel as an extra peer
        cols = np.column_stack([pv, ref * 1.0])                    # last column: the reference panel
        has = np.r_[inv, True]
        per_ghi = np.where(sun & learn[:, None], cols / ghi[:, None], np.nan)
        k = np.nanmedian(per_ghi, axis=0)
        r = np.where(sun & has, cols / (k * ghi[:, None]), np.nan)
        r[:, :H] = np.where(live, r[:, :H], np.nan)
        # the street's reference: the clean reference panel when the site has one (with only 3 opt-in
        # homes, two faulty panels at once would drag a median down), else the opt-in homes' median
        if ref_kind == "reference_panel":
            peer = r[:, H:]
        else:
            peer = np.nanmedian(r[:, :H], axis=1, keepdims=True)
        qi = (r / peer)[:, :H]
        trip = sun & live & (r[:, :H] < P.pv_trip) & (peer > 0.5) & inv
        f["pv_inverter_trip"] = trip & (pd.DataFrame(trip.astype(float)).rolling(2, min_periods=2).sum().to_numpy() >= 2)
        same = (pv == np.roll(pv, 1, axis=0)) & (pv > 0.05) & sun
        run = pd.DataFrame(same.astype(float)).rolling(P.pv_stuck_run, min_periods=P.pv_stuck_run).sum().to_numpy()
        f["pv_reading_stuck"] = _forward_fill_run(run >= P.pv_stuck_run, P.pv_stuck_run)
        # readings already explained by a trip or a frozen value are left out
        qi = np.where(trip | same | f["pv_reading_stuck"], np.nan, qi)
        dq_i = _sunny_rolling(qi, lit, 16, None)                   # about 4 sunny hours
        sq_i = _sunny_rolling(qi, lit, 8, None)                    # about 2 sunny hours
        f["pv_over_report"] = live & inv & (sq_i > P.pv_high)
        q[:, inv], day_q[:, inv] = qi[:, inv], dq_i[:, inv]

        # ---- estimated homes, from the net meter
        if est.any():
            x = -D.m["active_power_kw"][:, est] / np.where(ref > 0.05, ref, np.nan)[:, None]
            cap = np.nanquantile(np.where((sun & learn[:, None]), x, np.nan), 0.9, axis=0)
            re = np.where(sun & live[:, est], x / cap, np.nan)
            peer_e = np.nanmedian(re, axis=1, keepdims=True)
            trip_e = (re < P.pv_trip) & (peer_e > 0.5)
            n = P.pv_est_trip_run
            run_e = pd.DataFrame(trip_e.astype(float)).rolling(n, min_periods=n).sum().to_numpy() >= n
            f["pv_inverter_trip"][:, est] = _forward_fill_run(run_e, n) & trip_e
            env = _sunny_rolling(np.where(trip_e, np.nan, re), lit, P.pv_est_window, 0.9)
            dq_e = env / np.nanmedian(env, axis=1, keepdims=True)
            q[:, est] = re / peer_e
            day_q[:, est] = dq_e
    solar = inv | est
    f["pv_underperform"] = live & solar & (day_q < P.pv_low)
    ev = {"pv_ratio_to_peers": q, "pv_ratio_day": day_q,
          "pv_ratio_change_1d": day_q - np.roll(day_q, 96, axis=0)}
    return f, ev


def head_flags(D: Data, R: Residuals, P: Params) -> dict:
    h = D.h
    V = h["voltage_v"]
    zero = V < 1
    all_zero = zero.all(axis=2, keepdims=True)
    f = {
        "head_outage": zero & all_zero,
        "head_phase_loss": zero & ~all_zero,
        "head_sag": (h["voltage_min_v"] < 0.9 * V) & ~zero,
        "head_frequency": np.abs(h["frequency_hz"] - 50) > P.freq_tol,
    }
    step = np.abs(R.head_dv) > P.head_step
    mv = step.reshape(D.T, -1).all(axis=1)          # an MV event moves every head together
    f["head_mv_step"] = np.repeat(mv[:, None, None], V.shape[1], 1).repeat(3, 2) & ~zero
    return f


def isolation_forest(D: Data, R: Residuals, P: Params, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-row Isolation Forest on physics residuals and load-shape features. Train split only."""
    Pk = D.m["active_power_kw"]
    hour = D.idx.hour.values
    train = D.split == "train"
    prof_mu = np.zeros((24, D.H))
    prof_sd = np.ones((24, D.H))
    for hr in range(24):
        rows = train & (hour == hr)
        prof_mu[hr] = np.nanmean(Pk[rows], axis=0)
        prof_sd[hr] = np.nanstd(Pk[rows], axis=0) + 0.02
    pz = (Pk - prof_mu[hour]) / prof_sd[hour]
    rv_sd = pd.DataFrame(R.r_v).rolling(8, min_periods=4).std().to_numpy()
    # physics residuals only: load-shape features learned in winter would flag every summer AC peak
    del pz
    feats = np.stack([R.r_v, rv_sd, R.r_id, np.nan_to_num(R.r_reg)], axis=-1)
    X = feats.reshape(-1, feats.shape[-1])
    ok = np.isfinite(X).all(1)
    tr = np.repeat(train, D.H) & ok
    rng = np.random.default_rng(seed)
    sample = rng.choice(np.where(tr)[0], size=min(200_000, tr.sum()), replace=False)
    model = IsolationForest(n_estimators=200, max_samples=4096, random_state=seed).fit(X[sample])
    score = np.full(len(X), np.nan)
    score[ok] = -model.score_samples(X[ok])
    thr = np.quantile(score[sample], P.if_quantile)
    score = score.reshape(D.T, D.H)
    forest = {"model": model, "features": feats, "background": X[sample[:1000]]}
    return score > thr, score, forest


IF_FEATURES = ("r_v", "rv_sd", "r_id", "r_reg")   # voltage gap to the physics model, its 2-hour jumpiness,
                                                  # power-identity residual, register residual


def explain_forest(al: pd.DataFrame, forest: dict, if_score: np.ndarray, meters: list) -> tuple[pd.DataFrame, dict]:
    """SHAP on the Isolation Forest: for each forest alert, at its peak-score row,
    each signal's push toward "unusual" (positive = more anomalous) and the signal that pushed most."""
    import time

    import shap
    t0 = time.perf_counter()
    al = al.copy()
    al["shap"], al["shap_top"] = None, None
    rows = al.index[al.rule == "iforest"]
    if not len(rows):
        return al, {"forest_alerts": 0}
    col = {m: j for j, m in enumerate(meters)}
    peaks, X = [], []
    for i in rows:
        j, s, e = col[al.at[i, "asset"]], int(al.at[i, "start_idx"]), int(al.at[i, "end_idx"])
        t = s + int(np.nanargmax(np.nan_to_num(if_score[s:e, j], nan=-np.inf)))
        peaks.append(t)
        X.append(np.nan_to_num(forest["features"][t, j]))
    X = np.asarray(X)
    bg = forest["background"]
    ex = shap.TreeExplainer(forest["model"], shap.maskers.Independent(bg, max_samples=len(bg)))
    # the forest's output grows with normality, our score is its negative: flip the sign
    sv = -np.asarray(ex.shap_values(X, check_additivity=False))
    for k, i in enumerate(rows):
        al.at[i, "shap"] = {f: round(float(v), 4) for f, v in zip(IF_FEATURES, sv[k])}
        top = int(np.argmax(sv[k]))
        al.at[i, "shap_top"] = {"feature": IF_FEATURES[top], "value": round(float(X[k, top]), 4)}
    dt = time.perf_counter() - t0
    return al, {"forest_alerts": int(len(rows)), "shap_seconds_total": round(dt, 2),
                "shap_ms_per_alert": round(dt / len(rows) * 1000, 1)}


# rule alerts a forest alert may lead with, and the signal each one speaks for
LINK_RULE_SIGNAL = {"group_voltage_drop": "r_v", "voltage_offset": "r_v", "group_voltage_rise": "r_v"}
LINK_HOURS = 6.0   # the forest never overlaps a rule alert on the same home


def link_rule_alerts(al: pd.DataFrame, hours: float = LINK_HOURS, rules=tuple(LINK_RULE_SIGNAL)) -> pd.DataFrame:
    """A forest alert leads its explanation with the reason of the nearest voltage-rule
    alert on the same home within `hours` (SHAP second). A rule alert and a forest alert never overlap
    on one home, since each reading gets one winning check, hence the window. Changes nothing in the
    forest, its signals, confidence or routing."""
    al = al.copy()
    al["linked_rule_alert"] = None
    cand = al[al.rule.isin(rules)]
    for i in al.index[al.rule == "iforest"]:
        a = al.loc[i]
        o = cand[cand.asset == a.asset]
        if not len(o):
            continue
        gap = np.maximum((o.t_start - a.t_end).dt.total_seconds(), (a.t_start - o.t_end).dt.total_seconds()) / 3600
        gap = gap[gap <= hours]
        if len(gap):
            k = gap.idxmin()
            al.at[i, "linked_rule_alert"] = {"alert_id": al.at[k, "alert_id"], "rule": al.at[k, "rule"],
                                             "hours_apart": round(float(max(gap[k], 0.0)), 2)}
    return al


@dataclass
class Detection:
    alerts: pd.DataFrame
    meter_rule: np.ndarray      # (T, H) index into RULES of the winning rule, -1 = no flag
    head_rule: np.ndarray       # (T, F, 3)
    tx_flag: np.ndarray         # (T,)
    flags: dict
    evidence: dict
    if_score: np.ndarray
    timing: dict = None


def _runs(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive), merging gaps <= gap."""
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    out, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            out.append((s, prev + 1))
            s = i
        prev = i
    out.append((s, prev + 1))
    return out


def transformer_load(D: Data) -> np.ndarray:
    """Transformer load in pu of its rating, from the substation reading (or the stand-in built from
    the homes when there is none): apparent power summed over feeders and phases."""
    S = np.hypot(D.h["active_power_kw"], D.h["reactive_power_kvar"])
    return np.nansum(S, axis=(1, 2)) / D.kva


def overload_flags(D: Data, P: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """No sensor needed: top oil is estimated from load and city air temperature with the IEC
    60076-7 model the simulator uses. Notice: load above `overload_pu` for more than `overload_hours`.
    Alarm: estimated top oil above `overload_oil_c`. Returns (notice, alarm, load_pu, oil_est)."""
    from ..sim.grid import top_oil_temperature
    load = transformer_load(D)
    amb = D.ctx["ambient_temp_c"].reindex(D.idx).interpolate(limit_direction="both").to_numpy()
    oil = top_oil_temperature(load, amb, 15.0)
    n = int(round(P.overload_hours * 4)) + 1                       # more than 1 h = 5 intervals
    over = load > P.overload_pu
    run = pd.Series(over.astype(float)).rolling(n, min_periods=n).sum().to_numpy() >= n
    notice = _forward_fill_run(run, n).ravel() & over
    return notice, oil > P.overload_oil_c, load, oil


def detected_index(rule: str, s: int, T: int, batch: int, fast_ok: bool = True) -> int:
    """Interval at which the alert reaches the operator: alarms pushed at
    once arrive one interval after the start; everything else waits for the next profile batch."""
    if rule in FAST_RULES and fast_ok:
        return min(s + 1, T)
    return min(-(-(s + 1) // batch) * batch, T)


def detect(D: Data, R: Residuals, P: Params | None = None, use_iforest: bool = True) -> Detection:
    with warnings.catch_warnings():
        # empty windows (nights, lost batches) are expected and give NaN, which every rule treats as "no flag"
        warnings.simplefilter("ignore", RuntimeWarning)
        return _detect(D, R, P or Params(), use_iforest)


def _detect(D: Data, R: Residuals, P: Params, use_iforest: bool) -> Detection:
    site = D.site
    has_head = site.has_head if site is not None else True
    mf, ev = meter_flags(D, R, P)
    hf = head_flags(D, R, P)
    if use_iforest:
        mf["iforest"], if_score, forest = isolation_forest(D, R, P)
    else:
        mf["iforest"], if_score, forest = np.zeros((D.T, D.H), bool), np.zeros((D.T, D.H)), None

    # head context propagates to the meters it feeds: a real event explains their readings
    head_rule = np.full(hf["head_outage"].shape, -1)
    for name in reversed([r[0] for r in RULES if r[0].startswith("head_")]):
        head_rule[hf[name]] = RULE_IDX[name]
    meter_rule = np.full((D.T, D.H), -1)
    for name, *_ in reversed(RULES):
        if name.startswith("head_") or name in PHASE_RULES:
            continue
        meter_rule[mf[name]] = RULE_IDX[name]
    known = D.phase_known if D.phase_known is not None else np.ones(D.H, bool)
    # a home of unknown phase takes any event on its feeder (the highest-precedence one)
    any_phase = np.where(head_rule >= 0, head_rule, 10 ** 6).min(axis=2)
    any_phase = np.where(any_phase == 10 ** 6, -1, any_phase)
    from_head = np.where(known, head_rule[:, D.feeder, np.where(known, D.phase, 0)], any_phase[:, D.feeder])
    any_meter = meter_rule >= 0
    # a head event takes precedence for meters flagged at the same time (and zero-voltage meters)
    zero_v = D.m["voltage_v"] < 1
    use_head = (from_head >= 0) & (any_meter | zero_v)
    meter_rule = np.where(use_head, from_head, meter_rule)

    # energy balance: an alert on the group, unless a real head event already explains it.
    # Without a substation reading there is nothing to balance against.
    bal = balance_flags(D, R, P) if has_head else Balance([], np.zeros((D.T, D.H)),
                                                          np.full((D.T, D.H), np.nan), np.full((D.T, D.H), np.nan))
    for g in bal.groups:
        col = np.full(D.T, -1)
        col[g["slow"]] = RULE_IDX["balance_gap_phase_slow"]
        col[g["fast"]] = RULE_IDX["balance_gap_phase"]
        col[(head_rule[:, g["feeder"], list(g["phases"])] >= 0).any(1)] = -1
        g["rule"] = col

    # transformer: overload estimate always; cooling-fault check only with the oil monitor
    notice, alarm, load_pu, oil_est = overload_flags(D, P)
    ev["transformer_load_pu"], ev["top_oil_estimate_c"] = load_pu, oil_est
    monitor = bool(site.transformer_monitor) if site is not None else np.isfinite(R.oil_resid).any()
    oil = pd.Series(R.oil_resid).rolling(96, min_periods=48).mean().to_numpy()
    tx = {"oil_temperature_residual": (oil > P.oil_resid_c) if monitor else np.zeros(D.T, bool),
          "transformer_overload_alarm": alarm, "transformer_overload": notice & ~alarm}
    tx_flag = np.zeros(D.T, bool)
    for v in tx.values():
        tx_flag |= v

    alerts = []
    share = (meter_rule >= 0)
    feeder_share = np.zeros((D.T, D.H))
    for fd in np.unique(D.feeder):
        sel = D.feeder == fd
        feeder_share[:, sel] = share[:, sel].mean(1, keepdims=True)

    def add(asset, s, e, rules_idx, scores, share_val, feeder=None, phase=None):
        vals, counts = np.unique(rules_idx[rules_idx >= 0], return_counts=True)
        top = int(vals[np.argmax(counts)]) if len(vals) else int(rules_idx.max())
        rid, verdict, subtype, base = RULES[top]
        fired = sorted({RULES[i][0] for i in vals})
        # the measured value per rule replaces this prior once calibrate.apply runs;
        # confidence_hand_set keeps the old stated value (+0.02 per extra rule) for the before/after report
        alerts.append({"asset": asset, "feeder": feeder, "phase": phase, "start_idx": s, "end_idx": e,
                       "t_start": D.idx[s], "t_end": D.idx[e - 1] + pd.Timedelta(minutes=15),
                       "verdict": verdict, "subtype": subtype, "rule": rid, "rules_fired": fired,
                       "confidence": round(prior_confidence(rid, len(fired)), 3),
                       "confidence_hand_set": round(min(0.99, base + 0.02 * (len(fired) - 1)), 3),
                       "neighbors_affected": round(float(share_val), 3),
                       **scores})

    for j, mid in enumerate(D.meters):
        col = meter_rule[:, j]
        for s, e in _runs(col >= 0, P.gap):
            seg = col[s:e]
            sc = {
                "max_abs_voltage_resid_v": float(np.nanmax(np.abs(R.r_v[s:e, j]), initial=0)),
                "max_abs_identity_resid": float(np.nanmax(np.abs(R.r_id[s:e, j]), initial=0)),
                "voltage_slope_v_per_day": float(np.nan_to_num(ev["voltage_slope_v_per_day"][e - 1, j])),
                "iforest_score": float(np.nan_to_num(np.nanmax(if_score[s:e, j], initial=0))),
                "pv_ratio_to_peers": float(np.nan_to_num(np.nanmedian(ev["pv_ratio_to_peers"][s:e, j]), nan=1.0))
                if np.isfinite(ev["pv_ratio_to_peers"][s:e, j]).any() else np.nan,
                "pv_ratio_change_1d": float(ev["pv_ratio_change_1d"][s, j]),
            }
            ph = PH[D.phase[j]] if known[j] else None
            add(mid, s, e, seg, sc, feeder_share[s:e, j].max(), f"F{D.feeder[j] + 1}", ph)

    F = head_rule.shape[1]
    for fd in range(F):
        for p in range(3):
            col = head_rule[:, fd, p]
            for s, e in _runs(col >= 0, P.gap):
                add(f"F{fd + 1}-{PH[p]}", s, e, col[s:e], {}, 1.0, f"F{fd + 1}", PH[p])
    for g in bal.groups:
        col, sel, fd = g["rule"], g["meters"], g["feeder"]
        phase = PH[g["phases"][0]] if len(g["phases"]) == 1 else None
        for s, e in _runs(col >= 0, P.gap):
            sc = np.nan_to_num(bal.score[s:e, sel]).mean(0)
            rank = [k for k in np.argsort(-sc)[:N_SUSPECTS] if sc[k] > 0]
            n_strong = len(rank)
            if P.suspects_fallback and len(rank) < N_SUSPECTS:
                # fill with the homes whose coefficient best explains the gap's sign
                sg = np.sign(np.nanmean(g["gap_kw"][s:e]))
                tt = np.nan_to_num(np.nanmedian(bal.tstat[s:e, sel] * sg, axis=0), nan=-np.inf)
                rank += [k for k in np.argsort(-tt) if k not in rank and tt[k] > 0][:N_SUSPECTS - len(rank)]
            suspects = [{"meter": D.meters[sel[k]],
                         "beta": round(float(np.nanmedian(bal.beta[s:e, sel[k]])), 3),
                         "t": round(float(np.nanmedian(bal.tstat[s:e, sel[k]])), 1),
                         "evidence": "strong" if i < n_strong else "weak"} for i, k in enumerate(rank)]
            add(g["asset"], s, e, col[s:e],
                {"gap_kw": round(float(np.nanmean(g["gap_kw"][s:e])), 3), "suspects": suspects},
                1.0, f"F{fd + 1}", phase)
    for rule, flag in tx.items():
        verdict, subtype, conf = TX_RULES[rule]
        for s, e in _runs(flag, P.gap):
            alerts.append({"asset": "TX", "start_idx": s, "end_idx": e, "t_start": D.idx[s],
                           "t_end": D.idx[e - 1] + pd.Timedelta(minutes=15), "verdict": verdict,
                           "subtype": subtype, "rule": rule, "rules_fired": [rule], "confidence": conf,
                           "confidence_hand_set": conf,
                           "neighbors_affected": 0.0, "oil_resid_c": float(np.nanmax(oil[s:e])),
                           "load_pu_max": round(float(np.nanmax(load_pu[s:e])), 3),
                           "top_oil_estimate_c": round(float(np.nanmax(oil_est[s:e])), 1)})

    al = pd.DataFrame(alerts)
    if len(al):
        # drift vs bias: a clear trend in the voltage residual means drift
        is_off = al.rule == "voltage_offset"
        al.loc[is_off & (al.voltage_slope_v_per_day.abs() > 0.15), "subtype"] = "drift"
        al = _panel_dust_or_aging(al, ev, D, P)
        # when the alert reaches a person: at once for pushed alarms, else at the next profile batch.
        # Without a substation reading an outage is pushed by the homes' power-cut alarms.
        batch = int((site.batch_hours if site is not None else 12) * 4)
        cut = np.zeros((D.T, int(D.feeder.max()) + 1), bool)
        if D.alarms is not None and len(D.alarms):
            a = D.alarms[D.alarms.alarm == "power_cut"]
            for t, j in zip(a.t.astype(int), a.meter.astype(int)):
                cut[max(t - 2, 0):t + 3, D.feeder[j]] = True
        def det_idx(r):
            fast_ok = True
            if not has_head and r.rule in ("head_outage", "head_phase_loss") and isinstance(r.feeder, str):
                fast_ok = bool(cut[r.start_idx, int(r.feeder[1:]) - 1])
            return detected_index(r.rule, int(r.start_idx), D.T, batch, fast_ok)
        al["detected_idx"] = [det_idx(r) for r in al.itertuples()]
        al["detected_at"] = [D.idx[min(i, D.T - 1)] + pd.Timedelta(minutes=15 if i >= D.T else 0)
                             for i in al.detected_idx]
        al = al.sort_values("t_start").reset_index(drop=True)
        al.insert(0, "alert_id", [f"AL{i + 1:05d}" for i in range(len(al))])
    timing = {}
    if forest is not None and len(al):
        al, timing = explain_forest(al, forest, if_score, D.meters)
    if len(al):
        al = link_rule_alerts(al)
    flags = {**mf, **hf, **{f"tx_{k}": v for k, v in tx.items()}}
    return Detection(al, meter_rule, head_rule, tx_flag, flags, ev, if_score, timing)


def _panel_dust_or_aging(al: pd.DataFrame, ev: dict, D: Data, P: Params) -> pd.DataFrame:
    """Panels losing output. A one-day drop of more than 8 points is a failed string
    (degradation). A slow slide is first `pv_soiling_likely`; a sudden recovery (the ratio jumps back
    up by more than 8 points within a day) confirms dust; no recovery within 60 days relabels it
    `pv_aging_inspect` for a person to inspect."""
    al = al.copy()
    is_low = al.rule == "pv_underperform"
    al.loc[is_low, "subtype"] = "pv_soiling_likely"
    al.loc[is_low & (al.pv_ratio_change_1d < -P.pv_recovery), "subtype"] = "pv_degradation"
    al["pv_status"] = None
    al["pv_resolved_at"] = pd.NaT
    ch = ev["pv_ratio_change_1d"]
    col = {m: j for j, m in enumerate(D.meters)}
    limit = P.pv_aging_days * 96
    for i in al.index[al.subtype == "pv_soiling_likely"]:
        j, s = col[al.at[i, "asset"]], int(al.at[i, "start_idx"])
        look = ch[s:min(s + limit, D.T), j]
        up = np.where(look > P.pv_recovery)[0]
        if len(up):
            al.at[i, "subtype"], al.at[i, "pv_status"] = "pv_soiling", "dust_confirmed_by_recovery"
            al.at[i, "pv_resolved_at"] = D.idx[s + up[0]]
        elif s + limit < D.T:
            al.at[i, "subtype"], al.at[i, "pv_status"] = "pv_aging_inspect", "no_recovery_in_60_days"
            al.at[i, "pv_resolved_at"] = D.idx[s + limit]
        else:
            al.at[i, "pv_status"] = "open"
    return al


def to_verdict_records(alerts: pd.DataFrame) -> list[dict]:
    """Goal 1 -> Goal 2/3 hand-off format (matches the escalation router's Event.from_verdict)."""
    out = []
    for a in alerts.itertuples():
        scores = {k: getattr(a, k) for k in ("max_abs_voltage_resid_v", "max_abs_identity_resid",
                                              "voltage_slope_v_per_day", "iforest_score", "gap_kw",
                                              "oil_resid_c", "pv_ratio_to_peers") if hasattr(a, k) and pd.notna(getattr(a, k))}
        evidence = {"rules_fired": list(a.rules_fired), "scores": scores,
                    "neighbors_affected": float(a.neighbors_affected)}
        if isinstance(getattr(a, "shap", None), dict):
            # which signal made the forest call this reading unusual
            evidence["shap"], evidence["shap_top"] = a.shap, a.shap_top
        if isinstance(getattr(a, "linked_rule_alert", None), dict):
            evidence["linked_rule_alert"] = a.linked_rule_alert
        if a.rule in PHASE_RULES:
            # homes to check, ranked by the balance regression; the system names no culprit
            evidence["suspects"] = list(a.suspects)
        out.append({
            "event_id": a.alert_id, "t_start": str(a.t_start), "t_end": str(a.t_end),
            "assets": [a.asset], "verdict": a.verdict, "subtype": a.subtype,
            "confidence": float(a.confidence),
            "evidence": evidence,
        })
    return out
