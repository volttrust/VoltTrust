"""Load the dataset into wide (time x asset) arrays and build physics residuals.

Only observed channels are read here. Labels are loaded separately, for evaluation only.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import phase as phase_id
from .site import Site

CHANNELS = ["voltage_v", "voltage_min_v", "voltage_max_v", "current_a", "active_power_kw",
            "reactive_power_kvar", "power_factor", "energy_import_kwh", "energy_export_kwh",
            "pv_power_kw"]       # inverter production reading, NaN for homes without PV
HEAD_CHANNELS = ["voltage_v", "voltage_min_v", "current_a", "active_power_kw", "reactive_power_kvar",
                 "frequency_hz"]


@dataclass
class Data:
    idx: pd.DatetimeIndex
    meters: list            # meter ids, column order
    feeder: np.ndarray      # (H,) 0-based
    phase: np.ndarray       # (H,) 0..2
    pole: np.ndarray        # (H,)
    m: dict                 # channel -> (T, H)
    h: dict                 # channel -> (T, F, 3)
    tx: pd.DataFrame
    ctx: pd.DataFrame
    split: np.ndarray       # (T,)
    meta: pd.DataFrame
    site: Site = None
    alarms: pd.DataFrame = None   # pushed at once: power cut / restore, tamper (t index, meter index)
    collected: np.ndarray = None  # (T, H) False where the head-end log says the profile read failed
    pv_source: np.ndarray = None  # (H,) "inverter_optin" | "estimated" | ""
    phase_known: np.ndarray = None   # (H,) False: compared per feeder until a technician checks it
    phase_margin: np.ndarray = None  # (H,) score margin of the inferred phase (inf with STEG's map)
    kva: float = 100.0

    @property
    def T(self):
        return len(self.idx)

    @property
    def H(self):
        return len(self.meters)


def load(data_dir: str | Path, site: Site | None = None) -> Data:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _load(data_dir, site or Site())


def _load(data_dir: str | Path, site: Site) -> Data:
    d = Path(data_dir)
    topo = json.loads((d / "topology.json").read_text())
    tm = pd.DataFrame(topo["meters"])
    mt = pd.read_parquet(d / "meters.parquet", columns=["timestamp", "meter_id", "split"] + CHANNELS)
    meters = list(tm.meter_id)
    m = {c: mt.pivot(index="timestamp", columns="meter_id", values=c)[meters].to_numpy(np.float64)
         for c in CHANNELS}
    idx = pd.DatetimeIndex(mt.timestamp.unique()).sort_values()
    split = mt.drop_duplicates("timestamp").set_index("timestamp").loc[idx, "split"].astype(str).values
    feeder = tm.feeder.str[1:].astype(int).values - 1
    phase = tm.phase.map({"A": 0, "B": 1, "C": 2}).values
    F = topo["feeders"]
    T, H = len(idx), len(meters)

    # power factor from active and reactive power, not a stored value
    P, Q = m["active_power_kw"], m["reactive_power_kvar"]
    S = np.hypot(P, Q)
    m["power_factor"] = np.where(np.isnan(S), np.nan, np.where(S > 1e-4, np.abs(P) / np.maximum(S, 1e-9), 1.0))

    meta = pd.read_csv(d / "meters.csv").set_index("meter_id").loc[meters].reset_index()
    if "pv_source" in meta:
        pv_source = meta.pv_source.fillna("").astype(str).values
    else:
        pv_source = np.where(meta.has_pv, "inverter_optin", "")
    if site.pv_source == "estimated":
        pv_source = np.where(pv_source != "", "estimated", "")
    m["pv_power_kw"] = np.where((pv_source == "inverter_optin")[None, :], m["pv_power_kw"], np.nan)

    # profile reads the head-end knows it failed to collect
    collected = np.ones((T, H), bool)
    if (d / "collection_log.csv").exists():
        cl = pd.read_csv(d / "collection_log.csv", parse_dates=["batch_start", "batch_end"])
        col = {mid: j for j, mid in enumerate(meters)}
        for r in cl.itertuples():
            a, b = idx.searchsorted(r.batch_start), idx.searchsorted(r.batch_end)
            collected[a:b, col[r.meter_id]] = False
    alarms = pd.DataFrame(columns=["t", "meter", "alarm"])
    if (d / "alarms.csv").exists():
        al = pd.read_csv(d / "alarms.csv", parse_dates=["timestamp"])
        col = {mid: j for j, mid in enumerate(meters)}
        alarms = pd.DataFrame({"t": idx.searchsorted(al.timestamp), "meter": al.meter_id.map(col),
                               "alarm": al.alarm})

    h = None
    if site.has_head:
        hd = pd.read_parquet(d / "feeder_head.parquet")
        h = {}
        for c in HEAD_CHANNELS:
            w = hd.pivot_table(index="timestamp", columns=["feeder", "phase"], values=c, observed=True,
                               dropna=False, aggfunc="first")
            h[c] = w.to_numpy(np.float64).reshape(T, F, 3)
    margin = np.full(H, np.inf)
    if site.phase_source == "inferred":
        # learn the map over a quiet window at the start of the data (days 3 to 22)
        win = slice(2 * 96, (2 + site.phase_learn_days) * 96)
        phase, margin, _ = phase_id.infer(m["voltage_v"], m["active_power_kw"], feeder,
                                          h["voltage_v"] if h else None,
                                          h["active_power_kw"] if h else None, win)
    known = margin >= site.phase_unclear_margin
    if h is None:
        h = pseudo_head(m, feeder, np.where(known, phase, -1), F)
    tx = pd.read_parquet(d / "transformer.parquet", columns=["timestamp", "load_pct", "top_oil_temp_c"])
    if not site.transformer_monitor:
        tx["top_oil_temp_c"] = np.nan          # no oil sensor: only the overload estimate runs
    return Data(
        idx=idx, meters=meters, feeder=feeder, phase=phase,
        pole=tm.pole.values, m=m, h=h, tx=tx,
        ctx=pd.read_parquet(d / "context.parquet").set_index("timestamp"),
        split=split, meta=meta, site=site, alarms=alarms, collected=collected, pv_source=pv_source,
        phase_known=known, phase_margin=margin,
        kva=float(topo.get("transformer_kva", 100.0)),
    )


def pseudo_head(m: dict, feeder: np.ndarray, phase: np.ndarray, F: int) -> dict:
    """No substation reading (head_meter = none): stand-in feeder-phase readings built from the homes
    on that phase. Voltages are the median home (robust to one lying meter), powers are sums.
    There is no frequency, and the energy balance cannot run."""
    T = m["voltage_v"].shape[0]
    h = {c: np.full((T, F, 3), np.nan) for c in HEAD_CHANNELS}
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        for f in range(F):
            for p in range(3):
                sel = (feeder == f) & (phase == p)
                if not sel.any():
                    continue
                h["voltage_v"][:, f, p] = np.nanmedian(m["voltage_v"][:, sel], axis=1)
                h["voltage_min_v"][:, f, p] = np.nanmedian(m["voltage_min_v"][:, sel], axis=1)
                h["current_a"][:, f, p] = np.nansum(m["current_a"][:, sel], axis=1)
                h["active_power_kw"][:, f, p] = np.nansum(m["active_power_kw"][:, sel], axis=1)
                h["reactive_power_kvar"][:, f, p] = np.nansum(m["reactive_power_kvar"][:, sel], axis=1)
    return h


def fill_power_from_registers(P: np.ndarray, e_imp: np.ndarray, e_exp: np.ndarray) -> np.ndarray:
    """Missing 15-minute power, rebuilt from the energy registers: the first reading after a gap
    carries the energy used during it, which is spread evenly over the missing intervals."""
    out = P.copy()
    T, H = P.shape
    for j in range(H):
        miss = np.isnan(P[:, j])
        if not miss.any():
            continue
        ok = np.where(~miss & np.isfinite(e_imp[:, j]) & np.isfinite(e_exp[:, j]))[0]
        if len(ok) < 2:
            continue
        gaps = np.where(np.diff(ok) > 1)[0]
        for g in gaps:
            a, b = ok[g], ok[g + 1]
            de = (e_imp[b, j] - e_imp[a, j]) - (e_exp[b, j] - e_exp[a, j])
            n = b - a - 1
            out[a + 1:b, j] = (de - 0.25 * P[b, j]) / (0.25 * n)
    return out


def _lstsq_trimmed(X: np.ndarray, y: np.ndarray, passes: int = 3) -> tuple[np.ndarray, float]:
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    coef = np.zeros(X.shape[1])
    for _ in range(passes):
        coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        r = y - X @ coef
        mad = np.nanmedian(np.abs(r[ok] - np.nanmedian(r[ok]))) * 1.4826
        ok = ok & (np.abs(r) < 4 * mad + 1e-9)
    return coef, float(mad)


@dataclass
class Residuals:
    r_v: np.ndarray          # (T, H) voltage minus physics-expected voltage (V)
    sigma_v: np.ndarray      # (H,)
    r_id: np.ndarray         # (T, H) power identity residual, relative
    r_reg: np.ndarray        # (T, H) register increment minus integrated power (kWh)
    bal_groups: list         # energy balance per feeder-phase (or feeder): asset, meters, r (T,) kW, sigma
    P_fill: np.ndarray       # (T, H) reported power with gaps rebuilt from the energy registers
    head_dv: np.ndarray      # (T, F, 3) head voltage deviation from its 1-day rolling median (pu)
    p_signed_i: np.ndarray   # (T, H) signed current (A), + import
    oil_resid: np.ndarray    # (T,) transformer top-oil residual (deg C)
    models: dict


def residuals(D: Data) -> Residuals:
    """Physics-based residuals. Models are fitted on the train split only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # empty slices (lost batches) give NaN
        return _residuals(D)


def _residuals(D: Data) -> Residuals:
    m, h = D.m, D.h
    train = D.split == "train"
    V, I, P, Q, PF = (m[c] for c in ("voltage_v", "current_a", "active_power_kw", "reactive_power_kvar",
                                     "power_factor"))
    Is = I * np.sign(np.where(P == 0, 1, P))
    known = D.phase_known if D.phase_known is not None else np.ones(D.H, bool)
    ph = np.where(known, D.phase, 0)
    head_p = h["active_power_kw"]
    head_i_signed = h["current_a"] * np.sign(np.where(head_p == 0, 1, head_p))
    # a home whose phase is not known yet is compared with its feeder as a whole
    head_v = np.where(known, h["voltage_v"][:, D.feeder, ph], np.nanmean(h["voltage_v"], axis=2)[:, D.feeder])
    head_i = np.where(known, head_i_signed[:, D.feeder, ph], np.nansum(head_i_signed, axis=2)[:, D.feeder])
    P_fill = fill_power_from_registers(P, m["energy_import_kwh"], m["energy_export_kwh"])

    # 1) voltage profile model per meter: V = a + b * V_head + c * I_own + d * I_head(phase)
    r_v = np.full_like(V, np.nan)
    sig = np.zeros(D.H)
    coefs = {}
    for j in range(D.H):
        X = np.column_stack([np.ones(D.T), head_v[:, j], Is[:, j], head_i[:, j]])
        valid = (V[:, j] > 50) & (head_v[:, j] > 50)
        y = np.where(valid, V[:, j], np.nan)
        Xt = np.where(train[:, None], X, np.nan)
        coef, s = _lstsq_trimmed(Xt, np.where(train, y, np.nan))
        r_v[:, j] = y - X @ coef
        sig[j] = s
        coefs[D.meters[j]] = coef.tolist()

    # 2) power identity: |P| = V * I * PF
    s_kva = V * I / 1000
    r_id = (np.abs(P) - s_kva * PF) / np.maximum(s_kva, 0.05)

    # 3) register consistency
    dE = np.diff(m["energy_import_kwh"], axis=0, prepend=np.nan)
    r_reg = dE - np.clip(P, 0, None) * 0.25
    prev_nan = np.isnan(np.roll(V, 1, axis=0))
    r_reg[prev_nan] = np.nan                                   # catch-up after a gap is legitimate

    # 4) energy balance: head P = sum(meter P) + k * I_head^2, per feeder-phase, or per feeder when
    # a home's phase is not known yet. Missing readings are rebuilt from the energy registers.
    F = h["voltage_v"].shape[1]
    bal_groups, kbal = [], {}
    for f in range(F):
        on_f = D.feeder == f
        if known[on_f].all():
            groups = [(f"F{f + 1}-{'ABC'[p]}", (p,), on_f & (D.phase == p)) for p in range(3)]
        else:
            groups = [(f"F{f + 1}", (0, 1, 2), on_f)]
        for asset, phases, sel in groups:
            tot = P_fill[:, sel].sum(1)                        # NaN only if a reading is still missing
            y = head_p[:, f, list(phases)].sum(1) - tot
            X = np.column_stack([np.ones(D.T), (h["current_a"][:, f, list(phases)] ** 2).sum(1)])
            rebuilt = (np.isnan(P[:, sel]) & np.isfinite(P_fill[:, sel])).any(1)
            fit = train & ~rebuilt
            coef, s = _lstsq_trimmed(np.where(fit[:, None], X, np.nan), np.where(fit, y, np.nan))
            r = y - X @ coef
            # a rebuilt reading is right in total over its gap but flat inside it, so over a stretch
            # with rebuilt readings only the stretch's mean gap is known
            blk = np.cumsum(np.r_[True, rebuilt[1:] != rebuilt[:-1]])
            mean_blk = pd.Series(r).groupby(blk).transform("mean").to_numpy()
            bal_groups.append({"asset": asset, "feeder": f, "phases": phases, "meters": np.where(sel)[0],
                               "r": np.where(rebuilt, mean_blk, r), "sigma": s})
            kbal[asset] = coef.tolist()

    # 5) head voltage step vs its own recent past
    hv = h["voltage_v"].reshape(D.T, -1)
    med = pd.DataFrame(hv).rolling(96, min_periods=24).median().shift(1).to_numpy()
    head_dv = ((hv - med) / med).reshape(D.T, F, 3)

    # 6) transformer top-oil, only with a retrofit monitor: oil - ambient = a + b * (12 h mean of load^2)
    amb = D.ctx["ambient_temp_c"].reindex(D.idx).to_numpy()
    oil_resid = np.full(D.T, np.nan)
    coef = np.full(2, np.nan)
    if np.isfinite(D.tx["top_oil_temp_c"].to_numpy()).any():
        lp = (D.tx["load_pct"].to_numpy() / 100) ** 2
        lp12 = pd.Series(lp).ewm(span=48).mean().to_numpy()
        y = D.tx["top_oil_temp_c"].to_numpy() - amb
        X = np.column_stack([np.ones(D.T), lp12])
        coef, _ = _lstsq_trimmed(np.where(train[:, None], X, np.nan), np.where(train, y, np.nan))
        oil_resid = y - X @ coef

    return Residuals(r_v=r_v, sigma_v=sig, r_id=r_id, r_reg=r_reg, bal_groups=bal_groups, P_fill=P_fill,
                     head_dv=head_dv, p_signed_i=Is, oil_resid=oil_resid,
                     models={"voltage_profile": coefs, "balance": kbal, "oil": coef.tolist()})
