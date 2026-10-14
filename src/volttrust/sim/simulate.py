"""Generate the VoltTrust Phase 1 dataset.

    python -m volttrust.sim.simulate --out data --seed 42

Pipeline: context -> household loads and PV -> physical events -> power flow -> true values
-> meter noise -> sensing faults -> energy registers -> reporting faults -> files + labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import anomalies as an
from . import context, grid, loads
from .config import SimConfig


def _split(idx: pd.DatetimeIndex, cfg: SimConfig) -> np.ndarray:
    return np.where(idx <= cfg.train_end, "train", np.where(idx <= cfg.val_end, "val", "test"))


def simulate(cfg: SimConfig, verbose: bool = True) -> dict:
    t0 = time.time()
    log = (lambda *a: print(f"[{time.time() - t0:5.1f}s]", *a)) if verbose else (lambda *a: None)
    rng = np.random.default_rng(cfg.seed)
    idx = context.time_index(cfg.start, cfg.end, cfg.freq_min)
    T = len(idx)
    ctx = context.build_context(idx, cfg.tz_offset_h, rng)
    log(f"context: {T} intervals")

    topo = grid.build_topology(cfg, rng)
    H, F = len(topo.feeder), topo.n_feeders
    homes = loads.make_homes(H, cfg, rng)
    names = list(cfg.meter_models)
    models = rng.choice(names, size=H, p=list(cfg.meter_models.values()))

    P_load, Q_load, P_pv, P_ac = (np.zeros((T, H)) for _ in range(4))
    for h, home in enumerate(homes):
        P_load[:, h], Q_load[:, h], P_pv[:, h], P_ac[:, h] = loads.home_load(home, idx, ctx, rng)
    log("household loads and PV")

    # ---------------------------------------------------------------- physical layer
    sch = an.Scheduler(idx, topo, cfg, rng)
    sch.meter_models = models
    sch.has_pv = np.array([h.has_pv for h in homes])
    # separate generator for what STEG's systems provide, so earlier draws keep their values
    rng2 = np.random.default_rng(cfg.seed + 7919)
    pv_homes = np.where(sch.has_pv)[0]
    sch.pv_optin = np.sort(rng2.choice(pv_homes, size=min(cfg.n_pv_optin, len(pv_homes)), replace=False))
    phys = an.PhysicalInputs(T, H, F, topo.n_poles)
    an.schedule_physical(sch, phys)

    # MV supply: slow random walk plus a dip when the wider grid is loaded in the evening
    walk = np.zeros(T)
    for t in range(1, T):
        walk[t] = 0.995 * walk[t - 1] + rng.normal(0, 0.0006)
    sys_load = pd.Series(P_load.sum(1)).rolling(8, min_periods=1).mean().values
    mv = 1 + walk - 0.015 * (sys_load - sys_load.mean()) / sys_load.std() / 3
    v_src = topo.v_src_nom * mv * phys.v_src_scale

    # inverters stop when their phase is de-energized (anti-islanding)
    home_on = phys.energized[:, topo.feeder, topo.phase]
    P_pv = P_pv * phys.pv_scale * home_on
    p_net = P_load * phys.load_scale + phys.extra_p - P_pv
    q_net = Q_load * phys.load_scale + phys.extra_q
    pf_res = grid.power_flow(topo, p_net, q_net, v_src, extra_r=phys.extra_r, energized=phys.energized)
    log("power flow solved")

    # ---------------------------------------------------------------- true meter values
    V = pf_res["v_home"]
    fluct = np.abs(rng.normal(0, 0.006, (T, H)))
    true = {
        "voltage_v": V,
        "voltage_min_v": V * (1 - fluct - phys.sag_depth[:, None]),
        "voltage_max_v": V * (1 + np.abs(rng.normal(0, 0.006, (T, H)))),
        "current_a": pf_res["i_home"],
        "active_power_kw": pf_res["p_home"],
        "reactive_power_kvar": pf_res["q_home"],
    }
    S = np.hypot(true["active_power_kw"], true["reactive_power_kvar"])
    true["power_factor"] = np.where(S > 1e-4, np.abs(true["active_power_kw"]) / np.maximum(S, 1e-9), 1.0)

    # ---------------------------------------------------------------- measurement layer
    noise = {"voltage_v": 0.001, "voltage_min_v": 0.001, "voltage_max_v": 0.001,
             "current_a": 0.003, "active_power_kw": 0.003, "reactive_power_kvar": 0.005}
    meas = {c: true[c] * (1 + rng.normal(0, sd, (T, H))) for c, sd in noise.items()}
    meas["_hour"] = idx.hour.values
    meas["_pf_frozen"] = np.zeros((T, H), bool)
    mev = an.schedule_measurement(sch)
    an.apply_sensing(mev, meas, rng)

    Sm = np.hypot(meas["active_power_kw"], meas["reactive_power_kvar"])
    pf_m = np.where(Sm > 1e-4, np.abs(meas["active_power_kw"]) / np.maximum(Sm, 1e-9), 1.0)
    meas["power_factor"] = np.where(meas["_pf_frozen"], true["power_factor"], pf_m)
    imp0 = rng.uniform(3000, 40000, H)
    exp0 = np.array([rng.uniform(300, 5000) if h.has_pv else 0.0 for h in homes])
    meas["energy_import_kwh"] = imp0 + np.cumsum(np.clip(meas["active_power_kw"], 0, None), 0) * 0.25
    meas["energy_export_kwh"] = exp0 + np.cumsum(np.clip(-meas["active_power_kw"], 0, None), 0) * 0.25
    true["energy_import_kwh"] = imp0 + np.cumsum(np.clip(true["active_power_kw"], 0, None), 0) * 0.25
    true["energy_export_kwh"] = exp0 + np.cumsum(np.clip(-true["active_power_kw"], 0, None), 0) * 0.25

    rep = {c: meas[c].copy() for c in an.REPORT}
    an.apply_reporting(mev, rep, rng)

    # inverter production reading (a separate device from the meter), solar homes only
    has_pv = np.array([h.has_pv for h in homes])
    true["pv_power_kw"] = np.where(has_pv, P_pv, np.nan)
    rep["pv_power_kw"] = np.where(has_pv, P_pv * (1 + rng.normal(0, 0.01, (T, H))), np.nan)
    an.apply_inverter(mev, rep)
    # STEG installs only a net meter: production readings exist only where the owner shares them
    optin = np.isin(np.arange(H), sch.pv_optin)
    rep["pv_power_kw"][:, ~optin] = np.nan
    log(f"{len(sch.labels.events)} events injected")

    # ---------------------------------------------------------------- what STEG's systems add
    alarms = _alarms(idx, true["voltage_v"], sch.labels.events, homes, cfg, rng2)
    lost, collection_log = _collection_loss(idx, H, cfg, rng2)
    for c in an.REPORT:
        rep[c] = np.where(lost, np.nan, rep[c])
    # a measured reference panel per neighborhood (kW per kWp), no faults
    if cfg.reference_panel_kwp > 0:
        g, ta = ctx["ghi_wm2"].values, ctx["ambient_temp_c"].values
        ref = g / 1000 * 0.82 * (1 - 0.004 * (ta + 0.03 * g - 25))
        ctx["ref_pv_kw_per_kwp"] = np.clip(ref, 0, None) * (1 + rng2.normal(0, 0.01, T))

    # ---------------------------------------------------------------- feeder heads, transformer
    freq = np.zeros(T)
    for t in range(1, T):
        freq[t] = 0.9 * freq[t - 1] + rng.normal(0, 0.006)
    freq = 50 + freq + phys.freq_offset
    nonlinear = (P_ac + 0.3 * P_load).sum(1) / cfg.transformer_kva
    head = {
        "voltage_v": pf_res["v_head"] * (1 + rng.normal(0, 0.0005, (T, F, 3))),
        "current_a": pf_res["i_head"] * (1 + rng.normal(0, 0.001, (T, F, 3))),
        "active_power_kw": pf_res["p_head"] * (1 + rng.normal(0, 0.001, (T, F, 3))),
        "reactive_power_kvar": pf_res["q_head"] * (1 + rng.normal(0, 0.002, (T, F, 3))),
    }
    head["voltage_min_v"] = head["voltage_v"] * (1 - np.abs(rng.normal(0, 0.003, (T, F, 3))) - phys.sag_depth[:, None, None])
    Sh = np.hypot(head["active_power_kw"], head["reactive_power_kvar"])
    head["power_factor"] = np.where(Sh > 1e-3, np.abs(head["active_power_kw"]) / np.maximum(Sh, 1e-9), 1.0)
    head["frequency_hz"] = np.repeat(np.repeat(freq[:, None, None], F, 1), 3, 2)
    thd = 1.2 + 6.0 * nonlinear + rng.normal(0, 0.08, T)
    head["thd_v_pct"] = np.repeat(np.repeat(thd[:, None, None], F, 1), 3, 2)
    energized = phys.energized
    for c in ("frequency_hz", "thd_v_pct"):
        head[c] = np.where(energized, head[c], np.nan)

    load_pu = pf_res["s_tx_kva"] / cfg.transformer_kva
    tx = pd.DataFrame({
        "timestamp": idx,
        "load_pct": load_pu * 100,
        "top_oil_temp_c": grid.top_oil_temperature(load_pu, ctx["ambient_temp_c"].values, cfg.freq_min,
                                                   dtheta_rated=55.0 * phys.cooling_factor)
                          + rng.normal(0, 0.2, T),
        "losses_kw": pf_res["loss_kw"],
    })
    log("feeder heads and transformer")

    return {"cfg": cfg, "idx": idx, "ctx": ctx, "topo": topo, "homes": homes, "models": models,
            "true": true, "rep": rep, "head": head, "tx": tx, "labels": sch.labels,
            "P_pv": P_pv, "pv_optin": sch.pv_optin, "alarms": alarms, "collection_log": collection_log,
            "collection_lost": lost}


def _alarms(idx, v_true, events, homes, cfg, rng) -> pd.DataFrame:
    """Events the meters push at once, outside the 12-hourly profile reads: power cut and restore
    (from the real supply), and tamper alarms (terminal cover or magnet)."""
    rows = []
    off = v_true < 1
    edge = np.diff(off.astype(np.int8), axis=0, prepend=0)
    for t, m in zip(*np.where(edge == 1)):
        rows.append((idx[t], m, "power_cut", ""))
    for t, m in zip(*np.where(edge == -1)):
        rows.append((idx[t], m, "power_restore", ""))
    for e in events:
        if e["anomaly_type"] == "theft_bypass" and rng.random() < cfg.p_theft_tamper_alarm:
            m = int(e["meters"][1:]) - 1
            rows.append((idx[e["start_idx"]], m, "tamper_cover", e["event_id"]))
    for _ in range(cfg.n_nuisance_tamper_alarms):
        rows.append((idx[int(rng.integers(len(idx)))], int(rng.integers(len(homes))),
                     str(rng.choice(["tamper_cover", "tamper_magnet"])), ""))
    al = pd.DataFrame(rows, columns=["timestamp", "m", "alarm", "event_id"])
    al["meter_id"] = [homes[m].meter_id for m in al.m]
    return al.drop(columns="m").sort_values("timestamp").reset_index(drop=True)[
        ["timestamp", "meter_id", "alarm", "event_id"]]


def _collection_loss(idx, H, cfg, rng) -> tuple[np.ndarray, pd.DataFrame]:
    """The concentrator reads each meter's load profile every `batch_hours`. A failed read loses that
    meter's whole batch; the head-end knows which reads failed (collection log)."""
    T = len(idx)
    per = cfg.batch_hours * 60 // cfg.freq_min
    nb = -(-T // per)
    p = rng.uniform(*cfg.collection_loss, H)
    fail = rng.random((nb, H)) < p[None, :]
    reason = np.where(fail, "read_failed", "").astype(object)
    for b in rng.choice(nb, size=cfg.n_concentrator_offline, replace=False):
        fail[b] = True
        reason[b] = "concentrator_offline"
    lost = np.repeat(fail, per, axis=0)[:T]
    rows = [(idx[b * per], idx[min((b + 1) * per, T) - 1] + pd.Timedelta(minutes=cfg.freq_min), f"M{m + 1:03d}", reason[b, m])
            for b, m in zip(*np.where(fail))]
    return lost, pd.DataFrame(rows, columns=["batch_start", "batch_end", "meter_id", "reason"])


# --------------------------------------------------------------------------------- writing
ROUND = {"voltage_v": 1, "voltage_min_v": 1, "voltage_max_v": 1, "current_a": 2,
         "active_power_kw": 3, "reactive_power_kvar": 3, "power_factor": 3,
         "energy_import_kwh": 3, "energy_export_kwh": 3, "frequency_hz": 3, "thd_v_pct": 2,
         "pv_power_kw": 3}


def _label_cols(lab: np.ndarray, events: pd.DataFrame) -> dict:
    flat = lab.ravel()
    has = flat >= 0
    out = {"is_anomaly": has.astype(np.int8)}
    for c in ("event_id", "anomaly_type", "family", "verdict"):
        vals = events[c].values
        col = np.where(has, vals[np.maximum(flat, 0)], "normal" if c != "event_id" else "")
        out[c] = pd.Categorical(col)
    return out


def write(sim: dict, out: str | Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, idx, topo = sim["cfg"], sim["idx"], sim["topo"]
    T, H = len(idx), len(topo.feeder)
    F = topo.n_feeders
    events = pd.DataFrame(sim["labels"].events)
    meter_ids = np.array([h.meter_id for h in sim["homes"]])
    split = _split(idx, cfg)

    base = {
        "timestamp": np.repeat(idx.values, H),
        "meter_id": pd.Categorical(np.tile(meter_ids, T)),
        "feeder": pd.Categorical(np.tile([f"F{f + 1}" for f in topo.feeder], T)),
        "phase": pd.Categorical(np.tile([grid.PHASES[p] for p in topo.phase], T)),
        "pole": np.tile(topo.pole, T).astype(np.int8),
    }
    obs = dict(base)
    for c in an.REPORT + ["pv_power_kw"]:
        v = np.round(sim["rep"][c], ROUND[c]).ravel()
        obs[c] = v if c.startswith("energy") else v.astype(np.float32)
    obs["split"] = pd.Categorical(np.repeat(split, H))
    obs.update(_label_cols(sim["labels"].meter, events))
    pd.DataFrame(obs).to_parquet(out / "meters.parquet", index=False)

    tru = dict(base)
    for c in an.REPORT + ["pv_power_kw"]:
        v = np.round(sim["true"][c], ROUND[c]).ravel()
        tru[c] = v if c.startswith("energy") else v.astype(np.float32)
    pd.DataFrame(tru).to_parquet(out / "meters_true.parquet", index=False)

    hd = {
        "timestamp": np.repeat(idx.values, F * 3),
        "feeder": pd.Categorical(np.tile(np.repeat([f"F{f + 1}" for f in range(F)], 3), T)),
        "phase": pd.Categorical(np.tile(list(grid.PHASES) * F, T)),
    }
    for c in ("voltage_v", "voltage_min_v", "current_a", "active_power_kw", "reactive_power_kvar",
              "power_factor", "frequency_hz", "thd_v_pct"):
        hd[c] = np.round(sim["head"][c], ROUND[c]).ravel().astype(np.float32)
    hd["split"] = pd.Categorical(np.repeat(split, F * 3))
    hd.update(_label_cols(sim["labels"].head, events))
    pd.DataFrame(hd).to_parquet(out / "feeder_head.parquet", index=False)

    tx = sim["tx"].copy()
    tx[["load_pct", "top_oil_temp_c", "losses_kw"]] = tx[["load_pct", "top_oil_temp_c", "losses_kw"]].round(3)
    tx["split"] = split
    tx = pd.concat([tx, pd.DataFrame(_label_cols(sim["labels"].tx, events))], axis=1)
    tx.to_parquet(out / "transformer.parquet", index=False)

    sim["alarms"].to_csv(out / "alarms.csv", index=False)
    sim["collection_log"].to_csv(out / "collection_log.csv", index=False)

    ctx = sim["ctx"].copy()
    ctx[["ambient_temp_c", "ghi_wm2"]] = ctx[["ambient_temp_c", "ghi_wm2"]].round(2)
    if "ref_pv_kw_per_kwp" in ctx:
        ctx["ref_pv_kw_per_kwp"] = ctx["ref_pv_kw_per_kwp"].round(4)
    ctx.index.name = "timestamp"
    ctx.reset_index().to_parquet(out / "context.parquet", index=False)

    events.drop(columns=["start_idx", "end_idx"]).to_csv(out / "events.csv", index=False)

    meta = pd.DataFrame([{
        "meter_id": h.meter_id, "feeder": f"F{topo.feeder[i] + 1}", "pole": int(topo.pole[i]),
        "phase": grid.PHASES[topo.phase[i]], "meter_model": sim["models"][i],
        "has_pv": h.has_pv, "pv_kwp": h.pv_kwp,
        "pv_source": ("inverter_optin" if i in sim["pv_optin"] else "estimated") if h.has_pv else "", "has_ac": h.has_ac, "has_electric_heater": h.has_heater,
        "has_electric_water_heater": h.has_water_heater, "stay_home_household": h.stay_home,
    } for i, h in enumerate(sim["homes"])])
    meta.to_csv(out / "meters.csv", index=False)

    topo_d = topo.to_dict(meter_ids)
    topo_d["transformer_kva"] = cfg.transformer_kva
    (out / "topology.json").write_text(json.dumps(topo_d, indent=2))
    cfg.to_json(out / "config.json")

    # small CSV sample for a quick look without parquet tools: one week in August, all meters
    m = pd.read_parquet(out / "meters.parquet")
    wk = m[(m.timestamp >= "2025-08-04") & (m.timestamp < "2025-08-11")]
    wk.to_csv(out / "sample_meters_week.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None, help="JSON config (see SimConfig)")
    a = ap.parse_args(argv)
    cfg = SimConfig.from_json(a.config) if a.config else SimConfig()
    if a.seed is not None:
        cfg.seed = a.seed
    sim = simulate(cfg)
    write(sim, a.out)
    print(f"wrote {a.out}/")


if __name__ == "__main__":
    main()
