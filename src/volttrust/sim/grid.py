"""LV network: topology, vectorized power flow, feeder-head meters and transformer thermal model.

Each phase of each radial feeder is solved as a per-phase equivalent with a backward/forward
sweep, for every time step at once. This is exact for radial networks (same physics as
pandapower's radial solvers); `tests/test_powerflow.py` cross-checks a snapshot against pandapower.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASES = ("A", "B", "C")


@dataclass
class Topology:
    feeder: np.ndarray        # (H,) feeder index per home
    pole: np.ndarray          # (H,) pole index 1..K per home
    phase: np.ndarray         # (H,) phase index 0..2 per home
    n_feeders: int
    n_poles: int
    z_seg: complex            # per span, per-phase equivalent (ohm)
    z_drop: complex           # service drop (ohm)
    z_tx: complex             # transformer, referred to LV, per phase (ohm)
    r_seg: float
    v_src_nom: float          # no-load phase voltage at the busbar (V)

    def to_dict(self, meter_ids) -> dict:
        return {
            "feeders": self.n_feeders, "poles_per_feeder": self.n_poles,
            "segment_impedance_ohm": [self.z_seg.real, self.z_seg.imag],
            "service_drop_impedance_ohm": [self.z_drop.real, self.z_drop.imag],
            "transformer_impedance_ohm": [self.z_tx.real, self.z_tx.imag],
            "busbar_no_load_phase_voltage_v": self.v_src_nom,
            "meters": [
                {"meter_id": m, "feeder": f"F{f + 1}", "pole": int(p), "phase": PHASES[ph]}
                for m, f, p, ph in zip(meter_ids, self.feeder, self.pole, self.phase)
            ],
        }


def build_topology(cfg, rng: np.random.Generator) -> Topology:
    F, K, per = cfg.n_feeders, cfg.poles_per_feeder, cfg.homes_per_pole
    feeder, pole, phase = [], [], []
    for f in range(F):
        # balanced-ish phase assignment with some local imbalance, as in real LV networks
        ph = np.tile([0, 1, 2], K * per // 3 + 1)[:K * per]
        ph = ph[rng.permutation(len(ph))]
        for k in range(K):
            for j in range(per):
                feeder.append(f)
                pole.append(k + 1)
                phase.append(int(ph[k * per + j]))
    r = (cfg.r_phase_ohm_km + cfg.neutral_return_share * cfg.r_neutral_ohm_km) * cfg.span_km
    x = cfg.x_ohm_km * cfg.span_km
    rd = 2 * cfg.r_drop_ohm_km * cfg.drop_km
    z_base = 0.4 ** 2 / (cfg.transformer_kva / 1000)
    z = cfg.transformer_z_pct / 100 * z_base
    r_tx = z / np.sqrt(1 + cfg.transformer_xr ** 2)
    return Topology(
        feeder=np.array(feeder), pole=np.array(pole), phase=np.array(phase),
        n_feeders=F, n_poles=K,
        z_seg=complex(r, x), z_drop=complex(rd, 0.002), z_tx=complex(r_tx, r_tx * cfg.transformer_xr),
        r_seg=r, v_src_nom=cfg.nominal_v * cfg.tap_pu,
    )


def power_flow(topo: Topology, p_kw: np.ndarray, q_kvar: np.ndarray, v_src: np.ndarray,
               extra_r: np.ndarray | None = None, energized: np.ndarray | None = None,
               n_iter: int = 8) -> dict:
    """Solve all time steps.

    p_kw, q_kvar: (T, H) net consumption per home (negative = export).
    v_src: (T,) busbar no-load phase voltage (V); MV events act here.
    extra_r: (T, F, K, 3) added series resistance per span and phase (high-resistance joints).
    energized: (T, F, 3) bool; False = feeder phase de-energized (outage, blown fuse).
    """
    T, H = p_kw.shape
    F, K = topo.n_feeders, topo.n_poles
    if energized is None:
        energized = np.ones((T, F, 3), bool)
    home_on = energized[:, topo.feeder, topo.phase]                       # (T, H)
    s = (p_kw + 1j * q_kvar) * 1000 * home_on
    z_seg = np.full((T, F, K, 3), topo.z_seg, dtype=complex)
    if extra_r is not None:
        z_seg = z_seg + extra_r

    v_home = np.repeat(v_src[:, None], H, axis=1).astype(complex)
    for _ in range(n_iter):
        i_home = np.conj(s / np.where(home_on, v_home, 1.0)) * home_on
        inj = np.zeros((T, F, K, 3), complex)
        np.add.at(inj, (slice(None), topo.feeder, topo.pole - 1, topo.phase), i_home)
        i_seg = np.flip(np.cumsum(np.flip(inj, axis=2), axis=2), axis=2)  # current through span k
        i_tx = i_seg[:, :, 0, :].sum(axis=1)                               # (T, 3)
        v_bus = v_src[:, None] - topo.z_tx * i_tx                          # (T, 3)
        v_pole = v_bus[:, None, None, :] - np.cumsum(z_seg * i_seg, axis=2)  # (T, F, K, 3)
        v_home = v_pole[:, topo.feeder, topo.pole - 1, topo.phase] - topo.z_drop * i_home

    i_home = np.conj(s / np.where(home_on, v_home, 1.0)) * home_on
    v_home = v_home * home_on
    i_head = i_seg[:, :, 0, :]                                             # (T, F, 3)
    v_head = np.repeat(v_bus[:, None, :], F, axis=1) * energized
    s_head = v_head * np.conj(i_head) / 1000                               # kVA
    loss_kw = ((np.abs(i_seg) ** 2 * z_seg.real).sum(axis=(1, 2, 3))
               + (np.abs(i_home) ** 2).sum(axis=1) * topo.z_drop.real) / 1000
    s_tx = (v_bus * np.conj(i_tx)).sum(axis=1) / 1000
    return {
        "v_home": np.abs(v_home), "i_home": np.abs(i_home),
        "p_home": (v_home * np.conj(i_home)).real / 1000, "q_home": (v_home * np.conj(i_home)).imag / 1000,
        "v_head": np.abs(v_head), "i_head": np.abs(i_head),
        "p_head": s_head.real, "q_head": s_head.imag,
        "loss_kw": loss_kw, "s_tx_kva": np.abs(s_tx), "p_tx_kw": s_tx.real,
    }


def top_oil_temperature(load_pu: np.ndarray, ambient: np.ndarray, dt_min: float,
                        dtheta_rated: float | np.ndarray = 55.0, R: float = 6.0, n: float = 0.8,
                        tau_min: float = 180.0) -> np.ndarray:
    """IEC 60076-7 exponential top-oil model (ONAN), first-order discrete form."""
    dtheta_rated = np.broadcast_to(dtheta_rated, load_pu.shape)
    ult = dtheta_rated * ((load_pu ** 2 * R + 1) / (R + 1)) ** n
    out = np.empty_like(load_pu)
    rise = ult[0]
    a = dt_min / tau_min
    for t in range(len(load_pu)):
        rise += a * (ult[t] - rise)
        out[t] = ambient[t] + rise
    return out
