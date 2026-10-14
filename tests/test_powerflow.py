"""Cross-check the vectorized feeder power flow against pandapower on one snapshot."""
import numpy as np
import pytest

from volttrust.sim import grid
from volttrust.sim.config import SimConfig


def test_sweep_matches_pandapower():
    pp = pytest.importorskip("pandapower")
    cfg = SimConfig()
    rng = np.random.default_rng(0)
    topo = grid.build_topology(cfg, rng)
    H = len(topo.feeder)
    # phase A only, so the per-phase equivalent maps onto a balanced pandapower model
    on = topo.phase == 0
    p = np.where(on, rng.uniform(0.2, 3.0, H), 0.0)
    q = 0.3 * p
    v_src = np.array([topo.v_src_nom])
    r = grid.power_flow(topo, p[None], q[None], v_src)

    net = pp.create_empty_network()
    vn = 0.4
    src = pp.create_bus(net, vn)
    pp.create_ext_grid(net, src, vm_pu=topo.v_src_nom / (vn * 1000 / np.sqrt(3)))
    bus = pp.create_bus(net, vn)
    pp.create_line_from_parameters(net, src, bus, 1.0, topo.z_tx.real, topo.z_tx.imag, 0, 1e3)
    home_bus = {}
    for f in range(topo.n_feeders):
        prev = bus
        poles = []
        for k in range(topo.n_poles):
            b = pp.create_bus(net, vn)
            pp.create_line_from_parameters(net, prev, b, 1.0, topo.z_seg.real, topo.z_seg.imag, 0, 1e3)
            poles.append(b)
            prev = b
        for h in np.where((topo.feeder == f) & on)[0]:
            hb = pp.create_bus(net, vn)
            pp.create_line_from_parameters(net, poles[topo.pole[h] - 1], hb, 1.0,
                                           topo.z_drop.real, topo.z_drop.imag, 0, 1e3)
            pp.create_load(net, hb, p_mw=3 * p[h] / 1000, q_mvar=3 * q[h] / 1000)
            home_bus[h] = hb
    pp.runpp(net)
    v_pp = np.array([net.res_bus.vm_pu.at[home_bus[h]] for h in sorted(home_bus)]) * vn * 1000 / np.sqrt(3)
    v_us = r["v_home"][0, sorted(home_bus)]
    assert np.max(np.abs(v_pp - v_us)) < 0.05      # volts
