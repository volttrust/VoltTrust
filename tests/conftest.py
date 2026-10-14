import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volttrust.sim.config import SimConfig  # noqa: E402
from volttrust.sim.simulate import simulate  # noqa: E402


@pytest.fixture(scope="session")
def small_sim():
    cfg = SimConfig(seed=7, start="2025-06-01 00:00", end="2025-08-31 23:45",
                    train_end="2025-06-30 23:45", val_end="2025-07-31 23:45")
    return simulate(cfg, verbose=False)
