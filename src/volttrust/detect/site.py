"""What a neighborhood's equipment provides. The detector runs with any combination
and gets better as equipment is added.

    head_meter           substation reading per phase: "concentrator" (its built-in metering),
                         "retrofit" (our three-phase CT meter on the LV board) or "none"
    phase_source         which phase each home is on: "steg_map" (STEG's map) or "inferred" from
                         voltage + power changes over a quiet learning window (detect/phase.py);
                         homes whose margin is under `phase_unclear_margin` get a technician check
                         and are compared per feeder until their phase is known
    transformer_monitor  retrofit oil-temperature monitor fitted (cooling-fault check)
    pv_source            "as_data": each solar home as recorded (`inverter_optin` homes share their
                         inverter reading, the others are estimated from the net meter);
                         "estimated": ignore every inverter reading
    use_reference_panel  use the neighborhood's measured reference panel when the data has one;
                         otherwise production is estimated from hourly city weather
    batch_hours          the concentrator reads load profiles every N hours; alarms arrive at once
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

HEAD_METERS = ("concentrator", "retrofit", "none")


@dataclass
class Site:
    head_meter: str = "concentrator"
    phase_source: str = "steg_map"
    phase_unclear_margin: float = 0.10
    phase_learn_days: int = 20
    transformer_monitor: bool = False
    pv_source: str = "as_data"
    use_reference_panel: bool = True
    batch_hours: int = 12

    def __post_init__(self):
        assert self.head_meter in HEAD_METERS, self.head_meter
        assert self.phase_source in ("steg_map", "inferred"), self.phase_source
        assert self.pv_source in ("as_data", "estimated"), self.pv_source

    @property
    def has_head(self) -> bool:
        return self.head_meter != "none"

    def to_dict(self) -> dict:
        return asdict(self)
