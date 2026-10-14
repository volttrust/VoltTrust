"""Escalation router: decides the action tier for a flagged event and logs it.

Rules first (hard triggers), then calibrated-confidence bands.
Decisions go to an append-only, hash-chained SQLite log. Every row records the policy,
detector and confidence-table versions, the data window the verdict was made on, and a
reference (path + SHA-256) to a snapshot of the raw readings in that window.
Snapshots live outside the chained table, so deleting one after the 1-year retention
never breaks the chain. Households appear by meter number only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

TIERS = {
    "T0": "log only",
    "T1": "auto-act, logged",
    "T2": "auto-act + notify, undoable",
    "T3": "human approval required",
    "T4": "emergency alert (protection acts)",
}


# meter numbers and network assets only (M037, F2, F2-A, TX); never names or addresses
ASSET_ID = re.compile(r"^(M\d+|F\d+(-[ABC])?|TX\d*)$")


class SnapshotStore:
    """Raw readings behind each decision, one gzip file per event, kept outside the log.
    The log keeps only the reference and the SHA-256 of the stored bytes, so a snapshot can
    be checked against the log, and deleting it (retention) leaves the chain intact."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, event_id: str, data: bytes) -> dict:
        blob = gzip.compress(data, mtime=0)
        path = self.root / f"{event_id}.csv.gz"
        path.write_bytes(blob)
        return {"id": event_id, "path": str(path), "sha256": hashlib.sha256(blob).hexdigest(),
                "bytes": len(blob), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def load(self, ref: dict) -> bytes:
        """Raw bytes, refused if they no longer match the hash recorded in the log."""
        blob = Path(ref["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != ref["sha256"]:
            raise ValueError(f"snapshot {ref['id']} does not match its logged SHA-256")
        return gzip.decompress(blob)


@dataclass
class Event:
    event_id: str
    category: str            # Goal 1 verdict: "grid_event" | "equipment_fault" | "data_quality"
    subtype: str             # e.g. bias, drift, stuck, falsified, theft, sag, outage
    confidence: float        # Goal 2 final confidence P(anomaly) x P(class), calibrated
    uncertain: bool = False  # Goal 2 conformal set holds more than one class
    novelty: float = 0.0     # 0..1
    proposed_action: str = "flag_reading"
    customers_affected: int = 1
    kw_affected: float = 0.0
    detectors_agree: bool = True
    correlated_meters: int = 1
    reasons: list[str] = field(default_factory=list)   # plain-language SHAP reasons

    @classmethod
    def from_verdict(cls, rec: dict, meters_on_feeder: int = 50, **kw) -> "Event":
        """Build from the Goal 1/Goal 2 verdict JSON record."""
        ev = rec.get("evidence", {})
        return cls(
            event_id=rec["event_id"], category=rec["verdict"], subtype=rec.get("subtype", ""),
            confidence=rec["confidence"], uncertain=rec.get("uncertain", False),
            correlated_meters=max(len(rec.get("assets", [])),
                                  round(ev.get("neighbors_affected", 0) * meters_on_feeder)),
            **kw,
        )


@dataclass
class Decision:
    event_id: str
    tier: str
    rule: str
    policy_version: str
    latency_ms: float


class Router:
    def __init__(self, policy_path: str | Path, db_path: str | Path = ":memory:",
                 versions: dict | None = None):
        """versions: {"calibration_version", "detector_version"} recorded on every row
        (see volttrust.provenance)."""
        self.policy = yaml.safe_load(Path(policy_path).read_text())
        v = versions or {}
        self.versions = {"calibration_version": v.get("calibration_version"),
                         "detector_version": v.get("detector_version")}
        self.db = sqlite3.connect(str(db_path))
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, event_id TEXT, actor TEXT, tier TEXT, rule TEXT,
                policy_version TEXT, payload TEXT, prev_hash TEXT, hash TEXT)"""
        )
        self._auto_times: deque[float] = deque()

    # ---- routing -------------------------------------------------------
    def _hard_trigger(self, e: Event) -> tuple[str, str] | None:
        h = self.policy["hard_triggers"]
        if e.subtype in h["safety_classes"]:
            return "T4", "safety event"
        if self.policy.get("safe_mode"):
            return "T3", "safe mode on"
        if e.proposed_action in h["physical_actions"]:
            return "T3", "physical grid action"
        if e.subtype in h["accusatory_classes"]:
            return "T3", "verdict implies wrongdoing"
        if e.subtype in h.get("always_human_subtypes", []):
            return "T3", "always reviewed by a person"
        if e.uncertain:
            return "T3", "conformal set uncertain"
        if e.novelty >= h["novelty_threshold"]:
            return "T3", "novel pattern"
        if not e.detectors_agree:
            return "T3", "detectors disagree"
        if e.correlated_meters >= h["correlated_meters"]:
            return "T3", "correlated anomalies"
        if e.customers_affected > h["max_auto_customers"] or e.kw_affected > h["max_auto_kw"]:
            return "T3", "impact above auto limit"
        if self._breaker_tripped():
            return "T3", "automation circuit breaker"
        return None

    def _confidence_tier(self, e: Event) -> tuple[str, str]:
        bands = self.policy["confidence_bands"][e.category]
        for tier in ("T1", "T2", "T3"):
            if tier in bands and e.confidence >= bands[tier]:
                return tier, f"confidence {e.confidence:.2f} >= {bands[tier]}"
        return "T0", f"confidence {e.confidence:.2f} below review band"

    def _breaker_tripped(self) -> bool:
        cb = self.policy["circuit_breaker"]
        now = time.time()
        while self._auto_times and now - self._auto_times[0] > cb["window_minutes"] * 60:
            self._auto_times.popleft()
        return len(self._auto_times) >= cb["max_auto_actions"]

    def route(self, e: Event, data_window: dict | None = None,
              snapshot_ref: dict | None = None) -> Decision:
        """data_window: {"assets", "start", "end", ...}; snapshot_ref: from SnapshotStore.save."""
        t0 = time.perf_counter()
        hard = self._hard_trigger(e)
        action = e.proposed_action
        if hard:
            tier, rule = hard
        elif e.subtype in self.policy.get("auto_notify_subtypes", []):
            tier, rule = "T2", "dust: owner notified automatically"
            action = self.policy.get("auto_notify_action", action)
        else:
            tier, rule = self._confidence_tier(e)
        if tier == "T1" and e.subtype in self.policy.get("never_silent_subtypes", []):
            tier, rule = "T2", rule + "; never silent"
        if tier in ("T1", "T2"):
            self._auto_times.append(time.time())
        d = Decision(e.event_id, tier, rule, self.policy["version"],
                     (time.perf_counter() - t0) * 1000)
        self.log(e.event_id, "ai", tier, rule, {"event": asdict(e), "action": action, "latency_ms": d.latency_ms},
                 data_window=data_window, snapshot_ref=snapshot_ref)
        return d

    def last_decision(self, event_id: str, actor: str | None = None) -> dict | None:
        q = "SELECT ts,actor,tier,rule,payload FROM decision_log WHERE event_id=?"
        args = [event_id]
        if actor:
            q, args = q + " AND actor=?", args + [actor]
        row = self.db.execute(q + " ORDER BY id DESC LIMIT 1", args).fetchone()
        if not row:
            return None
        return dict(zip(("ts", "actor", "tier", "rule"), row[:4]), payload=json.loads(row[4]))

    def history(self, event_id: str) -> list[dict]:
        rows = self.db.execute("SELECT ts,actor,tier,rule,payload FROM decision_log"
                               " WHERE event_id=? ORDER BY id", (event_id,)).fetchall()
        return [dict(zip(("ts", "actor", "tier", "rule"), r[:4]), payload=json.loads(r[4])) for r in rows]

    # ---- operator actions ---------------------------------------------
    def operator_decision(self, event_id: str, operator: str, choice: str, reason: str = "") -> None:
        """choice: approve | override | undo | escalate | more_data"""
        if choice in ("override", "undo") and not reason:
            raise ValueError("override/undo requires a reason")
        # the operator decided on the same readings the AI routed on
        ai = self.last_decision(event_id, "ai")
        p = ai["payload"] if ai else {}
        self.log(event_id, f"operator:{operator}", "-", choice, {"reason": reason},
                 data_window=p.get("data_window"), snapshot_ref=p.get("snapshot_ref"))

    # ---- tamper-evident log -------------------------------------------
    def log(self, event_id, actor, tier, rule, payload, data_window=None, snapshot_ref=None) -> None:
        w = data_window or {}
        for a in [*w.get("assets", []), *([w["alert_asset"]] if w.get("alert_asset") else [])]:
            if not ASSET_ID.match(str(a)):
                raise ValueError(f"log keeps meter numbers only, got asset id {a!r}")
        payload = {**payload, **self.versions, "data_window": data_window,
                   "snapshot_ref": snapshot_ref}
        row = self.db.execute("SELECT hash FROM decision_log ORDER BY id DESC LIMIT 1").fetchone()
        prev = row[0] if row else "GENESIS"
        ts = time.time()
        body = json.dumps([ts, event_id, actor, tier, rule, self.policy["version"], payload, prev],
                          sort_keys=True, default=str)
        h = hashlib.sha256(body.encode()).hexdigest()
        self.db.execute(
            "INSERT INTO decision_log (ts,event_id,actor,tier,rule,policy_version,payload,prev_hash,hash)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, event_id, actor, tier, rule, self.policy["version"],
             json.dumps(payload, sort_keys=True, default=str), prev, h),
        )
        self.db.commit()

    def verify_log(self) -> bool:
        prev = "GENESIS"
        for ts, eid, actor, tier, rule, ver, payload, p, h in self.db.execute(
            "SELECT ts,event_id,actor,tier,rule,policy_version,payload,prev_hash,hash"
            " FROM decision_log ORDER BY id"
        ):
            body = json.dumps([ts, eid, actor, tier, rule, ver, json.loads(payload), p],
                              sort_keys=True, default=str)
            if p != prev or hashlib.sha256(body.encode()).hexdigest() != h:
                return False
            prev = h
        return True


if __name__ == "__main__":
    r = Router(Path(__file__).with_name("policy.yaml"))
    demo = [
        Event("e1", "data_quality", "sensor_drift", 0.95, reasons=["voltage 4% above neighbours for 6 h"]),
        Event("e2", "equipment_fault", "transformer_overheat", 0.78),
        Event("e3", "data_quality", "theft", 0.97),
        Event("e4", "grid_event", "surge", 0.93),
        Event("e5", "grid_event", "outage", 0.88, proposed_action="isolate_section", customers_affected=40),
        Event("e6", "data_quality", "unknown", 0.60, novelty=0.91),
        Event("e7", "grid_event", "overvoltage_trip", 0.99),
        Event("e8", "data_quality", "bias", 0.62, uncertain=True),
        Event("e9", "data_quality", "noise", 0.20),
    ]
    for e in demo:
        d = r.route(e)
        print(f"{e.event_id} {e.category:12} {e.subtype:16} conf={e.confidence:.2f} -> "
              f"{d.tier} ({TIERS[d.tier]}): {d.rule}  [{d.latency_ms:.3f} ms]")
    r.operator_decision("e5", "op1", "approve")
    r.operator_decision("e3", "op1", "override", "meter replaced yesterday, not tampering")
    print("log intact:", r.verify_log())
