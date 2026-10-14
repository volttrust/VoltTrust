"""Build and execute the Phase 1 notebooks (Kaggle-ready).

    python scripts/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = r'''
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Works from the repo (notebooks/) and on Kaggle (the VoltTrust dataset attached under /kaggle/input)
cands = [Path("../data"), Path("data")] + [Path(p).parent for p in glob.glob("/kaggle/input/**/data/events.csv", recursive=True)]
DATA = next(p for p in cands if (p / "events.csv").exists())
for p in [Path("../src"), Path("src")] + [Path(p).parent for p in glob.glob("/kaggle/input/**/src/volttrust", recursive=True)]:
    if p.exists(): sys.path.insert(0, str(p.resolve()))

# Reference categorical palette (fixed order), recessive grid
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "gray": "#8a8a85"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#b0afa8", "lines.linewidth": 1.6, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
print("data:", DATA)
'''


def nb1():
    c = []
    md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
    code = lambda s: c.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731
    md("""
# VoltTrust Tunisia: dataset exploration (Phase 1)

One year (2025) of a simulated smart-metered neighborhood near Tunis at 15-minute resolution: a 100 kVA
STEG-style MV/LV substation, two LV feeders and 48 homes, driven by the **real 2025 Tunis weather**, with
genuine grid events, equipment faults and data problems injected on purpose and labelled. How the data
and anomalies were generated: `docs/anomalies.md`.

This notebook checks that the **clean baseline is realistic and physically consistent**, compares it with
real data (STEG monthly sales, Moroccan feeder loads, the Tunisian household benchmark), then shows what
each injected anomaly looks like next to the true values.
""")
    code(SETUP)
    md("## 1. Files")
    code('''
meters = pd.read_parquet(DATA / "meters.parquet")
truth = pd.read_parquet(DATA / "meters_true.parquet")
head = pd.read_parquet(DATA / "feeder_head.parquet")
tx = pd.read_parquet(DATA / "transformer.parquet")
ctx = pd.read_parquet(DATA / "context.parquet").set_index("timestamp")
events = pd.read_csv(DATA / "events.csv", parse_dates=["start", "end"])
meta = pd.read_csv(DATA / "meters.csv")
for n, d in [("meters", meters), ("meters_true", truth), ("feeder_head", head), ("transformer", tx),
             ("context", ctx), ("events", events), ("meters (metadata)", meta)]:
    print(f"{n:18s} {d.shape}")
meters.head()
''')
    code('''
print("Anomalous meter rows by split:")
print(meters.groupby("split", observed=True).is_anomaly.mean().round(4))
pd.crosstab([events.family, events.anomaly_type], events.split)
''')
    md("## 2. Context: Tunis weather, Ramadan, holidays, PV")
    code('''
fig, ax = plt.subplots(figsize=(10, 3))
daily = ctx.ambient_temp_c.resample("D").agg(["min", "max", "mean"])
ax.fill_between(daily.index, daily["min"], daily["max"], color=C["blue"], alpha=0.18, lw=0, label="daily range")
ax.plot(daily.index, daily["mean"], color=C["blue"], label="daily mean")
ram = ctx[ctx.is_ramadan].index
ax.axvspan(ram.min(), ram.max(), color=C["gray"], alpha=0.15, lw=0)
ax.text(ram.min(), ax.get_ylim()[1] * 0.92, " Ramadan", color="#55554f")
hw = ctx.heatwave.resample("D").max().astype(bool)
runs = (hw != hw.shift()).cumsum()[hw]
for _, g in hw[hw].groupby(runs):
    ax.axvspan(g.index[0], g.index[-1] + pd.Timedelta(days=1), color=C["orange"], alpha=0.2, lw=0)
ax.text(hw[hw].index[0], ax.get_ylim()[1] * 0.92, " heatwaves", color="#55554f")
ax.set_title("Ambient temperature, Tunis 2025, real hourly data from Open-Meteo (°C)"); ax.legend(loc="lower center", ncol=2)
plt.show()
''')
    md("## 3. Household load: seasons and Ramadan")
    code('''
m = truth.merge(ctx[["is_weekend", "is_holiday", "is_ramadan"]], left_on="timestamp", right_index=True)
m["hour"] = m.timestamp.dt.hour + m.timestamp.dt.minute / 60
m["month"] = m.timestamp.dt.month
work = ~(m.is_weekend | m.is_holiday)
groups = {
    "Winter weekday (Jan)": C["blue"],
    "Summer weekday (Jul)": C["orange"],
    "Ramadan weekday": C["aqua"],
}
sel = {"Winter weekday (Jan)": work & (m.month == 1) & ~m.is_ramadan,
       "Summer weekday (Jul)": work & (m.month == 7),
       "Ramadan weekday": work & m.is_ramadan}
fig, ax = plt.subplots(figsize=(8, 3.2))
for k, col in groups.items():
    prof = m[sel[k]].groupby("hour").active_power_kw.mean()
    ax.plot(prof.index, prof.values, color=col, label=k)
ax.set_xlabel("hour of day"); ax.set_title("Average net power per home (kW, negative = PV export)")
ax.set_xticks(range(0, 25, 3)); ax.legend()
plt.show()
ann = truth.groupby("meter_id", observed=True).active_power_kw.apply(lambda s: s.clip(lower=0).sum() * 0.25)
print("Annual imported energy per home (kWh):", ann.describe()[["min", "50%", "max"]].round(0).to_dict())
''')
    md("""
The lunch and evening peaks, the summer air-conditioning load and the Ramadan shift (suhoor around
03:00, a dip at iftar, a late-evening peak) are all there. Midday PV export pulls the average down.
""")
    md("""
## 3b. Calibration against real data

The household model was calibrated so the clean baseline matches three public references
(`inputs/README.md`): the Tunisian household benchmark (about 1,450 kWh a year), STEG's national
monthly sales (2013 to 2021) and the weekday load curve of residential district feeders in Laayoune,
Morocco (10-minute data, 2022 to 2024).
""")
    code('''
INP = next(p for p in [Path("../inputs"), Path("inputs")] + [Path(p).parent for p in glob.glob("/kaggle/input/**/inputs/steg_monthly_index_2013_2021.csv", recursive=True)] if p.exists())
steg = pd.read_csv(INP / "steg_monthly_index_2013_2021.csv")
mor = pd.read_csv(INP / "morocco_laayoune_hourly_profile.csv")
# homes without PV: what the meter sees is the household's whole consumption
nopv = meta.loc[~meta.has_pv, "meter_id"]
cons = truth[truth.meter_id.isin(nopv)]
ann_c = cons.groupby("meter_id", observed=True).active_power_kw.sum() * 0.25
tot = cons.groupby("timestamp").active_power_kw.sum()
mo = tot.groupby(tot.index.month).mean(); mo = mo / mo.mean()
wd = tot[tot.index.dayofweek < 5]
hr = wd.groupby(wd.index.hour).mean(); hr = hr / hr.mean()

fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
axs[0].hist(ann_c, bins=12, color=C["blue"])
axs[0].axvline(1450, color="#55554f", ls="--", lw=1); axs[0].text(1450, axs[0].get_ylim()[1] * 0.9, " Tunisian benchmark", color="#55554f")
axs[0].set_title(f"Annual consumption, {len(nopv)} homes without PV (kWh)"); axs[0].set_yticks([])
axs[1].plot(steg.month, steg.steg_bt_index, color=C["gray"], marker="o", ms=3, label="STEG low voltage (billed)")
axs[1].plot(steg.month, steg.steg_mt_index, color=C["yellow"], marker="o", ms=3, label="STEG medium voltage")
axs[1].plot(mo.index, mo.values, color=C["blue"], marker="o", ms=3, label="simulated homes")
axs[1].set_xticks(range(1, 13)); axs[1].set_title("Month / yearly mean"); axs[1].legend(fontsize=7)
for z, col in zip(["laayoune_zone2", "laayoune_zone3", "laayoune_zone4"], [C["gray"]] * 3):
    axs[2].plot(mor.hour, mor[z], color=col, lw=1)
axs[2].plot([], [], color=C["gray"], label="Laayoune residential feeders")
axs[2].plot(hr.index, hr.values, color=C["blue"], label=f"simulated, {len(nopv)} homes")
axs[2].set_xticks(range(0, 25, 3)); axs[2].set_title("Weekday load curve / daily mean"); axs[2].legend(fontsize=7)
plt.tight_layout(); plt.show()
night = wd.between_time("02:00", "04:59").mean() / wd.mean()
print(f"median home {ann_c.median():.0f} kWh/yr | summer peak month {mo.max():.2f}x | night (02-05h) {night:.2f}x | evening peak {hr.max():.2f}x")
''')
    md("""
STEG's low-voltage sales are billed, so their summer peak shows up in September to November; the
medium-voltage series is metered monthly and peaks in July and August. The simulated months sit between
them. A district of thousands of customers is smoother than a few dozen homes, so the simulated daily curve
has a deeper night and a sharper evening peak than the Moroccan feeders, as expected; it keeps their
shape (quiet night, midday shoulder, peak at 20:00 to 21:00). Homes with PV are left out of these three
panels because their meters only see what the panels do not cover.
""")
    md("## 4. Physics: voltages, feeder profile, power identity, energy balance")
    code('''
v = truth.voltage_v[truth.voltage_v > 50]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
axs[0].hist(v, bins=120, color=C["blue"])
for lim in (207, 253):
    axs[0].axvline(lim, color="#55554f", lw=1, ls="--")
axs[0].set_title("Home voltage, all clean intervals (V)"); axs[0].set_yticks([])
axs[0].text(207, axs[0].get_ylim()[1] * 0.9, " EN 50160 limits", color="#55554f")
print(f"share of intervals inside 207-253 V: {((v > 207) & (v < 253)).mean():.4f}")

t = truth[truth.feeder == "F1"].copy()
t["hour"] = t.timestamp.dt.hour
summer = t.timestamp.dt.month.isin([7, 8])
spring = t.timestamp.dt.month.isin([4, 5])
for lab, mask, col in [("summer 21:00 (peak load)", summer & (t.hour == 21), C["orange"]),
                       ("spring 12:00 (PV backfeed)", spring & (t.hour == 12), C["aqua"])]:
    prof = t[mask].groupby("pole").voltage_v.mean()
    axs[1].plot(prof.index, prof.values, marker="o", ms=4, color=col, label=lab)
axs[1].set_xlabel("pole along feeder F1"); axs[1].set_title("Average voltage along the feeder (V)")
axs[1].legend()
plt.tight_layout(); plt.show()
''')
    code('''
clean = meters[meters.is_anomaly == 0]
s = clean.voltage_v * clean.current_a / 1000
ok = s > 0.05
ident = (clean.active_power_kw.abs() - s * clean.power_factor)[ok] / s[ok]
print(f"Power identity |P| = V*I*PF on clean observed rows: median |residual| {ident.abs().median():.4f}, "
      f"99.9th pct {ident.abs().quantile(0.999):.4f}")

hp = head.groupby(["timestamp", "feeder"], observed=True).active_power_kw.sum()
mp = truth.groupby(["timestamp", "feeder"], observed=True).active_power_kw.sum()
gap = (hp - mp).unstack()
fig, ax = plt.subplots(figsize=(10, 2.8))
wk = gap.loc["2025-02-03":"2025-02-09"]
ax.plot(wk.index, wk["F1"], color=C["blue"], label="F1"); ax.plot(wk.index, wk["F2"], color=C["orange"], label="F2")
ax.set_title("Feeder-head power minus sum of home meters (kW): the network losses, one week"); ax.legend()
plt.show()
''')
    md("""
Clean data obeys the physics the detectors rely on: the power identity holds to within meter noise,
and the feeder balance gap is just the network losses, which follow the load. A data problem has to
break one of these relations; a real grid event keeps them.
""")
    md("## 5. Anomaly gallery: observed (what the meter reported) vs true")
    code('''
def show(ax, etype, channel, pad_h=12, which=0):
    e = events[events.anomaly_type == etype].sort_values("start").iloc[which]
    mid = (e.meters or "").split(";")[0] if isinstance(e.meters, str) else None
    a, b = e.start - pd.Timedelta(hours=pad_h), e.end + pd.Timedelta(hours=pad_h)
    long = (e.end - e.start) > pd.Timedelta(days=2)
    if long:
        a, b = e.start - pd.Timedelta(days=3), e.end + pd.Timedelta(days=3)
    o = meters[(meters.meter_id == mid) & meters.timestamp.between(a, b)].set_index("timestamp")[channel]
    tr = truth[(truth.meter_id == mid) & truth.timestamp.between(a, b)].set_index("timestamp")[channel]
    if long:
        o, tr = o.resample("6h").mean(), tr.resample("6h").mean()
    ax.plot(tr.index, tr.values, color=C["gray"], lw=1.2, label="true")
    ax.plot(o.index, o.values, color=C["orange"], lw=1.2, label="observed")
    ax.axvspan(e.start, e.end, color=C["yellow"], alpha=0.12, lw=0)
    ax.set_title(f"{etype} · {mid} · {channel}", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7, rotation=30)

panels = [("drift_voltage", "voltage_v"), ("drift_current", "current_a"), ("bias_voltage", "voltage_v"),
          ("bias_ct_gain", "active_power_kw"), ("stuck", "voltage_v"), ("spike", "current_a"),
          ("dropout", "active_power_kw"), ("clock_skew", "voltage_v"), ("theft_bypass", "active_power_kw"),
          ("tamper_report", "active_power_kw"), ("replay", "voltage_v"), ("coordinated_fdi", "voltage_v"),
          ("meter_end_of_life", "voltage_v"), ("leakage", "reactive_power_kvar"),
          ("high_resistance_joint", "voltage_v"), ("outage", "voltage_v"),
          ("pv_soiling", "active_power_kw"), ("pv_inverter_trip", "active_power_kw"), ("pv_degradation", "active_power_kw"),
          ("pv_reading_stuck", "pv_power_kw"), ("pv_inflated_report", "pv_power_kw")]
fig, axs = plt.subplots(6, 4, figsize=(13, 15))
for ax in axs.ravel()[len(panels):]:
    ax.axis("off")
for ax, (et, ch) in zip(axs.ravel(), panels):
    show(ax, et, ch)
axs[0, 0].legend(loc="upper left", fontsize=7)
plt.suptitle("Shaded = injected event window. Long events are shown as 6-hour means.", y=1.0, fontsize=9)
plt.tight_layout(); plt.show()
''')
    md("""
Where the true and observed lines overlap, the anomaly is **physical** (leakage, joint, outage, dusty or
tripped panels): the meter reports the truth and the grid itself changed. Where they split, the meter is lying. Note the
hard cases: current drift, CT gain error and bypass theft keep P = V·I·PF intact, so only the feeder
energy balance reveals them; the coordinated attack moves several meters together, like a real swell.
""")
    md("## 6. Hard negative: the heatwave is real demand, not an anomaly")
    code('''
fig, ax = plt.subplots(figsize=(10, 3))
w = tx.set_index("timestamp").loc["2025-07-12":"2025-08-02"]
ax.plot(w.index, w.load_pct, color=C["blue"], label="transformer load (%)")
ax.plot(w.index, w.top_oil_temp_c, color=C["orange"], label="top-oil temperature (°C)")
for d in ctx.loc["2025-07-12":"2025-08-02"].heatwave.resample("D").max().pipe(lambda s: s[s.astype(bool)]).index:
    ax.axvspan(d, d + pd.Timedelta(days=1), color=C["orange"], alpha=0.1, lw=0)
ax.set_title("Heatwave week (shaded): load and oil temperature rise, nothing is labelled"); ax.legend(ncol=2)
plt.show()
''')
    return c


def nb2():
    c = []
    md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
    code = lambda s: c.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731
    md("""
# VoltTrust Tunisia: Phase 1 baseline detector

Implements the Goal 1 strategy on the Phase 1 dataset:

- **L1 physics and rules**: voltage-profile model per meter (fitted on the train split), power identity,
  register consistency, stuck values, dropouts (not collection gaps), meter tamper alarms, feeder-phase
  energy balance (an alert on the phase, with a regression ranking the homes to check),
  head-meter events (outage, phase loss, sag, MV voltage step, frequency), transformer overload from
  an estimated top-oil temperature (and the oil residual where a monitor is fitted), rooftop panel checks.
- **L2**: Isolation Forest on the physics residuals, with SHAP naming the signal behind each forest
  alert; persistent-offset test on the voltage residual (bias vs drift by its slope).
- **L3**: a rule table that gives every alert a verdict, *grid_event*, *equipment_fault* or *data_quality*.

Thresholds were set on the **val** split. Training uses the **train** split (January to April, no data
problems). The **test** split contains two attack types never seen before (replay, coordinated false data).

This notebook runs the default equipment (`Site()`): the substation concentrator's metering, STEG's phase
map, no oil monitor, inverter readings from the 3 opt-in solar homes and the net meter for the other 11.
Load profiles arrive every 12 hours, so "time to operator" includes that wait; alarms arrive at once.
""")
    code(SETUP)
    code('''
import time
from volttrust.detect import features, detectors, evaluate, calibrate
t0 = time.perf_counter()
D = features.load(DATA)
R = features.residuals(D)
det = detectors.detect(D, R)
print(f"detector ran on {D.T} intervals x {D.H} meters in {time.perf_counter() - t0:.1f} s")
events = pd.read_csv(DATA / "events.csv")
meter_phase = {m: f"F{f + 1}-{'ABC'[p]}" for m, f, p in zip(D.meters, D.feeder, D.phase)}
res, al = evaluate.match(det.alerts, events, D.idx, meter_phase, collected=D.collected, meters=D.meters)
point = evaluate.point_metrics(det.meter_rule, DATA / "meters.parquet", D.meters, D.idx, D.split)
summary = evaluate.summarize(res, al, D.idx, D.split, point)
pd.DataFrame(summary).T
''')
    md("""
## Against the proposal targets

Targets: recall of at least 0.90, at most 1 false alert per feeder per day, at least 0.85 correct
verdicts; processing under 1 s, alarms under 5 minutes, other checks within one 12-hour batch.
Some events fall entirely inside readings the concentrator failed to collect; `event_recall_observable`
leaves those out.
""")
    code('''
cols = ["event_recall", "event_recall_observable", "verdict_accuracy_on_detected", "false_alerts_per_feeder_per_day",
        "median_time_to_detect_h", "median_time_to_operator_h"]
pd.DataFrame({sp: {k: summary[sp][k] for k in cols} for sp in ("val", "test")})
''')
    md("## Recall and verdict accuracy by anomaly type (val + test)")
    code('''
pt = evaluate.per_type(res)
pt
''')
    code('''
pt2 = pt.sort_values("recall")
fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(pt2))
ax.barh(y - 0.2, pt2.recall, height=0.38, color=C["blue"], label="detected (recall)")
ax.barh(y + 0.2, pt2.verdict_acc.fillna(0), height=0.38, color=C["aqua"], label="right verdict, when detected")
ax.set_yticks(y, [f"{t}{' (held out)' if h else ''}" for t, h in zip(pt2.anomaly_type, pt2.held_out)])
ax.set_xlim(0, 1.05); ax.legend(loc="lower right"); ax.set_title("Event-level results, val + test")
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()
''')
    md("""
## Rooftop panels: inverter homes vs estimated homes

The 3 opt-in homes are checked on their inverter reading; the other 11 on an estimate from the net
meter and the neighborhood's reference panel. A frozen or inflated inverter reading can only exist where
there is an inverter reading, so those two types are only on opt-in homes.
""")
    code('''
evaluate.by_pv_source(res, al, D)
''')
    md("""
## Confidence: measured per rule

Each alert's confidence is how often its rule gave the right verdict in May to Jul, pulled toward the
hand-set value (worth 10 alerts) and capped at 0.97. Alerts where 2 or more rules agree have their own
value. Calibration error (ECE) is the average gap between stated confidence and how often alerts are
right, before (hand-set) and after (measured), on the test months.
""")
    code('''
cal = calibrate.fit(al, res, D.split, "val", D.idx, source="data")
al = calibrate.apply(al, cal)
ece, per_rule = calibrate.report(al, res, D.split, ("test",))
print(ece)
per_rule.sort_values("alerts", ascending=False).head(12)
''')
    md("""
## Why the Isolation Forest flagged a reading: SHAP

Forest alerts only say "unusual pattern". SHAP splits the forest's score at the alert's peak reading into
the push from each of its 4 signals: the voltage gap to the physics model (`r_v`), how jumpy that gap is
over 2 hours (`rv_sd`), the power-identity residual (`r_id`) and the register residual (`r_reg`).
Positive means "pushed toward unusual".
""")
    code('''
fa = al[al.rule == "iforest"]
ex = fa[fa.matched_event.notna()].iloc[0]
etype = events.set_index("event_id").anomaly_type.get(ex.matched_event)
fig, ax = plt.subplots(figsize=(6, 2.2))
ax.barh(list(ex.shap), list(ex.shap.values()), color=C["blue"])
ax.set_title(f"{ex.asset} at {ex.t_start:%d %b %H:%M} (true type: {etype}): main signal {ex.shap_top['feature']}")
ax.axvline(0, color="#55554f", lw=0.8); plt.tight_layout(); plt.show()
print("SHAP time:", {k: v for k, v in det.timing.items()})
''')
    code('''
# the main signal per forest alert, by the true anomaly type it matched (a first check that explanations are right)
ev_type = dict(zip(res.event_id, res.anomaly_type))
pd.crosstab(fa.matched_event.map(ev_type).fillna("no event (false alert)"),
            fa.shap_top.map(lambda x: x["feature"]), rownames=["true type"], colnames=["main signal"])
''')
    md("""
## Explanation quality: the four measures against their targets

Scored against the injected ground truth, on both seeds and on the val and test months:
verdict accuracy, the right home among the 3 to check for energy gaps, calibration error (ECE) and
whether SHAP's main signal is the signal the injected fault corrupts, both raw and as shown to the
operator (a pattern alert leads with a nearby voltage rule's reason, SHAP second). Built by
`python -m volttrust.detect.quality` into `results/explanation_quality.csv`. Seed 42's val ECE is
in-sample (the confidence table was fitted on those months); seed 2026 uses seed 42's table.
""")
    code('''
q = pd.read_csv(DATA.parent / "results" / "explanation_quality.csv")
tab = q[q.fault_type == "all"].pivot_table(index=["measure", "target"], columns=["seed", "window"],
                                            values="value", aggfunc="first")
tab.loc[[("verdict_accuracy", ">= 0.85"), ("right_home_in_top3", ">= 0.7"),
         ("calibration_error_ece", "<= 0.08"), ("shap_main_signal_match", ">= 0.8"),
         ("shap_main_signal_match_as_shown", ">= 0.8")]]
''')
    code('''
# SHAP match per fault type, so the dominant noise type can't hide weak ones
q[(q.measure == "shap_main_signal_match") & (q.fault_type != "all")][
    ["seed", "window", "fault_type", "n", "value", "note"]]
''')
    md("## Grid event or data problem? Verdict confusion (events)")
    code('''
evaluate.confusion(res)
''')
    md("""
## From alert to verdict to escalation

Each alert becomes a verdict record and goes through the
escalation router (`escalation/`). Confidence here is the measured value per rule
above; Phase 2 replaces it with the calibrated two-stage confidence model.
""")
    code('''
sys.path.insert(0, str((DATA.parent / "escalation").resolve()))
from router import Router, Event, TIERS

TEMPLATES = {
    "grid_event": "Likely real {subtype} on {asset}: seen at the feeder head, so the sensors agree with each other.",
    "equipment_fault": "Likely equipment fault ({subtype}) at {asset}: readings are physically consistent but the network behaves abnormally.",
    "data_quality": "Likely data problem ({subtype}) at meter {asset}: {why}",
}
WHY = {
    "voltage_offset": "its voltage is {v:.1f} V away from what the feeder physics predicts, while its neighbours agree with the model.",
    "power_identity_persist": "reported power no longer equals voltage x current x power factor.",
    "group_voltage_rise": "several meters on the feeder report the same voltage rise that the feeder head does not see.",
}

def explain(a):
    if a.rule in detectors.PHASE_RULES:
        # the system names no home: it hands a person the list of homes to check
        homes = ", ".join(f"{x['meter']}" + ("" if x["evidence"] == "strong" else " (weak)") for x in a.suspects)
        where = f"feeder {a.asset[1]}" + (f" phase {a.asset[-1]}" if "-" in a.asset else "")
        return (f"About {abs(a.gap_kw):.2f} kW is {'missing' if a.gap_kw > 0 else 'over-reported'} on "
                f"{where}; homes to check: {homes or 'none stands out'}.")
    why = WHY.get(a.rule, "rules fired: " + ", ".join(a.rules_fired) + ".")
    why = why.format(v=getattr(a, "max_abs_voltage_resid_v", 0) or 0, b=getattr(a, "balance_beta", 0) or 0)
    return TEMPLATES[a.verdict].format(subtype=a.subtype.replace("_", " "), asset=a.asset, why=why)

router = Router(DATA.parent / "escalation" / "policy.yaml")
test_al = al[(al.t_start >= "2025-08-01") & al.matched_event.notna()]
picks = pd.concat([test_al[test_al.rule == r].head(1) for r in
                   ["head_sag", "head_outage", "group_voltage_drop", "voltage_offset", "balance_gap_phase",
                    "power_identity_persist", "group_voltage_rise", "stuck"]])
rows = []
for rec, a in zip(detectors.to_verdict_records(picks), picks.itertuples()):
    d = router.route(Event.from_verdict(rec, meters_on_feeder=24))
    rows.append({"asset": a.asset, "verdict": a.verdict, "subtype": a.subtype, "confidence": a.confidence,
                 "tier": f"{d.tier}: {TIERS[d.tier]}", "explanation": explain(a)})
pd.set_option("display.max_colwidth", 200)
print("decision log intact:", router.verify_log())
pd.DataFrame(rows)
''')
    md("""
## What this baseline does not do yet (Phase 2)

- Confidence is measured per rule on May to Jul, not yet per alert. Phase 2: a calibrated two-stage model and conformal "uncertain" verdicts (Goal 2).
- Current-channel problems (CT drift, CT gain error, bypass theft) are the hardest: they keep the power
  identity intact and only show in the feeder balance. The alert names the feeder phase and a ranked list
  of homes to check, never a culprit; small errors still take days to confirm.
- Clock skew and replay are caught, but not named; the verdict is right, the subtype is generic.
- No autoencoder yet; SHAP explains only the Isolation Forest alerts.
- The other equipment modes (no substation reading, inferred phases, oil monitor, all-estimated solar)
  are run from the command line, `python -m volttrust.detect.run --help`; their results are in
  `results/equipment_modes.csv`.
""")
    return c


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
    nbf.write(nb, NB / name)
    print("wrote", NB / name)


if __name__ == "__main__":
    import sys
    NB.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["1", "2"]
    if "1" in which:
        build("01_data_exploration.ipynb", nb1())
    if "2" in which:
        build("02_baseline_detection.ipynb", nb2())
