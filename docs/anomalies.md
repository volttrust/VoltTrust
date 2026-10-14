# How the anomalies were generated

This is the documentation the spec requires ("Teams must document how the anomalies were generated"). Code: `src/volttrust/sim/anomalies.py`. Parameters: `src/volttrust/sim/config.py`. Everything is reproducible from the seed (`data/config.json`).

## 1. Two injection layers

| Layer | What it changes | Consequence |
|---|---|---|
| **Physical** | Inputs of the power flow: MV source voltage, household consumption, line resistance, which feeder phases are energized, system frequency, transformer cooling | Every sensor (household meters, feeder-head meters, transformer) sees the event consistently, because the physics propagates it |
| **Measurement: sensing** | What one meter *measures*, inside its measurement chain (voltage sensor, current transformer, metrology firmware) | The energy registers integrate the wrong values too |
| **Measurement: reporting** | What one meter *reports* to the head-end system (communications, data store, attacker in the middle) | Only the reported row changes; the meter's internal register keeps counting |

True values (before meter noise and measurement anomalies) are in `data/meters_true.parquet` for every row.

Base meter noise on every row (clean or not), Gaussian, relative: voltage 0.1%, current 0.3%, active power 0.3%, reactive power 0.5% (IEC class 1 meters). Feeder-head meters are utility grade (0.05 to 0.2%) and are never given measurement anomalies: they are the trusted anchor.

## 2. Catalog

Durations are in 15-minute intervals unless marked "days". Ranges are uniform draws. "Fingerprint" is what the physics checks of Goals 1 and 2 should see.

### 2.1 Genuine grid events (physical, verdict `grid_event`)

| Type | Generation model | Parameters | Assets | Fingerprint |
|---|---|---|---|---|
| `voltage_sag` | A short MV dip (about 1 s). The interval's minimum voltage drops by the depth; the 15-min mean barely moves | depth 15 to 50% of `voltage_min_v`, mean −0.2 to −1%; 1 interval | all meters + all heads | every meter and head sees it at once |
| `mv_undervoltage` | MV source voltage reduced | −4 to −9%; 2 to 12 intervals | all meters + heads | all voltages fall together; head falls too; power identity holds |
| `tap_swell` | MV source voltage raised (tap change), placed between 09:00 and 15:00 so it stacks on PV backfeed | +4 to +7%; 4 to 32 intervals | all meters + heads | all voltages rise together, some above 253 V |
| `outage` | Feeder (70%) or whole transformer (30%) de-energized, then **cold-load pickup**: consumption ×(1 + 0.6·e^(−k/3)) for 8 intervals after restoration | 2 to 16 intervals off + 8 pickup | feeder or all meters, heads, transformer if whole | V = I = P = 0 on all of them, head too; registers stop; then a surge |
| `frequency_excursion` | System frequency dip (loss of generation elsewhere) | −0.3 to −0.7 Hz; 1 to 2 intervals | heads only (meters don't report frequency) | all heads, same value |

### 2.2 Equipment faults (physical, verdict `equipment_fault`)

| Type | Generation model | Parameters | Assets | Fingerprint |
|---|---|---|---|---|
| `phase_loss` | One feeder phase de-energized (blown fuse) | 2 to 12 intervals | meters on that phase + that head phase | V = 0 only on one phase of one feeder |
| `high_resistance_joint` | Corroding connector at one pole: series resistance added to that span, growing quadratically (slow, then accelerating) | up to 0.15 to 0.5 Ω over 5 to 20 days | meters on that phase downstream of the pole + head phase | downstream voltages sag more as load rises, upstream neighbors unaffected; physics consistent; extra losses at head |
| `leakage` | Failing appliance / insulation leak: extra constant load at PF 0.5, growing over the first 70% of the event | 0.2 to 0.8 kW; 3 to 15 days | one meter | real extra consumption: head sees it, balance holds, PF drops, night base load rises |
| `transformer_cooling` | Degraded cooling (blocked radiator, low oil): rated top-oil rise ×1.3 to 1.6 in the IEC 60076-7 thermal model | 5 to 15 days | transformer | oil temperature too high for the load and ambient |

### 2.3 Degraded sensors (sensing, verdict `data_quality`)

| Type | Generation model | Parameters | Fingerprint |
|---|---|---|---|
| `drift_voltage` | Voltage sensor gain drifts linearly: g(t) = 1 + rate·days. P and Q (computed from v·i inside the meter) drift with it | ±0.05 to ±0.5 %/day; 10 to 45 days. 70% of drifting meters are model MB-200 | voltage departs from peers on the same phase; P = V·I·PF still holds; balance slightly off |
| `drift_current` | Current transformer gain drifts the same way; I, P, Q and the import register follow | same ranges | **power identity holds**; only the feeder energy balance and the peer comparison show it |
| `noise_growth` | Extra Gaussian noise on V, I, P | ×3 to ×10 the class-1 noise; 3 to 14 days | rough signal, identity residual grows |
| `stuck` | All reported channels frozen at the last value (including registers) | 4 to 192 intervals | zero variance; register jumps when it unfreezes |

### 2.4 Biased readings (sensing / reporting, verdict `data_quality`)

| Type | Generation model | Parameters | Fingerprint |
|---|---|---|---|
| `bias_voltage` | Constant voltage offset (miscalibrated replacement meter); P scaled by (V+b)/V | ±2 to ±8 V; 20 to 60 days | steady offset vs peers |
| `bias_ct_gain` | Wrong CT ratio: I, P, Q × gain | 0.88 to 0.97 or 1.03 to 1.12; 20 to 60 days | identity holds; steady balance residual |
| `clock_skew` | Reported values lag the true time | 1 to 4 intervals (15 to 60 min); 3 to 20 days | cross-correlation with peers peaks at a lag |

### 2.5 Corrupted readings (reporting, verdict `data_quality`)

| Type | Generation model | Parameters | Fingerprint |
|---|---|---|---|
| `spike` | One interval, one channel corrupted: voltage ×1.25 to 1.6 or ×0.3 to 0.6; current or power ×3 to 10; or a register bit flip (+2^10 to +2^16 kWh) | 1 interval | impossible value, identity broken, no neighbor sees it |
| `dropout` | All channels missing (NaN): communication loss | 1 to 96 intervals | gaps; register catches up afterwards |

### 2.6 Falsified readings (verdict `data_quality`)

| Type | Generation model | Parameters | Fingerprint |
|---|---|---|---|
| `theft_bypass` | Part of the load wired around the meter: the meter sees only a fraction of the imported current. Mode `continuous` or `evening` (17:00 to 23:00 only) | fraction metered 0.3 to 0.7; 3 to 30 days | **identity holds** (the meter measures correctly what passes through it); feeder balance residual opens |
| `tamper_report` | Firmware / data tampering: reported P, Q and the import register scaled down; V, I and PF keep reporting true values | fraction 0.3 to 0.7; 3 to 30 days | P ≠ V·I·PF; balance residual |
| `replay` **(held out)** | Reported channels replaced by the same meter's values from 7 days earlier; register rebuilt from the replayed power | 1 to 3 days; test split only | plausible shape, but voltage no longer tracks the neighbors; small balance gap |
| `coordinated_fdi` **(held out)** | 3 to 8 meters on one feeder report the same voltage rise, ramped in over 1 hour; current lowered so P = V·I·PF still holds | +8 to +14 V; 8 to 48 intervals; test split only | looks like a swell, but the head meter sees nothing and the affected meters jump away from the voltage the feeder physics predicts for them, while their neighbours do not |

### 2.7 Natural failure (verdict `data_quality`)

| Type | Generation model | Parameters | Fingerprint |
|---|---|---|---|
| `meter_end_of_life` | Noise grows ×1 to ×8 over the event; from 40% of the event, intermittent dropouts with probability rising 5% to 60%; last 20%: dead (no data) | 10 to 30 days | noise, then gaps, then silence |

### 2.8 Rooftop solar panels (verdict depends on the type)

About 30% of homes (14) have rooftop PV. STEG installs only a net meter, so the inverter's own production reading `pv_power_kw` (a separate device from the meter, 1% noise) is shared by **3 opt-in homes** only; it is NaN for the other 11 solar homes and for homes without PV (`pv_source` in `meters.csv`: `inverter_optin` or `estimated`). It reads 0 whenever the home's phase is de-energized (anti-islanding). A clean reference panel per neighborhood is measured in `context.parquet: ref_pv_kw_per_kwp`. All panel events start in val or test, alternating, so the train split stays clean for learning each home's normal output. Physical panel events can hit any of the 14 solar homes; the two reading problems only the 3 opt-in homes, since the others have no reading to corrupt.

| Type | Layer | Generation model | Parameters | Verdict | Fingerprint |
|---|---|---|---|---|---|
| `pv_soiling` | physical | Dust and sand: output falls linearly, then recovers abruptly at the end (cleaning or rain) | loss reaches 10 to 30%; 14 to 40 days | `equipment_fault` | ratio to the other solar homes slides down slowly; the meter's export falls too |
| `pv_inverter_trip` | physical | Inverter stops in full sun | output 0; 8 to 192 intervals, starting 08:00 to 14:00 | `equipment_fault` | zero output while every other solar home produces; the home imports more |
| `pv_degradation` | physical | Failed string or bypass diode: a step down until repaired | loss 15 to 25%; 30 to 60 days | `equipment_fault` | ratio to peers drops in one step and stays there |
| `pv_reading_stuck` | inverter reading | Reading frozen at its last value; the panels keep working | 8 to 96 intervals, starting 08:00 to 14:00 | `data_quality` | identical readings in changing sunshine; the meter still shows normal solar behavior |
| `pv_inflated_report` (detector subtype `pv_reading_high`, "inverter reading too high") | inverter reading, biased | Reported production multiplied; the panels and the meter's export do not change. STEG pays on the net meter, so this earns the owner nothing and blames nobody. The generated files keep the original type name and the `falsified_reading` family | ×1.2 to ×1.5; 3 to 20 days | `data_quality` | ratio to peers jumps above 1 while the meter's export does not rise |

Physical panel events change the power flow (the meter's net power and the feeder head see them); reading problems change only `pv_power_kw`.

### 2.9 Hard negatives (not labelled as anomalies)

Detectors should not flag these. They are in the baseline on purpose.

| Condition | Where |
|---|---|
| Heatwaves from the real 2025 weather: days in a 3-day spell averaging 30.5 °C or more (13 to 15 July, 19 to 26 July with 46.1 °C on the 21st, 26 to 28 August, 7 to 9 September) | `context.parquet: heatwave` |
| Ramadan (1 to 29 March 2025): suhoor peak ~03:00, quiet days, late evenings | `is_ramadan` |
| Weekends and public holidays | `is_weekend`, `is_holiday` |
| PV backfeed at noon (negative power, voltage rise on PV feeders) and cloud ramps; clouds lower every solar home's output together | `ghi_wm2`, `pv_power_kw` |
| Natural voltage drop along the feeder in the evening peak | topology |

## 3. Scheduling rules

- Physical events can occur in any split. Measurement events start only in `val` or `test`, alternating so every non-held-out type appears in both.
- `replay` and `coordinated_fdi` start only in `test` (held out, `held_out = True` in `events.csv`).
- Rooftop PV events (physical and reading) start only in `val` or `test`, alternating, on homes with PV; reading events only on the opt-in homes.
- Events never overlap on the same meter, except that a meter-level event may overlap a feeder- or transformer-wide event; there the row carries the meter-level label.
- `drift_*` events pick meter model MB-200 70% of the time (MB-200 is 35% of the fleet), so a per-model drift rate is learnable (Goal 4 calibration insights).
- An event may start inside one split and finish in the next; `events.csv: split` is the split of its start.

## 4. Files and labels

| File | Content |
|---|---|
| `events.csv` | One row per event: `event_id`, `anomaly_type`, `family`, `verdict`, `layer`, `scope`, `meters`, `heads`, `transformer`, `channels`, `start`, `end`, `duration_intervals`, `split`, `held_out`, `params` (JSON) |
| `meters.parquet` | Observed rows (timestamp × meter), including the inverter's `pv_power_kw`, with `is_anomaly`, `event_id`, `anomaly_type`, `family`, `verdict` |
| `feeder_head.parquet` | Feeder-head rows (timestamp × feeder × phase) with the same labels |
| `transformer.parquet` | Transformer load, top-oil temperature and losses with labels |
| `meters_true.parquet` | True values for every meter row |
| `alarms.csv` | Alarms the meters push at once: `power_cut`, `power_restore` (from the true voltage), `tamper_cover` / `tamper_magnet` (at the start of about half the `theft_bypass` events, plus 4 false alarms a year); `event_id` is for scoring only |
| `collection_log.csv` | Profile reads the concentrator failed to collect: `batch_start`, `batch_end`, `meter_id`, `reason` (`read_failed`, 5 to 15% of each meter's 12-hour batches; `concentrator_offline`, 2 whole batches a year). Those readings are blank in `meters.parquet`; an injected event inside them can be invisible |

`verdict` is the three-way answer Goal 1 must give: `grid_event`, `equipment_fault`, `data_quality`. `family` is the spec's own wording (grid event, equipment fault, degraded sensor, biased reading, corrupted reading, falsified reading, natural failure).

## 5. Known simplifications

- 15-minute resolution. Sub-second events are represented by the interval minimum voltage, the way meters log them.
- Each phase of each feeder is solved as a per-phase equivalent with part of the neutral return included; neutral-shift effects of unbalance are not modeled.
- Household loads are synthetic (no Tunisian meter-level data is public), driven by the real 2025 Tunis weather and calibrated against real data: the Tunisian household benchmark (about 1,450 kWh a year), STEG's monthly national sales and residential feeder loads from Laayoune, Morocco (`inputs/README.md`). A public Portuguese smart-meter dataset was screened as a source of real load shapes and dropped: none of its 370 clients behaves like a home.
