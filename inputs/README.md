# Real-data inputs

| File | What | Source | Used for |
|---|---|---|---|
| `open_meteo_tunis_2025_hourly.csv` | Hourly temperature (°C), shortwave radiation (W/m², mean of the preceding hour) and cloud cover (%), Tunis 36.80N 10.17E, 1 Jan to 31 Dec 2025, 8,760 rows | Open-Meteo historical weather API (CC BY 4.0) | Drives the simulation: air conditioning, heating, fridges, lighting, PV output, heatwave flags |
| `steg_monthly_index_2013_2021.csv` | Month / yearly mean of STEG's national electricity sales, low voltage (BT) and medium voltage (MT), averaged over 2013 to 2021, with standard deviation | STEG open data, monthly consumption by voltage level (2005 to 2021); broken months removed (Feb–Mar 2012, Jan–Feb and Aug–Sep 2016, Nov–Dec 2018, Apr–May 2020) | Calibration check of the seasonal swing |
| `morocco_laayoune_hourly_profile.csv` | Weekday load curve / daily mean, by hour, of three residential district feeders in Laayoune (zones 2 to 4) | Moroccan 10-minute zone loads, Sep 2022 to May 2024 (UCI dataset 1158, CC BY 4.0) | Calibration check of the daily curve |

Benchmark used for the level: a Tunisian household uses about **1,450 kWh a year** (UNESCWA, 2017).

What the calibration targets mean (`src/volttrust/sim/loads.py`):

- **Level:** median home close to 1,450 kWh a year.
- **Months:** summer about 1.2x the yearly mean, between STEG's billed low-voltage series (its peak lags to September–November) and its metered medium-voltage series (peak 1.17x in July).
- **Daily curve:** the Laayoune feeders aggregate thousands of customers (night 0.7x, evening peak 1.25x the daily mean). 48 homes cannot smooth that much, so the simulated feeder is allowed a deeper night (about 0.6x) and a sharper evening peak (about 1.8x), with the same shape.

The Portuguese UCI dataset 321 was screened for household-like clients and dropped: none of its 370 clients passed as a home.
