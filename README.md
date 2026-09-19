# Industrial Agent

An LLM-powered agentic system for industrial equipment monitoring and diagnostics, combining **heat exchanger fouling analysis** and **bearing vibration fault detection** into a unified tool-calling agent.

---

## Architecture

```
                  INDUSTRIAL AGENT
                        │
               ┌────────┴────────┐
               │                 │
         HX-101 / HX-102      Bearing
               │                 │
        Heat exchanger       Vibration
           dataset            dataset
               │                 │
               └────────┬────────┘
                        │
                   DATA TOOLS
                        │
                        ▼
                      LLM
                        │
                 Tool selection
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
        Analysis     Engineering    RAG
         tools         tools        tools
```

---

## Datasets

### Heat Exchanger — `heat_exchanger_fouling_clean.csv`
Simulated crude oil preheat train (5 exchangers in series: E01–E05).

| Column Group | Description |
|---|---|
| `Crude_API`, `Crude_Chlorides`, `Crude_TAN` | Crude oil quality properties |
| `E0x_Crude_Tube_T_In/Out_degC` | Tube-side (crude) inlet/outlet temperatures per exchanger |
| `E0x_Crude_Tube_m_kg_s` | Crude mass flow rate per exchanger |
| `E0x_*_Shell_T_In/Out_degC` | Shell-side (product) temperatures per exchanger |
| `E0x_*_Shell_m_kg_s` | Product-side mass flow rates |
| `Time_hr` | Simulation time in hours |

- **64,001 rows** of time-series simulation data
- Products on shell side: Heavy Naphtha, Kerosene, Light Diesel, LVGO, Heavy Diesel

---

### Bearings — `bearings_all_features.csv`
IMS (University of Cincinnati) bearing run-to-failure dataset. Vibration features extracted from raw accelerometer signals.

| Column | Description |
|---|---|
| `timestamp` | Measurement timestamp |
| `test_id` | Test run identifier (1, 2, or 3) |
| `Bearing{1-4}_{x,y}_{feature}` | Per-axis (x/y) vibration features per bearing |
| `Bearing{1-4}_{feature}` | Combined vibration features per bearing |
| `hours_elapsed` | Time since test start (hours) |
| `failed_bearing` | Which bearing(s) failed |
| `failure_type` | Fault mode (e.g., `outer_race`) |

**Extracted features per bearing/axis:** `rms`, `peak`, `peak_to_peak`, `kurtosis`, `crest_factor`, `std`, `skewness`, `variance`, `mean_abs`, `shape_factor`, `impulse_factor`

- **9,464 rows** across 3 test runs (Test 1: 2156, Test 2: 984, Test 3: 6324)
- **138 columns** total
- Per-test CSVs also available: `bearings_test1_features.csv`, `bearings_test2_features.csv`, `bearings_test3_features.csv`

---

## Agent Tool Categories

### Analysis Tools
- Fouling resistance / heat transfer coefficient calculation
- Bearing health index trending
- Anomaly / threshold detection
- Statistical degradation metrics

### Engineering Tools
- LMTD / NTU-effectiveness calculations
- Bearing fault frequency estimation (BPFO, BPFI, BSF, FTF)
- Maintenance scheduling logic
- Remaining useful life (RUL) estimation

### RAG Tools
- Retrieval over maintenance manuals / P&IDs
- Fault codebook lookup
- Historical incident similarity search

---

## Directory Structure

```
industrial_agent/
├── README.md                          <- This file
├── heat_exchanger_fouling_clean.csv   <- HX dataset (64k rows)
├── heat_exchanger_fouling_dataset.csv <- HX dataset (raw/full)
├── bearings_all_features.csv          <- Bearings combined (9464 rows, 138 cols)
├── bearings_test1_features.csv        <- Test 1 only (2156 rows)
├── bearings_test2_features.csv        <- Test 2 only (984 rows)
├── bearings_test3_features.csv        <- Test 3 only (6324 rows)
└── bearings_dataset/                  <- Raw vibration signal files
```
