"""
Industrial Agent — Configuration
==================================
Resolves dataset paths and defines physical equipment constants.

Path resolution order:
  1. Environment variable  INDUSTRIAL_DATA_DIR
  2. C:/Projects/industrial_agent
  3. Repository root (two levels up from this file)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset Path Resolution
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> Path:
    """Locate the directory that contains the dataset CSV files."""
    # 1. Explicit env override
    env_dir = os.environ.get("INDUSTRIAL_DATA_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)

    # 2. Fixed project path (outside OneDrive)
    fixed = Path("C:/Projects/industrial_agent")
    if fixed.is_dir():
        return fixed

    # 3. Repository root (this file lives at agent/config.py)
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root


DATA_DIR: Path = _resolve_data_dir()

# Dataset file paths
HX_CLEAN_CSV: Path = DATA_DIR / "heat_exchanger_fouling_clean.csv"
HX_RAW_CSV: Path   = DATA_DIR / "heat_exchanger_fouling_dataset.csv"

BEARINGS_ALL_CSV: Path    = DATA_DIR / "bearings_all_features.csv"
BEARINGS_TEST1_CSV: Path  = DATA_DIR / "bearings_test1_features.csv"
BEARINGS_TEST2_CSV: Path  = DATA_DIR / "bearings_test2_features.csv"
BEARINGS_TEST3_CSV: Path  = DATA_DIR / "bearings_test3_features.csv"


def validate_paths() -> dict:
    """Check which dataset files are present and return a status dict."""
    files = {
        "hx_clean":      HX_CLEAN_CSV,
        "hx_raw":        HX_RAW_CSV,
        "bearings_all":  BEARINGS_ALL_CSV,
        "bearings_test1": BEARINGS_TEST1_CSV,
        "bearings_test2": BEARINGS_TEST2_CSV,
        "bearings_test3": BEARINGS_TEST3_CSV,
    }
    return {name: {"path": str(p), "exists": p.is_file()} for name, p in files.items()}


# ---------------------------------------------------------------------------
# Heat Exchanger Equipment Metadata
# ---------------------------------------------------------------------------

# Shell-side fluid name and approximate specific heat capacity (kJ/kg·K)
HX_EXCHANGER_META: dict = {
    "E01": {
        "shell_fluid":    "Heavy Naphtha",
        "shell_cp_kJ_kgK": 2.20,
        "tube_fluid":     "Crude Oil",
        "tube_cp_kJ_kgK":  2.30,
        "description":    "Crude preheat vs Heavy Naphtha side-cut",
    },
    "E02": {
        "shell_fluid":    "Kerosene",
        "shell_cp_kJ_kgK": 2.15,
        "tube_fluid":     "Crude Oil",
        "tube_cp_kJ_kgK":  2.30,
        "description":    "Crude preheat vs Kerosene side-cut",
    },
    "E03": {
        "shell_fluid":    "Light Diesel",
        "shell_cp_kJ_kgK": 2.10,
        "tube_fluid":     "Crude Oil",
        "tube_cp_kJ_kgK":  2.30,
        "description":    "Crude preheat vs Light Diesel side-cut",
    },
    "E04": {
        "shell_fluid":    "LVGO (Light Vacuum Gas Oil)",
        "shell_cp_kJ_kgK": 2.05,
        "tube_fluid":     "Crude Oil",
        "tube_cp_kJ_kgK":  2.30,
        "description":    "Crude preheat vs LVGO side-cut",
    },
    "E05": {
        "shell_fluid":    "Heavy Diesel",
        "shell_cp_kJ_kgK": 2.00,
        "tube_fluid":     "Crude Oil",
        "tube_cp_kJ_kgK":  2.30,
        "description":    "Crude preheat vs Heavy Diesel side-cut",
    },
}

# Column prefixes in the CSV for each exchanger
HX_COLUMN_PREFIX: dict = {
    "E01": {"tube": "E01_Crude_Tube",       "shell": "E01_HeavyNaphtha_Shell"},
    "E02": {"tube": "E02_Crude_Tube",       "shell": "E02_Kero_Shell"},
    "E03": {"tube": "E03_Crude_Tube",       "shell": "E03_LightDiesel_Shell"},
    "E04": {"tube": "E04_Crude_Tube",       "shell": "E04_LVGO_Shell"},
    "E05": {"tube": "E05_Crude_Tube",       "shell": "E05_HeavyDiesel_Shell"},
}

VALID_EXCHANGER_IDS = list(HX_EXCHANGER_META.keys())


# ---------------------------------------------------------------------------
# IMS Bearing Test Metadata
# ---------------------------------------------------------------------------

BEARING_TEST_META: dict = {
    1: {
        "n_channels":    8,
        "bearings":      ["Bearing1_x", "Bearing1_y", "Bearing2_x", "Bearing2_y",
                          "Bearing3_x", "Bearing3_y", "Bearing4_x", "Bearing4_y"],
        "failed_bearing": "Bearing3 & Bearing4 (outer race)",
        "failure_type":  "outer_race",
        "n_snapshots":   2156,
        "axis":          "dual",   # x and y axes per bearing
    },
    2: {
        "n_channels":    4,
        "bearings":      ["Bearing1", "Bearing2", "Bearing3", "Bearing4"],
        "failed_bearing": "Bearing1 (outer race)",
        "failure_type":  "outer_race",
        "n_snapshots":   984,
        "axis":          "single",
    },
    3: {
        "n_channels":    4,
        "bearings":      ["Bearing1", "Bearing2", "Bearing3", "Bearing4"],
        "failed_bearing": "Bearing3 (outer race)",
        "failure_type":  "outer_race",
        "n_snapshots":   6324,
        "axis":          "single",
    },
}

BEARING_METRICS = [
    "rms", "peak", "peak_to_peak", "kurtosis",
    "crest_factor", "std", "skewness", "variance",
    "mean_abs", "shape_factor", "impulse_factor",
]

VALID_TEST_IDS = list(BEARING_TEST_META.keys())

# ISO 10816 vibration severity thresholds (RMS velocity mm/s — class II machinery)
ISO_10816_ZONES = {
    "A": (0.0, 2.3),    # New machinery — GOOD
    "B": (2.3, 4.5),    # Acceptable for long-term operation
    "C": (4.5, 7.1),    # Restricted operation — plan maintenance
    "D": (7.1, float("inf")),  # Danger — immediate action required
}
