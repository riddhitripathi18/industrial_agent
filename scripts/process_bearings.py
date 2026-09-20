"""
IMS Bearing Dataset Processor
Processes raw IMS bearing vibration data into feature CSVs.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

BASE = os.getenv("IMS_DATA_DIR", r"C:\Projects\industrial_agent\bearings_dataset\IMS_extracted")
OUT_DIR = os.getenv("INDUSTRIAL_DATA_DIR", r"c:\Users\riddh\OneDrive\Desktop\industrial_agent")


TEST_DIRS = {
    1: {"path": os.path.join(BASE, "1st_test"), "channels": 8},
    2: {"path": os.path.join(BASE, "2nd_test"), "channels": 4},
    3: {"path": os.path.join(BASE, "4th_test", "txt"), "channels": 4},
}

CHANNEL_NAMES = {
    8: ["Bearing1_x", "Bearing1_y", "Bearing2_x", "Bearing2_y",
        "Bearing3_x", "Bearing3_y", "Bearing4_x", "Bearing4_y"],
    4: ["Bearing1", "Bearing2", "Bearing3", "Bearing4"],
}

FAILURE_INFO = {
    1: {"failed_bearing": "Bearing3_x/y & Bearing4_x/y", "failure_type": "outer_race"},
    2: {"failed_bearing": "Bearing1", "failure_type": "outer_race"},
    3: {"failed_bearing": "Bearing3", "failure_type": "outer_race"},
}

def parse_timestamp(filename: str) -> datetime:
    name = os.path.basename(filename)
    parts = name.split(".")
    try:
        return datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                        int(parts[3]), int(parts[4]), int(parts[5]))
    except Exception:
        return datetime(1970, 1, 1)

def extract_features(signal: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    peak2peak = float(np.max(signal) - np.min(signal))
    kurt = float(stats.kurtosis(signal, fisher=True))
    crest = float(peak / rms) if rms != 0 else 0.0
    std_val = float(np.std(signal))
    skew = float(stats.skew(signal))
    variance = float(np.var(signal))
    mean_abs = float(np.mean(np.abs(signal)))
    shape_fac = float(rms / mean_abs) if mean_abs != 0 else 0.0
    impulse_fac = float(peak / mean_abs) if mean_abs != 0 else 0.0
    return {
        "rms": rms, "peak": peak, "peak_to_peak": peak2peak,
        "kurtosis": kurt, "crest_factor": crest, "std": std_val,
        "skewness": skew, "variance": variance, "mean_abs": mean_abs,
        "shape_factor": shape_fac, "impulse_factor": impulse_fac,
    }

def process_test(test_id: int) -> pd.DataFrame:
    cfg = TEST_DIRS[test_id]
    data_dir = cfg["path"]
    n_ch = cfg["channels"]
    ch_names = CHANNEL_NAMES[n_ch]

    files = sorted(glob.glob(os.path.join(data_dir, "*")))
    files = [f for f in files if os.path.isfile(f)]
    print(f"\n[Test {test_id}] Processing {len(files)} files...")

    rows = []
    total = len(files)
    for i, fpath in enumerate(files):
        if (i + 1) % 500 == 0 or i == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  {i+1}/{total} ({pct:.0f}%)  {os.path.basename(fpath)}")

        try:
            arr = np.loadtxt(fpath, delimiter="\t")
        except Exception as e:
            continue

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        ts = parse_timestamp(fpath)
        row = {"timestamp": ts, "test_id": test_id, "file": os.path.basename(fpath)}

        for ch_idx, ch_name in enumerate(ch_names):
            if ch_idx < arr.shape[1]:
                feats = extract_features(arr[:, ch_idx])
                for feat_name, feat_val in feats.items():
                    row[f"{ch_name}_{feat_name}"] = feat_val
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    t0 = df["timestamp"].iloc[0]
    df["hours_elapsed"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0
    info = FAILURE_INFO[test_id]
    df["failed_bearing"] = info["failed_bearing"]
    df["failure_type"] = info["failure_type"]
    return df

def main():
    print("=" * 60)
    print("IMS Bearing Dataset Processing")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_dfs = []
    for test_id in [1, 2, 3]:
        out_path = os.path.join(OUT_DIR, f"bearings_test{test_id}_features.csv")
        if os.path.exists(out_path):
            print(f"  [Test {test_id}] Already processed: {out_path}")
            df = pd.read_csv(out_path)
        else:
            df = process_test(test_id)
            df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}  ({len(df)} rows x {len(df.columns)} cols)")
        all_dfs.append(df)

    combined_path = os.path.join(OUT_DIR, "bearings_all_features.csv")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined saved: {combined_path} ({len(combined)} rows)")

if __name__ == "__main__":
    main()
