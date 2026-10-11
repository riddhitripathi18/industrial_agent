"""
extract_bearing_frequency_features.py
Extracts raw FFT defect peaks, envelope spectrum defect harmonics (BPFO, 2*BPFO, BPFI, BSF),
and spectral kurtosis from raw IMS vibration waveforms across all 3 endurance runs.
"""

import os, time, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import signal

BASE = r"C:\Projects\industrial_agent\bearings_dataset\IMS_extracted"
OUT_CSV = r"c:\Users\riddh\OneDrive\Desktop\industrial_agent\bearings_freq_raw_features.csv"

TEST_DIRS = {
    1: {"path": os.path.join(BASE, "1st_test"), "channels": 8},
    2: {"path": os.path.join(BASE, "2nd_test"), "channels": 4},
    3: {"path": os.path.join(BASE, "4th_test", "txt"), "channels": 4},
}

def process_single_file(args):
    fpath, test_id, n_ch = args
    fname = os.path.basename(fpath)
    try:
        df = pd.read_csv(fpath, sep='\t', header=None, engine='c')
        data = df.values
        if data.shape[0] < 20000:
            return None
        
        fs = 20000.0
        n_samples = data.shape[0]
        freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
        
        # Characteristic defect frequencies at 2000 RPM (fr = 33.33 Hz)
        # Rexnord ZA-2115: BPFO ~ 236.4 Hz, 2*BPFO ~ 472.8 Hz, BPFI ~ 296.9 Hz, BSF ~ 139.9 Hz
        bpfo_mask  = (freqs >= 231.0) & (freqs <= 242.0)
        bpfo2_mask = (freqs >= 467.0) & (freqs <= 479.0)
        bpfi_mask  = (freqs >= 291.0) & (freqs <= 303.0)
        bsf_mask   = (freqs >= 135.0) & (freqs <= 145.0)
        
        rec = {"file": fname, "test_id": test_id}
        
        # Map channels to bearings
        # Test 1 (8 channels): b1:(0,1), b2:(2,3), b3:(4,5), b4:(6,7)
        # Test 2 & 3 (4 channels): b1:0, b2:1, b3:2, b4:3
        for b_idx in [1, 2, 3, 4]:
            b_name = f"Bearing{b_idx}"
            if n_ch == 8:
                ch_x = (b_idx - 1) * 2
                ch_y = ch_x + 1
                sig = data[:, ch_x]  # Primary x-axis
            else:
                ch = b_idx - 1
                sig = data[:, ch]
            
            s_zero = sig - np.mean(sig)
            
            # 1. Direct FFT
            fft_mag = np.abs(np.fft.rfft(s_zero)) * (2.0 / n_samples)
            fft_bpfo = float(np.max(fft_mag[bpfo_mask]))
            fft_bpfi = float(np.max(fft_mag[bpfi_mask]))
            fft_bsf  = float(np.max(fft_mag[bsf_mask]))
            
            # 2. Envelope spectrum via Hilbert transform
            env = np.abs(signal.hilbert(s_zero))
            env_zero = env - np.mean(env)
            env_mag = np.abs(np.fft.rfft(env_zero)) * (2.0 / n_samples)
            
            env_bpfo  = float(np.max(env_mag[bpfo_mask]))
            env_bpfo2 = float(np.max(env_mag[bpfo2_mask]))
            env_bpfi  = float(np.max(env_mag[bpfi_mask]))
            env_bsf   = float(np.max(env_mag[bsf_mask]))
            
            # 3. Spectral Kurtosis via STFT
            _, _, Zxx = signal.stft(s_zero, fs=fs, nperseg=256, noverlap=128)
            Z_mag = np.abs(Zxx)
            m2 = np.mean(Z_mag**2, axis=1) + 1e-12
            m4 = np.mean(Z_mag**4, axis=1)
            sk = (m4 / (m2**2)) - 2.0
            spec_kurt_mean = float(np.mean(sk))
            spec_kurt_max  = float(np.max(sk))
            
            rec[f"{b_name}_env_bpfo"]  = env_bpfo
            rec[f"{b_name}_env_bpfo2"] = env_bpfo2
            rec[f"{b_name}_env_bpfi"]  = env_bpfi
            rec[f"{b_name}_env_bsf"]   = env_bsf
            rec[f"{b_name}_fft_bpfo"]  = fft_bpfo
            rec[f"{b_name}_fft_bpfi"]  = fft_bpfi
            rec[f"{b_name}_fft_bsf"]   = fft_bsf
            rec[f"{b_name}_sk_mean"]   = spec_kurt_mean
            rec[f"{b_name}_sk_max"]    = spec_kurt_max
            
        return rec
    except Exception as e:
        return None

def main():
    print("=" * 60)
    print("Extracting frequency-domain features from raw IMS waveforms...")
    print("=" * 60)
    t0 = time.time()
    
    tasks = []
    for test_id in [1, 2, 3]:
        cfg = TEST_DIRS[test_id]
        p = cfg["path"]
        n_ch = cfg["channels"]
        files = sorted(os.listdir(p))
        files = [os.path.join(p, f) for f in files if os.path.isfile(os.path.join(p, f))]
        print(f"Test {test_id}: {len(files)} files to process (n_channels={n_ch})")
        for f in files:
            tasks.append((f, test_id, n_ch))
            
    print(f"\nTotal files queued across all tests: {len(tasks)}")
    n_workers = min(os.cpu_count() or 4, 16)
    print(f"Launching ProcessPoolExecutor with {n_workers} worker processes...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(process_single_file, tasks, chunksize=32))
        
    valid_results = [r for r in results if r is not None]
    dt = time.time() - t0
    print(f"\nExtraction complete! Successfully processed {len(valid_results)}/{len(tasks)} files in {dt:.1f} s ({dt/len(valid_results)*1000:.1f} ms/file)")
    
    df_out = pd.DataFrame(valid_results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved frequency-domain features to: {OUT_CSV} (shape: {df_out.shape})")

if __name__ == "__main__":
    main()
