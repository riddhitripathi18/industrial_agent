import os, glob, time
import numpy as np
import pandas as pd
from scipy import signal, stats
from concurrent.futures import ProcessPoolExecutor

BASE = r"C:\Projects\industrial_agent\bearings_dataset\IMS_extracted"

def process_file_feats(args):
    fpath, n_ch = args
    fname = os.path.basename(fpath)
    try:
        # Fast load
        data = np.fromfile(fpath, sep='\t').reshape(-1, n_ch)
        if data.shape[0] != 20480:
            return None
        
        fs = 20000.0
        n_samples = data.shape[0]
        freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
        
        # Characteristic frequencies (2000 RPM = 33.33 Hz shaft freq)
        # Rexnord ZA-2115: BPFO = 236.4 Hz, 2*BPFO = 472.8 Hz, BPFI = 296.9 Hz, BSF = 139.9 Hz
        bpfo_mask  = (freqs >= 231.0) & (freqs <= 242.0)
        bpfo2_mask = (freqs >= 467.0) & (freqs <= 479.0)
        bpfi_mask  = (freqs >= 291.0) & (freqs <= 303.0)
        bsf_mask   = (freqs >= 135.0) & (freqs <= 145.0)
        
        res = {"file": fname}
        
        for ch in range(n_ch):
            s = data[:, ch]
            # 1. Direct FFT spectrum
            s_zero = s - np.mean(s)
            fft_mag = np.abs(np.fft.rfft(s_zero)) * (2.0 / n_samples)
            
            # 2. Envelope spectrum via Hilbert transform
            env = np.abs(signal.hilbert(s_zero))
            env_zero = env - np.mean(env)
            env_mag = np.abs(np.fft.rfft(env_zero)) * (2.0 / n_samples)
            
            # Extract fault harmonic amplitudes from envelope spectrum
            env_bpfo  = float(np.max(env_mag[bpfo_mask]))
            env_bpfo2 = float(np.max(env_mag[bpfo2_mask]))
            env_bpfi  = float(np.max(env_mag[bpfi_mask]))
            env_bsf   = float(np.max(env_mag[bsf_mask]))
            
            # Raw FFT defect band energies
            fft_bpfo = float(np.max(fft_mag[bpfo_mask]))
            fft_bpfi = float(np.max(fft_mag[bpfi_mask]))
            
            # 3. Spectral Kurtosis via STFT
            # 256-point STFT -> 129 freq bins
            _, _, Zxx = signal.stft(s_zero, fs=fs, nperseg=256, noverlap=128)
            Z_mag = np.abs(Zxx)
            # Kurtosis across time slices for each frequency bin
            m2 = np.mean(Z_mag**2, axis=1) + 1e-12
            m4 = np.mean(Z_mag**4, axis=1)
            sk_per_bin = (m4 / (m2**2)) - 2.0
            mean_spec_kurt = float(np.mean(sk_per_bin))
            max_spec_kurt  = float(np.max(sk_per_bin))
            
            res[f"ch{ch}_env_bpfo"] = env_bpfo
            res[f"ch{ch}_env_bpfo2"] = env_bpfo2
            res[f"ch{ch}_env_bpfi"] = env_bpfi
            res[f"ch{ch}_env_bsf"] = env_bsf
            res[f"ch{ch}_fft_bpfo"] = fft_bpfo
            res[f"ch{ch}_fft_bpfi"] = fft_bpfi
            res[f"ch{ch}_spec_kurt_mean"] = mean_spec_kurt
            res[f"ch{ch}_spec_kurt_max"] = max_spec_kurt
            
        return res
    except Exception as e:
        return None

if __name__ == "__main__":
    t0 = time.time()
    files2 = sorted(glob.glob(os.path.join(BASE, '2nd_test', '*')))[:50]
    args_list = [(f, 4) for f in files2]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(process_file_feats, args_list))
    
    valid = [r for r in results if r is not None]
    dt = time.time() - t0
    print(f"Processed {len(valid)} files in {dt:.2f} s ({dt/len(valid)*1000:.1f} ms/file)")
    if valid:
        print("Sample keys:", list(valid[0].keys())[:10])
