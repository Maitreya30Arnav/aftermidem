import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, pearsonr


def autocorr(x, max_lag=50):
    return np.array([
        np.corrcoef(x[:-lag], x[lag:])[0,1]
        for lag in range(1, max_lag)
    ])


def generate_synthetic_and_metrics(df, start_row, end_row):

    t_full = df["Relative_ms"].values
    x_full = df["Current_A"].values.astype(float)

    # -----------------------------
    # Threshold detection
    # -----------------------------
    hist, bins = np.histogram(x_full, bins=100)
    peaks, _ = find_peaks(hist)

    if len(peaks) >= 2:
        peak_vals = bins[peaks]
        thr = np.mean(sorted(peak_vals)[:2])
    else:
        thr = np.percentile(x_full, 45)

    state_full = (x_full > thr).astype(int)

    # -----------------------------
    # Extract blocks
    # -----------------------------
    on_blocks, off_blocks = [], []

    start = 0
    for i in range(1, len(state_full)):
        if state_full[i] != state_full[i-1]:
            block = x_full[start:i]
            if state_full[i-1] == 1:
                on_blocks.append(block)
            else:
                off_blocks.append(block)
            start = i

    block = x_full[start:]
    if state_full[-1] == 1:
        on_blocks.append(block)
    else:
        off_blocks.append(block)

    # -----------------------------
    # Generate synthetic
    # -----------------------------
    N = end_row - start_row
    rng = np.random.default_rng()

    syn = []
    cur_state = state_full[start_row]

    acf_lag1 = np.corrcoef(x_full[:-1], x_full[1:])[0,1]
    phi = np.clip(acf_lag1, 0.7, 0.98)

    while len(syn) < N:

        if cur_state == 1:
            block = on_blocks[rng.integers(len(on_blocks))].copy()
        else:
            block = off_blocks[rng.integers(len(off_blocks))].copy()

        L = len(block)

        sigma = np.std(block) * 0.03
        noise = np.zeros(L)

        for i in range(1, L):
            noise[i] = phi * noise[i-1] + rng.normal(0, sigma)

        block = block + noise

        # Smooth transition
        if len(syn) > 10:
            fade = min(10, L)
            for k in range(fade):
                alpha = k / fade
                block[k] = (1 - alpha) * syn[-1] + alpha * block[k]

        syn.extend(block)
        cur_state = 1 - cur_state

    syn = np.array(syn[:N])

    x_real = x_full[start_row:end_row]
    t_real = t_full[start_row:end_row]

    # =====================================================
    # 🔥 NEW: POWER CALCULATION
    # =====================================================
    V = 25  # constant voltage

    power_real = V * x_real
    power_syn  = V * syn

    # =====================================================
    # METRICS
    # =====================================================
    rms_real = np.sqrt(np.mean(x_real**2))
    rms_syn  = np.sqrt(np.mean(syn**2))
    rms_error = abs(rms_real - rms_syn) / rms_real * 100

    dt = np.median(np.diff(t_real))
    fs = 1000 / dt

    f_real, psd_real = welch(x_real, fs=fs, nperseg=1024)
    f_syn,  psd_syn  = welch(syn, fs=fs, nperseg=1024)

    psd_peak_error = abs(np.max(psd_real) - np.max(psd_syn)) / np.max(psd_real) * 100

    acf_real = autocorr(x_real)
    acf_syn  = autocorr(syn)

    acf_corr, _ = pearsonr(acf_real, acf_syn)

    kurt_real = kurtosis(x_real)
    kurt_syn  = kurtosis(syn)
    kurt_error = abs(kurt_real - kurt_syn) / abs(kurt_real) * 100

    metrics = {
        "RMS % Error": rms_error,
        "PSD Peak % Difference": psd_peak_error,
        "ACF Correlation": acf_corr,
        "Kurtosis % Error": kurt_error
    }

    return (
        x_real, syn, t_real,
        f_real, psd_real, f_syn, psd_syn,
        acf_real, acf_syn,
        power_real, power_syn,   # 👈 NEW
        metrics
    )