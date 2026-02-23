"""
dsp_features.py — Estrazione feature audio con scipy/numpy puro
Sostituisce interamente librosa con implementazioni equivalenti.

Funzioni esportate (stessa interfaccia di analyze_mp3.py originale):
  extract_features(y, sr) → dict
"""

import numpy as np
from scipy import signal
from scipy.fft import dct


# ─────────────────────────────────────────────────────────────────────────────
# Costanti Krumhansl-Schmuckler (identiche all'originale)
# ─────────────────────────────────────────────────────────────────────────────
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ─────────────────────────────────────────────────────────────────────────────
# Parametri STFT
# ─────────────────────────────────────────────────────────────────────────────
N_FFT     = 2048
HOP       = 512
WIN       = np.hanning(N_FFT)


def _stft_magnitude(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Ritorna (S, freqs):
      S     = magnitudine STFT  shape (n_freqs, n_frames)
      freqs = array Hz per ogni bin
    """
    _, _, Zxx = signal.stft(y, fs=sr, window=WIN,
                             nperseg=N_FFT, noverlap=N_FFT - HOP)
    S     = np.abs(Zxx)                         # (n_freqs, n_frames)
    freqs = np.linspace(0, sr / 2, S.shape[0])
    return S, freqs


# ─────────────────────────────────────────────────────────────────────────────
# Feature individuali
# ─────────────────────────────────────────────────────────────────────────────

def _rms(y: np.ndarray) -> tuple[float, float, float]:
    """Ritorna (rms_mean, rms_std, dynamic_range)."""
    hop = HOP
    frames = [y[i:i+N_FFT] for i in range(0, len(y) - N_FFT, hop)]
    if not frames:
        return 0.055, 0.01, 3.0
    rms_frames = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    mean = float(np.mean(rms_frames))
    std  = float(np.std(rms_frames))
    dyn  = float(np.max(rms_frames) / (mean + 1e-8))
    return mean, std, dyn


def _spectral_centroid(S: np.ndarray, freqs: np.ndarray) -> float:
    mag_sum = S.sum(axis=0) + 1e-8
    return float(np.mean((freqs[:, None] * S).sum(axis=0) / mag_sum))


def _spectral_rolloff(S: np.ndarray, freqs: np.ndarray,
                       pct: float = 0.85) -> float:
    cumsum = np.cumsum(S, axis=0)
    total  = cumsum[-1, :] + 1e-8
    frames = []
    for t in range(S.shape[1]):
        idx = np.searchsorted(cumsum[:, t], pct * total[t])
        idx = min(idx, len(freqs) - 1)
        frames.append(freqs[idx])
    return float(np.mean(frames))


def _spectral_bandwidth(S: np.ndarray, freqs: np.ndarray,
                         centroid: float) -> float:
    mag_sum = S.sum(axis=0) + 1e-8
    dev     = (freqs[:, None] - centroid) ** 2
    bw      = np.sqrt((dev * S).sum(axis=0) / mag_sum)
    return float(np.mean(bw))


def _zcr(y: np.ndarray) -> float:
    hops = [y[i:i+N_FFT] for i in range(0, len(y) - N_FFT, HOP)]
    if not hops:
        return 0.065
    rates = [np.mean(np.abs(np.diff(np.sign(h))) / 2) for h in hops]
    return float(np.mean(rates))


def _spectral_contrast(S: np.ndarray, sr: int, n_bands: int = 7) -> np.ndarray:
    """
    Contrasto spettrale per n_bands bande (come librosa).
    Ritorna array shape (n_bands,).
    """
    n_freq  = S.shape[0]
    band_edges = np.logspace(np.log10(200), np.log10(sr / 2),
                              n_bands + 1)
    freq_bins  = np.linspace(0, sr / 2, n_freq)
    contrast   = []
    for i in range(n_bands):
        lo  = band_edges[i]
        hi  = band_edges[i + 1]
        idx = np.where((freq_bins >= lo) & (freq_bins < hi))[0]
        if len(idx) == 0:
            contrast.append(0.0)
            continue
        band   = S[idx, :]
        peaks  = np.percentile(band, 95, axis=0)
        valleys = np.percentile(band, 5, axis=0)
        contrast.append(float(np.mean(peaks - valleys + 1e-8)))
    return np.array(contrast)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 128) -> np.ndarray:
    """Matrice filtri mel  shape (n_mels, n_fft//2+1)."""
    mel_min  = 2595 * np.log10(1 + 20 / 700)
    mel_max  = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
    bins     = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freq   = n_fft // 2 + 1
    filters  = np.zeros((n_mels, n_freq))
    for m in range(1, n_mels + 1):
        lo, cen, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, cen):
            if cen != lo:
                filters[m-1, k] = (k - lo) / (cen - lo)
        for k in range(cen, hi):
            if hi != cen:
                filters[m-1, k] = (hi - k) / (hi - cen)
    return filters


_MEL_CACHE: dict = {}

def _mel_spec(S: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    key = (sr, S.shape[0], n_mels)
    if key not in _MEL_CACHE:
        _MEL_CACHE[key] = _mel_filterbank(sr, (S.shape[0] - 1) * 2, n_mels)
    fb = _MEL_CACHE[key]
    # Adatta dimensione filtri
    n_freq = min(S.shape[0], fb.shape[1])
    return fb[:, :n_freq] @ S[:n_freq, :]


def _mfcc(S: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    mel = _mel_spec(S, sr)
    log_mel = np.log(mel + 1e-8)
    coeffs  = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc, :]
    return coeffs.mean(axis=1)


def _chroma(S: np.ndarray, sr: int) -> np.ndarray:
    """12 chroma bins per ottava, media sulle frame."""
    n_freq  = S.shape[0]
    freqs   = np.linspace(0, sr / 2, n_freq)
    chroma  = np.zeros((12, S.shape[1]))
    for b, f in enumerate(freqs):
        if f < 16.35:
            continue
        try:
            midi = 12 * np.log2(f / 16.35)
            pc   = int(round(midi)) % 12
            chroma[pc] += S[b]
        except (ValueError, FloatingPointError):
            pass
    norm = chroma.sum(axis=0, keepdims=True) + 1e-8
    chroma /= norm
    return chroma.mean(axis=1)


def _onset_strength(S: np.ndarray, sr: int) -> tuple[float, float]:
    """(onset_mean, onset_std) usando la variazione frame-to-frame del mel."""
    mel = _mel_spec(S, sr, n_mels=64)
    # Positive half-wave rectification of mel diff
    diff = np.diff(mel, axis=1)
    onset = np.maximum(0, diff).sum(axis=0)
    return float(np.mean(onset)), float(np.std(onset))


def _beat_tempo(y: np.ndarray, sr: int,
                onset_env: np.ndarray = None) -> float:
    """
    Stima BPM via autocorrelazione dell'onset envelope.
    Range cercato: 60-200 BPM.
    """
    if onset_env is None:
        # Calcola onset semplice da energia frame
        hop = HOP
        frames = [y[i:i+N_FFT] for i in range(0, len(y) - N_FFT, hop)]
        onset_env = np.array([np.sqrt(np.mean(f**2)) for f in frames])

    fps = sr / HOP   # frames per second

    # Autocorrelazione
    n = len(onset_env)
    if n < 2:
        return 120.0

    # Lag range per 60-200 BPM
    lag_min = int(fps * 60 / 200)  # 200 BPM
    lag_max = int(fps * 60 / 60)   # 60 BPM
    lag_max = min(lag_max, n - 1)
    lag_min = max(lag_min, 1)

    if lag_min >= lag_max:
        return 120.0

    # Autocorrelazione via FFT
    padded = np.zeros(2 * n)
    padded[:n] = onset_env - onset_env.mean()
    spectrum   = np.fft.rfft(padded)
    acf        = np.fft.irfft(spectrum * np.conj(spectrum))[:n]

    if lag_max >= len(acf):
        lag_max = len(acf) - 1

    best_lag = lag_min + np.argmax(acf[lag_min:lag_max + 1])
    bpm      = 60.0 * fps / best_lag

    # Evita armoniche: se BPM > 160 prova metà
    if bpm > 160:
        bpm2 = bpm / 2
        if 60 <= bpm2 <= 160:
            bpm = bpm2

    return float(np.clip(bpm, 60.0, 200.0))


def _mode_major(chroma: np.ndarray) -> tuple[int, float]:
    """Ritorna (mode_major 0/1, mode_strength)."""
    best_major, best_minor = 0.0, 0.0
    if np.std(chroma) > 1e-6:
        best_major, best_minor = -np.inf, -np.inf
        for shift in range(12):
            m = float(np.corrcoef(np.roll(_KS_MAJOR, shift), chroma)[0, 1])
            n = float(np.corrcoef(np.roll(_KS_MINOR, shift), chroma)[0, 1])
            if np.isfinite(m) and m > best_major:
                best_major = m
            if np.isfinite(n) and n > best_minor:
                best_minor = n
        if not np.isfinite(best_major): best_major = 0.0
        if not np.isfinite(best_minor): best_minor = 0.0

    mode_major    = 1 if best_major >= best_minor else 0
    mode_strength = float(abs(best_major - best_minor))
    return mode_major, mode_strength


# ─────────────────────────────────────────────────────────────────────────────
# Entry point principale
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Estrae tutte le feature audio da (y, sr).
    Interfaccia compatibile con analyze_mp3.extract_features().

    Args:
        y  : array float32 mono PCM
        sr : sample rate Hz

    Returns:
        dict con le stesse chiavi della versione librosa
    """
    # STFT condivisa
    S, freqs = _stft_magnitude(y, sr)

    # Energia
    rms_mean, rms_std, dyn_range = _rms(y)

    # Spettro
    centroid  = _spectral_centroid(S, freqs)
    rolloff   = _spectral_rolloff(S, freqs)
    bandwidth = _spectral_bandwidth(S, freqs, centroid)
    zcr       = _zcr(y)

    # Contrasto
    contrast       = _spectral_contrast(S, sr)
    contrast_ratio = float(
        np.mean(contrast[4:]) / (np.mean(contrast[:3]) + 1e-8)
    )

    # MFCC
    mfcc = _mfcc(S, sr, n_mfcc=13)

    # Chroma
    chroma_vec = _chroma(S, sr)
    chroma_std = float(np.std(chroma_vec))

    # Modo tonale
    mode_maj, mode_str = _mode_major(chroma_vec)

    # Onset + BPM
    onset_mean, onset_std = _onset_strength(S, sr)

    # Costruisci onset env per beat tracking
    mel    = _mel_spec(S, sr, n_mels=64)
    diff   = np.diff(mel, axis=1)
    o_env  = np.maximum(0, diff).sum(axis=0)
    tempo  = _beat_tempo(y, sr, o_env)

    # Durata
    duration = float(len(y) / sr)

    raw = {
        "tempo":          tempo,
        "rms":            rms_mean,
        "rms_std":        rms_std,
        "dynamic_range":  dyn_range,
        "centroid":       centroid,
        "rolloff":        rolloff,
        "bandwidth":      bandwidth,
        "zcr":            zcr,
        "contrast":       contrast.tolist(),
        "contrast_ratio": contrast_ratio,
        "mfcc":           mfcc.tolist(),
        "chroma":         chroma_vec.tolist(),
        "chroma_std":     chroma_std,
        "mode_major":     mode_maj,
        "mode_strength":  mode_str,
        "onset_mean":     onset_mean,
        "onset_std":      onset_std,
        "duration":       duration,
    }

    # Sanifica NaN/Inf
    return _sanitize(raw)


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def _sanitize(f: dict) -> dict:
    scalar_defaults = {
        "tempo": 120.0, "rms": 0.055, "rms_std": 0.01,
        "dynamic_range": 3.0, "centroid": 2200.0, "rolloff": 4000.0,
        "bandwidth": 2500.0, "zcr": 0.065, "contrast_ratio": 1.0,
        "chroma_std": 0.08, "mode_major": 1, "mode_strength": 0.1,
        "onset_mean": 1.8, "onset_std": 0.8, "duration": 180.0,
    }
    for k, d in scalar_defaults.items():
        if k in f:
            f[k] = _safe_float(f[k], d)
    for k in ("mfcc", "chroma", "contrast"):
        if k in f and isinstance(f[k], list):
            f[k] = [_safe_float(v, 0.0) for v in f[k]]
    return f
