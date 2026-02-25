"""
dsp_features.py — DSP con SOLO numpy (zero scipy)
numpy ha recipe p4a stabile; scipy spesso fallisce la compilazione ARM.

numpy.fft sostituisce scipy.fft
numpy signal processing sostituisce scipy.signal
"""

import numpy as np
from numpy.fft import rfft, irfft

# ── Costanti ──────────────────────────────────────────────────────────────────
N_FFT = 2048
HOP   = 512

_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,
                       2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,
                       2.54,4.75,3.98,2.69,3.34,3.17])

# Finestra di Hann precalcolata
_WIN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)


# ─────────────────────────────────────────────────────────────────────────────
# STFT con numpy.fft
# ─────────────────────────────────────────────────────────────────────────────
def _stft(y: np.ndarray, sr: int):
    """Ritorna (S, freqs): magnitudine shape (n_freq, n_frames)."""
    frames = []
    for i in range(0, len(y) - N_FFT, HOP):
        frame = y[i:i + N_FFT] * _WIN
        frames.append(frame)
    if not frames:
        frames = [np.zeros(N_FFT)]
    S_complex = np.array([rfft(f) for f in frames]).T   # (n_freq, n_frames)
    S = np.abs(S_complex)
    freqs = np.linspace(0, sr / 2, S.shape[0])
    return S, freqs


# ─────────────────────────────────────────────────────────────────────────────
# Feature singole
# ─────────────────────────────────────────────────────────────────────────────
def _rms(y):
    rms_frames = np.array([
        np.sqrt(np.mean(y[i:i+N_FFT]**2))
        for i in range(0, len(y)-N_FFT, HOP)
    ])
    if len(rms_frames) == 0:
        return 0.055, 0.01, 3.0
    m = float(np.mean(rms_frames))
    return m, float(np.std(rms_frames)), float(np.max(rms_frames)/(m+1e-8))


def _centroid(S, freqs):
    mag = S.sum(axis=0) + 1e-8
    return float(np.mean((freqs[:,None]*S).sum(axis=0)/mag))


def _rolloff(S, freqs, pct=0.85):
    cum = np.cumsum(S, axis=0)
    tot = cum[-1,:] + 1e-8
    out = []
    for t in range(S.shape[1]):
        idx = np.searchsorted(cum[:,t], pct*tot[t])
        out.append(freqs[min(idx, len(freqs)-1)])
    return float(np.mean(out))


def _bandwidth(S, freqs, centroid):
    mag = S.sum(axis=0) + 1e-8
    dev = (freqs[:,None]-centroid)**2
    return float(np.mean(np.sqrt((dev*S).sum(axis=0)/mag)))


def _zcr(y):
    rates = [np.mean(np.abs(np.diff(np.sign(y[i:i+N_FFT])))/2)
             for i in range(0, len(y)-N_FFT, HOP)]
    return float(np.mean(rates)) if rates else 0.065


def _contrast(S, sr, n_bands=7):
    n_freq = S.shape[0]
    edges  = np.logspace(np.log10(200), np.log10(sr/2), n_bands+1)
    fbins  = np.linspace(0, sr/2, n_freq)
    out = []
    for i in range(n_bands):
        idx = np.where((fbins>=edges[i]) & (fbins<edges[i+1]))[0]
        if len(idx) == 0:
            out.append(0.0); continue
        band = S[idx,:]
        out.append(float(np.mean(
            np.percentile(band,95,axis=0) - np.percentile(band,5,axis=0) + 1e-8
        )))
    return np.array(out)


# ── Mel filterbank ────────────────────────────────────────────────────────────
_MEL_CACHE = {}

def _mel_fb(sr, n_freq, n_mels=128):
    key = (sr, n_freq, n_mels)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    mel_min = 2595*np.log10(1+20/700)
    mel_max = 2595*np.log10(1+(sr/2)/700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels+2)
    hz_pts  = 700*(10**(mel_pts/2595)-1)
    n_fft2  = (n_freq-1)*2
    bins    = np.floor((n_fft2+1)*hz_pts/sr).astype(int)
    fb      = np.zeros((n_mels, n_freq))
    for m in range(1, n_mels+1):
        lo,cn,hi = bins[m-1],bins[m],bins[m+1]
        for k in range(lo, cn):
            if cn!=lo: fb[m-1,k]=(k-lo)/(cn-lo)
        for k in range(cn, hi):
            if hi!=cn: fb[m-1,k]=(hi-k)/(hi-cn)
    _MEL_CACHE[key] = fb
    return fb


def _mel_spec(S, sr, n_mels=128):
    fb = _mel_fb(sr, S.shape[0], n_mels)
    return fb @ S


def _mfcc(S, sr, n_mfcc=13):
    mel = _mel_spec(S, sr)
    log_mel = np.log(mel + 1e-8)
    # DCT-II con numpy
    N = log_mel.shape[0]
    n = np.arange(N)
    k = np.arange(n_mfcc)[:,None]
    dct_mat = np.cos(np.pi/N * k * (n + 0.5))  # (n_mfcc, N)
    coeffs  = dct_mat @ log_mel                  # (n_mfcc, n_frames)
    return coeffs.mean(axis=1)


def _chroma(S, sr):
    n_freq  = S.shape[0]
    freqs   = np.linspace(0, sr/2, n_freq)
    chroma  = np.zeros((12, S.shape[1]))
    for b, f in enumerate(freqs):
        if f < 16.35: continue
        try:
            pc = int(round(12*np.log2(f/16.35))) % 12
            chroma[pc] += S[b]
        except Exception:
            pass
    norm = chroma.sum(axis=0,keepdims=True)+1e-8
    chroma /= norm
    return chroma.mean(axis=1)


def _onset(S, sr):
    mel  = _mel_spec(S, sr, n_mels=64)
    diff = np.diff(mel, axis=1)
    env  = np.maximum(0, diff).sum(axis=0)
    return float(np.mean(env)), float(np.std(env)), env


def _tempo(y, sr, onset_env):
    fps = sr / HOP
    n   = len(onset_env)
    if n < 4: return 120.0

    lag_min = max(int(fps*60/200), 1)
    lag_max = min(int(fps*60/60), n-1)
    if lag_min >= lag_max: return 120.0

    # Autocorrelazione via FFT (numpy)
    pad  = np.zeros(2*n)
    pad[:n] = onset_env - onset_env.mean()
    sp   = rfft(pad)
    acf  = irfft(sp * np.conj(sp))[:n].real
    lag  = lag_min + int(np.argmax(acf[lag_min:lag_max+1]))
    bpm  = 60.0*fps/lag
    if bpm > 160: bpm /= 2
    return float(np.clip(bpm, 60.0, 200.0))


def _mode(chroma):
    bm, bn = 0.0, 0.0
    if np.std(chroma) > 1e-6:
        bm, bn = -np.inf, -np.inf
        for shift in range(12):
            m = float(np.corrcoef(np.roll(_KS_MAJOR,shift), chroma)[0,1])
            n = float(np.corrcoef(np.roll(_KS_MINOR,shift), chroma)[0,1])
            if np.isfinite(m) and m>bm: bm=m
            if np.isfinite(n) and n>bn: bn=n
        if not np.isfinite(bm): bm=0.0
        if not np.isfinite(bn): bn=0.0
    return (1 if bm>=bn else 0), float(abs(bm-bn))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(y: np.ndarray, sr: int) -> dict:
    S, freqs = _stft(y, sr)

    rms_m, rms_s, dyn  = _rms(y)
    centroid            = _centroid(S, freqs)
    rolloff             = _rolloff(S, freqs)
    bandwidth           = _bandwidth(S, freqs, centroid)
    zcr                 = _zcr(y)
    contrast            = _contrast(S, sr)
    contrast_ratio      = float(np.mean(contrast[4:])/(np.mean(contrast[:3])+1e-8))
    mfcc                = _mfcc(S, sr, 13)
    chroma_vec          = _chroma(S, sr)
    chroma_std          = float(np.std(chroma_vec))
    mode_maj, mode_str  = _mode(chroma_vec)
    onset_m, onset_s, env = _onset(S, sr)
    tempo               = _tempo(y, sr, env)
    duration            = float(len(y)/sr)

    raw = {
        "tempo": tempo, "rms": rms_m, "rms_std": rms_s,
        "dynamic_range": dyn, "centroid": centroid, "rolloff": rolloff,
        "bandwidth": bandwidth, "zcr": zcr,
        "contrast": contrast.tolist(), "contrast_ratio": contrast_ratio,
        "mfcc": mfcc.tolist(), "chroma": chroma_vec.tolist(),
        "chroma_std": chroma_std, "mode_major": mode_maj,
        "mode_strength": mode_str, "onset_mean": onset_m,
        "onset_std": onset_s, "duration": duration,
    }
    return _sanitize(raw)


def _sf(x, d=0.0):
    try:
        v=float(x); return d if (np.isnan(v) or np.isinf(v)) else v
    except: return d

def _sanitize(f):
    defs = {"tempo":120.,"rms":.055,"rms_std":.01,"dynamic_range":3.,
            "centroid":2200.,"rolloff":4000.,"bandwidth":2500.,"zcr":.065,
            "contrast_ratio":1.,"chroma_std":.08,"mode_major":1,
            "mode_strength":.1,"onset_mean":1.8,"onset_std":.8,"duration":180.}
    for k,d in defs.items():
        if k in f: f[k]=_sf(f[k],d)
    for k in ("mfcc","chroma","contrast"):
        if k in f and isinstance(f[k],list):
            f[k]=[_sf(v,0.) for v in f[k]]
    return f
