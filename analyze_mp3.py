"""
analyze_mp3.py — Orchestrazione analisi audio (versione Android)
Path e logging sicuri per Android: nessun FileHandler a livello modulo.
"""

import os
import json
import hashlib
import logging
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from audio_decode import load_audio
from dsp_features import extract_features as _dsp_extract

# ---------------------------------------------------------------------------
# Path sicuro per Android (usa CWD che p4a imposta sulla dir dati dell'app)
# ---------------------------------------------------------------------------
def _get_data_dir() -> Path:
    """Restituisce una directory scrivibile su qualsiasi piattaforma."""
    # p4a imposta CWD sulla dir privata dell'app (/data/data/<pkg>/files/)
    # che è sempre scrivibile. Su desktop usa la home.
    cwd = Path.cwd()
    if os.access(str(cwd), os.W_OK):
        return cwd
    # Fallback: directory temporanea
    import tempfile
    return Path(tempfile.gettempdir())

DATA_DIR   = _get_data_dir()
CACHE_FILE = DATA_DIR / "music_cache.json"
LOG_FILE   = DATA_DIR / "music_analyzer.log"

# ---------------------------------------------------------------------------
# Logging — FileHandler solo se la directory è scrivibile (mai a livello modulo!)
# ---------------------------------------------------------------------------
def _setup_logging():
    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(str(LOG_FILE), encoding="utf-8"))
    except Exception:
        pass  # Non crashare se il file non è apribile
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

_setup_logging()
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_VERSION = 6

_FEATURE_DEFAULTS = {
    "rms_std": 0.01, "dynamic_range": 3.0,
    "chroma_std": 0.08, "contrast_ratio": 1.0, "mode_strength": 0.1,
}


def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning("Cache corrotta: %s", e)
        return {}
    if data.get("__version__") != _CACHE_VERSION:
        return {"__version__": _CACHE_VERSION}
    return data


def _save_cache(cache: dict):
    cache["__version__"] = _CACHE_VERSION
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Impossibile salvare cache: %s", e)


def _ensure_defaults(feats: dict) -> dict:
    for k, v in _FEATURE_DEFAULTS.items():
        feats.setdefault(k, v)
    return feats


def _file_hash(filepath: str) -> str:
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            h.update(f.read(65536))
    except OSError:
        h.update(filepath.encode())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(filepath: str) -> dict:
    y, sr = load_audio(filepath, duration=90.0)
    return _dsp_extract(y, sr)


# ─────────────────────────────────────────────────────────────────────────────
# Mood
# ─────────────────────────────────────────────────────────────────────────────
def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def _mood_sigmoid(x, scale=1.5):
    return float(1.0 / (1.0 + np.exp(-np.clip(x / scale, -10, 10))))


_MOOD_ANCHORS = {
    "tempo":      (115.0, 30.0),
    "rms":        (0.055, 0.035),
    "centroid":   (2200.0, 900.0),
    "onset_mean": (1.8,    1.0),
    "zcr":        (0.065,  0.035),
    "chroma_std": (0.085,  0.03),
}


def _norm_abs(value, key):
    mean, std = _MOOD_ANCHORS[key]
    return float(np.clip((value - mean) / std, -3.0, 3.0))


def compute_mood(features_list: list) -> list:
    if not features_list:
        return features_list
    raw_a_list, raw_v_list = [], []
    for f in features_list:
        t_n  = _norm_abs(f["tempo"],      "tempo")
        r_n  = _norm_abs(f["rms"],        "rms")
        o_n  = _norm_abs(f["onset_mean"], "onset_mean")
        c_n  = _norm_abs(f["centroid"],   "centroid")
        z_n  = _norm_abs(f["zcr"],        "zcr")
        ch_n = _norm_abs(f["chroma_std"], "chroma_std")
        raw_a = 0.30*r_n + 0.25*o_n + 0.20*t_n + 0.15*z_n + 0.10*c_n
        mode_scaled = f["mode_major"]*2.0 - 1.0
        mode_w = 0.35 * f.get("mode_strength", 0.5)
        raw_v = (mode_scaled*(0.30+mode_w) + 0.20*c_n + 0.15*t_n
                 + 0.10*r_n - 0.10*ch_n)
        raw_a_list.append(raw_a)
        raw_v_list.append(raw_v)
    a_off = float(np.nanmedian(raw_a_list)) * 0.4
    v_off = float(np.nanmedian(raw_v_list)) * 0.4
    if np.isnan(a_off): a_off = 0.0
    if np.isnan(v_off): v_off = 0.0
    results = []
    for i, f in enumerate(features_list):
        results.append({
            **f,
            "arousal": _mood_sigmoid(raw_a_list[i] - a_off, scale=1.2),
            "valence": _mood_sigmoid(raw_v_list[i] - v_off, scale=1.2),
        })
    return results


def assign_mood(data: list) -> list:
    if not data: return data
    valences = np.array([d["valence"] for d in data], dtype=float)
    arousals = np.array([d["arousal"] for d in data], dtype=float)
    v_thresh = float(np.clip(np.nanmedian(valences), 0.40, 0.60))
    a_thresh = float(np.clip(np.nanmedian(arousals), 0.40, 0.60))
    if np.isnan(v_thresh): v_thresh = 0.5
    if np.isnan(a_thresh): a_thresh = 0.5
    _Q = {(True,True):"Energetic", (True,False):"Positive",
          (False,True):"Aggressive", (False,False):"Melancholic"}
    vm, am = 0.05, 0.05
    for d in data:
        v, a = d["valence"], d["arousal"]
        if   v >= v_thresh+vm and a >= a_thresh+am: d["mood"] = "Energetic"
        elif v >= v_thresh+vm and a <  a_thresh-am: d["mood"] = "Positive"
        elif v <  v_thresh-vm and a >= a_thresh+am: d["mood"] = "Aggressive"
        elif v <  v_thresh-vm and a <  a_thresh-am: d["mood"] = "Melancholic"
        else: d["mood"] = _Q[(v>=v_thresh, a>=a_thresh)]
        dist = ((v-v_thresh)**2 + (a-a_thresh)**2)**0.5
        d["mood_confidence"] = round(float(np.clip(dist/0.707, 0, 1)), 3)
        vs, as_ = v >= v_thresh, a >= a_thresh
        d["mood_alt"] = _Q[(not vs, as_)] if abs(v-v_thresh) < abs(a-a_thresh) \
                        else _Q[(vs, not as_)]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Genre
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid(x, center, steepness=5.0):
    arg = float(np.clip(-steepness*(x-center), -30, 30))
    return float(1/(1+np.exp(arg)))

def _bell(x, center, width):
    z = float(np.clip((x-center)/max(width,1e-9), -6, 6))
    return float(np.exp(-0.5*z*z))

def _compute_ds(data):
    def med(k): return float(np.nanmedian([d[k] for d in data]))
    def std(k):
        vals = np.array([d[k] for d in data], dtype=float)
        q75, q25 = np.nanpercentile(vals, [75, 25])
        return float(max((q75-q25)/1.349, 1e-9))
    return {
        "med_rms": med("rms"),          "std_rms": std("rms"),
        "med_centroid": med("centroid"),"std_centroid": std("centroid"),
        "med_bandwidth": med("bandwidth"),"std_bandwidth": std("bandwidth"),
        "med_zcr": med("zcr"),          "std_zcr": std("zcr"),
        "med_onset_mean": med("onset_mean"),"std_onset_mean": std("onset_mean"),
        "med_onset_std": med("onset_std"), "std_onset_std": std("onset_std"),
        "med_tempo": med("tempo"),      "std_tempo": std("tempo"),
        "med_dynrange": med("dynamic_range"),"std_dynrange": std("dynamic_range"),
        "med_chroma_std": med("chroma_std"),"std_chroma_std": std("chroma_std"),
        "med_contrast": float(np.nanmedian(
            [float(np.mean(d.get("contrast",[10.]))) for d in data])),
        "std_contrast": float(max(np.nanstd(
            [float(np.mean(d.get("contrast",[10.]))) for d in data]),1e-9)),
    }

def _zs(v, med, std, lo=-3, hi=3):
    return float(np.clip((v-med)/std, lo, hi))

def _score_genre(d, ds):
    z_rms = _zs(d["rms"],          ds["med_rms"],       ds["std_rms"])
    z_c   = _zs(d["centroid"],     ds["med_centroid"],  ds["std_centroid"])
    z_bw  = _zs(d["bandwidth"],    ds["med_bandwidth"], ds["std_bandwidth"])
    z_zcr = _zs(d["zcr"],          ds["med_zcr"],       ds["std_zcr"])
    z_om  = _zs(d["onset_mean"],   ds["med_onset_mean"],ds["std_onset_mean"])
    z_os  = _zs(d["onset_std"],    ds["med_onset_std"], ds["std_onset_std"])
    z_dyn = _zs(d["dynamic_range"],ds["med_dynrange"],  ds["std_dynrange"])
    z_chs = _zs(d["chroma_std"],   ds["med_chroma_std"],ds["std_chroma_std"])
    cm    = _safe_float(np.mean(d.get("contrast",[10.])))
    z_co  = _zs(cm, ds["med_contrast"], ds["std_contrast"])
    t, maj = _safe_float(d["tempo"]), float(d["mode_major"])
    hi_ = lambda z: _sigmoid(z, 0.0,  1.5)
    lo_ = lambda z: _sigmoid(z, 0.0, -1.5)
    return {
        "Electronic": float(np.mean([_bell(t,135,22), hi_(z_c),  hi_(z_zcr), lo_(z_dyn), lo_(z_os)])),
        "Rock":       float(np.mean([hi_(z_rms), hi_(z_bw), hi_(z_om), hi_(z_os), _bell(t,128,30)])),
        "HipHop":     float(np.mean([_bell(t,90,18), hi_(z_zcr), lo_(z_os), lo_(z_c),  lo_(z_co)])),
        "Acoustic":   float(np.mean([lo_(z_rms), hi_(z_dyn), lo_(z_bw), lo_(z_c),  lo_(z_om)])),
        "Pop":        float(np.mean([_bell(t,118,16), maj*0.7+0.3, _bell(z_rms,0,.8), _bell(z_c,0,.8), lo_(z_chs)])),
        "Jazz":       float(np.mean([hi_(z_co), hi_(z_chs), lo_(z_zcr), hi_(z_dyn), _bell(t,112,38)])),
        "Classical":  float(np.mean([hi_(z_co), hi_(z_dyn), lo_(z_om), lo_(z_zcr), hi_(z_bw), lo_(z_rms)])),
    }

def assign_genre(data: list) -> list:
    if not data: return data
    if len(data) < 3:
        for d in data:
            d["genre"] = "Pop"; d["genre_scores"] = {}; d["genre_confidence"] = 0.0
        return data
    ds = _compute_ds(data)
    for d in data:
        scores  = {k: float(np.clip(v,0,1)) for k,v in _score_genre(d,ds).items()}
        sorted_ = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        best, bscore = sorted_[0]
        sec,  sscore = sorted_[1]
        d["genre"]            = best
        d["genre_scores"]     = {k: round(v,3) for k,v in scores.items()}
        d["genre_confidence"] = round(bscore - sscore, 3)
        d["genre_alt"]        = sec if (bscore-sscore) < 0.05 else None
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Anno da tag ID3
# ─────────────────────────────────────────────────────────────────────────────
def _extract_year(filepath: str) -> Optional[int]:
    try:
        import re
        from mutagen.easyid3 import EasyID3
        from mutagen.mp3 import MP3
        try:
            audio = EasyID3(filepath)
            if "date" in audio:
                m = re.search(r"\d{4}", audio["date"][0])
                if m: return int(m.group())
        except Exception:
            pass
        try:
            audio = MP3(filepath)
            if "TDRC" in audio:
                m = re.search(r"\d{4}", str(audio["TDRC"]))
                if m: return int(m.group())
        except Exception:
            pass
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# scan_folders
# ─────────────────────────────────────────────────────────────────────────────
def scan_folders(folder_paths: list,
                 progress_callback: Optional[Callable] = None) -> list:
    cache = _load_cache()
    mp3_files = []
    for folder in folder_paths:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".mp3"):
                    mp3_files.append(os.path.join(root, file))

    total = len(mp3_files)
    if total == 0:
        log.warning("Nessun MP3 trovato.")
        return []

    raw_features, songs_meta = [], []
    cache_hits = 0

    for idx, full_path in enumerate(mp3_files):
        filename = os.path.basename(full_path)
        if progress_callback:
            progress_callback(idx+1, total, filename)

        file_key = _file_hash(full_path)

        if file_key in cache and not file_key.startswith("__"):
            feats = _ensure_defaults(cache[file_key])
            raw_features.append(feats)
            songs_meta.append({"title": filename, "path": full_path,
                                "year": _extract_year(full_path)})
            cache_hits += 1
            continue

        try:
            feats = extract_features(full_path)
            cache[file_key] = feats
            raw_features.append(feats)
            songs_meta.append({"title": filename, "path": full_path,
                                "year": _extract_year(full_path)})
            log.info("Analizzato [%d/%d]: %s", idx+1, total, filename)
        except Exception:
            log.error("Errore %s:\n%s", full_path, traceback.format_exc())

    _save_cache(cache)
    log.info("Scansione: %d brani (%d cache, %d nuovi).",
              len(raw_features), cache_hits, len(raw_features)-cache_hits)

    if not raw_features:
        return []

    enriched = compute_mood(raw_features)
    for i in range(len(enriched)):
        songs_meta[i].update(enriched[i])
    songs_meta = assign_mood(songs_meta)
    songs_meta = assign_genre(songs_meta)
    return songs_meta
