"""
audio_decode.py — Decodifica MP3 → PCM float32 mono

Android: usa MediaExtractor + MediaCodec via pyjnius (Java API nativa)
Desktop: usa scipy.io.wavfile o soundfile come fallback per sviluppo/test
"""

import os
import numpy as np
from typing import Tuple


def load_audio(path: str, duration: float = 90.0) -> Tuple[np.ndarray, int]:
    """
    Carica un file MP3 e ritorna (y, sr) con:
      y  = array float32 mono, normalizzato in [-1, 1]
      sr = sample rate (Hz)

    Su Android usa MediaCodec; su desktop usa fallback.
    Analizza al massimo `duration` secondi (default 90s).
    """
    try:
        from jnius import autoclass
        return _load_android(path, duration)
    except ImportError:
        return _load_desktop(path, duration)


# ─────────────────────────────────────────────────────────────────────────────
# Android: MediaExtractor + MediaCodec
# ─────────────────────────────────────────────────────────────────────────────
def _load_android(path: str, max_dur: float) -> Tuple[np.ndarray, int]:
    from jnius import autoclass
    import array as pyarray

    MediaExtractor  = autoclass("android.media.MediaExtractor")
    MediaCodec      = autoclass("android.media.MediaCodec")
    MediaFormat     = autoclass("android.media.MediaFormat")
    BufferInfo      = autoclass("android.media.MediaCodec$BufferInfo")

    extractor = MediaExtractor()
    extractor.setDataSource(path)

    # Trova traccia audio
    audio_idx = -1
    fmt = None
    for i in range(extractor.getTrackCount()):
        f = extractor.getTrackFormat(i)
        mime = f.getString(MediaFormat.KEY_MIME)
        if mime and mime.startswith("audio/"):
            audio_idx = i
            fmt = f
            break
    if audio_idx < 0:
        raise RuntimeError(f"Nessuna traccia audio in {path}")

    extractor.selectTrack(audio_idx)
    sr       = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
    channels = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

    codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME))
    codec.configure(fmt, None, None, 0)
    codec.start()

    pcm_bytes = bytearray()
    info      = BufferInfo()
    TIMEOUT   = 10_000   # µs
    max_bytes = int(max_dur * sr * channels * 2)  # int16 = 2 bytes
    done      = False

    while not done and len(pcm_bytes) < max_bytes:
        # Feed input
        in_idx = codec.dequeueInputBuffer(TIMEOUT)
        if in_idx >= 0:
            buf   = codec.getInputBuffer(in_idx)
            n     = extractor.readSampleData(buf, 0)
            if n < 0:
                codec.queueInputBuffer(in_idx, 0, 0, 0,
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                done = True
            else:
                pts = extractor.getSampleTime()
                codec.queueInputBuffer(in_idx, 0, n, pts, 0)
                extractor.advance()

        # Drain output
        out_idx = codec.dequeueOutputBuffer(info, TIMEOUT)
        if out_idx >= 0:
            buf = codec.getOutputBuffer(out_idx)
            buf.limit(info.size)
            chunk = buf.array()[buf.arrayOffset():buf.arrayOffset() + info.size]
            pcm_bytes.extend(chunk)
            codec.releaseOutputBuffer(out_idx, False)
            if info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM:
                done = True

    codec.stop(); codec.release()
    extractor.release()

    # int16 → float32 mono
    arr = np.frombuffer(bytes(pcm_bytes), dtype=np.int16).astype(np.float32)
    arr /= 32768.0
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)

    # Tronca a max_dur
    max_samples = int(max_dur * sr)
    arr = arr[:max_samples]
    return arr, sr


# ─────────────────────────────────────────────────────────────────────────────
# Desktop fallback (sviluppo/test)
# ─────────────────────────────────────────────────────────────────────────────
def _load_desktop(path: str, max_dur: float) -> Tuple[np.ndarray, int]:
    """
    Prova in ordine: soundfile → scipy.io.wavfile → errore.
    Per MP3 su desktop durante lo sviluppo usa soundfile (pip install soundfile).
    """
    ext = os.path.splitext(path)[1].lower()

    # 1. soundfile (supporta MP3 via libsndfile ≥ 1.1.0, WAV, FLAC, OGG…)
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y[:int(max_dur * sr)], int(sr)
    except Exception:
        pass

    # 2. scipy.io.wavfile (solo WAV)
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = data.astype(np.float32)
        if y.max() > 1.0:
            y /= 32768.0
        return y[:int(max_dur * sr)], int(sr)
    except Exception:
        pass

    raise RuntimeError(
        f"Impossibile caricare {path}. "
        "Su desktop installa 'soundfile' per supporto MP3."
    )
