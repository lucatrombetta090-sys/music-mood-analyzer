"""
audio_decode.py — Decodifica MP3 → PCM float32 mono

Android : MediaExtractor + MediaCodec via pyjnius (Java nativa, zero dipendenze extra)
Desktop  : soundfile (pip install soundfile) oppure wave per WAV
"""

import os
import numpy as np
from typing import Tuple


def load_audio(path: str, duration: float = 90.0) -> Tuple[np.ndarray, int]:
    try:
        from jnius import autoclass   # disponibile solo su Android
        return _android(path, duration)
    except ImportError:
        return _desktop(path, duration)


# ─────────────────────────────────────────────────────────────────────────────
# Android — MediaCodec (nessuna dipendenza Python esterna)
# ─────────────────────────────────────────────────────────────────────────────
def _android(path: str, max_dur: float) -> Tuple[np.ndarray, int]:
    from jnius import autoclass
    MediaExtractor = autoclass("android.media.MediaExtractor")
    MediaCodec     = autoclass("android.media.MediaCodec")
    MediaFormat    = autoclass("android.media.MediaFormat")
    BufferInfo     = autoclass("android.media.MediaCodec$BufferInfo")

    ext = MediaExtractor()
    ext.setDataSource(path)

    audio_idx, fmt = -1, None
    for i in range(ext.getTrackCount()):
        f = ext.getTrackFormat(i)
        mime = f.getString(MediaFormat.KEY_MIME)
        if mime and mime.startswith("audio/"):
            audio_idx, fmt = i, f
            break
    if audio_idx < 0:
        raise RuntimeError(f"Nessuna traccia audio: {path}")

    ext.selectTrack(audio_idx)
    sr       = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
    channels = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

    codec = MediaCodec.createDecoderByType(fmt.getString(MediaFormat.KEY_MIME))
    codec.configure(fmt, None, None, 0)
    codec.start()

    pcm      = bytearray()
    info     = BufferInfo()
    TIMEOUT  = 10_000
    max_bytes = int(max_dur * sr * channels * 2)
    done     = False

    while not done and len(pcm) < max_bytes:
        in_idx = codec.dequeueInputBuffer(TIMEOUT)
        if in_idx >= 0:
            buf = codec.getInputBuffer(in_idx)
            n   = ext.readSampleData(buf, 0)
            if n < 0:
                codec.queueInputBuffer(in_idx, 0, 0, 0,
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                done = True
            else:
                codec.queueInputBuffer(in_idx, 0, n, ext.getSampleTime(), 0)
                ext.advance()

        out_idx = codec.dequeueOutputBuffer(info, TIMEOUT)
        if out_idx >= 0:
            buf = codec.getOutputBuffer(out_idx)
            buf.limit(info.size)
            pcm.extend(buf.array()[buf.arrayOffset():buf.arrayOffset()+info.size])
            codec.releaseOutputBuffer(out_idx, False)
            if info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM:
                done = True

    codec.stop(); codec.release(); ext.release()

    arr = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr[:int(max_dur * sr)], sr


# ─────────────────────────────────────────────────────────────────────────────
# Desktop fallback — soundfile (solo per sviluppo/test, NON serve su Android)
# ─────────────────────────────────────────────────────────────────────────────
def _desktop(path: str, max_dur: float) -> Tuple[np.ndarray, int]:
    # Prova soundfile (supporta MP3 se libsndfile >= 1.1.0)
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y[:int(max_dur * sr)], int(sr)
    except Exception:
        pass

    # Fallback WAV con wave stdlib
    try:
        import wave, struct
        with wave.open(path, "rb") as wf:
            sr  = wf.getframerate()
            ch  = wf.getnchannels()
            sw  = wf.getsampwidth()
            n   = min(wf.getnframes(), int(max_dur * sr))
            raw = wf.readframes(n)
        fmt  = {1: "b", 2: "h", 4: "i"}.get(sw, "h")
        data = np.array(struct.unpack(f"{len(raw)//sw}{fmt}", raw), dtype=np.float32)
        if sw == 2: data /= 32768.0
        elif sw == 1: data = (data - 128) / 128.0
        if ch > 1: data = data.reshape(-1, ch).mean(axis=1)
        return data, sr
    except Exception:
        pass

    raise RuntimeError(
        f"Impossibile caricare {path}.\n"
        "Su desktop: pip install soundfile"
    )
