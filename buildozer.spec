[app]

# ── Informazioni app ──────────────────────────────────────────────────────────
title            = Music Mood Analyzer
package.name     = musicmoodanalyzer
package.domain   = com.example
source.dir       = .
source.include_exts = py,kv,json,txt,db

version          = 1.0
requirements     = python3,kivy==2.3.0,numpy,scipy,librosa,mutagen,plyer,sqlite3,pillow

# ── Orientamento ──────────────────────────────────────────────────────────────
orientation      = portrait

# ── Icon e splash ─────────────────────────────────────────────────────────────
# icon.filename    = %(source.dir)s/icon.png
# presplash.filename = %(source.dir)s/presplash.png

# ── Android specifiche ────────────────────────────────────────────────────────
android.minapi           = 24
android.api              = 33
android.ndk              = 25b
android.sdk              = 33
# Forza build-tools stabile (evita RC come 37.0.0-rc1)
android.build_tools_version = 34.0.0
android.archs            = arm64-v8a,armeabi-v7a
android.allow_backup     = True

android.permissions      = \
    READ_EXTERNAL_STORAGE,\
    WRITE_EXTERNAL_STORAGE,\
    READ_MEDIA_AUDIO,\
    INTERNET

# Supporto file audio MP3 su Android 13+
android.features         = android.hardware.audio.output

# Accetta automaticamente le licenze SDK
android.accept_sdk_license = True

# ── Python-for-Android ────────────────────────────────────────────────────────
p4a.branch = develop

# ── Build ────────────────────────────────────────────────────────────────────
[buildozer]
log_level = 2
warn_on_root = 1
