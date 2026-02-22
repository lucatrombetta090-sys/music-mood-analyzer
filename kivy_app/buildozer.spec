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
android.api              = 34
android.ndk              = 25b
android.sdk              = 34
android.archs            = arm64-v8a,armeabi-v7a
android.allow_backup     = True

android.permissions      = \
    READ_EXTERNAL_STORAGE,\
    WRITE_EXTERNAL_STORAGE,\
    READ_MEDIA_AUDIO,\
    INTERNET

# Supporto file audio MP3 su Android 13+
android.features         = android.hardware.audio.output

# Gradle
android.gradle_dependencies = com.google.android.gms:play-services-base:18.0.1

# ── Python-for-Android recipes personalizzate ─────────────────────────────────
# Le seguenti recipe compilano librosa e le sue dipendenze per ARM.
# Assicurati di avere p4a aggiornato (python-for-android >= 2024.01)
p4a.branch = develop

# Eventuale fork con recipe librosa se non presente in upstream
# p4a.source_dir = /path/to/custom/p4a

# ── Build ────────────────────────────────────────────────────────────────────
[buildozer]
log_level = 2
warn_on_root = 1
