[app]

title            = Music Mood Analyzer
package.name     = musicmoodanalyzer
package.domain   = com.example
source.dir       = .
source.include_exts = py,kv,json,txt,db

version          = 2.1

# numpy pinned a 1.24.4 — l'ultima versione compatibile con p4a v2024.01.21
# (numpy >= 1.25 causa "lapack_lite ld returned 1 exit status" sul build host)
requirements     = python3,kivy==2.3.0,numpy==1.24.4,pyjnius,android,mutagen,plyer,pillow

orientation      = portrait

# Android 16 (API 36) — targetapi deve essere >= 35 altrimenti il sistema
# forza il compatibility mode e alcune API crashano.
# minapi=21 copre il 99% dei dispositivi Android attivi.
android.minapi           = 21
android.api              = 35
android.ndk              = 25b
android.build_tools_version = 34.0.0
android.archs            = arm64-v8a,armeabi-v7a
android.allow_backup     = False
android.accept_sdk_license = True

# Permessi — MANAGE_EXTERNAL_STORAGE rimosso dal manifest:
# su Android 11+ viene richiesto a runtime aprendo le Impostazioni,
# dichiararlo nel manifest senza la meta-data corretta crasha l'app.
android.permissions = \
    READ_EXTERNAL_STORAGE,\
    WRITE_EXTERNAL_STORAGE,\
    READ_MEDIA_AUDIO,\
    READ_MEDIA_IMAGES,\
    INTERNET

p4a.branch = v2024.01.21

[buildozer]
log_level = 2
warn_on_root = 1
