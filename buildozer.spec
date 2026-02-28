[app]

title            = Music Mood Analyzer
package.name     = musicmoodanalyzer
package.domain   = com.example
source.dir       = .
source.include_exts = py,kv,json,txt,db

version          = 2.1

# NO p4a.local_recipes — il file __init__.py nella root rompeva tutti gli import.
# Il fix numpy URL è gestito direttamente nel workflow GitHub Actions.
requirements     = python3,kivy==2.3.0,numpy,pyjnius,android,mutagen,plyer,pillow

orientation      = portrait

android.minapi           = 21
android.api              = 35
android.ndk              = 25b
android.build_tools_version = 34.0.0
android.archs            = arm64-v8a
android.allow_backup     = False
android.accept_sdk_license = True

android.permissions = \
    READ_EXTERNAL_STORAGE,\
    WRITE_EXTERNAL_STORAGE,\
    READ_MEDIA_AUDIO,\
    INTERNET

p4a.branch = v2024.01.21

[buildozer]
log_level = 2
warn_on_root = 1
