[app]

title            = Music Mood Analyzer
package.name     = musicmoodanalyzer
package.domain   = com.example
source.dir       = .
source.include_exts = py,kv,json,txt,db

version          = 2.0

# numpy, scipy, pyjnius hanno recipe p4a funzionanti
# librosa RIMOSSA — sostituita da dsp_features.py (scipy puro)
requirements     = python3,kivy==2.3.0,numpy,scipy,pyjnius,mutagen,plyer,pillow

orientation      = portrait

android.minapi           = 24
android.api              = 33
android.ndk              = 25b
android.build_tools_version = 34.0.0
android.archs            = arm64-v8a,armeabi-v7a
android.allow_backup     = True
android.accept_sdk_license = True

android.permissions      = \
    READ_EXTERNAL_STORAGE,\
    WRITE_EXTERNAL_STORAGE,\
    READ_MEDIA_AUDIO,\
    INTERNET

android.features         = android.hardware.audio.output

p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 1
