[app]

title            = Music Mood Analyzer
package.name     = musicmoodanalyzer
package.domain   = com.example
source.dir       = .
source.include_exts = py,kv,json,txt,db

version          = 2.0

# scipy RIMOSSA — tutto DSP riscritto con numpy puro (dsp_features.py)
# pyjnius per MediaCodec Android (decodifica MP3 nativa)
requirements     = python3,kivy==2.3.0,numpy,pyjnius,mutagen,plyer,pillow

orientation      = portrait

android.minapi           = 24
android.api              = 33
# NDK 28c: obbligatorio per recipe fortran (dipendenza numpy/openblas)
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

# develop usa Python 3.14 (incompatibile con Kivy 2.3.0 - Py_UNICODE rimosso)
p4a.branch = v2024.01.21

[buildozer]
log_level = 2
warn_on_root = 1
