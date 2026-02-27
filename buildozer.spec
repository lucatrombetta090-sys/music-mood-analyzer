[app]

title            = Music Mood Analyzer
package.name     = musicmoodanalyzer
package.domain   = com.example
source.dir       = .
source.include_exts = py,kv,json,txt,db

version          = 2.1

# numpy senza pin versione — la versione è gestita dal recipe locale in ./recipes/numpy/
# Il recipe locale corregge l'URL (pypi.python.org è deprecato → files.pythonhosted.org)
requirements     = python3,kivy==2.3.0,numpy,pyjnius,android,mutagen,plyer,pillow

# Percorso recipe personalizzati (override URL numpy)
p4a.local_recipes = ./recipes

orientation      = portrait

android.minapi           = 21
android.api              = 35
android.ndk              = 25b
android.build_tools_version = 34.0.0
android.archs            = arm64-v8a,armeabi-v7a
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
