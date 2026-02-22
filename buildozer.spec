[app]
title = Babita Ultimate AI
package.name = babitaultimate
package.domain = org.babita

source.dir = .
source.include_exts = py,png,jpg,kv,ttf,txt,db

version = 0.1
# version.regex = __version__ = ['"](.*)['"]
# version.filename = %(source.dir)s/main.py

requirements = python3,kivy,plyer,cryptography,requests,beautifulsoup4,pyjnius,android

# ANDROID SPECIFIC
android.api = 33
android.minapi = 21
android.ndk = 25b
# 🔥 FIXED: android.arch -> android.archs
android.archs = arm64-v8a
android.ndk_path = $(ANDROID_HOME)/ndk/25.2.9519653
android.sdk_path = $(ANDROID_SDK_ROOT)
android.gradle_dependencies = 'com.google.android.gms:play-services-location:21.0.1'

# Permissions
android.permissions = INTERNET,ACCESS_NETWORK_STATE,ACCESS_FINE_LOCATION,ACCESS_COARSE_LOCATION,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,VIBRATE

# Extra Java source
android.add_src = java/

# Metadata
android.meta_data = com.google.android.gms.version=@integer/google_play_services_version

# App icon
icon.filename = %(source.dir)s/assets/icon.png
presplash.filename = %(source.dir)s/assets/splash.png

# Orientation
orientation = portrait

# Fullscreen
fullscreen = 0

# Log level
log_level = 2

# Signing (debug only)
android.debug = 1

# SQLite support
android.ndk_libraries = libc++_shared.so,libcrypto.so,libssl.so,libsqlite3.so

# Wake lock
wakelock = True

# Window size
window.size = 450x700

# Recipes for Android
android.recipes = sqlite3

[buildozer]
log_level = 2
warn_on_root = 1
