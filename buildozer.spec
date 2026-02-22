[app]
title = Babita Ultimate AI
package.name = babitaultimate
package.domain = org.babita

source.dir = .
source.include_exts = py,png,jpg,kv,ttf,txt,db

version = 0.1
version.regex = __version__ = ['"](.*)['"]
version.filename = %(source.dir)s/main.py

requirements = python3,kivy,plyer,cryptography,requests,beautifulsoup4,openssl,sqlite3

# Android specific
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33
android.gradle_dependencies = 'com.google.android.gms:play-services-location:21.0.1'

# Permissions
android.permissions = INTERNET,ACCESS_NETWORK_STATE,ACCESS_FINE_LOCATION,ACCESS_COARSE_LOCATION,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,VIBRATE

android.add_src =
android.add_src += java/

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

# Architecture
android.arch = arm64-v8a

# Java source
android.add_src = java/

# Signing (debug only)
android.debug = 1

# Build options
android.accept_sdk_license = True
android.ndk_path = /path/to/ndk  # Auto-detected usually

# Libraries
android.library_references = 

# Extra Java classes
android.add_src = 

# Services
android.services = 

# Wake lock
wakelock = True

# Window size
window.size = 450x700

[buildozer]
log_level = 2
warn_on_root = 1
