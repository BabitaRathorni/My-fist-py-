[app]
title = Babita Ultimate AI
package.name = babitaultimate
package.domain = org.babita

source.dir = .
source.include_exts = py,png,jpg,kv,ttf,txt,db

version = 0.1

requirements = python3,kivy,plyer,requests,pyjnius

# Android specific
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33

# Permissions
android.permissions = INTERNET,ACCESS_NETWORK_STATE,VIBRATE

# App icon
icon.filename = %(source.dir)s/assets/icon.png

# Orientation
orientation = portrait

# Architecture
android.archs = arm64-v8a

# Debug
android.debug = 1

[buildozer]
log_level = 2
