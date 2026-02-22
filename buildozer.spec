[app]
title = Babita Ultimate AI
package.name = babitaultimate
package.domain = org.babita

source.dir = .
source.include_exts = py,png,jpg,kv,ttf,txt

version = 0.1

requirements = python3,kivy,requests

# Android specific - NO android.sdk line
android.api = 33
android.minapi = 21
android.ndk = 25b

android.permissions = INTERNET

orientation = portrait

android.archs = arm64-v8a

android.debug = 1

[buildozer]
log_level = 2
