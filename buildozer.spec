[app]

# (str) Title of your application
title = Babita AI

# (str) Package name
package.name = babita_ultimate

# (str) Package domain (needed for android packaging)
package.domain = org.test

# (str) Source code where the main.py is located
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,db,ttf

# (str) Application versioning
version = 0.1

# (list) Application requirements
# Yahan sari libraries hain jo tune code mein use ki hain
requirements = python3,kivy==2.3.0,plyer,cryptography,requests,beautifulsoup4,sqlite3

# (str) Supported orientation
orientation = portrait

# (bool) Indicate if the application should be fullscreen
fullscreen = 0

# (list) Permissions
android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,ACCESS_FINE_LOCATION,ACCESS_COARSE_LOCATION

# (int) Target Android API
android.api = 33

# (int) Minimum API your APK will support
android.minapi = 21

# (str) Android SDK directory (if empty, it will be automatically downloaded)
android.sdk_build_tools_revision = 33.0.0

# (str) Android NDK version
android.ndk = 25b

# (bool) Accept SDK license without operator interaction
android.accept_sdk_license = True

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = false, 1 = true)
warn_on_root = 1

      - name: Upload APK
        uses: actions/upload-artifact@v4
        with:
          name: Babita-Ultimate-APK
          path: bin/*.apk
