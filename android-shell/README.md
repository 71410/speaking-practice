# Android Shell

This folder contains a Capacitor Android wrapper for the Streamlit app:

https://speaking-practice-bhpcjdevjsrfpx99jckqpy.streamlit.app/

The APK is only a native shell. AI calls, Supabase, login, and secrets stay on
the hosted Streamlit app.

## Build

Install Android Studio first, including:

- Android SDK Platform
- Android SDK Build-Tools
- Android Platform-Tools
- JDK 17 or newer

Then run:

```powershell
cd android-shell
npm install
npm run android:add
npm run android:sync
cd android
.\gradlew assembleDebug
```

The debug APK will be created at:

```text
android-shell/android/app/build/outputs/apk/debug/app-debug.apk
```

## Build On GitHub

This repository also includes `.github/workflows/build-android-apk.yml`.
After pushing these files to GitHub, open GitHub Actions, choose
`Build Android APK`, run the workflow, then download the
`ielts-trainer-debug-apk` artifact.

## Change The Streamlit URL

Edit `capacitor.config.ts` and `www/index.html`, then run:

```powershell
npm run android:sync
```
