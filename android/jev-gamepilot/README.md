# GamePilot Jev Dino Preview (Android)

This is an experimental **Chrome Dino only** Android adaptation of the capture,
vision, and action loop in `newuser7171/jev-gamepilot`. It now queries TypeSafe
Jev System One for a `jump` or `wait` Choice when its vision code sees an
approaching obstacle. The desktop Python modules, local Laya inference, other
game profiles, and ADB controller are not in this APK.

## Use

1. Install the APK and enable **GamePilot Dino Preview** in Android's Accessibility settings.
2. Open the app and enter your TypeSafe API key if you want Jev decisions.
   The key stays in memory for the session, is sent only to `api.typesafe.ai`,
   and is not saved in preferences. Without a key, the local reflex still runs.
3. Tap **Test Jev key** to check the connection, then tap **Start pilot with Jev**,
   grant screen capture, and choose **Entire screen**.
4. Open `chrome://dino` in Chrome and start the game. The pilot finds the ground
   line, measures near/ahead foreground clusters, and taps only while Chrome is
   in the foreground. The Android notification shows diagnostics and Jev status.
5. Stop capture from the app or notification. Disable Accessibility when finished.

The detector searches for a ground line and uses relative screen regions. It
is not calibrated across browsers, orientations, or game layouts. It may miss
obstacles or jump at the wrong time. Local reflexes jump at close obstacles;
Jev can also request an early jump on a fresh observation. Only numerical
scene measurements go to TypeSafe; no screen images are uploaded. The user
supplies their own API key; the APK has no bundled key.

## Build

Install Android SDK platform 35 and build-tools 35.0.0, Java with `javac`, and
OpenSSL. Set `ANDROID_HOME` to the SDK directory and run `./build.sh`. The script
creates a disposable signing key for this preview, so builds made on different
machines cannot install as updates over each other.
