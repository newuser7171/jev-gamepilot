#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sdk_root="${ANDROID_HOME:?Set ANDROID_HOME to the Android SDK directory}"
tools_dir="$sdk_root/build-tools/35.0.0"
framework="$sdk_root/platforms/android-35/android.jar"
mkdir -p build/res build/classes build/dex
"$tools_dir/aapt2" compile --dir res -o build/res
"$tools_dir/aapt2" link -o build/unsigned.apk -I "$framework" --manifest AndroidManifest.xml --min-sdk-version 29 --target-sdk-version 35 build/res/*.flat
javac -source 8 -target 8 -bootclasspath "$framework" -d build/classes src/com/newuser/gamepilot/*.java
"$tools_dir/d8" --min-api 29 --lib "$framework" --output build/dex $(find build/classes -name '*.class')
cp build/unsigned.apk build/combined.apk
(cd build/dex && zip -q -u ../combined.apk classes.dex)
"$tools_dir/zipalign" -f 4 build/combined.apk build/aligned.apk
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -keyout build/debug.key -out build/debug.crt -days 3650 -subj '/CN=GamePilot Dino Preview/' >/dev/null 2>&1
openssl pkcs12 -export -inkey build/debug.key -in build/debug.crt -out build/debug.p12 -name gamepilot -passout pass:preview
"$tools_dir/apksigner" sign --ks build/debug.p12 --ks-type PKCS12 --ks-key-alias gamepilot --ks-pass pass:preview --out GamePilot-Dino-Preview.apk build/aligned.apk
"$tools_dir/apksigner" verify GamePilot-Dino-Preview.apk
