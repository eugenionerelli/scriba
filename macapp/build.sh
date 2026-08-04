#!/bin/zsh
# Builds Scriba.app.
#
# No .xcodeproj: SwiftPM produces the executable, and the .app bundle is assembled
# around it by hand here. Less elegant than a project file. In exchange everything
# stays in git as text and it rebuilds from the terminal without opening Xcode.
set -euo pipefail

cd "$(dirname "$0")"
APP="Scriba.app"
CONF="${1:-release}"

echo "==> building ($CONF)"
swift build -c "$CONF" --disable-sandbox

BIN="$(swift build -c "$CONF" --show-bin-path)/Scriba"
[[ -x "$BIN" ]] || { echo "executable not found: $BIN"; exit 1 }

echo "==> assembling the bundle"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/Scriba"
cp Resources/Info.plist "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

# Ad-hoc signature: enough to run locally. Without it, macOS 26 refuses to launch
# an unsigned bundle even when you compiled it yourself.
#
# The entitlement is not optional now that the app records. Signed without it the
# microphone is not refused, it is granted and then silent: the access request
# returns denied in two milliseconds with no dialog, the audio engine starts
# anyway, and every frame arrives at amplitude zero.
#
# One consequence worth knowing: an ad-hoc signature identifies the binary by its
# hash, so every rebuild is a different application as far as the privacy database
# is concerned and macOS asks for the microphone again. That is the price of not
# having a Developer ID, not a fault in the build.
codesign --force --deep --sign - --entitlements Resources/Scriba.entitlements "$APP" 2>/dev/null || \
    echo "   (signing failed: the app still starts if you right-click > Open the first time)"

echo "==> done: $(pwd)/$APP"
echo "    open it with:  open $APP"
