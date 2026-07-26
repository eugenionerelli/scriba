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
codesign --force --deep --sign - "$APP" 2>/dev/null || \
    echo "   (signing failed: the app still starts if you right-click > Open the first time)"

echo "==> done: $(pwd)/$APP"
echo "    open it with:  open $APP"
