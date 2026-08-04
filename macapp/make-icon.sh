#!/bin/zsh
# Builds Resources/Scriba.icns from the mark the site also uses.
#
# Generated rather than committed as a binary, for the same reason the rest of
# this repository is text: docs/img/mark.svg can be read and reviewed, an .icns
# cannot. Run it when the mark changes.
set -euo pipefail

cd "$(dirname "$0")"
WORK="$(mktemp -d)"
ICONSET="$WORK/Scriba.iconset"
mkdir -p Resources

echo "==> drawing the tiles"
swiftc -O -parse-as-library Tools/make-icon.swift -o "$WORK/make-icon"
"$WORK/make-icon" "$ICONSET"

echo "==> packing"
iconutil -c icns "$ICONSET" -o Resources/Scriba.icns
echo "==> Resources/Scriba.icns ($(du -h Resources/Scriba.icns | cut -f1))"
