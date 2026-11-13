#!/bin/bash
# Baut "Laufbass Vorbereiten.app" und legt sie in den Programme-Ordner.
# Wird von install.sh aufgerufen, kann aber auch allein laufen.
# Hinweis: kompatibel mit macOS-bash 3.2 (kein Heredoc in $(...)).
set -e

HIER="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HIER/.." && pwd)"
NAME="Laufbass Vorbereiten"

if [ -w "/Applications" ]; then ZIEL="/Applications"; else ZIEL="$HOME/Applications"; mkdir -p "$ZIEL"; fi
APP="$ZIEL/$NAME.app"

echo "   Baue $NAME.app ..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/practice"

cp "$HIER/app/gui.py" "$HIER/app/ui.html" "$HIER/app/icon.png" "$HIER/laufbass_prep.py" "$HIER/bass_v2.py" "$APP/Contents/Resources/"
cp "$ROOT/app/index.html" "$APP/Contents/Resources/practice/index.html"

# --- Info.plist ---
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Laufbass Vorbereiten</string>
  <key>CFBundleDisplayName</key><string>Laufbass Vorbereiten</string>
  <key>CFBundleIdentifier</key><string>de.laufbass.vorbereiten</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Laufbass</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.music</string>
</dict>
</plist>
PLIST

# --- Startprogramm ---
cat > "$APP/Contents/MacOS/Laufbass" <<'LAUNCH'
#!/bin/bash
R="$(cd "$(dirname "$0")/../Resources" && pwd)"
PY="$HOME/.laufbass/venv/bin/python"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
mkdir -p "$HOME/.laufbass"
if [ ! -x "$PY" ]; then
  /usr/bin/osascript -e 'display alert "Laufbass ist noch nicht installiert" message "Bitte zuerst install.sh im Laufbass-Ordner ausführen."'
  exit 1
fi
if ! "$PY" -c "import webview" >/dev/null 2>&1; then
  /usr/bin/osascript -e 'display alert "Ein Baustein fehlt" message "Bitte install.sh im Laufbass-Ordner noch einmal ausführen."'
  exit 1
fi
exec "$PY" "$R/gui.py" >>"$HOME/.laufbass/app.log" 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/Laufbass"

# --- Icon (sips + iconutil sind auf jedem Mac vorhanden) ---
if command -v iconutil >/dev/null 2>&1 && command -v sips >/dev/null 2>&1; then
  SET="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$SET"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$HIER/app/icon.png" --out "$SET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z $d $d "$HIER/app/icon.png" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$SET" -o "$APP/Contents/Resources/AppIcon.icns"
  rm -rf "$(dirname "$SET")"
fi

xattr -cr "$APP" 2>/dev/null || true
touch "$APP"
LSREG="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[ -x "$LSREG" ] && "$LSREG" -f "$APP" >/dev/null 2>&1 || true

echo "   Fertig: $APP"
