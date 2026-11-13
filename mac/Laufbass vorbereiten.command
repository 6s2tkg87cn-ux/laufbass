#!/bin/bash
# Laufbass - Song(s) vorbereiten (Doppelklick oder Dateien auf das Terminal-Fenster ziehen)
HIER="$(cd "$(dirname "$0")" && pwd)"
VENV="$HOME/.laufbass/venv"

for p in /opt/homebrew/bin /usr/local/bin; do
  [ -d "$p" ] && PATH="$p:$PATH"
done
export PATH

pause_end() { echo ""; read -n 1 -s -r -p "Taste druecken zum Schliessen ..."; echo ""; }

if [ ! -x "$VENV/bin/python" ]; then
  echo "Laufbass ist noch nicht installiert."
  echo "Bitte zuerst 'install.sh' ausfuehren (siehe README)."
  pause_end
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [ "$#" -eq 0 ]; then
  # Native Dateiauswahl; mehrere Dateien moeglich
  AUSWAHL="$(osascript <<'OSA' 2>/dev/null
try
  set dateien to choose file of type {"public.audio"} with prompt "Welche(n) Song(s) soll Laufbass vorbereiten?" with multiple selections allowed
  set pfade to {}
  repeat with d in dateien
    set end of pfade to POSIX path of d
  end repeat
  set AppleScript's text item delimiters to linefeed
  return pfade as text
on error
  return ""
end try
OSA
)"
  if [ -z "$AUSWAHL" ]; then
    echo "Keine Datei ausgewaehlt - Abbruch."
    pause_end
    exit 0
  fi
  DATEIEN=()
  while IFS= read -r zeile; do
    [ -n "$zeile" ] && DATEIEN+=("$zeile")
  done <<< "$AUSWAHL"
else
  DATEIEN=("$@")
fi

ERGEBNISSE=()
FEHLER=0
for datei in "${DATEIEN[@]}"; do
  if python "$HIER/laufbass_prep.py" "$datei"; then
    ZIP="$(dirname "$datei")/$(basename "${datei%.*}").laufbass.zip"
    [ -f "$ZIP" ] && ERGEBNISSE+=("$ZIP")
  else
    FEHLER=1
  fi
done

if [ "${#ERGEBNISSE[@]}" -gt 0 ]; then
  open -R "${ERGEBNISSE[@]}"
fi

echo ""
if [ "$FEHLER" -eq 0 ]; then
  echo "Alles erledigt. Die .laufbass.zip-Dateien liegen neben den Original-Songs."
else
  echo "Es gab Fehler - siehe Meldungen oben."
fi
pause_end
exit $FEHLER
