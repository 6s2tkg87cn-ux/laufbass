#!/bin/bash
# Laufbass - einmalige Installation (macOS, Apple Silicon)
# Kann gefahrlos mehrfach ausgefuehrt werden.
set -e

HIER="$(cd "$(dirname "$0")" && pwd)"
VENV="$HOME/.laufbass/venv"

echo ""
echo "=============================================="
echo "  Laufbass - Installation"
echo "=============================================="

if [ "$(uname -s)" != "Darwin" ]; then
  echo "FEHLER: Dieses Skript ist nur fuer macOS gedacht."
  exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  echo "Hinweis: Dies ist kein Apple-Silicon-Mac (M1/M2/M3). Es sollte trotzdem laufen, ist aber langsamer."
fi

# Homebrew-Pfad bekannt machen (fuer Doppelklick-Start)
for p in /opt/homebrew/bin /usr/local/bin; do
  [ -d "$p" ] && PATH="$p:$PATH"
done
export PATH

# --- 1) Homebrew ------------------------------------------------------------
echo ""
echo "Schritt 1/5: Homebrew pruefen ..."
if ! command -v brew >/dev/null 2>&1; then
  echo ""
  echo "FEHLER: Homebrew ist nicht installiert."
  echo "Homebrew ist ein kostenloser Paketmanager fuer den Mac. So installierst du ihn:"
  echo "  1. Oeffne https://brew.sh im Browser."
  echo "  2. Kopiere den dort angezeigten Befehl in dieses Terminal-Fenster und druecke Enter."
  echo "  3. Folge den Hinweisen am Ende der Installation (\"Next steps\")."
  echo "  4. Starte danach diese Installation erneut."
  exit 1
fi
echo "   Homebrew gefunden: $(brew --prefix)"

# --- 2) python@3.11 + ffmpeg ------------------------------------------------
echo ""
echo "Schritt 2/5: Python 3.11 und ffmpeg pruefen ..."
if ! brew list --versions python@3.11 >/dev/null 2>&1; then
  echo "   Installiere python@3.11 (kann einige Minuten dauern) ..."
  brew install python@3.11
else
  echo "   python@3.11 ist bereits installiert."
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "   Installiere ffmpeg (kann einige Minuten dauern) ..."
  brew install ffmpeg
else
  echo "   ffmpeg ist bereits installiert."
fi
PY="$(brew --prefix python@3.11)/bin/python3.11"
if [ ! -x "$PY" ]; then
  echo "FEHLER: python3.11 wurde nicht gefunden unter $PY"
  exit 1
fi

# --- 3) virtuelle Umgebung --------------------------------------------------
echo ""
echo "Schritt 3/5: Python-Umgebung unter $VENV ..."
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
  echo "   Vorhandene Umgebung hat die falsche Python-Version und wird neu angelegt."
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$HOME/.laufbass"
  "$PY" -m venv "$VENV"
  echo "   Umgebung angelegt."
else
  echo "   Umgebung ist bereits vorhanden."
fi

# --- 4) Pakete --------------------------------------------------------------
echo ""
echo "Schritt 4/5: Pakete installieren (beim ersten Mal ca. 1 GB, 5-10 Minuten) ..."
"$VENV/bin/python" -m pip install --upgrade pip wheel -q
"$VENV/bin/python" -m pip install -r "$HIER/requirements.txt"

# --- 5) Modell vorab laden --------------------------------------------------
echo ""
echo "Schritt 5/5: KI-Modell laden und testen (ca. 80 MB, Internet noetig) ..."
"$VENV/bin/python" - <<'PYEOF'
import sys, warnings
warnings.filterwarnings("ignore")
try:
    import torch, numpy as np
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    m = get_model("htdemucs"); m.eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.zeros(1, 2, 44100 * 2)
    try:
        apply_model(m, x, device=dev, split=True, shifts=0)
    except Exception as e:
        print("   Hinweis: Test auf", dev, "fehlgeschlagen (", str(e)[:80], ") - es wird dann die CPU genutzt.")
        apply_model(m, x, device="cpu", split=True, shifts=0)
    print("   Demucs-Modell OK (Rechenwerk:", dev + ")")
    import librosa, pretty_midi, soundfile
    from basic_pitch.inference import predict  # noqa
    print("   Alle Pakete lassen sich laden.")
except Exception as e:
    print("   WARNUNG: Der Test ist fehlgeschlagen:", e)
    print("   Die Installation ist moeglicherweise unvollstaendig. Bitte pruefe die Internetverbindung")
    print("   und starte install.sh erneut.")
    sys.exit(2)
PYEOF

chmod +x "$HIER/Laufbass vorbereiten.command" 2>/dev/null || true
echo ""
echo "=============================================="
echo "  Fertig! Jetzt kannst du 'Laufbass vorbereiten.command'"
echo "  doppelklicken und einen Song auswaehlen."
echo "=============================================="
