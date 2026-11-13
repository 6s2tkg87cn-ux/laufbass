# Laufbass vorbereiten (Mac)

Dieses Werkzeug macht aus einem Song (MP3, M4A, WAV ...) ein **Laufbass-Paket**
(`Songname.laufbass.zip`), das du in der Laufbass-Übe-App öffnest. Darin stecken:

- vier getrennte Spuren: Bass, Schlagzeug, Gesang, Rest (als MP3)
- die erkannte Basslinie und das erkannte Schlagzeug (als MIDI und für die App)
- Tempo und Taktschläge

Alles läuft **lokal auf deinem Mac** – nichts wird hochgeladen.

## Einmalige Installation

Voraussetzung: MacBook mit Apple-Chip (M1/M2/M3 ...) und Internet.

1. Öffne das Programm **Terminal** (Spotlight: `Terminal`).
2. Ziehe die Datei `install.sh` aus diesem Ordner in das Terminal-Fenster und drücke **Enter**.
   (Alternativ: `cd` in den Ordner und `bash install.sh`.)
3. Warte, bis „Fertig!" erscheint (5–15 Minuten, einmalig, ca. 1–2 GB Download).

Falls **Homebrew** fehlt, sagt dir das Skript, wie du es installierst (https://brew.sh),
und du startest danach `install.sh` noch einmal. Das Skript kann beliebig oft wiederholt werden.

Wenn macOS beim Doppelklick auf `Laufbass vorbereiten.command` meldet, die Datei sei nicht
geöffnet worden: **Rechtsklick → Öffnen → Öffnen**. Falls die Datei „nicht ausführbar" ist:
im Terminal `chmod +x "Laufbass vorbereiten.command"` im Ordner ausführen.

## Benutzung

**Variante A (am einfachsten):** Doppelklick auf `Laufbass vorbereiten.command`.
Es öffnet sich ein Fenster zur Dateiauswahl (mehrere Songs gleichzeitig sind möglich).
Danach läuft die Verarbeitung im Terminal-Fenster. Am Ende zeigt der Finder die fertige
`.laufbass.zip` – sie liegt im selben Ordner wie der Original-Song.

**Variante B:** Songdatei(en) auf das geöffnete Terminal-Fenster ziehen, nachdem du
`Laufbass vorbereiten.command` gestartet hast – oder direkt:

```
~/.laufbass/venv/bin/python laufbass_prep.py "Mein Song.mp3" -o ~/Desktop
```

Die fertige `.laufbass.zip` öffnest du dann in der Laufbass-App.

## Wie lange dauert es?

Auf einem MacBook Air M1 ungefähr **1–3 Minuten für einen Song von 4 Minuten**
(der größte Teil ist die Instrumententrennung; das allererste Mal kommt der Modell-Download dazu).
Das Fenster bitte währenddessen offen lassen; das MacBook zieht dabei Strom und wird warm.

## Grenzen der Erkennung

- **Bass:** Funktioniert am besten bei klaren, einstimmigen Basslinien. Sehr schnelle Läufe,
  Slap, Akkorde oder stark verzerrter Bass werden fehlerhafter erkannt. Einzelne Oktavfehler
  und falsche Wiederholungen sind möglich.
- **Schlagzeug:** Es wird mit einem einfachen Verfahren nach Frequenzbereichen erkannt.
  Erkannt werden **Bassdrum, Snare, Hi-Hat** und (nur sehr vorsichtig) **Crash**.
  **Toms, Ride, offene Hi-Hat** werden nicht erkannt. Bei dichten Fills, leisen Ghost-Notes
  und Hi-Hat genau unter der Snare sind Fehler normal. Rechne mit einem Rohentwurf, nicht mit Perfektion.
- **Tempo/Takt:** Das Tempo wird automatisch bestimmt und kann bei manchen Songs doppelt oder halb
  so groß sein. Die „Eins" des Takts wird über die Bassdrum geraten. Songs mit Tempowechseln
  oder ohne feste Zählzeit funktionieren schlecht.
- Es wird immer von 4/4 ausgegangen.

## Noten in Logic korrigieren und neu laden

1. Entpacke die `.laufbass.zip` (Doppelklick) – darin liegen `bass.mid` und `drums.mid`.
2. Ziehe die MIDI-Datei in ein Logic-Projekt (oder GarageBand), am besten zusammen mit
   `bass.mp3` bzw. `drums.mp3` als Hörvorlage. Das Tempo der MIDI-Datei entspricht dem erkannten
   Songtempo; stelle in Logic ggf. „Tempo aus MIDI-Datei übernehmen" ein.
3. Korrigiere Noten (falsche Töne löschen, fehlende ergänzen).
   - Bass: beliebige Spur, Tonhöhen im Bereich einer 4-Saiten-Bassgitarre (E1 bis G4).
   - Drums: General-MIDI-Belegung (Bassdrum C1, Snare D1, Hi-Hat F#1 usw.), **Kanal 10**.
4. Exportiere die Spur wieder als MIDI-Datei (Datei → Exportieren → Auswahl als MIDI-Datei).
5. Öffne in der Laufbass-App den Song und lade die korrigierte `.mid` hinein: Spuren auf
   Kanal 10 ersetzen die Drum-Schläge, alle anderen die Basslinie.

## Probleme?

- „ffmpeg wurde nicht gefunden": im Terminal `brew install ffmpeg`.
- „Demucs ... nicht installiert": `install.sh` erneut ausführen.
- Modell-Download schlägt fehl: Internetverbindung prüfen und es noch einmal versuchen.
- Mehr Details bei Fehlern: im Terminal `LAUFBASS_DEBUG=1` vor den Befehl setzen.
