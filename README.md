# Laufbass

Übe-App für **Bass und Schlagzeug**. Ein Song wird auf dem Mac einmal vorbereitet: Er wird in Spuren getrennt, und die Noten werden erkannt. Danach kann man auf Mac, iPad oder iPhone mitspielen. Das virtuelle Griffbrett bzw. Schlagzeug zeigt dabei, was zu spielen ist.

```
MP3 ──► Mac-Skript (Demucs + Basic Pitch + Drum-Erkennung) ──► Song.laufbass.zip ──► Übe-App
```

## Teile

| Ordner | Inhalt |
|---|---|
| `app/index.html` | Die Übe-App: eine einzelne HTML-Datei ohne Installation |
| `mac/` | Vorbereitung auf dem Mac: App „Laufbass Vorbereiten“ + Skript, Anleitung in [`mac/README.md`](mac/README.md) |
| `docs/FORMAT.md` | Aufbau einer `.laufbass.zip` |
| `mac/testdata/` | Kleiner synthetischer Testsong zum Ausprobieren |

## Übe-App

- **Bass:** Tab-Laufband, Griffbrett mit aktueller Note und Vorschau der nächsten, Bund- oder Notennamen, 4 oder 5 Saiten. Der Fingersatz wird automatisch berechnet: tiefe Lagen, wenige Lagenwechsel.
- **Schlagzeug:** Laufband pro Instrument und Drumset von oben. Die Pads leuchten beim Schlag auf und kündigen den nächsten an.
- **Üben:** Tempo 40–120 % ohne Tonhöhenänderung, Takt-Loop (1/2/4 Takte), A-B-Loop, Einzähler, Metronom-Klick.
- **Mixer:** Jede Spur hat Lautstärke und Stummschaltung. Es gibt die Voreinstellungen „Mitspielen (ohne Bass)“ und „Nur Bass“.
- **Bibliothek:** Geöffnete Songs bleiben im Browser gespeichert.
- **Korrekturen:** Eine in Logic korrigierte `.mid` einfach zusätzlich öffnen. Sie ersetzt dann die erkannten Noten.
- **Bedienung:** Leertaste = Play/Pause, ← → = spulen, L = Loop, B/S = Bass/Schlagzeug. Durch Ziehen am Laufband wird gespult.

**Starten:** `app/index.html` in Safari oder Chrome öffnen. Nach Aktivierung von GitHub Pages geht es auch online unter `https://<benutzer>.github.io/laufbass/`. Auf dem iPad die `.laufbass.zip` über „Öffnen“ aus der Dateien-App wählen.

## Grenzen

Die Erkennung ist gut genug zum Üben, aber nicht perfekt:

- **Bass:** Gelegentlich gibt es Oktavfehler, und schnelle Tonwiederholungen werden verschluckt.
- **Schlagzeug:** Kick, Snare und Hi-Hat werden erkannt, Toms, Ride und offene Hi-Hat noch nicht.
- **Tempo und Takt:** Das Tempo kann doppelt oder halb erkannt werden. Es wird ein 4/4-Takt angenommen.

Die Erkennung der Drums steckt in einer einzigen Funktion (`transcribe_drums_bands`) und lässt sich später durch ein besseres Modell ersetzen.
