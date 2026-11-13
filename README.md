# Laufbass

Übe-App für **Bass und Schlagzeug**. Ein Song wird auf dem Mac einmal vorbereitet: Er wird in Spuren getrennt, und die Noten werden erkannt. Danach kann man auf Mac, iPad oder iPhone mitspielen. Das virtuelle Griffbrett bzw. Schlagzeug zeigt dabei, was zu spielen ist.

```
MP3 ──► Mac-App (Demucs + Bass-Erkennung + ADTOF + Beat This!) ──► Song.laufbass.zip ──► Übe-App
```

## Teile

| Ordner | Inhalt |
|---|---|
| `app/index.html` | Die Übe-App: eine einzelne HTML-Datei ohne Installation |
| `mac/` | Vorbereitung auf dem Mac: App „Laufbass Vorbereiten“ + Skript, Anleitung in [`mac/README.md`](mac/README.md) |
| `docs/FORMAT.md` | Aufbau einer `.laufbass.zip` |
| `mac/testdata/` | Kleiner synthetischer Testsong zum Ausprobieren |

## Übe-App

- **Bass:** Tab-Laufband, Griffbrett mit aktueller Note und Vorschau der nächsten, Bund- oder Notennamen. Der Fingersatz wird automatisch berechnet: tiefe Lagen, wenige Lagenwechsel.
- **Einfach-Modus (Taste E):** Bass zeigt pro Schlag nur den Grundton. Schlagzeug: Der Song wird in Teile mit gleichem Groove zerlegt, pro Teil gibt es ein festes Pattern aus Bassdrum, Snare und Hi-Hat (Achtel), ohne Fills.
- **Transponieren:** ♭/♯ verschiebt Bassnoten und Wiedergabe um Halbtöne, das Tempo bleibt gleich.
- **Raster korrigieren:** „Hier ist die 1“, Tempo ×2/÷2, Raster ±10 ms. Wird pro Song gespeichert.
- **Schlagzeug:** Laufband pro Instrument und Drumset von oben. Die Pads leuchten beim Schlag auf und kündigen den nächsten an.
- **Üben:** Tempo 40–120 % ohne Tonhöhenänderung, Takt-Loop (1/2/4 Takte), A-B-Loop, Einzähler, Metronom-Klick.
- **Mixer:** Jede Spur hat Lautstärke und Stummschaltung. Es gibt die Voreinstellungen „Mitspielen (ohne Bass)“ und „Nur Bass“.
- **Bibliothek:** Geöffnete Songs bleiben im Browser gespeichert.
- **Korrekturen:** Eine in Logic korrigierte `.mid` einfach zusätzlich öffnen. Sie ersetzt dann die erkannten Noten.
- **Bedienung:** Leertaste = Play/Pause, ← → = spulen, L = Loop, B/S = Bass/Schlagzeug, E = Einfach. Durch Ziehen am Laufband wird gespult.

**Starten:** `app/index.html` in Safari oder Chrome öffnen. Nach Aktivierung von GitHub Pages geht es auch online unter `https://<benutzer>.github.io/laufbass/`. Auf dem iPad die `.laufbass.zip` über „Öffnen“ aus der Dateien-App wählen.

## Grenzen

Die Erkennung ist gut genug zum Üben, aber nicht perfekt:

- **Bass:** Es werden nur Noten gezeigt, bei denen sich zwei Verfahren einig sind. An unklaren Stellen fehlt also eher eine Note. Slap, Akkorde und Slides bleiben schwierig.
- **Schlagzeug:** Leise Ghost-Notes und offene Hi-Hat fehlen oft. Crash und Ride werden nur nach dem Spielmuster unterschieden.
- **Tempo und Takt:** 4/4 und 3/4. Falls die „1“ oder das Tempo nicht stimmt: in der App unter „Raster“ korrigieren.

Entwickler-Tests: `mac/dev/synth_test.py` (ganzer Ablauf) und `mac/dev/bass_bench.py` (harte Bass-Testlinie mit Achteln, Oktaven und Klavier-Resten).
