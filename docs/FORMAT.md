# Laufbass-Paketformat (Version 1)

Das Mac-Skript erzeugt pro Song **eine Datei** `<Titel>.laufbass.zip`. Die Übe-App öffnet genau diese Datei.

## Inhalt der ZIP-Datei

```
song.json      Pflicht – Metadaten, Beats, Bassnoten, Drum-Schläge
bass.mp3       Stem
drums.mp3      Stem
vocals.mp3     Stem
other.mp3      Stem (Gitarre, Keys, Rest)
bass.mid       Bass als MIDI (zum Korrigieren in Logic/GarageBand)
drums.mid      Drums als MIDI (General-MIDI-Drum-Map, Kanal 10)
```

- MP3-Dateien werden **unkomprimiert gespeichert** (`ZIP_STORED`), `song.json` und `.mid` dürfen `ZIP_DEFLATED` nutzen. Andere Kompressionsverfahren sind nicht erlaubt.
- Alle Dateien liegen auf oberster Ebene der ZIP (keine Unterordner).

## song.json

```json
{
  "format": "laufbass",
  "version": 1,
  "title": "Songtitel",
  "source": "original.mp3",
  "duration": 213.4,
  "tempo": 118.2,
  "timeSignature": [4, 4],
  "beatMode": "fixed",
  "beats": [0.52, 1.03, 1.54],
  "firstDownbeat": 0,
  "stems": {
    "bass": "bass.mp3",
    "drums": "drums.mp3",
    "vocals": "vocals.mp3",
    "other": "other.mp3"
  },
  "bass": {
    "tuning": [28, 33, 38, 43],
    "notes": [ { "t": 0.52, "d": 0.31, "p": 40, "v": 96 } ]
  },
  "drums": {
    "hits": [ { "t": 0.52, "k": "kick", "v": 110 } ]
  }
}
```

| Feld | Bedeutung |
|---|---|
| `duration` | Länge in Sekunden |
| `tempo` | Tempo in BPM (`60 / Schlagabstand`; bei `"variable"` der Mittelwert) |
| `timeSignature` | Taktart `[Schläge pro Takt, 4]`; derzeit `[4, 4]` oder `[3, 4]` |
| `beatMode` | optional: `"fixed"` = konstantes Tempo (Raster mit exakt gleichen Abständen) oder `"variable"` = Tempo schwankt, Abstände der `beats` sind unterschiedlich. Fehlt das Feld, gilt `"fixed"`. Die App darf in beiden Fällen einfach `beats` benutzen. |
| `beats` | Zeitpunkte aller Schläge in Sekunden, aufsteigend |
| `firstDownbeat` | Index in `beats`, der die erste „1“ eines Takts ist |
| `bass.tuning` | MIDI-Tonhöhen der leeren Saiten, tiefste zuerst (Standard E1 A1 D2 G2) |
| `bass.notes[]` | `t` Start (s), `d` Dauer (s), `p` MIDI-Tonhöhe, `v` Velocity 1–127 |
| `drums.hits[]` | `t` Zeit (s), `k` Instrument, `v` Velocity 1–127 |
| `analysis` | optional: womit erkannt wurde, z. B. `{"version": "2", "separation": "htdemucs", "bass": "v2", "drums": "adtof", "beats": "beat_this"}` |

Erlaubte Werte für `k`: `kick`, `snare`, `hihat`, `hihat_open`, `tom_high`, `tom_mid`, `tom_low`, `crash`, `ride`.

Alle Arrays sind nach Zeit sortiert. Fehlt ein Stem, fehlt auch sein Eintrag in `stems`.

## MIDI-Zuordnung (drums.mid, Kanal 10)

| k | MIDI |
|---|---|
| kick | 36 |
| snare | 38 |
| hihat | 42 |
| hihat_open | 46 |
| tom_low | 45 |
| tom_mid | 48 |
| tom_high | 50 |
| crash | 49 |
| ride | 51 |

Die App kann zusätzlich eine einzelne `.mid`-Datei laden und damit Bassnoten oder Drum-Schläge ersetzen (z. B. nach Korrektur in Logic). Spuren auf Kanal 10 gelten als Drums, alle anderen als Bass.
