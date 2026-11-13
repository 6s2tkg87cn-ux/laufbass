#!/usr/bin/env python3
"""Dev benchmark for the bass transcription.

  python bass_bench.py make OUTDIR            -> OUTDIR/bass.wav + truth.json (hard synthetic bass line)
  python bass_bench.py eval OUTDIR [old|v2]   -> precision / recall / pitch accuracy

The bass is a Karplus-Strong plucked string (realistic harmonics, decaying), with
  - repeated legato eighths (the "two eighths become one note" case),
  - octave jumps, dead-note-free fills, rests,
  - a quiet intro where only piano bleed is present (Demucs residue), plus piano bleed under the bass.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
rng = np.random.default_rng(3)


def ks_pluck(p, d, vel, bright=0.5):
    f = 440 * 2 ** ((p - 69) / 12)
    n = int(d * SR)
    period = SR / f
    N = int(period)
    frac = period - N
    buf = rng.uniform(-1, 1, N)
    # soften the excitation (finger pluck)
    for _ in range(2):
        buf = 0.5 * (buf + np.roll(buf, 1))
    out = np.zeros(n)
    idx = 0
    damp = 0.996 - 0.004 * (1 - bright)
    prev = 0.0
    for i in range(n):
        cur = buf[idx]
        nxt = buf[(idx + 1) % N]
        val = (1 - frac) * cur + frac * nxt
        new = damp * 0.5 * (val + prev)
        prev = val
        buf[idx] = new
        out[i] = cur
        idx = (idx + 1) % N
    env = np.minimum(1, (d - np.arange(n) / SR) / 0.015)  # finger mute at the end
    return vel * out * np.clip(env, 0, 1)


def piano(p, d, vel):
    f = 440 * 2 ** ((p - 69) / 12)
    n = int(d * SR)
    t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * f * k * t) for k, a in ((1, 1), (2, 0.5), (3, 0.3), (4, 0.15)))
    return vel * x * np.exp(-t * 2.0) * np.minimum(1, t / 0.003)


def add(buf, sig, t):
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def make(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    bpm = 104
    beat = 60 / bpm
    intro = 8 * beat  # 2 bars piano only
    bars = 16
    dur = intro + bars * 4 * beat + 2
    N = int(dur * SR)
    bass = np.zeros(N)
    truth = []
    # patterns: (start in 8ths, midi, length in 8ths)
    pats = [
        [(0, 28, 1), (1, 28, 1), (2, 28, 1), (3, 28, 1), (4, 31, 1), (5, 31, 1), (6, 33, 1), (7, 35, 1)],   # legato 8ths
        [(0, 33, 2), (2, 45, 1), (3, 33, 1), (4, 33, 1), (5, 45, 1), (6, 31, 2)],                           # octaves
        [(0, 38, 3), (3, 38, 1), (4, 36, 2), (6, 33, 1), (7, 35, 1)],
        [(0, 40, 1), (1, 40, 1), (2, 40, 1), (3, 40, 1), (4, 40, 1), (5, 40, 1), (6, 43, 1), (7, 45, 1)],   # pumping 8ths
    ]
    for b in range(bars):
        t0 = intro + b * 4 * beat
        for st, p, ln in pats[b % 4]:
            ts = t0 + st * beat / 2 + rng.normal(0, 0.006)
            d = ln * beat / 2 * rng.uniform(0.92, 0.99)  # legato: almost no gap
            add(bass, ks_pluck(p, d, rng.uniform(0.5, 0.8)), ts)
            truth.append({"t": ts, "d": d, "p": p})
    # piano bleed: chords through the whole song, intro relatively stronger in the (otherwise empty) stem
    bleed = np.zeros(N)
    chords = [(48, 52, 55), (45, 48, 52), (50, 53, 57), (43, 47, 50)]
    for k in range(int(dur / (2 * beat))):
        for p in chords[k % 4]:
            add(bleed, piano(p, 2 * beat, 0.01), k * 2 * beat)
    x = bass + bleed + 0.0005 * rng.standard_normal(N)
    x *= 0.9 / np.abs(x).max()
    sf.write(str(outdir / "bass.wav"), x, SR, subtype="PCM_16")
    json.dump({"notes": truth, "intro_end": intro}, open(outdir / "truth.json", "w"))
    print("written", outdir, len(truth), "notes")


def score(det, truth, tol=0.05):
    used = set()
    tp = pit = 0
    for t in truth:
        best, bd = None, tol
        for i, d in enumerate(det):
            if i not in used and abs(d["t"] - t["t"]) <= bd:
                best, bd = i, abs(d["t"] - t["t"])
        if best is not None:
            used.add(best)
            tp += 1
            pit += det[best]["p"] == t["p"]
    fp = len(det) - tp
    return dict(truth=len(truth), found=len(det), onset_tp=tp, fp=fp, correct_pitch=pit,
                precision=round(pit / max(1, len(det)), 3), recall=round(pit / max(1, len(truth)), 3))


def evaluate(outdir: Path, engine: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    truth = json.load(open(outdir / "truth.json"))
    wav = outdir / "bass.wav"
    if engine == "old":
        import laufbass_prep as lp
        notes = [dict(t=n.t, p=n.p) for n in lp.transcribe_bass_basic_pitch(wav)]
    else:
        import bass_v2
        notes = [dict(t=n.t, p=n.p) for n in bass_v2.transcribe(wav, log=print)]
    intro = [n for n in notes if n["t"] < truth["intro_end"] - 0.1]
    print(engine, score(notes, truth["notes"]), "notes in intro (false):", len(intro))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(Path(sys.argv[2]))
    else:
        evaluate(Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "v2")
