#!/usr/bin/env python3
"""Dev tool: synthesize a test song (stems + truth) and evaluate a result package.

  python synth_test.py make OUTDIR      -> OUTDIR/{mix,bass,drums,vocals,other}.wav + truth.json (+ mix.wav)
  python synth_test.py eval ZIP TRUTH   -> detection accuracy
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120
BEAT = 60 / BPM
DUR = 30.0
LEAD = 0.25  # song starts a bit after t=0
rng = np.random.default_rng(7)


def hp_noise(n, lo, hi):
    """Band-limited noise via FFT masking."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n)


def kick(vel):
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    f = 50 + 110 * np.exp(-t * 35)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return vel * np.sin(ph) * np.exp(-t * 14) * 0.9


def snare(vel):
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 28) * 0.4
    noise = hp_noise(n, 1500, 9000) * np.exp(-t * 20)
    noise = noise / (np.abs(noise).max() + 1e-9) * 0.7
    return vel * (body + noise)


def hihat(vel):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    x = hp_noise(n, 6500, 16000) * np.exp(-t * 90)
    return vel * x / (np.abs(x).max() + 1e-9) * 0.35


def add(buf, sig, t):
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def bass_note(p, d, vel):
    f = 440 * 2 ** ((p - 69) / 12)
    n = int(d * SR)
    t = np.arange(n) / SR
    env = np.minimum(1, t / 0.005) * np.exp(-t * 2.5) * np.minimum(1, (d - t) / 0.03)
    return vel * env * (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.15 * np.sin(6 * np.pi * f * t))


def make(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    N = int(DUR * SR)
    drums = np.zeros(N)
    bass = np.zeros(N)
    truth = {"tempo": BPM, "lead": LEAD, "kick": [], "snare": [], "hihat": [], "bass": []}
    nbars = int((DUR - LEAD - 1) // (4 * BEAT))
    # bass line per bar: (beat offset, midi, length in beats)
    riffs = [
        [(0, 28, 1.5), (2, 28, 0.75), (3, 31, 0.9)],
        [(0, 33, 1.5), (2, 33, 0.75), (3, 35, 0.9)],
        [(0, 38, 1.0), (1, 38, 0.9), (2, 36, 1.5), (3.5, 33, 0.4)],
        [(0, 31, 1.5), (2, 33, 0.9), (3, 28, 0.9)],
    ]
    for b in range(nbars):
        t0 = LEAD + b * 4 * BEAT
        # kick on 1, 3 (+ "and" of 3 on odd bars); snare on 2, 4; hats on 8ths (accented on beats)
        for beat in (0, 2):
            v = 0.9 if beat == 0 else 0.75
            add(drums, kick(v), t0 + beat * BEAT)
            truth["kick"].append(t0 + beat * BEAT)
        if b % 2:
            add(drums, kick(0.6), t0 + 2.5 * BEAT)
            truth["kick"].append(t0 + 2.5 * BEAT)
        for beat in (1, 3):
            add(drums, snare(0.9), t0 + beat * BEAT)
            truth["snare"].append(t0 + beat * BEAT)
        for e in range(8):
            v = 0.8 if e % 2 == 0 else 0.5
            add(drums, hihat(v), t0 + e * BEAT / 2)
            truth["hihat"].append(t0 + e * BEAT / 2)
        for off, p, ln in riffs[b % 4]:
            ts = t0 + off * BEAT
            d = ln * BEAT * 0.9
            add(bass, bass_note(p, d, 0.6), ts)
            truth["bass"].append({"t": ts, "d": d, "p": p})
    # other: soft pad chord; vocals: quiet vibrato sine
    t = np.arange(N) / SR
    other = 0.05 * (np.sin(2 * np.pi * 261.6 * t) + np.sin(2 * np.pi * 329.6 * t) + np.sin(2 * np.pi * 392 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))
    vocals = 0.05 * np.sin(2 * np.pi * (440 + 6 * np.sin(2 * np.pi * 5 * t)) * t) * (np.sin(2 * np.pi * 0.25 * t) > 0)
    # light noise floor on drums to avoid digital silence
    drums += 0.001 * rng.standard_normal(N)
    mix = bass + drums + other + vocals
    pk = np.abs(mix).max()
    g = 0.9 / pk
    for name, x in (("bass", bass), ("drums", drums), ("vocals", vocals), ("other", other), ("mix", mix)):
        sf.write(str(outdir / f"{name}.wav"), np.stack([x * g, x * g], axis=1), SR, subtype="PCM_16")
    json.dump(truth, open(outdir / "truth.json", "w"))
    print("Synthetic song written to", outdir)


def match(det, tru, tol=0.04):
    det, tru = sorted(det), sorted(tru)
    used = set()
    tp, errs = 0, []
    for t in tru:
        best, bd = None, tol
        for i, d in enumerate(det):
            if i in used:
                continue
            if abs(d - t) <= bd:
                best, bd = i, abs(d - t)
        if best is not None:
            used.add(best)
            tp += 1
            errs.append(det[best] - t)
    return tp, len(det) - tp, len(tru) - tp, errs


def evaluate(zip_path: Path, truth_path: Path):
    z = zipfile.ZipFile(zip_path)
    song = json.loads(z.read("song.json"))
    truth = json.load(open(truth_path))
    for k in ("kick", "snare", "hihat"):
        det = [h["t"] for h in song["drums"]["hits"] if h["k"] == k]
        tp, fp, fn, errs = match(det, truth[k])
        print(f"{k:6s}: truth {len(truth[k]):3d}  found {len(det):3d}  TP {tp:3d}  FP {fp:3d}  FN {fn:3d}  "
              f"median timing err {1000*np.median(errs) if errs else float('nan'):+.1f} ms")
    other = [h for h in song["drums"]["hits"] if h["k"] not in ("kick", "snare", "hihat")]
    print("other drum classes (false):", len(other))
    det = song["bass"]["notes"]
    tn = truth["bass"]
    tp, fp, fn, errs = match([n["t"] for n in det], [n["t"] for n in tn], tol=0.06)
    pit = 0
    for t in tn:
        c = [n for n in det if abs(n["t"] - t["t"]) <= 0.06]
        if c and min(c, key=lambda n: abs(n["t"] - t["t"]))["p"] == t["p"]:
            pit += 1
    print(f"bass  : truth {len(tn)}  found {len(det)}  onset TP {tp}  FP {fp}  FN {fn}  correct pitch {pit}/{len(tn)}  "
          f"median timing err {1000*np.median(errs) if errs else float('nan'):+.1f} ms")
    print(f"tempo : {song['tempo']} (truth {truth['tempo']}), beats {len(song['beats'])}, firstDownbeat idx {song['firstDownbeat']}")
    b = song["beats"]
    fd = song["firstDownbeat"]
    print(f"        first beats {b[:5]}; downbeat time {b[fd]:.3f} (truth bar starts at {truth['lead']} + n*2.0)")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(Path(sys.argv[2]))
    else:
        evaluate(Path(sys.argv[2]), Path(sys.argv[3]))
