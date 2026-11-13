#!/usr/bin/env python3
"""Dev: drum transcription accuracy on MDB-Drums (23 MedleyDB songs with every hit annotated).

  python drum_eval.py MDB_DIR [--source drum_only|demucs|mix] [--engine adtof|bands]

F-measure per class with +-50 ms tolerance (standard in drum transcription papers).
Classes: KD kick, SD snare, HH hi-hat, TT toms, CY cymbals (crash/ride).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import laufbass_prep as lp  # noqa: E402

MAP = {"kick": "KD", "snare": "SD", "hihat": "HH", "hihat_open": "HH", "tom_low": "TT", "tom_mid": "TT",
       "tom_high": "TT", "crash": "CY", "ride": "CY"}
CLASSES = ["KD", "SD", "HH", "TT", "CY"]
TOL = 0.05


def load_ref(path):
    ref = {c: [] for c in CLASSES}
    for line in open(path):
        p = line.split()
        if len(p) >= 2 and p[1] in ref:
            ref[p[1]].append(float(p[0]))
    return {c: np.array(sorted(v)) for c, v in ref.items()}


def match(det, ref):
    det, ref = np.sort(det), np.sort(ref)
    used = np.zeros(len(det), bool)
    tp = 0
    for t in ref:
        j = np.flatnonzero((np.abs(det - t) <= TOL) & ~used)
        if len(j):
            used[j[np.argmin(np.abs(det[j] - t))]] = True
            tp += 1
    return tp, len(det) - tp, len(ref) - tp


def separate_drums(mix_wav: Path, tmp: Path) -> Path:
    stems = lp.separate_stems(mix_wav, tmp)
    return stems["drums"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mdb")
    ap.add_argument("--source", default="drum_only", choices=["drum_only", "demucs", "mix"])
    ap.add_argument("--engine", default="adtof", choices=["adtof", "bands"])
    a = ap.parse_args()
    root = Path(a.mdb)
    tot = {c: [0, 0, 0] for c in CLASSES}
    for ann in sorted((root / "annotations" / "class").glob("*.txt")):
        name = ann.name.split("_class")[0]
        if a.source == "drum_only":
            wav = root / "audio" / "drum_only" / f"{name}_Drum.wav"
        else:
            wav = root / "audio" / "full_mix" / f"{name}_MIX.wav"
        if a.source == "demucs":
            cache = root.parent / "demucs_drums" / f"{name}.wav"
            if not cache.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as td:
                    import shutil
                    shutil.copy(separate_drums(wav, Path(td)), cache)
            wav = cache
        hits = lp.transcribe_drums_adtof(wav) if a.engine == "adtof" else lp.transcribe_drums_bands(wav)
        ref = load_ref(ann)
        row = []
        for c in CLASSES:
            det = np.array([h.t for h in hits if MAP.get(h.k) == c])
            tp, fp, fn = match(det, ref[c])
            for k, v in enumerate((tp, fp, fn)):
                tot[c][k] += v
            row.append(f"{c} {2 * tp / max(1, 2 * tp + fp + fn):.2f}" if len(ref[c]) else f"{c}  -  ")
        print(f"{name:32s} " + "  ".join(row), flush=True)
    print("-" * 80)
    allf = []
    for c in CLASSES:
        tp, fp, fn = tot[c]
        f = 2 * tp / max(1, 2 * tp + fp + fn)
        allf.append((f, tp + fn))
        print(f"{c}: F {f:.3f}  (Praezision {tp / max(1, tp + fp):.3f}, gefunden {tp / max(1, tp + fn):.3f}, Schlaege {tp + fn})")
    n = sum(w for _, w in allf)
    print(f"gesamt (nach Schlaegen gewichtet): F {sum(f * w for f, w in allf) / max(1, n):.3f}   [{a.source}, {a.engine}]")


if __name__ == "__main__":
    main()
