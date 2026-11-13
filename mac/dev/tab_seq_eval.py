#!/usr/bin/env python3
"""Dev: compare detected bass notes with a bass tab copied from the internet (text, G-D-A-E lines).

Tabs rarely give exact rhythm, so this compares the ORDER of notes, not their timing:
  - the tab is read block by block (4 string lines; "x8" / "(x 2)" repeat a block), giving a pitch sequence
  - the detected notes (song.json) give a pitch sequence too
  - longest common subsequence (LCS) of both, on pitch classes, after the best transposition
    (handles tuned-down guitars, e.g. Nirvana a whole step down)
Scores: recall = LCS / tab notes, precision = LCS / detected notes in the tab's time span, octave = share
of matched pairs that are also in the right octave. With --riffs the tab is a set of riffs without song
order: then only the share of detected 3-note sequences that occur in some riff is reported.

  python tab_seq_eval.py TAB.txt SONG.json|PACKAGE.zip [--riffs]
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

OPEN = {"G": 43, "D": 38, "A": 33, "E": 28}
LINE = re.compile(r"^\s*(?:\([^)]*\)\s*)?([GDAE])-?\|(.*)$")
REP = re.compile(r"x\s*(\d+)|\(x\s*(\d+)\)", re.I)


def parse_tab(text):
    """-> list of (pitch sequence, repeat count) per block"""
    blocks, cur = [], {}
    def flush():
        if cur:
            notes = {}
            rep = 1
            for s, (body, tail) in cur.items():
                for m in re.finditer(r"\d+", body):
                    # "10h12": h/p already separate the notes. A run like "00000000" (written without
                    # dashes) is one note per digit; 1-2 digits are one fret number.
                    g = m.group()
                    parts = [(m.start(), g)] if len(g) <= 2 else [(m.start() + i, c) for i, c in enumerate(g)]
                    for col, fret in parts:
                        notes.setdefault(col, []).append(OPEN[s] + int(fret))
                mm = REP.search(tail)
                if mm:
                    rep = int(mm.group(1) or mm.group(2))
            seq = [min(notes[c]) for c in sorted(notes)]  # double stops -> lowest note
            if seq:
                blocks.append((seq, rep))
            cur.clear()
    for raw in text.splitlines():
        m = LINE.match(raw)
        if not m:
            if cur and not raw.strip():
                flush()
            continue
        s, rest = m.group(1), m.group(2)
        # body = up to the last '|' or end; tail (after the tab) may hold "x8"
        last = rest.rfind("|")
        body, tail = (rest[:last], rest[last + 1:]) if last >= 0 else (rest, "")
        if s in cur:
            flush()
        cur[s] = (body, tail)
        if len(cur) == 4:
            flush()
    flush()
    return blocks


def lcs_pairs(a, b):
    """indices of an LCS of sequences a, b (lists of ints)"""
    n, m = len(a), len(b)
    L = np.zeros((n + 1, m + 1), dtype=np.int32)
    A = np.asarray(a)
    for i in range(1, n + 1):
        eq = (np.asarray(b) == A[i - 1])
        row, prev = L[i], L[i - 1]
        for j in range(1, m + 1):
            row[j] = prev[j - 1] + 1 if eq[j - 1] else (prev[j] if prev[j] >= row[j - 1] else row[j - 1])
    pairs, i, j = [], n, m
    while i and j:
        if a[i - 1] == b[j - 1]:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif L[i - 1, j] >= L[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def load_notes(path):
    p = Path(path)
    song = json.loads(zipfile.ZipFile(p).read("song.json")) if p.suffix == ".zip" else json.load(open(p))
    return [(n["t"], n["p"]) for n in song["bass"]["notes"]]


def best_shift(ref, det):
    hr = np.bincount(np.asarray(ref) % 12, minlength=12).astype(float)
    hd = np.bincount(np.asarray(det) % 12, minlength=12).astype(float)
    return int(np.argmax([np.dot(hr, np.roll(hd, -k)) for k in range(12)]))


def evaluate(tab_text, notes, riffs=False):
    blocks = parse_tab(tab_text)
    det = [p for _, p in sorted(notes)]
    if riffs:
        grams = set()
        for seq, _ in blocks:
            s = seq * 3
            grams |= {tuple(x % 12 for x in s[i:i + 3]) for i in range(len(s) - 2)}
        best = None
        for k in range(12):
            d = [(p + k) % 12 for p in det]
            g = [tuple(d[i:i + 3]) for i in range(len(d) - 2)]
            ok = sum(x in grams for x in g) / max(1, len(g))
            if best is None or ok > best[0]:
                best = (ok, k)
        return dict(mode="riffs", trigram_precision=best[0], shift=best[1], det=len(det))
    ref = [p for seq, rep in blocks for _ in range(rep) for p in seq]
    a = [p % 12 for p in ref]
    best = None
    # tuning: the recording may be tuned down (Eb: +1, D: +2 semitones to reach the tab's fret pitches);
    # wider shifts would let a repeated riff "match" a different key by chance
    for k in (0, 1, 2):
        pr = lcs_pairs(a, [(p + k) % 12 for p in det])
        if best is None or len(pr) > len(best[1]):
            best = (k, pr)
    k, pairs = best
    # octave: compare after the octave shift that fits most matched pairs
    diffs = [(ref[i] - det[j]) for i, j in pairs]
    oct_ok = 0
    if diffs:
        vals, cnt = np.unique(diffs, return_counts=True)
        oct_ok = int(cnt.max())
    # precision only over detected notes between the first and last matched note (intro/outro noise
    # outside the tab is not counted, the tab may skip them)
    span = (pairs[-1][1] - pairs[0][1] + 1) if pairs else len(det)
    rec = len(pairs) / max(1, len(ref))
    prec = len(pairs) / max(1, span)
    return dict(mode="song", ref=len(ref), det=len(det), lcs=len(pairs), recall=rec, precision=prec,
                f1=2 * rec * prec / max(1e-9, rec + prec), octave=oct_ok / max(1, len(pairs)), shift=k)


def fmt(r):
    if r["mode"] == "riffs":
        return (f"3-Ton-Folgen, die im Tab vorkommen: {100 * r['trigram_precision']:.1f}% "
                f"({r['det']} Noten, Transposition {r['shift']})")
    return (f"F1 {r['f1']:.3f} | Tab-Noten gefunden {100 * r['recall']:.1f}% | Praezision {100 * r['precision']:.1f}% "
            f"| richtige Oktave {100 * r['octave']:.1f}% | Tab {r['ref']} / erkannt {r['det']} Noten, Transp. {r['shift']}")


if __name__ == "__main__":
    tab = open(sys.argv[1]).read()
    print(fmt(evaluate(tab, load_notes(sys.argv[2]), "--riffs" in sys.argv)))
