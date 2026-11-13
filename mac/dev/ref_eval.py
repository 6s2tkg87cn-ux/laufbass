#!/usr/bin/env python3
"""Dev: compare detected bass notes with a reference given in bars/16th steps (e.g. typed from a tab).

The reference is a list of sections, each a list of bars, each bar a list of (step 0..15, midi).
It is placed on the song's beat grid (beats + firstDownbeat from song.json). The bar offset of the
reference (where the bass starts) is searched automatically, per section within +-2 bars as well,
so a section count that is off by one bar does not ruin the rest.

Scores per section: recall (reference notes found with the right pitch class / exact octave),
precision (detected notes that hit a reference note with the right pitch class).
"""
import numpy as np

TOL = 0.05  # s


def beat_time(beats, i):
    n = len(beats)
    if i < 0:
        return beats[0] + i * (beats[1] - beats[0])
    if i >= n - 1:
        return beats[-1] + (i - n + 1) * (beats[-1] - beats[-2])
    return beats[i]


def step_time(beats, fd, bar, step):
    i = fd + bar * 4 + step // 4
    t0, t1 = beat_time(beats, i), beat_time(beats, i + 1)
    return t0 + (step % 4) / 4 * (t1 - t0)


def place(sections, beats, fd, start_bar):
    ref, bar = [], start_bar
    for name, bars in sections:
        for b in bars:
            for step, p in b:
                ref.append((step_time(beats, fd, bar, step), p, name, bar))
            bar += 1
    return ref


def score(det, ref, local=False):
    """det: list of (t, p); ref: list of (t, p, section, bar)"""
    dt = np.array([d[0] for d in det]) if det else np.zeros(0)
    dp = np.array([d[1] for d in det]) if det else np.zeros(0, int)
    used = np.zeros(len(det), bool)
    res = {}
    for t, p, sec, _ in ref:
        r = res.setdefault(sec, dict(n=0, pc=0, ex=0, on=0))
        r["n"] += 1
        if not len(dt):
            continue
        cand = np.flatnonzero((np.abs(dt - t) <= TOL) & ~used)
        if not len(cand):
            continue
        r["on"] += 1
        good = [k for k in cand if dp[k] % 12 == p % 12]
        k = min(good or cand, key=lambda k: abs(dt[k] - t))
        used[k] = True
        if dp[k] % 12 == p % 12:
            r["pc"] += 1
            r["ex"] += dp[k] == p
    # precision: detected notes inside the reference time span
    lo, hi = (ref[0][0] - 0.05, ref[-1][0] + 0.3) if local else (ref[0][0] - 0.2, ref[-1][0] + 0.5)
    inside = (dt >= lo) & (dt <= hi) if len(dt) else np.zeros(0, bool)
    tot = {k: sum(r[k] for r in res.values()) for k in ("n", "pc", "ex", "on")}
    hit_pc = int(sum(1 for k in np.flatnonzero(inside & used)
                     if True))  # used notes are matched; count those with right pitch class below
    ok = 0
    for k in np.flatnonzero(inside & used):
        # matched notes: right pitch class if some ref note at that time has the same pc
        ok += any(abs(r[0] - dt[k]) <= TOL and r[1] % 12 == dp[k] % 12 for r in ref)
    prec = ok / max(1, int(inside.sum()))
    rec = tot["pc"] / max(1, tot["n"])
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(sections=res, recall_pc=rec, recall_exact=tot["ex"] / max(1, tot["n"]),
                onset_recall=tot["on"] / max(1, tot["n"]), precision_pc=prec, f1=f1, n_det_inside=int(inside.sum()))


def best_alignment(det, sections, beats, fd, search=range(-2, 5), slack=8):
    """Each section is placed on its own: start searched within +-slack bars of where the previous
    section ended (tab repeat counts like "19x" are often approximate), but never before the previous
    section started. Where sections overlap, the earlier one is cut (its repeat count was too high)."""
    starts, pos = [], None
    for i, sec in enumerate(sections):
        if pos is None:
            rng = search
        else:
            rng = range(max(starts[-1] + 1, pos - slack), pos + slack + 1)
        best = None
        for b0 in rng:
            s = score(det, place([sec], beats, fd, b0), local=True)
            if best is None or s["f1"] > best[1]:
                best = (b0, s["f1"])
        starts.append(best[0])
        pos = best[0] + len(sec[1])
    ref = []
    for k, (sec, b0) in enumerate(zip(sections, starts)):
        end = starts[k + 1] if k + 1 < len(starts) else 10 ** 6
        ref += [r for r in place([sec], beats, fd, b0) if r[3] < end]
    return starts[0], [b - starts[0] for b in starts], score(det, ref)


def place_shifted(sections, beats, fd, start_bar, shifts):
    ref, bar = [], start_bar
    for (name, bars), sh in zip(sections, shifts):
        b0 = bar + sh
        for j, b in enumerate(bars):
            for step, p in b:
                ref.append((step_time(beats, fd, b0 + j, step), p, name, b0 + j))
        bar += len(bars)
    return ref


def report(label, det, sections, beats, fd):
    off, shifts, s = best_alignment(det, sections, beats, fd)
    print(f"{label:28s} F1 {s['f1']:.3f} | richtige Tonklasse gefunden {100*s['recall_pc']:5.1f}% "
          f"(exakt {100*s['recall_exact']:5.1f}%, Einsatz {100*s['onset_recall']:5.1f}%) | "
          f"Praezision {100*s['precision_pc']:5.1f}% | Abschnitte ab Takt {[off + x for x in shifts]}")
    return s
