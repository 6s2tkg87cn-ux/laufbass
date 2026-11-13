# -*- coding: utf-8 -*-
"""Bass transcription v2: onset segmentation + two independent pitch estimators.

Idea ("lieber weniger Noten, die dafuer stimmen"):
  1. Note starts come from the bass stem itself (attack = loudness rise, basic-pitch onset activation,
     or a stable pitch change). Two equal eighths in a row are therefore two notes, even when played
     legato.
  2. Each segment gets a pitch from basic-pitch (note salience, octave chosen towards the fundamental)
     AND from pYIN (monophonic f0 tracker). Only when both agree on the pitch class is the note kept
     with full confidence; clear single-estimator cases are kept only if that estimator is very sure.
  3. Where the bass stem is (almost) silent, nothing is reported - Demucs leaves faint piano/guitar
     residue there, which basic-pitch would otherwise normalise up and "hear" as notes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 22050
HOP = 256                 # 11.6 ms; same frame rate as basic-pitch
FPS = SR / HOP
BP_MIDI0 = 21             # basic-pitch note bin 0 = A0
BP_FPS = 86.0             # basic-pitch output frame rate
P_MIN, P_MAX = 26, 67     # D1 (drop D) .. G4
GATE_DB = 20.0            # notes need the stem within 20 dB of its typical loud level


@dataclass
class BassNote:
    t: float
    d: float
    p: int
    v: int
    conf: float = 1.0

    @property
    def end(self) -> float:
        return self.t + self.d


def _peaks(x: np.ndarray, thr: float, min_dist: int, prom: float | None = None) -> np.ndarray:
    from scipy.signal import find_peaks
    idx, _ = find_peaks(x, height=thr, distance=max(1, min_dist), prominence=prom)
    return idx


def sixteenth_grid(beats):
    """(times, index) of all 16th positions between the beats; even index = 8th position."""
    if beats is None or len(beats) < 2:
        return None
    b = np.asarray(beats, dtype=float)
    t = (b[:-1, None] + np.diff(b)[:, None] * np.arange(4)[None, :] / 4).ravel()
    return t, np.tile(np.arange(4), len(b) - 1)


def pluck_flux(y: np.ndarray, length: int) -> np.ndarray:
    """Log spectral flux 0.4-3 kHz (against the max of the 2 previous frames), one value per frame."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP, center=True))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    L = np.log1p(100.0 * S[(f >= 400) & (f < 3000)])
    fl = np.maximum(0.0, L[:, 2:] - np.maximum(L[:, 1:-1], L[:, :-2])).sum(axis=0)
    fl = np.pad(fl, (2, 0))
    return fl[:length] if len(fl) >= length else np.pad(fl, (0, length - len(fl)))


def basic_pitch_activations(wav: Path):
    """Returns (note [frames, 88], onset [frames, 88]) of basic-pitch, or (None, None)."""
    try:
        from basic_pitch.inference import run_inference
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model
    except Exception:
        return None, None
    path = None
    try:
        import onnxruntime  # noqa: F401
        cand = build_icassp_2022_model_path(FilenameSuffix.onnx)
        if Path(cand).exists():
            path = cand
    except Exception:
        pass
    model = Model(path) if path else Model(build_icassp_2022_model_path(FilenameSuffix.coreml))
    out = run_inference(str(wav), model)
    return np.asarray(out["note"]), np.asarray(out["onset"])


def pyin_track(y: np.ndarray):
    """pYIN f0 at FPS (computed at 11.025 kHz for speed). Returns (midi float or nan, voiced prob)."""
    import librosa
    y2 = librosa.resample(y, orig_sr=SR, target_sr=SR // 2)
    f0, voiced, vprob = librosa.pyin(y2, fmin=36.0, fmax=420.0, sr=SR // 2, frame_length=1024,
                                     hop_length=HOP // 2, center=True)
    midi = np.where(voiced & np.isfinite(f0), librosa.hz_to_midi(np.where(np.isfinite(f0), f0, 1.0)), np.nan)
    return midi, np.nan_to_num(vprob)


def crepe_track(y: np.ndarray, model: str = "full"):
    """CREPE (neural f0 tracker) at FPS. Returns (midi float or nan, voicing confidence)."""
    import torch
    import torchcrepe
    import librosa
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    x = torch.from_numpy(y16.astype(np.float32))[None]
    with torch.no_grad():
        f0, per = torchcrepe.predict(x, 16000, hop_length=160, fmin=36.0, fmax=420.0, model=model,
                                     decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                     batch_size=1024, device="cpu")
    f0, per = f0[0].numpy(), per[0].numpy()
    per = torchcrepe.filter.median(torch.from_numpy(per)[None], 3)[0].numpy()
    t_src = np.arange(len(f0)) * 0.01
    n = int(np.ceil(len(y) / HOP)) + 1
    t_dst = np.arange(n) / FPS
    idx = np.clip(np.round(t_dst / 0.01).astype(int), 0, len(f0) - 1)
    f, pr = f0[idx], per[idx]
    midi = np.where((pr > 0.35) & (f > 0), librosa.hz_to_midi(np.maximum(f, 1.0)), np.nan)
    return midi, pr


PITCH_ENGINE = "pyin"   # "pyin" or "crepe"
FLUX_FLOOR = 0.08       # pluck click must reach this share of the song's strong clicks
GRID_F8, GRID_F16 = 1.6, 2.0   # click / local median needed on an 8th resp. 16th grid position


def transcribe(wav: Path, log=lambda m: None, return_debug: bool = False, keep_unsure: bool = False,
               beats=None):
    import librosa
    from scipy.ndimage import median_filter, maximum_filter1d

    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    n = int(np.ceil(len(y) / HOP)) + 1

    # --- loudness and gate
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP, center=True)[0]
    db = 20 * np.log10(rms + 1e-7)
    ref = float(np.percentile(db[db > -70], 90)) if np.any(db > -70) else -20.0
    gate = db > ref - GATE_DB                       # bass stem clearly active
    gate = maximum_filter1d(gate.astype(np.uint8), 5).astype(bool)

    # --- estimators
    note_act, onset_act = basic_pitch_activations(wav)
    midi_py, vprob = crepe_track(y) if PITCH_ENGINE == "crepe" else pyin_track(y)

    def fit(a, length):
        """basic-pitch frames are not evenly spaced in time (overlapping 2 s windows, ~0.5 % drift
        if read as plain frames): map them with basic-pitch's own frame->time function onto our axis."""
        if a is None:
            return None
        try:
            from basic_pitch.note_creation import model_frames_to_time
            t_src = model_frames_to_time(len(a))  # includes basic-pitch's per-window correction
        except Exception:
            t_src = np.arange(len(a)) / BP_FPS
        t_dst = np.arange(length) / FPS
        idx = np.clip(np.searchsorted(t_src, t_dst), 1, len(a) - 1)
        w = np.clip((t_dst - t_src[idx - 1]) / (t_src[idx] - t_src[idx - 1]), 0.0, 1.0)[:, None]
        return (1 - w) * a[idx - 1] + w * a[idx]

    m = min(len(rms), len(midi_py))
    rms, db, gate, midi_py, vprob = rms[:m], db[:m], gate[:m], midi_py[:m], vprob[:m]
    note_act, onset_act = fit(note_act, m), fit(onset_act, m)

    # --- onset candidates
    rise = np.zeros(m)
    rise[2:] = np.maximum(0.0, db[2:] - db[:-2])
    # primary: attack = loudness jump (timing is exact to one frame) ...
    cand = list(_peaks(rise, 3.0, int(0.06 * FPS), prom=2.0))
    strength = {i: rise[i] / 8.0 for i in cand}
    # ... or the pluck "click": spectral flux 0.4-3 kHz. Catches re-plucks of a compressed, legato bass
    # where the level dips only 1-2 dB (steady eighths): on real songs it separates every eighth-note
    # attack from the positions between them.
    flux = pluck_flux(y, m)
    # Beat grid as prior: bass players play on the grid, so a weaker click is enough on an 8th / 16th
    # position (unaccented eighths, a bass buried in the full band), off the grid a clear one is needed.
    g16 = sixteenth_grid(beats)
    def grid_factor(i):
        if g16 is None:
            return 2.5
        t = i / FPS
        j = int(np.searchsorted(g16[0], t))
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(g16[0]) and (best is None or abs(g16[0][k] - t) < abs(g16[0][best] - t)):
                best = k
        if best is None or abs(g16[0][best] - t) > 0.03:
            return 2.8
        return GRID_F8 if g16[1][best] % 2 == 0 else GRID_F16
    from scipy.ndimage import median_filter as _mf
    base = _mf(flux, size=int(1.0 * FPS) | 1, mode="nearest")
    floor_ = FLUX_FLOOR * float(np.percentile(flux[gate], 99)) if np.any(gate) else 0.0
    for i in _peaks(flux, 0.0, int(0.07 * FPS)):
        # a muted string clicks too, but then the level drops; after a real pluck it holds or rises
        # (a pluck never loses 3 dB within 35 ms of its attack, a mute always does)
        nxt = db[i + 1:min(m, i + 4)]
        if len(nxt) and float(nxt.min()) < float(db[max(0, i - 1)]) - 3.0:
            continue
        if flux[i] > max(grid_factor(i) * base[i], floor_) and gate[min(m - 1, i + 2)]:
            k = i
            near = [c for c in cand if abs(c - i) <= 2]
            if near:
                continue  # same attack already found via loudness
            strength[k] = min(1.5, flux[i] / max(4.0 * base[i], 1e-9))
            cand.append(k)
    prio = {i: 3 for i in cand}  # 3 = attack (loudness or pluck click), 2 = basic-pitch onset, 1 = pitch change
    if onset_act is not None:
        # secondary: basic-pitch onset activation (broad, so only clearly prominent peaks, and only
        # where no loudness attack was found nearby - e.g. a soft re-pluck under a ringing note)
        lo, hi = P_MIN - BP_MIDI0, P_MAX - BP_MIDI0 + 1
        on = onset_act[:, lo:hi].max(axis=1)
        prim = np.asarray(cand) if cand else np.zeros(0, dtype=int)
        for i in _peaks(on, 0.5, int(0.06 * FPS), prom=0.3):
            if len(prim) and np.min(np.abs(prim - i)) <= int(0.06 * FPS):
                continue
            strength[i] = on[i]
            prio[i] = 2
            cand.append(i)
    # stable pitch changes (legato / hammer-on): pyin rounded pitch differs before/after for >= 50 ms
    pr = np.round(median_filter(np.nan_to_num(midi_py, nan=0.0), size=5))
    w = int(0.05 * FPS)
    for i in range(w, m - w):
        a, b = pr[i - w:i], pr[i:i + w]
        if a[0] > 0 and b[-1] > 0 and np.all(a == a[0]) and np.all(b == b[-1]) and abs(a[0] - b[-1]) >= 1 and pr[i - 1] != pr[i]:
            cand.append(i)
            strength[i] = max(strength.get(i, 0.0), 0.8)
            prio.setdefault(i, 1)
    # a pitch change right after an attack is just pYIN catching up (its 93 ms window still hears the
    # previous note) - not a new note
    att = np.asarray(sorted(i for i in cand if prio.get(i, 0) >= 2))
    def after_attack(i):
        if not len(att):
            return False
        j = np.searchsorted(att, i, side="right") - 1
        return j >= 0 and i - att[j] <= int(0.15 * FPS)
    cand = [c for c in cand if prio.get(c, 0) >= 2 or not after_attack(c)]
    cand = sorted(set(int(c) for c in cand if gate[min(m - 1, c + 2)]))
    # merge candidates closer than 90 ms (keep the most reliable kind, then the strongest)
    on_frames: list[int] = []
    key = lambda i: (prio.get(i, 0), strength.get(i, 0.0))
    for c in cand:
        if on_frames and c - on_frames[-1] < int(0.09 * FPS):
            if key(c) > key(on_frames[-1]):
                on_frames[-1] = c
            continue
        on_frames.append(c)

    # --- segments -> notes
    notes: list[BassNote] = []
    for k, s in enumerate(on_frames):
        e_max = on_frames[k + 1] if k + 1 < len(on_frames) else m
        seg_db = db[s:e_max]
        if len(seg_db) < 3:
            continue
        pk = float(seg_db[: max(3, int(0.08 * FPS))].max())
        # note ends when the stem falls 18 dB below the attack peak or the gate closes
        e = e_max
        below = np.flatnonzero((seg_db < pk - 18.0) | ~gate[s:e_max])
        below = below[below > int(0.04 * FPS)]
        if len(below):
            e = s + int(below[0])
        dur = (e - s) / FPS
        if dur < 0.06:
            continue
        a0 = s + int(0.02 * FPS)
        a1 = s + max(int(0.03 * FPS) + 1, min(int(0.6 * (e - s)), int(0.25 * FPS)))
        a1 = min(a1, e)
        if a1 <= a0:
            a1 = a0 + 1

        # estimator 1: pYIN (its analysis window is 93 ms wide and centred: start 50 ms after the attack)
        p0 = min(s + int(0.05 * FPS), max(a0, a1 - 2))
        p1 = max(a1, p0 + 3)
        seg_py = midi_py[p0:min(p1, e)] if min(p1, e) > p0 else midi_py[a0:a1]
        voiced = np.isfinite(seg_py)
        p_py = int(np.round(np.median(seg_py[voiced]))) if voiced.mean() >= 0.5 else None
        vconf = float(vprob[p0:max(p0 + 1, min(p1, e))].mean())

        # estimator 2: basic-pitch salience, octave pulled towards the fundamental
        p_bp, bconf = None, 0.0
        if note_act is not None:
            sal = note_act[a0:a1].mean(axis=0)
            lo, hi = P_MIN - BP_MIDI0, P_MAX - BP_MIDI0 + 1
            band = sal[lo:hi]
            j = int(np.argmax(band))
            bconf = float(band[j])
            p = j + P_MIN
            while p - 12 >= P_MIN and sal[p - 12 - BP_MIDI0] >= 0.3 * bconf:
                p -= 12
            p_bp = p

        if p_py is not None and p_bp is not None:
            if p_py % 12 == p_bp % 12:
                p, conf = p_py, 1.0           # agree on the note; pYIN knows the octave better
            elif abs(p_py - p_bp) == 1 and bconf < 0.5:
                p, conf = p_py, 0.6
            elif bconf >= 0.6 and vconf < 0.4:
                p, conf = p_bp, 0.6
            else:
                continue                        # disagreement -> rather show nothing
        elif p_py is not None:
            if vconf < 0.5:
                continue
            p, conf = p_py, 0.6
        elif p_bp is not None:
            if bconf < 0.55:
                continue
            p, conf = p_bp, 0.6
        else:
            continue
        while p < 28:
            p += 12  # standard 4-string range (transpose in the app for lower tunings)
        while p > P_MAX:
            p -= 12
        vel = int(np.clip(40 + 87 * (pk - (ref - GATE_DB)) / 30.0, 1, 127))
        notes.append(BassNote(s / FPS, dur, int(p), vel, conf))

    # merge same-pitch neighbours split by a weak "onset" (loudness ripple, no new attack)
    out: list[BassNote] = []
    for nt in notes:
        if out and out[-1].p == nt.p and nt.t - out[-1].end < 0.03:
            f = int(round(nt.t * FPS))
            if prio.get(f, 0) < 3 and rise[max(0, f - 2):f + 3].max() < 2.0:
                out[-1].d = nt.end - out[-1].t
                continue
        out.append(nt)
    # isolated octave flicks between two equal neighbours
    for i in range(1, len(out) - 1):
        a, b, c = out[i - 1], out[i], out[i + 1]
        if a.p == c.p and abs(b.p - a.p) == 12 and b.d < 0.2 and b.conf < 1.0:
            b.p = a.p
    if not keep_unsure:
        # only one estimator was sure: on real songs these are mostly wrong (checked against an
        # independent CQT pitch estimate: ~16 % right vs. ~87 % when both agree) -> leave them out
        n_unsure = sum(1 for x in out if x.conf < 1.0)
        out = [x for x in out if x.conf >= 1.0]
        if n_unsure:
            log(f"{n_unsure} unsichere Bassnoten weggelassen")
    log(f"{len(out)} Bassnoten ({sum(1 for x in out if x.conf >= 1.0)} sicher), "
        f"Pegel-Schwelle {ref - GATE_DB:.0f} dBFS")
    if return_debug:
        return out, dict(db=db, gate=gate, midi_py=midi_py, vprob=vprob, onsets=on_frames)
    return out
