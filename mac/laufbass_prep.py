#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laufbass-Vorbereitung (macOS, Apple Silicon)

Wandelt einen Song (MP3/WAV/M4A/...) in ein Laufbass-Paket um:
    <Titel>.laufbass.zip  (siehe docs/FORMAT.md)

Pipeline:
    1. Stems trennen        (Demucs htdemucs: bass / drums / vocals / other)
    2. Bass transkribieren  (Spotify basic-pitch, danach monophon bereinigt)
    3. Drums transkribieren (librosa, Frequenzband-Onsets; austauschbar)
    4. Beats + Tempo        (librosa)
    5. MIDI + song.json + ZIP schreiben

Aufruf:
    python laufbass_prep.py song.mp3 [-o AUSGABEORDNER] [--keep-temp] [--model htdemucs]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Homebrew-Pfade ergaenzen (bei Doppelklick-Start fehlen sie oft im PATH)
for _p in ("/opt/homebrew/bin", "/usr/local/bin"):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _p

# Fehlende MPS-Operatoren automatisch auf CPU ausweichen lassen (muss vor "import torch" stehen)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

FORMAT_NAME = "laufbass"
FORMAT_VERSION = 1
STEM_NAMES = ("bass", "drums", "vocals", "other")
BASS_TUNING = [28, 33, 38, 43]  # E1 A1 D2 G2
BASS_MIN_MIDI, BASS_MAX_MIDI = 28, 67
SEP_SR = 44100  # Samplerate fuer Demucs / Analyse

# General-MIDI-Drum-Map laut FORMAT.md
GM_DRUMS = {
    "kick": 36, "snare": 38, "hihat": 42, "hihat_open": 46,
    "tom_low": 45, "tom_mid": 48, "tom_high": 50, "crash": 49, "ride": 51,
}


class LaufbassError(Exception):
    """Fehler mit verstaendlicher deutscher Meldung."""


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------
def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(n: int, total: int, text: str) -> float:
    say(f"\nSchritt {n}/{total}: {text}")
    return time.time()


def done(t0: float) -> None:
    say(f"   fertig in {time.time() - t0:.1f} s")


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise LaufbassError(
            "ffmpeg wurde nicht gefunden.\n"
            "   Bitte in der Konsole 'brew install ffmpeg' ausfuehren "
            "(oder install.sh erneut starten)."
        )
    return exe


def load_audio_ffmpeg(path: Path, sr: int = SEP_SR, channels: int = 2) -> np.ndarray:
    """Decode any audio file with ffmpeg -> float32 array (channels, samples)."""
    ffmpeg = check_ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise LaufbassError(f"Die Audiodatei '{path.name}' konnte nicht gelesen werden.\n   ({err[:300]})")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    data = data[: (len(data) // channels) * channels].reshape(-1, channels).T
    return np.ascontiguousarray(data)


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), data.T, sr, subtype="FLOAT")


def encode_mp3(src: Path, dst: Path, bitrate_kbps: int = 256) -> None:
    ffmpeg = check_ffmpeg()
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src), "-vn", "-codec:a", "libmp3lame",
           "-b:a", f"{bitrate_kbps}k", "-map_metadata", "-1", str(dst)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise LaufbassError("MP3-Kodierung fehlgeschlagen: " + proc.stderr.decode("utf-8", "replace")[:300])


def to_mono(wav: np.ndarray) -> np.ndarray:
    return wav.mean(axis=0) if wav.ndim == 2 else wav


# --------------------------------------------------------------------------
# Schritt 1: Stem-Trennung
# --------------------------------------------------------------------------
def _pick_device(torch):
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def separate_stems(audio_path: Path, tmp_dir: Path, model_name: str = "htdemucs") -> dict[str, Path]:
    """Run Demucs through its Python API; returns {stem: wav_path} (float WAV, 44.1 kHz stereo)."""
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as e:
        raise LaufbassError(
            "Demucs bzw. PyTorch ist nicht installiert (" + str(e) + ").\n"
            "   Bitte install.sh ausfuehren."
        )

    wav = load_audio_ffmpeg(audio_path, SEP_SR, 2)
    if wav.shape[1] < SEP_SR:
        raise LaufbassError("Die Audiodatei ist kuerzer als eine Sekunde.")

    say("   Lade Modell (beim allerersten Mal werden ca. 80 MB heruntergeladen) ...")
    try:
        model = get_model(model_name)
    except Exception as e:
        raise LaufbassError(
            f"Das Demucs-Modell '{model_name}' konnte nicht geladen werden.\n"
            f"   Beim ersten Start wird eine Internetverbindung gebraucht. ({e})"
        )
    model.eval()
    if getattr(model, "samplerate", SEP_SR) != SEP_SR:
        raise LaufbassError(f"Unerwartete Modell-Samplerate: {model.samplerate}")

    ref = wav.mean(0)
    mean, std = float(ref.mean()), float(ref.std()) or 1.0
    mix = torch.from_numpy((wav - mean) / std)[None]

    device = _pick_device(torch)

    def run(dev: str):
        with torch.no_grad():
            return apply_model(model, mix, device=dev, shifts=1, split=True,
                               overlap=0.25, progress=True)[0]

    say(f"   Rechne auf: {'Apple-GPU (MPS)' if device == 'mps' else 'CPU'}")
    try:
        out = run(device)
    except Exception as e:
        if device == "mps":
            say(f"   Hinweis: GPU-Berechnung fehlgeschlagen ({str(e)[:120]}). Versuche es mit der CPU - das dauert laenger.")
            try:
                model.to("cpu")
                out = run("cpu")
            except Exception as e2:
                raise LaufbassError(f"Die Stem-Trennung ist fehlgeschlagen: {e2}")
        else:
            raise LaufbassError(f"Die Stem-Trennung ist fehlgeschlagen: {e}")

    out = (out.cpu().numpy() * std + mean).astype(np.float32)
    stems: dict[str, Path] = {}
    for name, data in zip(model.sources, out):
        peak = float(np.abs(data).max())
        if peak > 0.99:
            data = data * (0.99 / peak)
        p = tmp_dir / f"{name}.wav"
        write_wav(p, data, SEP_SR)
        stems[name] = p
    return stems


def stems_from_dir(stems_dir: Path, tmp_dir: Path) -> dict[str, Path]:
    """Dev helper: use ready-made stems (wav/mp3) instead of running Demucs."""
    stems: dict[str, Path] = {}
    for name in STEM_NAMES:
        for ext in (".wav", ".mp3", ".flac"):
            p = stems_dir / (name + ext)
            if p.exists():
                stems[name] = p
                break
    if "bass" not in stems or "drums" not in stems:
        raise LaufbassError(f"In {stems_dir} fehlen bass.* und/oder drums.*")
    return stems


# --------------------------------------------------------------------------
# Schritt 2: Bass-Transkription
# --------------------------------------------------------------------------
@dataclass
class Note:
    t: float  # start (s)
    d: float  # duration (s)
    p: int    # MIDI pitch
    v: int    # velocity 1..127

    @property
    def end(self) -> float:
        return self.t + self.d


def make_monophonic(notes: list[Note], tol: float = 0.03) -> list[Note]:
    """Reduce to one voice. On overlap keep the louder/longer note; for a clean
    successor note (overlap only at the tail) the earlier note is just cut off."""
    notes = sorted(notes, key=lambda n: (n.t, -n.v))
    out: list[Note] = []
    for n in notes:
        if not out or n.t >= out[-1].end - tol:
            if out and n.t < out[-1].end:  # tiny overlap -> trim
                out[-1].d = max(0.0, n.t - out[-1].t)
            out.append(n)
            continue
        prev = out[-1]
        # n starts before prev has ended
        if n.t - prev.t < 0.06 or n.end <= prev.end + tol:
            # near-simultaneous or n is fully inside prev: keep the stronger one (loudness * length)
            if n.v * n.d > prev.v * prev.d:
                out[-1] = n
            # else: drop n
        else:
            # n begins clearly later and extends beyond prev: new onset wins, cut prev
            prev.d = n.t - prev.t
            out.append(n)
    return [n for n in out if n.d > 0]


def merge_fragments(notes: list[Note], max_gap: float = 0.015) -> list[Note]:
    """A decaying bass note is often split into several same-pitch pieces by the model. Merge
    same-pitch neighbours that touch (gap <= 15 ms)."""
    out: list[Note] = []
    for n in notes:
        if out and out[-1].p == n.p and n.t - out[-1].end <= max_gap:
            prev = out[-1]
            prev.d = n.end - prev.t
            prev.v = max(prev.v, n.v)
        else:
            out.append(n)
    return out


def fix_octave_jumps(notes: list[Note], max_gap: float = 0.5, max_dur: float = 0.4) -> list[Note]:
    """A single short note far from both neighbours (which are close to each other) is an artefact:
    exactly an octave away -> shift it by an octave; other large jumps and very short -> drop."""
    drop: set[int] = set()
    for i in range(1, len(notes) - 1):
        a, b, c = notes[i - 1], notes[i], notes[i + 1]
        if b.t - a.end > max_gap or c.t - b.end > max_gap or b.d > max_dur:
            continue
        if abs(a.p - c.p) > 2:
            continue
        diff = b.p - (a.p + c.p) / 2
        if abs(abs(diff) - 12) <= 2:
            b.p += -12 if diff > 0 else 12
        elif abs(diff) >= 9 and b.d < 0.15:
            drop.add(i)  # e.g. a harmonic (octave + fifth) flashing up for a moment
    return [n for i, n in enumerate(notes) if i not in drop]


def fold_to_bass_range(notes: list[Note]) -> None:
    for n in notes:
        while n.p < BASS_MIN_MIDI:
            n.p += 12
        while n.p > BASS_MAX_MIDI:
            n.p -= 12


def postprocess_bass(raw: list[Note], min_dur: float = 0.06) -> list[Note]:
    notes = [n for n in raw if n.d >= min_dur]
    notes = make_monophonic(notes)
    notes = [n for n in notes if n.d >= min_dur]
    notes = merge_fragments(notes)
    notes = fix_octave_jumps(notes)
    fold_to_bass_range(notes)
    for n in notes:
        n.v = int(min(127, max(1, n.v)))
    return notes


def refine_onsets(notes: list[Note], bass_wav: Path, back: float = 0.08, fwd: float = 0.10) -> None:
    """basic-pitch onsets are quantised to ~12 ms frames and tend to sit early. Snap each onset to the
    steepest rise of the stem's loudness envelope near the detected start (in place)."""
    import librosa
    y, sr = librosa.load(str(bass_wav), sr=22050, mono=True)
    hop = 128
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-6)
    rise = np.maximum(0.0, db[2:] - db[:-2])  # 2-frame slope, index i -> frame i+1
    for i, n in enumerate(notes):
        end = n.end
        lo = max(0, int((n.t - back) * sr / hop))
        hi = min(len(rise), int((n.t + fwd) * sr / hop) + 1)
        if hi - lo < 3:
            continue
        seg = rise[lo:hi]
        j = int(np.argmax(seg))
        if seg[j] < 3.0:  # no clear attack (legato) -> keep
            continue
        new_t = (lo + j + 1) * hop / sr + 0.006
        prev_end = notes[i - 1].end if i else 0.0
        if new_t < end - 0.04:
            n.t = max(new_t, 0.0)
            n.d = end - n.t
            if i and prev_end > n.t:
                notes[i - 1].d = max(0.02, n.t - notes[i - 1].t)


def transcribe_bass_basic_pitch(bass_wav: Path) -> list[Note]:
    try:
        from basic_pitch.inference import predict
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    except Exception as e:
        raise LaufbassError(
            "basic-pitch ist nicht installiert oder konnte nicht geladen werden "
            f"({e}).\n   Bitte install.sh ausfuehren."
        )
    # Prefer ONNX (robust on Apple Silicon); fall back to the package default (CoreML).
    model_path = None
    try:
        import onnxruntime  # noqa: F401
        cand = build_icassp_2022_model_path(FilenameSuffix.onnx)
        if Path(cand).exists():
            model_path = cand
    except Exception:
        pass
    kwargs = dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60,
                  minimum_frequency=30.0, maximum_frequency=400.0, melodia_trick=True)
    try:
        if model_path is not None:
            _, _, events = predict(str(bass_wav), model_path, **kwargs)
        else:
            _, _, events = predict(str(bass_wav), **kwargs)
    except Exception as e:
        raise LaufbassError(f"Die Bass-Erkennung (basic-pitch) ist fehlgeschlagen: {e}")
    raw = []
    for ev in events:
        s, e_, pitch, amp = float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3])
        raw.append(Note(s, e_ - s, pitch, int(round(40 + 80 * min(1.0, max(0.0, amp))))))
    notes = postprocess_bass(raw)
    try:
        refine_onsets(notes, bass_wav)
    except Exception:
        pass  # refinement is optional
    return notes


def transcribe_bass_pyin(bass_wav: Path) -> list[Note]:
    """TEST ONLY: monophonic pYIN tracker (no basic-pitch needed)."""
    import librosa
    y, sr = librosa.load(str(bass_wav), sr=22050, mono=True)
    hop = 256
    f0, voiced, _ = librosa.pyin(y, fmin=30, fmax=400, sr=sr, frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    midi = np.full(n, np.nan)
    ok = voiced[:n] & ~np.isnan(f0[:n])
    midi[ok] = np.round(librosa.hz_to_midi(f0[:n][ok]))
    rms_ref = max(float(rms.max()), 1e-9)
    notes: list[Note] = []
    cur, start = None, 0
    for i in range(n + 1):
        m = midi[i] if i < n else np.nan
        # also split a note on a sharp energy rise (re-pluck of the same pitch)
        rise = i < n and i > 2 and cur is not None and rms[i] > 1.8 * rms[i - 2] and rms[i] > 0.1 * rms_ref
        if cur is not None and (np.isnan(m) or int(m) != cur or rise):
            notes.append(_pyin_note(cur, start, i, hop, sr, rms, rms_ref))
            cur = None
        if cur is None and i < n and not np.isnan(m):
            cur, start = int(m), i
    return postprocess_bass(notes)


def _pyin_note(pitch, i0, i1, hop, sr, rms, rms_ref) -> Note:
    seg = rms[i0:i1]
    v = 30 + 97 * float(seg.max()) / rms_ref
    return Note(i0 * hop / sr, (i1 - i0) * hop / sr, pitch, int(min(127, v)))


# --------------------------------------------------------------------------
# Schritt 3: Drum-Transkription (librosa, bandweise)
# --------------------------------------------------------------------------
@dataclass
class DrumHit:
    t: float
    k: str
    v: int


D_NFFT = 1024
D_HOP = 256
D_GAIN = 8.0  # log compression gain; low value keeps kick-band flux from being flooded by snare transients
BANDS = {
    "kick": (30.0, 150.0),
    "snare_body": (150.0, 1000.0),
    "snare_noise": (1500.0, 5000.0),
    "hihat": (6000.0, 16000.0),
}


def drum_band_envelopes(y: np.ndarray, sr: int) -> tuple[dict[str, np.ndarray], float]:
    """Band-wise spectral-flux onset envelopes. Returns ({band: env}, frame_duration_s)."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=D_NFFT, hop_length=D_HOP, center=True))
    peak = float(S.max())
    if peak <= 0:
        z = np.zeros(S.shape[1])
        return {b: z for b in BANDS}, D_HOP / sr
    S = S / peak
    L = np.log1p(D_GAIN * S)  # mild compression
    freqs = librosa.fft_frequencies(sr=sr, n_fft=D_NFFT)
    # flux against the max of the two previous frames (suppresses sustained energy)
    prev = np.maximum(np.pad(L, ((0, 0), (1, 0)))[:, :-1], np.pad(L, ((0, 0), (2, 0)))[:, :-2])
    flux = np.maximum(0.0, L - prev)
    env = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        env[name] = flux[m].mean(axis=0)
    return env, D_HOP / sr


def _local_maxima(env: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d
    mx = maximum_filter1d(env, size=2 * radius + 1, mode="nearest")
    return np.flatnonzero((env == mx) & (env > 0))


def _pick_peaks(env: np.ndarray, frame_s: float, rel_thresh: float = 0.2,
                abs_floor: float = 0.006, min_sep_s: float = 0.04) -> tuple[np.ndarray, float]:
    """Adaptive peak picking. Returns (frame indices, scale) where scale ~ level of a typical strong hit."""
    from scipy.ndimage import uniform_filter1d
    cand = _local_maxima(env, max(1, int(round(min_sep_s / frame_s))))
    cand = cand[env[cand] > abs_floor]
    if len(cand) == 0:
        return cand, 0.0
    vals = env[cand]
    p95 = float(np.percentile(vals, 95))
    scale = float(np.median(vals[vals >= 0.25 * p95]))  # median of the strong peaks
    local = uniform_filter1d(env, size=int(0.4 / frame_s) | 1, mode="nearest")
    thr = np.maximum(rel_thresh * scale, 2.0 * local[cand])
    return cand[vals > thr], scale


def transcribe_drums_bands(drum_wav: Path, return_debug: bool = False):
    """Classical band-wise drum detector (kick / snare / hihat / crash).

    This function is the single swap point: replace it by a neural model
    (e.g. ADTOF) as long as it returns a list of DrumHit.
    """
    import librosa
    y, sr = librosa.load(str(drum_wav), sr=SEP_SR, mono=True)
    env, fs = drum_band_envelopes(y, sr)

    # --- onset candidates per band
    kick_f, kick_scale = _pick_peaks(env["kick"], fs)
    body_f, body_scale = _pick_peaks(env["snare_body"], fs)
    noise_f, noise_scale = _pick_peaks(env["snare_noise"], fs)
    hat_f, hat_scale = _pick_peaks(env["hihat"], fs)

    def strength(name: str, f: int, scale: float, w: int = 2) -> float:
        if scale <= 0:
            return 0.0
        lo, hi = max(0, f - w), min(len(env[name]), f + w + 1)
        return float(env[name][lo:hi].max() / scale)

    # unified onset list (frames), merged within 30 ms
    allf = np.unique(np.concatenate([kick_f, body_f, noise_f, hat_f])) if len(kick_f) + len(body_f) + len(noise_f) + len(hat_f) else np.array([], dtype=int)
    merged: list[int] = []
    for f in allf:
        if merged and (f - merged[-1]) * fs < 0.03:
            continue
        merged.append(int(f))

    hits: list[DrumHit] = []
    raw = []
    for f in merged:
        k = strength("kick", f, kick_scale)
        sb = strength("snare_body", f, body_scale)
        sn = strength("snare_noise", f, noise_scale)
        h = strength("hihat", f, hat_scale)
        raw.append((f, k, sb, sn, h))

    snare_k = []  # kick-band leak at snare hits (used to tell real kicks from snare transients)
    classified = []
    for f, k, sb, sn, h in raw:
        snare_score = min(sb, sn)  # body AND noise band must fire
        is_snare = snare_score >= 0.5 and sn >= 0.5
        classified.append([f, k, sb, sn, h, is_snare])
        if is_snare:
            snare_k.append(k)
    # a snare transient also leaks into the kick band; a kick coinciding with a snare must clearly exceed that leak
    leak = float(np.percentile(snare_k, 50)) if snare_k else 0.0
    snare_times = []
    for f, k, sb, sn, h, is_snare in classified:
        t = f * fs
        is_kick = k >= 0.5 and (not is_snare or k >= max(1.0, 1.8 * leak))
        # the snare noise also lights up the hihat band, so with a snare only a very strong top band counts
        is_hat = h >= 0.5 and (not is_snare or h >= 2.5)
        if is_kick:
            hits.append(DrumHit(t, "kick", _vel(k)))
        if is_snare:
            hits.append(DrumHit(t, "snare", _vel(min(sb, sn))))
            snare_times.append(t)
        if is_hat:
            hits.append(DrumHit(t, "hihat", _vel(h)))
    hits.extend(_infill_hats_under_snares(hits, snare_times))

    # --- crash: very strong top-band onset whose HF energy keeps ringing (conservative)
    hits.extend(_detect_crashes(y, sr, [(f, h) for f, _, _, _, h in raw], fs))
    hits.sort(key=lambda x: (x.t, x.k))
    if return_debug:
        return hits, env, fs
    return hits


def _infill_hats_under_snares(hits: list[DrumHit], snare_times: list[float]) -> list[DrumHit]:
    """Hats are hidden by snare noise. If the hat pattern is steady (hat before and after a snare at
    equal distance), assume the hat also plays on the snare."""
    hat_t = np.array(sorted(h.t for h in hits if h.k == "hihat"))
    hat_v = {round(h.t, 4): h.v for h in hits if h.k == "hihat"}
    out: list[DrumHit] = []
    if len(hat_t) < 4:
        return out
    for t in snare_times:
        if np.any(np.abs(hat_t - t) < 0.04):
            continue
        i = int(np.searchsorted(hat_t, t))
        if i == 0 or i >= len(hat_t):
            continue
        dp, dn = t - hat_t[i - 1], hat_t[i] - t
        if 0.12 <= dp <= 0.7 and abs(dp - dn) < 0.03:
            out.append(DrumHit(t, "hihat", int((hat_v[round(hat_t[i - 1], 4)] + hat_v[round(hat_t[i], 4)]) / 2)))
    return out


def _vel(strength_ratio: float) -> int:
    r = min(max(strength_ratio, 0.2), 1.4) / 1.4
    return int(round(30 + r * 97))


def _detect_crashes(y: np.ndarray, sr: int, cand: list[tuple[int, float]], fs: float) -> list[DrumHit]:
    import librosa
    S = np.abs(librosa.stft(y, n_fft=D_NFFT, hop_length=D_HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=D_NFFT)
    hf = S[(freqs >= 6000) & (freqs < 16000)].sum(axis=0)
    out: list[DrumHit] = []
    last_t = -10.0
    for f, h in cand:
        if h < 3.0:  # needs to be much stronger than a normal hihat hit
            continue
        t = f * fs
        pk = hf[max(0, f - 1): f + 3].max()
        a, b = int(0.25 / fs), int(0.9 / fs)
        tail = hf[f + a: f + b]
        if pk <= 0 or len(tail) < (b - a) // 2:
            continue
        if tail.mean() / pk >= 0.30 and t - last_t > 1.5:
            out.append(DrumHit(t, "crash", _vel(h / 3.0)))
            last_t = t
    return out


# --------------------------------------------------------------------------
# Schritt 4: Beats / Tempo / erster Taktschlag
# --------------------------------------------------------------------------
def track_beats(mix_mono: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    import librosa
    y = librosa.resample(mix_mono, orig_sr=sr, target_sr=22050) if sr != 22050 else mix_mono
    hop = 256
    oenv = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=22050, hop_length=hop, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=22050, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beats) >= 8:
        tempo = float(60.0 / np.median(np.diff(beats)))
    return tempo, np.asarray(beats, dtype=float)


def find_first_downbeat(beats: np.ndarray, kick_env: np.ndarray, frame_s: float, beats_per_bar: int = 4) -> int:
    """Index (0..3) whose beats carry the highest average kick strength."""
    if len(beats) < beats_per_bar:
        return 0
    strengths = []
    for b in beats:
        f = int(round(b / frame_s))
        lo, hi = max(0, f - 4), min(len(kick_env), f + 5)
        strengths.append(float(kick_env[lo:hi].max()) if hi > lo else 0.0)
    strengths = np.asarray(strengths)
    means = [strengths[i::beats_per_bar].mean() for i in range(beats_per_bar)]
    return int(np.argmax(means))


# --------------------------------------------------------------------------
# Schritt 5: MIDI, song.json, ZIP
# --------------------------------------------------------------------------
def write_bass_midi(path: Path, notes: list[Note], tempo: float) -> None:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=33, name="Bass")  # Electric Bass (finger)
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(n.v), pitch=int(n.p), start=n.t, end=n.t + max(n.d, 0.02)))
    pm.instruments.append(inst)
    pm.write(str(path))


def write_drums_midi(path: Path, hits: list[DrumHit], tempo: float) -> None:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")  # -> MIDI channel 10
    for h in hits:
        inst.notes.append(pretty_midi.Note(velocity=int(h.v), pitch=GM_DRUMS[h.k], start=h.t, end=h.t + 0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def build_song_json(title: str, source: str, duration: float, tempo: float, beats: np.ndarray,
                    first_downbeat: int, stems: list[str], notes: list[Note], hits: list[DrumHit]) -> dict:
    r3 = lambda x: round(float(x), 3)
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "title": title,
        "source": source,
        "duration": r3(duration),
        "tempo": round(float(tempo), 1),
        "timeSignature": [4, 4],
        "beats": [r3(b) for b in sorted(beats)],
        "firstDownbeat": int(first_downbeat),
        "stems": {s: f"{s}.mp3" for s in STEM_NAMES if s in stems},
        "bass": {
            "tuning": BASS_TUNING,
            "notes": [{"t": r3(n.t), "d": r3(n.d), "p": int(n.p), "v": int(n.v)}
                      for n in sorted(notes, key=lambda n: n.t)],
        },
        "drums": {
            "hits": [{"t": r3(h.t), "k": h.k, "v": int(h.v)} for h in sorted(hits, key=lambda h: h.t)],
        },
    }


def write_package(zip_path: Path, files: dict[str, Path]) -> None:
    """files: {archive name: path}. MP3 stored, everything else deflated, flat structure."""
    order = sorted(files, key=lambda n: (n != "song.json", n))
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in order:
            comp = zipfile.ZIP_STORED if name.endswith(".mp3") else zipfile.ZIP_DEFLATED
            z.write(files[name], arcname=name, compress_type=comp)


# --------------------------------------------------------------------------
# Gesamtablauf
# --------------------------------------------------------------------------
def process_file(audio: Path, out_dir: Path, model: str = "htdemucs", keep_temp: bool = False,
                 stems_dir: Path | None = None, bass_engine: str = "basic-pitch",
                 bitrate: int = 256) -> Path:
    t_all = time.time()
    title = audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{title}.laufbass.zip"
    tmp = Path(tempfile.mkdtemp(prefix="laufbass_"))
    TOTAL = 5
    say(f"\n=== {audio.name} ===")
    try:
        check_ffmpeg()

        # 1) stems
        t0 = step(1, TOTAL, "Instrumente trennen (Bass, Schlagzeug, Gesang, Rest) - das dauert am laengsten ...")
        stems = stems_from_dir(stems_dir, tmp) if stems_dir else separate_stems(audio, tmp, model)
        done(t0)

        # 2) bass
        t0 = step(2, TOTAL, "Basslinie erkennen ...")
        if bass_engine == "pyin":
            notes = transcribe_bass_pyin(stems["bass"])
        else:
            notes = transcribe_bass_basic_pitch(stems["bass"])
        say(f"   {len(notes)} Bassnoten gefunden")
        done(t0)

        # 3) drums
        t0 = step(3, TOTAL, "Schlagzeug erkennen ...")
        hits = transcribe_drums_bands(stems["drums"])
        counts = {k: sum(1 for h in hits if h.k == k) for k in GM_DRUMS}
        say("   " + ", ".join(f"{v} {k}" for k, v in counts.items() if v) if hits else "   keine Schlaege gefunden")
        done(t0)

        # 4) beats
        t0 = step(4, TOTAL, "Tempo und Taktschlaege bestimmen ...")
        mix = load_audio_ffmpeg(audio, SEP_SR, 1)[0]
        duration = len(mix) / SEP_SR
        tempo, beats = track_beats(mix, SEP_SR)
        import librosa
        d_y, _ = librosa.load(str(stems["drums"]), sr=SEP_SR, mono=True)
        env, fs = drum_band_envelopes(d_y, SEP_SR)
        first_db = find_first_downbeat(beats, env["kick"], fs)
        say(f"   Tempo ca. {tempo:.1f} BPM, {len(beats)} Schlaege, erste '1' = Schlag {first_db + 1}")
        done(t0)

        # 5) package
        t0 = step(5, TOTAL, "Paket schreiben (MP3-Stems, MIDI, song.json) ...")
        files: dict[str, Path] = {}
        for name, src in stems.items():
            dst = tmp / f"{name}.mp3"
            if src.suffix.lower() == ".mp3":
                shutil.copyfile(src, dst)
            else:
                encode_mp3(src, dst, bitrate)
            files[f"{name}.mp3"] = dst
        write_bass_midi(tmp / "bass.mid", notes, tempo)
        write_drums_midi(tmp / "drums.mid", hits, tempo)
        files["bass.mid"], files["drums.mid"] = tmp / "bass.mid", tmp / "drums.mid"
        song = build_song_json(title, audio.name, duration, tempo, beats, first_db, list(stems), notes, hits)
        (tmp / "song.json").write_text(json.dumps(song, ensure_ascii=False, indent=1), encoding="utf-8")
        files["song.json"] = tmp / "song.json"
        write_package(zip_path, files)
        done(t0)

        mb = zip_path.stat().st_size / 1e6
        say(f"\nFertig: {zip_path}  ({mb:.1f} MB, gesamt {time.time() - t_all:.0f} s)")
        return zip_path
    finally:
        if keep_temp:
            say(f"   Temporaere Dateien bleiben erhalten in: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Laufbass: Song fuer die Uebe-App vorbereiten")
    ap.add_argument("songs", nargs="+", help="Audiodatei(en), z. B. song.mp3")
    ap.add_argument("-o", "--outdir", help="Ausgabeordner (Standard: Ordner der Eingabedatei)")
    ap.add_argument("--keep-temp", action="store_true", help="Temporaere Dateien behalten")
    ap.add_argument("--model", default="htdemucs", help="Demucs-Modell (Standard: htdemucs)")
    # developer options (hidden)
    ap.add_argument("--stems-dir", help=argparse.SUPPRESS)
    ap.add_argument("--bass-engine", choices=["basic-pitch", "pyin"], default="basic-pitch", help=argparse.SUPPRESS)
    ap.add_argument("--bitrate", type=int, default=256, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    failed = 0
    for s in args.songs:
        p = Path(s).expanduser()
        if not p.is_file():
            say(f"\nFEHLER: Datei nicht gefunden: {p}")
            failed += 1
            continue
        out = Path(args.outdir).expanduser() if args.outdir else p.resolve().parent
        try:
            process_file(p, out, args.model, args.keep_temp,
                         Path(args.stems_dir) if args.stems_dir else None,
                         args.bass_engine, args.bitrate)
        except LaufbassError as e:
            say(f"\nFEHLER bei '{p.name}': {e}")
            failed += 1
        except KeyboardInterrupt:
            say("\nAbgebrochen.")
            return 130
        except Exception as e:  # unexpected
            say(f"\nUNERWARTETER FEHLER bei '{p.name}': {e}")
            if os.environ.get("LAUFBASS_DEBUG"):
                traceback.print_exc()
            else:
                say("   (Mit LAUFBASS_DEBUG=1 gibt es mehr Details.)")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
