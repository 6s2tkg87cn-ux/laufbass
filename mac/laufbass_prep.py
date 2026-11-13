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
    4. Beats + Tempo        (Beat This! auf dem Mix, dann Raster-Glaettung + Phasenkorrektur)
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
           "-b:a", f"{bitrate_kbps}k", "-map_metadata", "-1",
           # Xing/LAME-Header mit Encoder-Delay und Padding: Browser schneiden damit die Verzoegerung
           # des MP3-Encoders weg (sonst waeren die Stems ca. 25 ms gegen das Raster verschoben)
           "-write_xing", "1", str(dst)]
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


def transcribe_bass_v2(bass_wav: Path) -> list[Note]:
    """Default: onset segmentation + basic-pitch AND pYIN must agree (see bass_v2.py).
    Falls back to the plain basic-pitch path if something goes wrong."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import bass_v2
        res = bass_v2.transcribe(bass_wav, log=lambda m: say("   " + m))
        return [Note(n.t, n.d, n.p, n.v) for n in res]
    except Exception as e:
        say(f"   WARNUNG: neue Bass-Erkennung fehlgeschlagen ({str(e)[:160]}), nehme basic-pitch direkt.")
        return transcribe_bass_basic_pitch(bass_wav)


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


ADTOF_CLASSES = ("kick", "snare", "tom_mid", "hihat", "crash")  # order of the model outputs (GM 35 38 47 42 49)
ADTOF_THRESHOLDS = (0.22, 0.24, 0.32, 0.22, 0.30)               # per class, from the ADTOF paper
ADTOF_FPS = 100
ADTOF_LAG_S = 0.005  # ADTOF peaks sit ~5 ms before the attack (measured against the flux onsets)


def transcribe_drums_adtof(drum_wav: Path, return_activations: bool = False):
    """Neural drum transcription (ADTOF Frame-RNN, Zehren et al. 2021/2023, PyTorch port).
    Trained on real recordings: tells kick from snare reliably, also finds toms and cymbals.
    Returns a list of DrumHit (and the activation matrix [frames, 5] if requested)."""
    try:
        import torch
        from adtof_pytorch import (create_frame_rnn_model, calculate_n_bins, load_audio_for_model,
                                   load_pytorch_weights, get_default_weights_path, PeakPicker)
    except Exception as e:
        raise LaufbassError(f"Das Drum-Modell (ADTOF) ist nicht installiert ({e}).")
    import contextlib, io
    model = create_frame_rnn_model(calculate_n_bins())
    model.eval()
    w = get_default_weights_path()
    if not w or not Path(w).exists():
        raise LaufbassError("Die Gewichte des Drum-Modells (ADTOF) fehlen.")
    with contextlib.redirect_stdout(io.StringIO()):
        model = load_pytorch_weights(model, w, strict=False)
    x = load_audio_for_model(str(drum_wav))
    with torch.no_grad():
        act = model(x).cpu().numpy()[0]  # [frames, 5]
    picker = PeakPicker(thresholds=list(ADTOF_THRESHOLDS), fps=ADTOF_FPS)
    peaks = picker.pick(act[None], labels=list(range(5)), label_offset=0)[0]
    hits: list[DrumHit] = []
    for c, name in enumerate(ADTOF_CLASSES):
        for t in peaks.get(c, []):
            f = int(round(t * ADTOF_FPS))
            a = float(act[max(0, f - 1): f + 2, c].max())
            hits.append(DrumHit(float(t) + ADTOF_LAG_S, name, int(round(35 + 92 * min(1.0, a)))))
    # ADTOF has one class for all cymbals. A cymbal played as a steady stream (>= 3 more hits within
    # +-1.2 s) keeps time -> ride; a single accent -> crash.
    cym = np.array(sorted(h.t for h in hits if h.k == "crash"))
    for h in hits:
        if h.k == "crash" and int(np.sum(np.abs(cym - h.t) <= 1.2)) - 1 >= 3:
            h.k = "ride"
    hits.sort(key=lambda h: (h.t, h.k))
    return (hits, act) if return_activations else hits


def transcribe_drums(drum_wav: Path) -> tuple[list[DrumHit], str]:
    """ADTOF if available, else the band-wise fallback. Returns (hits, engine name)."""
    try:
        return transcribe_drums_adtof(drum_wav), "adtof"
    except LaufbassError as e:
        say(f"   WARNUNG: {e} Es wird das einfachere Verfahren benutzt (Kick/Snare weniger zuverlaessig).")
        return transcribe_drums_bands(drum_wav), "bands"


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
# Schritt 4: Beats / Tempo / Takt
#   a) Rohe Schlaege + Downbeats vom KI-Modell "Beat This!" (Mix, nicht Stem)
#      - Ausweichloesung: librosa
#   b) Raster-Glaettung: konstantes Tempo (robuste Regression) oder gleitende Regression
#   c) Phasenkorrektur gegen die Schlagzeug-Onsets (Kick/Snare)
#   d) Taktart und erste "Eins" aus den Modell-Downbeats
# --------------------------------------------------------------------------
BEAT_FIXED_RMS_MS = 12.0   # Rest-Abweichung, unter der ein festes Raster gewaehlt wird ...
BEAT_FIXED_P95_MS = 30.0   # ... und keine Stelle weiter als das danebenliegt
PHASE_MAX_SHIFT_S = 0.040  # Suchbereich der Phasenkorrektur (+-40 ms)
# Systematischer Versatz zwischen Flux-Peak und echtem Einsatz (s). Mit mac/dev/synth_test.py
# (bekannte Kicks/Snares) gemessen: ca. -1 ms, also deutlich unter einem Frame (5,8 ms) -> 0.
ENV_LAG_S = 0.0


def detect_beats_beat_this(mix: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Beat This! (CPJKU, ISMIR 2024) on the full mix. Returns (beats, downbeats) in seconds."""
    try:
        import torch  # noqa: F401
        from beat_this.inference import Audio2Beats
    except Exception as e:
        raise LaufbassError(f"Das Beat-Modell (beat_this) konnte nicht geladen werden ({e}).")
    try:
        # CPU: das Modell ist klein (ca. 20 MB), ein Song dauert nur wenige Sekunden - und die CPU ist
        # zuverlaessiger als MPS (Rotary-Embeddings). Beim ersten Mal wird der Checkpoint geladen.
        tracker = Audio2Beats(checkpoint_path="final0", device="cpu", float16=False, dbn=False)
    except Exception as e:
        raise LaufbassError(
            f"Die Gewichte des Beat-Modells konnten nicht geladen werden ({e}). "
            "Beim ersten Start wird eine Internetverbindung gebraucht.")
    try:
        beats, downbeats = tracker(np.ascontiguousarray(mix, dtype=np.float32), sr)
    except Exception as e:
        raise LaufbassError(f"Die Beat-Erkennung (beat_this) ist fehlgeschlagen ({e}).")
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def detect_beats_librosa(mix: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    y = librosa.resample(mix, orig_sr=sr, target_sr=22050) if sr != 22050 else mix
    hop = 256
    oenv = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=22050, hop_length=hop, units="frames")
    return np.asarray(librosa.frames_to_time(beat_frames, sr=22050, hop_length=hop), dtype=float)


def detect_beats(mix: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Returns (raw beats, model downbeats or None, engine name)."""
    try:
        beats, downbeats = detect_beats_beat_this(mix, sr)
        if len(beats) < 8:
            raise LaufbassError("das Modell hat zu wenige Schlaege gefunden")
        return beats, downbeats, "beat_this"
    except LaufbassError as e:
        say(f"   WARNUNG: {e}")
        say("   Es wird ersatzweise das einfachere Verfahren (librosa) benutzt - das Raster kann dann")
        say("   weniger genau sein und die erste '1' wird nur geraten. Bitte install.sh erneut ausfuehren.")
        return detect_beats_librosa(mix, sr), None, "librosa"


def clean_beat_sequence(beats: np.ndarray) -> np.ndarray:
    """Insert beats missing in gaps of ~2x the local interval, drop beats that follow far too early."""
    from scipy.ndimage import median_filter
    b = np.sort(np.asarray(beats, dtype=float))
    if len(b) < 5:
        return b
    d = np.diff(b)
    loc = median_filter(d, size=9, mode="nearest")  # local interval, index i = after beat i
    out = [b[0]]
    for i in range(1, len(b)):
        L = loc[i - 1]
        gap = b[i] - out[-1]
        r = int(round(gap / L)) if L > 0 else 1
        if r >= 2 and abs(gap / r - L) < 0.25 * L:
            out.extend(out[-1] + gap * np.arange(1, r) / r)
            out.append(b[i])
        elif gap < 0.5 * L:
            continue  # spurious extra beat
        else:
            out.append(b[i])
    return np.asarray(out, dtype=float)


def assign_beat_indices(beats: np.ndarray, win: int = 24, tol: float = 0.3,
                        max_skip: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Give every detector beat an integer position on the beat grid.

    The local beat period is the median interval over +-win beats, so a short stretch where the model
    hears double time (or skips beats) cannot change it. Each beat advances the count by
    round(gap / period). Beats that land between grid positions (|r - round(r)| > tol, e.g. half-beat
    extras in a double-time stretch) are dropped instead of being counted - otherwise the bar count
    shifts and the '1' ends up on the wrong beat for the rest of the song.

    Returns (kept_times, integer_indices)."""
    b = np.sort(np.asarray(beats, dtype=float))
    n = len(b)
    if n < 3:
        return b, np.arange(n)
    d = np.diff(b)
    loc = np.empty(n - 1)
    for i in range(n - 1):
        lo, hi = max(0, i - win), min(n - 1, i + win + 1)
        loc[i] = float(np.median(d[lo:hi]))
    keep_t, keep_i = [b[0]], [0]
    skipped = 0
    for i in range(1, n):
        L = loc[min(i - 1, n - 2)]
        r = (b[i] - keep_t[-1]) / L if L > 0 else 1.0
        k = int(round(r))
        if k >= 1 and (abs(r - k) <= tol or skipped >= max_skip):
            keep_t.append(b[i])
            keep_i.append(keep_i[-1] + k)
            skipped = 0
        else:
            skipped += 1  # off-grid extra beat
    return np.asarray(keep_t), np.asarray(keep_i, dtype=int)


def _robust_line(x: np.ndarray, y: np.ndarray, w0: np.ndarray | None = None,
                 floor_s: float = 0.004) -> tuple[float, float, np.ndarray]:
    """y ~ a + c*x by iteratively reweighted least squares (Huber, then Tukey). Returns (a, c, residuals)."""
    w = np.ones(len(x)) if w0 is None else w0.astype(float).copy()

    def wfit(w):
        sw = w.sum()
        if sw <= 0:
            return None
        xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
        var = (w * (x - xm) ** 2).sum()
        c = (w * (x - xm) * (y - ym)).sum() / var if var > 0 else 0.0
        return ym - c * xm, c

    base = w.copy()
    a, c = wfit(base)
    for kind, iters in (("huber", 12), ("tukey", 12)):
        for _ in range(iters):
            res = y - (a + c * x)
            s = max(1.4826 * float(np.median(np.abs(res[base > 0]))), floor_s)
            u = np.abs(res) / s
            if kind == "huber":
                k = 1.345
                rw = np.where(u <= k, 1.0, k / np.maximum(u, 1e-9))
            else:
                k = 4.685
                rw = np.where(u < k, (1 - (u / k) ** 2) ** 2, 0.0)
            w = base * rw
            if (w > 1e-6).sum() < max(3, 0.3 * (base > 0).sum()):
                break
            a, c = wfit(w)
    return a, c, y - (a + c * x)


def _trimmed_rms(res: np.ndarray, keep: float = 0.9) -> float:
    r = np.sort(np.abs(res))
    r = r[: max(3, int(np.ceil(keep * len(r))))]
    return float(np.sqrt(np.mean(r ** 2)))


def _robust_poly_at0(xs: np.ndarray, ys: np.ndarray, w0: np.ndarray, deg: int = 2,
                     floor_s: float = 0.004) -> float:
    """Weighted robust (Tukey) polynomial fit of ys over xs; returns the fitted value at x = 0."""
    A = np.vander(xs, deg + 1, increasing=True)
    w = w0.copy()
    coef = None
    for it in range(8):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], ys * sw, rcond=None)[0]
        res = ys - A @ coef
        s = max(1.4826 * float(np.median(np.abs(res))), floor_s)
        u = np.abs(res) / (4.685 * s)
        w = w0 * np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        if (w > 1e-6).sum() < deg + 3:
            break
    return float(coef[0])


def smooth_variable_grid(b: np.ndarray, idx: np.ndarray | None = None, half_window: int = 16) -> np.ndarray:
    """Moving robust regression (local line + curvature) over +-half_window beats (~8 bars in 4/4):
    follows tempo changes but removes the frame-to-frame jitter of the detector. The curvature term
    keeps the result unbiased during accelerando / ritardando.

    idx: integer grid position of each beat (gaps allowed, see assign_beat_indices). The result has one
    beat for EVERY position from idx[0] to idx[-1], so missing beats are filled in from the local fit."""
    b = np.asarray(b, dtype=float)
    x = np.arange(len(b), dtype=float) if idx is None else np.asarray(idx, dtype=float)
    pos = np.arange(int(x[0]), int(x[-1]) + 1, dtype=float)
    out = np.empty(len(pos))
    full = 2 * half_window + 1
    span = x[-1] - x[0]
    for j, i in enumerate(pos):
        lo, hi = i - half_window, i + half_window
        if span >= full - 1:  # edges: keep the window size by shifting it
            if lo < x[0]:
                lo, hi = x[0], x[0] + full - 1
            elif hi > x[-1]:
                lo, hi = x[-1] - full + 1, x[-1]
        m = (x >= lo) & (x <= hi)
        xs, ys = x[m] - i, b[m]
        if len(xs) < 6:
            k = int(np.argmin(np.abs(x - i)))
            T = float(np.median(np.diff(b) / np.diff(x))) if len(b) > 1 else 0.5
            out[j] = b[k] + (i - x[k]) * T
            continue
        hw = max(half_window, float(np.max(np.abs(xs))))
        tri = np.clip((1 - (np.abs(xs) / (hw + 1)) ** 3) ** 3, 1e-3, None)  # tricube weights
        ref = float(np.interp(i, x, b))
        out[j] = _robust_poly_at0(xs, ys - ref, tri) + ref
    return out


def extend_grid(grid: np.ndarray, duration: float, n_ref: int = 8) -> tuple[np.ndarray, int]:
    """Continue the grid with the local tempo of its first/last beats to cover 0..duration.
    Returns (grid, number of beats added in front)."""
    if len(grid) < 2:
        return grid, 0
    T0 = float(np.median(np.diff(grid[:n_ref + 1])))
    T1 = float(np.median(np.diff(grid[-n_ref - 1:])))
    front = []
    t = grid[0] - T0
    while t >= -1e-6 and T0 > 0.15:
        front.append(t)
        t -= T0
    back = []
    t = grid[-1] + T1
    while t <= duration + 1e-6 and T1 > 0.15:
        back.append(t)
        t += T1
    return np.concatenate([np.asarray(front[::-1]), grid, np.asarray(back)]), len(front)


def combined_onset_env(drum_env: dict[str, np.ndarray]) -> np.ndarray:
    """Kick-weighted onset strength: kick + snare (noise band) + a little hi-hat, each normalised."""
    parts = []
    for name, wgt in (("kick", 1.0), ("snare_noise", 0.7), ("snare_body", 0.3), ("hihat", 0.2)):
        e = drum_env.get(name)
        if e is None or len(e) == 0:
            continue
        ref = float(np.percentile(e, 99)) or 1.0
        parts.append(wgt * np.minimum(e / ref, 2.0))
    if not parts:
        return np.zeros(1)
    n = min(len(p) for p in parts)
    return np.sum([p[:n] for p in parts], axis=0)


def refine_phase(beats: np.ndarray, env: np.ndarray, fs: float, weight_mask: np.ndarray | None = None,
                 max_shift: float = PHASE_MAX_SHIFT_S) -> float:
    """Constant offset (s) that moves the grid onto the onset-strength peaks (+-max_shift)."""
    from scipy.ndimage import gaussian_filter1d
    if len(env) < 10 or float(np.max(env)) <= 0 or len(beats) < 4:
        return 0.0
    e = gaussian_filter1d(env, 1.0)
    times = np.arange(len(e)) * fs
    sel = beats if weight_mask is None else beats[weight_mask]
    sel = sel[(sel > max_shift) & (sel < times[-1] - max_shift)]
    if len(sel) < 4:
        return 0.0
    shifts = np.arange(-round(max_shift * 1000), round(max_shift * 1000) + 1) / 1000.0
    score = np.array([np.interp(sel + s + ENV_LAG_S, times, e).mean() for s in shifts])
    j = int(np.argmax(score))
    if j == 0 or j == len(shifts) - 1:
        return 0.0  # peak at the edge of the search range -> not trustworthy
    if score[j] < 1.1 * float(np.median(score)):
        return 0.0  # no clear peak
    # parabolic interpolation around the maximum
    y0, y1, y2 = score[j - 1], score[j], score[j + 1]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return float((shifts[j] + frac * 0.001))


def regularize_beats(raw: np.ndarray, duration: float, env: np.ndarray | None = None,
                     fs: float = 0.0, log=None) -> dict:
    """Raw detector beats -> clean grid. Returns dict(beats, tempo, mode, rms_ms, offset_ms)."""
    log = log or (lambda m: None)
    b, x = assign_beat_indices(np.asarray(raw, dtype=float))
    if len(b) < 4:
        raise LaufbassError("Es wurden zu wenige Schlaege gefunden, um ein Raster zu bestimmen.")
    dropped = len(raw) - len(b)
    filled = int(x[-1] - x[0] + 1) - len(b)
    if dropped or filled:
        log(f"Roh-Schlaege bereinigt: {dropped} Zwischen-Schlaege verworfen, {filled} fehlende ergaenzt.")
    x = x.astype(float)
    a, T, res = _robust_line(x, b)
    rms = _trimmed_rms(res) * 1000.0
    p95 = float(np.percentile(np.abs(res), 95)) * 1000.0
    first_t, last_t = float(b[0]), float(b[-1])
    # Fixed grid only for really steady songs (click track): a drummer drifting by 40-70 ms would
    # otherwise put the grid half a 16th off in places.
    if rms < BEAT_FIXED_RMS_MS and p95 < BEAT_FIXED_P95_MS:
        mode = "fixed"
        k0 = int(np.ceil(-a / T - 1e-9))              # first grid point at t >= 0
        k1 = int(np.floor((duration - a) / T + 1e-9))  # last grid point inside the song
        grid = a + T * np.arange(k0, k1 + 1)
        tempo = 60.0 / T
        log(f"Festes Tempo erkannt: Raster mit {tempo:.2f} BPM, Abweichung der Roh-Schlaege "
            f"{rms:.1f} ms (Grenze {BEAT_FIXED_RMS_MS:.0f} ms).")
    else:
        mode = "variable"
        grid = smooth_variable_grid(b, x)
        grid, _ = extend_grid(grid, duration)
        tempo = 60.0 / float(np.median(np.diff(grid)))
        log(f"Tempo schwankt (Abweichung vom festen Raster {rms:.1f} ms, 95 % < {p95:.0f} ms): "
            f"Raster wird abschnittsweise (ca. 8 Takte) geglaettet, mittleres Tempo {tempo:.2f} BPM.")
    offset = 0.0
    if env is not None and fs > 0:
        mask = (grid >= first_t - 0.25 * (60.0 / tempo)) & (grid <= last_t + 0.25 * (60.0 / tempo))
        offset = refine_phase(grid, env, fs, mask)
        if abs(offset) > 1e-4:
            grid = grid + offset
            log(f"Phasenkorrektur gegen die Schlagzeug-Einsaetze: Raster um {offset * 1000:+.1f} ms verschoben.")
        else:
            log("Phasenkorrektur: Raster liegt bereits auf den Schlagzeug-Einsaetzen (kein Versatz).")
    grid = grid[(grid >= 0.0) & (grid <= duration)]
    return {"beats": grid, "tempo": float(tempo), "mode": mode, "rms_ms": float(rms),
            "offset_ms": offset * 1000.0}


def analyze_bars(beats: np.ndarray, downbeats: np.ndarray | None) -> tuple[int, int] | None:
    """(first_downbeat_index, beats_per_bar) from the model downbeats, or None if unusable."""
    if downbeats is None or len(downbeats) < 3 or len(beats) < 8:
        return None
    T = float(np.median(np.diff(beats)))
    idx = []
    for d in downbeats:
        j = int(np.argmin(np.abs(beats - d)))
        if abs(beats[j] - d) <= 0.5 * T:
            idx.append(j)
    idx = np.unique(idx)
    if len(idx) < 3:
        return None
    gaps = np.diff(idx)
    c3, c4 = int(np.sum(gaps == 3)), int(np.sum(gaps == 4))
    bpb = 3 if (c3 >= 3 and c3 >= 0.75 * len(gaps)) else 4
    phase = np.bincount(idx % bpb, minlength=bpb)
    return int(np.argmax(phase)), bpb


def find_first_downbeat(beats: np.ndarray, kick_env: np.ndarray, frame_s: float, beats_per_bar: int = 4) -> int:
    """Fallback without model downbeats: index (0..3) whose beats carry the highest average kick strength."""
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
                    first_downbeat: int, stems: list[str], notes: list[Note], hits: list[DrumHit],
                    beats_per_bar: int = 4, beat_mode: str = "fixed") -> dict:
    r3 = lambda x: round(float(x), 3)
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "title": title,
        "source": source,
        "duration": r3(duration),
        "tempo": round(float(tempo), 1),
        "timeSignature": [int(beats_per_bar), 4],
        "beatMode": beat_mode,
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
                 stems_dir: Path | None = None, bass_engine: str = "v2",
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
        elif bass_engine == "basic-pitch":
            notes = transcribe_bass_basic_pitch(stems["bass"])
        else:
            notes = transcribe_bass_v2(stems["bass"])
        say(f"   {len(notes)} Bassnoten gefunden")
        done(t0)

        # 3) drums
        t0 = step(3, TOTAL, "Schlagzeug erkennen ...")
        hits, drum_engine = transcribe_drums(stems["drums"])
        say(f"   Verfahren: {'ADTOF (KI-Modell)' if drum_engine == 'adtof' else 'Frequenzbaender'}")
        counts = {k: sum(1 for h in hits if h.k == k) for k in GM_DRUMS}
        say("   " + ", ".join(f"{v} {k}" for k, v in counts.items() if v) if hits else "   keine Schlaege gefunden")
        done(t0)

        # 4) beats
        t0 = step(4, TOTAL, "Tempo und Taktschlaege bestimmen ...")
        mix = load_audio_ffmpeg(audio, SEP_SR, 1)[0]
        duration = len(mix) / SEP_SR
        # Beat-Modell laeuft auf dem kompletten Original-Mix (nicht auf einem Stem)
        say("   Suche Schlaege im Original-Mix ...")
        raw_beats, raw_downbeats, engine = detect_beats(mix, SEP_SR)
        say(f"   {len(raw_beats)} Roh-Schlaege ({'Beat This!' if engine == 'beat_this' else 'librosa'})")
        import librosa
        d_y, _ = librosa.load(str(stems["drums"]), sr=SEP_SR, mono=True)
        env, fs = drum_band_envelopes(d_y, SEP_SR)
        beat_env = combined_onset_env(env)
        grid = regularize_beats(raw_beats, duration, beat_env, fs, log=lambda m: say("   " + m))
        beats, tempo, beat_mode = grid["beats"], grid["tempo"], grid["mode"]
        bars = analyze_bars(beats, raw_downbeats)
        if bars is None:
            first_db, beats_per_bar = find_first_downbeat(beats, env["kick"], fs), 4
            how = "geraten ueber die Bassdrum"
        else:
            first_db, beats_per_bar = bars
            how = "aus den Modell-Downbeats"
        say(f"   Tempo {tempo:.1f} BPM, {beats_per_bar}/4-Takt, {len(beats)} Schlaege, "
            f"erste '1' = Schlag {first_db + 1} ({how})")
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
        song = build_song_json(title, audio.name, duration, tempo, beats, first_db, list(stems), notes, hits,
                               beats_per_bar, beat_mode)
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
    ap.add_argument("--bass-engine", choices=["v2", "basic-pitch", "pyin"], default="v2", help=argparse.SUPPRESS)
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
