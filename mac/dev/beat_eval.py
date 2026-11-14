#!/usr/bin/env python3
"""Dev: beat grid + downbeat ("1") accuracy on MDB-Drums (beat annotations with bar position).

  python beat_eval.py MDB_DIR

Per song: beat F-measure (+-70 ms), tempo octave (0.5x / 1x / 2x), share of annotated downbeats on which
our grid also has a "1" (the error the user saw: "die 1 war vertauscht").
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import laufbass_prep as lp  # noqa: E402


def fmeasure(det, ref, tol=0.07):
    used = np.zeros(len(det), bool)
    tp = 0
    for t in ref:
        j = np.flatnonzero((np.abs(det - t) <= tol) & ~used)
        if len(j):
            used[j[0]] = True
            tp += 1
    return 2 * tp / max(1, len(det) + len(ref))


def main():
    root = Path(sys.argv[1])
    rows = []
    for ann in sorted((root / "annotations" / "beats").glob("*.beats")):
        name = ann.name.replace("_MIX.beats", "")
        a = np.loadtxt(ann)
        ref_b, ref_pos = a[:, 0], a[:, 1].astype(int)
        ref_down = ref_b[ref_pos == 1]
        wav = root / "audio" / "full_mix" / f"{name}_MIX.wav"
        mix = lp.load_audio_ffmpeg(wav, lp.SEP_SR, 1)[0]
        dur = len(mix) / lp.SEP_SR
        raw, rdown, _ = lp.detect_beats(mix, lp.SEP_SR)
        drums = root / "audio" / "drum_only" / f"{name}_Drum.wav"  # stand-in for the separated drum stem
        import librosa
        y, _ = librosa.load(str(drums), sr=lp.SEP_SR, mono=True)
        env, fs = lp.drum_band_envelopes(y, lp.SEP_SR)
        g = lp.regularize_beats(raw, dur, lp.combined_onset_env(env), fs)
        beats = g["beats"]
        bars = lp.analyze_bars(beats, rdown)
        fd, bpb = bars if bars else (lp.find_first_downbeat(beats, env["kick"], fs), 4)
        our_down = beats[fd::bpb]
        f = fmeasure(beats, ref_b)
        ratio = np.median(np.diff(ref_b)) / np.median(np.diff(beats))
        dhit = np.mean([np.min(np.abs(our_down - t)) <= 0.07 for t in ref_down]) if len(ref_down) else np.nan
        meter = int(ref_pos.max())
        rows.append((name, f, ratio, dhit, meter, bpb))
        print(f"{name:30s} Beat-F {f:.2f}  Tempo x{ratio:.2f}  '1' richtig {100 * dhit:5.1f}%  Takt {meter}/4 erkannt {bpb}/4",
              flush=True)
    F = np.array([r[1] for r in rows]); D = np.array([r[3] for r in rows]); R = np.array([r[2] for r in rows])
    print("-" * 80)
    print(f"Mittel Beat-F {F.mean():.3f} | Tempo richtig (x0.9-1.1) {np.mean((R > 0.9) & (R < 1.1)) * 100:.0f}% | "
          f"'1' richtig im Mittel {np.nanmean(D) * 100:.1f}% | Songs mit '1' >= 90% richtig: {np.sum(D >= 0.9)}/{len(D)}")


if __name__ == "__main__":
    main()
