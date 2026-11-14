# -*- coding: utf-8 -*-
"""Stem separation with BS-RoFormer SW (6 stems: bass, drums, other, vocals, guitar, piano).

Model by jarredou, trained with ZFTurbo's Music-Source-Separation-Training (architecture vendored in
roformer/, MIT). Measured against user tabs it gives clearly cleaner bass stems than Demucs
(Just Like Heaven bass F1: htdemucs 0.57, htdemucs_ft 0.66, BS-RoFormer SW 0.76).

Weights (~700 MB) are downloaded once to ~/.laufbass/models and checked by SHA-256.
Inference follows MSST's demix(): overlapping chunks (50 %), linear fades, reflect padding.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

MODEL_FILE = "BS-Roformer-SW.ckpt"
MODEL_URL = ("https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/"
             + MODEL_FILE)
MODEL_SHA256 = "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
MODEL_DIR = Path(os.environ.get("LAUFBASS_MODELS", Path.home() / ".laufbass" / "models"))
INSTRUMENTS = ["bass", "drums", "other", "vocals", "guitar", "piano"]
SR = 44100
CHUNK = 588800
NUM_OVERLAP = 2
CONFIG = {
    "dim": 256, "depth": 12, "stereo": True, "num_stems": 6, "time_transformer_depth": 1,
    "freq_transformer_depth": 1, "linear_transformer_depth": 0,
    "freqs_per_bands": (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129),
    "dim_head": 64, "heads": 8, "attn_dropout": 0.1, "ff_dropout": 0.1, "flash_attn": True,
    "dim_freqs_in": 1025, "stft_n_fft": 2048, "stft_hop_length": 512, "stft_win_length": 2048,
    "stft_normalized": False, "mask_estimator_depth": 2, "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256), "multi_stft_hop_size": 147,
    "multi_stft_normalized": False, "mlp_expansion_factor": 4, "use_torch_checkpoint": False,
    "skip_connection": False,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_weights(log=print) -> Path:
    path = MODEL_DIR / MODEL_FILE
    if path.exists() and path.stat().st_size > 100_000_000:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log("   Lade Trennmodell BS-RoFormer (einmalig ca. 700 MB) ...")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if _sha256(tmp) != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Download des Trennmodells beschaedigt (Pruefsumme stimmt nicht)")
    tmp.rename(path)
    return path


def load_model(device: str = "cpu"):
    import torch
    from roformer.bs_roformer import BSRoformer
    model = BSRoformer(**CONFIG)
    state = torch.load(ensure_weights(), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Modell passt nicht zu den Gewichten (fehlend {len(missing)}, unerwartet {len(unexpected)})")
    return model.eval().to(device)


def separate(wav: np.ndarray, device: str = "cpu", progress=None, model=None) -> dict[str, np.ndarray]:
    """wav: float32 (2, samples) at 44.1 kHz. Returns {instrument: (2, samples)}."""
    import torch
    import torch.nn.functional as F
    if model is None:
        model = load_model(device)
    mix = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    step = CHUNK // NUM_OVERLAP
    border = CHUNK - step
    fade = CHUNK // 10
    length0 = mix.shape[-1]
    if length0 > 2 * border:
        mix = F.pad(mix[None], (border, border), mode="reflect")[0]
    win = torch.ones(CHUNK)
    win[:fade] = torch.linspace(0, 1, fade)
    win[-fade:] = torch.linspace(1, 0, fade)
    n = mix.shape[-1]
    result = torch.zeros((len(INSTRUMENTS),) + tuple(mix.shape))
    counter = torch.zeros(n)
    starts = list(range(0, n, step))
    with torch.no_grad():
        for k, i in enumerate(starts):
            part = mix[:, i:i + CHUNK]
            L = part.shape[-1]
            part = F.pad(part[None], (0, CHUNK - L), mode="reflect" if L > CHUNK // 2 else "constant")[0]
            x = model(part[None].to(device)).float().cpu()[0]  # (stems, 2, CHUNK)
            w = win.clone()
            if i == 0:
                w[:fade] = 1
            if i + step >= n:
                w[-fade:] = 1
            result[..., i:i + L] += x[..., :L] * w[:L]
            counter[i:i + L] += w[:L]
            if progress:
                progress(k + 1, len(starts))
    est = (result / counter.clamp_min(1e-8)).numpy()
    if length0 > 2 * border:
        est = est[..., border:-border]
    return {name: np.nan_to_num(est[j]) for j, name in enumerate(INSTRUMENTS)}
