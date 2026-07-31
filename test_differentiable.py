#!/usr/bin/env python3
"""Gate test: the differentiable forward must match stock Kokoro exactly.

Everything downstream (semantic Jacobians, voice inversion) assumes our
reimplementation is the same function Kokoro actually runs. This asserts that,
and doubles as the regression check against future Kokoro releases.

Note the seeding. Kokoro's vocoder is stochastic — `istftnet.py` draws a random
initial phase (line 150) and injects noise (lines 205, 253) in the harmonic-plus-
noise source module. Two runs with identical inputs differ by ~0.13 peak
amplitude. Seeding immediately before each synthesis pins the noise realization,
which is what makes an exact comparison possible at all.

Usage:
    uv run python test_differentiable.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*", category=UserWarning)

from core import SpeechGenerator
from core.differentiable_kokoro import DifferentiableKokoro

PROJECT_ROOT = Path(__file__).parent
VOICES_DIR = PROJECT_ROOT / "voices"
SEED = 1234

TEXTS = [
    "Hello, my name is Alex.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing one two three.",
]


def load_voice(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def main() -> int:
    print("Initializing Kokoro...")
    gen = SpeechGenerator()
    pipeline = gen.pipeline
    model = pipeline.model
    diff = DifferentiableKokoro(model)

    voice = load_voice(VOICES_DIR / "af_heart.pt")
    failures = 0

    for text in TEXTS:
        ps = diff.phonemize(pipeline, text)
        row = len(ps) - 1
        ref_s = voice[row].to(diff.device)

        # Stock path
        torch.manual_seed(SEED)
        stock = model(ps, ref_s, 1.0, return_output=True)
        stock_audio = stock.audio.detach().cpu().numpy()

        # Ours — same seed pins the vocoder noise to the same realization
        ctx = diff.build_context(ps)
        torch.manual_seed(SEED)
        out = diff.forward(ctx, ref_s, 1.0)
        our_audio = out.audio.detach().cpu().numpy()

        if stock_audio.shape != our_audio.shape:
            print(f"FAIL shape  '{text}': {stock_audio.shape} vs {our_audio.shape}")
            failures += 1
            continue

        max_err = float(np.max(np.abs(stock_audio - our_audio)))
        rms = float(np.sqrt(np.mean(stock_audio**2)))
        # Same seed, same graph — this should be bit-identical, not merely close.
        ok = max_err < 1e-6
        print(
            f"{'PASS' if ok else 'FAIL'}  '{text[:40]}'  "
            f"phonemes={len(ps)} row={row} samples={len(our_audio)} "
            f"max_err={max_err:.3e} (signal rms={rms:.3f})"
        )
        if not ok:
            failures += 1

    # Gradient check — the whole point of this module.
    print("\nGradient flow:")
    ps = diff.phonemize(pipeline, TEXTS[0])
    ctx = diff.build_context(ps)
    ref_s = voice[len(ps) - 1].clone().to(diff.device).requires_grad_(True)
    out = diff.forward(ctx, ref_s, 1.0)

    checks = {
        "audio (timbre+prosody)": out.audio.pow(2).mean(),
        "duration (pacing)": out.duration.sum(),
        "f0_pred (pitch)": out.f0_pred.mean(),
        "n_pred (energy)": out.n_pred.mean(),
    }
    for name, scalar in checks.items():
        if ref_s.grad is not None:
            ref_s.grad = None
        scalar.backward(retain_graph=True)
        g = ref_s.grad
        timbre_g = float(g[:, :128].abs().sum())
        prosody_g = float(g[:, 128:].abs().sum())
        alive = timbre_g + prosody_g > 0
        print(
            f"  {'PASS' if alive else 'FAIL'}  {name:24s} "
            f"|grad| timbre={timbre_g:.4e}  prosody={prosody_g:.4e}"
        )
        if not alive:
            failures += 1

    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
