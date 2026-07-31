#!/usr/bin/env python3
"""Compare voice-match candidates against real reference audio.

Scores each candidate voice by Resemblyzer similarity to the reference
recording, on text the optimizer never saw. Resemblyzer is deliberately *not*
what `invert_voice.py` optimizes (that's the WavLM pooled-statistics loss), so
it stays an independent check rather than a number we tuned against.

Two figures to keep in mind when reading the output:

  - Kokoro's vocoder is stochastic, so two syntheses of the same voice score
    ~0.9985 rather than 1.0. That's the effective ceiling.
  - Comparing synthetic speech to a real human recording is harder than
    comparing two synthetic clips; expect lower absolute numbers than the
    built-in-voice sanity check produces.

Usage:
    uv run python evaluate_match.py --reference input/reference.wav \
        --candidates voices/af_heart.pt output/my_voice.pt
"""

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*RNN module weights are not part of single contiguous chunk.*", category=UserWarning)

from core import SpeechGenerator, FitnessScorer
from core.differentiable_kokoro import DifferentiableKokoro

SEED = 1234
KOKORO_SR = 24000

# Held out from every optimization run in this project.
EVAL_TEXTS = [
    "The lighthouse keeper watched the storm roll in across the harbor.",
    "I wasn't expecting anyone to call this late in the evening.",
    "Numbers like seventeen and forty-three are surprisingly hard to say clearly.",
]


def load_voice(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="Real reference recording (.wav)")
    ap.add_argument("--candidates", nargs="+", required=True, help="Voice .pt files to score")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-audio", action="store_true",
                    help="Write one sample per candidate for listening")
    args = ap.parse_args()

    gen = SpeechGenerator(device=args.device, seed=SEED)
    diff = DifferentiableKokoro(gen.pipeline.model)

    ref_audio, _ = librosa.load(args.reference, sr=KOKORO_SR, mono=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, ref_audio, KOKORO_SR)
    fitness = FitnessScorer(tmp.name, device=args.device)

    print(f"Reference: {Path(args.reference).name} ({len(ref_audio)/KOKORO_SR:.1f}s)")
    print(f"Scoring on {len(EVAL_TEXTS)} held-out texts\n")
    print(f"{'candidate':<34s} {'mean':>8s} {'min':>8s} {'max':>8s}")
    print("-" * 62)

    results = []
    for path in args.candidates:
        voice = load_voice(path)
        sims = []
        for text in EVAL_TEXTS:
            ps = diff.phonemize(gen.pipeline, text)
            ctx = diff.build_context(ps)
            torch.manual_seed(SEED)
            with torch.no_grad():
                audio = diff.forward(ctx, voice[len(ps) - 1].to(args.device)).audio.cpu().numpy()
            sims.append(float(fitness.target_similarity(audio)))
            if args.save_audio and text == EVAL_TEXTS[0]:
                out = Path("output") / f"eval_{Path(path).stem}.wav"
                sf.write(str(out), audio, KOKORO_SR)
        results.append((path, float(np.mean(sims)), min(sims), max(sims)))
        print(f"{Path(path).name:<34s} {np.mean(sims):>8.4f} {min(sims):>8.4f} {max(sims):>8.4f}")

    print("-" * 62)
    best = max(results, key=lambda r: r[1])
    print(f"Best: {Path(best[0]).name} ({best[1]:.4f})")
    if args.save_audio:
        print("Wrote output/eval_*.wav for listening.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
