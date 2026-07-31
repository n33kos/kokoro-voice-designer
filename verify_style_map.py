#!/usr/bin/env python3
"""Empirically verify the style map by actually synthesizing.

`build_style_map.py` reports a *linear* prediction of what each slider does.
The Jacobian is a local linearization of a nonlinear model, so that prediction
is only trustworthy if it survives real synthesis. This applies each slider at
a given magnitude, synthesizes, measures the features, and compares what
actually happened to what was promised.

Usage:
    uv run python verify_style_map.py
    uv run python verify_style_map.py --magnitude 1.0 --text "Some other text."
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*RNN module weights are not part of single contiguous chunk.*", category=UserWarning)

from core import SpeechGenerator
from core.differentiable_kokoro import DifferentiableKokoro
from core.spectral_features import AUDIO_FEATURES, prosody_features

PROJECT_ROOT = Path(__file__).parent
SEED = 1234


def measure(diff, ctx, ref_s):
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = diff.forward(ctx, ref_s)
        feats = {k: float(v) for k, v in prosody_features(out).items()}
        for name, fn in AUDIO_FEATURES.items():
            feats[name] = float(fn(out.audio.unsqueeze(0)).squeeze())
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(PROJECT_ROOT / "catalog" / "style_map.json"))
    ap.add_argument("--voice", default=str(PROJECT_ROOT / "voices" / "af_heart.pt"))
    ap.add_argument("--text", default="She sells seashells by the seashore.",
                    help="Held-out by default — not one of the probe texts")
    ap.add_argument("--magnitude", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.map) as f:
        smap = json.load(f)
    names = smap["feature_names"]
    directions = np.array(smap["directions"])
    target = smap["target_change"] * args.magnitude

    gen = SpeechGenerator(device=args.device)
    diff = DifferentiableKokoro(gen.pipeline.model)
    voice = torch.load(args.voice, weights_only=True)

    ps = diff.phonemize(gen.pipeline, args.text)
    ctx = diff.build_context(ps)
    row = min(len(ps) - 1, voice.shape[0] - 1)
    base_ref = voice[row].to(args.device)

    base = measure(diff, ctx, base_ref)
    print(f"Text: '{args.text}' ({len(ps)} phonemes, row {row})")
    print(f"Slider magnitude {args.magnitude:+.2f} -> predicted {target:+.1%} change\n")
    print(f"{'slider':18s} {'actual own':>11s} {'predicted':>10s} "
          f"{'worst leak':>11s}  {'leaked into':<18s}")
    print("-" * 76)

    own_errors, leaks = [], []
    for i, name in enumerate(names):
        delta = torch.tensor(directions[i] * args.magnitude, dtype=torch.float32)
        ref_s = (base_ref + delta.unsqueeze(0).to(args.device))
        got = measure(diff, ctx, ref_s)

        rel = {k: (got[k] - base[k]) / abs(base[k]) if abs(base[k]) > 1e-12 else 0.0
               for k in base}
        own = rel[name]
        others = {k: v for k, v in rel.items() if k != name}
        worst_k = max(others, key=lambda k: abs(others[k]))
        own_errors.append(abs(own - target))
        leaks.append(abs(others[worst_k]))
        print(f"{name:18s} {own:>+10.1%} {target:>+10.1%} "
              f"{others[worst_k]:>+10.1%}  {worst_k:<18s}")

    print("-" * 76)
    print(f"mean |own - predicted| = {np.mean(own_errors):.1%}   "
          f"mean worst-leak = {np.mean(leaks):.1%}")
    print("\nA slider is behaving if its own change is near the prediction and "
          "its worst leak is well below it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
