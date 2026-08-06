#!/usr/bin/env python3
"""Measure every built-in voice so a run can pick the right starting point.

Voice matching moves a base voice toward a target, and the manifold constraint
limits how far it may travel. Starting from a voice whose register is far from
the target spends that budget on pitch instead of identity — a match started
20% below the speaker's register came out 27% high with its pitch asymmetry
inverted, while runs started near their target landed within a few percent.

Writes catalog/voice_registry.json. Rebuild after adding voices.

Usage:
    uv run python build_voice_registry.py
"""
import json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import librosa, numpy as np, torch
from scipy.stats import gaussian_kde

from core import SpeechGenerator
from core.differentiable_kokoro import DifferentiableKokoro
from resemblyzer import VoiceEncoder, preprocess_wav

ROOT = Path(__file__).parent
VOICES = ROOT / "voices"
OUT = ROOT / "catalog" / "voice_registry.json"
TEXT = ("The morning after the storm, the whole village came down to the water "
        "to see what had washed in, and nobody wanted to be first to speak.")


def profile(audio):
    # Same tracker as every other pitch measurement in the project. yin with a
    # fixed band reports 8-13% octave jumps on real speech; see core.pitch.
    from core import pitch

    track = pitch.track(audio, 24000)
    v = track.values
    if len(v) < 20:
        return None
    lg = np.log(v)
    kde = gaussian_kde(lg); grid = np.linspace(lg.min(), lg.max(), 400)
    return {
        "register_hz": round(float(np.exp(grid[np.argmax(kde(grid))])), 1),
        "median_hz": round(float(np.median(v)), 1),
        "up_ratio": round(float(np.percentile(v, 90)) / float(np.median(v)), 2),
        "down_ratio": round(float(np.median(v)) / float(np.percentile(v, 10)), 2),
    }


def main():
    gen = SpeechGenerator(device="cpu", seed=1234)
    diff = DifferentiableKokoro(gen.pipeline.model)
    encoder = VoiceEncoder(device="cpu", verbose=False)
    ps = diff.phonemize(gen.pipeline, TEXT)
    ctx = diff.build_context(ps)

    entries = {}
    names = [f[:-3] for f in sorted(os.listdir(VOICES)) if f.endswith(".pt")]
    for i, name in enumerate(names, 1):
        v = torch.load(VOICES / f"{name}.pt", weights_only=True).reshape(510, 256).float()
        torch.manual_seed(1234)
        with torch.no_grad():
            audio = diff.forward(ctx, v[len(ps) - 1].unsqueeze(0)).audio.numpy()
        p = profile(audio)
        if p:
            # Cache the speaker embedding too. Register alone is a poor way to
            # choose a starting voice — matching only median pitch picked a male
            # voice for a female speaker and a non-English voice for an English
            # one. Embedding similarity captures who a voice sounds like.
            p["embedding"] = [round(float(x), 5) for x in
                              encoder.embed_utterance(preprocess_wav(audio, source_sr=24000))]
            entries[name] = p
        print(f"  {i}/{len(names)} {name}: {p['register_hz'] if p else 'n/a'} Hz", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(entries, indent=2, sort_keys=True))
    print(f"\nWrote {len(entries)} voices to {OUT}")
    reg = sorted((e["register_hz"], n) for n, e in entries.items())
    print(f"register range: {reg[0][0]} Hz ({reg[0][1]}) to {reg[-1][0]} Hz ({reg[-1][1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
