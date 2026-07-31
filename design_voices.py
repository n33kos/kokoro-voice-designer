#!/usr/bin/env python3
"""Design original voices that fill gaps in Kokoro's built-in set.

No reference audio is involved. Voices are built by steering a base voice along
the named style axes from `build_style_map.py`, so every result is a synthetic
construction described by what was changed — "lower pitch, warmer, slower" — not
a likeness of anyone.

Two approaches were measured before settling on this one:

  - **Blending built-in voices.** Convex blends stay inside the hull of the
    existing library, whose interior is by definition average-sounding. Even
    pushed radially outward until 62% of dimensions clamped, the closest
    built-in was still 0.853 — no more distinct than the most similar pair of
    real built-ins (0.862).
  - **Discovery directions from the catalog.** Worse: 0.91-0.99 similarity even
    at 20x scale. Those directions were selected for audio-feature impact, not
    speaker identity, so they change how a voice sounds without changing who it
    sounds like.

Style-axis steering reaches 0.69-0.71, which sits in the normal range for two
genuinely different voices (built-in pairs: median 0.63, max 0.862).

Candidates are screened on measured acoustics — pitch, high-frequency content
and pitch stability all have to land inside the range the built-in voices span,
so nothing ships that Kokoro renders badly.

Usage:
    uv run python design_voices.py --n-voices 6 --out-dir output/designed
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*RNN module weights are not part of single contiguous chunk.*", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent
VOICES_DIR = PROJECT_ROOT / "voices"
STYLE_MAP = PROJECT_ROOT / "catalog" / "style_map.json"
SEED = 1234
KOKORO_SR = 24000

# The pipeline runs lang_code="a", so restrict to English packs. Voices fitted to
# reference audio are excluded — they are likenesses, not raw material.
ENGLISH_PREFIXES = ("af_", "am_", "bf_", "bm_")
EXCLUDE = {"af_kate_reading", "am_michael_kramer"}

PREVIEW_TEXT = ("The lighthouse keeper watched the storm roll in across the harbor, "
                "and thought about everything he had left behind.")


def load_voice(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def load_library() -> tuple[list[str], torch.Tensor]:
    names, tensors = [], []
    for f in sorted(os.listdir(VOICES_DIR)):
        if not f.endswith(".pt"):
            continue
        stem = f[:-3]
        if stem in EXCLUDE or not stem.startswith(ENGLISH_PREFIXES):
            continue
        names.append(stem)
        tensors.append(load_voice(VOICES_DIR / f).reshape(510, 256).float())
    return names, torch.stack(tensors)


def measure(audio: np.ndarray) -> dict:
    f0 = librosa.yin(audio, fmin=50, fmax=400, sr=KOKORO_SR)
    voiced = f0[(f0 > 55) & (f0 < 350)]
    median = float(np.median(voiced)) if len(voiced) else 0.0
    return {
        "pitch_hz": median,
        # Zero-crossing rate stands in for high-frequency content; too much
        # reads as grainy or sibilant.
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
        # Frequency of large upward pitch jumps. Excess sounds unstable.
        "pitch_excursions": (float(np.mean(voiced > 1.5 * median))
                             if len(voiced) else 1.0),
        "brightness_hz": float(np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=KOKORO_SR))),
    }


def describe(coeffs: np.ndarray, features: list[str], top_n: int = 3) -> str:
    """Plain-language summary of what was steered, for the voice's description."""
    order = np.argsort(-np.abs(coeffs))[:top_n]
    parts = []
    for i in order:
        if abs(coeffs[i]) < 0.3:
            continue
        direction = "more" if coeffs[i] > 0 else "less"
        parts.append(f"{direction} {features[i]}")
    return ", ".join(parts) if parts else "subtle adjustment"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-voices", type=int, default=6)
    ap.add_argument("--n-candidates", type=int, default=48,
                    help="Candidates to synthesize and screen")
    ap.add_argument("--strength", type=float, default=3.0,
                    help="Max magnitude per style axis")
    ap.add_argument("--max-similarity", type=float, default=0.80,
                    help="Reject candidates closer than this to any built-in voice")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "designed"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not STYLE_MAP.exists():
        print(f"No style map at {STYLE_MAP}. Run build_style_map.py first.")
        return 1

    from core import SpeechGenerator
    from core.differentiable_kokoro import DifferentiableKokoro
    from resemblyzer import VoiceEncoder, preprocess_wav

    rng = np.random.default_rng(args.seed)
    names, library = load_library()
    lo, hi = library.min(0).values, library.max(0).values

    smap = json.loads(STYLE_MAP.read_text())
    features = smap["feature_names"]
    directions = torch.tensor(smap["directions"], dtype=torch.float32)

    gen = SpeechGenerator(device="cpu", seed=args.seed)
    diff = DifferentiableKokoro(gen.pipeline.model)
    encoder = VoiceEncoder(device="cpu", verbose=False)
    ps = diff.phonemize(gen.pipeline, PREVIEW_TEXT)
    ctx = diff.build_context(ps)

    def synth(voice_rows: torch.Tensor) -> np.ndarray:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            return diff.forward(ctx, voice_rows[len(ps) - 1].unsqueeze(0)).audio.numpy()

    def embed(audio: np.ndarray) -> np.ndarray:
        return encoder.embed_utterance(preprocess_wav(audio, source_sr=KOKORO_SR))

    print(f"Library: {len(names)} built-in English voices")
    print("Measuring built-in voices for reference ranges...")
    builtin_embeds, builtin_stats = [], []
    for i, name in enumerate(names):
        audio = synth(library[i])
        builtin_embeds.append(embed(audio))
        builtin_stats.append(measure(audio))
    builtin_embeds = np.stack(builtin_embeds)

    # Accept only what the built-in voices themselves demonstrate is renderable.
    zcr_lo, zcr_hi = (min(s["zcr"] for s in builtin_stats),
                      max(s["zcr"] for s in builtin_stats))
    exc_hi = max(s["pitch_excursions"] for s in builtin_stats)
    pitch_lo, pitch_hi = (min(s["pitch_hz"] for s in builtin_stats),
                          max(s["pitch_hz"] for s in builtin_stats))
    print(f"  pitch {pitch_lo:.0f}-{pitch_hi:.0f} Hz, ZCR {zcr_lo:.3f}-{zcr_hi:.3f}, "
          f"pitch excursions up to {exc_hi:.1%}")

    print(f"\nGenerating and screening {args.n_candidates} candidates...")
    kept = []
    for k in range(args.n_candidates):
        base_idx = int(rng.integers(len(names)))
        coeffs = rng.uniform(-args.strength, args.strength, size=len(features))
        delta = (torch.tensor(coeffs, dtype=torch.float32).unsqueeze(0) @ directions).squeeze(0)
        cand = torch.clamp(library[base_idx] + delta.unsqueeze(0), lo, hi)

        audio = synth(cand)
        stats = measure(audio)
        if not (zcr_lo <= stats["zcr"] <= zcr_hi):
            continue
        if stats["pitch_excursions"] > exc_hi:
            continue
        if not (pitch_lo * 0.9 <= stats["pitch_hz"] <= pitch_hi * 1.1):
            continue

        e = embed(audio)
        sims = builtin_embeds @ e
        top = float(sims.max())
        if top > args.max_similarity:
            continue

        kept.append({
            "tensor": cand, "audio": audio, "embed": e, "stats": stats,
            "coeffs": coeffs, "base": names[base_idx],
            "closest_builtin": names[int(sims.argmax())], "closest_similarity": top,
        })

    print(f"  {len(kept)} of {args.n_candidates} passed screening")
    if not kept:
        print("  Try raising --max-similarity or lowering --strength.")
        return 1

    # Greedy max-min so the final set is varied rather than several takes on the
    # same idea.
    print(f"\nSelecting {args.n_voices} mutually distinct voices...")
    chosen = [min(kept, key=lambda c: c["closest_similarity"])]
    remaining = [c for c in kept if c is not chosen[0]]
    while len(chosen) < args.n_voices and remaining:
        best = max(remaining,
                   key=lambda c: -max(float(c["embed"] @ o["embed"]) for o in chosen))
        chosen.append(best)
        # Remove by identity: these dicts hold tensors, so `list.remove` would
        # compare them elementwise and raise.
        remaining = [c for c in remaining if c is not best]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, c in enumerate(chosen, 1):
        stem = f"designed_{i:02d}"
        torch.save(c["tensor"].unsqueeze(1), out_dir / f"{stem}.pt")
        sf.write(str(out_dir / f"{stem}.wav"), c["audio"], KOKORO_SR)
        peer = max((float(c["embed"] @ o["embed"]) for o in chosen if o is not c),
                   default=0.0)
        entry = {
            "name": stem,
            "description": describe(c["coeffs"], features),
            "pitch_hz": round(c["stats"]["pitch_hz"], 1),
            "brightness_hz": round(c["stats"]["brightness_hz"], 1),
            "closest_builtin": c["closest_builtin"],
            "closest_similarity": round(c["closest_similarity"], 3),
            "closest_peer_similarity": round(peer, 3),
        }
        meta.append(entry)
        print(f"  {stem}: {entry['description']}")
        print(f"     pitch {entry['pitch_hz']} Hz, closest built-in "
              f"{entry['closest_builtin']} at {entry['closest_similarity']}")

    (out_dir / "designed.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {len(chosen)} voices to {out_dir}")
    print(f"Highest similarity to any built-in: "
          f"{max(m['closest_similarity'] for m in meta):.3f}")
    print(f"Highest similarity between two of these: "
          f"{max(m['closest_peer_similarity'] for m in meta):.3f}")
    print("For reference, built-in voices sit at 0.63 median / 0.86 max to each other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
