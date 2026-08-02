#!/usr/bin/env python3
"""Design original voices that fill gaps in Kokoro's built-in set.

No reference audio is involved. Voices are built by steering a base voice along
the named style axes from `build_style_map.py`, so every result is a synthetic
construction described by what was changed — "lower pitch, warmer, slower" — not
a likeness of anyone.

Three approaches were measured:

  - **Blending built-in voices.** Convex blends stay inside the hull of the
    library, whose interior is average-sounding by construction. Sounded clean
    but reached only 0.85-0.90 similarity to the nearest built-in — no more
    distinct than the most similar pair of real built-ins (0.862).
  - **Discovery directions from the catalog.** Worse: 0.91-0.99 even at 20x
    scale. Those directions were selected for audio-feature impact, not speaker
    identity, so they change how a voice sounds without changing who it sounds
    like.
  - **Style-axis steering.** Reaches 0.69-0.78, genuinely distinct — but pushed
    naively it produces grainy, strange-sounding voices.

The trap in the third approach is worth stating, because it is easy to repeat.
Constraining each dimension to the range the built-in voices span is far weaker
than staying near the voices themselves. A point can be inside every
dimension's range and still be somewhere no real voice is — the way a person
can be within human height range and within human weight range at seven feet
and ninety pounds. Naive style pushes landed 2.8-3.8 from their nearest
built-in when real voices sit 1.5-2.5 apart, and they sounded like it.

So the gate here is **distance to the nearest built-in voice**, capped at what
the library itself demonstrates is normal. Each candidate is then pushed as far
along its style direction as that cap allows, which maximizes distinctness
without leaving the region Kokoro renders well. Acoustic screens on pitch,
high-frequency content and pitch stability catch anything that slips through.

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

# Per-axis limits on how far a candidate may be steered, derived from listening
# to a generated set. These axes change perceived audio *quality* much faster
# than they change identity, so a large push buys little distinctness and costs
# a lot of naturalness:
#
#   sibilance up      -> high-frequency ringing
#   warmth up         -> muffled, or a band-limited "phone call" sound
#   pace up           -> rushed delivery
#
# The remaining axes (pitch, pitch variation, volume, fullness, energy,
# dynamics) moved identity without audible artifacts and are left unconstrained.
AXIS_LIMITS = {
    "sibilance": (-0.8, 0.15),
    "warmth": (-0.6, 0.15),
    "brightness": (-0.8, 0.8),
    # Upward breathiness appeared in both voices rejected on listening, so it is
    # capped tighter than downward.
    "breathiness": (-0.8, 0.25),
    "pace": (-1.2, 0.15),
}

# Warmth and sibilance are complementary band-energy ratios, so pushing both the
# same way asks for more low *and* more high energy at once. The optimizer
# satisfies that with odd spectral shaping that sounds simultaneously muffled
# and ringing.
EXCLUSIVE_PAIRS = [("warmth", "brightness"), ("warmth", "sibilance")]

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
    ap.add_argument("--manifold-headroom", type=float, default=0.95,
                    help="Cap on distance to the nearest built-in, as a fraction of "
                         "the largest gap the library itself shows. Above ~1.0 the "
                         "result is further out than any real voice and sounds it")
    ap.add_argument("--max-similarity", type=float, default=0.82,
                    help="Reject candidates closer than this to any built-in voice")
    ap.add_argument("--keep", action="append", default=[],
                    help="Existing .pt voices to keep. New voices are selected to be "
                         "distinct from these as well as from the built-ins.")
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

    # How far apart real voices actually sit. This is the gate: a candidate is
    # allowed to be as isolated as the most isolated built-in, and no more.
    lib_mean = library.mean(1)
    D = torch.cdist(lib_mean, lib_mean)
    D.fill_diagonal_(float("inf"))
    nn_lib = D.min(1).values
    max_gap = float(nn_lib.max()) * args.manifold_headroom
    print(f"  built-in voices sit {float(nn_lib.min()):.2f}-{float(nn_lib.max()):.2f} "
          f"apart; capping candidates at {max_gap:.2f}")

    def nn_distance(cand: torch.Tensor) -> float:
        return float(torch.cdist(cand.mean(0, keepdim=True), lib_mean).min())

    print(f"\nGenerating and screening {args.n_candidates} candidates...")
    kept = []
    for k in range(args.n_candidates):
        base_idx = int(rng.integers(len(names)))
        base = library[base_idx]
        raw = rng.normal(size=len(features))
        # Don't push two opposed spectral axes the same way; drop the weaker one.
        for a, b in EXCLUSIVE_PAIRS:
            if a in features and b in features:
                ia, ib = features.index(a), features.index(b)
                if raw[ia] * raw[ib] > 0:
                    if abs(raw[ia]) < abs(raw[ib]):
                        raw[ia] = 0.0
                    else:
                        raw[ib] = 0.0
        if np.linalg.norm(raw) < 1e-9:
            continue
        raw /= np.linalg.norm(raw)
        delta = (torch.tensor(raw, dtype=torch.float32).unsqueeze(0) @ directions).squeeze(0)

        # Push as far along this direction as the manifold cap allows. Distance
        # grows monotonically with scale, so bisection finds the limit exactly,
        # and it costs nothing — no synthesis involved.
        scale_lo, scale_hi = 0.0, 64.0
        if nn_distance(torch.clamp(base + delta.unsqueeze(0) * scale_hi, lo, hi)) <= max_gap:
            scale = scale_hi
        else:
            for _ in range(24):
                mid = (scale_lo + scale_hi) / 2
                if nn_distance(torch.clamp(base + delta.unsqueeze(0) * mid, lo, hi)) <= max_gap:
                    scale_lo = mid
                else:
                    scale_hi = mid
            scale = scale_lo

        # Also respect the per-axis quality limits: the usable scale is whatever
        # the tightest constrained axis allows.
        for axis, (amin, amax) in AXIS_LIMITS.items():
            if axis not in features:
                continue
            v = float(raw[features.index(axis)])
            if v > 1e-9:
                scale = min(scale, amax / v)
            elif v < -1e-9:
                scale = min(scale, amin / v)
        if scale <= 1e-6:
            continue

        coeffs = raw * scale
        cand = torch.clamp(base + delta.unsqueeze(0) * scale, lo, hi)

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
            "nn_distance": nn_distance(cand),
            "closest_builtin": names[int(sims.argmax())], "closest_similarity": top,
        })

    print(f"  {len(kept)} of {args.n_candidates} passed screening")
    if not kept:
        print("  Try raising --max-similarity or lowering --strength.")
        return 1

    # Greedy max-min so the final set is varied rather than several takes on the
    # same idea.
    # Voices being kept from a previous run count as already chosen, so the new
    # ones are picked to differ from them too.
    prior = []
    for path in args.keep:
        v = load_voice(path).reshape(510, 256).float()
        prior.append({"embed": embed(synth(v)), "name": Path(path).stem})
    if prior:
        print(f"\nKeeping {len(prior)} existing voice(s); "
              f"selecting {args.n_voices} more distinct from them")
        kept = [c for c in kept
                if max(float(c["embed"] @ o["embed"]) for o in prior) <= args.max_similarity]
        print(f"  {len(kept)} candidates remain after excluding near-duplicates")
        if not kept:
            print("  none left; try more --n-candidates or a higher --max-similarity")
            return 1

    print(f"\nSelecting {args.n_voices} mutually distinct voices...")
    pool = prior + kept
    chosen = [min(kept, key=lambda c: max(
        [c["closest_similarity"]] + [float(c["embed"] @ o["embed"]) for o in prior]))]
    remaining = [c for c in kept if c is not chosen[0]]
    chosen_all = prior + chosen
    while len(chosen) < args.n_voices and remaining:
        best = max(remaining,
                   key=lambda c: -max(float(c["embed"] @ o["embed"]) for o in chosen_all))
        chosen.append(best)
        chosen_all.append(best)
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
            "manifold_distance": round(c["nn_distance"], 2),
            "closest_peer_similarity": round(peer, 3),
        }
        meta.append(entry)
        print(f"  {stem}: {entry['description']}")
        print(f"     pitch {entry['pitch_hz']} Hz, closest built-in "
              f"{entry['closest_builtin']} at {entry['closest_similarity']}, "
              f"manifold distance {entry['manifold_distance']}")

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
