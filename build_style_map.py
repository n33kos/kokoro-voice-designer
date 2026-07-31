#!/usr/bin/env python3
"""Build semantic slider directions in Kokoro's native 256-dim style space.

Replaces `build_semantic_map.py`, which perturbed 1055 PCA/discovery components
in the flattened 130,560-dim tensor, synthesized once per component, and diffed
librosa features — 1055 syntheses producing a noisy [1055, 20] matrix that was
then pseudo-inverted, hopelessly underdetermined.

This works in the space the model actually uses (see plan.md F1-F3):

  - An utterance uses one 256-dim row of the voice pack, indexed by phoneme count.
  - `ref_s[:, :128]` drives the decoder (timbre); `ref_s[:, 128:]` drives the
    prosody predictor (pace, pitch, energy).

So the Jacobian is [n_features, 256], computed by backpropagation rather than
finite differences — exact, and one backward pass per feature instead of one
synthesis per component. Inverting it is well-conditioned.

Two further improvements over the old map:

  - Each axis is masked to the half that architecturally controls it, so a pitch
    slider structurally cannot move timbre.
  - Directions are calibrated: slider = 1.0 changes its feature by a set
    percentage of the base voice's value, rather than an arbitrary scale.

Usage:
    uv run python build_style_map.py
    uv run python build_style_map.py --voice voices/af_bella.pt --target-change 0.3
"""

import argparse
import json
import sys
import time
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
VOICES_DIR = PROJECT_ROOT / "voices"
CATALOG_DIR = PROJECT_ROOT / "catalog"
DEFAULT_OUTPUT = CATALOG_DIR / "style_map.json"
SEED = 1234

# Which half of the style vector each axis is allowed to touch. Masking makes
# disentanglement structural instead of something the inversion has to discover.
PROSODY_HALF = "prosody"
TIMBRE_HALF = "timbre"

FEATURE_HALVES = {
    "pace": PROSODY_HALF,
    "pitch": PROSODY_HALF,
    "pitch variation": PROSODY_HALF,
    "energy": PROSODY_HALF,
    "energy variation": PROSODY_HALF,
    "volume": TIMBRE_HALF,
    "brightness": TIMBRE_HALF,
    "fullness": TIMBRE_HALF,
    "breathiness": TIMBRE_HALF,
    "sibilance": TIMBRE_HALF,
    "warmth": TIMBRE_HALF,
    "dynamics": TIMBRE_HALF,
}

# Averaging the Jacobian over several lengths keeps sliders from being tuned to
# one utterance size.
# Features whose natural measurement runs opposite to the slider's name. `pace`
# is measured as total duration, so a larger value means slower speech; the
# slider is negated so that dragging it up speeds the voice up.
INVERTED_FEATURES = {"pace"}

PROBE_TEXTS = [
    "Hello, my name is Alex.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "I think we should probably talk about this in the morning.",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_voice(path):
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def compute_jacobian(diff, pipeline, voice, device):
    """[n_features, 256] Jacobian plus the base value of each feature.

    Averaged over probe texts. Prosody features come from model internals and
    need no vocoder pass; audio features need a full forward through it.
    """
    rows, base_vals = {}, {}

    for text in PROBE_TEXTS:
        ps = diff.phonemize(pipeline, text)
        ctx = diff.build_context(ps)
        row_idx = min(len(ps) - 1, voice.shape[0] - 1)
        ref_s = voice[row_idx].clone().to(device).requires_grad_(True)

        torch.manual_seed(SEED)
        out = diff.forward(ctx, ref_s)

        feats = dict(prosody_features(out))
        for name, fn in AUDIO_FEATURES.items():
            feats[name] = fn(out.audio.unsqueeze(0)).squeeze()

        for name, value in feats.items():
            if ref_s.grad is not None:
                ref_s.grad = None
            value.backward(retain_graph=True)
            g = ref_s.grad.detach().clone().squeeze(0)
            rows.setdefault(name, []).append(g.cpu().numpy())
            base_vals.setdefault(name, []).append(float(value))

        log(f"  probed '{text[:40]}' ({len(ps)} phonemes)")

    feature_names = sorted(rows.keys())
    J = np.stack([np.mean(rows[n], axis=0) for n in feature_names])  # [M, 256]
    base = np.array([float(np.mean(base_vals[n])) for n in feature_names])
    return feature_names, J, base


def build(args) -> int:
    log("Initializing Kokoro...")
    gen = SpeechGenerator(device=args.device)
    diff = DifferentiableKokoro(gen.pipeline.model)

    voice = load_voice(args.voice)
    log(f"Base voice: {Path(args.voice).name}  shape={tuple(voice.shape)}")

    log(f"Computing Jacobian over {len(PROBE_TEXTS)} probe texts "
        f"({len(AUDIO_FEATURES) + 5} features, one backward pass each)...")
    names, J, base = compute_jacobian(diff, gen.pipeline, voice, args.device)
    log(f"Jacobian: {J.shape[0]} features x {J.shape[1]} style dims")

    # --- Mask each feature to the half that controls it ---
    mask = np.zeros_like(J)
    for i, name in enumerate(names):
        half = FEATURE_HALVES.get(name)
        if half == PROSODY_HALF:
            mask[i, 128:] = 1.0
        elif half == TIMBRE_HALF:
            mask[i, :128] = 1.0
        else:
            mask[i, :] = 1.0
    J_masked = J * mask

    # Report how much signal masking discards — if a "timbre" feature is mostly
    # driven by the prosody half, the assignment in FEATURE_HALVES is wrong.
    log("Feature sensitivity by half (|grad| share):")
    for i, name in enumerate(names):
        t = float(np.abs(J[i, :128]).sum())
        p = float(np.abs(J[i, 128:]).sum())
        total = t + p + 1e-12
        log(f"  {name:18s} timbre={100*t/total:5.1f}%  prosody={100*p/total:5.1f}%  "
            f"-> masked to {FEATURE_HALVES.get(name, 'both')}")

    # --- Right pseudo-inverse with ridge ---
    # Work in *relative* change units first. Raw gradients differ by orders of
    # magnitude between features (pace is in frames, pitch in Hz, sibilance is a
    # ratio), which makes the Gram matrix badly scaled and lets a single ridge
    # term swamp the small eigenvalues. Dividing each row by its base value puts
    # every feature on a "fraction of itself" footing.
    M = J_masked.shape[0]
    J_rel = J_masked / np.maximum(np.abs(base)[:, None], 1e-12)

    # Solve J_rel @ d_i = e_i. The Gram matrix is only [M, M] (~12x12), so this
    # is well-conditioned — the opposite of inverting [1055, 20].
    G = J_rel @ J_rel.T
    scale = float(np.trace(G)) / M
    directions = (J_rel.T @ np.linalg.inv(G + args.ridge * scale * np.eye(M))).T  # [M, 256]

    # Slider at 1.0 now means "change this feature by target_change of its base".
    directions *= args.target_change

    for i, name in enumerate(names):
        if name in INVERTED_FEATURES:
            directions[i] *= -1.0

    # --- Verify: does moving one slider actually move only its feature? ---
    # achieved[i, j] is the absolute change in feature i when slider j is at 1.0.
    # Features are in wildly different units (pace in frames, pitch in Hz), so
    # normalize each row by its own base value to get a comparable fraction.
    achieved = J_masked @ directions.T  # [M, M]
    relative = achieved / np.maximum(np.abs(base)[:, None], 1e-12)
    diag = np.abs(np.diag(relative))
    leak = np.abs(relative - np.diag(np.diag(relative)))
    log(f"Disentanglement (fraction of each feature's base value):")
    log(f"  intended change on-diagonal: mean={diag.mean():.3f} "
        f"(target {args.target_change})")
    log(f"  worst cross-talk: {leak.max():.3f} "
        f"({names[int(np.argmax(leak) // M)]} <- {names[int(np.argmax(leak) % M)]})")
    log(f"  mean cross-talk: {leak.mean():.4f}  "
        f"signal/leak ratio: {diag.mean()/max(leak.mean(), 1e-12):.1f}x")

    result = {
        "version": 2,
        "space": "style_256",
        "feature_names": names,
        "feature_halves": [FEATURE_HALVES.get(n, "both") for n in names],
        "directions": directions.tolist(),
        "base_values": base.tolist(),
        "target_change": args.target_change,
        "base_voice": Path(args.voice).name,
        "probe_texts": PROBE_TEXTS,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Saved {out}  ({M} features x 256 dims)")
    log(f"Features: {', '.join(names)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default=str(VOICES_DIR / "af_heart.pt"))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="Ridge factor, relative to the Jacobian's own scale")
    ap.add_argument("--target-change", type=float, default=0.3,
                    help="Fractional change in a feature at slider = 1.0")
    return build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
