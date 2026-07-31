#!/usr/bin/env python3
"""Build a semantic map from the discovery catalog.

For each of the N components in the catalog, perturbs the voice and measures
the delta across ALL audio features. Builds an N x M sensitivity matrix,
computes its pseudo-inverse to get M semantic directions (one per audio
feature), and saves the result to catalog/semantic_map.json.

Each semantic direction is a set of component weights that maximally changes
one audio feature while minimally affecting others.

Usage:
    uv run python build_semantic_map.py
    uv run python build_semantic_map.py --voice voices/af_heart.pt
    uv run python build_semantic_map.py --text "Hello, how are you today?"
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Suppress noisy warnings from Kokoro/PyTorch internals
warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=".*RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message=".*is deprecated in favor of*", category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*dropout option adds dropout after all but last recurrent layer*",
    category=UserWarning,
)

from core import SpeechGenerator, VoiceAnalyzer, FitnessScorer

PROJECT_ROOT = Path(__file__).parent
VOICES_DIR = PROJECT_ROOT / "voices"
CATALOG_DIR = PROJECT_ROOT / "catalog"
DEFAULT_CATALOG = CATALOG_DIR / "discovery_catalog.pt"
DEFAULT_OUTPUT = CATALOG_DIR / "semantic_map.json"

FEATURE_LABELS = {
    "pitch_mean": "pitch",
    "pitch_std": "pitch variation",
    "spectral_centroid_mean": "brightness",
    "spectral_bandwidth_mean": "fullness",
    "spectral_rolloff_mean": "crispness",
    "spectral_contrast_mean": "clarity",
    "spectral_flatness_mean": "breathiness",
    "rms_energy": "volume",
    "energy_mean": "energy",
    "energy_std": "energy variation",
    "mfcc1_mean": "timbre",
    "mfcc2_mean": "nasality",
    "mfcc3_mean": "resonance",
    "mfcc4_mean": "texture",
    "chroma_mean": "harmonics",
    "tonnetz_mean": "tonality",
    "tempo": "pace",
    "audio_std": "dynamics",
    "harmonic_ratio": "harmonic richness",
    "zero_crossing_rate": "sibilance",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_voice(path: str | Path) -> torch.Tensor:
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def build_semantic_map(
    catalog_path: Path,
    voice_path: Path,
    voice_folder: str,
    text: str,
    output_path: Path,
) -> None:
    # --- Load catalog ---
    if not catalog_path.exists():
        print(f"Error: catalog not found at {catalog_path}")
        print("Run build_catalog.py first.")
        sys.exit(1)

    log("Loading catalog...")
    catalog = torch.load(catalog_path, weights_only=False)

    n_pca = catalog.get("pca_n_components", 0)
    pca_comps = catalog["pca_components"][:n_pca]
    pca_sv = catalog["pca_singular_values"][:n_pca]
    pca_mean = catalog["pca_mean"]

    disc_comps = catalog.get("discovered_components")
    if disc_comps is not None and disc_comps.shape[0] > 0:
        avg_sv = float(pca_sv.mean())
        disc_sv = torch.full((disc_comps.shape[0],), avg_sv)
        all_components = torch.cat([pca_comps, disc_comps], dim=0)
        all_sv = torch.cat([pca_sv, disc_sv], dim=0)
    else:
        all_components = pca_comps
        all_sv = pca_sv

    n_components = all_components.shape[0]
    log(f"Catalog: {n_pca} PCA + {n_components - n_pca} discoveries = {n_components} total")

    # --- Load voice library for component ranges ---
    pt_files = sorted(f for f in os.listdir(voice_folder) if f.endswith(".pt"))
    voice_tensors = [load_voice(os.path.join(voice_folder, f)) for f in pt_files]
    n_voices = len(voice_tensors)

    flat = torch.stack(voice_tensors).reshape(n_voices, -1).float()
    centered = flat - pca_mean
    projections = centered @ all_components.T
    component_ranges = projections.max(dim=0).values - projections.min(dim=0).values

    # Fix discovery ranges (same as server.py)
    if disc_comps is not None and disc_comps.shape[0] > 0:
        avg_pca_range = float(component_ranges[:n_pca].mean())
        component_ranges[n_pca:] = avg_pca_range

    # --- Initialize Kokoro ---
    log("Initializing Kokoro pipeline...")
    speech_gen = SpeechGenerator()

    # --- Load base voice ---
    base_voice = load_voice(voice_path)
    voice_shape = base_voice.shape
    base_flat = base_voice.reshape(-1).float()

    base_min = float(base_flat.min())
    base_max = float(base_flat.max())
    clamp_lo = base_min - abs(base_min) * 2
    clamp_hi = base_max + abs(base_max) * 2

    # --- Generate base audio and features ---
    log(f"Generating base audio with voice: {voice_path.name}")
    base_audio = speech_gen.generate_audio(text, base_voice)
    base_features = FitnessScorer.extract_features(base_audio)

    # Only use features that are in FEATURE_LABELS
    feature_keys = [k for k in base_features.keys() if k in FEATURE_LABELS]
    feature_names = [FEATURE_LABELS[k] for k in feature_keys]
    n_features = len(feature_keys)
    log(f"Tracking {n_features} audio features")

    # --- Build sensitivity matrix ---
    log(f"Building sensitivity matrix ({n_components} components x {n_features} features)...")
    log(f"This requires {n_components} Kokoro syntheses.")

    S = np.zeros((n_components, n_features))
    n_skipped = 0

    for i in range(n_components):
        component = all_components[i]
        scale = float(all_sv[i]) * 0.1
        perturbed_flat = base_flat + component * scale
        perturbed_flat = torch.nan_to_num(perturbed_flat, nan=0.0, posinf=clamp_hi, neginf=clamp_lo)
        perturbed_flat = perturbed_flat.clamp(clamp_lo, clamp_hi)
        perturbed_voice = perturbed_flat.reshape(voice_shape)

        try:
            audio = speech_gen.generate_audio(text, perturbed_voice)
            features = FitnessScorer.extract_features(audio)
            for j, key in enumerate(feature_keys):
                bv = base_features[key]
                nv = features[key]
                S[i, j] = (nv - bv) / abs(bv) if abs(bv) > 1e-8 else (nv - bv)
        except Exception as e:
            log(f"  Skipping component {i} (error: {str(e)[:80]})")
            n_skipped += 1

        if (i + 1) % 8 == 0 or i == n_components - 1:
            log(f"  Analyzed {i + 1}/{n_components} ({n_skipped} skipped)")

    # --- Scale by component ranges ---
    # The sensitivity was measured at scale = sv * 0.1, but sliders use
    # component_ranges. Scale each row so S reflects actual slider impact.
    for i in range(n_components):
        analysis_scale = float(all_sv[i]) * 0.1
        slider_scale = float(component_ranges[i])
        if analysis_scale > 0:
            S[i, :] *= slider_scale / analysis_scale

    # --- Filter out near-zero rows (components with negligible impact) ---
    row_norms = np.linalg.norm(S, axis=1)
    threshold = np.median(row_norms) * 0.01
    active_mask = row_norms > threshold
    n_active = int(active_mask.sum())
    log(f"Active components: {n_active}/{n_components} (threshold={threshold:.6f})")

    if n_active < n_features:
        log(f"Warning: fewer active components ({n_active}) than features ({n_features}). "
            f"Using all components for better coverage.")
        S_active = S
        active_indices = list(range(n_components))
    else:
        S_active = S[active_mask]
        active_indices = [i for i, m in enumerate(active_mask) if m]

    # --- Compute pseudo-inverse ---
    log("Computing pseudo-inverse for disentangled directions...")
    # S_active is [N_active, M], pseudo-inverse is [M, N_active]
    S_pinv = np.linalg.pinv(S_active)  # [M, N_active]

    # Map back to full component space
    directions = np.zeros((n_features, n_components))
    for col_idx, comp_idx in enumerate(active_indices):
        directions[:, comp_idx] = S_pinv[:, col_idx]

    # --- Normalize directions ---
    # Scale each direction so that semantic coefficient=1.0 produces a
    # perturbation comparable in magnitude to a single raw slider at 1.0.
    #
    # The server computes: perturbation = sum_j(raw[j] * component_ranges[j] * direction_j)
    # With max-abs normalization, many raw[j] can be near +-1 simultaneously,
    # so the total perturbation is far larger than a single raw slider — causing
    # distortion when semantic sliders exceed ~+-0.25.
    #
    # Fix: normalize each direction row by its L1 norm (sum of absolute weights)
    # so that when a semantic slider is at 1.0, the summed absolute raw
    # coefficients equal 1.0. This means the total perturbation magnitude is
    # bounded to roughly what one raw slider at 1.0 would produce.
    for i in range(n_features):
        l1_norm = np.sum(np.abs(directions[i]))
        if l1_norm > 1e-10:
            directions[i] /= l1_norm

    # --- Save ---
    result = {
        "feature_names": feature_names,
        "raw_feature_keys": feature_keys,
        "directions": directions.tolist(),
        "component_count": n_components,
        "feature_count": n_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    log(f"Saved semantic map to {output_path}")
    log(f"  {n_features} semantic features x {n_components} components")
    log(f"  Features: {', '.join(feature_names)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build semantic map from discovery catalog",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(DEFAULT_CATALOG),
        help="Path to discovery catalog (.pt)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=str(VOICES_DIR / "af_heart.pt"),
        help="Base voice for sensitivity analysis",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, how are you today?",
        help="Text for synthesis",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output path for semantic map JSON",
    )
    parser.add_argument(
        "--voice-folder",
        default=str(VOICES_DIR),
        help="Folder containing .pt voice files",
    )
    args = parser.parse_args()

    build_semantic_map(
        catalog_path=Path(args.catalog),
        voice_path=Path(args.voice),
        voice_folder=args.voice_folder,
        text=args.text,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
