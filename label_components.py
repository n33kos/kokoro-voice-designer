#!/usr/bin/env python3
"""Label each component in the discovery catalog with its dominant audio feature.

Runs sensitivity analysis: for each component, perturb the voice, generate audio,
measure feature changes, and assign the FEATURE_LABEL with the largest delta.

Saves the result as a simple JSON list to catalog/component_labels.json.

Usage:
    uv run python label_components.py
    uv run python label_components.py --catalog catalog/discovery_catalog.pt
    uv run python label_components.py --voice voices/af_heart.pt --text "Hello there."
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch

# Suppress noisy deprecation warnings from Kokoro/PyTorch internals
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

from core import FitnessScorer, SpeechGenerator, VoiceAnalyzer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
VOICES_DIR = PROJECT_ROOT / "voices"
CATALOG_DIR = PROJECT_ROOT / "catalog"
DEFAULT_CATALOG = CATALOG_DIR / "discovery_catalog.pt"
DEFAULT_LABELS_OUTPUT = CATALOG_DIR / "component_labels.json"

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


def label_components(
    catalog_path: Path,
    voice_folder: Path,
    base_voice_path: Path | None,
    text: str,
    output_path: Path,
) -> list[str]:
    """Run sensitivity analysis on every catalog component and return labels."""

    # --- Load catalog ---
    if not catalog_path.exists():
        print(f"Error: catalog not found at {catalog_path}")
        sys.exit(1)

    log(f"Loading catalog from {catalog_path}...")
    catalog = torch.load(catalog_path, weights_only=False)

    n_pca = catalog.get("pca_n_components", 0)
    pca_comps = catalog["pca_components"][:n_pca]
    pca_sv = catalog["pca_singular_values"][:n_pca]
    pca_mean = catalog["pca_mean"]

    # Discovery components
    disc_comps = catalog.get("discovered_components")
    if disc_comps is not None and disc_comps.shape[0] > 0:
        avg_sv = float(pca_sv.mean())
        disc_sv = torch.full((disc_comps.shape[0],), avg_sv)
        all_components = torch.cat([pca_comps, disc_comps], dim=0)
        all_sv = torch.cat([pca_sv, disc_sv], dim=0)
    else:
        all_components = pca_comps
        all_sv = pca_sv

    n_total = all_components.shape[0]
    log(f"Catalog has {n_pca} PCA + {n_total - n_pca} discovery = {n_total} total components")

    # --- Load voice library for component ranges ---
    log(f"Loading voices from {voice_folder}...")
    pt_files = sorted(f for f in os.listdir(voice_folder) if f.endswith(".pt"))
    if len(pt_files) < 2:
        print(f"Error: need at least 2 .pt files in {voice_folder}, found {len(pt_files)}")
        sys.exit(1)

    voice_tensors = [load_voice(voice_folder / f) for f in pt_files]
    n_voices = len(voice_tensors)

    # Component ranges
    flat = torch.stack(voice_tensors).reshape(n_voices, -1).float()
    centered = flat - pca_mean
    projections = centered @ all_components.T
    component_ranges = projections.max(dim=0).values - projections.min(dim=0).values

    # Fix discovery ranges (same logic as server.py)
    if disc_comps is not None and disc_comps.shape[0] > 0:
        avg_pca_range = float(component_ranges[:n_pca].mean())
        component_ranges[n_pca:] = avg_pca_range

    # --- Pick base voice ---
    if base_voice_path is not None:
        base_voice = load_voice(base_voice_path)
        log(f"Using base voice: {base_voice_path.name}")
    else:
        # Default: af_heart if available, else first voice
        heart_idx = next((i for i, f in enumerate(pt_files) if f == "af_heart.pt"), 0)
        base_voice = voice_tensors[heart_idx]
        log(f"Using base voice: {pt_files[heart_idx]}")

    # --- Initialize Kokoro ---
    log("Initializing Kokoro pipeline...")
    speech_gen = SpeechGenerator()

    # --- Sensitivity analysis ---
    log(f"Running sensitivity analysis on {n_total} components (1 synthesis each)...")
    base_flat = base_voice.reshape(-1).float()
    base_audio = speech_gen.generate_audio(text, base_voice)
    base_features = FitnessScorer.extract_features(base_audio)

    base_min = float(base_flat.min())
    base_max = float(base_flat.max())
    clamp_lo = base_min - abs(base_min) * 2
    clamp_hi = base_max + abs(base_max) * 2

    sensitivity: dict[int, dict] = {}
    n_skipped = 0
    t_start = time.time()

    for i in range(n_total):
        component = all_components[i]
        scale = float(all_sv[i]) * 0.1
        perturbed_flat = base_flat + component * scale
        perturbed_flat = torch.nan_to_num(perturbed_flat, nan=0.0, posinf=clamp_hi, neginf=clamp_lo)
        perturbed_flat = perturbed_flat.clamp(clamp_lo, clamp_hi)
        perturbed_voice = perturbed_flat.reshape(base_voice.shape)

        try:
            audio = speech_gen.generate_audio(text, perturbed_voice)
            features = FitnessScorer.extract_features(audio)
            deltas = {}
            for key in base_features:
                bv = base_features[key]
                nv = features[key]
                deltas[key] = (nv - bv) / abs(bv) if abs(bv) > 1e-8 else (nv - bv)
            sensitivity[i] = deltas
        except Exception as e:
            log(f"  Skipping component {i} (error: {str(e)[:80]})")
            sensitivity[i] = {k: 0.0 for k in base_features}
            n_skipped += 1

        # Progress
        done = i + 1
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n_total - done) / rate if rate > 0 else 0
        if done % 5 == 0 or done == n_total:
            log(f"  [{done}/{n_total}] {done/n_total:.0%} done, "
                f"{rate:.1f} comp/s, ETA {eta:.0f}s ({n_skipped} skipped)")

    # --- Assign labels ---
    log("Assigning labels...")
    name_counts: dict[str, int] = {}
    labels: list[str] = []

    for i in range(n_total):
        deltas = sensitivity.get(i, {})
        analysis_scale = float(all_sv[i]) * 0.1
        slider_scale = float(component_ranges[i]) if i < len(component_ranges) else 1.0
        scale_factor = slider_scale / analysis_scale if analysis_scale > 0 else 1.0

        scaled = {k: abs(v) * scale_factor for k, v in deltas.items()}
        sorted_feats = sorted(scaled.items(), key=lambda x: x[1], reverse=True)

        # Find the dominant feature label (allow duplicates with numbering)
        name = f"d{i}"
        for feat, _ in sorted_feats:
            if feat in FEATURE_LABELS:
                name = FEATURE_LABELS[feat]
                break

        # Add suffix for duplicates: "pitch", "pitch 2", "pitch 3", etc.
        count = name_counts.get(name, 0) + 1
        name_counts[name] = count
        if count > 1:
            name = f"{name} {count}"

        labels.append(name)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)

    log(f"Saved {len(labels)} labels to {output_path}")

    # Summary
    named = [l for l in labels if not l.startswith("d")]
    unnamed = [l for l in labels if l.startswith("d")]
    log(f"  Named: {len(named)} ({', '.join(named[:10])}{'...' if len(named) > 10 else ''})")
    log(f"  Unnamed (dN): {len(unnamed)}")
    log(f"  Skipped: {n_skipped}")

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label catalog components via sensitivity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(DEFAULT_CATALOG),
        help="Path to discovery catalog (.pt)",
    )
    parser.add_argument(
        "--voice-folder",
        type=str,
        default=str(VOICES_DIR),
        help="Folder containing .pt voice files",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Base voice .pt file for synthesis (default: af_heart.pt or first voice)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, how are you today?",
        help="Text to synthesize for sensitivity measurement",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_LABELS_OUTPUT),
        help="Output JSON path for labels",
    )
    args = parser.parse_args()

    label_components(
        catalog_path=Path(args.catalog),
        voice_folder=Path(args.voice_folder),
        base_voice_path=Path(args.voice) if args.voice else None,
        text=args.text,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
