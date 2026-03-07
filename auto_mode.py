#!/usr/bin/env python3
"""Auto mode: continuously run discovery → analysis → auto-tune, loop indefinitely.

Each iteration:
  1. Run N discovery probes (accumulates into discovery cache)
  2. Load components from catalog (if provided) or compute PCA + load discoveries
  3. Sensitivity analysis on all active components
  4. Coordinate descent auto-tune
  5. Save tuned voice (timestamped + overwrites designed_voice.pt)
  6. Repeat using tuned voice as new base (unless --keep-base)

Stop at any time with Ctrl+C.

Usage:
    # With a pre-built catalog (skips PCA computation each iteration):
    python auto_mode.py \\
        --base-voice voices/af_heart.pt \\
        --target-audio my_target.wav \\
        --target-text "Hello, my name is Alex." \\
        --catalog catalog/discovery_catalog.pt \\
        --n-probes 50 --n-pca 32 --n-discovery 64 \\
        --passes 3

    # Without catalog (computes PCA fresh each iteration):
    python auto_mode.py \\
        --base-voice voices/af_heart.pt \\
        --target-audio my_target.wav \\
        --target-text "Hello, my name is Alex." \\
        --n-probes 200 --n-pca 32 --n-discovery 32 \\
        --passes 3
"""

import argparse
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

# Suppress noisy deprecation warnings from Kokoro/PyTorch internals
warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*", category=UserWarning)

from core import DiscoveryAnalysis, FitnessScorer, SpeechGenerator, VoiceAnalyzer

OUTPUT_DIR = Path(__file__).parent / "output"
CATALOG_DIR = Path(__file__).parent / "catalog"
DEFAULT_CATALOG = CATALOG_DIR / "discovery_catalog.pt"

FEATURE_LABELS = {
    "pitch_mean": "pitch",
    "pitch_std": "pitch_variation",
    "spectral_centroid_mean": "brightness",
    "spectral_bandwidth_mean": "fullness",
    "spectral_rolloff_mean": "crispness",
    "spectral_contrast_mean": "clarity",
    "spectral_flatness_mean": "breathiness",
    "rms_energy": "volume",
    "energy_mean": "energy",
    "energy_std": "energy_variation",
    "mfcc1_mean": "timbre",
    "mfcc2_mean": "nasality",
    "mfcc3_mean": "resonance",
    "mfcc4_mean": "texture",
    "chroma_mean": "harmonics",
    "tonnetz_mean": "tonality",
    "tempo": "pace",
    "audio_std": "dynamics",
    "harmonic_ratio": "harmonic_richness",
    "zero_crossing_rate": "sibilance",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_voice(path: str) -> torch.Tensor:
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def convert_audio(path: str) -> str:
    audio, _ = librosa.load(path, sr=24000, mono=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, 24000)
    return tmp.name


def compute_impacts_and_names(
    sensitivity: dict,
    n_active: int,
    all_singular_values: torch.Tensor,
    component_ranges: torch.Tensor,
) -> tuple[dict, dict, dict]:
    """Compute per-slider impact scores and human-readable names."""
    used_names: set[str] = set()
    comp_names: dict[int, str] = {}
    impacts: dict[int, float] = {}
    dominant_sens: dict[int, float] = {}

    for i in range(n_active):
        analysis_scale = float(all_singular_values[i]) * 0.1
        slider_scale = float(component_ranges[i]) if i < len(component_ranges) else 1.0
        scale_factor = slider_scale / analysis_scale if analysis_scale > 0 else 1.0

        deltas = sensitivity.get(i, {})
        scaled = {k: abs(v) * scale_factor for k, v in deltas.items()}
        impacts[i] = sum(scaled.values())

        sorted_feats = sorted(scaled.items(), key=lambda x: x[1], reverse=True)
        name = f"comp_{i}"
        dom = 0.0
        for feat, feat_impact in sorted_feats:
            if feat in FEATURE_LABELS:
                base_name = FEATURE_LABELS[feat]
                if base_name not in used_names:
                    name = base_name
                    dom = feat_impact
                    used_names.add(base_name)
                    break
        comp_names[i] = name
        dominant_sens[i] = dom

    return comp_names, impacts, dominant_sens


# ---------------------------------------------------------------------------
# Core iteration
# ---------------------------------------------------------------------------

def load_catalog(catalog_path: Path) -> dict | None:
    """Load catalog if it exists, return None otherwise."""
    if not catalog_path.exists():
        return None
    try:
        cat = torch.load(catalog_path, weights_only=False)
        log(f"Loaded catalog: {catalog_path.name} "
            f"({cat.get('pca_n_components', 0)} PCA + "
            f"{len(cat.get('discovered_impacts', []))} discoveries)")
        return cat
    except Exception as e:
        log(f"Warning: could not load catalog ({e}) — falling back to fresh PCA")
        return None


def run_iteration(
    iteration: int,
    base_voice: torch.Tensor,
    voice_shape: tuple,
    voice_folder: str,
    speech_gen: SpeechGenerator,
    fitness: FitnessScorer,
    target_text: str,
    n_probes: int,
    n_pca: int,
    n_discovery: int,
    passes: int,
    start_step: float,
    mag_steps: int,
    output_dir: Path,
    catalog: dict | None = None,
) -> tuple[torch.Tensor, float, float]:
    """Run one full iteration. Returns (tuned_voice, best_similarity, base_similarity)."""

    sep = "=" * 60
    log(f"\n{sep}")
    log(f"Iteration {iteration}")
    log(sep)

    # Load voice library
    pt_files = sorted(f for f in os.listdir(voice_folder) if f.endswith(".pt"))
    voice_tensors = [load_voice(os.path.join(voice_folder, f)) for f in pt_files]

    # ------------------------------------------------------------------
    # Step 1: Discovery (accumulates into working cache)
    # ------------------------------------------------------------------
    log(f"Step 1: Running {n_probes} discovery probes...")
    discovery = DiscoveryAnalysis(output_dir)
    last_frac = [-1]

    def disc_progress(frac: float, desc: str) -> None:
        bucket = int(frac * 10)
        if bucket != last_frac[0]:
            last_frac[0] = bucket
            log(f"  [{frac:.0%}] {desc}")

    discovery.run(voice_tensors, pt_files, speech_gen, n_probes, progress_fn=disc_progress)

    # ------------------------------------------------------------------
    # Step 2: Load components from catalog (if available) or compute fresh
    # ------------------------------------------------------------------
    if catalog is not None:
        # Use PCA from catalog — no recomputation needed
        n_pca_from_catalog = min(n_pca, catalog.get("pca_n_components", 0))
        log(f"Step 2: Loading {n_pca_from_catalog} PCA from catalog + "
            f"{n_discovery} discoveries...")
        pca_comps = catalog["pca_components"][:n_pca_from_catalog]
        pca_sv = catalog["pca_singular_values"][:n_pca_from_catalog]
        per_comp_var = catalog["pca_variance"][:n_pca_from_catalog]
        pca_mean = catalog["pca_mean"]  # needed for component ranges
        analyzer_mean = pca_mean
    else:
        n_pca_actual = min(n_pca, len(voice_tensors) - 1)
        log(f"Step 2: PCA ({n_pca_actual}) + discovery ({n_discovery}) components...")
        analyzer = VoiceAnalyzer(voice_tensors, n_components=n_pca_actual)
        pca_comps = analyzer.components
        pca_sv = analyzer.singular_values
        per_comp_var = analyzer.per_component_variance
        analyzer_mean = analyzer.mean

    # Load top-N discoveries: merge catalog discoveries + new working cache
    disc_comps = None
    disc_sv = None

    # Collect from catalog first (already ranked by impact)
    cat_disc = None
    if catalog is not None and catalog.get("discovered_components") is not None:
        cat_all = catalog["discovered_components"]
        cat_impacts = catalog.get("discovered_impacts", [])
        if cat_all.shape[0] > 0:
            cat_disc = cat_all  # already in impact-descending order

    # Then from working cache (may have newer/additional discoveries)
    cache_data = discovery.load_cache()
    discovered_all = cache_data.get("components")
    working_disc = None
    if discovered_all is not None and discovered_all.shape[0] > 0:
        ranked = cache_data.get("ranked_indices", list(range(discovered_all.shape[0])))
        top_indices = ranked[:n_discovery]
        working_disc = discovered_all[top_indices]
        log(f"  Working cache: {working_disc.shape[0]} of {discovered_all.shape[0]} discoveries")

    # Assemble discovery components (catalog first, then working cache additions)
    disc_parts = []
    if cat_disc is not None:
        n_cat = min(n_discovery, cat_disc.shape[0])
        disc_parts.append(cat_disc[:n_cat])
        log(f"  Catalog discoveries: {n_cat} of {cat_disc.shape[0]}")
    if working_disc is not None:
        # Add up to n_discovery total, filling from working cache
        already = sum(d.shape[0] for d in disc_parts)
        remaining = max(0, n_discovery - already)
        if remaining > 0:
            extra = working_disc[:remaining]
            disc_parts.append(extra)
            log(f"  Working cache additions: {extra.shape[0]}")

    if disc_parts:
        disc_comps = torch.cat(disc_parts, dim=0)
        avg_sv = float(pca_sv.mean())
        disc_sv = torch.full((disc_comps.shape[0],), avg_sv)
        log(f"  Total discoveries: {disc_comps.shape[0]}")

    # Assemble
    all_comps_parts = [pca_comps]
    all_sv_parts = [pca_sv]
    if disc_comps is not None:
        all_comps_parts.append(disc_comps)
        all_sv_parts.append(disc_sv)

    all_components = torch.cat(all_comps_parts, dim=0)    # [N, D]
    all_singular_values = torch.cat(all_sv_parts, dim=0)  # [N]
    n_active = all_components.shape[0]

    # Component ranges (project voice library for slider scaling)
    n_voices = len(voice_tensors)
    flat = torch.stack(voice_tensors).reshape(n_voices, -1).float()
    centered = flat - analyzer_mean
    projections = centered @ all_components.T
    component_ranges = projections.max(dim=0).values - projections.min(dim=0).values

    # Discovery components are orthogonal to PCA, so the voice library barely projects
    # onto them — their component_ranges are tiny, making auto-tune steps negligibly
    # small. Replace discovery ranges with the average PCA range so coord descent can
    # actually move the voice along these directions.
    n_pca_actual = pca_comps.shape[0]
    if disc_comps is not None and disc_comps.shape[0] > 0:
        avg_pca_range = float(component_ranges[:n_pca_actual].mean())
        component_ranges[n_pca_actual:] = avg_pca_range

    # Pad per_component_variance with zeros for discovery components
    n_extra = all_components.shape[0] - len(per_comp_var)
    if n_extra > 0:
        per_comp_var = torch.cat([per_comp_var, torch.zeros(n_extra)])

    # ------------------------------------------------------------------
    # Step 3: Sensitivity analysis
    # ------------------------------------------------------------------
    log(f"Step 3: Sensitivity analysis on {n_active} components...")
    base_audio = speech_gen.generate_audio(target_text, base_voice)
    base_flat = base_voice.reshape(-1).float()
    base_features = FitnessScorer.extract_features(base_audio)

    # Reference value range from the base voice for clamping
    base_min = float(base_flat.min())
    base_max = float(base_flat.max())
    clamp_lo = base_min - abs(base_min) * 2
    clamp_hi = base_max + abs(base_max) * 2

    sensitivity: dict[int, dict] = {}
    n_skipped = 0
    for i in range(n_active):
        component = all_components[i]
        scale = float(all_singular_values[i]) * 0.1
        perturbed_flat = base_flat + component * scale
        # Guard against NaN/Inf and extreme values that crash MPS
        perturbed_flat = torch.nan_to_num(perturbed_flat, nan=0.0, posinf=clamp_hi, neginf=clamp_lo)
        perturbed_flat = perturbed_flat.clamp(clamp_lo, clamp_hi)
        perturbed_voice = perturbed_flat.reshape(voice_shape)
        try:
            audio = speech_gen.generate_audio(target_text, perturbed_voice)
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

        if (i + 1) % 8 == 0 or i == n_active - 1:
            log(f"  Analyzed {i + 1}/{n_active} components ({n_skipped} skipped)")

    comp_names, impacts, dominant_sensitivity = compute_impacts_and_names(
        sensitivity, n_active, all_singular_values, component_ranges
    )
    impact_order = sorted(range(n_active), key=lambda i: impacts.get(i, 0), reverse=True)
    top5 = ", ".join(f"{comp_names[i]}={impacts[i]:.1f}" for i in impact_order[:5])
    log(f"  Top 5 impacts: {top5}")

    # ------------------------------------------------------------------
    # Step 4: Auto-tune (coordinate descent)
    # ------------------------------------------------------------------
    log(f"Step 4: Auto-tune ({passes} passes, {n_active} components)...")

    base_sim = float(fitness.target_similarity(base_audio))
    log(f"  Baseline similarity: {base_sim:.4f}")

    coeffs = torch.zeros(all_components.shape[0])
    best_sim = base_sim
    eval_count = 0

    def score(c: torch.Tensor) -> float:
        nonlocal eval_count
        scaled = c * component_ranges
        p = (scaled.unsqueeze(0) @ all_components).squeeze(0)
        voice_flat = (base_voice.reshape(-1).float() + p)
        voice_flat = torch.nan_to_num(voice_flat, nan=0.0, posinf=clamp_hi, neginf=clamp_lo)
        voice_flat = voice_flat.clamp(clamp_lo, clamp_hi)
        voice = voice_flat.reshape(voice_shape)
        try:
            audio = speech_gen.generate_audio(target_text, voice)
            eval_count += 1
            return float(fitness.target_similarity(audio))
        except Exception:
            eval_count += 1
            return 0.0

    sens_values = [dominant_sensitivity.get(i, 0) for i in range(n_active)]
    median_sens = sorted(sens_values)[len(sens_values) // 2] if sens_values else 1.0
    median_sens = max(median_sens, 1e-6)

    for pass_idx in range(passes):
        pass_start = best_sim

        for i in impact_order:
            name = comp_names.get(i, f"comp_{i}")
            current = coeffs[i].item()
            best_val = current
            best_comp_sim = best_sim

            # Per-component step sizes scaled by sensitivity
            s = dominant_sensitivity.get(i, 0)
            ratio = max(0.1, min(10.0, median_sens / s)) if s > 0 else 1.0
            base_s = start_step * ratio
            comp_steps = [base_s / (10 ** m) for m in range(mag_steps)]

            for step in comp_steps:
                for sign in [1.0, -1.0]:
                    candidate = max(-2.0, min(2.0, current + sign * step))
                    if abs(candidate - current) < 1e-6:
                        continue
                    coeffs[i] = candidate
                    sim = score(coeffs)
                    if sim > best_comp_sim:
                        best_comp_sim = sim
                        best_val = candidate

                # Found improvement — bisect then stop
                if best_val != current:
                    mid = (current + best_val) / 2.0
                    coeffs[i] = mid
                    mid_sim = score(coeffs)
                    if mid_sim > best_comp_sim:
                        best_comp_sim = mid_sim
                        best_val = mid
                    break

            coeffs[i] = best_val
            if best_val != current:
                best_sim = best_comp_sim

        improvement = best_sim - pass_start
        log(f"  Pass {pass_idx + 1}/{passes}: sim={best_sim:.4f} (+{improvement:.4f}), evals={eval_count}")

    delta = best_sim - base_sim
    log(f"  Final: sim={best_sim:.4f} (delta={delta:+.4f}), {eval_count} evaluations")

    # ------------------------------------------------------------------
    # Step 5: Save
    # ------------------------------------------------------------------
    scaled = coeffs * component_ranges
    p = (scaled.unsqueeze(0) @ all_components).squeeze(0)
    tuned_flat = (base_voice.reshape(-1).float() + p)
    tuned_flat = torch.nan_to_num(tuned_flat, nan=0.0, posinf=clamp_hi, neginf=clamp_lo)
    tuned_flat = tuned_flat.clamp(clamp_lo, clamp_hi)
    tuned_voice = tuned_flat.reshape(voice_shape)

    ts = time.strftime("%Y%m%d_%H%M%S")
    iter_path = output_dir / f"auto_voice_iter{iteration:04d}_{ts}_sim{best_sim:.4f}.pt"
    torch.save(tuned_voice, iter_path)
    torch.save(tuned_voice, output_dir / "designed_voice.pt")
    log(f"  Saved: {iter_path.name}")

    return tuned_voice, best_sim, base_sim


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto mode: continuous discovery → analysis → auto-tune loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-voice", required=True, help="Base .pt voice file")
    parser.add_argument("--target-audio", required=True, help="Target audio file")
    parser.add_argument("--target-text", required=True, help="Text for synthesis comparison")
    parser.add_argument(
        "--voice-folder",
        default=str(Path(__file__).parent / "voices"),
        help="Folder containing .pt voice files",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(DEFAULT_CATALOG) if DEFAULT_CATALOG.exists() else None,
        help="Path to discovery catalog (.pt). Skips PCA recomputation if provided. "
             f"Auto-detected from {DEFAULT_CATALOG} if present.",
    )
    parser.add_argument("--n-probes", type=int, default=200, help="Discovery probes per iteration")
    parser.add_argument("--n-pca", type=int, default=32, help="PCA components to use")
    parser.add_argument("--n-discovery", type=int, default=32, help="Top discovery components to load")
    parser.add_argument("--passes", type=int, default=3, help="Auto-tune passes per iteration")
    parser.add_argument("--start-step", type=float, default=0.1, help="Starting step size")
    parser.add_argument("--mag-steps", type=int, default=3, help="Magnitude step reductions (10x each)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument(
        "--keep-base",
        action="store_true",
        help="Always start from the original base voice instead of carrying over tuning",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Initializing Kokoro pipeline...")
    speech_gen = SpeechGenerator()

    log("Loading target audio and initializing fitness scorer...")
    target_wav_path = convert_audio(args.target_audio)
    fitness = FitnessScorer(target_wav_path, device=speech_gen.device)

    original_base = load_voice(args.base_voice)
    voice_shape = original_base.shape
    current_voice = original_base

    # Load catalog if provided or auto-detected
    catalog = None
    if args.catalog:
        catalog = load_catalog(Path(args.catalog))

    log("Auto mode started. Press Ctrl+C to stop.")
    log(f"  Catalog     : {args.catalog or 'none (fresh PCA each iteration)'}")
    log(f"  Probes/iter : {args.n_probes}")
    log(f"  PCA comps   : {args.n_pca}")
    log(f"  Discovery   : {args.n_discovery}")
    log(f"  Passes      : {args.passes}")
    log(f"  Step        : {args.start_step}  Mag steps: {args.mag_steps}")
    log(f"  Base mode   : {'fixed (original)' if args.keep_base else 'rolling (use tuned voice)'}")

    iteration = 0
    try:
        while True:
            iteration += 1
            tuned_voice, best_sim, iter_base_sim = run_iteration(
                iteration=iteration,
                base_voice=current_voice,
                voice_shape=voice_shape,
                voice_folder=args.voice_folder,
                speech_gen=speech_gen,
                fitness=fitness,
                target_text=args.target_text,
                n_probes=args.n_probes,
                n_pca=args.n_pca,
                n_discovery=args.n_discovery,
                passes=args.passes,
                start_step=args.start_step,
                mag_steps=args.mag_steps,
                output_dir=output_dir,
                catalog=catalog,
            )
            if not args.keep_base:
                if best_sim >= iter_base_sim:
                    current_voice = tuned_voice
                    log(f"Carrying tuned voice into iteration {iteration + 1} (sim={best_sim:.4f} should be next baseline).")
                else:
                    log(f"Iteration {iteration} regressed ({best_sim:.4f} < {iter_base_sim:.4f}) — keeping previous voice.")
    except KeyboardInterrupt:
        log(f"\nStopped after {iteration} iteration(s).")
        log(f"Best result saved to: {output_dir}/designed_voice.pt")

        # Synthesize a WAV preview from the best voice
        try:
            audio = speech_gen.generate_audio(args.target_text, current_voice)
            wav_path = output_dir / "designed_voice.wav"
            sf.write(str(wav_path), audio, 24000)
            log(f"WAV saved to: {wav_path}")
        except Exception as e:
            log(f"Warning: could not synthesize WAV ({e})")

        sys.exit(0)


if __name__ == "__main__":
    main()
