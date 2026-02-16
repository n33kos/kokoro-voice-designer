#!/usr/bin/env python3
"""Voice Designer — interactive PCA-based voice crafting with Gradio UI."""

import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from core import VoiceAnalyzer, SpeechGenerator, FitnessScorer, DiscoveryAnalysis

VOICES_DIR = Path(__file__).parent / "voices"
OUTPUT_DIR = Path(__file__).parent / "output"


def ensure_voices():
    """Download Kokoro voices from HuggingFace if not already present."""
    if VOICES_DIR.exists() and any(VOICES_DIR.glob("*.pt")):
        return
    print("Downloading Kokoro voices from hexgrad/Kokoro-82M...")
    local = snapshot_download(
        repo_id="hexgrad/Kokoro-82M",
        allow_patterns="voices/*.pt",
        local_dir=Path(__file__).parent / "_hf_cache",
    )
    src = Path(local) / "voices"
    VOICES_DIR.mkdir(exist_ok=True)
    for pt in sorted(src.glob("*.pt")):
        dest = VOICES_DIR / pt.name
        if not dest.exists():
            dest.symlink_to(pt.resolve())
    print(f"Linked {len(list(VOICES_DIR.glob('*.pt')))} voices to {VOICES_DIR}")


def get_discovery_cache_info(cache_path: Path) -> dict | None:
    """Return dict with path, count, and timestamp if discovery cache exists."""
    if not cache_path.exists():
        return None
    try:
        cache = torch.load(cache_path, weights_only=False)
        n_components = cache.get("components", torch.zeros(0, 1)).shape[0]
        timestamp = cache.get("timestamp", 0)
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        return {
            "path": str(cache_path),
            "count": n_components,
            "timestamp": timestamp_str,
        }
    except Exception:
        return None


# Human-readable labels for audio features
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

MAX_SLIDERS = 64

# Module-level state (Kokoro pipeline isn't picklable for gr.State)
_state: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def convert_audio(path: str) -> str:
    """Convert any audio file to mono WAV at 24kHz, return path to temp file."""
    audio, _ = librosa.load(path, sr=24000, mono=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, 24000)
    return tmp.name


def load_voice(path: str) -> torch.Tensor:
    """Load a .pt voice tensor, trying weights_only=True first."""
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def analyze_components(sensitivity: dict, n_components: int,
                       singular_values: torch.Tensor,
                       component_ranges: torch.Tensor,
                       ) -> tuple[dict[int, str], dict[int, float], dict[int, float]]:
    """Name components and compute per-slider-unit impact scores.

    Sensitivity was measured at perturbation = singular_values[i] * 0.1.
    Sliders perturb by component_ranges[i] per unit. We scale the deltas
    so impact reflects "what happens when you move this slider by 1.0".

    Returns:
        names: component index -> human-readable name
        impacts: component index -> total audio impact per slider unit (sum of |scaled deltas|)
        dominant_sensitivity: component index -> dominant feature's absolute sensitivity per slider unit
    """
    used_names: set[str] = set()
    names: dict[int, str] = {}
    impacts: dict[int, float] = {}
    dominant_sensitivity: dict[int, float] = {}

    for i in range(n_components):
        if i not in sensitivity:
            names[i] = f"component_{i}"
            impacts[i] = 0.0
            dominant_sensitivity[i] = 0.0
            continue

        # Scale factor: convert from analysis perturbation to per-slider-unit
        analysis_scale = float(singular_values[i]) * 0.1
        slider_scale = float(component_ranges[i]) if i < len(component_ranges) else 1.0
        scale_factor = slider_scale / analysis_scale if analysis_scale > 0 else 1.0

        deltas = sensitivity[i]
        scaled_deltas = {k: abs(v) * scale_factor for k, v in deltas.items()}

        # Total impact = sum of all scaled absolute deltas
        impacts[i] = sum(scaled_deltas.values())

        # Name by dominant feature
        sorted_features = sorted(scaled_deltas.items(), key=lambda x: x[1], reverse=True)

        chosen_name = f"component_{i}"
        dominant_sens = 0.0
        for feat_name, feat_impact in sorted_features:
            if feat_name in FEATURE_LABELS:
                base_name = FEATURE_LABELS[feat_name]
                if base_name not in used_names:
                    chosen_name = base_name
                    dominant_sens = feat_impact
                    used_names.add(base_name)
                    break

        names[i] = chosen_name
        dominant_sensitivity[i] = dominant_sens

    return names, impacts, dominant_sensitivity


def compute_sorted_gaps(current_features: dict, target_features: dict) -> list[tuple[str, str, float]]:
    """Compute sorted (label, key, gap_pct) tuples."""
    gaps = []
    for key in target_features:
        if key in current_features:
            t_val = target_features[key]
            c_val = current_features[key]
            if abs(t_val) > 1e-8:
                gap_pct = (t_val - c_val) / abs(t_val) * 100
            else:
                gap_pct = (t_val - c_val) * 100
            label = FEATURE_LABELS.get(key, key)
            gaps.append((label, key, gap_pct))
    gaps.sort(key=lambda x: abs(x[2]), reverse=True)
    return gaps


def build_slider_updates(n_active, comp_names, per_component_variance,
                         dominant_sensitivity=None, preserve_values=None):
    """Build gr.update list for all MAX_SLIDERS slots."""
    updates = []
    for i in range(MAX_SLIDERS):
        if i < n_active:
            name = comp_names.get(i, f"component_{i}")
            var_pct = float(per_component_variance[i]) * 100 if i < len(per_component_variance) else 0
            val = preserve_values[i] if preserve_values is not None else 0.0
            # Show dominant feature sensitivity per slider unit
            if dominant_sensitivity and i in dominant_sensitivity and dominant_sensitivity[i] > 0:
                sens = dominant_sensitivity[i]
                label = f"{name} ({var_pct:.1f}%, \u0394{sens:.0%}/unit)"
            else:
                label = f"{name} ({var_pct:.1f}%)"
            updates.append(gr.update(label=label, interactive=True, value=val))
        else:
            updates.append(gr.update(label="(unused)", interactive=False, value=0.0))
    return updates


# ---------------------------------------------------------------------------
# Backend functions
# ---------------------------------------------------------------------------

def run_discovery(voice_folder, n_probes, progress=gr.Progress()):
    """Run offline discovery analysis exploring orthogonal directions beyond PCA.

    Probes random directions in voice parameter space to find impactful
    dimensions not captured by PCA. Results are cached as .pt tensors.
    """
    if not voice_folder or not os.path.isdir(voice_folder):
        raise gr.Error(f"Voice folder not found: {voice_folder}")

    n_probes = int(n_probes)

    # Load voice files
    pt_files = sorted(f for f in os.listdir(voice_folder) if f.endswith(".pt"))
    if len(pt_files) < 3:
        raise gr.Error(f"Need at least 3 .pt files in {voice_folder}, found {len(pt_files)}")

    progress(0.0, desc="Loading voices for discovery...")
    voice_tensors = []
    for f in pt_files:
        voice_tensors.append(load_voice(os.path.join(voice_folder, f)))

    # Reuse speech generator if available, otherwise create
    if "speech_gen" in _state:
        speech_gen = _state["speech_gen"]
    else:
        progress(0.02, desc="Initializing Kokoro pipeline...")
        speech_gen = SpeechGenerator()
        _state["speech_gen"] = speech_gen

    def progress_cb(frac, desc):
        progress(frac, desc=desc)

    discovery = DiscoveryAnalysis(OUTPUT_DIR)
    results = discovery.run(voice_tensors, pt_files, speech_gen, n_probes, progress_fn=progress_cb)

    # Summarize results
    total_probes = results.get("total_probes_run", n_probes)
    components = results.get("components")
    n_discovered = components.shape[0] if components is not None else 0
    impacts = results.get("impacts", [])
    ranked = results.get("ranked_indices", [])

    # Show top 5 impacts
    top_lines = []
    for idx in ranked[:5]:
        if idx < len(impacts):
            top_lines.append(f"  #{idx}: impact={impacts[idx]:.3f}")

    summary = (
        f"Discovery complete! {total_probes} total probes run, "
        f"{n_discovered} impactful directions found.\n"
    )
    if top_lines:
        summary += "Top impacts:\n" + "\n".join(top_lines)

    return summary


def run_analysis(base_voice_file, target_audio, target_text, target_text_file, voice_folder, n_components, use_pca=True, use_discovery=True, progress=gr.Progress()):
    """Load voices, build PCA, run sensitivity analysis.

    Incremental: if state already exists with sensitivity data, only analyzes new components.

    Args:
        use_pca: Whether to use PCA components
        use_discovery: Whether to use discovered components
    """
    if base_voice_file is None:
        raise gr.Error("Please upload a base voice .pt file.")
    if target_audio is None:
        raise gr.Error("Please upload a target audio file.")

    # Handle text from file or textbox
    if target_text_file:
        with open(target_text_file, 'r') as f:
            target_text = f.read().strip()

    if not target_text or not target_text.strip():
        raise gr.Error("Please enter target text or upload a .txt file.")
    if not voice_folder or not os.path.isdir(voice_folder):
        raise gr.Error(f"Voice folder not found: {voice_folder}")

    n_requested = int(n_components)

    # Check if we can reuse existing state (incremental expansion)
    have_existing = "analyzer" in _state and "sensitivity" in _state
    existing_sensitivity = _state.get("sensitivity", {})

    progress(0.0, desc="Loading base voice...")
    base_voice = load_voice(base_voice_file)

    # Load all .pt voices from folder
    progress(0.05, desc="Loading voice library...")
    voice_tensors = []
    pt_files = sorted(f for f in os.listdir(voice_folder) if f.endswith(".pt"))
    if len(pt_files) < 3:
        raise gr.Error(f"Need at least 3 .pt files in {voice_folder}, found {len(pt_files)}")

    for f in pt_files:
        voice_tensors.append(load_voice(os.path.join(voice_folder, f)))

    max_possible = min(len(voice_tensors) - 1, MAX_SLIDERS)
    n_active = min(n_requested, max_possible)

    # Reuse speech generator if available, otherwise create
    if have_existing and "speech_gen" in _state:
        progress(0.1, desc="Reusing Kokoro pipeline...")
        speech_gen = _state["speech_gen"]
    else:
        progress(0.1, desc="Initializing Kokoro pipeline...")
        speech_gen = SpeechGenerator()

    if have_existing and "fitness" in _state:
        progress(0.15, desc="Reusing fitness scorer...")
        fitness = _state["fitness"]
    else:
        progress(0.15, desc="Converting target audio...")
        target_wav_path = convert_audio(target_audio)
        fitness = FitnessScorer(target_wav_path, device=speech_gen.device)

    # PCA decomposition — always compute with max components for headroom
    progress(0.2, desc=f"Running PCA ({max_possible} components)...")
    analyzer = VoiceAnalyzer(voice_tensors, n_components=max_possible)

    # Load discoveries if cache exists
    discovery = DiscoveryAnalysis(OUTPUT_DIR)
    all_components = analyzer.components
    all_singular_values = analyzer.singular_values
    discovery_cache_data = None

    if discovery.cache_exists():
        try:
            progress(0.22, desc="Loading discovery cache...")
            discovery_cache_data = discovery.load_cache()
            cache_voice_hash = discovery_cache_data.get("voice_hash")
            current_voice_hash = discovery.compute_voice_hash(pt_files)
            if cache_voice_hash == current_voice_hash:
                discovered_components = discovery_cache_data.get("components")
                if discovered_components is not None and discovered_components.shape[0] > 0:
                    # Mix PCA + discoveries
                    all_components = torch.cat([all_components, discovered_components], dim=0)
                    # Extend singular values with average for discoveries
                    avg_sv = analyzer.singular_values.mean()
                    discovery_sv = torch.full((discovered_components.shape[0],), avg_sv)
                    all_singular_values = torch.cat([all_singular_values, discovery_sv], dim=0)
                    progress(0.23, desc=f"Loaded {discovered_components.shape[0]} discovered dimensions")
        except Exception as e:
            progress(0.22, desc=f"Could not load discoveries: {str(e)[:50]}")

    # Filter components based on user selection
    component_source_changed = False
    if not use_pca and not use_discovery:
        raise gr.Error("Must select at least one component source (PCA or Discoveries).")
    elif use_pca and not use_discovery:
        # PCA only: discard discoveries
        all_components = analyzer.components
        all_singular_values = analyzer.singular_values
        max_possible = min(len(voice_tensors) - 1, MAX_SLIDERS)
        n_active = min(n_requested, max_possible)
        component_source_changed = _state.get("component_source") != "pca"
        _state["component_source"] = "pca"
        progress(0.23, desc="Using PCA components only")
    elif use_discovery and not use_pca:
        # Discoveries only: use only discovered components
        if discovery_cache_data is None:
            raise gr.Error("No discoveries cached. Run Discovery first or select PCA.")
        discovered_components = discovery_cache_data.get("components")
        if discovered_components is None or discovered_components.shape[0] == 0:
            raise gr.Error("No discoveries found in cache. Run Discovery first or select PCA.")
        all_components = discovered_components
        D = int(torch.tensor(analyzer.voice_shape).prod().item())
        avg_sv = analyzer.singular_values.mean()
        all_singular_values = torch.full((all_components.shape[0],), avg_sv)
        max_possible = min(all_components.shape[0], MAX_SLIDERS)
        n_active = min(n_requested, max_possible)
        component_source_changed = _state.get("component_source") != "discovery"
        _state["component_source"] = "discovery"
        progress(0.23, desc=f"Using {all_components.shape[0]} discovered components only")
    else:
        # Both sources: already mixed above
        component_source_changed = _state.get("component_source") != "both"
        _state["component_source"] = "both"
        progress(0.23, desc="Using PCA + discovered components")

    # Clear sensitivity cache if component source changed to force re-analysis
    if component_source_changed:
        existing_sensitivity = {}

    # Compute component ranges for intuitive scaling
    n_voices = len(voice_tensors)
    flat = torch.stack(voice_tensors).reshape(n_voices, -1).float()
    centered = flat - analyzer.mean
    projections = centered @ all_components.T
    component_ranges = projections.max(dim=0).values - projections.min(dim=0).values

    # Determine which components still need sensitivity analysis
    components_to_analyze = [i for i in range(n_active) if i not in existing_sensitivity]
    total_to_analyze = len(components_to_analyze)

    # Generate base audio for comparison (reuse if already cached with same text)
    if "base_audio" not in _state or _state.get("target_text") != target_text:
        progress(0.25, desc="Generating base voice audio...")
        base_audio_np = speech_gen.generate_audio(target_text, base_voice)
    else:
        base_audio_np = _state["base_audio"]

    if total_to_analyze > 0:
        progress(0.28, desc=f"Sensitivity analysis on {total_to_analyze} components...")
        base_flat = base_voice.reshape(-1).float()
        base_features = fitness.extract_features(base_audio_np)

        for idx, i in enumerate(components_to_analyze):
            progress(0.28 + 0.65 * (idx / total_to_analyze),
                     desc=f"Analyzing component {i+1} ({idx+1}/{total_to_analyze} new)...")
            component = all_components[i]
            scale = float(all_singular_values[i]) * 0.1

            perturbed_flat = base_flat + component * scale
            perturbed_voice = perturbed_flat.reshape(analyzer.voice_shape)

            audio = speech_gen.generate_audio(target_text, perturbed_voice)
            features = fitness.extract_features(audio)

            deltas = {}
            for key in base_features:
                base_val = base_features[key]
                new_val = features[key]
                if abs(base_val) > 1e-8:
                    deltas[key] = (new_val - base_val) / abs(base_val)
                else:
                    deltas[key] = new_val - base_val
            existing_sensitivity[i] = deltas
    else:
        progress(0.93, desc="All components already analyzed, reusing...")

    analyzer.sensitivity = existing_sensitivity

    # Analyze components: names, impact scores, dominant sensitivity
    progress(0.95, desc="Computing impact scores...")
    comp_names, impacts, dominant_sens = analyze_components(
        existing_sensitivity, n_active, all_singular_values, component_ranges)

    # Rank by impact for auto-tune ordering
    impact_order = sorted(range(n_active), key=lambda i: impacts.get(i, 0), reverse=True)

    # Store everything in module-level state
    _state["analyzer"] = analyzer
    _state["all_components"] = all_components
    _state["all_singular_values"] = all_singular_values
    _state["base_voice"] = base_voice
    _state["base_audio"] = base_audio_np
    _state["speech_gen"] = speech_gen
    _state["fitness"] = fitness
    _state["component_names"] = comp_names
    _state["component_ranges"] = component_ranges
    _state["target_text"] = target_text
    _state["sensitivity"] = existing_sensitivity
    _state["n_active"] = n_active
    _state["impacts"] = impacts
    _state["dominant_sensitivity"] = dominant_sens
    _state["impact_order"] = impact_order

    progress(1.0, desc="Done!")

    slider_updates = build_slider_updates(
        n_active, comp_names, analyzer.per_component_variance, dominant_sens)

    new_count = total_to_analyze
    reused_count = n_active - total_to_analyze

    # Build impact ranking summary
    ranking_lines = []
    for rank, i in enumerate(impact_order[:10]):
        name = comp_names.get(i, f"comp_{i}")
        imp = impacts.get(i, 0)
        ds = dominant_sens.get(i, 0)
        ranking_lines.append(f"  {rank+1}. {name} (impact={imp:.1f}, dominant \u0394={ds:.0%}/unit)")

    pca_info = (
        f"PCA: {n_active} active / {analyzer.n_components} total components (max possible: {len(voice_tensors)-1} from {len(voice_tensors)} voices), "
        f"{analyzer.explained_variance_ratio:.1%} total variance. "
        f"({new_count} newly analyzed, {reused_count} reused)\n\n"
        f"Impact ranking (auto-tune order):\n" + "\n".join(ranking_lines)
    )

    return (*slider_updates, pca_info,
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))


def generate_voice(*args):
    """Build voice from slider values and generate audio."""
    slider_values = args[:MAX_SLIDERS]
    target_text = args[MAX_SLIDERS]

    if "analyzer" not in _state:
        raise gr.Error("Run analysis first!")

    analyzer = _state["analyzer"]
    all_components = _state.get("all_components", _state["analyzer"].components)
    n_active = _state["n_active"]
    base_voice = _state["base_voice"]
    speech_gen = _state["speech_gen"]
    fitness = _state["fitness"]
    component_ranges = _state["component_ranges"]
    base_audio_np = _state.get("base_audio")

    text = target_text.strip() if target_text and target_text.strip() else _state.get("target_text", "Hello world.")

    # Regenerate base audio if text changed
    if base_audio_np is None or text != _state.get("target_text", ""):
        base_audio_np = speech_gen.generate_audio(text, base_voice)
        _state["base_audio"] = base_audio_np

    # Build coefficient vector — only use active components
    coeffs = torch.zeros(all_components.shape[0])
    for i in range(n_active):
        coeffs[i] = float(slider_values[i])

    # Apply PCA perturbation
    scaled = coeffs * component_ranges
    perturbation_flat = (scaled.unsqueeze(0) @ all_components).squeeze(0)
    designed_voice = base_voice + perturbation_flat.reshape(analyzer.voice_shape)

    audio = speech_gen.generate_audio(text, designed_voice)
    target_sim = fitness.target_similarity(audio)
    base_sim = fitness.target_similarity(base_audio_np)

    current_features = fitness.extract_features(audio)
    gaps = compute_sorted_gaps(current_features, fitness.target_features)
    significant = [(l, g) for l, _, g in gaps if abs(g) > 5]

    score_text = f"Target similarity: {target_sim:.3f} (base: {base_sim:.3f}, delta: {target_sim - base_sim:+.3f})"
    gap_lines = [f"{label:<22} {gap_pct:+.1f}%" for label, gap_pct in significant[:10]]
    gap_text = "\n".join(gap_lines) if gap_lines else "All features within 5% of target."

    pt_path = str(OUTPUT_DIR / "designed_voice.pt")
    torch.save(designed_voice, pt_path)

    return (24000, audio), (24000, base_audio_np), score_text, gap_text, pt_path


def auto_tune(*args):
    """Coordinate descent generator: yields live slider + score updates on every evaluation."""
    slider_values = list(args[:MAX_SLIDERS])
    target_text_val = args[MAX_SLIDERS]
    max_passes = int(args[MAX_SLIDERS + 1])
    start_step = float(args[MAX_SLIDERS + 2])
    mag_steps = int(args[MAX_SLIDERS + 3])

    if "analyzer" not in _state:
        raise gr.Error("Run analysis first!")

    analyzer = _state["analyzer"]
    all_components = _state.get("all_components", _state["analyzer"].components)
    n_active = _state["n_active"]
    base_voice = _state["base_voice"]
    speech_gen = _state["speech_gen"]
    fitness = _state["fitness"]
    component_ranges = _state["component_ranges"]
    comp_names = _state["component_names"]
    impact_order = _state.get("impact_order", list(range(n_active)))
    text = (target_text_val.strip() if target_text_val and target_text_val.strip()
            else _state.get("target_text", "Hello world."))

    NO_CHANGE = gr.update()

    def score(coeffs):
        scaled = coeffs * component_ranges
        p = (scaled.unsqueeze(0) @ all_components).squeeze(0)
        voice = base_voice + p.reshape(analyzer.voice_shape)
        audio = speech_gen.generate_audio(text, voice)
        return float(fitness.target_similarity(audio))

    def make_slider_updates(coeffs):
        return [gr.update(value=round(coeffs[j].item(), 3)) if j < n_active
                else gr.update() for j in range(MAX_SLIDERS)]

    # Start from current slider positions
    coeffs = torch.zeros(all_components.shape[0])
    for i in range(n_active):
        coeffs[i] = float(slider_values[i])

    best_sim = score(coeffs)
    eval_count = 1
    log = [f"Baseline similarity: {best_sim:.4f}",
           f"Active components: {n_active} (tuning highest-impact first)"]

    # Per-component step sizes, scaled by inverse sensitivity:
    # High-impact components get smaller steps (they're sensitive),
    # low-impact ones get bigger steps (need more to move the needle).
    dominant_sens = _state.get("dominant_sensitivity", {})
    sens_values = [dominant_sens.get(i, 0) for i in range(n_active)]
    median_sens = sorted(sens_values)[len(sens_values) // 2] if sens_values else 1.0
    median_sens = max(median_sens, 1e-6)

    def steps_for_component(i):
        s = dominant_sens.get(i, 0)
        if s > 0:
            # Scale: if this component is 2x more sensitive than median,
            # use half the step size. If 0.5x, use double.
            ratio = median_sens / s
            ratio = max(0.1, min(10.0, ratio))  # clamp to 10x range
            base = start_step * ratio
        else:
            base = start_step
        return [base / (10 ** m) for m in range(mag_steps)]

    log.append(f"Base step: {start_step}, scaled per-component by sensitivity")

    yield (*make_slider_updates(coeffs), NO_CHANGE, NO_CHANGE,
           f"Baseline: {best_sim:.4f}", NO_CHANGE, NO_CHANGE,
           "\n".join(log))

    for pass_idx in range(max_passes):
        pass_start = best_sim

        for rank, i in enumerate(impact_order):
            name = comp_names.get(i, f"comp_{i}")
            current = coeffs[i].item()
            component_best_val = current
            component_best_sim = best_sim
            comp_steps = steps_for_component(i)

            for step in comp_steps:
                for sign in [1.0, -1.0]:
                    candidate = current + sign * step
                    candidate = max(-2.0, min(2.0, candidate))
                    if abs(candidate - current) < 1e-6:
                        continue

                    coeffs[i] = candidate
                    sim = score(coeffs)
                    eval_count += 1

                    improved = sim > component_best_sim
                    if improved:
                        component_best_sim = sim
                        component_best_val = candidate

                    marker = " +" if improved else ""
                    score_line = (
                        f"Pass {pass_idx+1}/{max_passes} | "
                        f"[{rank+1}/{n_active}] {name} = {candidate:+.4f} (step={step:.4g}) | "
                        f"sim = {sim:.4f}{marker} | "
                        f"best = {best_sim:.4f} | "
                        f"evals = {eval_count}"
                    )
                    yield (*make_slider_updates(coeffs), NO_CHANGE, NO_CHANGE,
                           score_line, NO_CHANGE, NO_CHANGE,
                           "\n".join(log))

                # Found improvement at this step size — refine then stop
                if component_best_val != current:
                    mid = (current + component_best_val) / 2.0
                    coeffs[i] = mid
                    mid_sim = score(coeffs)
                    eval_count += 1
                    if mid_sim > component_best_sim:
                        component_best_sim = mid_sim
                        component_best_val = mid

                    score_line = (
                        f"Pass {pass_idx+1}/{max_passes} | "
                        f"Refining {name} = {mid:+.4f} (bisect) | "
                        f"sim = {mid_sim:.4f} | "
                        f"best = {best_sim:.4f} | "
                        f"evals = {eval_count}"
                    )
                    yield (*make_slider_updates(coeffs), NO_CHANGE, NO_CHANGE,
                           score_line, NO_CHANGE, NO_CHANGE,
                           "\n".join(log))
                    break

            # Commit best value for this component
            coeffs[i] = component_best_val
            if component_best_val != current:
                best_sim = component_best_sim
                log.append(f"  {name}: {current:.4f} -> {component_best_val:.4f} (sim={best_sim:.4f})")
                yield (*make_slider_updates(coeffs), NO_CHANGE, NO_CHANGE,
                       f"Locked {name} = {component_best_val:+.4f} | best sim = {best_sim:.4f}",
                       NO_CHANGE, NO_CHANGE, "\n".join(log))

        improvement = best_sim - pass_start
        log.append(f"Pass {pass_idx+1}: sim={best_sim:.4f} (+{improvement:.4f})")

    # Final: generate audio, compute gaps, save .pt
    log.append(f"Done. Total evaluations: {eval_count}")
    scaled = coeffs * component_ranges
    p = (scaled.unsqueeze(0) @ all_components).squeeze(0)
    designed_voice = base_voice + p.reshape(analyzer.voice_shape)
    audio = speech_gen.generate_audio(text, designed_voice)

    current_features = fitness.extract_features(audio)
    gaps = compute_sorted_gaps(current_features, fitness.target_features)
    significant = [(l, g) for l, _, g in gaps if abs(g) > 5]
    gap_lines = [f"{l:<22} {g:+.1f}%" for l, g in significant[:10]]
    gap_text = "\n".join(gap_lines) if gap_lines else "All features within 5% of target."

    pt_path = str(OUTPUT_DIR / "designed_voice.pt")
    torch.save(designed_voice, pt_path)

    base_audio_np = _state.get("base_audio")
    base_sim = fitness.target_similarity(base_audio_np) if base_audio_np is not None else 0.0
    base_audio_val = (24000, base_audio_np) if base_audio_np is not None else NO_CHANGE

    yield (*make_slider_updates(coeffs), (24000, audio), base_audio_val,
           f"Target similarity: {best_sim:.4f} (base: {base_sim:.3f}, delta: {best_sim - base_sim:+.3f})",
           gap_text, pt_path,
           "\n".join(log))


def reset_sliders():
    """Reset all sliders to 0."""
    return [gr.update(value=0.0) for _ in range(MAX_SLIDERS)]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Voice Designer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Voice Designer\nCraft voices by exploring Kokoro's voice parameter space.")

        # --- Setup ---
        with gr.Accordion("Setup", open=True):
            with gr.Row():
                base_voice_input = gr.File(label="Base Voice (.pt)", file_types=[".pt"])
                target_audio_input = gr.Audio(label="Target Audio", type="filepath")
            with gr.Row():
                target_text_input = gr.Textbox(
                    label="Target Text",
                    placeholder="Enter the text to generate for comparison...",
                    lines=2,
                )
                target_text_file = gr.File(
                    label="Upload .txt file",
                    file_types=[".txt"],
                )
            with gr.Row():
                voice_folder_input = gr.Textbox(
                    label="Voice Folder",
                    value=str(VOICES_DIR),
                    info="Path to directory containing .pt voice files for PCA",
                )

        # --- Analysis ---
        with gr.Accordion("Analysis", open=True):
            with gr.Row():
                n_components_input = gr.Number(
                    label="Components", value=20, minimum=5, maximum=MAX_SLIDERS,
                    step=5, precision=0,
                    info="PCA dimensions to use. Start with 20, increase to add subtler controls. Re-click Analyze to expand (incremental).",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

            # Discovery cache status
            discovery_cache_info = get_discovery_cache_info(OUTPUT_DIR / "discovery_cache.pt")
            if discovery_cache_info:
                cache_status = f"Discovery cache loaded: {discovery_cache_info['count']} components | {discovery_cache_info['path']} | Updated {discovery_cache_info['timestamp']}"
            else:
                cache_status = "No discovery cache found. Run Discovery to create one."
            discovery_cache_status = gr.Textbox(
                label="Discovery Cache",
                value=cache_status,
                interactive=False,
                lines=2
            )

        # --- Discovery ---
        with gr.Accordion("Discovery", open=False):
            gr.Markdown(
                "Explore orthogonal directions beyond PCA to find impactful voice dimensions "
                "that standard analysis might miss. Results are cached as .pt tensors for reuse."
            )
            with gr.Row():
                discovery_file_input = gr.File(
                    label="Load Discovery Cache (.pt)",
                    file_types=[".pt"]
                )
                discovery_path_input = gr.Textbox(
                    label="Or enter cache path",
                    value=str(OUTPUT_DIR / "discovery_cache.pt"),
                    info="Path to discovery_cache.pt file"
                )
            with gr.Row():
                n_probes_input = gr.Number(
                    label="Number of probes", value=200, minimum=50, maximum=130000,
                    step=50, precision=0,
                    info="How many random directions to probe. More probes = better coverage but slower.",
                )
                discovery_btn = gr.Button("Run Discovery", variant="secondary")
            discovery_status = gr.Textbox(label="Discovery Status", interactive=False, lines=3)

        # --- Voice Parameters ---
        gr.Markdown("### Voice Parameters")
        sliders = []
        cols_per_row = 4
        sliders_per_col = MAX_SLIDERS // cols_per_row
        with gr.Row():
            for col in range(cols_per_row):
                with gr.Column():
                    for row in range(sliders_per_col):
                        idx = col * sliders_per_col + row
                        s = gr.Slider(
                            minimum=-2.0, maximum=2.0, step=0.05, value=0.0,
                            label=f"component_{idx}",
                            interactive=False,
                        )
                        sliders.append(s)

        # --- Tuning ---
        gr.Markdown("### Tuning")
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary", interactive=False)
            reset_btn = gr.Button("Reset Sliders", interactive=False)

        # --- Component Sources ---
        gr.Markdown("### Component Sources")
        with gr.Row():
            use_pca_checkbox = gr.Checkbox(label="Use PCA Components", value=True)
            use_discovery_checkbox = gr.Checkbox(label="Use Discovered Components", value=True)

        with gr.Row():
            auto_tune_btn = gr.Button("Auto-Tune", variant="secondary", interactive=False)
            passes_input = gr.Number(label="Passes", value=3, minimum=1, maximum=100, step=1, precision=0)
            step_size_input = gr.Number(label="Starting step", value=0.1, minimum=0.0001, maximum=2.0, step=0.001)
            mag_steps_input = gr.Number(label="Magnitude steps", value=3, minimum=1, maximum=10, step=1, precision=0,
                                        info="10x reductions (e.g. 3 = step, step/10, step/100)")

        # --- Results ---
        gr.Markdown("### Results")
        with gr.Row():
            audio_output = gr.Audio(label="Designed Voice", autoplay=True)
            base_audio_output = gr.Audio(label="Base Voice (unmodified)")
        with gr.Row():
            with gr.Column():
                score_output = gr.Textbox(label="Target Similarity", interactive=False)
                gap_output = gr.Textbox(label="Feature Gaps", interactive=False, lines=8)
            download_output = gr.File(label="Download Voice (.pt)")
        tune_log = gr.Textbox(label="Auto-Tune Log", interactive=False, lines=6, visible=True)

        # --- Wiring ---
        discovery_btn.click(
            fn=run_discovery,
            inputs=[voice_folder_input, n_probes_input],
            outputs=[discovery_status],
        )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[base_voice_input, target_audio_input, target_text_input, target_text_file, voice_folder_input, n_components_input, use_pca_checkbox, use_discovery_checkbox],
            outputs=[*sliders, status_text, generate_btn, reset_btn, auto_tune_btn],
        )

        generate_btn.click(
            fn=generate_voice,
            inputs=[*sliders, target_text_input],
            outputs=[audio_output, base_audio_output, score_output, gap_output, download_output],
        )

        reset_btn.click(
            fn=reset_sliders,
            inputs=[],
            outputs=sliders,
        )

        auto_tune_btn.click(
            fn=auto_tune,
            inputs=[*sliders, target_text_input, passes_input, step_size_input, mag_steps_input],
            outputs=[*sliders, audio_output, base_audio_output, score_output, gap_output, download_output, tune_log],
        )

    return demo


if __name__ == "__main__":
    ensure_voices()
    OUTPUT_DIR.mkdir(exist_ok=True)
    demo = build_ui()
    demo.launch()
