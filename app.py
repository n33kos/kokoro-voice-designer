#!/usr/bin/env python3
"""Voice Designer — interactive PCA-based voice crafting with Gradio UI."""

import os
import tempfile
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from core import VoiceAnalyzer, SpeechGenerator, FitnessScorer

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

MAX_SLIDERS = 40

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


def name_components(sensitivity: dict, n_components: int) -> dict[int, str]:
    """Assign human-readable names to PCA components based on dominant feature effect."""
    used_names: set[str] = set()
    names: dict[int, str] = {}

    for i in range(n_components):
        if i not in sensitivity:
            names[i] = f"component_{i}"
            continue

        deltas = sensitivity[i]
        sorted_features = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)

        chosen_name = f"component_{i}"
        for feat_name, _ in sorted_features:
            if feat_name in FEATURE_LABELS:
                base_name = FEATURE_LABELS[feat_name]
                if base_name not in used_names:
                    chosen_name = base_name
                    used_names.add(base_name)
                    break

        names[i] = chosen_name

    return names


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


def build_slider_updates(n_active, comp_names, per_component_variance, preserve_values=None):
    """Build gr.update list for all MAX_SLIDERS slots."""
    updates = []
    for i in range(MAX_SLIDERS):
        if i < n_active:
            label = comp_names.get(i, f"component_{i}")
            var_pct = float(per_component_variance[i]) * 100 if i < len(per_component_variance) else 0
            val = preserve_values[i] if preserve_values is not None else 0.0
            updates.append(gr.update(label=f"{label} ({var_pct:.1f}%)", interactive=True, value=val))
        else:
            updates.append(gr.update(label="(unused)", interactive=False, value=0.0))
    return updates


# ---------------------------------------------------------------------------
# Backend functions
# ---------------------------------------------------------------------------

def run_analysis(base_voice_file, target_audio, target_text, voice_folder, n_components, progress=gr.Progress()):
    """Load voices, build PCA, run sensitivity analysis.

    Incremental: if state already exists with sensitivity data, only analyzes new components.
    """
    if base_voice_file is None:
        raise gr.Error("Please upload a base voice .pt file.")
    if target_audio is None:
        raise gr.Error("Please upload a target audio file.")
    if not target_text or not target_text.strip():
        raise gr.Error("Please enter target text.")
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

    # Compute component ranges for intuitive scaling
    n_voices = len(voice_tensors)
    flat = torch.stack(voice_tensors).reshape(n_voices, -1).float()
    centered = flat - analyzer.mean
    projections = centered @ analyzer.components.T
    component_ranges = projections.max(dim=0).values - projections.min(dim=0).values

    # Determine which components still need sensitivity analysis
    components_to_analyze = [i for i in range(n_active) if i not in existing_sensitivity]
    total_to_analyze = len(components_to_analyze)

    if total_to_analyze > 0:
        progress(0.25, desc=f"Sensitivity analysis on {total_to_analyze} components...")
        base_flat = base_voice.reshape(-1).float()
        base_audio = speech_gen.generate_audio(target_text, base_voice)
        base_features = fitness.extract_features(base_audio)

        for idx, i in enumerate(components_to_analyze):
            progress(0.25 + 0.7 * (idx / total_to_analyze),
                     desc=f"Analyzing component {i+1} ({idx+1}/{total_to_analyze} new)...")
            component = analyzer.components[i]
            scale = float(analyzer.singular_values[i]) * 0.1

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
        progress(0.95, desc="All components already analyzed, reusing...")

    analyzer.sensitivity = existing_sensitivity

    # Name components
    progress(0.95, desc="Naming components...")
    comp_names = name_components(existing_sensitivity, n_active)

    # Store everything in module-level state
    _state["analyzer"] = analyzer
    _state["base_voice"] = base_voice
    _state["speech_gen"] = speech_gen
    _state["fitness"] = fitness
    _state["component_names"] = comp_names
    _state["component_ranges"] = component_ranges
    _state["target_text"] = target_text
    _state["sensitivity"] = existing_sensitivity
    _state["n_active"] = n_active

    progress(1.0, desc="Done!")

    slider_updates = build_slider_updates(n_active, comp_names, analyzer.per_component_variance)

    new_count = total_to_analyze
    reused_count = n_active - total_to_analyze
    pca_info = (
        f"PCA: {n_active} active / {analyzer.n_components} total components, "
        f"{analyzer.explained_variance_ratio:.1%} total variance across {len(voice_tensors)} voices. "
        f"({new_count} newly analyzed, {reused_count} reused)"
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
    n_active = _state["n_active"]
    base_voice = _state["base_voice"]
    speech_gen = _state["speech_gen"]
    fitness = _state["fitness"]
    component_ranges = _state["component_ranges"]

    text = target_text.strip() if target_text and target_text.strip() else _state.get("target_text", "Hello world.")

    # Build coefficient vector — only use active components
    coeffs = torch.zeros(analyzer.n_components)
    for i in range(n_active):
        coeffs[i] = float(slider_values[i])

    # Apply PCA perturbation
    scaled = coeffs * component_ranges
    perturbation_flat = (scaled.unsqueeze(0) @ analyzer.components).squeeze(0)
    designed_voice = base_voice + perturbation_flat.reshape(analyzer.voice_shape)

    audio = speech_gen.generate_audio(text, designed_voice)
    target_sim = fitness.target_similarity(audio)

    current_features = fitness.extract_features(audio)
    gaps = compute_sorted_gaps(current_features, fitness.target_features)
    significant = [(l, g) for l, _, g in gaps if abs(g) > 5]

    score_text = f"Target similarity: {target_sim:.3f}"
    gap_lines = [f"{label:<22} {gap_pct:+.1f}%" for label, gap_pct in significant[:10]]
    gap_text = "\n".join(gap_lines) if gap_lines else "All features within 5% of target."

    pt_path = str(OUTPUT_DIR / "designed_voice.pt")
    torch.save(designed_voice, pt_path)

    return (24000, audio), score_text, gap_text, pt_path


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
    n_active = _state["n_active"]
    base_voice = _state["base_voice"]
    speech_gen = _state["speech_gen"]
    fitness = _state["fitness"]
    component_ranges = _state["component_ranges"]
    comp_names = _state["component_names"]
    text = (target_text_val.strip() if target_text_val and target_text_val.strip()
            else _state.get("target_text", "Hello world."))

    NO_CHANGE = gr.update()

    def score(coeffs):
        scaled = coeffs * component_ranges
        p = (scaled.unsqueeze(0) @ analyzer.components).squeeze(0)
        voice = base_voice + p.reshape(analyzer.voice_shape)
        audio = speech_gen.generate_audio(text, voice)
        return float(fitness.target_similarity(audio))

    def make_slider_updates(coeffs):
        return [gr.update(value=round(coeffs[j].item(), 3)) if j < n_active
                else gr.update() for j in range(MAX_SLIDERS)]

    # Start from current slider positions
    coeffs = torch.zeros(analyzer.n_components)
    for i in range(n_active):
        coeffs[i] = float(slider_values[i])

    best_sim = score(coeffs)
    eval_count = 1
    log = [f"Baseline similarity: {best_sim:.4f}", f"Active components: {n_active}"]

    # Build step sizes: start_step, start_step/10, start_step/100, ...
    STEPS = [start_step / (10 ** m) for m in range(mag_steps)]
    log.append(f"Step sizes: {', '.join(f'{s:.4g}' for s in STEPS)}")

    yield (*make_slider_updates(coeffs), NO_CHANGE,
           f"Baseline: {best_sim:.4f}", NO_CHANGE, NO_CHANGE,
           "\n".join(log))

    for pass_idx in range(max_passes):
        pass_start = best_sim

        for i in range(n_active):
            name = comp_names.get(i, f"comp_{i}")
            current = coeffs[i].item()
            component_best_val = current
            component_best_sim = best_sim

            for step in STEPS:
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
                        f"Testing {name} = {candidate:+.4f} (step={step:.4g}) | "
                        f"sim = {sim:.4f}{marker} | "
                        f"best = {best_sim:.4f} | "
                        f"evals = {eval_count}"
                    )
                    yield (*make_slider_updates(coeffs), NO_CHANGE,
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
                    yield (*make_slider_updates(coeffs), NO_CHANGE,
                           score_line, NO_CHANGE, NO_CHANGE,
                           "\n".join(log))
                    break

            # Commit best value for this component
            coeffs[i] = component_best_val
            if component_best_val != current:
                best_sim = component_best_sim
                log.append(f"  {name}: {current:.4f} -> {component_best_val:.4f} (sim={best_sim:.4f})")
                yield (*make_slider_updates(coeffs), NO_CHANGE,
                       f"Locked {name} = {component_best_val:+.4f} | best sim = {best_sim:.4f}",
                       NO_CHANGE, NO_CHANGE, "\n".join(log))

        improvement = best_sim - pass_start
        log.append(f"Pass {pass_idx+1}: sim={best_sim:.4f} (+{improvement:.4f})")

        if improvement < 0.0005:
            log.append(f"Converged after {pass_idx+1} passes.")
            break

    # Final: generate audio, compute gaps, save .pt
    log.append(f"Done. Total evaluations: {eval_count}")
    scaled = coeffs * component_ranges
    p = (scaled.unsqueeze(0) @ analyzer.components).squeeze(0)
    designed_voice = base_voice + p.reshape(analyzer.voice_shape)
    audio = speech_gen.generate_audio(text, designed_voice)

    current_features = fitness.extract_features(audio)
    gaps = compute_sorted_gaps(current_features, fitness.target_features)
    significant = [(l, g) for l, _, g in gaps if abs(g) > 5]
    gap_lines = [f"{l:<22} {g:+.1f}%" for l, g in significant[:10]]
    gap_text = "\n".join(gap_lines) if gap_lines else "All features within 5% of target."

    pt_path = str(OUTPUT_DIR / "designed_voice.pt")
    torch.save(designed_voice, pt_path)

    yield (*make_slider_updates(coeffs), (24000, audio),
           f"Target similarity: {best_sim:.4f}", gap_text, pt_path,
           "\n".join(log))


def reset_sliders():
    """Reset all sliders to 0."""
    return [gr.update(value=0.0) for _ in range(MAX_SLIDERS)]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Voice Designer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Voice Designer\nCraft voices by adjusting PCA-derived parameters with real-time feedback.")

        # --- Setup section ---
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
                voice_folder_input = gr.Textbox(
                    label="Voice Folder (path to .pt files for PCA)",
                    value=str(VOICES_DIR),
                )
            with gr.Row():
                n_components_input = gr.Number(
                    label="Components", value=20, minimum=5, maximum=MAX_SLIDERS,
                    step=5, precision=0,
                    info="PCA dimensions to use. Start with 20, increase to add subtler controls. Re-click Analyze to expand (incremental).",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

        status_text = gr.Textbox(label="Status", interactive=False)

        # --- Sliders: 4 columns of 10 ---
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

        # --- Action buttons ---
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary", interactive=False)
            reset_btn = gr.Button("Reset Sliders", interactive=False)
        with gr.Row():
            auto_tune_btn = gr.Button("Auto-Tune", variant="secondary", interactive=False)
            passes_input = gr.Number(label="Passes", value=3, minimum=1, maximum=10, step=1, precision=0)
            step_size_input = gr.Number(label="Starting step", value=0.1, minimum=0.001, maximum=2.0, step=0.01)
            mag_steps_input = gr.Number(label="Magnitude steps", value=3, minimum=1, maximum=5, step=1, precision=0,
                                        info="10x reductions (e.g. 3 = step, step/10, step/100)")

        # --- Results ---
        gr.Markdown("### Results")
        with gr.Row():
            audio_output = gr.Audio(label="Generated Audio", autoplay=True)
            with gr.Column():
                score_output = gr.Textbox(label="Target Similarity", interactive=False)
                gap_output = gr.Textbox(label="Feature Gaps", interactive=False, lines=8)
                download_output = gr.File(label="Download Voice (.pt)")
        tune_log = gr.Textbox(label="Auto-Tune Log", interactive=False, lines=6, visible=True)

        # --- Wiring ---
        analyze_btn.click(
            fn=run_analysis,
            inputs=[base_voice_input, target_audio_input, target_text_input, voice_folder_input, n_components_input],
            outputs=[*sliders, status_text, generate_btn, reset_btn, auto_tune_btn],
        )

        generate_btn.click(
            fn=generate_voice,
            inputs=[*sliders, target_text_input],
            outputs=[audio_output, score_output, gap_output, download_output],
        )

        reset_btn.click(
            fn=reset_sliders,
            inputs=[],
            outputs=sliders,
        )

        auto_tune_btn.click(
            fn=auto_tune,
            inputs=[*sliders, target_text_input, passes_input, step_size_input, mag_steps_input],
            outputs=[*sliders, audio_output, score_output, gap_output, download_output, tune_log],
        )

    return demo


if __name__ == "__main__":
    ensure_voices()
    OUTPUT_DIR.mkdir(exist_ok=True)
    demo = build_ui()
    demo.launch()
