#!/usr/bin/env python3
"""Voice Designer — interactive PCA-based voice crafting with Gradio UI."""

import os
import tempfile

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

from core import VoiceAnalyzer, SpeechGenerator, FitnessScorer

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

NUM_SLIDERS = 20

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


def name_components(analyzer: VoiceAnalyzer) -> dict[int, str]:
    """Assign human-readable names to PCA components based on dominant feature effect."""
    if analyzer.sensitivity is None:
        return {i: f"component_{i}" for i in range(analyzer.n_components)}

    used_names: set[str] = set()
    names: dict[int, str] = {}

    for i in range(analyzer.n_components):
        deltas = analyzer.sensitivity[i]
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


# ---------------------------------------------------------------------------
# Backend functions
# ---------------------------------------------------------------------------

def run_analysis(base_voice_file, target_audio, target_text, voice_folder, progress=gr.Progress()):
    """Load voices, build PCA, run sensitivity analysis, name components."""
    if base_voice_file is None:
        raise gr.Error("Please upload a base voice .pt file.")
    if target_audio is None:
        raise gr.Error("Please upload a target audio file.")
    if not target_text or not target_text.strip():
        raise gr.Error("Please enter target text.")
    if not voice_folder or not os.path.isdir(voice_folder):
        raise gr.Error(f"Voice folder not found: {voice_folder}")

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

    # Initialize speech generator and fitness scorer
    progress(0.1, desc="Initializing Kokoro pipeline...")
    speech_gen = SpeechGenerator()

    progress(0.15, desc="Converting target audio...")
    target_wav_path = convert_audio(target_audio)
    fitness = FitnessScorer(target_wav_path, device=speech_gen.device)

    # PCA decomposition
    progress(0.2, desc="Running PCA decomposition...")
    analyzer = VoiceAnalyzer(voice_tensors)

    # Compute component ranges for intuitive scaling
    n_voices = len(voice_tensors)
    flat = torch.stack(voice_tensors).reshape(n_voices, -1).float()
    centered = flat - analyzer.mean
    projections = centered @ analyzer.components.T
    component_ranges = projections.max(dim=0).values - projections.min(dim=0).values

    # Run sensitivity analysis with progress updates
    progress(0.25, desc="Sensitivity analysis (this takes ~2 min)...")
    n = analyzer.n_components
    base_flat = base_voice.reshape(-1).float()
    base_audio = speech_gen.generate_audio(target_text, base_voice)
    base_features = fitness.extract_features(base_audio)

    sensitivity = {}
    for i in range(n):
        progress(0.25 + 0.7 * (i / n), desc=f"Analyzing component {i+1}/{n}...")
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
        sensitivity[i] = deltas

    analyzer.sensitivity = sensitivity

    # Name components
    progress(0.95, desc="Naming components...")
    comp_names = name_components(analyzer)

    # Store everything in module-level state
    _state["analyzer"] = analyzer
    _state["base_voice"] = base_voice
    _state["speech_gen"] = speech_gen
    _state["fitness"] = fitness
    _state["component_names"] = comp_names
    _state["component_ranges"] = component_ranges
    _state["target_text"] = target_text

    progress(1.0, desc="Done!")

    # Build slider updates: set label and enable each slider
    slider_updates = []
    for i in range(NUM_SLIDERS):
        if i < analyzer.n_components:
            label = comp_names.get(i, f"component_{i}")
            var_pct = float(analyzer.per_component_variance[i]) * 100
            slider_updates.append(gr.update(
                label=f"{label} ({var_pct:.1f}%)",
                interactive=True,
                value=0.0,
            ))
        else:
            slider_updates.append(gr.update(
                label=f"(unused)",
                interactive=False,
                value=0.0,
            ))

    pca_info = (
        f"PCA: {analyzer.n_components} components explain "
        f"{analyzer.explained_variance_ratio:.1%} of variation across "
        f"{len(voice_tensors)} voices."
    )

    return (*slider_updates, pca_info, gr.update(interactive=True), gr.update(interactive=True))


def generate_voice(*args):
    """Build voice from slider values and generate audio."""
    # Last arg is target_text override
    slider_values = args[:NUM_SLIDERS]
    target_text = args[NUM_SLIDERS]

    if "analyzer" not in _state:
        raise gr.Error("Run analysis first!")

    analyzer = _state["analyzer"]
    base_voice = _state["base_voice"]
    speech_gen = _state["speech_gen"]
    fitness = _state["fitness"]
    component_ranges = _state["component_ranges"]

    text = target_text.strip() if target_text and target_text.strip() else _state.get("target_text", "Hello world.")

    # Build coefficient vector from slider values
    coeffs = torch.zeros(analyzer.n_components)
    for i in range(analyzer.n_components):
        coeffs[i] = float(slider_values[i])

    # Apply PCA perturbation: coeffs * component_ranges @ components
    scaled = coeffs * component_ranges
    perturbation_flat = (scaled.unsqueeze(0) @ analyzer.components).squeeze(0)
    designed_voice = base_voice + perturbation_flat.reshape(analyzer.voice_shape)

    # Generate audio
    audio = speech_gen.generate_audio(text, designed_voice)
    target_sim = fitness.target_similarity(audio)

    # Feature gaps
    current_features = fitness.extract_features(audio)
    gaps = compute_sorted_gaps(current_features, fitness.target_features)
    significant = [(l, g) for l, _, g in gaps if abs(g) > 5]

    score_text = f"Target similarity: {target_sim:.3f}"

    gap_lines = []
    for label, gap_pct in significant[:10]:
        gap_lines.append(f"{label:<22} {gap_pct:+.1f}%")
    gap_text = "\n".join(gap_lines) if gap_lines else "All features within 5% of target."

    # Save .pt for download
    pt_path = os.path.join(tempfile.gettempdir(), "designed_voice.pt")
    torch.save(designed_voice, pt_path)

    return (24000, audio), score_text, gap_text, pt_path


def reset_sliders():
    """Reset all sliders to 0."""
    return [gr.update(value=0.0) for _ in range(NUM_SLIDERS)]


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
                    value=os.path.expanduser("~/kvoicewalk/voices"),
                )
            analyze_btn = gr.Button("Analyze", variant="primary")

        status_text = gr.Textbox(label="Status", interactive=False)

        # --- Sliders ---
        gr.Markdown("### Voice Parameters")
        sliders = []
        with gr.Row():
            with gr.Column():
                for i in range(10):
                    s = gr.Slider(
                        minimum=-2.0, maximum=2.0, step=0.05, value=0.0,
                        label=f"component_{i}",
                        interactive=False,
                    )
                    sliders.append(s)
            with gr.Column():
                for i in range(10, NUM_SLIDERS):
                    s = gr.Slider(
                        minimum=-2.0, maximum=2.0, step=0.05, value=0.0,
                        label=f"component_{i}",
                        interactive=False,
                    )
                    sliders.append(s)

        # --- Action buttons ---
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary", interactive=False)
            reset_btn = gr.Button("Reset Sliders", interactive=False)

        # --- Results ---
        gr.Markdown("### Results")
        with gr.Row():
            audio_output = gr.Audio(label="Generated Audio", autoplay=True)
            with gr.Column():
                score_output = gr.Textbox(label="Target Similarity", interactive=False)
                gap_output = gr.Textbox(label="Feature Gaps", interactive=False, lines=8)
                download_output = gr.File(label="Download Voice (.pt)")

        # --- Wiring ---
        analyze_btn.click(
            fn=run_analysis,
            inputs=[base_voice_input, target_audio_input, target_text_input, voice_folder_input],
            outputs=[*sliders, status_text, generate_btn, reset_btn],
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

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
