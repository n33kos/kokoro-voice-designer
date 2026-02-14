# Voice Designer

Interactive voice crafting tool built on PCA decomposition of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) voice tensors. Replaces the manual JSON-edit-and-rerun workflow with a Gradio web UI featuring real-time sliders, automated optimization, and live audio feedback.

## Quick Start

```bash
uv sync
uv run python app.py
```

On first launch, all 54 Kokoro voice `.pt` files are downloaded from HuggingFace automatically.

Open the URL shown in the terminal (usually `http://localhost:7860`).

## How It Works

Kokoro voice tensors live in a ~130K-dimensional space. Most of that is noise. PCA extracts the 20-40 principal directions that actually distinguish voices, and sensitivity analysis maps each direction to an audio feature (pitch, brightness, nasality, etc.).

You then adjust named sliders to sculpt a voice targeting a reference audio sample, with similarity scoring as feedback.

## Workflow

### 1. Setup
- **Base Voice**: Upload a `.pt` voice tensor as your starting point (any Kokoro voice from `voices/`)
- **Target Audio**: Upload a `.wav` of the voice you want to match
- **Target Text**: Enter text for Kokoro to generate (used for comparison)
- **Components**: Number of PCA dimensions (start with 20, increase for finer control)

### 2. Analyze
Click **Analyze** to run PCA decomposition and sensitivity analysis. This takes ~2 minutes for 20 components (one Kokoro inference per component). Results are cached — increasing components later only analyzes the new ones.

### 3. Manual Tuning
Each slider controls one PCA direction, labeled by its dominant audio effect (pitch, brightness, timbre, etc.) with variance percentage. Range is [-2, 2] where 1.0 = full observed voice range.

Click **Generate** to hear the result and see the target similarity score + feature gaps.

### 4. Auto-Tune
Automated coordinate descent that does the manual binary search process for you:

- **Passes**: Number of full sweeps across all components
- **Starting step**: Initial probe size (e.g., 0.1)
- **Magnitude steps**: How many 10x reductions to try (e.g., 3 = tries 0.1, 0.01, 0.001)

The UI updates live — sliders move to show what's being tested, and the similarity score updates on every evaluation.

### 5. Export
Download the designed `.pt` file from the Results section. Use it as `--starting_voice` in [kvoicewalk](https://github.com/RobertAgee/kvoicewalk) for further automated refinement.

## Project Structure

```
voice-designer/
├── app.py                  # Gradio UI + all orchestration
├── pyproject.toml          # uv-managed dependencies
├── core/
│   ├── voice_analyzer.py   # PCA decomposition + sensitivity analysis
│   ├── speech_generator.py # Kokoro TTS wrapper
│   └── fitness_scorer.py   # Audio feature extraction + similarity scoring
├── voices/                 # Kokoro .pt files (auto-downloaded, gitignored)
└── output/                 # Designed voices + audio (gitignored)
```

## Requirements

- Python 3.11-3.12
- ~2GB disk for dependencies + model weights
- macOS (MPS), Linux (CUDA), or CPU

## Tips

- Start with a base voice that's already somewhat close to your target — the closer the starting point, the faster Auto-Tune converges
- Use 20 components for coarse shaping, then expand to 30-40 for fine detail
- When Auto-Tune plateaus, reduce the starting step size (e.g., 0.01) and increase magnitude steps
- The designed `.pt` works as a starting point for kvoicewalk's random walk, combining manual design with automated exploration
