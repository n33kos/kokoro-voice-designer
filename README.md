# Voice Designer

Interactive voice crafting tool built on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Uses PCA decomposition of voice tensors plus orthogonal discovery probing to find and control the dimensions that shape how a voice sounds.

## Quick Start

```bash
uv sync
uv run python app.py
```

Voices are downloaded automatically on first launch. Open `http://localhost:7860`.

## What It Does

- **PCA Analysis**: Extracts the principal directions that distinguish Kokoro's 54 built-in voices, then maps each direction to an audio feature (pitch, brightness, nasality, etc.)
- **Discovery Mode**: Probes random directions orthogonal to PCA to find impactful voice dimensions that don't exist in the sample voices — expands what's possible beyond the original voice library
- **Manual Tuning**: Named sliders let you sculpt a voice targeting a reference audio sample, with real-time similarity scoring
- **Auto-Tune**: Automated coordinate descent that optimizes slider positions to maximize similarity to your target voice

## Workflow

1. **Setup** — Upload a base voice `.pt`, a target audio `.wav`, and enter the text for comparison
2. **Analyze** — Runs PCA and sensitivity analysis (~2 min for 20 components). Results are cached incrementally
3. **Discovery** (optional) — Probes orthogonal directions beyond PCA to find additional impactful dimensions. Takes ~10-20 min depending on probe count. Results accumulate across runs
4. **Tune** — Adjust sliders manually or use Auto-Tune to optimize toward your target
5. **Export** — Download the designed `.pt` voice file

## Discovery Mode

Standard PCA only captures variance between the 54 sample voices. Discovery explores the full ~130K-dimensional voice space to find directions PCA misses:

- Generates random directions orthogonal to all known components (PCA + prior discoveries)
- Tests each direction by perturbing the voice and measuring audio feature changes
- Keeps directions with significant impact, discards the rest
- Cache accumulates across runs — each session finds new directions

## Project Structure

```
voice-designer/
├── app.py                  # Gradio UI + orchestration
├── core/
│   ├── voice_analyzer.py   # PCA decomposition + sensitivity
│   ├── speech_generator.py # Kokoro TTS wrapper
│   ├── fitness_scorer.py   # Audio feature extraction + similarity
│   └── discovery.py        # Orthogonal probing beyond PCA
├── voices/                 # Kokoro .pt files (auto-downloaded)
└── output/                 # Designed voices + discovery cache
```

## Requirements

- Python 3.11-3.12
- ~2GB disk for dependencies + model weights
- macOS (MPS), Linux (CUDA), or CPU

## Tips

- Start with a base voice close to your target for faster convergence
- Use 20 components for coarse shaping, expand higher values for fine detail
- Run Discovery once to build a cache, then re-run with more probes to expand coverage. This file grows over time and captures the most impactful directions in voice space through cumulative discovery.
- When Auto-Tune plateaus, try reducing step size or increasing magnitude steps
