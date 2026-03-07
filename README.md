# Voice Designer

Interactive voice crafting tool built on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Uses PCA decomposition of voice tensors plus orthogonal discovery probing to find and control the dimensions that shape how a voice sounds.

## Full Workflow

The recommended end-to-end workflow, from raw voices to interactive tuning:

```bash
# 1. Install dependencies
uv sync
cd web && npm install && cd ..

# 2. Build the discovery catalog (PCA + discovered directions)
uv run python build_catalog.py

# 3. Label components via sensitivity analysis (one Kokoro synthesis per component)
uv run python label_components.py

# 4. Start the web UI
uv run uvicorn server:app --port 8000    # Terminal 1
cd web && npm run dev                     # Terminal 2
```

Open `http://localhost:5173`.

Steps 2 and 3 only need to be run once (or re-run when the catalog changes). The labels are saved to `catalog/component_labels.json` and loaded automatically by the server.

## Web UI (Recommended)

An interactive spider/radar chart interface for sculpting voices in real time with continuous audio playback and crossfade.

### Quick Start

```bash
uv sync

# Terminal 1 — start the backend API
uv run uvicorn server:app --port 8000

# Terminal 2 — start the frontend dev server
cd web && npm install && npm run dev
```

Open `http://localhost:5173`.

### How to Use

1. **Pick a base voice** from the dropdown (voices from the `voices/` directory)
2. **Import a custom voice** by clicking the **+** button next to the dropdown and uploading a `.pt` file
3. **Type text** that the voice will speak
4. **Press play** to start audio looping
5. **Drag points** on the spider chart to adjust voice dimensions (pitch, brightness, nasality, etc.)
6. Changes are debounced and synthesized automatically — new audio crossfades in mid-sentence

Each axis on the chart represents a voice component discovered via PCA or orthogonal probing. The center of each axis is -1.0, the outer edge is +1.0, and the dashed middle ring is neutral (0.0).

### Voice Import

Click the **+** button next to the voice selector dropdown to upload a custom `.pt` voice file. The file is validated, saved to the `voices/` directory, and automatically selected. The voice list refreshes to include the new file.

### Architecture

- **Backend**: FastAPI (`server.py`) serving synthesis API, catalog data, voice listings, and voice upload
- **Frontend**: React + TypeScript + Vite (`web/`) with Canvas-based spider chart and Web Audio API crossfade

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/voices` | GET | List available .pt voice files |
| `/api/catalog` | GET | Component names and count |
| `/api/synthesize` | POST | Synthesize WAV from voice + coefficients + text |
| `/api/upload-voice` | POST | Upload a .pt voice file to voices/ |

## Gradio UI (Legacy)

```bash
uv sync
uv run python app.py
```

Open `http://localhost:7860`.

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

## Auto Mode

Headless loop that runs continuously without the UI — discovery, analysis, auto-tune, save, repeat:

```bash
# With a pre-built catalog (recommended — skips PCA recomputation each iteration):
uv run python auto_mode.py \
    --base-voice voices/af_heart.pt \
    --target-audio my_target.wav \
    --target-text "Hello, my name is Alex." \
    --catalog catalog/discovery_catalog.pt \
    --n-probes 50 --n-pca 32 --n-discovery 64 \
    --passes 3

# Without catalog (computes PCA fresh each iteration):
uv run python auto_mode.py \
    --base-voice voices/af_heart.pt \
    --target-audio my_target.wav \
    --target-text "Hello, my name is Alex." \
    --n-probes 200 --n-pca 32 --n-discovery 32 \
    --passes 3
```

If `catalog/discovery_catalog.pt` exists it is loaded automatically — no `--catalog` flag needed.

Each iteration:
1. Runs `--n-probes` discovery probes (accumulates into `output/discovery_cache.pt`)
2. Loads PCA from catalog (or computes fresh) + top `--n-discovery` discovery components
3. Runs sensitivity analysis on all active components
4. Runs coordinate descent auto-tune (`--passes` passes)
5. Saves the tuned voice as `output/auto_voice_iter####_<timestamp>_sim<score>.pt` and overwrites `output/designed_voice.pt`
6. Repeats using the tuned voice as the new base

Stop at any time with **Ctrl+C**. The best result is always in `output/designed_voice.pt`.

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--catalog` | auto-detected | Path to discovery catalog (skips PCA recomputation) |
| `--n-probes` | 200 | Discovery probes per iteration |
| `--n-pca` | 32 | PCA components to include |
| `--n-discovery` | 32 | Top discovered components to include |
| `--passes` | 3 | Auto-tune coordinate descent passes |
| `--start-step` | 0.1 | Starting step size (scaled per-component by sensitivity) |
| `--mag-steps` | 3 | Magnitude refinement steps (each is 10× smaller) |
| `--keep-base` | off | Always start from original base voice instead of rolling forward |

## Discovery Catalog

The `catalog/` directory holds a version-controlled distillation of the best discovered voice directions, combined with pre-computed PCA. Using a catalog:
- Eliminates repeated PCA computation across iterations
- Packages the highest-value directions for sharing and reuse
- Lets anyone clone the repo and skip straight to tuning without running discovery

### Building the catalog

After accumulating discoveries in `output/discovery_cache.pt`:

```bash
# Build catalog with top 1000 discovered directions
uv run python build_catalog.py --n-discoveries 1000

# Preview size without writing anything
uv run python build_catalog.py --dry-run

# Write to a custom path
uv run python build_catalog.py --n-discoveries 500 --output catalog/v2.pt
```

Commit `catalog/discovery_catalog.pt` to version control. Use git-lfs for large catalogs.

### Labeling components

After building a catalog, run sensitivity analysis to assign human-readable labels (pitch, brightness, nasality, etc.) to each component:

```bash
# Label all components in the default catalog
uv run python label_components.py

# Use a specific base voice and text
uv run python label_components.py --voice voices/af_heart.pt --text "Hello there."

# Write labels to a custom path
uv run python label_components.py --output catalog/my_labels.json
```

This generates one Kokoro synthesis per component, so it takes a few minutes for large catalogs. Results are saved to `catalog/component_labels.json` as a simple JSON list. The web UI server loads these labels automatically on startup, replacing the default fallback names.

### Pruning the working cache

The working discovery cache (`output/discovery_cache.pt`) grows unboundedly. Prune it periodically to keep the top-N directions:

```bash
# Prune in-place to top 1000
uv run python prune_cache.py --keep 1000

# Preview what would be removed
uv run python prune_cache.py --dry-run
```

## Test Synthesis

Quickly generate audio from any `.pt` voice file:

```bash
# Synthesize with the current designed voice (output/designed_voice.pt)
uv run python synthesize.py

# Custom text
uv run python synthesize.py --text "Hello, this is a test."

# Specific voice file
uv run python synthesize.py --voice output/auto_voice_iter0012_*.pt

# Custom output path
uv run python synthesize.py --output /tmp/preview.wav
```

Output is saved to `output/test_synthesis.wav` by default.

## Project Structure

```
voice-designer/
├── server.py               # FastAPI backend for the web UI
├── web/                    # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/     # SpiderChart, PlaybackControls, VoiceSelector, TextInput
│   │   ├── hooks/          # useAudioEngine, useVoiceSynthesis
│   │   ├── api.ts          # API client
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts
├── app.py                  # Gradio UI + orchestration (legacy)
├── auto_mode.py            # Headless continuous refinement loop
├── synthesize.py           # Quick test synthesis from any .pt voice
├── build_catalog.py        # Distill PCA + top discoveries into a catalog
├── label_components.py     # Label components via sensitivity analysis
├── prune_cache.py          # Trim working discovery cache to top-N
├── catalog/                # Version-controlled discovery catalog
│   ├── discovery_catalog.pt
│   └── component_labels.json  # Human-readable labels for each component
├── core/
│   ├── voice_analyzer.py   # PCA decomposition + sensitivity
│   ├── speech_generator.py # Kokoro TTS wrapper
│   ├── fitness_scorer.py   # Audio feature extraction + similarity
│   └── discovery.py        # Orthogonal probing beyond PCA
├── voices/                 # Kokoro .pt files (auto-downloaded)
└── output/                 # Designed voices + working discovery cache (gitignored)
```

## Requirements

- Python 3.11-3.12
- ~2GB disk for dependencies + model weights
- macOS (MPS), Linux (CUDA), or CPU

## Tips

- Start with a base voice close to your target for faster convergence
- Use 20 components for coarse shaping, expand higher values for fine detail
- Run Discovery to accumulate a cache, then `build_catalog.py` to distill it — the catalog grows better over time
- When Auto-Tune plateaus, try reducing step size, increasing magnitude steps, or rebuilding the catalog with more discoveries
