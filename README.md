# Voice Designer

Interactive voice crafting tool built on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Uses PCA decomposition of voice tensors plus orthogonal discovery probing to find and control the dimensions that shape how a voice sounds.

## The Discovery Catalog and Git LFS

`catalog/discovery_catalog.pt` is ~530MB — well over GitHub's 100MB per-file limit — so it is stored with [Git LFS](https://git-lfs.com).

**If you have Git LFS**, `git clone` fetches it automatically. Otherwise install it and pull:

```bash
brew install git-lfs   # or: apt install git-lfs
git lfs install
git lfs pull
```

**If you don't want Git LFS**, the repository still clones and the code still runs — you'll get a small text pointer file instead of the catalog, and you can regenerate the real thing yourself (below). Nothing is broken; you just start without a pre-built catalog.

### Regenerating the catalog from scratch

The catalog is built from two inputs: the voice library in `voices/`, and a discovery cache accumulated by probing.

```bash
# 1. Accumulate discoveries. This is the slow part — the cache grows across
#    runs, so stop with Ctrl+C whenever you have enough and rerun later to add
#    more. Expect hours for a cache comparable to the shipped catalog.
uv run python auto_mode.py \
  --base-voice voices/af_heart.pt \
  --target-audio input/your_sample.wav \
  --target-text "transcript of your sample" \
  --n-probes 200

# 2. Distill the cache into a catalog (fast — seconds to minutes)
uv run python build_catalog.py --n-discoveries 1000

# 3. Rebuild the derived maps
uv run python build_style_map.py
uv run python label_components.py     # optional: names for raw components
```

Be aware of the real cost: step 1 writes `output/discovery_cache.pt`, which is gitignored and around 1GB for a mature cache, and it is genuinely slow — the shipped catalog represents many hours of probing. Step 2 onward is quick. If you only want the interactive designer and the style sliders, you do not need the catalog at all; `build_style_map.py` works directly from Kokoro and a base voice.

## Getting Started

The catalog and component labels are version-controlled, so you can clone and run immediately:

```bash
# Install dependencies
uv sync
cd web && npm install && cd ..

# Terminal 1 — start the backend API
uv run uvicorn server:app --reload --port 8000

# Terminal 2 — start the frontend dev server
cd web && npm run dev
```

Open `http://localhost:5173`.

## How It Works

Kokoro-82M represents each voice as a **[510, 1, 256] tensor** — 130,560 dimensions. This tool finds and controls the directions in that space that actually matter.

- **PCA** on Kokoro's 54 built-in voices identifies the axes of greatest variation — the most obvious differences between voices (pitch, timbre, resonance, etc.). This gives ~53 meaningful directions, but only within the span of the existing voice library.
- **Discovery** probes random directions orthogonal to PCA and measures their effect on audio features. This finds impactful dimensions the voice library doesn't vary along — subtler characteristics like breathiness or texture that no pair of built-in voices differs on.
- **Catalog** bundles the top PCA components and highest-impact discoveries into a single reusable file, so downstream tools skip recomputation.
- **Auto mode** runs coordinate descent optimization over these dimensions, iteratively tuning a voice to maximize Resemblyzer embedding similarity against a target audio sample.
- **Web UI** exposes the catalog dimensions as an interactive spider chart for manual voice sculpting with real-time audio preview.

**A note on labels:** Component labels like "pitch" and "brightness" are approximations. Each dimension doesn't control a single isolated audio attribute — voice characteristics are entangled across the high-dimensional space. The labeling script measures which audio feature changes most when a component is perturbed, but in practice each component subtly affects multiple features simultaneously. Think of the labels as the _dominant_ effect, not the only one.

## Web UI

An interactive spider/radar chart interface for sculpting voices in real time with continuous audio playback and crossfade.

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

| Endpoint            | Method | Description                                     |
| ------------------- | ------ | ----------------------------------------------- |
| `/api/health`       | GET    | Health check                                    |
| `/api/voices`       | GET    | List available .pt voice files                  |
| `/api/catalog`      | GET    | Component names and count                       |
| `/api/synthesize`   | POST   | Synthesize WAV from voice + coefficients + text |
| `/api/upload-voice` | POST   | Upload a .pt voice file to voices/              |

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

| Flag            | Default       | Description                                                      |
| --------------- | ------------- | ---------------------------------------------------------------- |
| `--catalog`     | auto-detected | Path to discovery catalog (skips PCA recomputation)              |
| `--n-probes`    | 200           | Discovery probes per iteration                                   |
| `--n-pca`       | 32            | PCA components to include                                        |
| `--n-discovery` | 32            | Top discovered components to include                             |
| `--passes`      | 3             | Auto-tune coordinate descent passes                              |
| `--start-step`  | 0.1           | Starting step size (scaled per-component by sensitivity)         |
| `--mag-steps`   | 3             | Magnitude refinement steps (each is 10× smaller)                 |
| `--keep-base`   | off           | Always start from original base voice instead of rolling forward |

## Development: Improving Voice Dimensions

The catalog (`catalog/discovery_catalog.pt`) and labels (`catalog/component_labels.json`) are committed to the repo, so regular users don't need to rebuild them. The steps below are for actively improving the voice model by discovering new directions and updating the shipped catalog.

### 1. Discover new voice directions

Run `auto_mode.py` to probe for impactful directions beyond PCA (see [Auto Mode](#auto-mode) above for full options):

```bash
uv run python auto_mode.py \
    --base-voice voices/af_heart.pt \
    --target-audio my_target.wav \
    --target-text "Hello, my name is Alex." \
    --n-probes 200
```

Discoveries accumulate in `output/discovery_cache.pt` across runs.

### 2. Build the catalog

Distill PCA + top discoveries into a single catalog file:

```bash
# Build catalog with top 1000 discovered directions
uv run python build_catalog.py --n-discoveries 1000

# Preview size without writing anything
uv run python build_catalog.py --dry-run

# Write to a custom path
uv run python build_catalog.py --n-discoveries 500 --output catalog/v2.pt
```

### 3. Label components

Run sensitivity analysis to assign human-readable labels (pitch, brightness, nasality, etc.) to each component:

```bash
# Label all components in the default catalog
uv run python label_components.py

# Use a specific base voice and text
uv run python label_components.py --voice voices/af_heart.pt --text "Hello there."

# Write labels to a custom path
uv run python label_components.py --output catalog/my_labels.json
```

This generates one Kokoro synthesis per component, so it takes a few minutes for large catalogs. Results are saved to `catalog/component_labels.json`. The web UI loads these labels automatically on startup.

### 4. Commit the updated files

```bash
git add catalog/discovery_catalog.pt catalog/component_labels.json
git commit -m "Update catalog and labels"
```

Use git-lfs for large catalogs.

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
├── auto_mode.py            # Headless continuous refinement loop
├── synthesize.py           # Quick test synthesis from any .pt voice
├── build_catalog.py        # Distill PCA + top discoveries into a catalog
├── build_semantic_map.py   # Build semantic directions from catalog sensitivity
├── label_components.py     # Label components via sensitivity analysis
├── prune_cache.py          # Trim working discovery cache to top-N
├── catalog/                # Version-controlled discovery catalog
│   ├── discovery_catalog.pt
│   ├── component_labels.json  # Human-readable labels for each component
│   └── semantic_map.json      # Semantic directions for disentangled sliders
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

## How Kokoro Stores a Voice

Worth understanding before the sections below, because it reframes what the tools are doing.

A `.pt` voice file is `[510, 1, 256]`, but it is **not** a 130,560-dimensional description of a voice. Kokoro's pipeline indexes it by phoneme count:

```python
# kokoro/pipeline.py
return model(ps, pack[len(ps)-1], speed, return_output=True)
```

It is **510 separate 256-dim style vectors**, and one utterance uses exactly one of them — the row matching its phoneme count. Within that row, the model splits the 256 dims cleanly:

```python
# kokoro/model.py
s = ref_s[:, 128:]                                            # prosody predictor
audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])    # decoder
```

- **`[:128]` — timbre.** Feeds the decoder. Vocal tract, formants, who is speaking.
- **`[128:]` — prosody.** Feeds duration, F0 and energy prediction. Pacing, rhythm, intonation.

Measured across the 510 rows, timbre varies only ~15% while prosody varies ~55% — the length indexing exists mostly to modulate prosody. This is why the newer tools optimize one shared timbre vector plus a smooth prosody profile rather than treating all 130,560 numbers as free.

Two further properties, both verified rather than assumed:

- **Synthesis is stochastic.** The vocoder draws a random initial phase and injects noise (`kokoro/istftnet.py`), so identical inputs produce different audio — peak difference ~0.13. Resemblyzer scores two identical runs at ~0.9985, not 1.0, which is a noise floor for any similarity-based search. Call `torch.manual_seed()` before synthesis when you need reproducibility.
- **The decoder emits exactly 600 audio samples per predicted duration frame** at 24kHz.

## Style Sliders (v2)

The spider chart's **Style** mode uses axes measured in Kokoro's native 256-dim style space. This supersedes the older Semantic mode (below), which worked in PCA/discovery component space.

Built by `build_style_map.py`, which differs from the older approach in four ways:

1. **Exact gradients, not finite differences.** `core/differentiable_kokoro.py` reimplements Kokoro's forward pass without its `@torch.no_grad()` decorator, so each feature's sensitivity comes from one backward pass. Twelve features means twelve backward passes, versus 1055 syntheses for the old map — seconds instead of minutes, and exact rather than noisy.
2. **The right space.** The Jacobian is `[12 features x 256 dims]`. Inverting that is well-conditioned, unlike pseudo-inverting `[1055, 20]`.
3. **Structural disentanglement.** Each axis is masked to the half that architecturally controls it, so a pitch slider *cannot* move timbre. Measured gradient share confirms the split is real: pace, pitch and energy are 100.0% prosody-half, while the spectral features are 80–92% timbre-half.
4. **Calibrated sliders.** Slider at 1.0 means "change this feature by 30% of the base voice's value," rather than an arbitrary scale.

Five axes are read straight off the model's internals rather than estimated from audio — `pace` from the continuous duration, `pitch`/`pitch variation` from `F0_pred`, `energy`/`energy variation` from `N_pred`. These are exact. The remaining seven (brightness, fullness, breathiness, sibilance, warmth, volume, dynamics) are measured on the waveform with differentiable spectral features in `core/spectral_features.py`.

```bash
# Build the map (about 40 seconds)
uv run python build_style_map.py

# Verify it by actually synthesizing, on held-out text
uv run python verify_style_map.py
```

`verify_style_map.py` matters: the Jacobian is a *local linearization* of a nonlinear model, so the predicted behavior has to be checked against real synthesis. Current measured results on held-out text, sliders at 1.0 with a 30% target:

- 9 of 12 axes land within a few points of +30%; breathiness (+13.8%) and brightness (+15.2%) under-deliver.
- Mean worst-case cross-talk is 8.4%. The linear prediction claims 0.02%, so most remaining entanglement is nonlinearity, not a flaw in the inversion.
- The largest leaks are between physically coupled features (brightness/breathiness, warmth/sibilance) — genuine perceptual correlation, not a bug.

Regenerate the map when you change base voice; it is a local approximation around whichever voice it was built on.

## Semantic Voice Sliders (v1, superseded)

Kept for comparison against Style mode. The web UI supports these modes for the spider chart, toggled above the chart:

- **Raw mode** (default): One axis per PCA/discovery component. Each axis represents one direction in voice tensor space. Labels are approximate — adjusting one component may subtly affect multiple audio features.
- **Semantic mode**: One axis per audio feature (pitch, brightness, breathiness, etc.). Each slider maps to a learned weighted combination of raw components that maximally changes one feature while minimally affecting others. This produces cleaner, more intuitive control.

### How it works

1. **Dense sensitivity matrix.** `build_semantic_map.py` perturbs each catalog component and measures the delta across all 20 tracked audio features (pitch, spectral centroid, MFCCs, energy, etc.), producing an N x M matrix `S[i,j]` = how much component `i` affects feature `j`.

2. **Pseudo-inverse for disentangled directions.** The pseudo-inverse `S+` (M x N) is computed so each row is a set of component weights that maximally changes one audio feature while minimally affecting others.

3. **Frontend mapping.** In semantic mode, the frontend multiplies the semantic coefficient vector by the directions matrix to compute raw coefficients: `raw_coeffs = semanticCoeffs @ directions`. The raw coefficients are sent to the same `/api/synthesize` endpoint.

### Generating / regenerating the semantic map

```bash
# Generate from the default catalog and voice
uv run python build_semantic_map.py

# Use a specific base voice and text
uv run python build_semantic_map.py --voice voices/af_heart.pt --text "Hello there."

# Custom output path
uv run python build_semantic_map.py --output catalog/semantic_map_v2.json
```

This requires one Kokoro synthesis per component, so it takes a few minutes for large catalogs. The result is saved to `catalog/semantic_map.json` and loaded automatically by the server on startup.

Re-running after catalog improvements (more discoveries, updated PCA) updates the mapping to reflect the new component space.

### Full workflow

1. **Discover** new voice directions with `auto_mode.py`
2. **Build catalog** with `uv run python build_catalog.py`
3. **Build semantic map** with `uv run python build_semantic_map.py`
4. **Use in web UI** — start the server and frontend, toggle to Semantic mode

### Limitations

Perfect disentanglement is not possible — some audio features are physically correlated (e.g., pitch and spectral centroid). The mapping provides substantially more intuitive control than raw components, but adjusting one semantic slider may still produce small shifts in correlated features.
