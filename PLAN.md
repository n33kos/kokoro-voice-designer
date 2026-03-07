# Voice Designer Web App - Implementation Plan

## Overview

Replace the existing Gradio UI with a clean, purpose-built web application for crafting custom Kokoro voices using an interactive spider/radar chart.

## Architecture

### Backend: FastAPI (Python) - `server.py`
- Serves Kokoro TTS synthesis API
- Loads catalog data (PCA + discovery components)
- Lists and loads voice files
- Returns synthesized WAV audio

### Frontend: React + TypeScript + Vite + CSS Modules - `web/`
- Interactive spider/radar chart (HTML Canvas)
- Clean, dark-themed UI
- Web Audio API for seamless crossfade playback

## Core Features

1. **Voice Selector** - Dropdown to pick a starting base voice from `voices/` directory
2. **Spider/Radar Chart** - Interactive chart where each axis = one voice component (PCA + top discoveries from catalog). User drags points to adjust voice. Labeled with dominant traits (pitch, brightness, nasality, etc.)
3. **Text Input** - User types the sentence the voice will speak
4. **Playback Controls** - Play/Pause, continuous loop. When spider chart changes, new audio synthesized and crossfaded mid-sentence
5. **Loading Indicator** - Subtle loading state during synthesis

## Audio Behavior

- Audio plays in a continuous loop
- On spider chart adjustment: debounce 300-500ms, then request new audio from backend
- Keep playing current audio while new audio generates
- When new audio arrives, crossfade mid-sentence (don't wait for finish)
- Use Web Audio API for precise crossfade control

## Backend API

```
GET  /api/voices     - List available .pt voice files
GET  /api/catalog    - Component info (names, count, ranges)
POST /api/synthesize - Accept coefficients + text, return WAV audio
GET  /api/health     - Health check
```

### Backend Implementation Details

1. Load catalog from `catalog/discovery_catalog.pt` (PCA components, discovery components, labels)
2. Load base voice from `voices/`
3. Apply coefficients: `voice = base_voice + sum(coeff[i] * component_range[i] * component[i])`
4. Use existing `SpeechGenerator` from `core/speech_generator.py`
5. Return WAV audio bytes
6. Component ranges: project voice library onto components (same as auto_mode.py lines 273-287)
7. Discovery component_ranges fix: replace tiny discovery ranges with average PCA range
8. Suppress PyTorch/Kokoro warnings

### Fallback
If catalog doesn't exist, compute PCA fresh from voices/ directory on startup.

## Frontend File Structure

```
web/
  package.json
  tsconfig.json
  vite.config.ts
  index.html
  src/
    main.tsx
    App.tsx
    App.module.css
    components/
      SpiderChart.tsx          # Interactive radar chart (Canvas)
      SpiderChart.module.css
      PlaybackControls.tsx     # Play/pause + loading
      PlaybackControls.module.css
      VoiceSelector.tsx        # Base voice dropdown
      VoiceSelector.module.css
      TextInput.tsx            # Synthesis text input
      TextInput.module.css
    hooks/
      useAudioEngine.ts        # Web Audio API crossfade logic
      useVoiceSynthesis.ts     # API calls + debouncing
    types.ts
    api.ts
```

## Spider Chart Details

- HTML Canvas-based
- Concentric polygon grid rings at 25%, 50%, 75%, 100%
- Axis lines from center to edge, labeled with component names
- Current voice profile shown as filled polygon
- Draggable axis points (mouse + touch)
- Each axis: -1.0 (center) to 1.0 (edge), default 0.0 (middle ring)
- Dark theme, subtle grid, accent color for voice polygon

## Design

- Dark theme, clean typography
- Minimal chrome - spider chart is the visual centerpiece
- Desktop-optimized, responsive
- CSS modules for all styling
- Modern, professional creative tool aesthetic

## Component Labels

Uses FEATURE_LABELS mapping from auto_mode.py:
- pitch, pitch_variation, brightness, fullness, crispness, clarity
- breathiness, volume, energy, energy_variation, timbre, nasality
- resonance, texture, harmonics, tonality, pace, dynamics
- harmonic_richness, sibilance

If catalog doesn't have pre-computed labels, compute via sensitivity analysis on startup.

## Changes Summary

### New Files
- `server.py` - FastAPI backend
- `web/` - Entire frontend directory

### Modified Files
- `README.md` - Document new web UI, keep existing docs

### Unchanged
- `app.py` - Old Gradio UI preserved (not modified)
- `auto_mode.py` - Headless refinement (separate feature)
- `synthesize.py` - Quick test synthesis
- `build_catalog.py` - Catalog builder
- `prune_cache.py` - Cache pruner
- `core/` - All core modules unchanged

## Auto Mode vs Web UI

These are separate features:
- **Auto Mode** (`auto_mode.py`): Headless, automated refinement targeting a specific voice via similarity scoring
- **Web UI** (`server.py` + `web/`): Interactive creative tool for manually crafting custom voices via spider chart

## Bug Fixes Applied in This Session

1. **Discovery component scaling** (auto_mode.py): Discovery components were effectively dead in auto-tune because component_ranges were tiny (orthogonal to PCA). Fixed by replacing discovery ranges with average PCA range.
2. **Improvement guard** (auto_mode.py): Added check to only carry forward tuned voice if `best_sim >= iter_base_sim`, preventing regression from MPS non-determinism.
3. **Warning suppression** (auto_mode.py): Added filters for PyTorch STFT resize warnings and librosa n_fft warnings.
