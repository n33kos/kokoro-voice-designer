# Voice Designer

Design synthetic voices for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — either by hand, with calibrated sliders for pitch, pace, brightness and warmth, or automatically, by fitting a voice to a reference recording with gradient descent.

Please read [Acceptable Use](#acceptable-use) before using the reference-fitting workflow.

---

## How Kokoro stores a voice

Everything here follows from one fact about the model, so it's worth understanding first.

A `.pt` voice file is `[510, 1, 256]`, but that is **not** a 130,560-dimensional description of a voice. Kokoro indexes it by phoneme count:

```python
# kokoro/pipeline.py
return model(ps, pack[len(ps)-1], speed, return_output=True)
```

It is **510 separate 256-dim style vectors**, and one utterance uses exactly one of them — the row matching its phoneme count. Inside that row, the model splits the 256 numbers cleanly:

```python
# kokoro/model.py
s = ref_s[:, 128:]                                            # prosody predictor
audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])    # decoder
```

- **`[:128]` — timbre.** Feeds the decoder. Vocal tract, formants, who is speaking.
- **`[128:]` — prosody.** Feeds duration, pitch and energy prediction. Pacing, rhythm, intonation.

Measured across the 510 rows, timbre varies only ~15% while prosody varies ~55% — the length indexing exists mostly to modulate prosody. So a voice is well described by one shared timbre vector plus a prosody profile that varies with utterance length.

Two more properties, both measured rather than assumed:

- **Synthesis is stochastic.** The vocoder draws a random initial phase and injects noise (`kokoro/istftnet.py`), so identical inputs give different audio. Two runs of the same voice score ~0.9985 on Resemblyzer, not 1.0 — a noise floor under any similarity metric. Seed before synthesis when you need reproducibility.
- **The decoder emits exactly 600 audio samples per predicted duration frame** at 24kHz.

---

## Getting Started

```bash
uv sync
cd web && npm install && cd ..

# Build the style map (~40s) — the web UI needs this
uv run python build_style_map.py

# Terminal 1 — backend
uv run uvicorn server:app --reload --port 8000

# Terminal 2 — frontend
cd web && npm run dev
```

Open `http://localhost:5173`.

---

## Designing a voice by hand

The web UI is a spider chart with one axis per voice characteristic. Pick a base voice, type some text, press play, and drag points — audio re-synthesizes and crossfades in continuously.

Twelve axes, all calibrated so that a slider at 1.0 changes that characteristic by 30% of the base voice's value:

| prosody axes | timbre axes |
| --- | --- |
| pace, pitch, pitch variation, energy, energy variation | volume, brightness, fullness, breathiness, sibilance, warmth, dynamics |

Each axis is masked to the half of the style vector that architecturally controls it, so **a pitch slider cannot move timbre**. That isn't a tuning choice; it's structural.

### How the axes are built

`build_style_map.py` computes the Jacobian of each characteristic with respect to the 256-dim style vector — one backward pass per axis, exact, about 40 seconds total.

Five axes are read straight off the model's internals rather than estimated from audio: `pace` from Kokoro's continuous duration, `pitch`/`pitch variation` from `F0_pred`, `energy`/`energy variation` from `N_pred`. These are exact. The other seven are measured on the waveform using differentiable spectral features (`core/spectral_features.py`).

Inverting a `[12 x 256]` Jacobian is well-conditioned, so the axes come out genuinely independent. Verify on your own base voice:

```bash
uv run python verify_style_map.py
```

That synthesizes with each slider at 1.0 on held-out text and reports what actually changed versus what was predicted. Measured on `af_heart`: 9 of 12 axes land within a few points of the 30% target, with mean cross-talk of 8.4%. The gap between the linear prediction and reality is nonlinearity — the Jacobian is a local approximation, so rebuild it if you change base voice substantially.

---

## Fitting a voice to a reference recording

`invert_voice.py` solves for a voice by gradient descent against a reference recording. Kokoro's 82M weights stay frozen; the only things optimized are a few hundred numbers describing one voice. This is the inverse of training, using the same machinery.

This produces a voice that resembles the reference. Only use audio you own or have documented permission to use — see [Acceptable Use](#acceptable-use).

**You need a recording plus its transcript.** The transcript matters more than you'd expect — see [why](#why-the-transcript-matters).

```bash
uv run python invert_voice.py \
  --target reference.wav --target-text reference.txt \
  --base voices/am_michael.pt \
  --steps 120 --lr 0.02 \
  --reg-weight 50.0 --speaker-weight 2.0 --f0-weight 5.0 \
  --duration-weight 2.0 \
  --out output/my_voice.pt
```

Roughly 30 minutes on CPU for a single 20-second clip. Every synthesis is seeded, so runs are reproducible.

### Multiple clips

Each distinct utterance length anchors one row of the voice pack, so a single clip only supports a constant prosody offset. Several clips of **different lengths** let prosody vary with length again — the basis expands automatically to match the number of distinct lengths found.

One long recording can be cut into many, which is usually better than gathering separate clips: same session means one microphone, one room, one noise floor.

```bash
# Cut at pauses and transcribe each segment with whisper.cpp
uv run python split_reference.py --audio long_recording.wav \
  --whisper --out-dir input/split

# Then pass each clip with its own transcript
uv run python invert_voice.py \
  --target input/split/seg02.wav --target-text input/split/seg02.txt \
  --target input/split/seg05.wav --target-text input/split/seg05.txt \
  ... --base voices/am_michael.pt --out output/my_voice.pt
```

`split_reference.py` verifies its own alignment by checking that every segment's implied speaking rate is plausible and consistent. This matters: matching an existing transcript to silence-delimited segments *looks* like it should work but often doesn't, because narrators pause mid-sentence and run sentences together. Use `--whisper` to transcribe each segment directly and skip the guesswork.

Cost scales linearly with clip count. Three to five clips spanning short to long is a good target; more clips of the *same* length add cost without unlocking any prosody flexibility.

### What the loss terms do

Each exists because a measurement showed the previous version was wrong in a specific way.

| flag | what it constrains |
| --- | --- |
| *(always on)* | WavLM layer-4 pooled statistics — the main "sounds like this voice" term |
| `--speaker-weight` | Differentiable Resemblyzer embedding. Constrains *identity* where WavLM constrains *texture* |
| `--reg-weight` | Hinge penalty on leaving the range spanned by the built-in voices. Zero cost inside that range |
| `--f0-weight` | Matches the reference's log-pitch mean and spread |
| `--duration-weight` | Lengthens phonemes Kokoro allocates too little time to |
| `--pacing-weight` | Matches overall speaking rate |

Two notes worth knowing:

- **With `--speaker-weight` set, Resemblyzer becomes a training target**, so its similarity score is no longer an independent metric. Judge results by ear, and by how far the result strays from the built-in voice range.
- **Gradient descent will walk the style vector outside the region Kokoro was trained on** if you let it, which sounds like graininess and pitch spiking. That's what `--reg-weight` is for.

### Why the transcript matters

The pooled WavLM statistics are far more content-dependent than "content-independent" suggests. Compared against matched content, the loss for the correct voice is *exactly zero*. But a target averaged across different texts leaves a residual no single utterance can reach, and the optimizer can only chase it by distorting the voice.

So supply a transcript. Without one, the tool falls back to generic texts and warns you.

### Evaluating

```bash
uv run python evaluate_match.py --reference reference.wav \
  --candidates voices/base.pt output/my_voice.pt --save-audio
```

Scores candidates on held-out text and writes samples for listening.

---

## Discovery and the component catalog

An older, separate line of work, still used by `auto_mode.py`. PCA over Kokoro's built-in voices finds the axes of greatest variation; discovery probes random directions orthogonal to PCA to find impactful dimensions the voice library doesn't vary along. `build_catalog.py` distills the results.

This powers `auto_mode.py`, a coordinate-descent optimizer that predates the gradient-based approach. It fits a voice to a reference recording the same way `invert_voice.py` does, but by black-box search rather than gradients, and it converges to a noticeably worse result for far more compute. Prefer `invert_voice.py`. Discovery remains useful for exploring directions outside the built-in voice distribution.

### The catalog and Git LFS

`catalog/discovery_catalog.pt` is ~530MB, over GitHub's 100MB per-file limit, so it is stored with [Git LFS](https://git-lfs.com):

```bash
brew install git-lfs   # or: apt install git-lfs
git lfs install && git lfs pull
```

Without LFS the repo still clones and everything except the catalog works — you'll get a pointer file instead. To rebuild it yourself:

```bash
# Accumulate discoveries (slow — hours for a mature cache; stop with Ctrl+C anytime)
uv run python auto_mode.py --base-voice voices/af_heart.pt \
  --target-audio input/sample.wav --target-text "transcript" --n-probes 200

# Distill into a catalog (fast)
uv run python build_catalog.py --n-discoveries 1000
```

Note the real cost: the discovery cache is ~1GB and represents many hours of probing. **You don't need the catalog for the designer or for voice matching** — `build_style_map.py` and `invert_voice.py` work directly from Kokoro and a base voice.

---

## Project structure

```
voice-designer/
├── invert_voice.py          # Gradient-based voice matching
├── build_style_map.py       # Build the slider axes (Jacobian in style space)
├── verify_style_map.py      # Check the axes against real synthesis
├── evaluate_match.py        # Score candidates against a reference
├── split_reference.py       # Cut a long recording into transcribed clips
├── test_differentiable.py   # Gate: our forward pass must match stock Kokoro
├── server.py                # FastAPI backend
├── auto_mode.py             # Older coordinate-descent loop + discovery
├── build_catalog.py         # Distill PCA + discoveries into a catalog
├── synthesize.py            # Quick test synthesis from any .pt voice
├── core/
│   ├── differentiable_kokoro.py  # Kokoro forward pass with gradients enabled
│   ├── perceptual_loss.py        # WavLM, pacing, F0 and duration losses
│   ├── speaker_loss.py           # Differentiable Resemblyzer + manifold bounds
│   ├── spectral_features.py      # Differentiable audio features
│   ├── voice_analyzer.py         # PCA decomposition
│   ├── speech_generator.py       # Kokoro wrapper (seedable)
│   ├── discovery.py              # Orthogonal direction probing
│   └── fitness_scorer.py         # Similarity metrics
├── catalog/style_map.json   # Slider axes (regenerate with build_style_map.py)
├── voices/                  # .pt voice files
└── web/                     # React + TypeScript frontend
```

### A note on `differentiable_kokoro.py`

Kokoro's `forward_with_tokens` is decorated with `@torch.no_grad()`, which makes gradient-based optimization impossible. That module mirrors those ~30 lines with gradients enabled, and exposes two things the stock path discards: the *continuous* duration before rounding (the only differentiable handle on pacing) and the `F0_pred`/`N_pred` contours.

Since this depends on Kokoro internals rather than a public API, `test_differentiable.py` asserts that under a fixed seed our output is **bit-identical** to stock Kokoro. Run it after upgrading Kokoro.

---

## Requirements

- Python 3.11–3.12, `uv`
- Node 18+ for the web UI
- ~4GB disk for models (Kokoro, WavLM, Resemblyzer)
- Apple Silicon, CUDA or CPU. Voice matching defaults to CPU — the model is small and MPS has gaps in backward-pass coverage.

---

## Acceptable Use

The code is MIT licensed (see [LICENSE](LICENSE)). This section is separate from the license and is not a legal restriction — use-restricted licenses aren't open source and are largely unenforceable anyway. It's how the project asks to be used.

### Only fit against audio you own or have permission to use

The reference-fitting workflow produces a voice that resembles whoever is in the recording. Use your own voice, audio you have documented permission to use, or public-domain or synthetic sources. Finding audio online is not permission, and neither is owning a copy of a recording — buying an audiobook doesn't license the narrator's voice.

If you want reference material for testing, the datasets built for speech research are a better fit than anything scraped: [VCTK](https://datashare.ed.ac.uk/handle/10283/3443), [Common Voice](https://commonvoice.mozilla.org/), [LibriSpeech](https://www.openslr.org/12), and [LJSpeech](https://keithito.com/LJ-Speech-Dataset/). They're consented, cleanly recorded, and give you many clips per speaker at varying lengths, which is what the multi-clip workflow wants.

### Publishing a voice file is permanent

A `.pt` voice file is not a recording — it's the ability to generate unlimited speech in that voice, for anyone who has a copy, indefinitely. You cannot un-publish it. Deleting the repo doesn't retract the copies.

**This applies to your own voice too.** Publishing a voice fitted to yourself hands everyone who downloads it the ability to make you say anything, permanently. That may still be a fine trade for you — just make it deliberately.

### Don't publish voices that imitate real identifiable people

Fitting a voice to a reference for your own use is one thing. Publishing or distributing a voice file that reproduces a real, identifiable person's voice is another, and this project asks you not to do it. That includes public figures, and it especially includes people whose voice is their livelihood — narrators, voice actors, broadcasters, performers.

If you publish voices made with this tool, say how they were made and what they were fitted to.

### Legal context

Not legal advice, and this is a fast-moving area. Tennessee's ELVIS Act, California's digital replica statutes, and the EU AI Act's disclosure duties for synthetic media all exist and all broadly target **whoever deploys or distributes a voice replica**, rather than whoever wrote the software. If you publish a voice or the audio it generates, that's you. Worth understanding what applies where you are before you publish.

### Roadmap: watermarking

Output watermarking is under consideration — likely [AudioSeal](https://github.com/facebookresearch/audioseal) or [Resemble Perth](https://github.com/resemble-ai/perth), both open source and both designed to survive ordinary audio processing. It would mark generated audio as synthetic and make provenance checkable, which is a better position than relying on good intentions. Not implemented yet.
