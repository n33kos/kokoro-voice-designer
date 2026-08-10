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

# Measure the built-in voices (~3 min) — lets voice matching pick a starting voice
uv run python build_voice_registry.py

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
  --f0-reference reference.wav \
  --steps 120 --lr 0.02 \
  --reg-weight 50.0 --speaker-weight 2.0 \
  --f0-weight 5.0 --f0-range-weight 1.5 \
  --pacing-weight 2.0 --energy-weight 5.0 \
  --duration-spread-weight 2.0 --creak-weight 3.0 \
  --tremor-weight 1.5 --punctuation-weight 2.0 \
  --constraint-start 0.5 --constraint-ramp 0.2 --trust-weight 600 \
  --out output/my_voice.pt
```

Those are the settings that produced the best results to date. The three
scheduling flags at the end matter as much as the weights — see
[the two-phase schedule](#the-two-phase-schedule).

The starting voice is chosen automatically — whichever built-in already sounds
most like the reference, by speaker-embedding similarity. Pass `--base` to
override.

Roughly 30 minutes on CPU for a single 20-second clip. Every synthesis is seeded,
so runs are reproducible, and checkpoints are written every 5 steps.

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
  ... --f0-reference long_recording.wav --out output/my_voice.pt
```

`split_reference.py` verifies its own alignment by checking that every segment's implied speaking rate is plausible and consistent. This matters: matching an existing transcript to silence-delimited segments *looks* like it should work but often doesn't, because narrators pause mid-sentence and run sentences together. Use `--whisper` to transcribe each segment directly and skip the guesswork.

Cost scales linearly with clip count. Three to five clips spanning short to long is a good target; more clips of the *same* length add cost without unlocking any prosody flexibility.

### What the loss terms do

Each exists because a measurement showed a specific defect, and each is a
**hinge with a deadband** rather than a target: zero cost while the voice is
already acceptable, so a term cannot perturb a voice that does not have the
defect it targets. That property is load-bearing — a plain squared error is never
zero, and one added as a target regressed the one voice that already sounded
right.

| flag | what it constrains |
| --- | --- |
| *(always on)* | WavLM layer-4 pooled statistics — the main "sounds like this voice" term |
| `--speaker-weight` | Differentiable Resemblyzer embedding. Constrains *identity* where WavLM constrains *texture* |
| `--reg-weight` | Hinge on leaving the range spanned by the built-in voices. Zero cost inside it |
| `--f0-weight` | Matches the whole log-pitch distribution across 25 quantiles — register and asymmetry, not an average |
| `--f0-range-weight` | One-sided hinge on p90/p94/p98. The distribution mean cannot feel a tail error: a 10% miss at p90 contributes 0.0004 |
| `--pacing-weight` | Overall speaking rate |
| `--energy-weight` | Spread of frame level in dB on rendered audio, plus a barrier against Kokoro's internal energy contour inverting |
| `--duration-spread-weight` | Keeps phoneme-length variation where the base voice had it. Without it, `pacing_loss` crushes the longest phonemes and the rhythm flattens |
| `--creak-weight` | One-sided, log-domain. Keeps frames below the register no more common than in the reference |
| `--tremor-weight` | Hinge on 3-10 Hz pitch modulation. Nothing else can see a steady wobble — creak counts frames below a threshold, and a tremor sits at the median |
| `--punctuation-weight` | Keeps pauses at punctuation from being swallowed, with separate targets for sentence ends and commas |
| `--noise-floor-weight` | Off by default. Improves the quantity it optimizes by up to 63 dB while the audible floor moves 0-3 dB |
| `--duration-weight` | Off by default — it flattened natural timing into staccato |

### The two-phase schedule

The prosody terms are barriers. They are zero once satisfied, but they block
regions of style space while the search is still moving, and switched on at full
strength they total roughly **seventeen times** the perceptual loss. That kicks
the voice out of the basin it had found: tracked across versions, the WavLM term
degraded from 0.135-0.181 to 0.157-0.225 as terms were added, while every prosody
metric improved.

Three flags fix that, and they are worth more than any individual weight:

- `--constraint-start 0.5` — optimize voice match alone for the first half of the
  run, so the constraints repair rather than steer.
- `--constraint-ramp 0.2` — fade them in over 20% of the run instead of switching
  them on, so the voice is never kicked.
- `--trust-weight 600` — penalize drift from the style vector as it stood when
  the constraints engaged.

Together these cut the post-engagement loss of voice match from 20% to 5%.

**Track the perceptual term across versions.** Its endpoint alone hides this —
the curve is what shows the damage, and five versions were spent blaming
individual terms for a problem that was about how abruptly they arrived.

### Measurement discipline

Nearly every bug in this project has had one shape: **two numbers that look like
the same measurement and are not.** A loss reports itself satisfied while the
audio is audibly wrong.

The pitch tracker is the worst case. Every target — register, distribution,
creak, contour — came from `librosa.yin` masked by a frequency band, which
reports **8-13% of adjacent voiced frames jumping more than an octave** on real
recordings. Those octave errors *were* the targets: the creak target asked for
7-12% of frames below the register when the truth is 0-1%, so the optimizer was
being asked to produce the croakiness. `core/pitch.py` replaced it with a
confidence-masked tracker (CREPE, or pyin as fallback). Read that module before
touching anything pitch-related.

Four rules that came out of it:

1. **Sweep the free parameter before optimizing against a statistic.** A value
   that changes when you halve the analysis window is not measuring the signal.
   Frame-to-frame pitch movement read the same at 12.5 ms and 25 ms — the tell
   that it was tracker noise. Declination was abandoned for the same reason: it
   moved 13x across segmentation settings.
2. **Bounds on Kokoro's internal `F0_pred`/`N_pred` need a bridge to the same
   statistic on rendered audio**, calibrated on the base voice. Those contours
   are not band-limited like tracked pitch — they ramp from zero at every voicing
   onset. A tremor bound set from tracked audio came out at 42.7 against a bound
   of 0.06, i.e. 99% of the objective.
3. **A bridge decays; a reference value does not.** Anything mapping between two
   domains must be re-measured on the current voice during the run, the way the
   pitch offset is. Fixed at step 0, the pause scale and tremor bound both went
   stale and their terms went quiet on bounds that were no longer true.
4. **Before running, check a new term reads zero on the voice that already sounds
   best.** If it is non-zero there, it will move it.

### Reproducibility

Two runs with identical settings differ by **4-5% on every pitch quantile**, and
39% relative on the style vector. Treat anything smaller than that in a
single-run comparison as noise. Averaging independent runs helps where the error
is scatter and not where it is bias — summed quantile error dropped 12.4% → 9.7%
on one voice and 4.7% → 3.3% on another, while two voices whose p90 was
reproducibly low were unchanged by it.

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

## Project structure

```
voice-designer/
├── invert_voice.py          # Gradient-based voice matching
├── build_style_map.py       # Build the slider axes (Jacobian in style space)
├── verify_style_map.py      # Check the axes against real synthesis
├── evaluate_match.py        # Score candidates against a reference
├── design_voices.py         # Design original voices from the style axes
├── split_reference.py       # Cut a long recording into transcribed clips
├── test_differentiable.py   # Gate: our forward pass must match stock Kokoro
├── server.py                # FastAPI backend
├── build_voice_registry.py  # Measure built-in voices (base-voice selection)
├── synthesize.py            # Quick test synthesis from any .pt voice
├── core/
│   ├── pitch.py                  # The one reference pitch tracker — read first
│   ├── differentiable_kokoro.py  # Kokoro forward pass with gradients enabled
│   ├── perceptual_loss.py        # WavLM, pitch, energy, duration and pause losses
│   ├── speaker_loss.py           # Differentiable Resemblyzer + manifold bounds
│   ├── spectral_features.py      # Differentiable audio features
│   ├── speech_generator.py       # Kokoro wrapper (seedable)
│   └── fitness_scorer.py         # Resemblyzer similarity, for evaluation
├── catalog/
│   ├── style_map.json       # Slider axes (build_style_map.py)
│   └── voice_registry.json  # Built-in voice measurements (build_voice_registry.py)
├── voices/                  # .pt voice files
└── web/                     # React + TypeScript frontend
```

### A note on `core/pitch.py`

Every reference pitch measurement in the project goes through it, and its
docstring explains why: the targets were wrong for a long time because the
tracker behind them was, and the losses reported success throughout. Anything
that measures pitch on a recording belongs here, not scattered across call sites
with its own band and hop.

### A note on `differentiable_kokoro.py`

Kokoro's `forward_with_tokens` is decorated with `@torch.no_grad()`, which makes gradient-based optimization impossible. That module mirrors those ~30 lines with gradients enabled, and exposes two things the stock path discards: the *continuous* duration before rounding (the only differentiable handle on pacing) and the `F0_pred`/`N_pred` contours.

`forward(..., decode=False)` stops before the vocoder. Peak memory in the
backward pass is dominated by retained vocoder activations — roughly 0.75 GB per
second of audio — so duration constraints can be applied at utterance lengths
that would be far too expensive to render. That is how pauses are constrained at
a 400-phoneme passage while training on 2-6 second clips.

Since this depends on Kokoro internals rather than a public API, `test_differentiable.py` asserts that under a fixed seed our output is **bit-identical** to stock Kokoro. Run it after upgrading Kokoro.

---

## Requirements

- Python 3.11–3.12, `uv`
- Node 18+ for the web UI
- ~4GB disk for models (Kokoro, WavLM, Resemblyzer, CREPE)
- Apple Silicon, CUDA or CPU. Voice matching defaults to CPU — the model is small and MPS has gaps in backward-pass coverage.

**Memory:** peak usage is roughly **0.75 GB per second of training audio**, because
the backward pass retains vocoder activations at 24 kHz. Cost scales with clip
*duration*, not count — 4.3 s needs ~3.6 GB, 9.7 s needs ~7.3 GB. A 14.6 s clip
was killed by the OS mid-run, and that failure is silent: the process vanishes
and a queue script moves on, so it looks as though the run never happened.
`--max-clip-seconds` (default 10) refuses over-long clips up front. Run one match
at a time.

`torchcrepe` is optional but recommended — `core/pitch.py` falls back to
`librosa.pyin`, which is faster but less accurate in the tails.

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
