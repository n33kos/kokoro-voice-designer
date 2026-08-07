#!/usr/bin/env python3
"""Gradient-based voice inversion: solve for the style vector that sounds like a target.

This is the inverse of training. Kokoro's 82M weights stay frozen; the only
things optimized are the few hundred numbers describing one voice. Where
auto_mode.py treats Kokoro as a black box (perturb, synthesize, score, keep if
better), this backpropagates through the model and moves every parameter at once.

Parameterization follows the architecture rather than guessing (see plan.md F1-F3):

  - Kokoro indexes a voice pack by phoneme count: `pack[len(ps)-1]`. Only one
    256-dim row is used per utterance.
  - `ref_s[:, :128]` drives the decoder (timbre); `ref_s[:, 128:]` drives the
    prosody predictor (pacing, pitch, energy).
  - Measured across the 510 rows, timbre varies ~15% while prosody varies ~55%.

So we optimize one shared timbre offset plus a smooth low-rank prosody profile
over the length axis — a few hundred parameters instead of 130,560.

Usage:
    # Prove the method works by recovering a voice we already know
    uv run python invert_voice.py --sanity-check af_bella.pt

    # Invert real reference audio
    uv run python invert_voice.py --target my_voice.wav --base voices/af_heart.pt
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*RNN module weights are not part of single contiguous chunk.*", category=UserWarning)

from core import SpeechGenerator, pitch
from core.differentiable_kokoro import DifferentiableKokoro
from core.perceptual_loss import (
    F0_DENSE_QUANTILES,
    FRAME_HOP,
    KOKORO_SR,
    WavLMPooledLoss,
    internal_voiced_mask,
    duration_floor_loss,
    energy_range_loss,
    energy_range_reference,
    energy_sign_loss,
    noise_floor_loss,
    noise_floor_of,
    NOISE_FLOOR_MARGIN_DB,
    duration_spread_loss,
    f0_contour_loss,
    f0_contour_reference,
    creak_fraction_loss,
    creak_reference_fraction,
    pause_share_loss,
    punctuation_duration_loss,
    boundary_duration_loss,
    subharmonic_loss,
    subharmonic_reference,
    declination_loss,
    declination_reference,
    f0_distribution_loss,
    f0_range_loss,
    tremor_loss,
    tremor_reference,
    tremor_share,
    voiced_threshold_for,
    f0_reference_distribution,
    pacing_loss,
    speaking_rate,
)
from core.speaker_loss import (
    SpeakerEmbeddingLoss,
    build_manifold_bounds,
    manifold_penalty,
)

# Voices produced by this project rather than shipped with Kokoro. Excluded when
# measuring what counts as a plausible voice.
DERIVED_VOICES = {
    "af_kate_reading", "am_michael_kramer",
    "af_mica", "af_quartz", "af_amber", "am_granite", "am_slate", "am_ash",
}

VOICE_REGISTRY = Path(__file__).parent / "catalog" / "voice_registry.json"

PROJECT_ROOT = Path(__file__).parent
VOICES_DIR = PROJECT_ROOT / "voices"
OUTPUT_DIR = PROJECT_ROOT / "output"
SEED = 1234

# Varied lengths so the prosody profile is anchored at several points on the
# length axis rather than fitted at one and extrapolated everywhere else.
TRAIN_TEXTS = [
    "Hello, my name is Alex.",
    "Testing one two three.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "I think we should probably talk about this in the morning.",
    "Yes.",
    "It was the best of times, it was the worst of times, and nobody knew quite what to make of it.",
]

HELD_OUT_TEXT = "She sells seashells by the seashore on a bright summer day."

# A multi-sentence passage used only for duration constraints, never rendered.
# Kokoro indexes its style pack by phoneme count, and beyond the trained rows the
# prosody profile is held flat (see StyleParameterization.voice). Those rows
# encode length-appropriate pacing, so freezing a 100-phoneme row and using it
# for a 400-phoneme utterance hands a long passage a short passage's pause
# structure — measured as sentence breaks 30-39% shorter than the base voice,
# heard as rushing. Constraining durations here costs almost nothing because the
# vocoder never runs.
LONG_PACING_TEXT = (
    "The morning after the storm, the whole village came down to the water to "
    "see what had washed in, and nobody wanted to be first to speak. There were "
    "crates from a ship nobody recognized, half buried in the sand, and a long "
    "stretch of rope that ran out past the breakers. She had never seen the "
    "ocean look like that before, flat and grey and patient, as though it were "
    "waiting to be asked a question."
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_base_voice(reference_audio: np.ndarray, exclude: set[str]) -> str | None:
    """Choose the built-in voice that already sounds most like the reference.

    The manifold constraint caps how far a voice may travel, so starting far
    from the target spends that budget on getting to the right vicinity rather
    than on identity. A match started ~20% below the speaker's register came out
    27% high with its pitch asymmetry inverted; runs started near their target
    landed within a few percent.

    Selection is by speaker-embedding similarity, not register. Register alone
    picked a male voice for a female speaker and a Portuguese voice for an
    English one — it says nothing about who a voice sounds like. Candidates are
    restricted to English packs because the pipeline runs lang_code="a".

    Requires catalog/voice_registry.json (build_voice_registry.py).
    """
    if not VOICE_REGISTRY.exists():
        return None
    import json
    from resemblyzer import VoiceEncoder, preprocess_wav

    registry = json.loads(VOICE_REGISTRY.read_text())
    if not any("embedding" in e for e in registry.values()):
        return None

    encoder = VoiceEncoder(device="cpu", verbose=False)
    ref = encoder.embed_utterance(preprocess_wav(reference_audio, source_sr=KOKORO_SR))

    best, best_sim = None, -1.0
    for name, entry in registry.items():
        if name in exclude or "embedding" not in entry:
            continue
        if not name.startswith(("af_", "am_", "bf_", "bm_")):
            continue
        if not (VOICES_DIR / f"{name}.pt").exists():
            continue
        sim = float(np.dot(ref, np.array(entry["embedding"])))
        if sim > best_sim:
            best, best_sim = name, sim
    return best


def load_voice(path) -> torch.Tensor:
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


class StyleParameterization(torch.nn.Module):
    """Shared timbre offset + smooth low-rank prosody profile over the 510 rows.

    Timbre is one 128-vector applied to every row (justified by F3: timbre only
    drifts ~15% across rows). Prosody gets a small cosine basis over the length
    axis so it varies smoothly with utterance length instead of independently
    per row — 510 free rows would be unconstrained by the handful of lengths we
    actually train on.
    """

    def __init__(self, base_voice: torch.Tensor, n_basis: int = 6, device: str = "cpu",
                 row_range: tuple[int, int] | None = None):
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be at least 1")
        self.device = device
        self.n_rows = base_voice.shape[0]
        self.base = base_voice.reshape(self.n_rows, 256).float().to(device)

        self.timbre_delta = torch.nn.Parameter(torch.zeros(128, device=device))
        self.prosody_coeffs = torch.nn.Parameter(torch.zeros(n_basis, 128, device=device))

        # Cosine (DCT-II style) basis over the normalized length axis.
        t = torch.linspace(0, 1, self.n_rows, device=device).unsqueeze(1)
        k = torch.arange(n_basis, device=device).unsqueeze(0)
        basis = torch.cos(np.pi * t * k)                          # [510, n_basis]

        # Only rows matching a training length are constrained; beyond them the
        # cosines keep curving, which drifts pitch on utterances longer or
        # shorter than anything seen. Measured: a profile trained on rows 30-180
        # rendered 97 Hz inside that span and 110 Hz at row 333. Holding the
        # basis flat outside the trained range removes that failure mode while
        # leaving the profile free where there is data.
        if row_range is not None:
            lo_row, hi_row = row_range
            lo_row = max(0, min(lo_row, self.n_rows - 1))
            hi_row = max(lo_row, min(hi_row, self.n_rows - 1))
            basis[:lo_row] = basis[lo_row]
            basis[hi_row + 1:] = basis[hi_row]
            self.row_range = (lo_row, hi_row)
        else:
            self.row_range = (0, self.n_rows - 1)
        self.register_buffer("basis", basis)

    def voice(self) -> torch.Tensor:
        """Full [510, 256] style tensor with current offsets applied.

        Outside the trained row range the whole prosody row is copied from the
        nearest trained row, not just the offset. Holding the offset flat is not
        enough: the final row is `base[row] + offset`, and the base voice's own
        prosody varies ~55% across rows, so beyond the trained range an offset
        fitted for one base row was being applied to a different one. Measured
        effect was severe — utterances past the trained range ran 46% to 118%
        too fast while trained lengths were accurate.
        """
        timbre = self.base[:, :128] + self.timbre_delta.unsqueeze(0)
        prosody = self.base[:, 128:] + (self.basis @ self.prosody_coeffs)
        lo_row, hi_row = self.row_range
        if lo_row > 0:
            prosody = torch.cat([prosody[lo_row:lo_row + 1].expand(lo_row, -1),
                                 prosody[lo_row:]], dim=0)
        if hi_row < self.n_rows - 1:
            prosody = torch.cat([prosody[:hi_row + 1],
                                 prosody[hi_row:hi_row + 1].expand(
                                     self.n_rows - hi_row - 1, -1)], dim=0)
        return torch.cat([timbre, prosody], dim=-1)

    def row(self, phoneme_count: int) -> torch.Tensor:
        """The [1, 256] style vector Kokoro would use for this utterance length."""
        idx = min(max(phoneme_count - 1, 0), self.n_rows - 1)
        return self.voice()[idx].unsqueeze(0)

    def as_voice_tensor(self) -> torch.Tensor:
        """Back to Kokoro's on-disk [510, 1, 256] layout."""
        return self.voice().detach().unsqueeze(1).cpu()

    def magnitude(self) -> tuple[float, float]:
        return float(self.timbre_delta.abs().mean()), float(
            (self.basis @ self.prosody_coeffs).abs().mean()
        )


def synth_numpy(diff, ctx, ref_s, seed=SEED) -> np.ndarray:
    torch.manual_seed(seed)
    with torch.no_grad():
        return diff.forward(ctx, ref_s).audio.detach().cpu().numpy()


def invert(
    diff: DifferentiableKokoro,
    pipeline,
    params: StyleParameterization,
    perceptual: WavLMPooledLoss,
    target_stats: torch.Tensor | dict[str, torch.Tensor],
    target_rates: dict[str, float] | None,
    steps: int,
    lr: float,
    pacing_weight: float,
    reg_weight: float,
    device: str,
    train_texts: list[str] | None = None,
    speaker: SpeakerEmbeddingLoss | None = None,
    target_embed: torch.Tensor | None = None,
    speaker_weight: float = 0.0,
    f0_target: list[float] | None = None,
    pitch_band: tuple[float, float] = (50.0, 400.0),
    f0_weight: float = 0.0,
    f0_range_weight: float = 0.0,
    tremor_weight: float = 0.0,
    tremor_bound: float | None = None,
    tremor_ref_share: float | None = None,
    contour_target: tuple[float, float] | None = None,
    contour_weight: float = 0.0,
    creak_target: tuple[float, float] | None = None,
    creak_weight: float = 0.0,
    declination_target: float | None = None,
    declination_weight: float = 0.0,
    energy_target: float | None = None,
    energy_weight: float = 0.0,
    duration_weight: float = 0.0,
    duration_min_frames: float = 2.0,
    pause_weight: float = 0.0,
    pause_reference: float | None = None,
    pause_base: float | None = None,
    duration_spread_weight: float = 0.0,
    noise_floor_weight: float = 0.0,
    punctuation_weight: float = 0.0,
    subharmonic_weight: float = 0.0,
    subharmonic_bound: tuple[float, float] | None = None,
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    ground_truth: torch.Tensor | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 5,
    out_path: Path | None = None,
) -> None:
    """Adam on the style parameters. Hyperparameters follow arXiv 2607.25351."""
    contexts = []
    for text in (train_texts or TRAIN_TEXTS):
        ps = diff.phonemize(pipeline, text)
        contexts.append((text, ps, diff.build_context(ps)))
    log(f"Training on {len(contexts)} texts, phoneme lengths: "
        f"{sorted(len(ps) for _, ps, _ in contexts)}")

    # Pause share of the *starting* voice, per text. The target is where the
    # base voice already puts its silence, not a measurement of the reference
    # recording — see `pause_share_loss`.
    # Duration spread of the starting voice, per text — a regularizer against
    # flattening the rhythm to hit a speaking rate. See `duration_spread_loss`.
    # The base voice's own noise floor on each training text, plus slack. A
    # fixed bound is wrong: measured on a 20 s sample the stock voices read -46
    # to -71 dB, but on these 2-6 s training utterances the same voices sit at
    # -75 to -141. v29's fixed -40 dB was 35-100 dB too lenient and the optimizer
    # parked exactly on it.
    # Base-voice duration for each punctuation class, per text. Sentence breaks
    # and commas are kept apart on purpose — see `punctuation_duration_loss`.
    punctuation_targets = {}
    boundary_targets = {}
    long_ctx = None
    pause_scale = 1.0
    if punctuation_weight > 0:
        # The base voice maps token duration to rendered silence; the reference
        # supplies the target. Targeting the base voice alone cannot reach it —
        # measured as silent gaps, one speaker pauses 525 ms at the median while
        # the voice fitted from his base pauses 319, and that base voice itself
        # only manages 325. See `punctuation_duration_loss`.
        # The base voice's own gaps, on these same training texts, measured by
        # the identical routine used on the reference.
        base_gaps = []
        for text, ps, ctx in contexts:
            torch.manual_seed(SEED)
            with torch.no_grad():
                bo = diff.forward(ctx, params.base[len(ps) - 1].unsqueeze(0))
            base_gaps.extend(pitch.silent_gaps(bo.audio.detach().cpu().numpy(),
                                               KOKORO_SR).tolist())
        pause_base = float(np.median(base_gaps)) if len(base_gaps) >= 3 else None
        if pause_reference is not None and pause_base:
            pause_scale = float(np.clip(pause_reference / pause_base, 0.75, 2.5))
            log(f"Pause scale toward the reference speaker: {pause_scale:.2f}x "
                f"(speaker {pause_reference:.0f}ms, base voice {pause_base:.0f}ms)")
        # The defect appears past the trained rows, so constrain a long passage
        # too. Duration-only, so no vocoder and no meaningful memory cost.
        long_ps = diff.phonemize(pipeline, LONG_PACING_TEXT)
        long_ctx = diff.build_context(long_ps)
        long_row = len(long_ps)
        with torch.no_grad():
            base_long = diff.forward(long_ctx, params.base[long_row - 1].unsqueeze(0),
                                     decode=False)
        dl = base_long.duration.squeeze()
        long_targets = []
        for m in (long_ctx.sentence_mask, long_ctx.comma_mask):
            mm = m[:dl.numel()]
            if int(mm.sum()):
                long_targets.append((m, float(dl[mm].mean())))
        sp = dl[long_ctx.phoneme_mask[:dl.numel()]]
        long_spread = float(sp.std() / sp.mean().clamp(min=1e-6))
        long_boundary = float(dl[0])
        log(f"Long-passage anchor: row {long_row}, base sentence break "
            f"{long_targets[0][1] * 600 / KOKORO_SR * 1000:.0f}ms, spread {long_spread:.2f}")
        for text, ps, ctx in contexts:
            torch.manual_seed(SEED)
            with torch.no_grad():
                base_out = diff.forward(ctx, params.base[len(ps) - 1].unsqueeze(0))
            d = base_out.duration.squeeze()
            entry = []
            for m in (ctx.sentence_mask, ctx.comma_mask):
                mm = m[:d.numel()]
                if int(mm.sum()):
                    entry.append((m, float(d[mm].mean())))
            punctuation_targets[text] = [(m, t * pause_scale) for m, t in entry]
            boundary_targets[text] = float(d[0])
        shown = [f"{t * 600 / KOKORO_SR * 1000:.0f}ms"
                 for e in punctuation_targets.values() for _, t in e]
        log("Punctuation length in the base voice: " + ", ".join(shown))

    noise_floor_bounds = {}
    if noise_floor_weight > 0:
        for text, ps, ctx in contexts:
            torch.manual_seed(SEED)
            with torch.no_grad():
                base_out = diff.forward(ctx, params.base[len(ps) - 1].unsqueeze(0))
            noise_floor_bounds[text] = noise_floor_of(base_out.audio) + NOISE_FLOOR_MARGIN_DB
        log("Noise-floor bound from the base voice: "
            + ", ".join(f"{v:.0f}dB" for v in noise_floor_bounds.values()))

    duration_spread_targets = {}
    if duration_spread_weight > 0:
        for text, ps, ctx in contexts:
            torch.manual_seed(SEED)
            with torch.no_grad():
                base_out = diff.forward(ctx, params.base[len(ps) - 1].unsqueeze(0))
            d = base_out.duration.squeeze()
            m = ctx.phoneme_mask[:d.numel()]
            sp = d[m]
            duration_spread_targets[text] = float(sp.std() / sp.mean().clamp(min=1e-6))
        log("Duration spread of the base voice: "
            + ", ".join(f"{v:.2f}" for v in duration_spread_targets.values()))

    pause_targets = {}
    if pause_weight > 0:
        for text, ps, ctx in contexts:
            torch.manual_seed(SEED)
            with torch.no_grad():
                d = params.base[len(ps) - 1].unsqueeze(0)
                base_out = diff.forward(ctx, d)
            dur = base_out.duration.squeeze()
            pm = ctx.pause_mask[:dur.numel()]
            pause_targets[text] = float(dur[pm].sum() / dur.sum()) if int(pm.sum()) else 0.0
        log("Pause share of the base voice: "
            + ", ".join(f"{100*v:.0f}%" for v in pause_targets.values()))

    opt = torch.optim.Adam(params.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=40, factor=0.5)

    # Resume if a checkpoint exists. These runs are ~1s/text/step on CPU, so
    # losing a long run to an interrupted process is expensive.
    start_step = 0
    best = float("inf")
    if checkpoint_path and checkpoint_path.exists():
        ck = torch.load(checkpoint_path, weights_only=False)
        params.load_state_dict(ck["params"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        best = ck.get("best", float("inf"))
        log(f"Resumed from {checkpoint_path.name} at step {start_step}")

    def save_checkpoint(step: int, best: float) -> None:
        if checkpoint_path is None:
            return
        torch.save(
            {"params": params.state_dict(), "opt": opt.state_dict(),
             "step": step, "best": best},
            checkpoint_path,
        )
        # Also keep the usable artifact current, so an interrupted run still
        # leaves a voice you can listen to.
        if out_path is not None:
            torch.save(params.as_voice_tensor(), out_path)

    # The F0 loss reads Kokoro's internal F0_pred, but the target was measured
    # with yin on a real waveform. Those disagree on identical audio by a factor
    # that varies with the voice (0.80-1.01 across the built-ins) and drifts as
    # the voice moves during training — am_michael starts at 0.99 and reached
    # 1.06 by the end of a run, so the loss reported success while the rendered
    # pitch sat 6% sharp. Re-measuring periodically closes that loop.
    f0_offset = [0.0] * len(F0_DENSE_QUANTILES)
    # Same cut for the loss and its calibration, or the two measure different
    # subsets of frames and the correction is meaningless.
    voiced_cut = voiced_threshold_for(f0_target) if f0_target else 30.0

    # The tremor bound comes from tracked pitch on a recording, but the loss
    # reads Kokoro's internal contour, which carries much more 3-10 Hz energy
    # because of voicing ramps — measured 42.7 against a bound of 0.06, i.e. the
    # term was 99% of the objective. Calibrate the two on the base voice, whose
    # internal contour and rendered audio can both be measured.
    if tremor_weight > 0 and tremor_ref_share is not None:
        ratios = []
        for text, ps, ctx in contexts:
            torch.manual_seed(SEED)
            with torch.no_grad():
                bo = diff.forward(ctx, params.base[len(ps) - 1].unsqueeze(0))
            rendered = tremor_reference(bo.audio.detach().cpu().numpy(), KOKORO_SR)
            internal = float(tremor_share(bo.f0_pred, voiced_cut))
            if rendered > 1e-6:
                ratios.append(internal / rendered)
        if ratios:
            scale = float(np.median(ratios))
            tremor_bound = tremor_ref_share * 1.15 * scale
            log(f"Tremor bound {100*tremor_bound:.1f}% internal "
                f"(reference {100*tremor_ref_share:.1f}% rendered, "
                f"internal/rendered {scale:.1f}x)")
        else:
            tremor_bound = None


    def recalibrate(f0_target):
        """Shift the F0 target so rendered pitch, not F0_pred, hits the mark.

        Averaged over every training length: the gap varies by row, so
        correcting from a single utterance leaves the other lengths off.
        """
        nonlocal f0_offset
        # One offset per quantile. A single scalar cannot correct this: measured
        # against two references, the low quantiles needed almost no correction
        # while the upper half ran 11-17% high. The error is a stretch, not a
        # shift, so it takes a per-quantile correction.
        rows = []
        for _, ps, ctx in contexts:
            torch.manual_seed(SEED)
            with torch.no_grad():
                out = diff.forward(ctx, params.row(len(ps)))
            f0 = out.f0_pred.squeeze()
            v = f0[internal_voiced_mask(f0, voiced_cut)]
            if v.numel() < 8:
                continue
            internal = np.quantile(np.log(v.detach().cpu().numpy()), F0_DENSE_QUANTILES)
            # Same tracker as the target, or the correction absorbs the
            # difference between two trackers instead of the difference between
            # internal and rendered pitch. Calibrating against yin was feeding
            # the octave errors this whole change exists to remove.
            rendered_track = pitch.track(out.audio.detach().cpu().numpy(), KOKORO_SR)
            if rendered_track.voiced.sum() < 10:
                continue
            rendered = np.quantile(np.log(rendered_track.values), F0_DENSE_QUANTILES)
            rows.append(internal - rendered)
        if not rows:
            return
        # Smooth the offset across quantile levels before using it. Each level
        # is estimated from the frames that land near it, and at p90/p98 on a
        # 2-6 s clip that is very few — so the tail offsets are dominated by
        # noise, which is exactly where `f0_range_loss` operates. Symptom: that
        # term finished at 0.00003 (satisfied) on a voice whose rendered p90 was
        # still 10% below target. A quadratic in the quantile level keeps the
        # real shape — the gap is a stretch, not a shift — while discarding
        # per-level jitter.
        raw = np.mean(rows, axis=0)
        levels = np.asarray(F0_DENSE_QUANTILES, dtype=np.float64)
        fit = np.polyfit(levels, raw, 2)
        f0_offset = list(np.polyval(fit, levels))
        return f0_offset

    for step in range(start_step, steps):
        if f0_target is not None and f0_weight > 0 and step % 10 == 0:
            off = recalibrate(f0_target)
            if off is not None and step % 30 == 0:
                log("  F0 calibration (rendered/internal) across distribution: "
                    + " ".join(f"{np.exp(-off[i]):.3f}" for i in (0, 6, 12, 18, 24)))
        # Full batch over every training text. Cycling one text per step is
        # high-variance SGD: loss levels differ several-fold between texts, so
        # individual steps chase whichever text came up rather than descending
        # the shared objective. Accumulating over all texts makes each step a
        # genuine descent direction.
        opt.zero_grad()
        totals = {"perceptual": 0.0, "pacing": 0.0, "speaker": 0.0, "f0": 0.0, "f0range": 0.0, "tremor": 0.0, "contour": 0.0, "creak": 0.0, "declination": 0.0, "pause": 0.0, "durspread": 0.0, "floor": 0.0, "punct": 0.0, "subharm": 0.0, "energy": 0.0, "dur": 0.0}
        total_loss = 0.0

        for text, ps, ctx in contexts:
            ref_s = params.row(len(ps))

            # Fixed seed pins the vocoder's random phase/noise so the loss
            # surface is deterministic (see F7); otherwise the optimizer partly
            # chases sampling noise.
            torch.manual_seed(SEED)
            out = diff.forward(ctx, ref_s)

            # Time-pooled WavLM stats are much more content-dependent than
            # "content-independent" implies: per-text against matched content
            # the self-loss is exactly 0, but a target averaged over different
            # texts leaves a residual reachable only by distorting the voice.
            # With matched transcripts, compare text-by-text.
            tgt = target_stats[text] if isinstance(target_stats, dict) else target_stats
            loss_perc = perceptual(out.audio.unsqueeze(0), tgt)
            loss = loss_perc
            totals["perceptual"] += float(loss_perc)

            # Rates are per-text: they vary substantially with utterance length,
            # so a global average is not a valid target for any single one.
            if target_rates and text in target_rates and pacing_weight > 0:
                lp = pacing_loss(out.duration, len(ps), target_rates[text])
                loss = loss + pacing_weight * lp
                totals["pacing"] += float(lp)

            # Speaker-embedding term. WavLM statistics and Resemblyzer optimize
            # different things — texture versus identity — and each alone admits
            # solutions the other rejects.
            if speaker is not None and target_embed is not None and speaker_weight > 0:
                ls = speaker(out.audio.unsqueeze(0), target_embed)
                loss = loss + speaker_weight * ls
                totals["speaker"] += float(ls)


            # Pitch distribution. Generated voices ran ~2x the reference's rate
            # of large pitch excursions, heard as spiking at word ends.
            if f0_target is not None and f0_weight > 0:
                # Aim the internal statistic at target + offset, so the audible
                # pitch lands on target.
                lf = f0_distribution_loss(out.f0_pred, f0_target, offset=f0_offset)
                loss = loss + f0_weight * lf
                totals["f0"] += float(lf)

                # The distribution mean cannot feel a tail error; this can.
                lr = f0_range_loss(out.f0_pred, f0_target, offset=f0_offset)
                loss = loss + f0_range_weight * lr
                totals["f0range"] += float(lr)

            if tremor_weight > 0 and tremor_bound is not None:
                lt = tremor_loss(out.f0_pred, tremor_bound, voiced_cut)
                loss = loss + tremor_weight * lt
                totals["tremor"] += float(lt)

            if contour_target is not None and contour_weight > 0:
                lct = f0_contour_loss(out.f0_pred, contour_target, voiced_cut)
                loss = loss + contour_weight * lct
                totals["contour"] += float(lct)

            if creak_target is not None and creak_weight > 0:
                lc = creak_fraction_loss(out.f0_pred, creak_target[0], creak_target[1])
                loss = loss + creak_weight * lc
                totals["creak"] += float(lc)

            if pause_weight > 0 and text in pause_targets:
                lpa = pause_share_loss(out.duration, ctx.pause_mask,
                                       pause_targets[text])
                loss = loss + pause_weight * lpa
                totals["pause"] += float(lpa)

            if declination_target is not None and declination_weight > 0:
                ldec = declination_loss(out.f0_pred, declination_target, voiced_cut)
                loss = loss + declination_weight * ldec
                totals["declination"] += float(ldec)

            if energy_target is not None and energy_weight > 0:
                # Deadband on how much loudness moves, plus a barrier against
                # the internal energy contour inverting. Both are zero on a
                # healthy voice, so neither can perturb one that is already right.
                le = (energy_range_loss(out.audio, energy_target)
                      + energy_sign_loss(out.n_pred))
                loss = loss + energy_weight * le
                totals["energy"] += float(le)

            if punctuation_weight > 0 and punctuation_targets.get(text):
                lpn = (punctuation_duration_loss(out.duration, punctuation_targets[text])
                       + boundary_duration_loss(out.duration, boundary_targets[text]))
                loss = loss + punctuation_weight * lpn
                totals["punct"] += float(lpn)

            if subharmonic_weight > 0 and subharmonic_bound is not None:
                lsh = subharmonic_loss(out.audio, subharmonic_bound[0],
                                       subharmonic_bound[1])
                loss = loss + subharmonic_weight * lsh
                totals["subharm"] += float(lsh)

            if noise_floor_weight > 0 and text in noise_floor_bounds:
                lnf = noise_floor_loss(out.audio, noise_floor_bounds[text])
                loss = loss + noise_floor_weight * lnf
                totals["floor"] += float(lnf)

            if duration_spread_weight > 0 and text in duration_spread_targets:
                lds = duration_spread_loss(out.duration, ctx.phoneme_mask,
                                           duration_spread_targets[text])
                loss = loss + duration_spread_weight * lds
                totals["durspread"] += float(lds)

            # Lengthen phonemes Kokoro swallows, without slowing everything.
            if duration_weight > 0:
                ld = duration_floor_loss(out.duration, ctx.phoneme_mask,
                                         duration_min_frames)
                loss = loss + duration_weight * ld
                totals["dur"] += float(ld)

            (loss / len(contexts)).backward()
            total_loss += float(loss) / len(contexts)

        if reg_weight > 0 and bounds is not None:
            # Hinge penalty on leaving the per-dimension range spanned by the
            # built-in voices. Zero inside the box, so it costs nothing until the
            # optimizer actually walks into territory Kokoro never saw — which is
            # where the croakiness and end-of-word pitch jumps come from.
            reg = manifold_penalty(params.voice(), bounds[0], bounds[1])
            (reg_weight * reg).backward()
            totals["reg"] = float(reg) * len(contexts)

        # Duration constraints at a long utterance length — once per step, not
        # per text, and backwarded on its own like the manifold penalty. The
        # defect lives past the trained rows, so it has to be measured there.
        # `decode=False` skips the vocoder, which is where the memory goes.
        if punctuation_weight > 0 and long_ctx is not None:
            lo = diff.forward(long_ctx, params.row(len(long_ctx.phonemes)),
                              decode=False)
            llp = (punctuation_duration_loss(lo.duration, long_targets)
                   + boundary_duration_loss(lo.duration, long_boundary)
                   + duration_spread_loss(lo.duration, long_ctx.phoneme_mask,
                                          long_spread))
            (punctuation_weight * llp).backward()
            totals["punct"] += float(llp) * len(contexts)

        torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0)
        opt.step()
        sched.step(total_loss)

        parts = {k: v / len(contexts) for k, v in totals.items() if v}
        if total_loss < best:
            best = total_loss

        if step % 5 == 0 or step == steps - 1:
            tm, pm = params.magnitude()
            extra = ""
            if bounds is not None:
                out_of_box = float(
                    ((params.voice() < bounds[0]) | (params.voice() > bounds[1])).float().mean()
                )
                extra = f" outside_manifold={100*out_of_box:.1f}%"
            if ground_truth is not None:
                err = float((params.voice() - ground_truth).abs().mean())
                base_err = float((params.base - ground_truth).abs().mean())
                extra += f" recov_err={err:.5f} (start {base_err:.5f})"
            msg = "  ".join(f"{k}={v:.5f}" for k, v in parts.items())
            log(f"  step {step:4d}/{steps}  loss={total_loss:.5f}  {msg}  "
                f"|timbre|={tm:.4f} |prosody|={pm:.4f}{extra}")

        if checkpoint_every and (step % checkpoint_every == 0 or step == steps - 1):
            save_checkpoint(step, best)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help="Reference audio (.wav). Repeatable.", action="append")
    ap.add_argument("--sanity-check", help="Built-in voice filename to recover (proof test)")
    ap.add_argument("--base", help="Starting voice. Defaults to the built-in whose "
                                   "register is closest to the reference.")
    ap.add_argument("--target-text", action="append",
                    help="Transcript for the corresponding --target, as a file path "
                         "or literal text. Repeat once per clip, in the same order. "
                         "Supplying these is strongly recommended: it makes the "
                         "content match and enables the pacing loss.")
    ap.add_argument("--steps", type=int, default=300)
    # The paper's 2e-4 is tuned for its own parameterization. Ours starts at a
    # zero offset and needs to travel ~0.08 mean-abs to reach a different voice,
    # which 2e-4 cannot cover in a few hundred steps.
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-basis", type=int, default=6, help="Prosody basis functions")
    # Was 0.1 against a pitch weight of 5.0, i.e. fifty times weaker, which left
    # speaking rate effectively unconstrained.
    ap.add_argument("--pacing-weight", type=float, default=2.0)
    ap.add_argument("--max-clip-seconds", type=float, default=10.0,
                    help="Refuse training clips longer than this. Peak memory is "
                         "roughly 0.75GB per second of audio, so a 15s clip needs "
                         "~11GB and will be killed by the OS mid-run.")
    ap.add_argument("--contour-weight", type=float, default=0.0,
                    help="Weight on matching frame-to-frame pitch movement — the "
                         "melody rather than the pitch histogram")
    ap.add_argument("--creak-weight", type=float, default=0.0,
                    help="Weight on keeping creak below the reference's own share "
                         "of frames under the register")
    ap.add_argument("--declination-weight", type=float, default=0.0,
                    help="Weight on matching how fast pitch falls across a phrase "
                         "(unreliable target -- see core/pitch.py)")
    ap.add_argument("--duration-spread-weight", type=float, default=0.0,
                    help="Weight on keeping phoneme-length variation where the "
                         "base voice had it")
    ap.add_argument("--tremor-weight", type=float, default=0.0,
                    help="Weight on keeping 3-10 Hz pitch wobble no worse than "
                         "the reference speaker's")
    ap.add_argument("--f0-range-weight", type=float, default=0.0,
                    help="Weight on keeping the top of the pitch range from "
                         "being compressed")
    ap.add_argument("--subharmonic-weight", type=float, default=0.0,
                    help="Weight on keeping energy below the fundamental down "
                         "(low-frequency graininess)")
    ap.add_argument("--punctuation-weight", type=float, default=0.0,
                    help="Weight on keeping pauses at punctuation as long as the "
                         "base voice made them")
    ap.add_argument("--noise-floor-weight", type=float, default=0.0,
                    help="Weight on keeping the gaps between words quiet")
    ap.add_argument("--pause-weight", type=float, default=0.0,
                    help="Weight on keeping the share of time spent in pauses "
                         "where the base voice had it")
    ap.add_argument("--energy-weight", type=float, default=0.0,
                    help="Weight on matching how much loudness varies")
    ap.add_argument("--speaker-weight", type=float, default=0.0,
                    help="Weight on the differentiable Resemblyzer speaker loss. "
                         "Complements the WavLM term, which optimizes texture "
                         "rather than identity. Note this makes Resemblyzer a "
                         "training target, so evaluate with something else too.")
    ap.add_argument("--f0-reference",
                    help="Audio to take the pitch target from. Use the original "
                         "unsegmented recording when training on split clips.")
    ap.add_argument("--f0-weight", type=float, default=0.0,
                    help="Weight on matching the reference's log-F0 mean and spread; "
                         "targets excess pitch excursions at word ends")
    ap.add_argument("--duration-weight", type=float, default=0.0,
                    help="Weight on lengthening under-allocated phonemes; targets "
                         "words that sound swallowed or rushed")
    ap.add_argument("--duration-min-frames", type=float, default=2.0,
                    help="Frames a speech phoneme should get at minimum (1 frame = 25ms)")
    ap.add_argument("--reg-margin", type=float, default=0.1,
                    help="Slack outside the built-in voice range, as a fraction of each dimension's span")
    ap.add_argument("--reg-weight", type=float, default=0.0,
                    help="Weight on the manifold hinge penalty — keeps style dims "
                         "inside the range spanned by the built-in voices")
    ap.add_argument("--device", default="cpu",
                    help="cpu is usually right — 82M params, and MPS backward has gaps")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "inverted_voice.pt"))
    ap.add_argument("--checkpoint", action="store_true", default=True,
                    help="Save resumable state next to --out (default on)")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--restart", action="store_true",
                    help="Ignore and delete any existing checkpoint")
    args = ap.parse_args()

    if not args.target and not args.sanity_check:
        ap.error("need --target or --sanity-check")

    OUTPUT_DIR.mkdir(exist_ok=True)

    log("Initializing Kokoro...")
    gen = SpeechGenerator(device=args.device)
    diff = DifferentiableKokoro(gen.pipeline.model)

    log("Loading WavLM...")
    perceptual = WavLMPooledLoss(device=args.device)

    if not args.base:
        ref_for_base = (librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)[0]
                        if args.f0_reference
                        else target_audio[0].detach().cpu().numpy())
        picked = pick_base_voice(ref_for_base, DERIVED_VOICES)
        if picked is None:
            log("No voice registry; defaulting to af_heart. "
                "Run build_voice_registry.py to enable automatic selection.")
            args.base = str(VOICES_DIR / "af_heart.pt")
        else:
            args.base = str(VOICES_DIR / f"{picked}.pt")
            log(f"Auto-selected base voice: {picked}")
    base_voice = load_voice(args.base)

    # --- Build target statistics ---
    ground_truth = None
    if args.sanity_check:
        # Recover a voice we already know the answer to. If this fails, the
        # method is broken; if it succeeds, it tells us the achievable ceiling
        # before real-world recording noise enters the picture.
        gt_path = VOICES_DIR / args.sanity_check
        gt_voice = load_voice(gt_path)
        ground_truth = gt_voice.reshape(gt_voice.shape[0], 256).float().to(args.device)
        log(f"Sanity check: recovering {gt_path.name} starting from {Path(args.base).name}")

        target_audio = []
        target_rates = {}
        train_texts = list(TRAIN_TEXTS)
        for text in train_texts:
            ps = diff.phonemize(gen.pipeline, text)
            ctx = diff.build_context(ps)
            clip = synth_numpy(diff, ctx, gt_voice[len(ps) - 1].to(args.device))
            target_audio.append(torch.from_numpy(clip).float().to(args.device))
            target_rates[text] = speaking_rate(len(ps), len(clip))
    else:
        target_audio = []
        target_rates = {}
        clips = []
        # Peak memory scales linearly with clip length — measured ~0.75GB per
        # second of audio, since the backward pass retains activations through
        # Kokoro's vocoder at 24kHz. A 14.6s clip needs ~11GB and gets killed by
        # the OS mid-run, which looks like an unexplained disappearance rather
        # than an error. Refuse it up front instead.
        keep_targets, keep_texts, skipped = [], [], []
        texts = args.target_text or [None] * len(args.target)
        for path, text in zip(args.target, texts):
            audio, _ = librosa.load(path, sr=KOKORO_SR, mono=True)
            seconds = len(audio) / KOKORO_SR
            if seconds > args.max_clip_seconds:
                skipped.append((Path(path).name, seconds))
                continue
            clips.append(torch.from_numpy(audio).float().to(args.device))
            keep_targets.append(path)
            keep_texts.append(text)
            log(f"  target: {Path(path).name}  {seconds:.1f}s")
        for name, sec in skipped:
            log(f"  SKIPPED {name} ({sec:.1f}s): over --max-clip-seconds "
                f"({args.max_clip_seconds:.0f}s, would need ~{sec*0.75:.1f}GB)")
        args.target = keep_targets
        if args.target_text:
            args.target_text = keep_texts
        if not clips:
            ap.error("no clips left after the length filter; raise --max-clip-seconds")

        if args.target_text:
            # Each clip is paired with its own transcript, by position. Pooled
            # WavLM stats are strongly content-dependent, so matching content
            # per clip collapses the loss floor from ~0.098 to 0 — the
            # difference between the optimizer having a reachable target and
            # distorting the voice chasing an average no single utterance can
            # produce.
            if len(args.target_text) != len(args.target):
                ap.error(
                    f"got {len(args.target)} --target but "
                    f"{len(args.target_text)} --target-text; each clip needs "
                    f"its own transcript, paired in order"
                )

            train_texts, target_audio = [], []
            for path, spec, clip in zip(args.target, args.target_text, clips):
                transcript = (Path(spec).read_text().strip()
                              if Path(spec).exists() else spec)
                ps = diff.phonemize(gen.pipeline, transcript)
                if len(ps) > 510:
                    log(f"  WARNING: {Path(path).name} transcript is {len(ps)} "
                        f"phonemes, over Kokoro's 510 limit; it will be "
                        f"truncated. Use a shorter clip.")
                if transcript in target_rates:
                    # Identical text means identical row and identical target;
                    # keeping both just doubles the cost of that one anchor.
                    log(f"  skipping {Path(path).name}: duplicate transcript")
                    continue
                train_texts.append(transcript)
                target_audio.append(clip)
                target_rates[transcript] = speaking_rate(len(ps), len(clip))
                log(f"  {Path(path).name}: {len(ps)} phonemes, "
                    f"{target_rates[transcript]:.1f} phonemes/sec")

            lengths = sorted({len(diff.phonemize(gen.pipeline, t)) for t in train_texts})
            log(f"  {len(train_texts)} clips, {len(lengths)} distinct lengths "
                f"({lengths[0]}-{lengths[-1]} phonemes)")
        else:
            # No transcript means no matched content and no phoneme count, so
            # neither the low loss floor nor the pacing term is available.
            target_audio = clips
            train_texts = list(TRAIN_TEXTS)
            log("  no --target-text: falling back to pooled stats over generic "
                "texts. Expect a worse result — supply a transcript if you can.")

    # The prosody profile is a cosine basis over the 510 rows, but only the rows
    # matching a training text's phoneme count are ever constrained. With more
    # basis functions than anchored lengths the rest is free to ring: a single
    # 229-phoneme transcript produced offsets of 0.86 at row 300 against a base
    # prosody magnitude of 0.13, which is what makes untrained lengths sound
    # broken. Cap the basis at the number of distinct anchored lengths — with one
    # training text that means a single constant offset, which is all the data
    # can actually support.
    n_anchors = len({len(diff.phonemize(gen.pipeline, t)) for t in train_texts})
    n_basis = min(args.n_basis, n_anchors)
    if n_basis < args.n_basis:
        log(f"Limiting prosody basis to {n_basis} ({n_anchors} distinct training "
            f"length(s); {args.n_basis} requested would be underdetermined)")
    train_rows = sorted(len(diff.phonemize(gen.pipeline, t)) - 1 for t in train_texts)
    params = StyleParameterization(base_voice, n_basis=n_basis, device=args.device,
                                   row_range=(train_rows[0], train_rows[-1]))
    log(f"Prosody profile free across rows {train_rows[0]}-{train_rows[-1]}, "
        f"held flat outside")

    log("Computing target statistics...")
    if args.sanity_check or args.target_text:
        # Matched content: one target per training text, so the loss floor is
        # genuinely zero rather than a content-mismatch residual.
        target_stats = {
            text: perceptual.target_stats(a.unsqueeze(0))
            for text, a in zip(train_texts, target_audio)
        }
    else:
        stats = [perceptual.target_stats(a.unsqueeze(0)) for a in target_audio]
        target_stats = torch.cat(stats, dim=0).mean(dim=0, keepdim=True)

    speaker = None
    target_embed = None
    if args.speaker_weight > 0:
        log("Loading Resemblyzer for the speaker loss...")
        speaker = SpeakerEmbeddingLoss(device=args.device)
        embeds = [speaker.target_embedding(a.unsqueeze(0)) for a in target_audio]
        target_embed = torch.cat(embeds, dim=0).mean(dim=0, keepdim=True)
        target_embed = target_embed / target_embed.norm(dim=1, keepdim=True)

    f0_target = None
    pitch_band = (50.0, 400.0)
    if args.f0_weight > 0:
        # Pool across every clip, not just the first. Individual clips vary a
        # lot: across one speaker's segments the per-clip median ranged 84-117 Hz
        # against 96 Hz for the whole recording, so taking clip zero aimed the
        # pitch target ~20% high and the result audibly missed. Concatenating
        # weights each clip by its length, which is what we want.
        if args.f0_reference:
            # Prefer the original unsegmented recording. Segments skew high
            # relative to the whole: pooling the clips gave 97.1 Hz where the
            # full recording is 96.3, and the result rendered ~10 Hz sharp at
            # every length.
            ref, _ = librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)
            log(f"F0 target from {Path(args.f0_reference).name} (full recording)")
        else:
            ref = np.concatenate([a.detach().cpu().numpy() for a in target_audio])
        ref_track = pitch.track(ref, KOKORO_SR)
        pitch_band = ref_track.band()
        log(f"Pitch tracked by {ref_track.tracker}: register "
            f"{ref_track.register():.0f} Hz, {100*ref_track.voiced.mean():.0f}% of "
            f"frames voiced, band {pitch_band[0]:.0f}-{pitch_band[1]:.0f} Hz")
        f0_target = f0_reference_distribution(ref, band=pitch_band)
        log(f"Voiced cut: {voiced_threshold_for(f0_target):.0f} Hz "
            f"(below this speaker's lowest real pitch)")
        log(f"F0 target: {len(f0_target)}-level distribution, "
            + " ".join(f"{np.exp(f0_target[i]):.0f}" for i in (0, 6, 12, 18, 24))
            + " Hz at p2/p26/p50/p74/p98")

    contour_target = None
    if args.contour_weight > 0 and args.f0_reference:
        ref_ct, _ = librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)
        contour_target = f0_contour_reference(ref_ct, band=pitch_band)
        log(f"Contour target: {contour_target[0]:.4f} per 12.5ms frame, "
            f"{contour_target[1]:.4f} per 150ms syllable "
            f"(ratio {contour_target[1] / max(contour_target[0], 1e-6):.1f})")

    creak_target = None
    if args.creak_weight > 0 and args.f0_reference:
        ref_c, _ = librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)
        creak_target = creak_reference_fraction(ref_c, band=pitch_band)
        log(f"Creak target: {100*creak_target[0]:.1f}% of frames below "
            f"{creak_target[1]:.0f} Hz")

    declination_target = None
    if args.declination_weight > 0 and args.f0_reference:
        ref_d, _ = librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)
        declination_target = declination_reference(ref_d)
        log(f"Declination target: {declination_target:+.3f} log-units of pitch "
            f"per second within a phrase")

    # Median silent gap in the reference recording, and in the base voice on the
    # training texts, so the punctuation target can be scaled from one to the
    # other. Both measured by identical code.
    pause_reference = pause_base = None
    if args.punctuation_weight > 0 and args.f0_reference:
        ref_p, _ = librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)
        gaps = pitch.silent_gaps(ref_p, KOKORO_SR)
        if len(gaps) >= 3:
            pause_reference = float(np.median(gaps))

    tremor_bound = tremor_ref_share = None
    if args.tremor_weight > 0 and args.f0_reference:
        ref_t, _ = librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)
        share = tremor_reference(ref_t)
        tremor_ref_share = share
        log(f"Reference tremor: {100*share:.1f}% of pitch modulation in 3-10 Hz")

    subharmonic_bound = None
    if args.subharmonic_weight > 0 and args.f0_reference:
        ref_sh, _ = librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)
        ref_f0 = float(np.median(pitch.track(ref_sh, KOKORO_SR).values))
        # Bound from the reference itself, with headroom. The base voice is not
        # a good source here: it is a different speaker at a different register.
        share = subharmonic_reference(ref_sh, ref_f0)
        subharmonic_bound = (ref_f0, max(share * 1.5, 0.004))
        log(f"Subharmonic bound: {100*subharmonic_bound[1]:.2f}% of energy below "
            f"{ref_f0*0.75:.0f} Hz (reference sits at {100*share:.2f}%)")

    energy_target = None
    if args.energy_weight > 0:
        ref_e = (librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)[0]
                 if args.f0_reference
                 else np.concatenate([a.detach().cpu().numpy() for a in target_audio]))
        energy_target = energy_range_reference(ref_e)
        log(f"Energy range target: {energy_target:.2f} dB spread of frame level")

    bounds = None
    if args.reg_weight > 0:
        # Only Kokoro's own voices define the manifold. Voices we generated sit
        # at its edge by construction, and reference-fitted ones were pushed
        # outward deliberately — including either widens the bounds the
        # constraint is supposed to enforce. Measured at 8% wider with six of
        # ours present, which is the constraint quietly loosening itself.
        lib = [load_voice(VOICES_DIR / f) for f in sorted(os.listdir(VOICES_DIR))
               if f.endswith(".pt") and f[:3] in ("af_", "am_", "bf_", "bm_",
                                                  "ef_", "em_", "ff_", "hf_",
                                                  "hm_", "if_", "im_", "jf_",
                                                  "jm_", "pf_", "pm_", "zf_",
                                                  "zm_")
               and f[:-3] not in DERIVED_VOICES]
        bounds = build_manifold_bounds(lib, margin=args.reg_margin)
        bounds = (bounds[0].to(args.device), bounds[1].to(args.device))
        log(f"Manifold bounds from {len(lib)} voices, margin {args.reg_margin}")

    log(f"Optimizing {sum(p.numel() for p in params.parameters())} parameters "
        f"({args.steps} steps, lr={args.lr}, device={args.device})...")
    t0 = time.time()
    ckpt = Path(args.out).with_suffix(".ckpt") if args.checkpoint else None
    if ckpt and args.restart and ckpt.exists():
        ckpt.unlink()
    invert(
        diff, gen.pipeline, params, perceptual, target_stats, target_rates,
        args.steps, args.lr, args.pacing_weight, args.reg_weight, args.device,
        train_texts=train_texts,
        speaker=speaker,
        target_embed=target_embed,
        speaker_weight=args.speaker_weight,
        f0_target=f0_target,
        f0_weight=args.f0_weight,
        f0_range_weight=args.f0_range_weight,
        tremor_weight=args.tremor_weight,
        tremor_bound=tremor_bound,
        tremor_ref_share=tremor_ref_share,
        pitch_band=pitch_band,
        contour_target=contour_target,
        contour_weight=args.contour_weight,
        creak_target=creak_target,
        creak_weight=args.creak_weight,
        declination_target=declination_target,
        declination_weight=args.declination_weight,
        pause_weight=args.pause_weight,
        duration_spread_weight=args.duration_spread_weight,
        noise_floor_weight=args.noise_floor_weight,
        punctuation_weight=args.punctuation_weight,
        pause_reference=pause_reference,
        pause_base=pause_base,
        subharmonic_weight=args.subharmonic_weight,
        subharmonic_bound=subharmonic_bound,
        energy_target=energy_target,
        energy_weight=args.energy_weight,
        duration_weight=args.duration_weight,
        duration_min_frames=args.duration_min_frames,
        bounds=bounds,
        ground_truth=ground_truth,
        checkpoint_path=ckpt,
        checkpoint_every=args.checkpoint_every,
        out_path=Path(args.out),
    )
    log(f"Done in {time.time()-t0:.1f}s")

    # --- Save + held-out preview ---
    out_voice = params.as_voice_tensor()
    torch.save(out_voice, args.out)
    log(f"Saved {args.out}")

    ps = diff.phonemize(gen.pipeline, HELD_OUT_TEXT)
    ctx = diff.build_context(ps)
    audio = synth_numpy(diff, ctx, params.row(len(ps)).detach())
    wav = Path(args.out).with_suffix(".wav")
    sf.write(str(wav), audio, KOKORO_SR)
    log(f"Held-out preview ({len(ps)} phonemes, unseen during optimization): {wav}")

    if ground_truth is not None:
        final = float((params.voice() - ground_truth).abs().mean())
        start = float((params.base - ground_truth).abs().mean())
        log(f"Recovery: {start:.5f} -> {final:.5f} mean abs error "
            f"({100*(1-final/start):.1f}% closer to ground truth)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
