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

from core import SpeechGenerator
from core.differentiable_kokoro import DifferentiableKokoro
from core.perceptual_loss import (
    F0_DENSE_QUANTILES,
    FRAME_HOP,
    KOKORO_SR,
    WavLMPooledLoss,
    duration_floor_loss,
    energy_dynamics_loss,
    energy_reference_cv,
    estimate_pitch_range,
    f0_distribution_loss,
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
    energy_target: float | None = None,
    energy_weight: float = 0.0,
    duration_weight: float = 0.0,
    duration_min_frames: float = 2.0,
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
            v = f0[f0 > 30.0]
            if v.numel() < 8:
                continue
            internal = np.quantile(np.log(v.detach().cpu().numpy()), F0_DENSE_QUANTILES)
            y = librosa.yin(out.audio.detach().cpu().numpy(),
                            fmin=pitch_band[0], fmax=pitch_band[1], sr=KOKORO_SR)
            voiced = y[(y > pitch_band[0] * 1.02) & (y < pitch_band[1] * 0.98)]
            if len(voiced) < 10:
                continue
            rendered = np.quantile(np.log(voiced), F0_DENSE_QUANTILES)
            rows.append(internal - rendered)
        if not rows:
            return
        f0_offset = list(np.mean(rows, axis=0))
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
        totals = {"perceptual": 0.0, "pacing": 0.0, "speaker": 0.0, "f0": 0.0, "energy": 0.0, "dur": 0.0}
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

            if energy_target is not None and energy_weight > 0:
                le = energy_dynamics_loss(out.n_pred, energy_target)
                loss = loss + energy_weight * le
                totals["energy"] += float(le)

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
        for path in args.target:
            audio, _ = librosa.load(path, sr=KOKORO_SR, mono=True)
            clips.append(torch.from_numpy(audio).float().to(args.device))
            log(f"  target: {Path(path).name}  {len(audio)/KOKORO_SR:.1f}s")

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
        pitch_band = estimate_pitch_range(ref)
        log(f"Pitch analysis band from reference: "
            f"{pitch_band[0]:.0f}-{pitch_band[1]:.0f} Hz")
        f0_target = f0_reference_distribution(ref, band=pitch_band)
        log(f"F0 target: {len(f0_target)}-level distribution, "
            + " ".join(f"{np.exp(f0_target[i]):.0f}" for i in (0, 6, 12, 18, 24))
            + " Hz at p2/p26/p50/p74/p98")

    energy_target = None
    if args.energy_weight > 0:
        ref_e = (librosa.load(args.f0_reference, sr=KOKORO_SR, mono=True)[0]
                 if args.f0_reference
                 else np.concatenate([a.detach().cpu().numpy() for a in target_audio]))
        energy_target = energy_reference_cv(ref_e)
        log(f"Energy variation target (coefficient of variation): {energy_target:.3f}")

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
        pitch_band=pitch_band,
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
