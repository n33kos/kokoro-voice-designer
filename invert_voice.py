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
    FRAME_HOP,
    KOKORO_SR,
    WavLMPooledLoss,
    pacing_loss,
    speaking_rate,
)

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

    def __init__(self, base_voice: torch.Tensor, n_basis: int = 6, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.n_rows = base_voice.shape[0]
        self.base = base_voice.reshape(self.n_rows, 256).float().to(device)

        self.timbre_delta = torch.nn.Parameter(torch.zeros(128, device=device))
        self.prosody_coeffs = torch.nn.Parameter(torch.zeros(n_basis, 128, device=device))

        # Cosine (DCT-II style) basis over the normalized length axis.
        t = torch.linspace(0, 1, self.n_rows, device=device).unsqueeze(1)
        k = torch.arange(n_basis, device=device).unsqueeze(0)
        self.register_buffer("basis", torch.cos(np.pi * t * k))  # [510, n_basis]

    def voice(self) -> torch.Tensor:
        """Full [510, 256] style tensor with current offsets applied."""
        timbre = self.base[:, :128] + self.timbre_delta.unsqueeze(0)
        prosody = self.base[:, 128:] + (self.basis @ self.prosody_coeffs)
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

    for step in range(start_step, steps):
        # Full batch over every training text. Cycling one text per step is
        # high-variance SGD: loss levels differ several-fold between texts, so
        # individual steps chase whichever text came up rather than descending
        # the shared objective. Accumulating over all texts makes each step a
        # genuine descent direction.
        opt.zero_grad()
        totals = {"perceptual": 0.0, "pacing": 0.0}
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

            (loss / len(contexts)).backward()
            total_loss += float(loss) / len(contexts)

        if reg_weight > 0:
            # Keep the style vector near the built-in voice manifold; gradient
            # descent will happily walk into regions Kokoro never saw.
            reg = params.timbre_delta.pow(2).mean() + params.prosody_coeffs.pow(2).mean()
            (reg_weight * reg).backward()
            totals["reg"] = float(reg)

        torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0)
        opt.step()
        sched.step(total_loss)

        parts = {k: v / len(contexts) for k, v in totals.items() if v}
        if total_loss < best:
            best = total_loss

        if step % 5 == 0 or step == steps - 1:
            tm, pm = params.magnitude()
            extra = ""
            if ground_truth is not None:
                err = float((params.voice() - ground_truth).abs().mean())
                base_err = float((params.base - ground_truth).abs().mean())
                extra = f" recov_err={err:.5f} (start {base_err:.5f})"
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
    ap.add_argument("--base", default=str(VOICES_DIR / "af_heart.pt"), help="Starting voice")
    ap.add_argument("--target-text", help="Transcript of --target audio; enables the pacing loss")
    ap.add_argument("--steps", type=int, default=300)
    # The paper's 2e-4 is tuned for its own parameterization. Ours starts at a
    # zero offset and needs to travel ~0.08 mean-abs to reach a different voice,
    # which 2e-4 cannot cover in a few hundred steps.
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-basis", type=int, default=6, help="Prosody basis functions")
    ap.add_argument("--pacing-weight", type=float, default=0.1)
    ap.add_argument("--reg-weight", type=float, default=0.0)
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

    base_voice = load_voice(args.base)
    params = StyleParameterization(base_voice, n_basis=args.n_basis, device=args.device)

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
        for path in args.target:
            audio, _ = librosa.load(path, sr=KOKORO_SR, mono=True)
            target_audio.append(torch.from_numpy(audio).float().to(args.device))
            log(f"  target: {Path(path).name}  {len(audio)/KOKORO_SR:.1f}s")

        if args.target_text:
            # Train on the reference clip's own transcript. Pooled WavLM stats
            # are strongly content-dependent, so matching content collapses the
            # loss floor from ~0.098 to 0 — the difference between the optimizer
            # having a reachable target and distorting the voice chasing an
            # average no single utterance can produce.
            transcript = Path(args.target_text).read_text().strip() \
                if Path(args.target_text).exists() else args.target_text
            train_texts = [transcript]
            ps = diff.phonemize(gen.pipeline, transcript)
            rate = speaking_rate(len(ps), len(target_audio[0]))
            target_rates = {transcript: rate}
            log(f"  transcript: {len(ps)} phonemes, "
                f"measured rate {rate:.2f} phonemes/sec")
            if len(ps) > 510:
                log(f"  WARNING: transcript exceeds Kokoro's 510-phoneme limit "
                    f"and will be truncated; use a shorter clip")
        else:
            # No transcript means no matched content and no phoneme count, so
            # neither the low loss floor nor the pacing term is available.
            train_texts = list(TRAIN_TEXTS)
            log("  no --target-text: falling back to pooled stats over generic "
                "texts. Expect a worse result — supply a transcript if you can.")

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
