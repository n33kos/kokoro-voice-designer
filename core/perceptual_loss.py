"""Differentiable perceptual losses for gradient-based voice inversion.

Two spectral-matching losses were tried and removed. A mean-normalized log-mel
LTAS minimized its own objective (0.466 -> 0.006) while moving the zero-crossing
rate it was meant to fix in the wrong direction, and a band-energy-ratio variant
ranked candidates opposite to how they actually sounded. Matching the pitch
distribution (`f0_stats_loss`) fixed the audible problem instead, and improved
spectral balance as a side effect.

The primary objective is a WavLM pooled-statistics loss: take layer-4 features
of the generated and target audio, collapse the time axis into per-dimension
mean and std, and compare with MSE. Because time is pooled away the loss is
*content-independent* — the generated speech doesn't have to say what the
reference clip says, so no transcript and no alignment are needed.

Method from "Extracting Voice Styles from Frozen TTS Models via Gradient-Based
Inverse Optimization" (arXiv 2607.25351).

Everything here stays on the autograd graph, including the 24kHz -> 16kHz
resample, so gradients reach Kokoro's style vector.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

KOKORO_SR = 24000
WAVLM_SR = 16000
DEFAULT_LAYER = 4
# Samples of audio the decoder emits per predicted duration frame. Verified
# empirically across several utterance lengths — exactly 600 every time.
FRAME_HOP = 600


def _sinc_kernel(ratio: float, half_width: int = 16, device=None) -> torch.Tensor:
    """Windowed-sinc lowpass for band-limited resampling.

    `F.interpolate` would alias badly on speech; this keeps the resample honest
    without pulling in torchaudio.
    """
    cutoff = min(1.0, ratio)
    n = torch.arange(-half_width, half_width + 1, dtype=torch.float32, device=device)
    kernel = cutoff * torch.sinc(cutoff * n)
    # Blackman window
    m = 2 * half_width
    k = torch.arange(0, m + 1, dtype=torch.float32, device=device)
    window = 0.42 - 0.5 * torch.cos(2 * math.pi * k / m) + 0.08 * torch.cos(4 * math.pi * k / m)
    kernel = kernel * window
    return kernel / kernel.sum()


def resample_24k_to_16k(audio: torch.Tensor) -> torch.Tensor:
    """Differentiable 3:2 decimation, band-limited.

    24000/16000 = 3/2, so upsample by 2, lowpass, then take every 3rd sample.
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    x = audio.unsqueeze(1)  # [B, 1, T]

    # Zero-stuff upsample by 2
    b, _, t = x.shape
    up = torch.zeros(b, 1, t * 2, device=x.device, dtype=x.dtype)
    up[:, :, ::2] = x * 2.0

    kernel = _sinc_kernel(0.5, half_width=16, device=x.device).to(x.dtype)
    kernel = kernel.view(1, 1, -1)
    filtered = F.conv1d(up, kernel, padding=kernel.shape[-1] // 2)

    return filtered[:, :, ::3].squeeze(1)


class WavLMPooledLoss(torch.nn.Module):
    """Time-pooled WavLM feature statistics, compared with MSE.

    Pooling over time is what makes this content-independent: we compare *how a
    voice sounds* rather than *what it says*, so target clips need no transcript
    and can be any length or content.
    """

    def __init__(self, device: str = "cpu", layer: int = DEFAULT_LAYER,
                 model_name: str = "microsoft/wavlm-large"):
        super().__init__()
        from transformers import WavLMModel

        self.layer = layer
        self.device = device
        self.model = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def features(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """[B, T] at 16kHz -> [B, frames, dim] hidden states from `self.layer`."""
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)
        # wavlm-large was trained with zero-mean/unit-var normalized input
        x = audio_16k - audio_16k.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-7)
        out = self.model(x, output_hidden_states=True)
        return out.hidden_states[self.layer]

    def pooled_stats(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """Kokoro-rate audio -> [B, 2*dim] concatenated per-dimension mean and std."""
        audio_16k = resample_24k_to_16k(audio_24k)
        feats = self.features(audio_16k)
        mean = feats.mean(dim=1)
        std = feats.std(dim=1)
        return torch.cat([mean, std], dim=-1)

    @torch.no_grad()
    def target_stats(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """Pooled stats for reference audio. Averaged if several clips are given."""
        stats = self.pooled_stats(audio_24k)
        return stats.mean(dim=0, keepdim=True)

    def forward(self, generated_24k: torch.Tensor, target_stats: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.pooled_stats(generated_24k), target_stats)


def estimate_pitch_range(audio: np.ndarray, sr: int = KOKORO_SR) -> tuple[float, float]:
    """Derive a pitch-analysis band centred on the speaker's own register.

    A fixed 50-400 Hz band silently clips anyone outside it, biasing every
    quantile downstream. But simply widening it is worse, not better: a 40-600 Hz
    band lets yin make octave errors, which land in the tails where the quantiles
    live. Measured on one reference, widening moved the 90th percentile from a
    correct 160 Hz to 520 Hz — the narrow band had been suppressing octave errors
    as much as it had been clipping.

    So estimate the register first, with a median (robust to exactly those tail
    errors), then bracket it by a factor covering the range a speaker actually
    uses. The result adapts to the voice while keeping a prior tight enough to
    reject octave confusion.

    The returned band must be reused for *every* comparison in a run. Measuring a
    reference and a generated clip through different bands makes the two numbers
    incomparable.
    """
    import librosa

    coarse = librosa.yin(audio, fmin=40.0, fmax=600.0, sr=sr)
    voiced = coarse[(coarse > 42) & (coarse < 580)]
    if len(voiced) < 20:
        return 50.0, 400.0
    register = float(np.median(voiced))
    return max(40.0, register / 2.5), min(700.0, register * 2.5)


# Quantiles pinned by the pitch loss. The middle one fixes the speaker's register;
# the outer ones fix how far the voice travels above and below it, separately.
F0_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def f0_reference_quantiles(audio: np.ndarray, sr: int = KOKORO_SR,
                           band: tuple[float, float] | None = None) -> list[float]:
    """Log-F0 quantiles of a reference recording — the pitch-loss target.

    Replaces matching mean and standard deviation, which cannot represent these
    distributions. Speech pitch is skewed, and the direction differs per speaker:
    one measured voice sits at a 78.6 Hz register and rises to 160 for emphasis
    (2.04x up, 1.23x down), while another sits at 159 Hz and dips *down* instead
    (1.22x up, 1.96x down). A symmetric spread around a mean describes neither,
    and the mean itself sits well above the register a listener hears — 96.3 Hz
    against a 78.6 Hz mode for the first voice.

    Quantiles capture register and asymmetry together, and unlike a modal
    estimate they have a differentiable counterpart in torch.
    """
    import librosa

    lo, hi = band if band is not None else estimate_pitch_range(audio, sr)
    f0 = librosa.yin(audio, fmin=lo, fmax=hi, sr=sr)
    voiced = f0[(f0 > lo * 1.02) & (f0 < hi * 0.98)]
    if len(voiced) == 0:
        return [0.0] * len(F0_QUANTILES)
    return [float(q) for q in np.quantile(np.log(voiced), F0_QUANTILES)]


# Quantile levels for distribution matching. Dense enough that the *shape*
# between landmarks is constrained, not just the landmarks: with five levels a
# match had near-perfect quantiles while its density was bimodal with a hole in
# the middle — a third of frames piled into vocal fry below the register and a
# bulge above it, audible as thin and croaky.
F0_DENSE_QUANTILES = tuple(round(0.02 + 0.96 * i / 24, 4) for i in range(25))


def f0_reference_distribution(audio: np.ndarray, sr: int = KOKORO_SR,
                              band: tuple[float, float] | None = None) -> list[float]:
    """Densely sampled log-F0 quantiles — the distribution-matching target."""
    import librosa

    lo, hi = band if band is not None else estimate_pitch_range(audio, sr)
    f0 = librosa.yin(audio, fmin=lo, fmax=hi, sr=sr)
    voiced = f0[(f0 > lo * 1.02) & (f0 < hi * 0.98)]
    if len(voiced) == 0:
        return [0.0] * len(F0_DENSE_QUANTILES)
    return [float(q) for q in np.quantile(np.log(voiced), F0_DENSE_QUANTILES)]


def voiced_threshold_for(target: list[float], factor: float = 0.28) -> float:
    """Voicing cut scaled to the speaker, in Hz.

    A fixed 30 Hz cut is 0.39x the register of a low voice but only 0.24x that of
    a higher one, so it admits far more sub-register junk the higher the speaker.
    Measured: 2% of counted frames sat below 0.6x register for a 77 Hz voice
    against 21% for a 126 Hz voice — and the loss then faithfully matched
    statistics computed over that junk, which is audible as excess creak.

    Kept deliberately low (~0.28x register). A higher cut removes creak frames
    from the distribution match, which stops constraining them rather than
    fixing them — see `creak_fraction_loss`, which targets their quantity
    instead. This cut exists only to drop unvoiced frames near 0 Hz.

    Keyed to the target's *median*, not its low tail — the low tail of the
    reference is itself partly creak, so scaling from it reproduces the problem.
    Half the median is roughly an octave down, below which a frame is far more
    likely to be sub-harmonic error than speech.
    """
    if not target:
        return 30.0
    median = np.exp(target[len(target) // 2])
    return float(median * factor)


def creak_reference_fraction(audio: np.ndarray, sr: int = KOKORO_SR,
                             band: tuple[float, float] | None = None,
                             factor: float = 0.6) -> tuple[float, float]:
    """How much of a reference sits below `factor` x its own register.

    Returns (fraction, threshold_hz). Real speech contains some creak — measured
    5-12% across three references — so the target is to *match* that amount, not
    to eliminate it.
    """
    import librosa

    lo, hi = band if band is not None else estimate_pitch_range(audio, sr)
    f0 = librosa.yin(audio, fmin=lo, fmax=hi, sr=sr)
    voiced = f0[(f0 > lo * 1.02) & (f0 < hi * 0.98)]
    if len(voiced) < 20:
        return 0.0, 0.0
    threshold = float(np.median(voiced) * factor)
    return float(np.mean(voiced < threshold)), threshold


def creak_fraction_loss(f0_pred: torch.Tensor, target_fraction: float,
                        threshold_hz: float, floor_hz: float = 20.0,
                        sharpness: float = 0.15) -> torch.Tensor:
    """Match the proportion of frames sitting below the creak threshold.

    Added because raising the voicing cut to exclude sub-register frames from the
    distribution match stopped *penalising* them: excluded frames are
    unconstrained, and the low-frame share rose from 5% to 14% (Michael) and 12%
    to 22% (Kate) once they were no longer measured. Constraining the share
    directly is the correction — the frames stay visible to the loss and their
    quantity is targeted.

    A sigmoid rather than a hard count, so it is differentiable.
    """
    f0 = f0_pred.squeeze()
    speech = f0[f0 > floor_hz]
    if speech.numel() < 8:
        return f0.sum() * 0.0
    below = torch.sigmoid((threshold_hz - speech) * sharpness)
    return (below.mean() - target_fraction) ** 2


def f0_distribution_loss(f0_pred: torch.Tensor, target: list[float],
                         offset: list[float] | None = None,
                         voiced_threshold: float | None = None) -> torch.Tensor:
    """Match the whole log-F0 distribution, not five points on it.

    Equivalent to a quantile-function (inverse-CDF) distance: comparing sorted
    distributions at many levels constrains density everywhere, so the optimizer
    cannot satisfy the landmarks while leaving a gap between them.

    `offset` is a per-level correction for the gap between Kokoro's internal
    F0_pred and pitch measured on rendered audio. That gap is a stretch, not a
    shift — measured 0.68x at the bottom of the distribution and 1.09x at the
    top — so it must be applied per level.
    """
    if voiced_threshold is None:
        voiced_threshold = voiced_threshold_for(target)
    f0 = f0_pred.squeeze()
    voiced = f0[f0 > voiced_threshold]
    if voiced.numel() < 8:
        return f0.sum() * 0.0
    log_f0 = torch.log(voiced.clamp(min=1.0))
    qs = torch.tensor(F0_DENSE_QUANTILES, device=f0.device, dtype=log_f0.dtype)
    got = torch.quantile(log_f0, qs)
    want = torch.tensor(target, device=f0.device, dtype=log_f0.dtype)
    if offset is not None:
        want = want + torch.tensor(offset, device=f0.device, dtype=log_f0.dtype)
    return ((got - want) ** 2).mean()


def f0_quantile_loss(f0_pred: torch.Tensor, target: list[float],
                     offset: float | list[float] = 0.0,
                     voiced_threshold: float = 30.0,
                     sharpness: float = 0.2) -> torch.Tensor:
    """Match the log-F0 quantiles of `F0_pred` to a reference.

    `offset` shifts the whole target to compensate for the gap between Kokoro's
    internal F0_pred and the pitch measured on rendered audio, which varies by
    voice and drifts during training.

    Unvoiced frames sit near 0 Hz and would dominate the low quantiles, so they
    are dropped by a hard threshold here rather than the soft mask used for
    moments — quantiles need an actual subset, and the threshold is far from any
    real pitch so no gradient signal is lost.
    """
    f0 = f0_pred.squeeze()
    voiced = f0[f0 > voiced_threshold]
    if voiced.numel() < 8:
        return f0.sum() * 0.0
    log_f0 = torch.log(voiced.clamp(min=1.0))
    qs = torch.tensor(F0_QUANTILES, device=f0.device, dtype=log_f0.dtype)
    got = torch.quantile(log_f0, qs)
    want = torch.tensor(target, device=f0.device, dtype=log_f0.dtype)
    off = torch.tensor(offset if isinstance(offset, (list, tuple)) else [offset] * len(target),
                       device=f0.device, dtype=log_f0.dtype)
    want = want + off
    return ((got - want) ** 2).mean()


def f0_reference_stats(audio: np.ndarray, sr: int = KOKORO_SR) -> tuple[float, float]:
    """Median and spread of log-F0 in a reference recording.

    Returns (mean_log_f0, std_log_f0) over voiced frames. Computed with librosa
    on real audio, so no gradients are involved — this is only the target.
    """
    import librosa

    f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr)
    voiced = f0[(f0 > 55) & (f0 < 350)]
    if len(voiced) == 0:
        return 0.0, 0.0
    log_f0 = np.log(voiced)
    return float(log_f0.mean()), float(log_f0.std())


def f0_stats_loss(f0_pred: torch.Tensor, target_mean: float, target_std: float,
                  voiced_threshold: float = 30.0, sharpness: float = 0.2) -> torch.Tensor:
    """Match the pitch distribution of `F0_pred` to a reference, in log space.

    Motivation: measured against a real recording, generated voices had roughly
    double the reference's rate of large pitch excursions — 25-31% of voiced
    frames above 1.5x median versus 12.3% for the reference — which is audible as
    the pitch spiking at the end of words. Matching mean *and* spread of log-F0
    constrains that directly, where the perceptual and speaker losses do not.

    Log space because pitch is perceived multiplicatively. The voiced mask is a
    sigmoid rather than a hard threshold so it stays differentiable; unvoiced
    frames sit at ~0 Hz and would otherwise dominate the statistics.
    """
    f0 = f0_pred.squeeze()
    weights = torch.sigmoid((f0 - voiced_threshold) * sharpness)
    total = weights.sum().clamp(min=1e-6)

    log_f0 = torch.log(f0.clamp(min=1.0))
    mean = (weights * log_f0).sum() / total
    var = (weights * (log_f0 - mean).pow(2)).sum() / total
    std = var.clamp(min=1e-8).sqrt()

    return (mean - target_mean) ** 2 + (std - target_std) ** 2


def energy_reference_cv(audio: np.ndarray, sr: int = KOKORO_SR) -> float:
    """Coefficient of variation of frame energy in a reference recording.

    Deliberately dimensionless. Kokoro's `N_pred` is an internal energy contour
    on no particular scale, so matching its absolute level against a waveform
    measurement would repeat the units mismatch that made the pitch loss report
    success while sounding wrong. A ratio of spread to mean is comparable across
    both domains.
    """
    import librosa

    rms = librosa.feature.rms(y=audio)[0]
    rms = rms[rms > np.percentile(rms, 10)]      # drop silence
    if len(rms) < 10 or float(rms.mean()) <= 0:
        return 0.0
    return float(rms.std() / rms.mean())


def energy_dynamics_loss(n_pred: torch.Tensor, target_cv: float) -> torch.Tensor:
    """Match how much the voice's loudness moves, relative to its own level.

    Energy variation is one of the twelve style dimensions but was only
    constrained indirectly, through the WavLM term. Unconstrained dimensions
    have been where every audible regression came from.
    """
    n = n_pred.squeeze()
    n = n[n > n.abs().mean() * 0.1]              # ignore near-silent frames
    if n.numel() < 10:
        return n_pred.sum() * 0.0
    cv = n.std() / n.mean().abs().clamp(min=1e-6)
    return (cv - target_cv) ** 2


def duration_floor_loss(duration: torch.Tensor, phoneme_mask: torch.Tensor,
                        min_frames: float = 2.0) -> torch.Tensor:
    """Penalize speech phonemes allocated too few frames.

    Kokoro's duration predictor is aggressive: on untouched `am_michael`, 24.7%
    of real phonemes get a single 25ms frame, which is far too short for a
    fricative (80-120ms is typical) and is audible as words being swallowed or
    rushed. Voice matching improves this incidentally (17.7% for v7) but doesn't
    target it.

    One-sided hinge on the *continuous* duration, before Kokoro rounds it — the
    only differentiable handle available. Costs nothing for phonemes already
    long enough, so it lengthens the swallowed ones rather than slowing
    everything down uniformly (which is what the pacing term would do).

    `min_frames` is in decoder frames; each is 600 samples, i.e. 25ms at 24kHz.
    """
    d = duration.squeeze()
    if phoneme_mask.shape[0] != d.shape[0]:
        n = min(phoneme_mask.shape[0], d.shape[0])
        d, phoneme_mask = d[:n], phoneme_mask[:n]
    shortfall = (min_frames - d).clamp(min=0) * phoneme_mask.float()
    return shortfall.pow(2).sum() / phoneme_mask.float().sum().clamp(min=1.0)


def speaking_rate(phoneme_count: int, n_samples: int, sr: int = KOKORO_SR) -> float:
    """Phonemes per second — the units the pacing loss works in."""
    return phoneme_count / (n_samples / sr)


def pacing_loss(duration: torch.Tensor, phoneme_count: int,
                target_rate: float, sr: int = KOKORO_SR) -> torch.Tensor:
    """Squared *relative* error on speaking rate, via Kokoro's continuous duration.

    `duration` is the pre-round tensor from `DifferentiableKokoro` — the only
    differentiable handle on pacing, since `round()` detaches everything after
    it. The decoder emits exactly 600 samples per predicted frame at 24kHz
    (measured, not assumed — see FRAME_HOP).

    Relative rather than absolute error, so the term is scale-free and stays
    comparable to the perceptual loss instead of dominating it: absolute rates
    are ~10-14 phonemes/sec, so squared absolute error lands in the tens while
    the perceptual loss sits around 0.1-0.3.

    Note `target_rate` must be the rate for *this* utterance. Rates vary
    substantially with length (measured 10.6 to 14.4 phonemes/sec across the
    same voice), so a single global average is not a valid target.
    """
    total_frames = duration.sum()
    seconds = total_frames * FRAME_HOP / sr
    rate = phoneme_count / seconds.clamp(min=1e-3)
    return (rate / target_rate - 1.0) ** 2
