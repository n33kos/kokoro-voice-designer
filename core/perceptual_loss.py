"""Differentiable perceptual losses for gradient-based voice inversion.

Two spectral-matching losses were tried and removed. A mean-normalized log-mel
LTAS minimized its own objective (0.466 -> 0.006) while moving the zero-crossing
rate it was meant to fix in the wrong direction, and a band-energy-ratio variant
ranked candidates opposite to how they actually sounded. Matching the pitch
distribution fixed the audible problem instead, and improved spectral balance as
a side effect.

Every reference pitch measurement here delegates to `core.pitch`. Read that
module before touching any pitch target — the targets were wrong for a long time
because the tracker behind them was, and the losses reported success throughout.

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

from . import pitch

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
        # Layers beyond `layer` are computed and discarded. Keeping all 24 cost
        # ~6x the activation memory a backward pass must retain — enough to push
        # a run to 7GB and trigger the OS memory killer.
        #
        # Keep layer+1, not layer: this is a stable-layer-norm encoder, so the
        # final layer_norm is applied to whichever hidden state comes last.
        # Truncating to exactly `layer` makes our target the last one, so it
        # picks up a normalization the full model never applied to it — measured
        # as a 66.2 max-abs divergence. One spare layer keeps it intermediate.
        self.model.encoder.layers = self.model.encoder.layers[:layer + 1]
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
    """Analysis band for the rendered side, from the speaker's own range.

    This used to bracket a yin-derived median by a factor of 2.5 in each
    direction, because a wide fixed band let yin make octave errors that landed
    in the tails where the quantiles live. The band was a workaround for a
    tracker that could not tell a pitch from its subharmonic. `core.pitch` uses
    a tracker with a per-frame voicing confidence instead, so the band is now
    just the speaker's own 1st-99th percentile with headroom.
    """
    return pitch.track(audio, sr).band()


# Quantile levels for distribution matching. Dense enough that the *shape*
# between landmarks is constrained, not just the landmarks: with five levels a
# match had near-perfect quantiles while its density was bimodal with a hole in
# the middle — a third of frames piled into vocal fry below the register and a
# bulge above it, audible as thin and croaky.
F0_DENSE_QUANTILES = tuple(round(0.02 + 0.96 * i / 24, 4) for i in range(25))


F0_RANGE_LEVELS = (0.90, 0.94, 0.98)


def f0_range_loss(f0_pred: torch.Tensor, target: list[float],
                  offset: list[float] | None = None,
                  voiced_threshold: float | None = None,
                  tolerance: float = 0.03) -> torch.Tensor:
    """Stop the top of the pitch range being compressed.

    Two voices came out with p90 **reproducibly ~10% low across independent
    runs** while their medians were within a few percent. Averaging the runs did
    not reduce it, so it is a bias rather than scatter: emphasis simply does not
    reach high enough.

    `f0_distribution_loss` cannot see it. It averages squared error over 25
    quantile levels, so a 10% miss at p90 contributes (ln 1.10)^2 / 25 = 0.0004 —
    nothing beside the other terms. Weighting the tails inside that mean barely
    helps either, because normalising by the weights cancels most of the gain
    (0.00047 against 0.00036). The tail needs its own term.

    **One-sided.** Only a range that is too *narrow* costs anything; overshooting
    is free, and is anyway covered by the distribution match. So this is silent
    on voices whose upper range is already right and pulls only the compressed
    ones. `tolerance` is a 3% deadband, well inside the ~5% run-to-run scatter
    measured between two runs with identical settings.
    """
    if voiced_threshold is None:
        voiced_threshold = voiced_threshold_for(target)
    f0 = f0_pred.squeeze()
    voiced = f0[internal_voiced_mask(f0, voiced_threshold)]
    if voiced.numel() < 8:
        return f0.sum() * 0.0
    log_f0 = torch.log(voiced.clamp(min=1.0))
    want_all = np.asarray(target, dtype=np.float64)
    if offset is not None:
        want_all = want_all + np.asarray(offset, dtype=np.float64)
    levels, wants = [], []
    for lv in F0_RANGE_LEVELS:
        i = int(np.argmin(np.abs(np.asarray(F0_DENSE_QUANTILES) - lv)))
        levels.append(F0_DENSE_QUANTILES[i])
        wants.append(float(want_all[i]))
    qs = torch.tensor(levels, device=f0.device, dtype=log_f0.dtype)
    got = torch.quantile(log_f0, qs)
    want = torch.tensor(wants, device=f0.device, dtype=log_f0.dtype)
    shortfall = (want - got - tolerance).clamp(min=0.0)
    return (shortfall ** 2).mean()


TREMOR_LO_HZ, TREMOR_HI_HZ = 3.0, 10.0
TREMOR_TOTAL_LO, TREMOR_TOTAL_HI = 0.5, 20.0


def _contour_rate(sr: int = KOKORO_SR) -> float:
    """Sample rate of Kokoro's internal F0 contour, in Hz."""
    return sr / pitch.INTERNAL_HOP


def tremor_share(f0_pred: torch.Tensor, voiced_threshold: float,
                 sr: int = KOKORO_SR) -> torch.Tensor:
    """Share of pitch modulation sitting in the 3-10 Hz tremor band.

    A steady wobble through the voice is periodic modulation of pitch at a few
    Hz. **Nothing else here can see it**: `creak_fraction_loss` counts frames
    below a threshold and a tremor sits at the median; `f0_contour_loss`
    constrains how far pitch moves per frame and per syllable, which a wobble
    satisfies as well as melody does.

    Measured against the references, it tracks the listening verdict on every
    voice tested: one fitted voice at 9.2% against its speaker's 5.4% was
    reported as "a constant vibration crackliness all the way through", while two
    others at 3.4% and 6.3% against 4.6% and 7.5% were reported as sounding good.
    It also rose 6.2% -> 9.2% over the two versions where the listener said that
    voice got worse, while every pitch quantile was improving.

    Computed on the internal contour, which is already sampled at 80 Hz, so no
    tracker and no alignment are involved. Unvoiced frames are held at the voiced
    mean rather than interpolated, keeping it differentiable.
    """
    f0 = f0_pred.squeeze()
    if f0.numel() < 64:
        return f0.sum() * 0.0
    voiced = internal_voiced_mask(f0, voiced_threshold)
    if int(voiced.sum()) < 32:
        return f0.sum() * 0.0
    lg = torch.log(f0.clamp(min=1.0))
    filled = torch.where(voiced, lg, lg[voiced].mean())
    x = filled - filled.mean()
    spec = torch.fft.rfft(x).abs() ** 2
    freqs = torch.fft.rfftfreq(x.numel(), d=1.0 / _contour_rate(sr)).to(x.device)
    band = (freqs > TREMOR_LO_HZ) & (freqs < TREMOR_HI_HZ)
    total = (freqs > TREMOR_TOTAL_LO) & (freqs < TREMOR_TOTAL_HI)
    return spec[band].sum() / spec[total].sum().clamp(min=1e-12)


def tremor_reference(audio: np.ndarray, sr: int = KOKORO_SR) -> float:
    """The same share, measured on a reference recording's tracked pitch."""
    track = pitch.track(audio, sr)
    if track.voiced.sum() < 64:
        return 0.0
    lg = np.log(np.clip(track.f0, 1.0, None))
    idx = np.arange(len(lg))
    lg = np.interp(idx, idx[track.voiced], lg[track.voiced])
    x = lg - lg.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=track.hop_seconds)
    band = (freqs > TREMOR_LO_HZ) & (freqs < TREMOR_HI_HZ)
    total = (freqs > TREMOR_TOTAL_LO) & (freqs < TREMOR_TOTAL_HI)
    return float(spec[band].sum() / max(spec[total].sum(), 1e-12))


def tremor_loss(f0_pred: torch.Tensor, max_share: float,
                voiced_threshold: float, sr: int = KOKORO_SR) -> torch.Tensor:
    """Hinge on tremor. Free up to `max_share`; less wobble is never penalised."""
    share = tremor_share(f0_pred, voiced_threshold, sr)
    return ((share - max_share).clamp(min=0.0) / max(max_share, 1e-6)) ** 2


def interharmonic_energy(audio: torch.Tensor, f0_hz: float,
                         sr: int = KOKORO_SR) -> torch.Tensor:
    """Energy sitting *between* the harmonics, relative to energy on them.

    Clean phonation puts nearly everything on multiples of f0; what lands in the
    gaps is noise or intermodulation, which is what roughness is made of.

    **The perceptual weight of this depends on register, so the bound must come
    from the speaker's own reference.** Measured across four voices, the highest
    reading belonged to a 228 Hz voice a listener called clean (136.3% against a
    110.7% reference), while an 82 Hz voice at 127.9% against 115.5% was
    described as "grainy... kind of robotic". At 228 Hz the harmonics are far
    apart and this energy reads as mild breathiness; at 82 Hz they are packed and
    it fills the gaps. Same number, different sound.

    Differentiable through `torch.stft`.
    """
    x = audio.squeeze()
    n_fft = 4096
    if x.numel() < n_fft or f0_hz <= 0:
        return x.sum() * 0.0
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=512, window=window,
                      return_complex=True).abs()
    freqs = torch.linspace(0, sr / 2, spec.shape[0], device=x.device, dtype=x.dtype)
    on = torch.zeros_like(freqs, dtype=torch.bool)
    between = torch.zeros_like(on)
    for k in range(1, int(3500 / f0_hz) + 1):
        c = k * f0_hz
        on |= (freqs > c - f0_hz * 0.15) & (freqs < c + f0_hz * 0.15)
        between |= (freqs > c + f0_hz * 0.30) & (freqs < c + f0_hz * 0.70)
    if not bool(on.any()) or not bool(between.any()):
        return x.sum() * 0.0
    return spec[between].sum() / spec[on].sum().clamp(min=1e-9)


def interharmonic_reference(audio: np.ndarray, f0_hz: float,
                            sr: int = KOKORO_SR) -> float:
    with torch.no_grad():
        return float(interharmonic_energy(
            torch.from_numpy(np.asarray(audio, dtype=np.float32)), f0_hz, sr))


def interharmonic_loss(audio: torch.Tensor, f0_hz: float, max_ratio: float,
                       sr: int = KOKORO_SR) -> torch.Tensor:
    """Hinge on inter-harmonic noise. Free up to the speaker's own level."""
    got = interharmonic_energy(audio, f0_hz, sr)
    return ((got - max_ratio).clamp(min=0.0) / max(max_ratio, 1e-6)) ** 2


def f0_reference_distribution(audio: np.ndarray, sr: int = KOKORO_SR,
                              band: tuple[float, float] | None = None) -> list[float]:
    """Densely sampled log-F0 quantiles — the distribution-matching target."""
    del band
    return pitch.track(audio, sr).quantiles(F0_DENSE_QUANTILES)


def voiced_threshold_for(target: list[float], factor: float = 0.9) -> float:
    """Voicing cut for Kokoro's internal F0, in Hz.

    Kokoro's prosody predictor emits values near zero — sometimes negative —
    wherever there is no voicing, then ramps continuously up to the speaker's
    register at each voicing onset. On one measured utterance 56% of frames sat
    below 10 Hz and a further 2% lay on the ramp between 23 and 114 Hz.

    This cut used to be 0.28x the reference median, which is 23 Hz for a 77 Hz
    speaker — far below the ramp. **Those 2% of ramp frames supplied roughly half
    of all measured frame-to-frame movement**, so the contour statistic was
    mostly voicing transitions, and the bottom quantiles of the distribution
    match were aimed at them too.

    The floor is now the speaker's own lowest real pitch: below the 2nd
    percentile of a confidence-masked reference track, a frame is not a pitch.
    `internal_voiced_mask` additionally requires the neighbouring frames to clear
    it, which is what actually removes the ramp.
    """
    if not target:
        return 50.0
    return float(np.exp(target[0]) * factor)


def internal_voiced_mask(f0: torch.Tensor, floor_hz: float,
                         guard: int = pitch.MIN_VOICED_RUN) -> torch.Tensor:
    """Which frames of Kokoro's internal F0 carry a pitch.

    A level test alone is not enough: the frames sitting on a voicing ramp clear
    any floor low enough to keep the speaker's real low notes. Requiring the
    neighbours to clear it as well drops onsets and offsets, which is where the
    spurious movement lived. Mirrors `core.pitch._stabilize` on the reference
    side, so both sides of every pitch loss select frames the same way.
    """
    voiced = f0 > floor_hz
    if guard <= 0:
        return voiced
    stable = voiced.clone()
    for k in range(1, guard + 1):
        stable[k:] &= voiced[:-k]
        stable[:-k] &= voiced[k:]
    return stable


# Lags for the contour match, in frames of Kokoro's 12.5 ms internal grid.
# `fast` is one frame; `slow` is 150 ms, roughly a syllable. Matching only the
# fast one is underdetermined — an oscillation satisfies it without any melody —
# and the ratio between them is the difference between intonation and a tremble.
# Measured on clean reference tracks, real speech runs slow/fast around 4.
CONTOUR_FAST_LAG = 1
CONTOUR_SLOW_LAG = 12


def f0_contour_reference(audio: np.ndarray, sr: int = KOKORO_SR,
                         band: tuple[float, float] | None = None) -> tuple[float, float]:
    """How far pitch travels over one frame and over one syllable, in log units.

    The distribution losses pin *which* pitches occur and how often. They say
    nothing about how pitch travels between them — two voices can have identical
    pitch histograms and completely different melodies. Measured across four
    matched voices, every one came out 13-22% flatter in frame-to-frame movement
    than its reference, always in the same direction, which is audible as
    intonation that rises and falls less naturally than the real speaker.

    Returns `(fast, slow)`: the spread of the log-F0 change over one 12.5 ms
    frame and over `CONTOUR_SLOW_LAG` frames (150 ms, roughly a syllable).

    **Both are needed.** v21 matched `fast` alone and the optimizer satisfied it
    the cheap way — dithering pitch up and down every frame instead of producing
    wider melodic arcs. The ratio `slow / fast` is the difference: real speech
    runs around 4, because movement accumulates in one direction across a
    syllable, while v21 hit 0.93 on one voice, meaning consecutive steps
    *cancelled*. That is a tremble, not intonation.

    **And both were previously measured wrong, twice over.** They came from yin
    masked by a frequency band, which reports 8-13% octave jumps on real
    recordings; the resulting `fast` target was 0.33 against a true 0.03, so the
    optimizer was asked for ten times the real pitch velocity. They were also
    measured on a 25 ms grid and applied to 12.5 ms predictions, doubling the
    demanded velocity again. Both sides now run on `core.pitch` at the internal
    frame rate. See that module.
    """
    del band
    track = pitch.track(audio, sr)
    return track.spread(CONTOUR_FAST_LAG), track.spread(CONTOUR_SLOW_LAG)


def f0_contour_loss(f0_pred: torch.Tensor, target: tuple[float, float],
                    voiced_threshold: float = 50.0) -> torch.Tensor:
    """Match pitch travel at both frame and syllable scale — the melody.

    Constraining the one-frame spread alone is underdetermined: an oscillation
    hits the target without any melody. Pinning the syllable-scale spread as well
    forces the movement to accumulate rather than cancel.
    """
    f0 = f0_pred.squeeze()
    if f0.numel() < 4 * CONTOUR_SLOW_LAG:
        return f0.sum() * 0.0
    lg = torch.log(f0.clamp(min=1.0))
    voiced = internal_voiced_mask(f0, voiced_threshold)

    def term(lag: int, want: float) -> torch.Tensor:
        d = lg[lag:] - lg[:-lag]
        span = voiced[lag:].clone()
        for k in range(lag):
            span = span & voiced[k:voiced.numel() - lag + k]
        if int(span.sum()) < 8:
            return f0.sum() * 0.0
        return (d[span].std() - want) ** 2

    fast, slow = target
    return term(CONTOUR_FAST_LAG, fast) + term(CONTOUR_SLOW_LAG, slow)


def creak_reference_fraction(audio: np.ndarray, sr: int = KOKORO_SR,
                             band: tuple[float, float] | None = None,
                             factor: float = 0.6) -> tuple[float, float]:
    """How much of a reference sits below `factor` x its own register.

    Returns (fraction, threshold_hz).

    This previously read 5-12% across the references and was documented as "real
    speech contains some creak, so match it rather than eliminate it." That was
    an artifact. Those frames were yin's octave errors — halved pitches sitting
    exactly where creak would sit. Two independent confidence-masked trackers put
    the true figure at 0.0-0.7% on the same recordings, so the target is now
    close to zero, and the term does what its name suggests. The optimizer had
    been *asked* for 7-12% creak, and delivered it.
    """
    del band
    return pitch.track(audio, sr).creak_fraction(factor)


def creak_fraction_loss(f0_pred: torch.Tensor, target_fraction: float,
                        threshold_hz: float, floor_hz: float = 50.0,
                        transition_semitones: float = 1.0) -> torch.Tensor:
    """Match the proportion of frames sitting below the creak threshold.

    Added because raising the voicing cut to exclude sub-register frames from the
    distribution match stopped *penalising* them: excluded frames are
    unconstrained, and the low-frame share rose from 5% to 14% (Michael) and 12%
    to 22% (Kate) once they were no longer measured. Constraining the share
    directly is the correction — the frames stay visible to the loss and their
    quantity is targeted.

    A sigmoid rather than a hard count, so it is differentiable.

    **One-sided and linear**, both deliberately. One-sided because there is never
    a reason to synthesize creak the reference does not have; the target is a
    ceiling, not a value to hit. Linear because a squared error on a fraction
    this small vanishes: with a target of 0 and a measured 1.8%, the squared term
    contributed 0.0003 to a loss of 0.24, so the optimizer had no reason to fix
    audible creak and did not. The linear form makes 1.8% cost 1.8%.

    The mask keeps a one-frame guard rather than the usual two. Creak is
    *isolated* low frames — a wider guard deletes exactly what this term is
    supposed to count, which is how the loss came to report itself satisfied at
    0.0003 while a listener heard the phrase ends drop into fry.

    **The sigmoid runs in log-F0 with a transition fixed in semitones**, not in
    Hz. It used to be `sigmoid((threshold - f0) * 0.15)` — an absolute width, the
    same number of Hz for every speaker, even though speakers live at different
    scales. Measured against each reference's own 10th percentile, that charged
    a 90 Hz speaker **10.9% creak on his genuine low notes** and a 158 Hz speaker
    0.5% on hers. It was punishing the lowest voice for being low, and it pushed
    his median from 83 Hz to 95 against a real 82. A ratio-based transition is
    the same width for everyone, and one semitone is sharp enough that a frame
    two semitones above the threshold is essentially unpenalized.
    """
    f0 = f0_pred.squeeze()
    speech = f0[internal_voiced_mask(f0, floor_hz, guard=1)]
    if speech.numel() < 8:
        return f0.sum() * 0.0
    width = transition_semitones / 12.0 * math.log(2.0)
    below = torch.sigmoid(
        (math.log(max(threshold_hz, 1.0)) - torch.log(speech.clamp(min=1.0))) / width)
    return (below.mean() - target_fraction).clamp(min=0.0)


def pause_share_loss(duration: torch.Tensor, pause_mask: torch.Tensor,
                     target_share: float) -> torch.Tensor:
    """Keep the share of an utterance spent in silence where it started.

    Measured on identical text, the optimization eats pauses. Stock `am_adam`
    renders that text 19.2% silent across 8 pauses; the voice fitted from it came
    out 8.3% silent across 2. Speech with the breaths taken out is heard as
    run-on and oddly emphasised, which is what a listener reported.

    The `pacing_loss` is what creates the freedom: it constrains phonemes per
    second over the *whole* utterance, and pauses are part of that total. Cutting
    silence and lengthening phonemes leaves the overall rate unchanged, so
    nothing pushed back.

    The target is the **base voice's own** share on the same text, not a
    measurement of the reference recording. Silence measured from audio depends
    on the dB threshold — Michael's reference reads 17.3% at -30 dB and 8.9% at
    -45 — and the reference speaks different text with different punctuation.
    The base voice is a real Kokoro voice that phrases correctly, rendering the
    same tokens, so it is both well-defined and known-good. This is a
    regularizer: don't eat the punctuation.
    """
    d = duration.squeeze()
    total = d.sum()
    if total <= 0 or int(pause_mask.sum()) == 0:
        return d.sum() * 0.0
    share = d[pause_mask[:d.numel()]].sum() / total
    return (share - target_share) ** 2


def punctuation_duration_loss(duration: torch.Tensor,
                              masks_and_targets: list[tuple[torch.Tensor, float]],
                              floor_ratio: float = 0.85) -> torch.Tensor:
    """Stop the optimizer from swallowing the breaks at punctuation.

    Measured on identical text, the compression is entirely in the punctuation:

    | | sentence end | comma | word space | speech phoneme |
    | --- | --- | --- | --- | --- |
    | stock am_adam | 167 ms | 71 | 48 | 50 |
    | fitted from it | 117 ms | 42 | 46 | 45 |
    | stock am_puck | 150 ms | 33 | 42 | 43 |
    | fitted from it | 92 ms | 33 | 45 | 45 |

    Sentence breaks lose 30-39%, commas up to 41%, while ordinary phonemes and
    word gaps are untouched. A listener hears it as rushing: "the pauses after a
    period or sentence finishes is shorter than it usually is."

    `pause_share_loss` missed this because it pooled all punctuation together,
    and one measured passage has 82 spaces against 3 sentence ends — the share is
    dominated by spaces, which do not move. **Classes that behave differently
    have to be measured separately or the signal averages away.**

    One-sided against a target duration for each class: shortening a break costs,
    lengthening one is free.

    **The target is the base voice's duration scaled toward the real speaker.**
    Targeting the base voice alone cannot reach: measured as silent gaps in the
    recordings, the real speakers pause 525 / 438 / 288 / 225 ms at the median
    while the stock voices they start from pause 325 / 412 / 412 / 475, and the
    fitted voices ended at 319 / 238 / 188 / 175 — every one 22-46% short of its
    speaker, and two of them short of a base voice that was itself short. A
    listener called it out on all four at once: "they all need to pause more
    after periods and commas, that feels universal."

    The base voice supplies the mapping from token duration to rendered silence,
    which is not one to one; the reference supplies the target. So callers scale
    the base voice's token duration by (real gap / base gap).
    """
    d = duration.squeeze()
    total = d.sum() * 0.0
    for mask, target_frames in masks_and_targets:
        m = mask[:d.numel()]
        if int(m.sum()) == 0 or target_frames <= 0:
            continue
        got = d[m].mean()
        total = total + ((target_frames * floor_ratio - got).clamp(min=0.0)
                         / target_frames) ** 2
    return total


def boundary_duration_loss(duration: torch.Tensor, base_frames: float,
                           ceiling_ratio: float = 1.5) -> torch.Tensor:
    """Cap the leading token, which nothing else constrains.

    Kokoro's token sequence is `[BOS, *phonemes, EOS]`. The BOS token is not in
    `phoneme_mask`, so `duration_spread_loss` cannot see it;
    `punctuation_duration_loss` is one-sided, so lengthening is free; and
    `pacing_loss` averages over the whole utterance, where one long token barely
    registers. It was completely unconstrained, and it ran away: measured at
    **975 ms on one fitted voice** against 250 ms for the stock voice it started
    from, 325-350 ms for the other fitted voices. A listener heard it as "an
    audible artifact at the start... a long robotic sounding stretching of a
    vowel" that cleared once speech began.

    A one-sided hinge above `ceiling_ratio` x the base voice's own leading token.
    Shortening is free; only running long costs.
    """
    d = duration.squeeze()
    if d.numel() < 2 or base_frames <= 0:
        return d.sum() * 0.0
    ceiling = base_frames * ceiling_ratio
    return ((d[0] - ceiling).clamp(min=0.0) / max(base_frames, 1e-6)) ** 2


def subharmonic_reference(audio: np.ndarray, f0_hz: float,
                          sr: int = KOKORO_SR) -> float:
    """Share of voiced-band energy sitting *below* the fundamental."""
    with torch.no_grad():
        return float(subharmonic_energy(
            torch.from_numpy(np.asarray(audio, dtype=np.float32)), f0_hz, sr))


def subharmonic_energy(audio: torch.Tensor, f0_hz: float,
                       sr: int = KOKORO_SR) -> torch.Tensor:
    """Energy below `0.75 x f0` as a fraction of energy below 4 kHz.

    Nothing can be lower than the fundamental except a subharmonic, and
    subharmonics are what roughness is. Measured against one reference: the real
    speaker sits at 0.39% and the stock voice at 0.50%, while the fitted voice
    reached 1.07% — heard as "a low frequency croakiness... almost grainy or
    growly". Jitter was normal on the same audio, so this is not period
    irregularity; it is genuine energy under the fundamental.

    Differentiable through `torch.stft`, so it can be applied to rendered audio.

    **Resolution matters more than it looks.** A first attempt used a 2048-point
    transform, giving 11.7 Hz bins. For a 90 Hz speaker the band below
    `0.75 x f0` is then about six bins wide and is dominated by spectral leakage
    from the fundamental itself plus DC rumble, not by subharmonics. Measured
    that way, a real speaker read 1.17% and the voice fitted to him 1.19% — no
    separation at all — while a clean stock voice read 15.75%, which is not
    physically plausible. A 8192-point transform (2.9 Hz bins) with a 40 Hz
    high-pass to drop rumble is what actually resolves the region.
    """
    x = audio.squeeze()
    n_fft = 8192
    if x.numel() < n_fft:
        return x.sum() * 0.0
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=1024, window=window,
                      return_complex=True).abs()
    freqs = torch.linspace(0, sr / 2, spec.shape[0], device=x.device, dtype=x.dtype)
    band = (freqs > 40.0) & (freqs < f0_hz * 0.75)
    below = spec[band].sum()
    total = spec[(freqs > 40.0) & (freqs < 4000.0)].sum().clamp(min=1e-9)
    return below / total


def subharmonic_loss(audio: torch.Tensor, f0_hz: float, max_share: float,
                     sr: int = KOKORO_SR) -> torch.Tensor:
    """Hinge on energy below the fundamental. Free up to `max_share`."""
    share = subharmonic_energy(audio, f0_hz, sr)
    return ((share - max_share).clamp(min=0.0) / max(max_share, 1e-6)) ** 2


def declination_loss(f0_pred: torch.Tensor, target_slope: float,
                     voiced_threshold: float = 50.0) -> torch.Tensor:
    """Match how fast pitch drifts down across a phrase.

    Real speech declines within a phrase and resets at the next. Nothing
    constrained this, and the fitted voices scattered around their own base
    voices in both directions — one landed at -0.44 log-units/s against a
    reference of -0.25 and a starting voice of -0.25.

    Two audible consequences, and a listener reported both. A phrase that falls
    too fast is heard as odd emphasis, the sentence running out of steam before
    it ends. It also *arrives* at creak: measured across five positions in a
    phrase, that voice's low frames ran 0.0/0.0/3.3/6.7/12.7% from start to end,
    against a reference that was flat at 0.6-4.4%. The creak was not spread
    through the voice, it was piled at the phrase ends, which is where a steep
    declination puts the pitch below the speaker's range.

    Slope is ordinary least squares, so it stays differentiable in `f0_pred`.
    Phrase boundaries come from a detached mask.
    """
    f0 = f0_pred.squeeze()
    if f0.numel() < 4 * pitch.PHRASE_MIN_FRAMES:
        return f0.sum() * 0.0
    voiced = internal_voiced_mask(f0, voiced_threshold)
    lg = torch.log(f0.clamp(min=1.0))
    mask = voiced.detach().cpu().numpy()

    slopes = []
    for a, b in pitch.phrase_spans(mask):
        idx = np.nonzero(mask[a:b])[0] + a
        if len(idx) < pitch.PHRASE_MIN_FRAMES:
            continue
        x = torch.tensor((idx - idx[0]) * pitch.INTERNAL_FRAME_SECONDS,
                         device=f0.device, dtype=lg.dtype)
        y = lg[torch.as_tensor(idx, device=f0.device)]
        xc = x - x.mean()
        slopes.append((xc * (y - y.mean())).sum() / (xc.pow(2).sum() + 1e-9))
    if not slopes:
        return f0.sum() * 0.0
    return (torch.stack(slopes).mean() - target_slope) ** 2


def declination_reference(audio: np.ndarray, sr: int = KOKORO_SR) -> float:
    """Mean within-phrase log-F0 slope of a reference, in log-units per second."""
    return pitch.track(audio, sr).declination()


def f0_distribution_loss(f0_pred: torch.Tensor, target: list[float],
                         offset: list[float] | None = None,
                         voiced_threshold: float | None = None,
                         tail_emphasis: float = 2.0) -> torch.Tensor:
    """Match the whole log-F0 distribution, not five points on it.

    Equivalent to a quantile-function (inverse-CDF) distance: comparing sorted
    distributions at many levels constrains density everywhere, so the optimizer
    cannot satisfy the landmarks while leaving a gap between them.

    `offset` is a per-level correction for the gap between Kokoro's internal
    F0_pred and pitch measured on rendered audio. That gap is a stretch, not a
    shift — measured 0.68x at the bottom of the distribution and 1.09x at the
    top — so it must be applied per level.

    **The tails are weighted up.** A flat mean over 25 levels cannot feel a tail
    error: a 10% miss at p90 contributes (ln 1.10)^2 / 25 = 0.0004, which is
    nothing beside the other terms. Two voices came out with p90 reproducibly
    ~10% low across independent runs — a bias, not scatter, since averaging runs
    did not reduce it — while their medians were within a few percent. That is
    the signature of a loss that nails the middle and cannot see the edges.
    `tail_emphasis` scales each level by 1 + tail_emphasis * |2q - 1|, so the
    extremes count roughly three times the median.
    """
    if voiced_threshold is None:
        voiced_threshold = voiced_threshold_for(target)
    f0 = f0_pred.squeeze()
    voiced = f0[internal_voiced_mask(f0, voiced_threshold)]
    if voiced.numel() < 8:
        return f0.sum() * 0.0
    log_f0 = torch.log(voiced.clamp(min=1.0))
    qs = torch.tensor(F0_DENSE_QUANTILES, device=f0.device, dtype=log_f0.dtype)
    got = torch.quantile(log_f0, qs)
    want = torch.tensor(target, device=f0.device, dtype=log_f0.dtype)
    if offset is not None:
        want = want + torch.tensor(offset, device=f0.device, dtype=log_f0.dtype)
    w = 1.0 + tail_emphasis * (2.0 * qs - 1.0).abs()
    return (((got - want) ** 2) * w).sum() / w.sum()


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
    """Deprecated: matched a ratio on Kokoro's signed internal energy contour.

    Kept only so old checkpoints and scripts still import. **Do not use it.**
    `N_pred` is signed and its mean can cross zero, so `std/mean` is not a
    coefficient of variation and does not correspond to the same ratio computed
    on waveform RMS, which is strictly positive.

    Measured on a voice fitted with this term: `N_pred` came out **77% negative**
    with mean -1.71 and peak +1.47, where the stock voice it started from had
    mean +4.42 and peak +9.10. The decoder was being told "near silent" for three
    quarters of every utterance — audible as words collapsing into a whisper —
    while the loss reported itself satisfied at 0.006, because `.abs()` on the
    mean hid the sign flip and the `n > n.abs().mean() * 0.1` filter then kept
    only a small unrepresentative slice near the top.

    Use `energy_range_reference` / `energy_range_loss`, which measure the same
    quantity on the rendered waveform on both sides.
    """
    n = n_pred.squeeze()
    n = n[n > n.abs().mean() * 0.1]
    if n.numel() < 10:
        return n_pred.sum() * 0.0
    cv = n.std() / n.mean().abs().clamp(min=1e-6)
    return (cv - target_cv) ** 2


ENERGY_FRAME = 1200   # 50 ms
ENERGY_HOP = 300      # 12.5 ms
ENERGY_FLOOR_DB = 40  # frames this far below the loudest are silence


def _frame_db(audio: torch.Tensor) -> torch.Tensor:
    """Per-frame level in dB relative to the loudest frame. Differentiable."""
    x = audio.squeeze()
    if x.numel() < ENERGY_FRAME:
        return x.new_zeros(1)
    frames = x.unfold(0, ENERGY_FRAME, ENERGY_HOP)
    rms = frames.pow(2).mean(dim=-1).clamp(min=1e-12).sqrt()
    db = 20.0 * torch.log10(rms)
    return db - db.max()


def energy_range_reference(audio: np.ndarray, sr: int = KOKORO_SR) -> float:
    """Spread of frame level, in dB, over the non-silent part of a recording."""
    db = _frame_db(torch.from_numpy(np.asarray(audio, dtype=np.float32)))
    keep = db > -ENERGY_FLOOR_DB
    return float(db[keep].std()) if int(keep.sum()) >= 10 else 0.0


def energy_range_loss(audio: torch.Tensor, target_std_db: float,
                      tolerance_db: float = 1.5) -> torch.Tensor:
    """Match how much the voice's loudness moves, measured the same way on both
    sides — frame level in dB on the rendered waveform.

    Replaces a ratio taken on Kokoro's internal `N_pred`, which is signed and so
    has no meaningful coefficient of variation; see `energy_dynamics_loss`. The
    reference and the generated audio go through identical code.

    The error is **relative to the target**, not absolute. dB spreads are O(10),
    so a squared absolute error is O(1-100) while every other term here is
    O(0.01-0.5): at weight 2 the absolute form was 63% of the objective at step
    zero. Dividing by the target makes it a fraction like the rest.

    **Deadband.** Nothing is charged while the spread is within `tolerance_db` of
    the target. A plain squared error is never zero, so it moves a voice that is
    already right: v27 pulled a voice whose spread was 9.83 dB against a target
    of 8.79 down to 7.70 — no closer, and its median pitch went 11% sharp in the
    process. A corrective term must be inert on a healthy case.
    """
    db = _frame_db(audio)
    keep = db > -ENERGY_FLOOR_DB
    if int(keep.sum()) < 10:
        return audio.sum() * 0.0
    scale = max(abs(target_std_db), 1e-3)
    excess = ((db[keep].std() - target_std_db).abs() - tolerance_db).clamp(min=0.0)
    return (excess / scale) ** 2


NOISE_FLOOR_MAX_DB = -40.0
NOISE_FLOOR_MARGIN_DB = 30.0   # slack above the base voice, on the same text.
# 12 dB (v30) drove the optimized quantity 20-63 dB better while the long-text
# floor barely moved, and cost one voice 1.8% -> 4.8% creak. Until the statistic
# is measured inside the gaps rather than as a global percentile, a tighter bound
# buys nothing audible and does do harm.


def noise_floor_of(audio: torch.Tensor) -> float:
    """The p10/p90 frame-energy ratio in dB. Same code the loss uses."""
    x = audio.squeeze()
    if x.numel() < ENERGY_FRAME:
        return 0.0
    frames = x.unfold(0, ENERGY_FRAME, ENERGY_HOP)
    rms = frames.pow(2).mean(dim=-1).clamp(min=1e-12).sqrt()
    qs = torch.tensor([0.10, 0.90], device=rms.device, dtype=rms.dtype)
    lo, hi = torch.quantile(rms, qs)
    return float(20.0 * torch.log10(lo / hi.clamp(min=1e-12)))


def noise_floor_loss(audio: torch.Tensor,
                     max_floor_db: float = NOISE_FLOOR_MAX_DB) -> torch.Tensor:
    """Keep the gaps between words quiet.

    The 10th percentile of frame energy against the 90th — how far the quiet
    moments sit below the loud ones. Fitted voices came out 20-45 dB louder in
    the gaps than the stock voices they started from, heard as breath or faint
    background sound in the spaces between words.

    That it is *reachable* is what justifies targeting it: across the 34 English
    stock voices this ratio spans -140 dB to -28 dB with a median of -64, while
    all four fitted voices land at -29 to -36 — worse than every stock voice but
    one. The style vector clearly controls this; ours had simply drifted to the
    bad end of the range.

    A **hinge**, free below `max_floor_db`. Not a target: the absolute value
    moves with the analysis window (one voice reads -28.5, -25.1 or -19.6 dB at
    25/50/100 ms frames), so only "worse than a bound" is well-posed, not "equal
    to a number".

    **The bound must be measured at the length the loss sees.** v29 used a fixed
    -40 dB, chosen from stock voices measured on a 20 s sample where they read
    -46 to -71. On the 2-6 s training utterances the loss actually evaluates,
    those same voices sit at **-75 to -141 dB** — so the bound was 35-100 dB too
    lenient. The result was textbook: the optimizer drove one voice to -40.1
    against a -40.0 threshold and stopped, parked exactly on a constraint that
    was not demanding anything. Callers should pass the base voice's own floor on
    the same text plus a margin, the way `duration_spread_loss` and
    `pause_share_loss` take their targets.

    The excess is scaled by 40 dB so a large violation stays comparable to the
    other terms rather than swamping them — a raw dB error squared is O(100-1000)
    against O(0.01-0.5) for everything else.
    """
    x = audio.squeeze()
    if x.numel() < ENERGY_FRAME:
        return audio.sum() * 0.0
    frames = x.unfold(0, ENERGY_FRAME, ENERGY_HOP)
    rms = frames.pow(2).mean(dim=-1).clamp(min=1e-12).sqrt()
    qs = torch.tensor([0.10, 0.90], device=rms.device, dtype=rms.dtype)
    lo, hi = torch.quantile(rms, qs)
    floor_db = 20.0 * torch.log10(lo / hi.clamp(min=1e-12))
    return ((floor_db - max_floor_db).clamp(min=0.0) / 40.0) ** 2


def energy_sign_loss(n_pred: torch.Tensor, max_negative: float = 0.40,
                     scale: float = 1.0) -> torch.Tensor:
    """Stop Kokoro's energy contour from inverting.

    `N_pred` drives the decoder's loudness. Fitted voices drove it deeply
    negative — 79%, 91% and 51% of frames below zero on three of four speakers,
    against 6-21% for the stock voices they started from — which the ear hears as
    words collapsing into a whisper.

    A **hinge**, not a target: zero cost while the negative share stays under
    `max_negative`. The one voice that sounded right sat at 26%, comfortably
    inside, and must not be moved. The broken ones sit far outside and are pulled
    back. Soft-counted through a sigmoid so it stays differentiable.
    """
    n = n_pred.squeeze()
    if n.numel() < 10:
        return n_pred.sum() * 0.0
    negative = torch.sigmoid(-n / scale).mean()
    return (negative - max_negative).clamp(min=0.0) ** 2


def duration_spread_loss(duration: torch.Tensor, phoneme_mask: torch.Tensor,
                         target_cv: float, floor_ratio: float = 0.85) -> torch.Tensor:
    """Keep the variation in phoneme length where the base voice had it.

    `pacing_loss` constrains phonemes per second across the whole utterance, and
    the cheapest way to hit a faster rate is to crush the longest phonemes.
    Measured on identical text, stock `am_puck` runs a duration spread of 1.21
    with one phoneme held for 28 frames; the voice fitted from it collapsed to
    0.46 with **nothing longer than 5 frames**. Every sound ends up the same
    length, so stressed syllables that should stretch get compressed — heard as
    pacing that lurches and words squeezed to nothing.

    Spread rather than the durations themselves, so the voice may still speak
    faster or slower overall; it just may not flatten the rhythm to do it.

    **One-sided, with a deadband.** Only a collapse below `floor_ratio` of the
    base voice costs anything; more variation than the base voice is free. As a
    plain squared error this term moved a voice sitting at 1.16 against a base of
    1.13 — already correct — down to 1.01, and cost it 11% on median pitch. A
    voice inside the band now gets exactly zero gradient.
    """
    d = duration.squeeze()
    m = phoneme_mask[:d.numel()]
    if int(m.sum()) < 10:
        return d.sum() * 0.0
    speech = d[m]
    cv = speech.std() / speech.mean().clamp(min=1e-6)
    return (target_cv * floor_ratio - cv).clamp(min=0.0) ** 2


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
