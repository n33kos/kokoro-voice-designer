"""Differentiable audio features for Jacobian-based semantic mapping.

The existing semantic map measures a component's effect by perturbing it,
synthesizing, and diffing librosa features — one synthesis per component, and
every measurement carries librosa's own noise. With a differentiable forward
pass we can instead backprop a feature straight to the style vector and get its
exact gradient.

That requires the features themselves to be differentiable, which rules out a
few librosa staples: spectral rolloff (a quantile search), zero-crossing rate
(a sign count), and beat-tracked tempo all sever the graph. Each is replaced
here by a differentiable quantity measuring the same perceptual thing — for
example high-band energy ratio instead of zero-crossing rate for sibilance.

Every function takes Kokoro-rate audio [T] or [B, T] and returns a scalar per
batch item that stays on the autograd graph.
"""

import torch

KOKORO_SR = 24000
N_FFT = 2048
HOP = 512


def _spectrogram(audio: torch.Tensor, n_fft: int = N_FFT, hop: int = HOP):
    """Magnitude spectrogram and its frequency axis."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=window,
                      return_complex=True, center=True)
    mag = stft.abs() + 1e-8  # [B, F, T]
    freqs = torch.linspace(0, KOKORO_SR / 2, mag.shape[1], device=audio.device,
                           dtype=audio.dtype)
    return mag, freqs


def volume(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    return audio.pow(2).mean(dim=-1).sqrt()


def brightness(audio: torch.Tensor) -> torch.Tensor:
    """Spectral centroid — the 'center of mass' of the spectrum, in Hz."""
    mag, freqs = _spectrogram(audio)
    weights = mag / mag.sum(dim=1, keepdim=True)
    centroid = (weights * freqs.view(1, -1, 1)).sum(dim=1)
    return centroid.mean(dim=-1)


def fullness(audio: torch.Tensor) -> torch.Tensor:
    """Spectral bandwidth — spread of energy around the centroid."""
    mag, freqs = _spectrogram(audio)
    weights = mag / mag.sum(dim=1, keepdim=True)
    f = freqs.view(1, -1, 1)
    centroid = (weights * f).sum(dim=1, keepdim=True)
    var = (weights * (f - centroid).pow(2)).sum(dim=1)
    return var.clamp(min=1e-8).sqrt().mean(dim=-1)


def breathiness(audio: torch.Tensor) -> torch.Tensor:
    """Spectral flatness — noise-like spectra score high, tonal ones low."""
    mag, _ = _spectrogram(audio)
    geo = mag.log().mean(dim=1).exp()
    arith = mag.mean(dim=1)
    return (geo / (arith + 1e-8)).mean(dim=-1)


def _band_ratio(audio: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    mag, freqs = _spectrogram(audio)
    band = ((freqs >= lo) & (freqs < hi)).to(mag.dtype).view(1, -1, 1)
    return ((mag * band).sum(dim=1) / mag.sum(dim=1)).mean(dim=-1)


def sibilance(audio: torch.Tensor) -> torch.Tensor:
    """High-band energy ratio. Differentiable stand-in for zero-crossing rate."""
    return _band_ratio(audio, 4000.0, KOKORO_SR / 2)


def warmth(audio: torch.Tensor) -> torch.Tensor:
    """Low-band energy ratio — the perceptual complement of sibilance."""
    return _band_ratio(audio, 0.0, 500.0)


def dynamics(audio: torch.Tensor) -> torch.Tensor:
    """Frame-energy variation — how much loudness moves across the utterance."""
    mag, _ = _spectrogram(audio)
    frame_energy = mag.sum(dim=1)
    return frame_energy.std(dim=-1) / (frame_energy.mean(dim=-1) + 1e-8)


# Audio-domain features, measured on the decoder output. These are the ones that
# need a full forward+backward through the vocoder.
AUDIO_FEATURES = {
    "volume": volume,
    "brightness": brightness,
    "fullness": fullness,
    "breathiness": breathiness,
    "sibilance": sibilance,
    "warmth": warmth,
    "dynamics": dynamics,
}


# Prosody features read straight off the model's own internals — exact, and far
# cheaper than going through the vocoder. `pace` in particular replaces
# beat-tracked tempo, which is close to meaningless on a few seconds of speech.
def prosody_features(out) -> dict[str, torch.Tensor]:
    """Named scalars from a `DiffOutput`, no audio analysis involved."""
    return {
        "pace": out.duration.sum(),
        "pitch": out.f0_pred.mean(),
        "pitch variation": out.f0_pred.std(),
        "energy": out.n_pred.mean(),
        "energy variation": out.n_pred.std(),
    }


PROSODY_FEATURE_NAMES = ["pace", "pitch", "pitch variation", "energy", "energy variation"]
