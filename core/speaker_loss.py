"""Differentiable Resemblyzer speaker-embedding loss.

Resemblyzer's `VoiceEncoder.forward` is an ordinary PyTorch LSTM and is fully
differentiable — the only things blocking gradients are `embed_utterance`'s
`no_grad` wrapper and the librosa/webrtcvad preprocessing in front of it. This
module rebuilds that path in torch so the speaker embedding can be used as a
training objective rather than only as an after-the-fact metric.

Why bother when we already have the WavLM pooled-statistics loss: the two
optimize genuinely different things. WavLM layer-4 statistics capture fine
acoustic and prosodic texture but are content-dependent and, on their own,
admit solutions that match the statistics without sounding like the target.
Resemblyzer is trained specifically to separate speaker identity from
everything else. Used together they constrain each other.

Two deliberate departures from stock Resemblyzer preprocessing:

  - **No VAD silence trimming.** `webrtcvad` is not differentiable. Synthesized
    speech has little leading silence, so the cost is small.
  - **Whole-utterance partials.** We mirror the 160-frame partial split the
    encoder was trained on, but slice deterministically rather than using
    `compute_partial_slices`.

Because both target and generated audio go through this same path, the
comparison stays self-consistent. It will not reproduce `embed_utterance`
exactly — keep the stock scorer for independent evaluation.
"""

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .perceptual_loss import resample_24k_to_16k

SR = 16000
N_FFT = 400          # 25ms at 16kHz
HOP = 160            # 10ms at 16kHz
N_MELS = 40
PARTIAL_FRAMES = 160  # 1600ms, what the encoder was trained on
# round((16000 / 1.3) / 160) — the frame stride stock Resemblyzer uses between
# overlapping partials, at its default rate of 1.3 partials per second.
PARTIAL_STEP = 77
TARGET_DBFS = -30


class SpeakerEmbeddingLoss(torch.nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        from resemblyzer import VoiceEncoder

        self.device = device
        self.encoder = VoiceEncoder(device=device, verbose=False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Take the filterbank straight from librosa so the mel scale matches
        # its melspectrogram exactly rather than being reimplemented.
        fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
        self.register_buffer("mel_fb", torch.tensor(fb, dtype=torch.float32, device=device))
        self.register_buffer("window", torch.hann_window(N_FFT, device=device))

    def _normalize_volume(self, wav: torch.Tensor) -> torch.Tensor:
        """Differentiable equivalent of Resemblyzer's dBFS normalization."""
        rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-9)
        dbfs = 20 * torch.log10(rms)
        gain = torch.pow(10.0, (TARGET_DBFS - dbfs) / 20)
        return wav * gain

    def mel(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """[B, T] at 16kHz -> [B, frames, 40] linear (not log) mel, as the encoder expects."""
        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)
        wav_16k = self._normalize_volume(wav_16k)
        stft = torch.stft(wav_16k, n_fft=N_FFT, hop_length=HOP, window=self.window,
                          return_complex=True, center=True, pad_mode="constant")
        power = stft.abs().pow(2)                       # [B, freq, frames]
        mels = self.mel_fb.unsqueeze(0) @ power         # [B, 40, frames]
        return mels.transpose(1, 2)                     # [B, frames, 40]

    def embed(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """Kokoro-rate audio -> L2-normed speaker embedding, on the autograd graph."""
        mels = self.mel(resample_24k_to_16k(audio_24k))
        n_frames = mels.shape[1]

        if n_frames < PARTIAL_FRAMES:
            partials = mels
        else:
            # Mirror `compute_partial_slices`: overlapping windows at ~1.3 per
            # second. Non-overlapping chunks measurably shift the embedding away
            # from what stock Resemblyzer produces, so match its stride.
            starts = list(range(0, max(1, n_frames - PARTIAL_FRAMES + 1), PARTIAL_STEP))
            if starts[-1] + PARTIAL_FRAMES < n_frames:
                starts.append(n_frames - PARTIAL_FRAMES)
            partials = torch.cat(
                [mels[:, s:s + PARTIAL_FRAMES, :] for s in starts], dim=0
            )

        embeds = self.encoder(partials)                 # [P, 256], already L2-normed
        mean = embeds.mean(dim=0, keepdim=True)
        return mean / mean.norm(dim=1, keepdim=True).clamp(min=1e-9)

    @torch.no_grad()
    def target_embedding(self, audio_24k: torch.Tensor) -> torch.Tensor:
        return self.embed(audio_24k)

    def forward(self, generated_24k: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        """Cosine distance in speaker-embedding space; 0 is identical."""
        return 1.0 - F.cosine_similarity(self.embed(generated_24k), target_embed, dim=-1).mean()


def build_manifold_bounds(voice_tensors: list[torch.Tensor], margin: float = 0.0):
    """Per-dimension [min, max] across the built-in voice library.

    Gradient descent will happily drive style dimensions into regions Kokoro
    never saw in training. Measured on a real match, 33% of dimensions ended up
    outside the range of *every* built-in voice, against 22% for coordinate
    descent and 0% for an untouched voice — which is a strong candidate for the
    croakiness and end-of-word pitch jumps that show up in listening.
    """
    stacked = torch.stack([v.reshape(v.shape[0], 256).float() for v in voice_tensors])
    lo = stacked.min(dim=0).values
    hi = stacked.max(dim=0).values
    span = (hi - lo).clamp(min=1e-6)
    return lo - margin * span, hi + margin * span


def manifold_penalty(voice: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Squared hinge on excursion outside the observed per-dimension range.

    Zero inside the box, so it costs nothing until the optimizer actually leaves
    the region Kokoro was trained on — unlike a plain L2 penalty, which fights
    every useful move equally.
    """
    below = (lo - voice).clamp(min=0)
    above = (voice - hi).clamp(min=0)
    return (below.pow(2) + above.pow(2)).mean()
