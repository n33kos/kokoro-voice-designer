"""One pitch tracker, used for every reference measurement in the project.

Every pitch target — register, distribution, creak, contour — used to come from
`librosa.yin` masked by a frequency band. That measurement is not trustworthy:
run on the reference recordings themselves, it reports **8-13% of adjacent voiced
frames jumping by more than an octave**, which is not something a human larynx
does. Those are the tracker's own octave errors, and they were being fed into the
targets as if they were the speaker's voice.

Everything downstream inherited them:

- the **creak** target read 7-12% of frames below the register, when the true
  figure is 0-1%. The optimizer was being *asked* for the croakiness.
- the **contour** target read 0.33 log-units of movement per frame against a true
  0.03. It was being asked for ten times the real pitch velocity, which is the
  warble.
- the bottom of the **distribution** sat an octave low — 42 Hz against a true
  60 Hz for one speaker — so the loss demanded a population of frames below the
  speaker's actual range.

The tell, in hindsight: frame-to-frame movement measured the same at a 12.5 ms
hop as at 25 ms. For any real signal, halving the time step roughly halves the
spread of the difference. Reading the same number at both hops means the
measurement is dominated by tracker noise, not by the voice.

Two independent trackers agree on the corrected numbers. CREPE (`torchcrepe`) is
used when installed, `librosa.pyin` otherwise; both report 0.0% octave jumps and
0-1% creak on the same recordings. Both carry a per-frame voicing confidence,
which is what a frequency band was standing in for and doing badly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

KOKORO_SR = 24000

# Kokoro's prosody predictor emits F0 on a 300-sample grid at 24 kHz. Reference
# tracks are measured on the same grid so that a lag of N frames means the same
# duration on both sides of every loss. It did not: targets were measured at
# 25 ms and compared against 12.5 ms predictions, asking for twice the pitch
# velocity before any other error.
INTERNAL_HOP = 300
INTERNAL_FRAME_SECONDS = INTERNAL_HOP / KOKORO_SR  # 12.5 ms

FMIN, FMAX = 55.0, 600.0
MIN_PERIODICITY = 0.5  # CREPE confidence below this is not a pitch
MIN_VOICED_RUN = 2     # frames either side that must also be voiced

# A phrase is a voiced stretch bounded by a pause. At 12.5 ms per frame these
# are 100 ms of silence and a 200 ms minimum length.
PHRASE_GAP_FRAMES = 8
PHRASE_MIN_FRAMES = 16


def phrase_spans(voiced: np.ndarray, gap: int = PHRASE_GAP_FRAMES,
                 min_len: int = PHRASE_MIN_FRAMES) -> list[tuple[int, int]]:
    """Split a voicing mask into phrases at pauses of at least `gap` frames."""
    spans, start, run = [], None, 0
    for i, on in enumerate(voiced):
        if on:
            if start is None:
                start = i
            run = 0
        elif start is not None:
            run += 1
            if run >= gap:
                if i - run - start >= min_len:
                    spans.append((start, i - run))
                start = None
    if start is not None and len(voiced) - start >= min_len:
        spans.append((start, len(voiced)))
    return spans


@dataclass
class PitchTrack:
    """A reference pitch contour and which of its frames are real."""

    f0: np.ndarray        # Hz, arbitrary where not voiced
    voiced: np.ndarray    # bool, same length
    hop_seconds: float
    tracker: str

    @property
    def values(self) -> np.ndarray:
        return self.f0[self.voiced]

    def quantiles(self, levels) -> list[float]:
        v = self.values
        if len(v) < 20:
            return [0.0] * len(levels)
        return [float(q) for q in np.quantile(np.log(v), levels)]

    def register(self) -> float:
        """The pitch the voice returns to most — the mode, not the mean."""
        v = self.values
        if len(v) < 20:
            return 0.0
        hist, edges = np.histogram(np.log(v), bins=40)
        i = int(hist.argmax())
        return float(np.exp(0.5 * (edges[i] + edges[i + 1])))

    def band(self) -> tuple[float, float]:
        """Analysis band for the rendered side, from the speaker's own range."""
        v = self.values
        if len(v) < 20:
            return FMIN, FMAX
        lo, hi = np.quantile(v, [0.01, 0.99])
        return float(max(FMIN, lo / 1.3)), float(min(FMAX, hi * 1.3))

    def spread(self, lag: int) -> float:
        """Spread of the log-F0 change across `lag` frames, over voiced spans.

        Every frame in the span must be voiced. Stepping across a voicing gap
        measures the gap, not the melody, and those steps are exactly where the
        old measurement's spurious movement came from.
        """
        lg = np.log(np.clip(self.f0, 1.0, None))
        d = lg[lag:] - lg[:-lag]
        span = self.voiced[lag:].copy()
        for k in range(lag):
            span &= self.voiced[k:len(self.voiced) - lag + k]
        return float(np.std(d[span])) if span.sum() >= 20 else 0.0

    def phrases(self, gap: int = PHRASE_GAP_FRAMES,
                min_len: int = PHRASE_MIN_FRAMES) -> list[tuple[int, int]]:
        """Voiced stretches separated by a pause, in frame indices."""
        return phrase_spans(self.voiced, gap, min_len)

    def declination(self) -> float:
        """Mean rate of pitch drift within a phrase, in log-units per second.

        Real speech drifts downward across a phrase and resets at the next one.
        Nothing in the loss constrained this, and the fitted voices came out
        scattered around their own base voices in both directions — one at
        -0.44/s against a reference of -0.25 and a base voice of -0.25. A phrase
        that dives that hard runs out of register before it ends, which is both
        the odd emphasis a listener hears and a way to arrive at creak.
        """
        lg = np.log(np.clip(self.f0, 1.0, None))
        slopes = []
        for a, b in self.phrases():
            idx = np.arange(a, b)[self.voiced[a:b]]
            if len(idx) < PHRASE_MIN_FRAMES:
                continue
            x = (idx - idx[0]) * self.hop_seconds
            slopes.append(np.polyfit(x, lg[idx], 1)[0])
        return float(np.mean(slopes)) if slopes else 0.0

    def creak_fraction(self, factor: float = 0.6) -> tuple[float, float]:
        """Share of voiced frames below `factor` x the median, and that cut."""
        v = self.values
        if len(v) < 20:
            return 0.0, 0.0
        threshold = float(np.median(v) * factor)
        return float(np.mean(v < threshold)), threshold


def _stabilize(voiced: np.ndarray, run: int = MIN_VOICED_RUN) -> np.ndarray:
    """Drop frames whose neighbours are unvoiced — onsets, offsets, glitches."""
    if run <= 0:
        return voiced
    stable = voiced.copy()
    for k in range(1, run + 1):
        stable[k:] &= voiced[:-k]
        stable[:-k] &= voiced[k:]
    return stable


def _track_crepe(audio: np.ndarray, sr: int, hop_seconds: float):
    import torch
    import torchcrepe

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    hop = max(1, int(round(16000 * hop_seconds)))
    f0, periodicity = torchcrepe.predict(
        torch.from_numpy(audio.astype(np.float32))[None], 16000, hop,
        fmin=FMIN, fmax=FMAX, model="full", batch_size=512,
        device="cpu", return_periodicity=True)
    periodicity = torchcrepe.filter.median(periodicity, 3)
    return f0[0].numpy().astype(np.float64), periodicity[0].numpy() > MIN_PERIODICITY


def _track_pyin(audio: np.ndarray, sr: int, hop_seconds: float):
    import librosa

    hop = max(1, int(round(sr * hop_seconds)))
    f0, voiced, _ = librosa.pyin(audio, fmin=FMIN, fmax=FMAX, sr=sr,
                                 frame_length=2048, hop_length=hop)
    return np.nan_to_num(f0, nan=1.0), voiced & np.isfinite(f0)


_CACHE: dict[tuple, PitchTrack] = {}


def track(audio: np.ndarray, sr: int = KOKORO_SR,
          hop_seconds: float = INTERNAL_FRAME_SECONDS,
          prefer: str = "crepe") -> PitchTrack:
    """Pitch-track a reference. Cached — several targets share one pass."""
    key = (hashlib.blake2b(np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
                           digest_size=16).hexdigest(), sr, round(hop_seconds, 6), prefer)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    tracker = prefer
    try:
        if prefer != "crepe":
            raise ImportError
        f0, voiced = _track_crepe(audio, sr, hop_seconds)
    except ImportError:
        tracker = "pyin"
        f0, voiced = _track_pyin(audio, sr, hop_seconds)

    result = PitchTrack(f0=f0, voiced=_stabilize(voiced),
                        hop_seconds=hop_seconds, tracker=tracker)
    _CACHE[key] = result
    return result


def silent_gaps(audio: np.ndarray, sr: int = KOKORO_SR,
                floor_db: float = -35.0, min_ms: float = 120.0) -> np.ndarray:
    """Lengths of the silent stretches in a recording, in milliseconds.

    How long a speaker actually pauses. Unlike token durations this is directly
    comparable between a reference recording and rendered audio, because it is
    measured the same way on both and does not depend on the text.
    """
    import librosa

    hop = INTERNAL_HOP
    rms = librosa.feature.rms(y=audio, frame_length=4 * hop, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max(rms))
    quiet = db < floor_db
    runs, n = [], 0
    for q in quiet:
        if q:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    ms = np.asarray(runs, dtype=np.float64) * 1000.0 * hop / sr
    return ms[ms >= min_ms]
