#!/usr/bin/env python3
"""Split a long reference recording into sentence-aligned clips.

Voice matching anchors one row of Kokoro's 510-row voice pack per distinct
training length, so a single clip can only ever support a constant prosody
offset. Several clips of *different* lengths let the prosody profile vary with
utterance length again.

Cutting one long recording is usually better than gathering separate clips: the
same session means one microphone, one room, one noise floor, so the optimizer
isn't asked to average over inconsistent recording conditions.

Two ways to get a transcript per segment:

  --transcript  Split an existing transcript into sentences and match them to
                segments in order. Fast, but only valid when silence boundaries
                really are sentence boundaries — often they aren't, because
                narrators pause mid-sentence and run sentences together. The
                speaking-rate check below exists to catch exactly that.

  --whisper     Transcribe each segment directly. No alignment guesswork, so
                this is the reliable option. Needs a whisper.cpp binary and
                model.

Either way, every segment's implied speaking rate is verified: for one speaker
it should be tight and plausible, so a wild rate means the transcript doesn't
match the audio and the clips must not be trained on.

Usage:
    uv run python split_reference.py --audio input/long_recording.wav \
        --whisper --out-dir input/split
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large.*", category=UserWarning)

SR = 24000
# Plausible range for read speech, in phonemes per second. Anything outside this
# means the sentence-to-segment mapping slipped.
RATE_MIN, RATE_MAX = 8.0, 18.0

WHISPER_BIN = Path.home() / ".claude/voice-multiplexer/whisper/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = Path.home() / ".claude/voice-multiplexer/whisper/models/ggml-large-v3-turbo.bin"


def transcribe(clip: np.ndarray, binary: Path, model: Path) -> str:
    """Transcribe one segment with whisper.cpp. Requires 16kHz mono input."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "seg.wav"
        sf.write(str(wav), librosa.resample(clip, orig_sr=SR, target_sr=16000), 16000)
        res = subprocess.run(
            [str(binary), "-m", str(model), "-f", str(wav), "-nt", "-np", "-l", "en"],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(res.stderr[-400:])
        return " ".join(res.stdout.split()).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def segment_audio(audio: np.ndarray, min_pause: float, top_db: int) -> list[tuple[int, int]]:
    """Cut at pauses longer than `min_pause`, keeping the pause inside the segment."""
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return []

    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        gap = (start - merged[-1][1]) / SR
        if gap > min_pause:
            merged.append([start, end])
        else:
            merged[-1][1] = end
    return [(int(a), int(b)) for a, b in merged]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--transcript", help="Existing transcript to split by sentence")
    ap.add_argument("--whisper", action="store_true",
                    help="Transcribe each segment instead of matching an existing transcript")
    ap.add_argument("--whisper-bin", default=str(WHISPER_BIN))
    ap.add_argument("--whisper-model", default=str(WHISPER_MODEL))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-pause", type=float, default=0.35,
                    help="Seconds of silence that counts as a sentence break")
    ap.add_argument("--top-db", type=int, default=32)
    ap.add_argument("--pad", type=float, default=0.15,
                    help="Seconds of audio kept either side of each segment")
    ap.add_argument("--min-duration", type=float, default=1.5,
                    help="Drop segments shorter than this")
    args = ap.parse_args()

    if not args.transcript and not args.whisper:
        ap.error("need --transcript or --whisper")

    audio, _ = librosa.load(args.audio, sr=SR, mono=True)
    segments = segment_audio(audio, args.min_pause, args.top_db)
    print(f"Audio: {len(audio)/SR:.1f}s -> {len(segments)} segments")

    if args.whisper:
        binary, model = Path(args.whisper_bin), Path(args.whisper_model)
        if not binary.exists() or not model.exists():
            print(f"whisper not found at {binary} / {model}")
            return 1
        print(f"Transcribing {len(segments)} segments with {model.name}...")
        sentences = []
        for i, (start, end) in enumerate(segments, 1):
            s0 = max(0, start - int(args.pad * SR))
            e0 = min(len(audio), end + int(args.pad * SR))
            sentences.append(transcribe(audio[s0:e0], binary, model))
            print(f"  {i}/{len(segments)}", end="\r", flush=True)
        print()
    else:
        sentences = split_sentences(Path(args.transcript).read_text())
        print(f"Transcript: {len(sentences)} sentences")

    if len(segments) != len(sentences):
        print(f"\nCounts differ ({len(segments)} vs {len(sentences)}), so ordered "
              f"matching is not safe. Try adjusting --min-pause; currently "
              f"{args.min_pause}s.")
        print("Segment durations:", [f"{(b-a)/SR:.1f}" for a, b in segments])
        return 1

    # Phoneme counts drive the row Kokoro selects, so estimate them the same way
    # the matcher will. Falls back to characters if Kokoro isn't importable.
    try:
        from core import SpeechGenerator
        from core.differentiable_kokoro import DifferentiableKokoro
        gen = SpeechGenerator(device="cpu")
        diff = DifferentiableKokoro(gen.pipeline.model)
        count = lambda s: len(diff.phonemize(gen.pipeline, s))
        unit = "phonemes"
    except Exception as e:
        print(f"(Kokoro unavailable, estimating from characters: {e})")
        count = len
        unit = "chars"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#':>3} {'dur':>7} {unit:>10} {'rate':>7}  text")
    print("-" * 72)
    rates, written = [], []
    for i, ((start, end), sentence) in enumerate(zip(segments, sentences), 1):
        s = max(0, start - int(args.pad * SR))
        e = min(len(audio), end + int(args.pad * SR))
        clip = audio[s:e]
        dur = len(clip) / SR
        n = count(sentence)
        rate = n / dur if dur > 0 else 0
        flag = "" if RATE_MIN <= rate <= RATE_MAX else "  <-- implausible rate"
        if dur < args.min_duration:
            # Short clips have unreliable rate estimates (padding and leading
            # silence dominate), and they're dropped anyway, so they must not
            # drag the alignment check.
            flag += "  (too short, skipped)"
        else:
            rates.append(rate)
            wav = out_dir / f"seg{i:02d}.wav"
            sf.write(str(wav), clip, SR)
            (out_dir / f"seg{i:02d}.txt").write_text(sentence)
            written.append((wav, n, dur))
        print(f"{i:>3} {dur:>6.1f}s {n:>10} {rate:>6.1f}  {sentence[:38]}{flag}")

    print("-" * 72)
    bad = [r for r in rates if not (RATE_MIN <= r <= RATE_MAX)]
    spread = float(np.std(rates)) / max(float(np.mean(rates)), 1e-6)
    print(f"Rate: mean {np.mean(rates):.1f}, relative spread {spread:.1%}, "
          f"{len(bad)} implausible")
    if bad or spread > 0.25:
        print("\nAlignment looks WRONG — rates should be tight for one speaker. "
              "Do not train on these without checking.")
        return 1

    print(f"\nWrote {len(written)} clips to {out_dir}")
    lengths = sorted({n for _, n, _ in written})
    print(f"Distinct {unit} counts: {len(lengths)} -> prosody basis can use "
          f"up to {len(lengths)} functions")
    print("\nTrain with:")
    args_str = " \\\n    ".join(
        f'--target "{w}" --target-text "{w.with_suffix(".txt")}"' for w, _, _ in written
    )
    print(f"  uv run python invert_voice.py \\\n    {args_str} \\\n"
          f"    --base voices/am_michael.pt --steps 120 --lr 0.02 \\\n"
          f"    --reg-weight 50.0 --speaker-weight 2.0 --f0-weight 5.0 \\\n"
          f"    --out output/match_multi.pt --restart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
