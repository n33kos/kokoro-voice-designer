#!/usr/bin/env python3
"""Quick synthesis test — generate audio from any .pt voice file.

Usage:
    # Use the latest designed voice with default text
    uv run python synthesize.py

    # Custom text
    uv run python synthesize.py --text "Hello, this is a test."

    # Specific voice file
    uv run python synthesize.py --voice output/auto_voice_iter0001_*.pt --text "Hi there."

    # Custom output path
    uv run python synthesize.py --output /tmp/test.wav
"""

import argparse
import warnings
from pathlib import Path

import soundfile as sf
import torch

warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*", category=UserWarning)

from core import SpeechGenerator

OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_VOICE = OUTPUT_DIR / "designed_voice.pt"
DEFAULT_OUTPUT = OUTPUT_DIR / "test_synthesis.wav"
DEFAULT_TEXT = "Hello, my name is Alex. How can I help you today?"


def load_voice(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a test audio clip from a .pt voice file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voice", type=str, default=str(DEFAULT_VOICE), help="Voice .pt file")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output .wav path")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    args = parser.parse_args()

    voice_path = Path(args.voice)
    if not voice_path.exists():
        print(f"Error: voice file not found: {voice_path}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Voice  : {voice_path}")
    print(f"Text   : {args.text}")
    print(f"Output : {output_path}")
    print("Loading Kokoro...")

    speech_gen = SpeechGenerator()
    voice = load_voice(voice_path)

    print("Synthesizing...")
    audio = speech_gen.generate_audio(args.text, voice, speed=args.speed)

    sf.write(str(output_path), audio, 24000)
    duration = len(audio) / 24000
    print(f"Done — {duration:.1f}s saved to {output_path}")


if __name__ == "__main__":
    main()
