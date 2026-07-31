import os
import warnings

import numpy as np
import torch
from kokoro import KPipeline


def get_device() -> str:
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SpeechGenerator:
    """Kokoro wrapper.

    `seed` makes synthesis reproducible. Kokoro's vocoder is stochastic — its
    harmonic-plus-noise source module draws a random initial phase and injects
    noise (see kokoro/istftnet.py) — so identical inputs otherwise produce
    different audio, with peak differences around 0.13. That puts a noise floor
    under any similarity metric: two runs of the *same* voice score ~0.9985 on
    Resemblyzer rather than 1.0.

    For search and comparison (auto mode, discovery), pass a seed so that a
    measured difference reflects the voice change rather than sampling noise.
    """

    def __init__(self, device: str | None = None, seed: int | None = None):
        surpressWarnings()
        self.device = device or get_device()
        self.seed = seed
        self.pipeline = KPipeline(lang_code="a", repo_id='hexgrad/Kokoro-82M', device=self.device)

    def generate_audio(self, text: str, voice: torch.Tensor,speed: float = 1.0) -> np.typing.NDArray[np.float32]:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        generator = self.pipeline(text, voice, speed)
        audio = []
        for gs, ps, chunk in generator:
            audio.append(chunk)
        return np.concatenate(audio)

def surpressWarnings():
    # Surpress all these warnings showing up from libraries cluttering the console
    warnings.filterwarnings(
        "ignore",
        message=".*RNN module weights are not part of single contiguous chunk of memory.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore", message=".*is deprecated in favor of*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*dropout option adds dropout after all but last recurrent layer*",
        category=UserWarning,
    )
