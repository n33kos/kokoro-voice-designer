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
    def __init__(self, device: str | None = None):
        surpressWarnings()
        self.device = device or get_device()
        self.pipeline = KPipeline(lang_code="a", repo_id='hexgrad/Kokoro-82M', device=self.device)

    def generate_audio(self, text: str, voice: torch.Tensor,speed: float = 1.0) -> np.typing.NDArray[np.float32]:
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
