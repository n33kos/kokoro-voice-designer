#!/usr/bin/env python3
"""FastAPI backend for the Kokoro Voice Designer web UI.

Serves voice synthesis, the style-slider definitions, and voice file listings.

Usage:
    uv run uvicorn server:app --reload --port 8000
"""

import io
import json
import os
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Suppress noisy warnings from Kokoro/PyTorch internals
warnings.filterwarnings("ignore", message=".*resized since it had shape.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*", category=UserWarning)
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

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from core import SpeechGenerator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
VOICES_DIR = PROJECT_ROOT / "voices"
CATALOG_DIR = PROJECT_ROOT / "catalog"
STYLE_MAP_FILE = CATALOG_DIR / "style_map.json"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Kokoro Voice Designer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (initialized on startup)
# ---------------------------------------------------------------------------
speech_gen: SpeechGenerator | None = None
# Named style axes in Kokoro's 256-dim style space. See build_style_map.py.
style_feature_names: list[str] | None = None
style_directions: torch.Tensor | None = None
has_style_map: bool = False


def apply_style_deltas(base_voice: torch.Tensor, coeffs: list[float]) -> torch.Tensor:
    """Apply style slider values as offsets in 256-dim style space.

    A voice file is [510, 1, 256] — 510 style vectors indexed by phoneme count.
    The offset is applied uniformly to every row so a slider means the same
    thing regardless of how long the utterance is.
    """
    if style_directions is None:
        return base_voice
    n = min(len(coeffs), style_directions.shape[0])
    c = torch.tensor(coeffs[:n], dtype=torch.float32)
    delta = (c.unsqueeze(0) @ style_directions[:n]).squeeze(0)  # [256]

    rows = base_voice.shape[0]
    out = base_voice.reshape(rows, 256).float() + delta.unsqueeze(0)
    return out.reshape(base_voice.shape)


def load_voice(path: str | Path) -> torch.Tensor:
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


@app.on_event("startup")
async def startup():
    global speech_gen

    print("[server] Initializing Kokoro pipeline...")
    speech_gen = SpeechGenerator()

    # Load style map (v2) if available
    global style_feature_names, style_directions, has_style_map
    if STYLE_MAP_FILE.exists():
        try:
            with open(STYLE_MAP_FILE, "r") as f:
                smap = json.load(f)
            style_feature_names = smap["feature_names"]
            style_directions = torch.tensor(smap["directions"], dtype=torch.float32)
            has_style_map = True
            print(f"[server] Loaded style map: {len(style_feature_names)} features "
                  f"in {style_directions.shape[1]}-dim style space")
        except Exception as e:
            print(f"[server] Could not load style map ({e})")

    print("[server] Ready.")




# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------
class SynthesizeRequest(BaseModel):
    voice: str  # base voice filename (e.g. "af_heart.pt")
    text: str
    speed: float = 1.0
    styleCoefficients: list[float]  # one per style feature, range [-1, 1]


class ExportVoiceRequest(BaseModel):
    voice: str  # base voice filename (e.g. "af_heart.pt")
    styleCoefficients: list[float]  # one per style feature, range [-1, 1]


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "styleMap": has_style_map}


@app.get("/api/voices")
async def list_voices():
    pt_files = sorted(f for f in os.listdir(VOICES_DIR) if f.endswith(".pt"))
    voices = [{"filename": f, "name": f.replace(".pt", "")} for f in pt_files]
    return {"voices": voices}


@app.get("/api/catalog")
async def get_catalog():
    result: dict = {}
    result["hasStyleMap"] = has_style_map
    if has_style_map and style_feature_names is not None:
        # Directions stay server-side: the client only sends slider values, and
        # the 256-dim offsets are applied during synthesis.
        result["styleFeatures"] = [
            {"index": i, "name": n} for i, n in enumerate(style_feature_names)
        ]
    return result


@app.post("/api/export-voice")
async def export_voice(req: ExportVoiceRequest):


    voice_path = VOICES_DIR / req.voice
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice not found: {req.voice}")

    base_voice = load_voice(voice_path)

    if not has_style_map:
        raise HTTPException(status_code=503,
                            detail="No style map loaded. Run build_style_map.py.")
    voice_flat = apply_style_deltas(base_voice, req.styleCoefficients).reshape(-1)

    # Clamp
    base_flat = base_voice.reshape(-1).float()
    base_min = float(base_flat.min())
    base_max = float(base_flat.max())
    clamp_lo = base_min - abs(base_min) * 2
    clamp_hi = base_max + abs(base_max) * 2
    voice_flat = torch.nan_to_num(voice_flat, nan=0.0, posinf=clamp_hi, neginf=clamp_lo)
    voice_flat = voice_flat.clamp(clamp_lo, clamp_hi)

    voice_tensor = voice_flat.reshape(base_voice.shape)

    # Save a copy to output/
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "designed_voice.pt"
    torch.save(voice_tensor, output_path)

    # Serialize tensor to bytes
    buf = io.BytesIO()
    torch.save(voice_tensor, buf)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="designed_voice.pt"'},
    )


@app.post("/api/upload-voice")
async def upload_voice(file: UploadFile):
    """Receive an uploaded .pt voice file and save it to the voices/ directory."""
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="File must be a .pt voice file")

    # Sanitize filename: keep only the basename
    safe_name = os.path.basename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    dest = VOICES_DIR / safe_name
    contents = await file.read()

    # Validate that it's a loadable torch tensor
    try:
        buf = io.BytesIO(contents)
        tensor = torch.load(buf, weights_only=True)
    except Exception:
        try:
            buf = io.BytesIO(contents)
            tensor = torch.load(buf, weights_only=False)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"File is not a valid PyTorch tensor: {str(e)[:100]}",
            )

    with open(dest, "wb") as f:
        f.write(contents)

    return {"filename": safe_name, "name": safe_name.replace(".pt", "")}


@app.post("/api/synthesize")
async def synthesize(req: SynthesizeRequest):
    if speech_gen is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    voice_path = VOICES_DIR / req.voice
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice not found: {req.voice}")

    base_voice = load_voice(voice_path)

    if not has_style_map:
        raise HTTPException(status_code=503,
                            detail="No style map loaded. Run build_style_map.py.")
    voice_flat = apply_style_deltas(base_voice, req.styleCoefficients).reshape(-1)

    # Clamp
    base_flat = base_voice.reshape(-1).float()
    base_min = float(base_flat.min())
    base_max = float(base_flat.max())
    clamp_lo = base_min - abs(base_min) * 2
    clamp_hi = base_max + abs(base_max) * 2
    voice_flat = torch.nan_to_num(voice_flat, nan=0.0, posinf=clamp_hi, neginf=clamp_lo)
    voice_flat = voice_flat.clamp(clamp_lo, clamp_hi)

    voice_tensor = voice_flat.reshape(base_voice.shape)

    # Synthesize
    try:
        audio = speech_gen.generate_audio(req.text, voice_tensor, speed=req.speed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    # Encode as WAV
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=synthesized.wav"},
    )
