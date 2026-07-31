#!/usr/bin/env python3
"""FastAPI backend for the Kokoro Voice Designer web UI.

Serves voice synthesis API, catalog/component data, and voice file listings.

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

from core import SpeechGenerator, VoiceAnalyzer, FitnessScorer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
VOICES_DIR = PROJECT_ROOT / "voices"
CATALOG_DIR = PROJECT_ROOT / "catalog"
DEFAULT_CATALOG = CATALOG_DIR / "discovery_catalog.pt"
COMPONENT_LABELS_FILE = CATALOG_DIR / "component_labels.json"
STYLE_MAP_FILE = CATALOG_DIR / "style_map.json"

FEATURE_LABELS = {
    "pitch_mean": "pitch",
    "pitch_std": "pitch variation",
    "spectral_centroid_mean": "brightness",
    "spectral_bandwidth_mean": "fullness",
    "spectral_rolloff_mean": "crispness",
    "spectral_contrast_mean": "clarity",
    "spectral_flatness_mean": "breathiness",
    "rms_energy": "volume",
    "energy_mean": "energy",
    "energy_std": "energy variation",
    "mfcc1_mean": "timbre",
    "mfcc2_mean": "nasality",
    "mfcc3_mean": "resonance",
    "mfcc4_mean": "texture",
    "chroma_mean": "harmonics",
    "tonnetz_mean": "tonality",
    "tempo": "pace",
    "audio_std": "dynamics",
    "harmonic_ratio": "harmonic richness",
    "zero_crossing_rate": "sibilance",
}

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
catalog_data: dict | None = None
component_names: list[str] = []
component_count: int = 0
pca_count: int = 0
all_components: torch.Tensor | None = None
component_ranges: torch.Tensor | None = None
pca_mean: torch.Tensor | None = None
voice_shape: tuple | None = None
# Style map (v2): directions live in Kokoro's native 256-dim style space rather
# than in PCA/discovery component space. See build_style_map.py.
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


def compute_component_labels(
    components: torch.Tensor,
    singular_values: torch.Tensor,
    comp_ranges: torch.Tensor,
    base_voice: torch.Tensor,
    gen: SpeechGenerator,
    text: str = "Hello, how are you today?",
) -> list[str]:
    """Run sensitivity analysis and assign human-readable labels to components."""
    base_flat = base_voice.reshape(-1).float()
    base_audio = gen.generate_audio(text, base_voice)
    base_features = FitnessScorer.extract_features(base_audio)

    n_active = components.shape[0]
    sensitivity: dict[int, dict] = {}

    base_min = float(base_flat.min())
    base_max = float(base_flat.max())
    clamp_lo = base_min - abs(base_min) * 2
    clamp_hi = base_max + abs(base_max) * 2

    for i in range(n_active):
        component = components[i]
        scale = float(singular_values[i]) * 0.1
        perturbed_flat = base_flat + component * scale
        perturbed_flat = torch.nan_to_num(perturbed_flat, nan=0.0, posinf=clamp_hi, neginf=clamp_lo)
        perturbed_flat = perturbed_flat.clamp(clamp_lo, clamp_hi)
        perturbed_voice = perturbed_flat.reshape(base_voice.shape)
        try:
            audio = gen.generate_audio(text, perturbed_voice)
            features = FitnessScorer.extract_features(audio)
            deltas = {}
            for key in base_features:
                bv = base_features[key]
                nv = features[key]
                deltas[key] = (nv - bv) / abs(bv) if abs(bv) > 1e-8 else (nv - bv)
            sensitivity[i] = deltas
        except Exception:
            sensitivity[i] = {k: 0.0 for k in base_features}

    # Assign names
    used_names: set[str] = set()
    names: list[str] = []
    for i in range(n_active):
        deltas = sensitivity.get(i, {})
        scale_factor = float(comp_ranges[i]) / (float(singular_values[i]) * 0.1) if float(singular_values[i]) * 0.1 > 0 else 1.0
        scaled = {k: abs(v) * scale_factor for k, v in deltas.items()}
        sorted_feats = sorted(scaled.items(), key=lambda x: x[1], reverse=True)
        name = f"component {i}"
        for feat, _ in sorted_feats:
            if feat in FEATURE_LABELS:
                base_name = FEATURE_LABELS[feat]
                if base_name not in used_names:
                    name = base_name
                    used_names.add(base_name)
                    break
        names.append(name)
    return names


def _default_component_names(n_pca_count: int, total: int) -> list[str]:
    """Fallback label assignment: cycle through FEATURE_LABELS for all components."""
    label_list = list(FEATURE_LABELS.values())
    name_counts: dict[str, int] = {}
    names = []
    for i in range(total):
        base_name = label_list[i % len(label_list)] if label_list else f"d{i}"
        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        if count > 1:
            names.append(f"{base_name} {count}")
        else:
            names.append(base_name)
    return names


@app.on_event("startup")
async def startup():
    global speech_gen, catalog_data, component_names, component_count, pca_count
    global all_components, component_ranges, pca_mean, voice_shape

    print("[server] Initializing Kokoro pipeline...")
    speech_gen = SpeechGenerator()

    # Load voices for PCA / ranges
    pt_files = sorted(f for f in os.listdir(VOICES_DIR) if f.endswith(".pt"))
    voice_tensors = [load_voice(VOICES_DIR / f) for f in pt_files]
    voice_shape_val = voice_tensors[0].shape
    voice_shape = voice_shape_val
    n_voices = len(voice_tensors)

    # Load catalog or compute PCA
    catalog = None
    if DEFAULT_CATALOG.exists():
        try:
            catalog = torch.load(DEFAULT_CATALOG, weights_only=False)
            print(f"[server] Loaded catalog: {catalog.get('pca_n_components', 0)} PCA + "
                  f"{len(catalog.get('discovered_impacts', []))} discoveries")
        except Exception as e:
            print(f"[server] Could not load catalog: {e}")

    if catalog is not None:
        n_pca = catalog.get("pca_n_components", 0)
        pca_comps = catalog["pca_components"][:n_pca]
        pca_sv = catalog["pca_singular_values"][:n_pca]
        pca_mean = catalog["pca_mean"]
        analyzer_mean = pca_mean

        # Discovery components
        disc_comps = catalog.get("discovered_components")
        if disc_comps is not None and disc_comps.shape[0] > 0:
            avg_sv = float(pca_sv.mean())
            disc_sv = torch.full((disc_comps.shape[0],), avg_sv)
            all_comps_parts = [pca_comps, disc_comps]
            all_sv_parts = [pca_sv, disc_sv]
        else:
            all_comps_parts = [pca_comps]
            all_sv_parts = [pca_sv]
            disc_comps = None

        all_components = torch.cat(all_comps_parts, dim=0)
        all_sv = torch.cat(all_sv_parts, dim=0)

        # Component ranges
        flat = torch.stack(voice_tensors).reshape(n_voices, -1).float()
        centered = flat - analyzer_mean
        projections = centered @ all_components.T
        component_ranges = projections.max(dim=0).values - projections.min(dim=0).values

        # Fix discovery ranges
        n_pca_actual = pca_comps.shape[0]
        if disc_comps is not None and disc_comps.shape[0] > 0:
            avg_pca_range = float(component_ranges[:n_pca_actual].mean())
            component_ranges[n_pca_actual:] = avg_pca_range

        catalog_data = catalog
    else:
        # Compute PCA from voices
        n_pca = min(n_voices - 1, 20)
        analyzer = VoiceAnalyzer(voice_tensors, n_components=n_pca)
        pca_comps = analyzer.components
        pca_sv = analyzer.singular_values
        pca_mean = analyzer.mean

        all_components = pca_comps
        all_sv = pca_sv

        flat = torch.stack(voice_tensors).reshape(n_voices, -1).float()
        centered = flat - pca_mean
        projections = centered @ all_components.T
        component_ranges = projections.max(dim=0).values - projections.min(dim=0).values

    component_count = all_components.shape[0]
    pca_count = pca_comps.shape[0]

    # Assign labels: prefer labels from file, fall back to FEATURE_LABELS / "d{N}"
    if COMPONENT_LABELS_FILE.exists():
        try:
            with open(COMPONENT_LABELS_FILE, "r") as f:
                file_labels = json.load(f)
            # Use file labels up to component_count, pad with dN if file is shorter
            component_names = []
            for i in range(component_count):
                if i < len(file_labels):
                    component_names.append(file_labels[i])
                else:
                    component_names.append(f"d{i}")
            print(f"[server] Loaded {len(file_labels)} labels from {COMPONENT_LABELS_FILE}")
        except Exception as e:
            print(f"[server] Could not load labels file ({e}), using defaults")
            component_names = _default_component_names(pca_comps.shape[0], component_count)
    else:
        component_names = _default_component_names(pca_comps.shape[0], component_count)

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

    print(f"[server] Ready. {component_count} components: {', '.join(component_names[:10])}...")


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------
class SynthesizeRequest(BaseModel):
    voice: str  # base voice filename (e.g. "af_heart.pt")
    coefficients: list[float]  # one per component, range [-1, 1]
    text: str
    speed: float = 1.0
    # When present, sliders are applied in 256-dim style space (style map v2)
    # instead of component space, and `coefficients` is ignored.
    styleCoefficients: list[float] | None = None


class ExportVoiceRequest(BaseModel):
    voice: str  # base voice filename (e.g. "af_heart.pt")
    coefficients: list[float]  # one per component, range [-1, 1]
    # When present, applied in 256-dim style space and `coefficients` is ignored.
    styleCoefficients: list[float] | None = None


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "components": component_count}


@app.get("/api/voices")
async def list_voices():
    pt_files = sorted(f for f in os.listdir(VOICES_DIR) if f.endswith(".pt"))
    voices = [{"filename": f, "name": f.replace(".pt", "")} for f in pt_files]
    return {"voices": voices}


@app.get("/api/catalog")
async def get_catalog():
    result: dict = {
        "components": [
            {"index": i, "name": component_names[i]}
            for i in range(component_count)
        ],
        "count": component_count,
        "pcaCount": pca_count,
    }
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
    if all_components is None or component_ranges is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    voice_path = VOICES_DIR / req.voice
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice not found: {req.voice}")

    base_voice = load_voice(voice_path)

    if req.styleCoefficients is not None and has_style_map:
        voice_flat = apply_style_deltas(base_voice, req.styleCoefficients).reshape(-1)
    else:
        # Build coefficient tensor
        coeffs = torch.zeros(component_count)
        for i, c in enumerate(req.coefficients[:component_count]):
            coeffs[i] = c

        # Apply: voice = base + sum(coeff * range * component)
        scaled = coeffs * component_ranges
        perturbation = (scaled.unsqueeze(0) @ all_components).squeeze(0)
        voice_flat = base_voice.reshape(-1).float() + perturbation

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
    if speech_gen is None or all_components is None or component_ranges is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    voice_path = VOICES_DIR / req.voice
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice not found: {req.voice}")

    base_voice = load_voice(voice_path)

    if req.styleCoefficients is not None and has_style_map:
        voice_flat = apply_style_deltas(base_voice, req.styleCoefficients).reshape(-1)
    else:
        # Build coefficient tensor
        coeffs = torch.zeros(component_count)
        for i, c in enumerate(req.coefficients[:component_count]):
            coeffs[i] = c

        # Apply: voice = base + sum(coeff * range * component)
        scaled = coeffs * component_ranges
        perturbation = (scaled.unsqueeze(0) @ all_components).squeeze(0)
        voice_flat = base_voice.reshape(-1).float() + perturbation

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
