"""Discovery mode: explore voice dimensions BEYOND PCA via orthogonal random probing.

The voice tensor is shape [510, 1, 256] = 130,560 dimensions.  PCA from ~54
voices captures at most ~53 components -- leaving ~130,507 dimensions unexplored.
This module probes random orthogonal directions in that unexplored subspace,
generates audio with Kokoro, and keeps directions that produce measurable changes
in audio features.

Results accumulate across runs and are cached as a .pt file (since the discovered
direction vectors are full-size tensors).
"""

import hashlib
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .fitness_scorer import FitnessScorer
from .voice_analyzer import VoiceAnalyzer

# Short prompt for fast probing -- one word keeps generation under ~0.5s each.
PROBE_PROMPT = "Hello"

CACHE_VERSION = 2


class DiscoveryAnalysis:
    """Discover impactful voice directions orthogonal to the PCA subspace.

    On each run, random directions in R^130560 are generated, orthogonalized
    against all known directions (PCA components + prior discoveries), and
    tested for audible impact by perturbing the mean voice and measuring
    feature deltas.  Discoveries accumulate across runs.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / "discovery_cache.pt"
        self._data: dict | None = None

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def cache_exists(self) -> bool:
        """Return True if a .pt cache file is present on disk."""
        return self.cache_path.exists()

    def load_cache(self) -> dict:
        """Load the cache from disk and store internally."""
        self._data = torch.load(self.cache_path, weights_only=False)
        return self._data

    def save_cache(self) -> None:
        """Persist the current results to disk as a .pt file."""
        if self._data is None:
            raise RuntimeError("Nothing to save -- run discovery first.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._data, self.cache_path)

    @property
    def data(self) -> dict | None:
        return self._data

    # ------------------------------------------------------------------
    # Voice hash (cache invalidation)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_voice_hash(voice_files: list[str]) -> str:
        """MD5 hash of sorted voice filenames -- changes when voices are added/removed."""
        canonical = sorted(voice_files)
        return hashlib.md5("\n".join(canonical).encode()).hexdigest()

    # ------------------------------------------------------------------
    # Gram-Schmidt orthogonalization
    # ------------------------------------------------------------------

    @staticmethod
    def _orthogonalize(direction: torch.Tensor, known_subspace: torch.Tensor) -> torch.Tensor | None:
        """Project out the known subspace from *direction* and normalize.

        Args:
            direction: [D] random vector.
            known_subspace: [K, D] matrix whose rows are orthonormal directions.

        Returns:
            Orthogonalized unit vector, or None if the residual is near-zero
            (meaning *direction* was already in the known subspace).
        """
        if known_subspace is not None and known_subspace.shape[0] > 0:
            # projections = known_subspace @ direction  -> [K]
            # projection_sum = projections @ known_subspace -> [D]
            projections = known_subspace @ direction  # [K]
            projection_sum = projections @ known_subspace  # [D]
            direction = direction - projection_sum

        norm = direction.norm()
        if norm < 1e-8:
            return None
        return direction / norm

    # ------------------------------------------------------------------
    # Main discovery sweep
    # ------------------------------------------------------------------

    def run(
        self,
        voice_tensors: list[torch.Tensor],
        voice_files: list[str],
        speech_gen,  # SpeechGenerator instance
        n_probes: int = 200,
        progress_fn: Callable[[float, str], None] | None = None,
    ) -> dict:
        """Probe random orthogonal directions and keep those with audible impact.

        Args:
            voice_tensors: All loaded voice .pt tensors.
            voice_files:   Corresponding filenames (for cache hashing).
            speech_gen:    A SpeechGenerator instance (Kokoro pipeline).
            n_probes:      Number of random directions to test this run.
            progress_fn:   Optional callback ``progress_fn(fraction, description)``.

        Returns:
            The results dict (also stored in ``self._data`` and saved to disk).
        """

        def _progress(frac: float, desc: str) -> None:
            if progress_fn is not None:
                progress_fn(frac, desc)

        voice_hash = self.compute_voice_hash(voice_files)
        n_voices = len(voice_tensors)
        # Use all possible PCA components
        n_pca = min(n_voices - 1, n_voices)

        _progress(0.0, f"Building PCA ({n_pca} components from {n_voices} voices)...")

        # --- Step 1: PCA via VoiceAnalyzer ---
        analyzer = VoiceAnalyzer(voice_tensors, n_components=n_pca)
        D = int(torch.tensor(analyzer.voice_shape).prod().item())  # 130560

        mean_voice = analyzer.mean.reshape(analyzer.voice_shape)
        base_flat = analyzer.mean  # [D], already flat

        # Perturbation scale: median singular value * 0.1
        median_sv = float(analyzer.singular_values.median())
        probe_scale = median_sv * 0.1

        # --- Step 2: Load existing discoveries if cache is valid ---
        existing_components = None  # [N_existing, D]
        existing_impacts: list[float] = []
        existing_sensitivity: list[dict[str, float]] = []
        total_probes_prior = 0

        if self.cache_exists():
            try:
                cached = torch.load(self.cache_path, weights_only=False)
                if (
                    cached.get("version") == CACHE_VERSION
                    and cached.get("voice_hash") == voice_hash
                ):
                    existing_components = cached["components"]  # [N, D]
                    existing_impacts = list(cached["impacts"])
                    existing_sensitivity = list(cached["sensitivity"])
                    total_probes_prior = cached.get("total_probes_run", 0)
                    _progress(
                        0.02,
                        f"Loaded {existing_components.shape[0]} prior discoveries "
                        f"from {total_probes_prior} total probes.",
                    )
            except Exception:
                pass  # corrupt cache -- start fresh

        # --- Step 3: Build known subspace [K, D] ---
        # PCA components are already orthonormal rows of shape [n_pca, D]
        known_parts: list[torch.Tensor] = [analyzer.components]  # [n_pca, D]
        if existing_components is not None and existing_components.shape[0] > 0:
            known_parts.append(existing_components)
        known_subspace = torch.cat(known_parts, dim=0).float()  # [K, D]

        _progress(
            0.03,
            f"Known subspace: {known_subspace.shape[0]} directions "
            f"({analyzer.n_components} PCA + "
            f"{0 if existing_components is None else existing_components.shape[0]} discovered). "
            f"Probing {n_probes} new directions...",
        )

        # --- Step 4: Generate base audio features ---
        _progress(0.05, "Generating base audio for feature comparison...")
        base_audio = speech_gen.generate_audio(PROBE_PROMPT, mean_voice)
        base_features = FitnessScorer.extract_features(base_audio)

        # --- Compute impact threshold from PCA components ---
        # Measure impact of a few PCA components to set a meaningful threshold.
        _progress(0.06, "Measuring PCA component impacts for threshold calibration...")
        pca_impacts: list[float] = []
        n_calibration = min(10, analyzer.n_components)
        for i in range(n_calibration):
            component = analyzer.components[i]
            scale = float(analyzer.singular_values[i]) * 0.1
            perturbed_flat = base_flat + component * scale
            perturbed_voice = perturbed_flat.reshape(analyzer.voice_shape)

            audio = speech_gen.generate_audio(PROBE_PROMPT, perturbed_voice)
            features = FitnessScorer.extract_features(audio)

            impact = 0.0
            for key in base_features:
                base_val = base_features[key]
                new_val = features[key]
                if abs(base_val) > 1e-8:
                    impact += abs((new_val - base_val) / abs(base_val))
                else:
                    impact += abs(new_val - base_val)
            pca_impacts.append(impact)

            frac = 0.06 + 0.04 * ((i + 1) / n_calibration)
            _progress(frac, f"Calibrating threshold: PCA component {i + 1}/{n_calibration}...")

        median_pca_impact = float(np.median(pca_impacts))
        impact_threshold = median_pca_impact * 0.5
        _progress(
            0.10,
            f"Threshold: {impact_threshold:.4f} "
            f"(median PCA impact {median_pca_impact:.4f} * 0.5)",
        )

        # --- Step 5: Random orthogonal probing ---
        new_components: list[torch.Tensor] = []
        new_impacts: list[float] = []
        new_sensitivity: list[dict[str, float]] = []
        n_rejected = 0

        for probe_idx in range(n_probes):
            frac = 0.10 + 0.85 * ((probe_idx + 1) / n_probes)
            _progress(
                frac,
                f"Probe {probe_idx + 1}/{n_probes} "
                f"({len(new_components)} found, {n_rejected} below threshold)...",
            )

            # 5a. Random direction in R^D
            raw = torch.randn(D)

            # 5b. Orthogonalize against known subspace
            direction = self._orthogonalize(raw, known_subspace)
            if direction is None:
                n_rejected += 1
                continue

            # 5c. Perturb mean voice
            perturbed_flat = base_flat + direction * probe_scale
            perturbed_voice = perturbed_flat.reshape(analyzer.voice_shape)

            # 5d. Generate audio
            audio = speech_gen.generate_audio(PROBE_PROMPT, perturbed_voice)

            # 5e. Extract features and compute deltas
            features = FitnessScorer.extract_features(audio)
            deltas: dict[str, float] = {}
            impact = 0.0
            for key in base_features:
                base_val = base_features[key]
                new_val = features[key]
                if abs(base_val) > 1e-8:
                    delta = (new_val - base_val) / abs(base_val)
                else:
                    delta = new_val - base_val
                deltas[key] = delta
                impact += abs(delta)

            # 5f. Keep if impact exceeds threshold
            if impact > impact_threshold:
                new_components.append(direction)
                new_impacts.append(impact)
                new_sensitivity.append(deltas)
                # Expand known subspace so future probes are orthogonal to this one too
                known_subspace = torch.cat(
                    [known_subspace, direction.unsqueeze(0)], dim=0
                )
            else:
                n_rejected += 1

        # --- Step 6: Merge new discoveries with existing ones ---
        _progress(0.96, "Merging and ranking discoveries...")

        all_components_list: list[torch.Tensor] = []
        all_impacts: list[float] = []
        all_sensitivity: list[dict[str, float]] = []

        if existing_components is not None and existing_components.shape[0] > 0:
            all_components_list.append(existing_components)
            all_impacts.extend(existing_impacts)
            all_sensitivity.extend(existing_sensitivity)

        if len(new_components) > 0:
            new_stack = torch.stack(new_components, dim=0)  # [N_new, D]
            all_components_list.append(new_stack)
            all_impacts.extend(new_impacts)
            all_sensitivity.extend(new_sensitivity)

        if len(all_components_list) > 0:
            all_components = torch.cat(all_components_list, dim=0)  # [N_total, D]
        else:
            all_components = torch.zeros(0, D)

        # --- Step 7: Rank by impact descending ---
        ranked_indices = sorted(
            range(len(all_impacts)),
            key=lambda idx: all_impacts[idx],
            reverse=True,
        )

        # --- Step 8: Save cache ---
        _progress(0.98, "Saving discovery cache...")

        self._data = {
            "version": CACHE_VERSION,
            "voice_hash": voice_hash,
            "n_pca_components": analyzer.n_components,
            "total_probes_run": total_probes_prior + n_probes,
            "timestamp": time.time(),
            "components": all_components,  # [N_discovered, D]
            "impacts": all_impacts,
            "sensitivity": all_sensitivity,
            "ranked_indices": ranked_indices,
        }

        self.save_cache()

        _progress(
            1.0,
            f"Discovery complete: {len(new_components)} new directions found "
            f"({all_components.shape[0]} total). "
            f"{n_rejected} probes below threshold.",
        )
        return self._data
