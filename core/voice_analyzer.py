import torch
from tqdm import tqdm
from typing import Any


class VoiceAnalyzer:
    """PCA decomposition of voice tensor space with sensitivity analysis.

    Reduces the search space from ~130K random dimensions to the ~20 principal
    directions that actually distinguish voices, and maps those directions to
    audio features (pitch, timbre, etc.) for targeted perturbation.
    """

    def __init__(self, voices: list[torch.Tensor], n_components: int = 20):
        stacked = torch.stack(voices)  # [N, 510, 1, 256]
        self.voice_shape = stacked.shape[1:]  # [510, 1, 256]
        n_voices = len(voices)

        flat = stacked.reshape(n_voices, -1).float()  # [N, 130560]
        self.mean = flat.mean(dim=0)
        centered = flat - self.mean

        # Element-wise std for perturbation magnitude reference
        self.element_std = flat.std(dim=0)
        self.reference_norm = float(self.element_std.norm())

        # SVD-based PCA (numerically stable for N < D)
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        self.n_components = min(n_components, len(S))
        self.components = Vt[:self.n_components]  # [K, D]
        self.singular_values = S[:self.n_components]

        total_var = (S ** 2).sum()
        self.explained_variance_ratio = float((S[:self.n_components] ** 2).sum() / total_var)
        self.per_component_variance = S[:self.n_components] ** 2 / total_var

        # Populated by run_sensitivity() and compute_component_weights()
        self.sensitivity: dict[int, dict[str, float]] | None = None
        self.component_weights: torch.Tensor | None = None
        self.signed_weights: torch.Tensor | None = None

    def run_sensitivity(self, base_voice: torch.Tensor, speech_generator,
                        fitness_scorer, target_text: str,
                        perturbation_scale: float = 0.1) -> dict[int, dict[str, float]]:
        """Perturb along each PCA component and measure audio feature changes.

        One-time cost (~20 Kokoro inferences) that maps PCA components
        to their effect on audio features (pitch, spectral centroid, MFCCs, etc.)
        """
        base_flat = base_voice.reshape(-1).float()
        base_audio = speech_generator.generate_audio(target_text, base_voice)
        base_features = fitness_scorer.extract_features(base_audio)

        sensitivity = {}
        print(f"Running sensitivity analysis on {self.n_components} PCA components...")

        for i in tqdm(range(self.n_components), desc="Sensitivity"):
            component = self.components[i]
            scale = float(self.singular_values[i]) * perturbation_scale

            perturbed_flat = base_flat + component * scale
            perturbed_voice = perturbed_flat.reshape(self.voice_shape)

            audio = speech_generator.generate_audio(target_text, perturbed_voice)
            features = fitness_scorer.extract_features(audio)

            deltas = {}
            for key in base_features:
                base_val = base_features[key]
                new_val = features[key]
                if abs(base_val) > 1e-8:
                    deltas[key] = (new_val - base_val) / abs(base_val)
                else:
                    deltas[key] = new_val - base_val
            sensitivity[i] = deltas

        self.sensitivity = sensitivity
        return sensitivity

    def compute_component_weights(self, target_features: dict[str, Any],
                                  current_features: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Use sensitivity map + feature gap to weight each PCA component.

        Returns (magnitude_weights, signed_weights):
        - magnitude_weights: [0.1, 1.0] per component — how much to perturb
        - signed_weights: directional bias for each component
        """
        if self.sensitivity is None:
            uniform = torch.ones(self.n_components)
            self.component_weights = uniform
            self.signed_weights = torch.zeros(self.n_components)
            return uniform, torch.zeros(self.n_components)

        # Normalized feature gap: positive means target > current
        feature_gap = {}
        for key in target_features:
            if key in current_features:
                t_val = target_features[key]
                c_val = current_features[key]
                if abs(t_val) > 1e-8:
                    feature_gap[key] = (t_val - c_val) / abs(t_val)
                else:
                    feature_gap[key] = t_val - c_val

        # Dot product of feature gap with each component's sensitivity = alignment
        signed = torch.zeros(self.n_components)
        for i in range(self.n_components):
            alignment = 0.0
            for key in feature_gap:
                if key in self.sensitivity[i]:
                    alignment += feature_gap[key] * self.sensitivity[i][key]
            signed[i] = alignment

        magnitudes = signed.abs()
        if magnitudes.max() > 0:
            magnitudes = magnitudes / magnitudes.max()
        magnitudes = magnitudes.clamp(min=0.1)  # floor so no component is zeroed out

        self.component_weights = magnitudes
        self.signed_weights = signed
        return magnitudes, signed

    def generate_perturbation(self, diversity: float,
                              direction_bias: float = 0.3) -> torch.Tensor:
        """Generate a perturbation in PCA space, optionally biased by feature-gap direction.

        Args:
            diversity: Overall perturbation scale (same meaning as VoiceGenerator.generate_voice)
            direction_bias: 0.0 = random in PCA space, 1.0 = fully directed toward feature gap
        """
        coeffs = torch.randn(self.n_components)

        # Bias each coefficient toward the direction that closes the feature gap
        if self.signed_weights is not None and direction_bias > 0:
            for i in range(self.n_components):
                if self.signed_weights[i] > 0:
                    directed = abs(coeffs[i])
                elif self.signed_weights[i] < 0:
                    directed = -abs(coeffs[i])
                else:
                    directed = coeffs[i]
                coeffs[i] = direction_bias * directed + (1 - direction_bias) * coeffs[i]

        # Scale by component importance and natural variance
        weights = self.component_weights if self.component_weights is not None else torch.ones(self.n_components)
        scaled = coeffs * weights * self.singular_values

        # Project back to original voice-tensor space
        perturbation = (scaled.unsqueeze(0) @ self.components).squeeze(0)  # [D]

        # Normalize magnitude to match expected random-perturbation scale
        p_norm = perturbation.norm()
        if p_norm > 0:
            perturbation = perturbation / p_norm * self.reference_norm * diversity

        return perturbation.reshape(self.voice_shape)
