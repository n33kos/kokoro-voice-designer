#!/usr/bin/env python3
"""Build a discovery catalog from PCA + top discovered directions.

The catalog bakes together:
  - PCA components (computed fresh from the voice library)
  - Top-N discovered directions (from the working discovery cache)

This eliminates per-run PCA computation and distills the best discoveries
into a version-controlled file that can be shared and reused.

Usage:
    # Build catalog from voices/ + output/discovery_cache.pt
    python build_catalog.py

    # Specify how many discovered directions to include
    python build_catalog.py --n-discoveries 500

    # Use a custom discovery cache or output path
    python build_catalog.py --cache output/my_cache.pt --output catalog/v2.pt

    # Preview what would be built (no files written)
    python build_catalog.py --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from core import VoiceAnalyzer

VOICES_DIR = Path(__file__).parent / "voices"
OUTPUT_DIR = Path(__file__).parent / "output"
CATALOG_DIR = Path(__file__).parent / "catalog"
DEFAULT_CACHE = OUTPUT_DIR / "discovery_cache.pt"
DEFAULT_CATALOG = CATALOG_DIR / "discovery_catalog.pt"

CATALOG_VERSION = 1


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_voice(path: str) -> torch.Tensor:
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)


def build_catalog(
    voice_folder: str,
    cache_path: Path | None,
    n_discoveries: int,
    output_path: Path,
    dry_run: bool = False,
) -> None:
    # --- Load voice library ---
    pt_files = sorted(f for f in os.listdir(voice_folder) if f.endswith(".pt"))
    if len(pt_files) < 3:
        print(f"Error: need at least 3 .pt files in {voice_folder}, found {len(pt_files)}")
        sys.exit(1)
    print(f"Loading {len(pt_files)} voices from {voice_folder}...")
    voice_tensors = [load_voice(os.path.join(voice_folder, f)) for f in pt_files]

    # --- Run PCA ---
    n_pca = min(len(voice_tensors) - 1, len(voice_tensors))
    print(f"Computing PCA ({n_pca} components)...")
    analyzer = VoiceAnalyzer(voice_tensors, n_components=n_pca)

    print(f"  PCA: {analyzer.n_components} components, "
          f"{analyzer.explained_variance_ratio:.1%} variance explained")
    print(f"  Voice shape: {analyzer.voice_shape}")
    print(f"  Mean tensor: {analyzer.mean.shape}")

    # --- Load discovery cache ---
    disc_components = None
    disc_impacts: list[float] = []
    disc_sensitivity: list[dict] = []
    total_probes = 0
    cache_voice_hash = None

    if cache_path and cache_path.exists():
        print(f"\nLoading discovery cache: {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
        all_comps = cache.get("components")
        all_impacts = cache.get("impacts", [])
        all_sens = cache.get("sensitivity", [])
        ranked = cache.get("ranked_indices", list(range(len(all_impacts))))
        total_probes = cache.get("total_probes_run", 0)
        cache_voice_hash = cache.get("voice_hash")

        if all_comps is not None and all_comps.shape[0] > 0:
            n_available = all_comps.shape[0]
            n_keep = min(n_discoveries, n_available)
            top_indices = ranked[:n_keep]
            disc_components = all_comps[top_indices]
            disc_impacts = [all_impacts[i] for i in top_indices]
            disc_sensitivity = [all_sens[i] for i in top_indices] if all_sens else []
            print(f"  {n_available:,} total discoveries → keeping top {n_keep:,} by impact")
            if disc_impacts:
                print(f"  Impact range: {min(disc_impacts):.2f} – {max(disc_impacts):.2f}")
        else:
            print("  Cache is empty — catalog will contain PCA only.")
    else:
        print("\nNo discovery cache found — catalog will contain PCA only.")
        if n_discoveries > 0:
            print("  Run discovery first with auto_mode.py or the Gradio UI.")

    # --- Size estimate ---
    pca_size = analyzer.components.numel() * 4
    disc_size = disc_components.numel() * 4 if disc_components is not None else 0
    total_size = pca_size + disc_size
    print(f"\nEstimated catalog size:")
    print(f"  PCA ({analyzer.n_components} components): {human_bytes(pca_size)}")
    if disc_components is not None:
        print(f"  Discoveries ({disc_components.shape[0]:,}): {human_bytes(disc_size)}")
    print(f"  Total: {human_bytes(total_size)}")

    if dry_run:
        print("\nDry run — no files written.")
        return

    # --- Build and save catalog ---
    catalog = {
        "version": CATALOG_VERSION,
        "voice_hash": cache_voice_hash,
        "voice_shape": list(analyzer.voice_shape),
        "voice_dim": analyzer.mean.shape[0],
        "pca_components": analyzer.components,         # [n_pca, D]
        "pca_singular_values": analyzer.singular_values,  # [n_pca]
        "pca_variance": analyzer.per_component_variance,   # [n_pca]
        "pca_mean": analyzer.mean,                      # [D]
        "pca_n_components": analyzer.n_components,
        "pca_explained_variance": float(analyzer.explained_variance_ratio),
        "n_voices": len(voice_tensors),
        "discovered_components": disc_components,       # [N, D] or None
        "discovered_impacts": disc_impacts,
        "discovered_sensitivity": disc_sensitivity,
        "n_total_probes": total_probes,
        "timestamp": time.time(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving catalog to {output_path}...")
    torch.save(catalog, output_path)
    saved_size = output_path.stat().st_size
    print(f"Saved: {human_bytes(saved_size)}")

    n_disc = disc_components.shape[0] if disc_components is not None else 0
    print(f"\nCatalog summary:")
    print(f"  PCA components : {analyzer.n_components}")
    print(f"  Discoveries    : {n_disc:,}")
    print(f"  Total directions: {analyzer.n_components + n_disc:,}")
    print(f"  Total probes run: {total_probes:,}")
    print(f"\nDone. Load with --catalog {output_path} in auto_mode.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a discovery catalog from PCA + top discovered directions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--voice-folder",
        default=str(VOICES_DIR),
        help="Folder containing .pt voice files",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(DEFAULT_CACHE),
        help="Discovery cache (.pt) to pull discovered directions from",
    )
    parser.add_argument(
        "--n-discoveries",
        type=int,
        default=1000,
        help="Number of top discovered directions to include",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_CATALOG),
        help="Output catalog path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print size estimates without writing any files",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache) if args.cache else None
    build_catalog(
        voice_folder=args.voice_folder,
        cache_path=cache_path,
        n_discoveries=args.n_discoveries,
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
