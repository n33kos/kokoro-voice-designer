#!/usr/bin/env python3
"""Prune the discovery cache to its top-N highest-impact directions.

This dramatically reduces file size (from ~11GB to ~300MB for top-1000)
while keeping the most valuable discovered voice directions.

Usage:
    # Keep top 1000 directions, overwrite in-place
    python prune_cache.py --keep 1000

    # Preview stats without modifying anything
    python prune_cache.py --dry-run

    # Save a "shareable catalog" of top 500, separate from the working cache
    python prune_cache.py --keep 500 --output output/catalog.pt

    # Prune a specific cache file
    python prune_cache.py --cache output/my_cache.pt --keep 500

The pruned file is a valid discovery cache that can be loaded by the
auto_mode.py and Gradio UI. Future discovery runs will continue to
accumulate into it.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_CACHE = OUTPUT_DIR / "discovery_cache.pt"


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_cache(path: Path) -> dict:
    print(f"Loading {path} ...")
    cache = torch.load(path, weights_only=False)
    return cache


def print_stats(cache: dict, label: str = "") -> None:
    comps = cache.get("components")
    impacts = cache.get("impacts", [])
    n = comps.shape[0] if comps is not None else 0
    total_probes = cache.get("total_probes_run", "?")
    ts = cache.get("timestamp", 0)
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "unknown"

    if label:
        print(f"\n{label}")
    print(f"  Directions  : {n:,}")
    print(f"  Total probes: {total_probes:,}")
    print(f"  Timestamp   : {ts_str}")
    if n > 0 and comps is not None:
        size_bytes = comps.numel() * 4  # float32
        print(f"  Tensor size : {human_bytes(size_bytes)}")
    if impacts:
        ranked = sorted(range(len(impacts)), key=lambda i: impacts[i], reverse=True)
        top5 = [impacts[i] for i in ranked[:5]]
        print(f"  Top 5 impacts: {[f'{v:.2f}' for v in top5]}")
        print(f"  Min impact  : {min(impacts):.4f}")
        print(f"  Max impact  : {max(impacts):.4f}")


def prune(cache: dict, keep: int) -> dict:
    impacts = cache.get("impacts", [])
    components = cache.get("components")
    sensitivity = cache.get("sensitivity", [])

    n_total = len(impacts)
    if n_total == 0 or components is None:
        print("Cache is empty — nothing to prune.")
        return cache

    # Ranked indices (already sorted by impact descending, but recompute to be safe)
    ranked = sorted(range(n_total), key=lambda i: impacts[i], reverse=True)

    n_keep = min(keep, n_total)
    top_indices = ranked[:n_keep]

    pruned_components = components[top_indices]
    pruned_impacts = [impacts[i] for i in top_indices]
    pruned_sensitivity = [sensitivity[i] for i in top_indices] if sensitivity else []
    new_ranked = list(range(n_keep))  # already in impact-descending order

    pruned = {
        "version": cache.get("version", 2),
        "voice_hash": cache.get("voice_hash"),
        "n_pca_components": cache.get("n_pca_components"),
        "total_probes_run": cache.get("total_probes_run", 0),
        "timestamp": time.time(),
        "components": pruned_components,
        "impacts": pruned_impacts,
        "sensitivity": pruned_sensitivity,
        "ranked_indices": new_ranked,
    }

    print(f"\n  Pruned {n_total:,} → {n_keep:,} directions "
          f"(removed {n_total - n_keep:,} lowest-impact).")
    return pruned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune discovery cache to top-N highest-impact directions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(DEFAULT_CACHE),
        help="Path to discovery_cache.pt to prune",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=1000,
        help="Number of top-impact directions to keep",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: overwrites input file). Use a different path to keep original.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and size estimates without modifying any files",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"Error: cache file not found: {cache_path}")
        sys.exit(1)

    original_size = cache_path.stat().st_size
    cache = load_cache(cache_path)
    print_stats(cache, label="Before pruning:")

    if args.dry_run:
        impacts = cache.get("impacts", [])
        n_total = len(impacts)
        n_keep = min(args.keep, n_total)
        comps = cache.get("components")
        if comps is not None:
            pruned_size = (n_keep / n_total) * comps.numel() * 4
            print(f"\nDry run — would keep {n_keep:,} of {n_total:,} directions.")
            print(f"  Estimated size: {human_bytes(int(pruned_size))} "
                  f"(currently {human_bytes(original_size)})")
        return

    pruned = prune(cache, args.keep)

    out_path = Path(args.output) if args.output else cache_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {out_path} ...")
    torch.save(pruned, out_path)

    new_size = out_path.stat().st_size
    print(f"  {human_bytes(original_size)} → {human_bytes(new_size)} "
          f"({100 * new_size / original_size:.1f}% of original)")
    print_stats(pruned, label="After pruning:")
    print("\nDone.")


if __name__ == "__main__":
    main()
