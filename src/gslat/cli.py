"""CLI for one-band GeoTIFF prediction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import rasterio

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GS-LAT on a single-band SAR raster")
    parser.add_argument("--input", required=True, type=Path, help="single-band georeferenced SAR GeoTIFF")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--representation", required=True, choices=("positive", "db"),
                        help="positive for supplied positive linear power/amplitude DN; db for supplied dB")
    args = parser.parse_args()
    names = ("mask", "seed", "core", "edges", "support", "anchor", "statistical_threshold",
             "threshold", "prior", "support_strength", "geometry_weight")
    paths = [args.output_dir/f"{name}.tif" for name in names] + [args.output_dir/"run.json"]
    if any(path.exists() for path in paths):
        parser.error("Output files already exist; choose a fresh output directory")
    with rasterio.open(args.input) as dataset:
        if dataset.count != 1:
            raise ValueError("Input must contain exactly one raster band")
        values = dataset.read(1).astype(np.float64)
        valid = (dataset.read_masks(1) > 0) & np.isfinite(values)
        grid = dataset.profile.copy()
    if args.representation == "positive" and np.any(values[valid] <= 0):
        raise ValueError("Positive-representation input contains non-positive valid pixels")
    arrays, metadata = run(values, valid, grid)
    digest = hashlib.sha256()
    with args.input.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    metadata["input_sha256"] = digest.hexdigest()
    metadata["representation_argument"] = args.representation
    metadata["software_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-image", "scikit-learn", "rasterio")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        is_mask = array.dtype == bool
        output_profile = {
            "driver": "GTiff", "height": values.shape[0], "width": values.shape[1],
            "crs": grid.get("crs"), "transform": grid["transform"],
            "dtype": "uint8" if is_mask else "float32", "count": 1, "compress": "deflate",
            "nodata": 255 if is_mask else np.nan, "BIGTIFF": "IF_SAFER",
        }
        encoded = np.asarray(array, dtype=output_profile["dtype"]).copy()
        encoded[~valid] = output_profile["nodata"]
        with rasterio.open(args.output_dir/f"{name}.tif", "w", **output_profile) as destination:
            destination.write(encoded, 1)
            destination.write_mask(valid.astype(np.uint8)*255)
    (args.output_dir/"run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    print(f"Wrote {args.output_dir/'mask.tif'} ({metadata['accepted_sources']} profile sources)")


if __name__ == "__main__":
    main()
