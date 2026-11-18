"""Shared numerical helpers for the selected GS-LAT workflow."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

def clean_mask(mask: np.ndarray, valid: np.ndarray, minimum_size: int) -> np.ndarray:
    """Apply 8-connected foreground cleanup and remove tiny water objects."""
    structure = np.ones((3, 3), dtype=bool)
    raw = mask & valid
    kernel = structure.astype(np.uint8)
    kernel[1, 1] = 0
    valid_neighbors = ndi.convolve(valid.astype(np.int16), kernel, mode="constant")
    foreground_neighbors = ndi.convolve(raw.astype(np.int16), kernel, mode="constant")
    same = np.where(raw, foreground_neighbors, valid_neighbors - foreground_neighbors)
    isolated = valid & (valid_neighbors > 0) & (same == 0)
    cleaned = raw.copy()
    cleaned[isolated] = foreground_neighbors[isolated] * 2 > valid_neighbors[isolated]
    labels, count = ndi.label(cleaned, structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels] & valid
