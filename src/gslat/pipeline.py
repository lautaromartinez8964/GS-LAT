"""Single-image GS-LAT prediction with no reference data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import research_geometry as geometry
from .numerics import clean_mask
from .seed import histogram_seed
from .statistics import empirical_threshold


def _algorithm(pixel_size: float) -> dict:
    raw = json.loads((Path(__file__).parent / "research_config.json").read_text(encoding="utf8"))
    return geometry.resolve_domain_algorithm("linear_power", raw, pixel_size)


def _summarize(raw_profiles: list[dict], scene_values: np.ndarray) -> dict | None:
    valid_profiles = []
    for profile in raw_profiles:
        result = geometry.ordinal_estimate(profile, "ordinal", {"native_pixel_size_m": 1.0})
        result.update(profile_index=profile["profile_index"], kind=profile["kind"],
                      step_pixels=profile["step_pixels"])
        if (result.get("status") == "accepted"
                and np.isfinite(result.get("local_threshold", np.nan))
                and result["core_median"] < result["local_threshold"] < result["bank_median"]):
            result["aggregation_inlier"] = True
            valid_profiles.append(result)
    if len(valid_profiles) < 2:
        return None
    contrasts = []
    for profile in valid_profiles:
        ranks = np.searchsorted(scene_values, [profile["core_median"], profile["bank_median"]],
                                side="right") / len(scene_values)
        contrasts.append(max(0.0, float(ranks[1]-ranks[0])))
    return {
        "profiles": valid_profiles,
        "threshold": geometry.observed_quantile(
            np.array([p["local_threshold"] for p in valid_profiles]), 0.5),
        "reliability": len(valid_profiles)/max(1, len(raw_profiles)),
        "scene_contrast": geometry.observed_quantile(np.asarray(contrasts), 0.5),
    }


def build_geometry(values: np.ndarray, valid: np.ndarray, profile: dict,
                   seed: np.ndarray) -> tuple[dict, list, int]:
    """Build profile supports from the supplied histogram seed."""
    pixel_size = abs(float(profile["transform"].a))
    if not pixel_size:
        raise ValueError("Raster pixel size must be nonzero")
    if seed.shape != valid.shape or np.any(seed & ~valid):
        raise ValueError("Seed must match the raster and remain inside its valid domain")
    algorithm = _algorithm(pixel_size)
    geom = geometry.build(values, valid, profile, algorithm, seed, 5, 3, True)
    geom, _ = geometry.relative_resample(geom, values, valid, algorithm)
    scene_values = np.sort(values[valid])
    pairs = []
    for source in geom["sources"]:
        primary = _summarize(source["profiles"], scene_values)
        fallback = _summarize(source["fallback_profiles"], scene_values)
        selected = primary if primary is not None else fallback
        if selected is None:
            continue
        contrast = selected["scene_contrast"]
        if np.isfinite(contrast) and contrast < 0.15:
            continue
        pairs.append((source, {"selected": selected,
                               "evidence": "double" if primary is not None else "single"}))
    boxes = {source["candidate_id"]: tuple(int(v) for v in source["box"])
             for source, _ in pairs}
    removed, _ = geometry.greedy_nested_prune(boxes, None)
    pairs = [(source, record) for source, record in pairs
             if source["candidate_id"] not in removed]
    if not pairs:
        raise RuntimeError("No source passed the ordered-profile and contrast gates")
    return geom, pairs, len(removed)


def anchor_field(shape: tuple[int, int], valid: np.ndarray,
                 pairs: list) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.array([
        record["selected"]["reliability"] * record["selected"]["scene_contrast"]
        * (1.0 if record["evidence"] == "double" else 0.5)
        for _, record in pairs
    ])
    thresholds = np.array([record["selected"]["threshold"] for _, record in pairs])
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("Accepted profile sources have zero total evidence weight")
    tau = geometry.weighted_median(thresholds, weights)
    numerator = np.zeros(shape, dtype=float)
    denominator = np.zeros(shape, dtype=float)
    absence = np.ones(shape, dtype=float)
    for (source, _), threshold, weight in zip(pairs, thresholds, weights):
        r0, r1, c0, c1 = source["support_box"]
        window = np.s_[r0:r1, c0:c1]
        local_weight = np.asarray(source["weight"], dtype=float)*weight
        numerator[window] += local_weight*threshold
        denominator[window] += local_weight
        absence[window] *= 1.0-local_weight
    local = np.full(shape, tau, dtype=float)
    present = denominator > 0
    local[present] = numerator[present]/denominator[present]
    q = 1.0-absence
    field = tau+q*(local-tau)
    field[~valid] = np.nan
    return field, q, float(tau)


def profile_radius(pairs: list) -> float:
    terms = [float(profile["step_pixels"])+float(profile.get("band_samples", 0))
             for _, record in pairs for profile in record["selected"]["profiles"]]
    return float(np.ceil(np.median(terms)))


def run(values: np.ndarray, valid: np.ndarray, profile: dict) -> tuple[dict, dict]:
    """Execute GS-LAT on one raster; reference labels are not accepted or read."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or valid.shape != values.shape or int(valid.sum()) < 100:
        raise ValueError("Input must be 2-D with at least 100 valid pixels")
    print("Building native histogram seed and profile geometry...", flush=True)
    seed, seed_cut, seed_info = histogram_seed(values, valid)
    initial_prior = float(np.clip(seed.sum()/valid.sum(), 0.01, 0.99))
    geom, pairs, pruned = build_geometry(values, valid, profile, seed)
    anchor, support, tau = anchor_field(values.shape, valid, pairs)
    radius = profile_radius(pairs)
    print(f"Estimating empirical classes ({len(pairs)} supported sources)...", flush=True)
    prior, statistical_threshold, stat_info = empirical_threshold(
        values, valid, pairs, radius, initial_prior)
    # w_geom=q; q=0 gives T_stat. Statistical hesitation is not a fusion factor.
    geometry_weight = support
    threshold = (1.0-geometry_weight)*statistical_threshold + geometry_weight*anchor
    mask = clean_mask(valid & (values <= threshold), valid, 5)
    arrays = {
        "mask": mask,
        "seed": seed,
        "core": geom["core"],
        "edges": geom["edges"],
        "support": support > 0,
        "support_strength": support,
        "geometry_weight": geometry_weight,
        "anchor": anchor,
        "statistical_threshold": statistical_threshold,
        "threshold": threshold,
        "prior": prior,
    }
    metadata = {
        "method": "GS-LAT empirical two-class support fusion without threshold floor",
        "sar_log_conversion": False,
        "input_representation": "supplied native values",
        "seed_cut_native": seed_cut,
        "seed_fraction": initial_prior,
        "seed_diagnostics": seed_info,
        "candidate_sources": len(geom["sources"]),
        "accepted_sources": len(pairs),
        "nested_boxes_pruned": pruned,
        "geometry_anchor_global_median_native": tau,
        "profile_radius_pixels": radius,
        "support_fraction": float(np.count_nonzero((support > 0) & valid)/valid.sum()),
        "empirical_classes": stat_info,
        "fusion": "T=q*M+(1-q)*T_stat; no entropy factor; no dark-mean floor",
        "cleanup": "8-connected k4, minimum foreground size 5",
        "water_pixels": int(mask.sum()),
    }
    return arrays, metadata
