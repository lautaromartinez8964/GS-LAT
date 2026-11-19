from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter
from sklearn.isotonic import IsotonicRegression

SHRINK = 0.5


def _accepted_profile_rows(source: dict, record: dict) -> list[tuple[dict, dict]]:
    key = "profiles" if record["evidence"] == "double" else "fallback_profiles"
    raw_map = {(p["profile_index"], p["kind"]): p for p in source[key]}
    rows = []
    for result in record["selected"]["profiles"]:
        if result.get("status") != "accepted" or not result.get("aggregation_inlier", True):
            continue
        raw = raw_map.get((result["profile_index"], result["kind"]))
        if raw is not None:
            rows.append((result, raw))
    return rows


def _valid_array(samples: np.ndarray, indices) -> np.ndarray:
    values = np.asarray(samples[np.asarray(indices, dtype=int)], dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _class_histograms(pairs: list, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    water = np.zeros(len(edges)-1, dtype=float)
    land = np.zeros_like(water)
    used_sources = used_profiles = core_n = bank_n = 0
    for source, record in pairs:
        profiles = []
        for result, raw in _accepted_profile_rows(source, record):
            samples = np.asarray(raw["samples"], dtype=np.float64)
            core = _valid_array(samples, result["core_indices"])
            parts = [_valid_array(samples, ix) for ix in result["bank_indices"]]
            bank = np.concatenate(parts) if parts else np.empty(0)
            if min(core.size, bank.size) >= 8:
                profiles.append((core, bank))
        selected = record["selected"]
        evidence_factor = 1.0 if record["evidence"] == "double" else 0.5
        source_weight = float(selected["reliability"] * selected["scene_contrast"] * evidence_factor)
        if not profiles or source_weight <= 0:
            continue
        used_sources += 1
        for core, bank in profiles:
            water += source_weight/len(profiles) * np.histogram(core, edges)[0] / len(core)
            land += source_weight/len(profiles) * np.histogram(bank, edges)[0] / len(bank)
            used_profiles += 1
            core_n += len(core)
            bank_n += len(bank)
    if water.sum() <= 0 or land.sum() <= 0:
        raise RuntimeError("No usable geometry pseudo samples for empirical classes")
    return water/water.sum(), land/land.sum(), dict(sources=used_sources, profiles=used_profiles,
        core_samples=core_n, bank_samples=bank_n)


def _project_classes(water: np.ndarray, land: np.ndarray,
                     scene: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    # Smooth the sampled class counts and add a small shared scene-mass pseudocount.
    w = gaussian_filter1d(water.astype(float), 1.0, mode="nearest")
    l = gaussian_filter1d(land.astype(float), 1.0, mode="nearest")
    w = w/w.sum() + 0.001*scene + 1e-12
    l = l/l.sum() + 0.001*scene + 1e-12
    w /= w.sum()
    l /= l.sum()
    mass = w+l
    ratio = w/mass
    monotone = IsotonicRegression(increasing=False).fit_transform(
        np.arange(len(w)), ratio, sample_weight=mass)
    wp = np.maximum(mass*monotone, 1e-12)
    lp = np.maximum(mass*(1-monotone), 1e-12)
    wp /= wp.sum()
    lp /= lp.sum()
    difference = wp-lp
    global_prior = float(np.clip(np.dot(scene-lp, difference)
                                 / max(np.dot(difference, difference), 1e-20), 0.01, 0.99))
    diagnostics = dict(projected_l1_change=float(np.abs(wp-w).sum()+np.abs(lp-l).sum()),
                       scene_fit_l1=float(np.abs(scene-(global_prior*wp+(1-global_prior)*lp)).sum()))
    return wp, lp, diagnostics


def _threshold_lookup(edges: np.ndarray, water: np.ndarray,
                      land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    priors = np.linspace(0.0, 1.0, 1001)
    fw = np.r_[0.0, np.cumsum(water)]
    fl = np.r_[0.0, np.cumsum(land)]
    risk = priors[:, None]*(1.0-fw) + (1.0-priors[:, None])*fl
    cuts = edges[np.argmin(risk, axis=1)].copy()
    cuts[cuts == edges[0]] = np.nextafter(edges[0], -np.inf)
    if (np.diff(cuts) < 0).any():
        raise RuntimeError("Empirical risk-threshold lookup must be monotone")
    return priors, cuts


def _update_prior(prior: np.ndarray, indices: np.ndarray, water: np.ndarray,
                  land: np.ndarray, valid: np.ndarray, window: int,
                  global_prior: float, denominator: np.ndarray) -> np.ndarray:
    water_likelihood = water[indices].astype(np.float32)
    land_likelihood = land[indices].astype(np.float32)
    a = np.asarray(prior, dtype=np.float32)*water_likelihood
    b = (1.0-np.asarray(prior, dtype=np.float32))*land_likelihood
    posterior = np.divide(a, a+b, out=np.zeros_like(a), where=(a+b) > 0)
    posterior[~valid] = 0.0
    local = uniform_filter(posterior, size=window, mode="constant")
    np.divide(local, denominator, out=local, where=denominator > 0)
    local = np.clip(SHRINK*local+(1.0-SHRINK)*global_prior, 0.01, 0.99)
    local[~valid] = global_prior
    return local.astype(np.float32)


def empirical_threshold(values: np.ndarray, valid: np.ndarray, pairs: list,
                        radius: float, initial_prior: float) -> tuple[np.ndarray, np.ndarray, dict]:
    scene = values[valid]
    edges = np.unique(np.quantile(scene, np.linspace(0.0, 1.0, 257)))
    if edges.size < 2:
        raise RuntimeError("A nonconstant image is required for empirical thresholds")
    indices = np.searchsorted(edges[1:-1], values, side="right")
    scene_hist = np.bincount(indices[valid], minlength=len(edges)-1).astype(float)
    scene_hist /= scene_hist.sum()
    water, land, sampling = _class_histograms(pairs, edges)
    water, land, projection = _project_classes(water, land, scene_hist)
    window = 2*max(1, int(np.ceil(radius))) + 1
    denominator = uniform_filter(valid.astype(np.float32), window, mode="constant")
    prior = np.full(values.shape, initial_prior, dtype=np.float32)
    prior = _update_prior(prior, indices, water, land, valid, window,
                          initial_prior, denominator)
    priors, lookup = _threshold_lookup(edges, water, land)
    threshold = np.interp(prior, priors, lookup)
    diagnostics = dict(**sampling, **projection, bins=int(edges.size-1),
                       window=window, initial_prior=float(initial_prior),
                       update_count=1, pseudocount_fraction=0.001,
                       class_balance="equal profile-normalized class histograms",
                       threshold_rule="minimum equal-cost Bayesian risk")
    return prior, threshold, diagnostics
