from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

PROM_FRAC = 0.08
DIP_MAX = 0.80


def _triangle_cut(smooth: np.ndarray, edges: np.ndarray) -> tuple[float, int, int, int]:
    peak = int(np.argmax(smooth))
    nonzero = np.flatnonzero(smooth[: peak + 1] > 0)
    start = int(nonzero[0]) if len(nonzero) else 0
    xs = np.arange(start, peak + 1)
    ys = smooth[start : peak + 1]
    x1, y1 = float(start), float(smooth[start])
    x2, y2 = float(peak), float(smooth[peak])
    numerator = np.abs((y2-y1)*xs - (x2-x1)*ys + x2*y1-y2*x1)
    denominator = np.hypot(y2-y1, x2-x1) + 1e-12
    k = int(xs[int(np.argmax(numerator / denominator))])
    return float(edges[k+1]), k, start, peak


def _histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    lo, hi = np.percentile(values, [0.1, 99.9])
    if not np.isfinite(lo) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        raise ValueError("A nonconstant image is required for the native histogram seed")
    q5, q40 = np.percentile(values, [5, 40])
    band = values[(values >= q5) & (values <= q40)]
    if band.size > 400_000:
        band = band[:: int(band.size // 400_000)]
    unique = np.unique(np.round(band.astype(np.float64), 5))
    gap = float(np.median(np.diff(unique))) if unique.size >= 8 else 0.0
    width = max((float(hi)-float(lo))/256.0, 2.0*max(gap, 0.0))
    bins = int(np.clip(round((float(hi)-float(lo))/width), 48, 256))
    hist, edges = np.histogram(values, bins=bins, range=(float(lo), float(hi)))
    bin_width = float(edges[1]-edges[0])
    sigma = float(np.clip((2.0*gap/bin_width) if gap else 1.5, 1.5, bins/16.0))
    smooth = gaussian_filter1d(hist.astype(np.float64), sigma)
    return smooth, edges, sigma, gap, bins


def _choose_cut(smooth: np.ndarray, edges: np.ndarray) -> tuple[float, dict]:
    prominence = PROM_FRAC*float(smooth.max())
    distance = max(6, len(smooth)//20)
    indices, props = find_peaks(smooth, prominence=prominence, distance=distance)
    detail = {
        "n_peaks": int(len(indices)),
        "peak_bins": [int(i) for i in indices],
        "peak_values": [float(edges[i]) for i in indices],
        "prominences": [float(v) for v in props["prominences"]] if len(indices) else [],
    }
    if len(indices) >= 2:
        order = np.argsort(props["prominences"])[::-1]
        left, right = sorted((int(indices[order[0]]), int(indices[order[1]])))
        valley = left + int(np.argmin(smooth[left:right+1]))
        dip = float(smooth[valley]/max(min(smooth[left], smooth[right]), 1e-12))
        separated = ((valley-left) >= max(6, len(smooth)//20)
                     and (right-valley) >= max(6, len(smooth)//20))
        detail.update(pair=[left, right], valley_bin=valley, dip=dip, separated=separated)
        if dip <= DIP_MAX and separated:
            detail["rule"] = "interpeak-valley"
            return float(edges[valley+1]), detail
    cut, index, start, peak = _triangle_cut(smooth, edges)
    detail.update(rule="triangle", triangle_bin=index, mode_bin=peak, dark_bin=start)
    return cut, detail


def _repaired_seed(values: np.ndarray, valid: np.ndarray, upper: float) -> tuple[float, dict]:
    sample = values[valid]
    lo, hi = np.percentile(sample, [0.1, upper])
    if hi <= lo:
        raise RuntimeError("Degenerate native histogram range")
    q5, q40 = np.percentile(sample, [5, 40])
    band = sample[(sample >= q5) & (sample <= q40)]
    if band.size > 400_000:
        band = band[:: max(1, band.size//400_000)]
    unique = np.unique(np.round(band, 5))
    gap = float(np.median(np.diff(unique))) if unique.size >= 8 else 0.0
    width = max((hi-lo)/256.0, 2.0*max(gap, 0.0))
    bins = int(np.clip(round((hi-lo)/width), 48, 256))
    hist, edges = np.histogram(sample, bins=bins, range=(float(lo), float(hi)))
    sigma = float(np.clip(2.0*gap/(edges[1]-edges[0]) if gap else 1.5, 1.5, bins/16.0))
    smooth = gaussian_filter1d(hist.astype(float), sigma)
    cut, detail = _choose_cut(smooth, edges)
    if detail["rule"] == "triangle":
        peak = int(np.argmax(smooth))
        endpoint = 0 if peak >= bins-1-peak else bins-1
        left, right = sorted((peak, endpoint))
        if right-left < 2:
            raise RuntimeError("No nondegenerate triangle tail")
        x = np.arange(left, right+1, dtype=float)
        y = smooth[left:right+1]
        chord = y[0] + (y[-1]-y[0])*(x-left)/(right-left)
        index = left + int(np.argmax(np.abs(y-chord)))
        cut = float(edges[index+1])
        detail.update(rule="longest-tail-triangle", triangle_bin=index, mode_bin=peak,
                      tail_endpoint_bin=endpoint, tail="dark" if endpoint == 0 else "bright")
    detail.update(histogram_lower_percentile=0.1, histogram_upper_percentile=upper,
                  bins=bins, smoothing_sigma=sigma, value_gap=gap)
    return float(cut), detail


def histogram_seed(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Build the selected native-value histogram seed, without transforming SAR values."""
    sample = np.asarray(values[valid], dtype=np.float64)
    if sample.size < 2 or not np.isfinite(sample).all():
        raise ValueError("At least two finite valid SAR values are required")
    smooth, edges, sigma, gap, bins = _histogram(sample)
    cut, detail = _choose_cut(smooth, edges)
    detail.update(smoothing_sigma=sigma, value_gap=gap, bins=bins)
    if detail["rule"] == "triangle" and detail.get("triangle_bin") == 0:
        cut, detail = _repaired_seed(values, valid, 99.0)
        detail["first_bin_repair"] = True
    seed = valid & (values <= cut)
    if not seed.any() or np.all(seed[valid]):
        raise RuntimeError("Native histogram seed is empty or covers the full valid image")
    return seed, float(cut), detail
