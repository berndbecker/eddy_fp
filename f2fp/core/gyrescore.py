from __future__ import annotations

def gyrescore(metrics: dict) -> float:
    """Compute a gyre score based on metrics dict."""
    try:
        # Extract metrics, default to 0.0 if missing
        circ = float(metrics.get('circularity', 0.0))
        plan = float(metrics.get('planarity', 0.0))
        scat = float(metrics.get('scattering', 0.0))
    except Exception as e:
        raise ValueError(f"Invalid metrics dict: {metrics}") from e
    # Weighted sum for score, penalize scattering
    score = 0.55*circ + 0.35*plan + 0.10*max(0.0, 1.0 - scat*3.0)
    # Clamp score to [0, 1]
    return max(0.0, min(1.0, score))

def gyreclass(metrics: dict, threshold: float = 0.6) -> str:
    """Classify gyre as ring-like or non-ring based on score threshold."""
    # Compare score to threshold
    return "ring-like" if gyrescore(metrics) >= threshold else "non-ring"
