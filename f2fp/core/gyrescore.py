from __future__ import annotations

def gyrescore(metrics: dict) -> float:
    circ = float(metrics.get('circularity', 0.0))
    plan = float(metrics.get('planarity', 0.0))
    scat = float(metrics.get('scattering', 0.0))
    score = 0.55*circ + 0.35*plan + 0.10*max(0.0, 1.0 - scat*3.0)
    return max(0.0, min(1.0, score))

def gyreclass(metrics: dict, threshold: float = 0.6) -> str:
    return "ring-like" if gyrescore(metrics) >= threshold else "non-ring"
