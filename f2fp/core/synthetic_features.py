import numpy as np
import json
from pathlib import Path

from core.synthetic import SynthConfig, ring_points, curtain_points, mix_with_noise
from core.fingerprints import build_fingerprint, to_dict, _stack_metrics
from core.cluster import embed_features, cluster_labels

def make_synthetic_samples(n_samples=10, label="ring", cfg=None):
    """Generate synthetic samples and return list of fingerprint dicts."""
    fps = []
    for i in range(n_samples):
        if cfg is None:
            cfg = SynthConfig(seed=42 + i)
        else:
            cfg = SynthConfig(**{**cfg.__dict__, "seed": cfg.seed + i})
        if label == "ring":
            lon, lat = ring_points(cfg)
        elif label == "curtain":
            lon, lat = curtain_points(cfg)
        else:
            raise ValueError("Unknown label")
        lon, lat = mix_with_noise(lon, lat, cfg)
        # Stack as (N,3) with dummy z=0
        X = np.stack([lon, lat, np.zeros_like(lon)], axis=1)
        fp = build_fingerprint(X, label=label)
        fps.append(to_dict(fp))
    return fps

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def make_synthetic_features(n_ring=10, n_curtain=10, cfg=None):
    """Generate synthetic ring and curtain fingerprints as dicts."""
    features = []
    for i in range(n_ring):
        cfg_ring = SynthConfig(seed=42 + i) if cfg is None else SynthConfig(**{**cfg.__dict__, "seed": cfg.seed + i})
        lon, lat = ring_points(cfg_ring)
        lon, lat = mix_with_noise(lon, lat, cfg_ring)
        X = np.stack([lon, lat, np.zeros_like(lon)], axis=1)
        fp = build_fingerprint(X, label="ring")
        features.append(to_dict(fp))
    for i in range(n_curtain):
        cfg_curtain = SynthConfig(seed=142 + i) if cfg is None else SynthConfig(**{**cfg.__dict__, "seed": cfg.seed + 100 + i})
        lon, lat = curtain_points(cfg_curtain)
        lon, lat = mix_with_noise(lon, lat, cfg_curtain)
        X = np.stack([lon, lat, np.zeros_like(lon)], axis=1)
        fp = build_fingerprint(X, label="curtain")
        features.append(to_dict(fp))
    return features

def save_features_json(features, path):
    with open(path, "w") as f:
        json.dump(features, f, indent=2)

def main():
    # Generate and store synthetic features
    rings = make_synthetic_samples(n_samples=10, label="ring")
    curtains = make_synthetic_samples(n_samples=10, label="curtain")
    all_fps = rings + curtains
    out_path = Path(__file__).parent / "synthetic_features.json"
    save_json(all_fps, out_path)

    # Read features and test clustering
    fps = load_json(out_path)
    keys = ["linearity", "planarity", "scattering", "circularity", "nnz"]
    M = _stack_metrics(fps, keys)
    emb = embed_features(M)
    labels = cluster_labels(emb)
    print("Cluster labels:", labels)
    print("Label counts:", {int(l): int(np.sum(labels==l)) for l in np.unique(labels)})

if __name__ == "__main__":
    main()
