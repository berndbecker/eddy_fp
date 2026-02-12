from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .canonical import pca_align, sparse_grid_12, Pose

@dataclass
class Fingerprint:
    pose: Pose
    grid: dict
    metrics: dict
    label: str | None = None

def _basic_metrics(Xc: np.ndarray, grid: dict) -> dict:
    cov = np.cov(Xc.T) if Xc.shape[0] > 3 else np.eye(3)
    w, _ = np.linalg.eigh(cov)
    w = np.sort(np.maximum(w, 1e-12))[::-1]
    linearity  = (w[0]-w[1])/w[0]
    planarity  = (w[1]-w[2])/w[0]
    scattering = w[2]/w[0]
    r = np.linalg.norm(Xc[:,:2], axis=1) if Xc.size else np.array([0.0])
    circularity = float(np.std(r) < 0.25)
    return dict(linearity=float(linearity),
                planarity=float(planarity),
                scattering=float(scattering),
                circularity=circularity,
                nnz=int(grid['nnz']))

def build_fingerprint(X: np.ndarray, label: str | None = None) -> Fingerprint:
    Xc, pose = pca_align(X)
    grid = sparse_grid_12(Xc)
    metrics = _basic_metrics(Xc, grid)
    return Fingerprint(pose=pose, grid=grid, metrics=metrics, label=label)

def to_dict(fp: Fingerprint) -> dict:
    return {
        'pose': dict(centroid=fp.pose.centroid.tolist(),
                     R=fp.pose.R.tolist(),
                     scale_s=float(fp.pose.scale_s)),
        'grid': fp.grid,
        'metrics': fp.metrics,
        'label': fp.label
    }
