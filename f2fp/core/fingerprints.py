from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .canonical import pca_align, sparse_grid_12, Pose

@dataclass
class Fingerprint:
    """Fingerprint representation for an eddy."""
    pose: Pose
    grid: dict
    metrics: dict
    label: str | None = None

def _basic_metrics(Xc: np.ndarray, grid: dict) -> dict:
    """Compute basic shape metrics for a point cloud."""
    assert 'nnz' in grid, "grid must contain 'nnz'"
    # Covariance for shape analysis
    cov = np.cov(Xc.T) if Xc.shape[0] > 3 else np.eye(3)
    # Eigenvalues for principal axes
    w, _ = np.linalg.eigh(cov)
    w = np.sort(np.maximum(w, 1e-12))[::-1]
    # Linearity, planarity, scattering metrics
    linearity  = (w[0]-w[1])/w[0]
    planarity  = (w[1]-w[2])/w[0]
    scattering = w[2]/w[0]
    # Radial distances for circularity
    r = np.linalg.norm(Xc[:,:2], axis=1) if Xc.size else np.array([0.0])
    circularity = float(np.std(r) < 0.25)
    return dict(linearity=float(linearity),
                planarity=float(planarity),
                scattering=float(scattering),
                circularity=circularity,
                nnz=int(grid['nnz']))

def build_fingerprint(X: np.ndarray, label: str | None = None) -> Fingerprint:
    """Build a fingerprint from input points."""
    assert X.ndim == 2 and X.shape[1] == 3, "Input X must be (N,3) array"
    # Canonical alignment
    Xc, pose = pca_align(X)
    # Sparse grid representation
    grid = sparse_grid_12(Xc)
    # Compute metrics
    metrics = _basic_metrics(Xc, grid)
    return Fingerprint(pose=pose, grid=grid, metrics=metrics, label=label)

def to_dict(fp: Fingerprint) -> dict:
    """Convert Fingerprint object to dictionary."""
    # Convert pose and other fields to dict
    return {
        'pose': dict(centroid=fp.pose.centroid.tolist(),
                     R=fp.pose.R.tolist(),
                     scale_s=float(fp.pose.scale_s)),
        'grid': fp.grid,
        'metrics': fp.metrics,
        'label': fp.label
    }
