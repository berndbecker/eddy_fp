from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Pose:
    """Pose information for canonical alignment."""
    centroid: np.ndarray  # Mean position
    R: np.ndarray         # Rotation matrix from SVD
    scale_s: float        # Scaling factor

def pca_align(X: np.ndarray) -> tuple[np.ndarray, Pose]:
    """Align points using PCA and return canonical coordinates and pose."""
    assert X.ndim == 2 and X.shape[1] == 3, "Input X must be (N,3) array"
    assert X.shape[0] > 0, "Input X must not be empty"
    # Compute centroid
    c = X.mean(axis=0, keepdims=True)
    X0 = X - c
    # SVD for principal axes
    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    R = Vt
    # Rotate points
    Xr = X0 @ R.T
    # Compute scale from 95th percentile of radius
    r = np.linalg.norm(Xr, axis=1)
    s = np.percentile(r, 95) if np.all(np.isfinite(r)) and r.size else 1.0
    s = s if s > 0 else 1.0
    # Normalize points
    Xc = Xr / s
    return Xc, Pose(centroid=c.ravel(), R=R, scale_s=s)

def sparse_grid_12(Xc: np.ndarray) -> dict:
    """Create a sparse 12x12x12 grid representation from canonical coordinates."""
    assert Xc.ndim == 2 and Xc.shape[1] == 3, "Input Xc must be (N,3) array"
    b = 12
    # Map coordinates to grid indices
    idx = np.clip(((Xc + 1.0) * 0.5 * b).astype(int), 0, b-1)
    # Find unique grid cells and their counts
    keys, counts = np.unique(idx, axis=0, return_counts=True)
    payload = []
    total = counts.sum() if counts.size else 1
    for (i,j,k), c in zip(keys, counts):
        # Store cell indices and normalized count
        payload += [int(i), int(j), int(k), float(c/total)]
    return {'nx':12, 'ny':12, 'nz':12, 'nnz': int(keys.shape[0]), 'payload': payload}
