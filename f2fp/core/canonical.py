from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Pose:
    centroid: np.ndarray
    R: np.ndarray
    scale_s: float

def pca_align(X: np.ndarray) -> tuple[np.ndarray, Pose]:
    assert X.ndim == 2 and X.shape[1] == 3
    c = X.mean(axis=0, keepdims=True)
    X0 = X - c
    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    R = Vt
    Xr = X0 @ R.T
    r = np.linalg.norm(Xr, axis=1)
    s = np.percentile(r, 95) if np.all(np.isfinite(r)) and r.size else 1.0
    s = s if s > 0 else 1.0
    Xc = Xr / s
    return Xc, Pose(centroid=c.ravel(), R=R, scale_s=s)

def sparse_grid_12(Xc: np.ndarray) -> dict:
    b = 12
    idx = np.clip(((Xc + 1.0) * 0.5 * b).astype(int), 0, b-1)
    keys, counts = np.unique(idx, axis=0, return_counts=True)
    payload = []
    total = counts.sum() if counts.size else 1
    for (i,j,k), c in zip(keys, counts):
        payload += [int(i), int(j), int(k), float(c/total)]
    return {'nx':12, 'ny':12, 'nz':12, 'nnz': int(keys.shape[0]), 'payload': payload}
