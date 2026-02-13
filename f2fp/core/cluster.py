from __future__ import annotations
import numpy as np

def _stack_metrics(fps: list[dict], keys: list[str]) -> np.ndarray:
    """Stack selected metrics from fingerprints into a matrix."""
    assert len(fps) > 0, "fps list is empty"
    M = []
    for d in fps:
        m = d['metrics']
        # Extract metrics for each fingerprint
        M.append([float(m.get(k, 0.0)) for k in keys])
    return np.asarray(M, dtype=float)

def embed_features(M: np.ndarray, n_components=2, random_state=42) -> np.ndarray:
    """Embed features using UMAP or PCA as fallback."""
    assert M.shape[0] > 0, "Input matrix M is empty"
    try:
        import umap
        # UMAP embedding
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(M)
    except Exception:
        # Fallback to PCA
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=random_state).fit_transform(M)

def cluster_labels(M: np.ndarray, min_cluster_size=20, random_state=42) -> np.ndarray:
    """Cluster embedded features using HDBSCAN, DBSCAN, or KMeans as fallback."""
    assert M.shape[0] > 0, "Input matrix M is empty"
    try:
        import hdbscan
        # HDBSCAN clustering
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(M)
    except Exception:
        from sklearn.cluster import DBSCAN, KMeans
        try:
            # DBSCAN clustering
            return DBSCAN(eps=0.5, min_samples=10).fit_predict(M)
        except Exception:
            # fallback: assign all to one cluster
            return np.zeros(M.shape[0], dtype=int)
