from __future__ import annotations
import numpy as np

def _stack_metrics(fps: list[dict], keys: list[str]) -> np.ndarray:
    M = []
    for d in fps:
        m = d['metrics']
        M.append([float(m.get(k, 0.0)) for k in keys])
    return np.asarray(M, dtype=float)

def embed_features(M: np.ndarray, n_components=2, random_state=42) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(M)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=random_state).fit_transform(M)

def cluster_labels(M: np.ndarray, min_cluster_size=20, random_state=42) -> np.ndarray:
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(M)
    except Exception:
        from sklearn.cluster import DBSCAN, KMeans
        try:
            return DBSCAN(eps=0.5, min_samples=10).fit_predict(M)
        except Exception:
            return KMeans(n_clusters=2, random_state=random_state).fit_predict(M)
