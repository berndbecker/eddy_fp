from __future__ import annotations
import numpy as np

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
        n_samples, n_features = M.shape
        n_comp = max(1, min(int(n_components), n_samples, n_features))
        emb = PCA(n_components=n_comp, random_state=random_state).fit_transform(M)
        if emb.shape[1] == 1 and int(n_components) == 2:
            emb = np.hstack([emb, np.zeros((emb.shape[0], 1), dtype=emb.dtype)])
        return emb

def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score safely; return -1.0 on invalid cases."""
    try:
        from sklearn.metrics import silhouette_score
        # Need at least 2 clusters (excluding noise)
        uniq = [u for u in np.unique(labels) if u != -1]
        if len(uniq) < 2:
            return -1.0
        return float(silhouette_score(X, labels))
    except Exception:
        return -1.0

def cluster_labels(M: np.ndarray, min_cluster_size=20, random_state=42, force_n_clusters: int | None = None) -> np.ndarray:
    """Cluster embedded features using HDBSCAN, DBSCAN, or KMeans as fallback."""
    assert M.shape[0] > 0, "Input matrix M is empty"
    try:
        import hdbscan
        # HDBSCAN clustering
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(M)
    except Exception:
        from sklearn.cluster import DBSCAN, KMeans
        try:
            # DBSCAN clustering
            return DBSCAN(eps=0.5, min_samples=10).fit_predict(M)
        except Exception:
            # fallback: KMeans if DBSCAN fails
            return KMeans(n_clusters=2, random_state=random_state).fit_predict(M)

    if force_n_clusters is not None:
        uniq = [u for u in np.unique(labels) if u != -1]
        if len(uniq) < force_n_clusters:
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=force_n_clusters, random_state=random_state).fit_predict(M)

    return labels

def cluster_labels_hdbscan_optimized(M: np.ndarray,
                                    min_cluster_size=20,
                                    random_state=42,
                                    max_steps=8) -> tuple[np.ndarray, int, float]:
    """
    Hill-climb (gradient descent-style) over min_cluster_size to maximize silhouette.
    Returns (labels, best_min_cluster_size, best_score).
    """
    assert M.shape[0] > 0, "Input matrix M is empty"
    try:
        import hdbscan
    except Exception:
        # fallback to existing clustering if hdbscan is unavailable
        return cluster_labels(M, min_cluster_size=min_cluster_size, random_state=random_state), min_cluster_size, -1.0

    best_mcs = max(2, int(min_cluster_size))
    best_labels = hdbscan.HDBSCAN(min_cluster_size=best_mcs).fit_predict(M)
    best_score = _silhouette_safe(M, best_labels)

    step = max(1, best_mcs // 2)
    for _ in range(max_steps):
        improved = False
        for candidate in (best_mcs - step, best_mcs + step):
            if candidate < 2:
                continue
            labels = hdbscan.HDBSCAN(min_cluster_size=candidate).fit_predict(M)
            score = _silhouette_safe(M, labels)
            if score > best_score:
                best_mcs, best_labels, best_score = candidate, labels, score
                improved = True
        if not improved:
            step //= 2
            if step == 0:
                break
    return best_labels, best_mcs, best_score

def available_backends() -> dict:
    """Return availability of optional clustering/embedding backends."""
    def _has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    return {
        "umap": _has("umap"),
        "hdbscan": _has("hdbscan"),
        "sklearn": _has("sklearn"),
    }
