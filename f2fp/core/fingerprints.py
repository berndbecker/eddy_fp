from __future__ import annotations
import numpy as np
#from fast_hdbscan import HDBSCAN
from dataclasses import dataclass
try:
    from .canonical import pca_align, sparse_grid_12, Pose
except Exception:
    from canonical import pca_align, sparse_grid_12, Pose

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

def _coerce_point_cloud(feat) -> np.ndarray:
    """Coerce feature input into (N,3) point cloud."""
    X = np.asarray(feat, dtype=object)
    if isinstance(feat, (list, tuple)):
        if len(feat) >= 4 and np.ndim(feat[0]) == 0 and np.ndim(feat[1]) >= 1:
            a, b, c = feat[1], feat[2], feat[3]
        elif len(feat) >= 3:
            a, b, c = feat[0], feat[1], feat[2]
        else:
            raise ValueError("Feature must contain at least 3 components")
        X = np.column_stack([np.asarray(a), np.asarray(b), np.asarray(c)])
    else:
        X = np.asarray(feat)

    if X.ndim == 1 and X.size == 3:
        X = X.reshape(1, 3)
    elif X.ndim == 2 and X.shape == (3, 1):
        X = X.T
    return np.asarray(X, dtype=float)

def build_fingerprint(X: np.ndarray, label: str | None = None) -> Fingerprint:
    X = _coerce_point_cloud(X)
    assert X.ndim == 2 and X.shape[1] == 3, "Input X must be (N,3) array"
    assert X.shape[0] > 0, "Input X must not be empty"
    Xc, pose = pca_align(X)
    grid = sparse_grid_12(Xc)
    metrics = _basic_metrics(Xc, grid)
    return Fingerprint(pose=pose, grid=grid, metrics=metrics, label=label)

def _normalize_metrics(metrics: dict) -> dict:
    """Normalize metrics keys and types for JSON stability."""
    return {
        "linearity": float(metrics.get("linearity", 0.0)),
        "planarity": float(metrics.get("planarity", 0.0)),
        "scattering": float(metrics.get("scattering", 0.0)),
        "circularity": float(metrics.get("circularity", 0.0)),
        "nnz": int(metrics.get("nnz", 0)),
    }

def _normalize_grid(grid: dict) -> dict:
    """Normalize grid keys and payload types for JSON stability."""
    return {
        "nx": int(grid.get("nx", 0)),
        "ny": int(grid.get("ny", 0)),
        "nz": int(grid.get("nz", 0)),
        "nnz": int(grid.get("nnz", 0)),
        "payload": [float(x) if i % 4 == 3 else int(x)
                    for i, x in enumerate(grid.get("payload", []))],
    }

def to_dict(fp: Fingerprint) -> dict:
    return {
        'pose': dict(centroid=fp.pose.centroid.tolist(),
                     R=fp.pose.R.tolist(),
                     scale_s=float(fp.pose.scale_s)),
        'grid': _normalize_grid(fp.grid),
        'metrics': _normalize_metrics(fp.metrics),
        'label': fp.label
    }

def _stack_metrics(fps: list[dict], keys: list[str]) -> np.ndarray:
    """Stack selected metrics from fingerprints into a matrix."""
    assert len(fps) > 0, "fps list is empty"
    M = []
    for d in fps:
        m = d['metrics']
        # Extract metrics for each fingerprint
        M.append([float(m.get(k, 0.0)) for k in keys])
    return np.asarray(M, dtype=float)

def load_fingerprints_json(path: str) -> list[dict]:
    """Load fingerprints from a JSON or JSONL file."""
    import json
    fps = []
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.read(1)
        fh.seek(0)
        if head == "[":
            fps = json.load(fh)
        else:
            for line in fh:
                line = line.strip()
                if line:
                    fps.append(json.loads(line))
    return fps

def load_metrics_matrix(path: str, keys: list[str]) -> np.ndarray:
    """Load fingerprints from JSON/JSONL and return stacked metrics matrix."""
    fps = load_fingerprints_json(path)
    return _stack_metrics(fps, keys)

def plot_features_from_json(path: str,
                            keys: list[str] | None = None,
                            title: str = "Embedded features",
                            show: bool = True):
    """Load JSON/JSONL fingerprints and plot embedded features."""
    import matplotlib.pyplot as plt
    try:
        from .cluster import embed_features
    except Exception:
        from cluster import embed_features

    fps = load_fingerprints_json(path)
    keys = keys or ["linearity", "planarity", "scattering", "circularity", "nnz"]
    M = _stack_metrics(fps, keys)
    emb = embed_features(M)

    labels = [fp.get("label", "unknown") for fp in fps]
    print(labels)
    label_set = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(label_set)}
    color_vals = [label_to_int[l] for l in labels]

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=color_vals, cmap="tab10", s=40)
    handles = [plt.Line2D([], [], marker="o", color="w",
                          markerfacecolor=scatter.cmap(i),
                          label=label, markersize=8)
               for i, label in enumerate(label_set)]
    plt.legend(handles=handles, title="Label")
    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("cluster_scatter.png")
    if show:
        plt.show()
    return emb

def load_point_clouds_json(path: str, key: str = "points") -> list[np.ndarray]:
    """Load point cloud(s) from JSON or JSONL."""
    import json
    clouds = []
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.read(1)
        fh.seek(0)
        data = json.load(fh) if head == "[" else [json.loads(line) for line in fh if line.strip()]
    # If list of points (Nx3)
    if data and isinstance(data, list) and isinstance(data[0], (list, tuple)) and len(data[0]) == 3:
        clouds.append(np.asarray(data, dtype=float))
        return clouds
    # If list of dicts with point arrays
    for rec in data:
        pts = rec.get(key, None) if isinstance(rec, dict) else None
        if pts is not None:
            clouds.append(np.asarray(pts, dtype=float))
    return clouds

def save_fingerprints_json(fps: list[dict], path: str) -> None:
    """Save fingerprints to JSON or JSONL, preserving structure."""
    import json
    is_jsonl = path.endswith(".jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        if is_jsonl:
            for rec in fps:
                fh.write(json.dumps(rec))
                fh.write("\n")
        else:
            json.dump(fps, fh, indent=2)

def _cluster_points_raw(points: np.ndarray, min_cluster_size=20, eps=0.5) -> np.ndarray:
    """Cluster raw points to select features (HDBSCAN -> DBSCAN fallback)."""
    cluster_selection_method = "leaf"
    algorithm = "kdtree"
    max_cluster_size = int(len(points) * 0.5)
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               cluster_selection_method=cluster_selection_method,
                               algorithm=algorithm,
                               max_cluster_size=max_cluster_size,).fit_predict(points)
    except Exception:
        try:
            from sklearn.cluster import DBSCAN
            return DBSCAN(eps=eps, min_samples=max(2, min_cluster_size // 2)).fit_predict(points)
        except Exception:
            return np.zeros(points.shape[0], dtype=int)

def _split_clusters(points: np.ndarray, labels: np.ndarray, min_size: int = 1) -> list[np.ndarray]:
    """Split points into clusters, dropping noise (-1)."""
    points = np.asarray(points)
    labels = np.asarray(labels)
    if points.ndim != 2 or points.shape[1] != 3:
        return []
    if labels.shape[0] != points.shape[0]:
        return []
    clusters = []
    for lbl in sorted(set(labels.tolist())):
        if lbl == -1:
            continue
        mask = labels == lbl
        if mask.sum() < max(1, int(min_size)):
            continue
        clusters.append(points[mask])
    return clusters

def _silhouette_safe_points(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score safely; return -1.0 on invalid cases."""
    try:
        from sklearn.metrics import silhouette_score
        uniq = [u for u in np.unique(labels) if u != -1]
        if len(uniq) < 2:
            return -1.0
        return float(silhouette_score(X, labels))
    except Exception:
        return -1.0

def _cluster_points_raw_adaptive(points: np.ndarray,
                                 min_cluster_size=20,
                                 eps=0.5) -> np.ndarray:
    """Try HDBSCAN and DBSCAN(auto-eps) and pick the best by silhouette."""
    points = np.asarray(points, dtype=float)
    best_labels = _cluster_points_raw(points, min_cluster_size=min_cluster_size, eps=eps)
    best_score = _silhouette_safe_points(points, best_labels)

    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.cluster import DBSCAN
        nn = NearestNeighbors(n_neighbors=6).fit(points)
        dists, _ = nn.kneighbors(points)
        kdist = np.sort(dists[:, -1])
        for q in (50, 70, 90):
            cand_eps = float(np.percentile(kdist, q))
            labels = DBSCAN(eps=cand_eps, min_samples=max(2, min_cluster_size // 2)).fit_predict(points)
            score = _silhouette_safe_points(points, labels)
            if score > best_score:
                best_labels, best_score = labels, score
    except Exception:
        pass

    return best_labels

def save_point_clouds_jsonl(path: str, clouds: list[np.ndarray]) -> None:
    """Save list of point clouds as JSONL (one cloud per line)."""
    import json
    with open(path, "w", encoding="utf-8") as fh:
        for i, pts in enumerate(clouds):
            rec = {"feature_id": int(i), "points": np.asarray(pts, dtype=float).tolist()}
            fh.write(json.dumps(rec))
            fh.write("\n")

def save_point_clouds_vtk(prefix: str, clouds: list[np.ndarray]) -> None:
    """Save list of point clouds as separate .vtk files."""
    try:
        from .synthetic import save_point_cloud_vtk
    except Exception:
        from synthetic import save_point_cloud_vtk
    for i, pts in enumerate(clouds):
        save_point_cloud_vtk(f"{prefix}_{i}.vtk", np.asarray(pts, dtype=float))

def pipeline_cluster_points(points: np.ndarray,
                            min_cluster_size=20,
                            eps=0.5,
                            random_state=42,
                            clusters_out: str | None = None,
                            clusters_out_vtk: str | None = None,
                            max_points: int | None = None,
                            parallel: bool = True,
                            max_workers: int | None = None,
                            first_stage: str = "iterative") -> list[dict]:
    """
    Two-stage clustering:
      1) HDBSCAN/DBSCAN on raw points to select features.
      2) Build fingerprints per feature.
      3) HDBSCAN/DBSCAN/KMeans on fingerprint embeddings to label eddies.
    Returns list of fingerprint dicts with consistent label strings.
    """
    try:
        from .cluster import embed_features, cluster_labels
    except Exception:
        from cluster import embed_features, cluster_labels

    points = _coerce_point_cloud(points)
    if points.size == 0:
        return []
    if max_points and points.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    if first_stage == "iterative":
        labels1 = _cluster_points_raw_iterative(points,
                                                min_cluster_size=min_cluster_size,
                                                min_samples=max(2, min_cluster_size // 5))
    elif first_stage == "adaptive":
        labels1 = _cluster_points_raw_adaptive(points,
                                               min_cluster_size=min_cluster_size,
                                               eps=eps)
    else:
        labels1 = _cluster_points_raw(points,
                                      min_cluster_size=min_cluster_size,
                                      eps=eps)

    clusters = _split_clusters(points, labels1)
    if not clusters and points.size:
        clusters = [points]

    if clusters_out:
        out = clusters_out.lower()
        if out.endswith(".vtk"):
            prefix = clusters_out[:-4]
            save_point_clouds_vtk(prefix, clusters)
            print("features saved to ", prefix, ".vtk")
        else:
            save_point_clouds_jsonl(clusters_out, clusters)
            print("features saved to ", clusters_out)
    if clusters_out_vtk:
        save_point_clouds_vtk(clusters_out_vtk, clusters)
        print("features saved to ", clusters_out_vtk)

    if parallel and len(clusters) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fps = list(ex.map(build_fingerprint, clusters))
    else:
        fps = [build_fingerprint(c) for c in clusters]
    fps_dicts = [to_dict(fp) for fp in fps]

    if not fps_dicts:
        return []

    keys = ["linearity", "planarity", "scattering", "circularity", "nnz"]
    M = _stack_metrics(fps_dicts, keys)
    emb = embed_features(M)
    labels2 = cluster_labels(emb, min_cluster_size=min_cluster_size, random_state=random_state)

    uniq = sorted(set(int(x) for x in labels2))
    mapping = {lbl: f"eddy_{i}" for i, lbl in enumerate(uniq)}
    for rec, lbl in zip(fps_dicts, labels2):
        rec["label"] = mapping.get(int(lbl), "eddy_-1")
    return fps_dicts

def run_pipeline_from_json(points_path: str,
                           out_path: str,
                           key: str = "points",
                           min_cluster_size=20,
                           eps=0.5,
                           random_state=42) -> list[dict]:
    """Load point cloud(s), run pipeline, and save fingerprint JSON."""
    clouds = load_point_clouds_json(points_path, key=key)
    if not clouds:
        return []
    points = clouds[0] if len(clouds) == 1 else np.vstack(clouds)

    fps_dicts = pipeline_cluster_points(points,
                                        min_cluster_size=min_cluster_size,
                                        eps=eps,
                                        random_state=random_state)
    save_fingerprints_json(fps_dicts, out_path)
    return fps_dicts

def run_pipeline_from_points(points: np.ndarray,
                             out_path: str,
                             min_cluster_size=20,
                             eps=0.5,
                             random_state=42) -> list[dict]:
    """Run pipeline directly on a point cloud array and save fingerprint JSON."""
    points = np.asarray(points, dtype=float)
    fps_dicts = pipeline_cluster_points(points,
                                        min_cluster_size=min_cluster_size,
                                        eps=eps,
                                        random_state=random_state,
                                        clusters_out_vtk="features.vtk")
    save_fingerprints_json(fps_dicts, out_path)
    print("fingerprints dictionary saved to ", out_path)
    return fps_dicts

def load_point_clouds_vtk(path: str) -> list[np.ndarray]:
    """Load point cloud(s) from a .vtk file."""
    try:
        import pyvista as pv
        data = pv.read(path)
        clouds = []
        if isinstance(data, pv.MultiBlock):
            for block in data:
                if block is None:
                    continue
                pts = np.asarray(block.points, dtype=float) if hasattr(block, "points") else None
                if pts is not None and pts.size:
                    clouds.append(pts)
        else:
            # Split single dataset into connected components
            try:
                conn = data.connectivity()
                region_ids = np.unique(conn["RegionId"])
                for rid in region_ids:
                    sub = conn.threshold([rid, rid], scalars="RegionId")
                    pts = np.asarray(sub.points, dtype=float)
                    if pts.size:
                        clouds.append(pts)
            except Exception:
                pts = np.asarray(data.points, dtype=float)
                if pts.size:
                    clouds.append(pts)
        return clouds
    except Exception:
        try:
            import vtk
            reader = vtk.vtkGenericDataObjectReader()
            reader.SetFileName(path)
            reader.Update()
            data = reader.GetOutput()
            pts_vtk = data.GetPoints()
            if pts_vtk is None:
                return []
            n = pts_vtk.GetNumberOfPoints()
            pts = np.asarray([pts_vtk.GetPoint(i) for i in range(n)], dtype=float)
            return [pts] if pts.size else []
        except Exception:
            return []

def run_pipeline_from_vtk(vtk_path: str,
                          out_path: str,
                          min_cluster_size=20,
                          eps=0.5,
                          random_state=42) -> list[dict]:
    """Load point cloud(s) from .vtk, run pipeline, and save fingerprint JSON."""
    clouds = load_point_clouds_vtk(vtk_path)
    if not clouds:
        return []
    points = clouds[0] if len(clouds) == 1 else np.vstack(clouds)

    fps_dicts = pipeline_cluster_points(points,
                                        min_cluster_size=min_cluster_size,
                                        eps=eps,
                                        random_state=random_state,
                                        clusters_out_vtk="features.vtk")
    save_fingerprints_json(fps_dicts, out_path)
    return fps_dicts

def reconstruct_points_from_fingerprint(fp: dict, n_points: int = 128, seed: int = 42) -> np.ndarray:
    """Approximate a point cloud from fingerprint grid + pose."""
    rng = np.random.default_rng(seed)
    grid = fp.get("grid", {})
    payload = grid.get("payload", [])
    if not payload:
        return np.zeros((0, 3), dtype=float)

    cells = []
    weights = []
    for i in range(0, len(payload), 4):
        ix, iy, iz, w = payload[i:i+4]
        cells.append((int(ix), int(iy), int(iz)))
        weights.append(float(w))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

    b = int(grid.get("nx", 12))
    idx = rng.choice(len(cells), size=n_points, p=weights)
    pts = []
    for j in idx:
        ix, iy, iz = cells[j]
        # cell center in [-1,1]
        cx = (ix + 0.5) / b * 2.0 - 1.0
        cy = (iy + 0.5) / b * 2.0 - 1.0
        cz = (iz + 0.5) / b * 2.0 - 1.0
        pts.append([cx, cy, cz])
    Xc = np.asarray(pts, dtype=float)

    pose = fp.get("pose", {})
    R = np.asarray(pose.get("R"), dtype=float)
    c = np.asarray(pose.get("centroid"), dtype=float)
    s = float(pose.get("scale_s", 1.0))
    Xr = Xc * s
    X0 = Xr @ R  # inverse of Xr = X0 @ R.T
    return X0 + c

def plot_compare_with_original(labels_json_path: str,
                               original_points,
                               index: int = 0,
                               n_points: int = 512):
    """Load labeled fingerprints, reconstruct a feature, and compare to original."""
    fps = load_fingerprints_json(labels_json_path)
    if not fps:
        return None, None

#   if isinstance(original_points, str):
    if original_points.endswith(".vtk"):
        clouds = load_point_clouds_vtk(original_points)
    else:
        clouds = load_point_clouds_json(original_points)

    # GeoVista if available, else PyVista/Matplotlib fallback
    try:
        import geovista as gv
        import pyvista as pv
        plotter = gv.GeoPlotter()
        plotter.add_base_layer()

        X_all = []
        for index, fp in enumerate(fps):
            print(index)
            X_recon = reconstruct_points_from_fingerprint(fp, n_points=n_points)
            print(index, fp, X_recon[:10])
            print(len(X_recon))
            X_orig = clouds[index] 
            print(X_orig[:10])
            plotter.add_mesh(pv.PolyData(X_orig), color="dodgerblue", point_size=5, render_points_as_spheres=True)
            plotter.add_mesh(X_recon, color="orange", point_size=5, render_points_as_spheres=True)
        plotter.show()
        
    except Exception:
        try:
            import pyvista as pv
            p = pv.Plotter()
            p.add_mesh(pv.PolyData(X_orig), color="dodgerblue", point_size=5, render_points_as_spheres=True)
            p.add_mesh(pv.PolyData(X_recon), color="orange", point_size=5, render_points_as_spheres=True)
            p.show()
        except Exception:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            if X_orig is not None:
                ax.scatter(X_orig[:, 0], X_orig[:, 1], X_orig[:, 2], s=2, c="dodgerblue", label="original")
            ax.scatter(X_recon[:, 0], X_recon[:, 1], X_recon[:, 2], s=2, c="orange", label="reconstructed")
            ax.legend()
            plt.show()

    return X_orig, X_recon

def main():
    """Plot embeddings for synthetic features and clustered outputs."""
    try:
        from .synthetic import SynthConfig, mixed_point_cloud, save_point_cloud_vtk
    except Exception:
        from synthetic import SynthConfig, mixed_point_cloud, save_point_cloud_vtk

    # process one big point cloud to separate into features 
    # save features to "synthetic_feature_point_cloud.vtk"
#   dummy = run_pipeline_from_vtk("synthetic_points.vtk", "single.jsonl")
    dummy = run_pipeline_from_vtk("NW_atlantic_soundspeed_on_soundspeed_fronts.vtk", "single.jsonl")
    # save fingerprints of features to "run_fp.jsonl")
    fps_dicts = run_pipeline_from_vtk("synthetic_feature_point_cloud.vtk", "run_fp.jsonl")

#   X = mixed_point_cloud(SynthConfig())
#   save_point_cloud_vtk("synthetic_points.vtk", X)

    # plot_features_from_json("core/synthetic_features.json", title="Synthetic Features")
    # plot_features_from_json("run_fp.jsonl", title="Clustered Features")
    # for index in range(0,31):
    plot_compare_with_original("run_fp.jsonl", "synthetic_feature_point_cloud.vtk")

if __name__ == "__main__":
    main()
