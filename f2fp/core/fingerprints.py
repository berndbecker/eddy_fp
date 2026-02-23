from __future__ import annotations
import sys

if sys.flags.interactive:
    print(
        "I am in interactive mode! , use standard matplotlib backend with X windows support"
    )
else:
    #   print('I am in batch mode!, use agg as backend for plotting ')
    #   print(' (in agg you get no interactive session to look at/monitor production, \
    #            ONLY if you imported matplotlib bfore mpl,use( Agg ))')
    import matplotlib as mpl

    mpl.use("Agg")

from pyvista.trame.jupyter import launch_server

import os
os.environ.pop("DISPLAY", None)

import time
from contextlib import contextmanager
import numpy as np
import pathlib
from fast_hdbscan import HDBSCAN as FastHDBSCAN
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
    
DEBUG_FP = True

def _dbg(*args, **kwargs):
    if DEBUG_FP:
        print(*args, **kwargs)

@contextmanager
def _timer(label: str):
    if not DEBUG_FP:
        yield
        return
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _dbg(f"{label}: {dt:.3f}s")

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

#def _stack_metrics(fps: list[dict], keys: list[str]) -> np.ndarray:
#    """Stack selected metrics from fingerprints into a matrix."""
#    assert len(fps) > 0, "fps list is empty"
#    M = []
#    for d in fps:
#       m = d['metrics']
#       # Extract metrics for each fingerprint
#       M.append([float(m.get(k, 0.0)) for k in keys])
#    return np.asarray(M, dtype=float)

def _stack_metrics(fps: list[dict], keys: list[str]) -> np.ndarray:
    """Stack selected metrics from fingerprints into a matrix."""
    assert len(fps) > 0, "fps list is empty"
    M = np.empty((len(fps), len(keys)), dtype=float)
    for i, d in enumerate(fps):
        m = d['metrics']
        M[i, :] = [float(m.get(k, 0.0)) for k in keys]
    return M

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

def load_point_clouds_vtm(path: str) -> list[np.ndarray]:
    """Load point clouds from a .vtm MultiBlock file."""
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
        return clouds
    except Exception:
        return []

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
    is_jsonl = path.name.endswith(".jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        if is_jsonl:
            for rec in fps:
                fh.write(json.dumps(rec))
                fh.write("\n")
        else:
            json.dump(fps, fh, indent=2)

def _cluster_points_raw(points: np.ndarray, min_cluster_size=20, eps=0.5) -> np.ndarray:
    """Cluster raw points using fast_hdbscan only."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        points = points.reshape(-1, 3)
    if points.size == 0 or points.shape[0] < 2:
        return np.full((points.shape[0],), -1, dtype=int)
    if FastHDBSCAN is None:
        raise RuntimeError("fast_hdbscan is required but not installed")
    clusterer = FastHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(2, int(min_cluster_size // 5)),
        metric="euclidean",
        cluster_selection_epsilon=0.0
    )
    return clusterer.fit_predict(points)

def _split_clusters(points: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    """Split points into clusters based on labels (ignore -1 noise)."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    clusters = []
    for lbl in sorted(set(labels)):
        if int(lbl) < 0:
            continue
        mask = labels == lbl
        if np.any(mask):
            clusters.append(points[mask])
    return clusters

def _cluster_points_raw_adaptive(points: np.ndarray,
                                 min_cluster_size=20,
                                 eps=0.5) -> np.ndarray:
    """No DBSCAN fallback; use fast_hdbscan only."""
    return _cluster_points_raw(points, min_cluster_size=min_cluster_size, eps=eps)

class IterativeHDBSCAN:
    """Iterative HDBSCAN clustering."""
    def __init__(self, min_cluster_size: int = 20, min_samples: int = 2, eps: float = 0.5):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.eps = eps
        self.metric = "euclidean"
        self.cluster_selection_epsilon = 0.0

    def _create_clusterer(self, min_cluster_size: int, min_samples: int):
        if FastHDBSCAN is None:
            raise RuntimeError("fast_hdbscan is required but not installed")
        return FastHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon
        )

    def fit(self, points: np.ndarray) -> list[dict]:
        """Fit the model to the points."""
        labels = _cluster_points_raw(points, min_cluster_size=self.min_cluster_size, eps=self.eps)
        clusters = _split_clusters(points, labels)
        if not clusters and points.size:
            clusters = [points]

        fps = []
        for c in clusters:
            fp = build_fingerprint(c)
            fps.append(fp)
        return fps

    def fit_predict(self, points: np.ndarray) -> list[dict]:
        """Fit the model to the points and return the labels."""
        labels = _cluster_points_raw(points, min_cluster_size=self.min_cluster_size, eps=self.eps)
        clusters = _split_clusters(points, labels)
        if not clusters and points.size:
            clusters = [points]

        fps = []
        for c in clusters:
            fp = build_fingerprint(c)
            fps.append(fp)
        return fps

def save_point_clouds_jsonl(path: str, clouds: list[np.ndarray]) -> None:
    """Save list of point clouds as JSONL (one cloud per line)."""
    import json
    with open(path, "w", encoding="utf-8") as fh:
        for i, pts in enumerate(clouds):
            rec = {"feature_id": int(i), "points": np.asarray(pts, dtype=float).tolist()}
            fh.write(json.dumps(rec))
            fh.write("\n")

def save_point_clouds_vtk(prefix: str, clouds: list[np.ndarray]) -> None:
    """Save list of point clouds as a single .vtk file (MultiBlock)."""
    path = prefix if prefix.name.endswith(".vtk") else f"{prefix}.vtk"
    if not clouds:
        try:
            from .synthetic import save_point_cloud_vtk
        except Exception:
            from synthetic import save_point_cloud_vtk
        save_point_cloud_vtk(path, np.zeros((0, 3), dtype=float))
        return

    try:
        print("save feature clouds as PolyData blocks as defined by cluster algorythm")
        import pyvista as pv
        blocks = pv.MultiBlock()
        for i, pts in enumerate(clouds):
            pts = np.asarray(pts, dtype=float)
            if pts.size == 0:
                continue
            blocks.append(pv.PolyData(pts))
        path = prefix if prefix.name.endswith(".vtm") else f"{prefix}.vtm"
        blocks.save(path)
    except Exception as e:
        print("failed: ", e)
        print(" Fallback: concatenate into a single point cloud")
        try:
            from .synthetic import save_point_cloud_vtk
        except Exception:
            from synthetic import save_point_cloud_vtk
        pts_all = np.vstack([np.asarray(pts, dtype=float) for pts in clouds if np.asarray(pts).size])
        print("save all feature clouds lumped together as mp array as it was before the cluster algorythm")
        path = prefix if prefix.name.endswith(".vtk") else f"{prefix}.vtk"
        save_point_cloud_vtk(path, pts_all)

#

def pipeline_cluster_points(points: np.ndarray,
                            min_cluster_size=20,
                            eps=0.5,
                            random_state=42,

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

    _dbg("clusters_out_vtk: ", clusters_out_vtk)
    points = np.asarray(points, dtype=float, order="C")
    len_points = points.shape[0]
    _dbg("points: ", len_points)
    if len_points == 0:
        return []
    if max_points and len_points > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len_points, size=max_points, replace=False)
        points = points[idx]
        len_points = points.shape[0]
    _dbg("max_points, points[:10] ", max_points, points[:10], len_points)

    with _timer("stage1 cluster"):
        if first_stage == "iterative":
            labels1 = _cluster_points_raw_adaptive(points,
                                                   min_cluster_size=min_cluster_size,
                                                   eps=eps,)
        elif first_stage == "adaptive":
            labels1 = _cluster_points_raw_adaptive(points,
                                                   min_cluster_size=min_cluster_size,
                                                   eps=eps)
        else:
            labels1 = _cluster_points_raw(points,
                                          min_cluster_size=min_cluster_size,
                                          eps=eps)

    with _timer("split clusters"):
        clusters = _split_clusters(points, labels1)
    _dbg("clusters[0] ", clusters[0] if clusters else None)
    _dbg("length clusters ", len(clusters))
    _dbg("labels1  ", labels1)
    if not clusters and len_points:
        _dbg("no clusters found! returning original point cloud")
        clusters = [points]

    if clusters_out_vtk:
        prefix = clusters_out_vtk.suffix
        if prefix == ".vtm":
            save_point_clouds_vtk(clusters_out_vtk, clusters)
            _dbg("features saved to ", clusters_out_vtk, " from save_point_clouds_vtk")
        else:
            save_point_clouds_jsonl(clusters_out, clusters)
            _dbg("features saved to ", clusters_out, " from save_point_clouds_jsonl")

    # Avoid ProcessPool overhead for small workloads
    use_parallel = parallel and len(clusters) > 4
    with _timer("build fingerprints"):
        if use_parallel:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                fps = list(ex.map(build_fingerprint, clusters))
        else:
            fps = [build_fingerprint(c) for c in clusters]
    fps_dicts = [to_dict(fp) for fp in fps]

    if not fps_dicts:
        _dbg("no fingerprints!")
        return []

    keys = ["linearity", "planarity", "scattering", "circularity", "nnz"]
    with _timer("stage2 embed+cluster"):
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

def run_pipeline_from_points(points: np.ndarray,
                             out_path: str,
                             min_cluster_size=20,
                             eps=0.5,
                             random_state=42,
                             features_out_vtk: str | None = None) -> list[dict]:
    """Run pipeline directly on a point cloud array and save fingerprint JSON."""
    points = np.asarray(points, dtype=float)
    fps_dicts = pipeline_cluster_points(points,
                                        min_cluster_size=min_cluster_size,
                                        eps=eps,
                                        random_state=random_state,
                                        clusters_out_vtk=features_out_vtk)
    save_fingerprints_json(fps_dicts, out_path)
    print("fingerprints dictionary saved to ", out_path)
    return fps_dicts

def run_pipeline_from_vtk(vtk_path: str,
                          out_path: str,
                          min_cluster_size=20,
                          eps=0.5,
                          random_state=42,
                          features_out_vtk: str | None = None) -> list[dict]:
    """Load point cloud(s) from .vtk, run pipeline(also save cluster feature point clouds),
    and save fingerprint JSON."""
#   feature_clusters_out = (features_out_vtk.parent/features_out_vtk.stem).with_suffix(".vtm")
    clouds = load_point_clouds_vtk(vtk_path)
    if not clouds:
        return []
    points = clouds[0] if len(clouds) == 1 else np.concatenate(clouds, axis=0)

    fps_dicts = pipeline_cluster_points(points,
                                        min_cluster_size=min_cluster_size,
                                        eps=eps,
                                        random_state=random_state,
#                                       clusters_out=feature_clusters_out,
                                        clusters_out_vtk=features_out_vtk)
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

def _ecef_to_lonlatalt(X: np.ndarray) -> np.ndarray:
    """Convert ECEF (meters) to lon/lat/alt (degrees, meters)."""
    from pyproj import Transformer
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    lon, lat, alt = transformer.transform(X[:, 0], X[:, 1], X[:, 2])
    return np.column_stack([lon, lat, alt])


def plot_compare_with_original(labels_json_path: str, 
                               original_points,
                               index: int = 0,
                               n_points: int = 128,
                               zlevel_scale: float = 1.0/6371.):
    
    """Load labeled fingerprints, reconstruct a feature, and compare to original."""
    fps = load_fingerprints_json(labels_json_path)
    if not fps:
        print(" no fingerprints!")
        return None, None

    #if isinstance(original_points, str) and original_points.endswith(".vtm"):
    clouds = load_point_clouds_vtm(original_points)
    #elif isinstance(original_points, str) and original_points.endswith(".vtk"):
    #    clouds = load_point_clouds_vtk(original_points)
    #else:
    #    clouds = load_point_clouds_json(original_points)

    if not clouds:
        print(" no clouds!")
        return None, None

    try:
        import geovista as gv
        import pyvista as pv
#       pv.start_xvfb() 
        plotter = gv.GeoPlotter(off_screen=True)
        #plotter = gv.GeoPlotter()

        # Add a globe layer if available (best-effort)
#       if hasattr(plotter, "add_base_layer"):
#           plotter.add_base_layer()

        n = min(len(fps), len(clouds))
        for i in range(n):
            fp = fps[i]
            _dbg("index, fp[\"label\"], fp[\'metrics\'][\'circularity\']", index, fp["label"], fp['metrics']['circularity'])
            X_recon = reconstruct_points_from_fingerprint(fp, n_points=n_points)
            X_orig = clouds[i]
            _dbg("reconstructed ", X_recon[:3])    
            _dbg("original      ", X_orig[:3])

            # lon, lat, alt -> cartesian on the globe
            points_orig = gv.common.to_cartesian(
                X_orig[:, 0], X_orig[:, 1],
                zlevel=X_orig[:, 2], zscale=zlevel_scale
            )
            points_recon = gv.common.to_cartesian(
                X_recon[:, 0], X_recon[:, 1],
                zlevel=X_recon[:, 2], zscale=zlevel_scale
            )

            # Debug: check for NaNs
            _dbg("points_orig nan?", np.isnan(points_orig).any(), "min/max", points_orig.min(), points_orig.max())
            _dbg("points_recon nan?", np.isnan(points_recon).any(), "min/max", points_recon.min(), points_recon.max())

            # Plot as points (more reliable in GeoPlotter HTML)
            #plotter.add_points(points_orig, color="blue", point_size=5, render_points_as_spheres=True)
            #plotter.add_points(points_recon, color="lightgreen", point_size=5, render_points_as_spheres=True)
           
            # Plot as point clouds on the globe
            plotter.add_mesh(pv.PolyData(points_orig), color="dodgerblue", point_size=5, render_points_as_spheres=True)
            plotter.add_mesh(pv.PolyData(points_recon), color="orange", point_size=5, render_points_as_spheres=True)

#       plotter.reset_camera() 
        html_file = pathlib.Path(pathlib.Path(labels_json_path).stem).with_suffix(".html")
        html_dir = pathlib.Path("/home/users/bernd.becker/public_html/FRONTAL/")
        plotfile = html_dir / html_file
        print(" make ", plotfile)
        try:
            plotter.export_html(plotfile)
        except Exception as e:
            html_dir = pathlib.Path("/home/users/orca12/public_html/research/FRONTAL" )
            plotfile = html_dir / html_file
            print(e, " save html to plotfile instead ", plotfile)
            plotter.export_html(plotfile)

#       plotter.show()
#       plotter.close()
#       return plotfile, None
    except Exception as e:
        print("plot failed:", e)
        try:
            import pyvista as pv
            p = pv.Plotter()
            p.add_mesh(pv.PolyData(X_orig), color="dodgerblue", point_size=5, render_points_as_spheres=True)
            p.add_mesh(pv.PolyData(X_recon), color="orange", point_size=5, render_points_as_spheres=True)
            p.show()
        except Exception as e:
            print("pyvista plot also failed:", e)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    n = min(len(fps), len(clouds))
    for i in range(n):
        fp = fps[i]
        print("index, fp[\"label\"], fp[\'metrics\'][\'circularity\']", index, fp["label"], fp['metrics']['circularity'])
        X_recon = reconstruct_points_from_fingerprint(fp, n_points=n_points)
        X_orig = clouds[i]

        if X_orig is not None:
#           ax.scatter(X_orig[:, 0], X_orig[:, 1], X_orig[:, 2], s=2, c="dodgerblue", label="original")
#           ax.scatter(X_recon[:, 0], X_recon[:, 1], X_recon[:, 2], s=2, c="orange", label="reconstructed")
            ax.scatter(X_orig[:, 0], X_orig[:, 1], X_orig[:, 2], s=2, c="dodgerblue")
            ax.scatter(X_recon[:, 0], X_recon[:, 1], X_recon[:, 2], s=2, c="orange")

    ax.legend()
    plt.savefig("compare_scatter.png")
    print("saved compare_scatter.png")
    plt.show()

    return X_orig, X_recon

def main():
    """Plot embeddings for synthetic features and clustered outputs."""
    try:
        from .synthetic import SynthConfig, mixed_point_cloud, save_point_cloud_vtk
    except Exception:
        from synthetic import SynthConfig, mixed_point_cloud, save_point_cloud_vtk

    # save features to "synthetic_feature_point_cloud.vtk"
#   X = mixed_point_cloud(SynthConfig())
#   save_point_cloud_vtk("synthetic_points.vtk", X)

    indir = "/data/scratch/orca12/BBecker_frontal_assessment/output/"
    filetag = "level1_gl_orca12_asm12_20260220_20260222T12/"

    canny_features_vtk = pathlib.Path(indir + filetag + "CSV/all_three_3D_cluster.vtk")
    #canny_features_vtk = pathlib.Path("synthetic_points.vtk")
    canny_features_vtm = (canny_features_vtk.parent/canny_features_vtk.stem).with_suffix(".vtm")
    fingerprints_jsonl = (canny_features_vtk.parent/canny_features_vtk.stem).with_suffix(".jsonl")

    # process one big point cloud to separate into features 
    fps_dicts = run_pipeline_from_vtk(canny_features_vtk, fingerprints_jsonl, features_out_vtk=canny_features_vtm)

    # plot_features_from_json("core/synthetic_features.json", title="Synthetic Features")
    # plot_features_from_json("run_fp.jsonl", title="Clustered Features")
    # for index in range(0,31):
    print("plot_compare_with_original( ", fingerprints_jsonl, canny_features_vtm, ")")
    plot_compare_with_original(fingerprints_jsonl, canny_features_vtm)

if __name__ == "__main__":
    main()
