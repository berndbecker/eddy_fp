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
    import os
    os.environ.pop("DISPLAY", None)

import argparse
import geovista as gv
import pyvista as pv
import matplotlib.pyplot as plt
import vtk
import json
#from pyvista.trame.jupyter import launch_server

import time
from contextlib import contextmanager
import numpy as np
import pathlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pyproj import CRS, Transformer
from collections import Counter
import matplotlib.cm as cm

#from geovista.pantry.meshes import ZLEVEL_SCALE_CLOUD  # == 0.00001 ,1.e-5
R_EARTH_M = 6_371_000.0
ZLEVEL_SCALE_CLOUD=1./R_EARTH_M   # NOT! == 0.00001 ,1.e-5
GRID_SIZE = 32  # higher resolution for tighter reconstruction



from fast_hdbscan import HDBSCAN as FastHDBSCAN
from dataclasses import dataclass
try:
    from .canonical import pca_align,  Pose
except Exception:
    from canonical import pca_align,  Pose

@dataclass
class Fingerprint:
    """Fingerprint representation for an eddy."""
    pose: Pose
    grid: dict
    metrics: dict
    label: str | None = None
    id: int | None = None
    
DEBUG_FP = True

def _progress(msg: str):
    print(msg, flush=True)

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
# ...existing code...
def build_fingerprint(X: np.ndarray, label: str | None = None, index: int | None = None) -> Fingerprint:
    """Build fingerprint in the input coordinate system (no geo conversion)."""

    # IMPORTANT: keep coordinates as-is (no lon/lat/depth conversion)

    X = _coerce_point_cloud(X)
    assert X.ndim == 2 and X.shape[1] == 3, "Input X must be (N,3) array"
    assert X.shape[0] > 0, "Input X must not be empty"
    # detect if X is ECEF unit sphere or Earth radius scale (norm 1 or R_EARTH_M)
    norms = np.linalg.norm(X, axis=1)
    if np.all((np.abs(norms - 1) < 0.1) | (np.abs(norms - R_EARTH_M) < 1000)):  # approximate check
        lonlatdepth = cloud_to_lonlatdepth(X)
    else:
        lonlatdepth = X

    X_local, geo = _to_local_tangent(lonlatdepth)
    Xc, pose = pca_align(X_local)
    _set_pose_geo(pose, geo)

    grid = sparse_grid_12(Xc)
    metrics = _basic_metrics(Xc, grid)

    # --- compact sparse payload (packed indices + quantized weights) ---
    payload = grid.get("payload", [])
    nx = int(grid.get("nx", 12))
    ny = int(grid.get("ny", 12))
    nz = int(grid.get("nz", 12))
    if payload and nx <= 63 and ny <= 63 and nz <= 63:
        cells = []
        weights = []
        for i in range(0, len(payload), 4):
            ix, iy, iz, w = payload[i:i+4]
            cells.append((int(ix), int(iy), int(iz)))
            weights.append(float(w))
        weights = np.asarray(weights, dtype=float)

        # choose exact scale if possible
        w_pos = weights[weights > 0]
        if w_pos.size > 0:
            w_scale = float(np.min(w_pos))
            w_q = weights / w_scale
            if np.allclose(w_q, np.round(w_q), rtol=0, atol=1e-12):
                w_q = np.round(w_q).astype(np.uint16)
                cells_packed = []
                for ix, iy, iz in cells:
                    packed = (ix << 12) | (iy << 6) | iz
                    cells_packed.append(int(packed))
                grid["cells_packed"] = cells_packed
                grid["w_q"] = w_q.tolist()
                grid["w_scale"] = w_scale
                grid.pop("payload", None)  # drop verbose payload
    # --- end compact payload ---
    metrics = _basic_metrics(Xc, grid)

    fp = Fingerprint(pose=pose, grid=grid, metrics=metrics, label=label, id=index)
    setattr(fp, "moments", _gaussian_moments(Xc))
    setattr(fp, "offset", grid.get("center", [0.0, 0.0, 0.0]))
    setattr(fp, "scale", grid.get("half", [1.0, 1.0, 1.0]))
# ...existing code...
    return fp
# ...existing code...


    
def reconstruct_points_from_fingerprint(fp: dict, n_points: int = 528, seed: int = 42,
                                        jitter: bool = True,
                                        cell_size_scale: float = 0.99,
                                        jitter_scale: float = 0.9,
                                        return_geo: bool = True,
                                        return_unit_sphere: bool = False) -> np.ndarray:
    """Approximate a point cloud from fingerprint grid + pose."""
    rng = np.random.default_rng(seed)
    grid = fp.get("grid", {})

    # accept compact or full payload
    payload = grid.get("payload", [])
    cells_packed = grid.get("cells_packed", [])
    if (not payload) and cells_packed:
        w_q = np.asarray(grid.get("w_q", []), dtype=float)
        w_scale = float(grid.get("w_scale", 1.0))
        payload = []
        for packed, wq in zip(cells_packed, w_q):
            ix = (packed >> 12) & 0x3F
            iy = (packed >> 6) & 0x3F
            iz = packed & 0x3F
            w = float(wq) * w_scale
            payload.extend([ix, iy, iz, w])

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
    counts = np.floor(weights * n_points).astype(int)
    rem = n_points - counts.sum()
    if rem > 0:
        extra = rng.choice(len(cells), size=rem, p=weights)
        for j in extra:
            counts[j] += 1

    pts = []
    cell_size = (2.0 / b) * cell_size_scale

    for (ix, iy, iz), c in zip(cells, counts):
        if c <= 0:
            continue
        cx = (ix + 0.5) / b * 2.0 - 1.0
        cy = (iy + 0.5) / b * 2.0 - 1.0
        cz = (iz + 0.5) / b * 2.0 - 1.0
        if jitter:
            jitter_xyz = (rng.random((c, 3)) - 0.5) * cell_size * jitter_scale
            base = np.array([cx, cy, cz], dtype=float)
            pts.append(base + jitter_xyz)
        else:
            pts.append(np.repeat([[cx, cy, cz]], c, axis=0))

    Xc = np.vstack(pts) if pts else np.zeros((0, 3), dtype=float)

    center = np.asarray(grid.get("center", [0.0, 0.0, 0.0]), dtype=float)
    half = np.asarray(grid.get("half", [1.0, 1.0, 1.0]), dtype=float)
    Xc = Xc * half + center

    pose = fp.get("pose", {})
    R = np.asarray(pose.get("R"), dtype=float)
    c = np.asarray(pose.get("centroid"), dtype=float)
    s = float(pose.get("scale_s", 1.0))
    Xr = Xc * s
    X0 = Xr @ R
    X_local = X0 + c

    geo = pose.get("geo") or fp.get("geo")

    if return_unit_sphere:
        if geo:
            X_geo = _from_local_tangent(X_local, geo)
            lon, lat, depth = X_geo[:, 0], X_geo[:, 1], X_geo[:, 2]
            h = -depth
            crs_geo = CRS.from_epsg(4979)
            crs_ecef = CRS.from_epsg(4978)
            tf = Transformer.from_crs(crs_geo, crs_ecef, always_xy=True)
            x, y, z = tf.transform(lon, lat, h)
            X_ecef = np.column_stack([x, y, z]) / R_EARTH_M
            return X_ecef.astype(float)

        # fallback: if already ECEF (meters), normalize to unit sphere
        r = np.linalg.norm(X_local, axis=1)
        r_med = float(np.median(r)) if r.size else 0.0
        if r_med > 1e5:
            return (X_local / r[:, None]).astype(float)

        return X_local.astype(float)

    if return_geo:
        if geo:
            X_geo = _from_local_tangent(X_local, geo)
            lon = X_geo[:, 0]
            lat = X_geo[:, 1]
            depth = X_geo[:, 2]
            mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(depth)
            mask &= (lat > -89.0) & (lat < 89.0)
            mask &= ~((np.abs(lon) < 1e-6) & (np.abs(lat) < 1e-6))
            return X_geo[mask]
        return cloud_to_lonlatdepth(X_local)

    return X_local

def cloud_to_lonlatdepth_gv(mesh):
    transformer = Transformer.from_crs(
        "EPSG:4978",   # ECEF meters
        "EPSG:4326",   # lon/lat/height (ellipsoidal)
        always_xy=True
    )

    if isinstance(mesh, np.ndarray):
        xyz = mesh
    else:
        xyz = mesh.points

    r_med = float(np.median(np.linalg.norm(xyz, axis=1)))
    if r_med <= 2.0:
        # unit-sphere ECEF
        xyz = xyz * R_EARTH_M
    elif r_med < 1e5:
        # unit box or unknown scale -> cannot compute depth
        xyz = xyz * ZLEVEL_SCALE_CLOUD
#       raise ValueError("cloud_to_lonlatdepth_gv: input is unit-box/unknown scale; depth undefined.")

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon, lat, h_ellip = transformer.transform(x, y, z)

    try:
        tf_geoid = Transformer.from_crs("EPSG:4326+5773", "EPSG:4326+5714", always_xy=True)
        _, _, h_ortho = tf_geoid.transform(lon, lat, h_ellip)
        depth = -h_ortho
    except Exception:
        depth = -h_ellip

    lon = ((lon + 180.0) % 360.0) - 180.0
    return np.column_stack([lon, lat, depth]).astype(float)


def build_fingerprint_geo(X: np.ndarray, label: str | None = None, index: int | None = None) -> Fingerprint:
    """ keys in fingerprint:
    pose
      centroid
      R
      scale_s
      geo
        crs
        lon0
        lat0
        depth_positive_down
    grid
      nx
      ny
      nz
      nnz
      payload
      center
      half
    metrics
      linearity
      planarity
      scattering
      circularity
      nnz
    label
    moments
      mean
      variance
      skewness
      kurtosis
    offset
    scale
    type_label
    id
    """
    X = _coerce_point_cloud(X)
    assert X.ndim == 2 and X.shape[1] == 3, "Input X must be (N,3) array"
    assert X.shape[0] > 0, "Input X must not be empty"
    # detect if X is ECEF unit sphere or Earth radius scale (norm 1 or R_EARTH_M)
    norms = np.linalg.norm(X, axis=1)
    if np.all((np.abs(norms - 1) < 0.1) | (np.abs(norms - R_EARTH_M) < 1000)):  # approximate check
        lonlatdepth = cloud_to_lonlatdepth(X)
    else:
        lonlatdepth = X

    X_local, geo = _to_local_tangent(lonlatdepth)
    Xc, pose = pca_align(X_local)
    _set_pose_geo(pose, geo)

    grid = sparse_grid_12(Xc)
    metrics = _basic_metrics(Xc, grid)

    # we need an index here for the process running number sequence.
    #use enumerate when you loop the fingerprints, But this fails when you filter?"
    # JSONL records need a dict with explicit id key
    fp = Fingerprint(pose=pose, grid=grid, metrics=metrics, label=label, id=index)
#   if index is not None:
#       fp.index = int(index)
    # attach moments + offset/scale for reconstruction
    setattr(fp, "moments", _gaussian_moments(Xc))
    setattr(fp, "offset", grid.get("center", [0.0, 0.0, 0.0]))
    setattr(fp, "scale", grid.get("half", [1.0, 1.0, 1.0]))

    _dbg("fp.id =", getattr(fp, "id", None))
    return fp

def _normalize_metrics(metrics: dict) -> dict:
    """Normalize metrics keys and types for JSON stability."""
    return {
        "linearity": float(metrics.get("linearity", 0.0)),
        "planarity": float(metrics.get("planarity", 0.0)),
        "scattering": float(metrics.get("scattering", 0.0)),
        "circularity": float(metrics.get("circularity", 0.0)),
        "nnz": int(metrics.get("nnz", 0)),
    }

def sparse_grid_12(Xc: np.ndarray) -> dict:
    """Create a sparse grid representation from canonical coordinates."""
    assert Xc.ndim == 2 and Xc.shape[1] == 3, "Input Xc must be (N,3) array"
    b = GRID_SIZE

    xyz_min = Xc.min(axis=0)
    xyz_max = Xc.max(axis=0)
    center = (xyz_min + xyz_max) * 0.5
    half = (xyz_max - xyz_min) * 0.5
    half[half == 0] = 1.0
    Xn = (Xc - center) / half
    Xn = np.clip(Xn, -1.0, 1.0)

    idx = np.clip(((Xn + 1.0) * 0.5 * b).astype(int), 0, b-1)
    keys, counts = np.unique(idx, axis=0, return_counts=True)

    payload = []
    total = counts.sum() if counts.size else 1
    for (i, j, k), c in zip(keys, counts):
        payload += [int(i), int(j), int(k), float(c / total)]

    return {
        'nx': b, 'ny': b, 'nz': b,
        'nnz': int(keys.shape[0]),
        'payload': payload,
        'center': center.tolist(),
        'half': half.tolist(),
    }

def _normalize_grid(grid: dict) -> dict:
    out = dict(grid)
    # ensure payload is a plain list (if needed)
    if "payload" in out and not isinstance(out["payload"], list):
        out["payload"] = list(out["payload"])
    # keep center/half for reconstruction
    if "center" in grid:
        out["center"] = grid["center"]
    if "half" in grid:
        out["half"] = grid["half"]
    return out

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
                            show: bool = False):
    """Load JSON/JSONL fingerprints and plot embedded features."""
    try:
        from .cluster import embed_features
    except Exception:
        from cluster import embed_features

    fps = load_fingerprints_json(path)
    keys = keys or ["linearity", "planarity", "scattering", "circularity", "nnz"]
    M = _stack_metrics(fps, keys)
    emb = embed_features(M)

    labels = [fp.get("type_label", "unknown") for fp in fps]
    label_set = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(label_set)}
    color_vals = [label_to_int[l] for l in labels]

#   plt.figure(figsize=(7, 5))
    plt.figure()
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

    scat_file = pathlib.Path(pathlib.Path(path).stem+"_scatt").with_suffix(".png")
    scat_dir = pathlib.Path(path).parent.parent/"Plots/"
    plotfile = scat_dir / scat_file
    plt.savefig(plotfile)
    print("cluster_scatter.png saved in ", plotfile)
    #del plotfile
    if sys.flags.interactive:
#       if show:
#           plt.show()
        pass
    else:
        pass
    plt.close()

    return emb

def load_point_clouds_vtm(path: str) -> list[np.ndarray]:
    """Load point clouds from a .vtm MultiBlock file."""
    try:
        data = pv.read(path)
        clouds = []
        if isinstance(data, pv.MultiBlock):
            for block in data:
                if block is None:
                    continue
                pts = np.asarray(block.points, dtype=float) if hasattr(block, "points") else None
                if pts is not None and pts.size:
                    clouds.append(pts)
        return clouds, data
    except Exception:
        return []

def load_point_clouds_json(path: str, key: str = "points") -> list[np.ndarray]:
    """Load point cloud(s) from JSON or JSONL."""
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

    min_samples=max(2, int(min_cluster_size // 5))
    print("hasthdbscan with min_cluster_size, min_samples, eps)", min_cluster_size, min_samples, eps)
    clusterer = FastHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=eps
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
        for idx, c in enumerate(clusters):
            fp = build_fingerprint(c, index=idx)
            fps.append(fp)
        return fps

    def fit_predict(self, points: np.ndarray) -> list[dict]:
        """Fit the model to the points and return the labels."""
        labels = _cluster_points_raw(points, min_cluster_size=self.min_cluster_size, eps=self.eps)
        clusters = _split_clusters(points, labels)
        if not clusters and points.size:
            clusters = [points]

        fps = []
        for idx, c in enumerate(clusters):
            fp = build_fingerprint(c, index=idx)
            fps.append(fp)
        return fps

def save_point_clouds_jsonl(path: str, clouds: list[np.ndarray]) -> None:
    """Save list of point clouds as JSONL (one cloud per line)."""
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
        print("save feature clouds as PolyData blocks as defined by cluster algorithm")
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
        print("save all feature clouds lumped together as mp array as it was before the cluster algorithm")
        path = prefix if prefix.name.endswith(".vtk") else f"{prefix}.vtk"
        save_point_cloud_vtk(path, pts_all)

def _cluster_points_raw_iterative(points: np.ndarray, \
                                  min_cluster_size=20, \
                                  eps=0.5, \
                                  max_iters: int = 10, \
                                  eps_decay: float = 0.7, \
                                  min_eps: float = 1e-8):
    """Iteratively tighten eps to reach a stable clustering, with hard stop."""
    eps_i = float(eps)
    last_labels = None
    for _ in range(max_iters):
        labels = _cluster_points_raw(points, \
                                     min_cluster_size=min_cluster_size, \
                                     eps=eps_i)
        if last_labels is not None and np.array_equal(labels, last_labels):
            return labels
        last_labels = labels
        eps_i = max(min_eps, eps_i * eps_decay)
        min_cluster_size = int(round(min_cluster_size * eps_decay))
    return last_labels if last_labels is not None else _cluster_points_raw(points,
                                                                          min_cluster_size=min_cluster_size,
                                                                          eps=eps)
def _type_label_from_metrics(m: dict) -> str:
    c = float(m.get("circularity", 0.0))
    l = float(m.get("linearity", 0.0))
    p = float(m.get("planarity", 0.0))
    s = float(m.get("scattering", 0.0))

# Eddy: strong circularity
    if c >= 0.9:
        return "eddy"

    # Front: high planarity, low circularity
    if p >= 0.2 and c < 0.1:
        return "front"

    # Curtain: high linearity, low planarity
    if l >= 0.9 and p < 0.1:
        return "curtain"

    # Swirl: moderate linearity/planarity, low circularity
    if 0.5 <= l < 0.9 and 0.05 <= p < 0.2 and c < 0.1:
        return "swirl"
    return "other"

def pipeline_cluster_points(points: np.ndarray,
                            min_cluster_size=20,
                            eps=0.005,
                            random_state=42,
#                           clusters_out: str | None = None,
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
    len_points = points.shape[0]
    with _timer("stage1 cluster"):
        if first_stage == "iterative":
            labels1 = _cluster_points_raw_iterative(points,
                                                    min_cluster_size=min_cluster_size,
                                                    eps=eps)
        elif first_stage == "adaptive":
            labels1 = _cluster_points_raw_adaptive(points,
                                                   min_cluster_size=min_cluster_size,
                                                   eps=eps)
        else:
            labels1 = _cluster_points_raw(points,
                                          min_cluster_size=min_cluster_size,
                                          eps=eps)
        _progress(f"[fp] stage1 clustering on {len_points} points...")
 
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

    clusters = _split_clusters(points, labels1)

    _progress(f"[fp] stage1 done, {len(clusters)} clusters")
    _dbg("clusters[0] ", clusters[0][:3] if clusters else None)
    _dbg("length clusters ", len(clusters))
#   _dbg("labels1  ", labels1)
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
    indexed = [(c, i) for i, c in enumerate(clusters)]
    with _timer("build fingerprints"):
        if use_parallel:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                fps = list(ex.map(_build_fp_with_index, indexed))
        else:
            fps = [build_fingerprint(c, index=i) for i, c in indexed]
        _progress(f"[fp] building fingerprints (parallel={use_parallel})...")

    fps_dicts = [to_dict(fp) for fp in fps]

    _progress(f"[fp] fingerprints built: {len(fps_dicts)}")

    if not fps_dicts:
        _dbg("no fingerprints!")
        return []

    keys = ["linearity", "planarity", "scattering", "circularity", "nnz"]
    with _timer("stage2 embed+cluster"):
        M = _stack_metrics(fps_dicts, keys)
        emb = embed_features(M)
        labels2 = cluster_labels(emb, min_cluster_size=min_cluster_size, random_state=random_state)
        _progress("[fp] stage2 embedding+clustering...")

    uniq = sorted(set(int(x) for x in labels2))
    mapping = {lbl: f"feature_kind_{i}" for i, lbl in enumerate(uniq)}
    
    for rec, lbl in zip(fps_dicts, labels2):
        rec["label"] = mapping.get(int(lbl), "feature_kind_-1")
        rec["type_label"] = _type_label_from_metrics(rec["metrics"])
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
    """Load point cloud(s) from a .vtk file.
    return mpoints and mesh"""
    print(path)
    try:
        data = pv.read(path)

        mesh = data.combine()
        del blocks
        mesh = mesh.clean(tolerance=1e-6)
        print("point cloud read from ", path)
        print("len(mesh.points) = ",len(mesh.points))
        mpoints = mesh.points
        print("len(mpoints) = ",len(mpoints))
        return mpoints, mesh

        # end!

        clouds = []
        if isinstance(data, pv.MultiBlock):
            for block in data:
                if block is None:
                    continue
                pts = np.asarray(block.points, dtype=float) if hasattr(block, "points") else None
                if pts is not None and pts.size:
                    clouds.append(pts)
        else:
            print(" Split single dataset into connected components")
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
        return clouds, data
    except Exception:
        try:
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
# ...existing code...
def _points_from_vtk(vtk_path: str) -> np.ndarray:
    res = load_point_clouds_vtk(vtk_path)
    if not res:
        return np.zeros((0, 3), dtype=float)
    # res can be (points, mesh) or (clouds, data) or list
    if isinstance(res, tuple):
        clouds = res[0]
        if isinstance(clouds, list):
            return clouds[0] if len(clouds) == 1 else np.concatenate(clouds, axis=0)
        if isinstance(clouds, np.ndarray):
            return np.asarray(clouds, dtype=float)
    if isinstance(res, list):
        return res[0] if len(res) == 1 else np.concatenate(res, axis=0)
    return np.asarray(res, dtype=float)

def detect_features_from_vtk(vtk_path: str,
                             features_out_vtk: str,
                             min_cluster_size=1450,
                             eps=0.0001,
                             random_state=42,
                             first_stage: str = "iterative") -> list[np.ndarray]:
    """Stage 1: feature detection (cluster only) and save clusters as .vtm."""
    points = _points_from_vtk(vtk_path)
    if points.size == 0:
        return []
    if first_stage == "iterative":
        labels1 = _cluster_points_raw_iterative(points, min_cluster_size=min_cluster_size, eps=eps)
    elif first_stage == "adaptive":
        labels1 = _cluster_points_raw_adaptive(points, min_cluster_size=min_cluster_size, eps=eps)
    else:
        labels1 = _cluster_points_raw(points, min_cluster_size=min_cluster_size, eps=eps)

    clusters = _split_clusters(points, labels1)
    if not clusters and points.size:
        clusters = [points]

    save_point_clouds_vtk(pathlib.Path(features_out_vtk), clusters)
    return clusters

def fingerprint_features_from_vtm(vtm_path: str,
                                  out_path: str,
                                  random_state=42,
                                  max_workers: int | None = None) -> list[dict]:
    """Stage 2: build fingerprints from cluster VTM."""
    clouds, _ = load_point_clouds_vtm(vtm_path)
    if not clouds:
        return []
    indexed = [(c, i) for i, c in enumerate(clouds)]
    use_parallel = len(clouds) > 4
    if use_parallel:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fps = list(ex.map(_build_fp_with_index, indexed))
    else:
        fps = [build_fingerprint(c, index=i) for i, c in indexed]

    fps_dicts = [to_dict(fp) for fp in fps]
    save_fingerprints_json(fps_dicts, pathlib.Path(out_path))
    return fps_dicts

def identify_features_from_fingerprints(json_path: str,
                                        min_cluster_size=20,
                                        random_state=42) -> list[dict]:
    """Stage 3: feature identification from fingerprints."""
    try:
        from .cluster import embed_features, cluster_labels
    except Exception:
        from cluster import embed_features, cluster_labels

    fps = load_fingerprints_json(json_path)
    if not fps:
        return []

    keys = ["linearity", "planarity", "scattering", "circularity", "nnz"]
    M = _stack_metrics(fps, keys)
    emb = embed_features(M)
    labels2 = cluster_labels(emb, min_cluster_size=min_cluster_size, random_state=random_state)

    uniq = sorted(set(int(x) for x in labels2))
    mapping = {lbl: f"feature_kind_{i}" for i, lbl in enumerate(uniq)}

    for rec, lbl in zip(fps, labels2):
        rec["label"] = mapping.get(int(lbl), "feature_kind_-1")
        rec["type_label"] = _type_label_from_metrics(rec["metrics"])

    save_fingerprints_json(fps, pathlib.Path(json_path))
    return fps

def run_pipeline_from_vtk(vtk_path: str,
                          out_path: str,
                          min_cluster_size=1450,
                          eps=0.0001,
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
                                        clusters_out_vtk=features_out_vtk)
    save_fingerprints_json(fps_dicts, out_path)

    return fps_dicts

def _ecef_to_lonlat_depth(X_ecef: np.ndarray):
    """ECEF (meters) -> lon/lat (deg), depth (m, +down)."""
    X = np.asarray(X_ecef, dtype=float).reshape(-1, 3)
    crs_ecef = CRS.from_epsg(4978)   # WGS84 geocentric
    crs_geo = CRS.from_epsg(4979)    # WGS84 geodetic 3D
    tf = Transformer.from_crs(crs_ecef, crs_geo, always_xy=True)
    lon, lat, h = tf.transform(X[:, 0], X[:, 1], X[:, 2])  # h = height (m, +up)
    depth = -h
    lon = ((lon + 180.0) % 360.0) - 180.0
#   out = np.column_stack([lon, lat, depth])
#   return out[0] if X_ecef.ndim == 1 else out
    return lon, lat, h

 # ...existing code...
def _to_local_tangent(lon_lat_depth: np.ndarray, depth_positive_down: bool = True):
    """lon/lat/depth(m) -> local x/y/z(m) in azimuthal equidistant CRS (pyproj)."""
    lon = lon_lat_depth[:, 0]
    lat = lon_lat_depth[:, 1]
    depth = lon_lat_depth[:, 2]

    # auto-detect radians
    if np.nanmax(np.abs(lon)) <= 2 * np.pi and np.nanmax(np.abs(lat)) <= np.pi:
        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)

    # robust center using unit-vector mean (avoids dateline/pole bias)
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    x_m, y_m, z_m = np.mean(x), np.mean(y), np.mean(z)
    lon0 = np.rad2deg(np.arctan2(y_m, x_m))
    hyp = np.hypot(x_m, y_m)
    lat0 = np.rad2deg(np.arctan2(z_m, hyp))

    crs_geo = CRS.from_epsg(4326)
    crs_loc = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +datum=WGS84 +type=crs")
    fwd = Transformer.from_crs(crs_geo, crs_loc, always_xy=True)
    x, y = fwd.transform(lon, lat)

    z = -depth if depth_positive_down else depth
    X_local = np.column_stack([x, y, z]).astype(float)
    return X_local, {"crs": "aeqd", "lon0": lon0, "lat0": lat0, "depth_positive_down": depth_positive_down}
# ...existing code...

def _from_local_tangent(X_local: np.ndarray, geo: dict):
    """local x/y/z(m) -> lon/lat/depth(m) (pyproj)."""
    lon0 = float(geo["lon0"])
    lat0 = float(geo["lat0"])
    depth_positive_down = bool(geo.get("depth_positive_down", True))

    X_local = np.asarray(X_local, dtype=float)
    single = (X_local.ndim == 1)
    X_local = X_local.reshape(-1, 3)
    
    crs_geo = CRS.from_epsg(4326)
    crs_loc = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +datum=WGS84 +type=crs")
    inv = Transformer.from_crs(crs_loc, crs_geo, always_xy=True)

    lon, lat = inv.transform(X_local[:, 0], X_local[:, 1])
    depth = -X_local[:, 2] if depth_positive_down else X_local[:, 2]
    return np.column_stack([lon, lat, depth]).astype(float)
 
def _set_pose_geo(pose, geo: dict):
    if isinstance(pose, dict):
        pose["geo"] = geo
    elif hasattr(pose, "__dict__"):
        setattr(pose, "geo", geo)
    else:
        raise TypeError("Pose object does not support geo assignment")

def _gaussian_moments(X: np.ndarray) -> dict:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    var = X.var(axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    z = (X - mu) / std
    skew = (z**3).mean(axis=0)
    kurt = (z**4).mean(axis=0)
    return {
        "mean": mu.tolist(),
        "variance": var.tolist(),
        "skewness": skew.tolist(),
        "kurtosis": kurt.tolist(),
    }

def _build_fp_with_index(args):
    c, idx = args
    return build_fingerprint(c, index=idx)


def to_dict(fp: Fingerprint) -> dict:
    pose = dict(centroid=fp.pose.centroid.tolist(),
                R=fp.pose.R.tolist(),
                scale_s=float(fp.pose.scale_s))
    if hasattr(fp.pose, "geo") and fp.pose.geo:
        pose["geo"] = fp.pose.geo

    out = {
        'pose': pose,
        'grid': _normalize_grid(fp.grid),
        'metrics': _normalize_metrics(fp.metrics),
        'label': fp.label
    }

    if hasattr(fp, "id"):
        out["id"] = int(fp.id)
    if hasattr(fp, "moments"):
        out["moments"] = fp.moments
    if hasattr(fp, "offset"):
        out["offset"] = fp.offset
    if hasattr(fp, "scale"):
        out["scale"] = fp.scale
    return out


def summarize_metrics(fps: list[dict]):
    keys = ["linearity", "planarity", "scattering", "circularity"]
    for k in keys:
        vals = np.array([fp["metrics"].get(k, 0.0) for fp in fps], dtype=float)
        q = np.quantile(vals, [0, 0.1, 0.5, 0.9, 1.0])
        print(k, "q0/q10/q50/q90/q100 =", q, "mean", vals.mean())

    labels2 = [fp.get("type_label", "unknown") for fp in fps]
    _dbg("counts(labels2)  ", Counter(labels2))

def _downsample_points(X: np.ndarray, max_points: int | None, seed: int = 42) -> np.ndarray:
    if X is None:
        return X
    X = np.asarray(X, dtype=float)
    if max_points is None or X.shape[0] <= max_points:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx]

def plot_compare_with_original_gv(labels_json_path: str, 
                               original_points,
                               index: int = 0,
                               n_points: int = 612,
                               zlevel_scale: float = ZLEVEL_SCALE_CLOUD,
                               max_features: int | None = None,
                               max_points_per_cloud: int | None = 256,
                               indices: int | None = None):
    
    """Load labeled fingerprints, reconstruct feature(s), and compare to original."""
    fps = load_fingerprints_json(labels_json_path)

    _dbg("fps length", len(fps))
    _dbg("fps[0].keys", fps[0].keys() if fps else None)
    _dbg("fps[0][\"grid\"].keys() ", fps[0]["grid"].keys())

    summarize_metrics(fps)

    if not fps:
        print(" no fingerprints!")
        return None, None

    clouds, mesh = load_point_clouds_vtm(original_points)
    # clouds 3rd dimension ranges from [ -1. , 1]  minus one to one.
    if not clouds:
        print(" no clouds!")
        return None, None


    n_total = min(len(fps), len(clouds))
    step = max(1, int(indices)) if indices is not None else 1

    if max_features is None or max_features <= 0:
        indices = list(range(0, n_total, step))
    elif max_features == 1:
        indices = [min(max(index, 0), n_total - 1)]
    else:
        n = min(n_total, max_features)
        indices = list(range(0, n, step))

    print("indices ", indices)
    print("ZLEVEL_SCALE_CLOUD= ", ZLEVEL_SCALE_CLOUD)
    print("zlevel_scale= ", zlevel_scale)
    if sys.flags.interactive:
        plotter = gv.GeoPlotter(off_screen=True)
    else:
        plotter = gv.GeoPlotter(off_screen=False)
    
    label_set = sorted({fp.get("type_label", fp.get("label", "unknown")) for fp in fps})
    cmap = mpl.colormaps["tab10"].resampled(len(label_set))
    color_map = {lab: cmap(i) for i, lab in enumerate(label_set)}

    def _subtype_color(fp):
        lab = fp.get("type_label", fp.get("label", "unknown"))
        return color_map[lab]

    selection = ["eddy", ]
    selection = ["eddy", "curtain"]
    for i in indices:
        fp = fps[i]
        if fp.get("type_label") in selection:
            X_orig = clouds[fp.get("id")]
            len_X_orig = len(X_orig)
            _dbg("i, index, label, type_label, circularity, pose, len(cloud) ",
                i, fp.get("id"), fp.get("label"), fp.get("type_label"), fp["metrics"].get("circularity"), 
                fp.get("pose"), len_X_orig)
    
            n_points=len_X_orig
            max_points_per_cloud = min(len_X_orig, n_points)
            X_recon = reconstruct_points_from_fingerprint(
                       fp, 
                       n_points=max_points_per_cloud, 
                       return_unit_sphere=True, 
                       return_geo=False 
                       )

            X_orig = _downsample_points(X_orig, max_points_per_cloud, seed=42)
            #_dbg("X_orig [:3] ", X_orig [:3])     # ranges [ -1, 1  ] ranges [ -1, 1  ] ranges [-0.9 to -0.55 ] or so. 
            #_dbg("X_recon [:3] ", X_recon [:3])   # ranges [-45, 45 ] ranges [-45, 45 ] ranges [-0.9 to -0.55 ] or so. 
            #_dbg("fp  ", fp)

            X_recon = _downsample_points(X_recon, max_points_per_cloud, seed=42)
            # create the point-cloud from the sample data
            mesh_orig = pv.PolyData(X_orig)
            mesh_recon = pv.PolyData(X_recon)

            #_dbg("shape  of plotted fields: orig recon ", mesh_orig.points.shape, mesh_recon.points.shape)

            plotter.add_mesh(mesh_orig, color="dodgerblue", point_size=5, render_points_as_spheres=True)
            plotter.add_mesh(mesh_recon, color=_subtype_color(fp), point_size=5, render_points_as_spheres=True)

            #sys.exit()

    plotter.add_coastlines("110m", color="black")
    html_file = pathlib.Path(pathlib.Path(labels_json_path).stem).with_suffix(".html")
    html_dir = pathlib.Path("/home/users/bernd.becker/public_html/FRONTAL/")
    plotfile = html_dir / html_file
    print(" make ", plotfile)
    try:
        _progress("[fp] exporting html...")
        plotter.export_html(plotfile)
        _progress("[fp] export done  ...")
    except Exception as e:
        html_dir = pathlib.Path("/home/users/orca12/public_html/research/FRONTAL")
        plotfile = html_dir / html_file
        print(e, " \n save html to plotfile instead ", plotfile)
        _progress("[fp] exporting html...")
        plotter.export_html(plotfile)
        _progress("[fp] export done  ...")

    if sys.flags.interactive:

        print("plotter.show(return_cpos=True) despite running  batch")
        #plotter.show(return_cpos=True)
    else:
        pass

    return 

def depth_km_from(ds, h_m):
    for k in ds.point_data.keys():
        if k.lower() == "depth":
            d = np.asarray(ds.point_data[k], dtype=float).ravel()
            _dbg("depth in keys ", d)
            return d  # assumed km; change if your depth is in meters
    d = np.asarray(h_m)/1000.0
    _dbg("depth not in keys  d= h_m/1000. ", d)
    return d 

def as_points_xyz_lonlatdepth(lon_deg, lat_deg, depth_km):
    """
    Stack lon, lat, depth_km into a single (N,3) float64 array:
    columns = [lon_deg, lat_deg, depth_km].
    """
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    dep = np.asarray(depth_km, dtype=np.float64).ravel()
    if not (lon.shape == lat.shape == dep.shape):
        raise ValueError(f"Shapes must match: lon={lon.shape}, lat={lat.shape}, depth={dep.shape}")
    return np.stack([lon, lat, dep], axis=1)  # (N,3)

def scale_to_m(points: np.ndarray) -> np.ndarray:
    """
    Convert cartesian points to meters.
    Accepts unit-sphere, kilometers, or meters.
    ∥[x,y,z]∥=sqr(x^2+y^2+z^2)
    """
    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or X.size == 0:
        return X

    r = float(np.median(np.linalg.norm(X, axis=1)))
    if r <= 2.0:          # unit sphere
        return X * R_EARTH_M
    if r <= 2e4:          # kilometers (<= 20,000 km)
        return X * 1000.0
    return X              # meters

def print_keys(d, indent=0):
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        print("  " * indent + str(k))
        if isinstance(v, dict):
            print_keys(v, indent + 1)
        elif isinstance(v, list):
            # If list contains dicts, traverse them too
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    print("  " * (indent + 1) + f"[{i}]")
                    print_keys(item, indent + 2)

def cloud_to_lonlatdepth(cloud: np.ndarray) -> np.ndarray:
    """
    Convert cartesian cloud (unit sphere or meters) to lon/lat/depth.
    Returns (N,3): [lon_deg, lat_deg, depth_m(+down)].
    """
    X = np.asarray(cloud, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        X = X.reshape(-1, 3)

    X_m = scale_to_m(X)  # unit-sphere -> meters (ECEF)
    crs_ecef = CRS.from_epsg(4978)
    crs_geo = CRS.from_epsg(4979)
    tf = Transformer.from_crs(crs_ecef, crs_geo, always_xy=True)
    lon, lat, h = tf.transform(X_m[:, 0], X_m[:, 1], X_m[:, 2])  # h = height (+up)
    depth = -h
    lon = ((lon + 180.0) % 360.0) - 180.0
    return np.column_stack([lon, lat, depth]).astype(float)

def plot_from_fp_plt(labels_json_path: str, 
                     original_points,
                     index: int = 0,
                     n_points: int = 265,
                     max_features: int | None = None,
                     max_points_per_cloud: int | None = 256,
                     show_matplotlib: bool = False,
                     mercator_2d: bool = False,
                     indices: int | None = None):

    """Load labeled fingerprints, reconstruct feature(s), and compare to original."""
    fps = load_fingerprints_json(labels_json_path)
    clouds, mesh = load_point_clouds_vtm(original_points)
    # clouds 3rd dimension ranges from [ -1. , 1]  minus one to one.
    if not clouds:
        print(" no clouds!")
        return None, None


    lon_all1, lat_all1, dep_all1 = [], [], []
    lon_all2, lat_all2, dep_all2 = [], [], []
    circles = []  # list of dicts: {"id":..., "lon":..., "lat":..., "radius_m":...}
    cid = 0

    # only plot eddies
    selection = ["eddy", ]
    for i, fp in enumerate(fps):

        if fp.get("type_label") in selection:  # filter out what is in selections, eddies only.
            X_orig = cloud_to_lonlatdepth_gv(clouds[fp.get("id")])
            #_dbg(" orig feature in lon, lat, depth (10 points) ", X_orig[:10])
            lon1, lat1, dep1 = X_orig[:, 0], X_orig[:, 1], X_orig[:, 2]

            # ---- 2) Cartesian -> lon/lat ----
            # If your XYZ are already on a unit sphere (as GeoVista commonly uses), this is perfect.
            # If they are ECEF in meters, geo conversion still works (it normalizes to the sphere) when return_geo=True.
            X_recon_ct = reconstruct_points_from_fingerprint(fp, 
                         n_points=n_points,
                         return_unit_sphere=True, 
                         return_geo=False 
                         )
            X_recon = cloud_to_lonlatdepth_gv(X_recon_ct)
            #_dbg(" reconstructed feature in lon, lat, depth (10 points) ", X_recon[:10])
            lon2, lat2, dep2 = X_recon[:, 0], X_recon[:, 1], X_recon[:, 2]
            #_dbg("index, type_label, circularity, pose, len(cloud) ",
            #     i, fp.get("index"), fp.get("label"), fp.get("type_label"), fp["metrics"].get("circularity"), 
            #     fp.get("pose"), n_points)

            lon_all1.append(lon1); lat_all1.append(lat1); dep_all1.append(dep1)
            lon_all2.append(lon2); lat_all2.append(lat2); dep_all2.append(dep2)

            cid += 1
            geo = fp["pose"].get("geo")

            # unit-sphere ECEF -> meters -> local-tangent meters
            #X_orig_local, geo_loc = _to_local_tangent(clouds[fp.get("id")], depth_positive_down=True)
            
            X_orig_local, geo_loc = _to_local_tangent(X_recon, depth_positive_down=True)

            #_dbg ("X_orig_local ", X_orig_local[:10], geo_loc )

            pos_x, pos_y, pos_z, radius =  estimate_eddy_geometry(X_orig_local)  # needs cartesian coordinates, not geo!

            _dbg("index, pos_x, pos_y, pos_z, radius ",  i, pos_x, pos_y, pos_z, radius )

            lon, lat, depth =np.median(X_orig, axis=0)  # lon lat depth
           
            _dbg("index, lon, lat, depth, radius ",  i, lon, lat, depth, radius) 
            #sys.exit()
            # store circle center from geo_loc
            if geo_loc:
                circles.append({
                    "id": int(fp.get("id")) if fp.get("id") is not None else int(i),
                    "lon": float(geo_loc["lon0"]),
                    "lat": float(geo_loc["lat0"]),
                    "radius_m": float(radius),
                })
    if cid == 0: raise RuntimeError("No point-bearing blocks found.")

    # write eddy centers to JSONL alongside canny_feature_vtk
    if isinstance(original_points, (str, pathlib.Path)):
        p = pathlib.Path(original_points)
        out_path = p.parent / f"{p.stem}_eddies.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in circles:
                f.write(json.dumps(row) + "\n")
        print("saved eddy centers to", out_path)

    lon1 = np.concatenate([np.asarray(d).ravel() for d in lon_all1]) if lon_all1 else np.array([])
    lon2 = np.concatenate([np.asarray(d).ravel() for d in lon_all2]) if lon_all2 else np.array([])
    lat1 = np.concatenate([np.asarray(d).ravel() for d in lat_all1]) if lat_all1 else np.array([])
    lat2 = np.concatenate([np.asarray(d).ravel() for d in lat_all2]) if lat_all2 else np.array([])
    dep1 = np.concatenate([np.asarray(d).ravel() for d in dep_all1]) if dep_all1 else np.array([])
    dep2 = np.concatenate([np.asarray(d).ravel() for d in dep_all2]) if dep_all2 else np.array([])
    #_dbg("lalo ", min(lon_all1), max(lon_all1), min(lat_all1), max(lat_all1))
    #_dbg("lalo ", min(lon_all2), max(lon_all2), min(lat_all2), max(lat_all2))
    _dbg("depth 1 and 2 min and max ", min(dep1), max(dep1), min(dep2), max(dep2))

    # Optional: if you have a depth or scalar to color by

    # ---- 3) Plot on lat lon map ----
    fig = plt.subplots(figsize=(8, 5))
    central_longitude = 0.
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))
    # Apply extent (in geographic coords)
    extent = [-180,180,-80,80]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add simple map features
    ax.coastlines(resolution="110m", linewidth=0.7)
    ax.add_feature(cfeature.LAND, edgecolor="none", facecolor="0.95")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    # plot circles (approx meters -> degrees)
    for c in circles:
        idx = c.get("id")
        lon0 = c.get("lon")
        lat0 = c.get("lat")
        r_m = c.get("radius_m")
        if not np.isfinite([lon0, lat0, r_m]).all():
            continue
        # degrees per meter
        dlat = (r_m / 111320.0)
        dlon = dlat / max(np.cos(np.deg2rad(lat0)), 1e-6)
        theta = np.linspace(0, 2*np.pi, 200)
        lon_c = lon0 + dlon * np.cos(theta)
        lat_c = lat0 + dlat * np.sin(theta)
        lon_c = ((lon_c + 180.0) % 360.0) - 180.0
        ax.plot(lon_c, lat_c, color="black", linewidth=0.8, alpha=0.7,
                transform=ccrs.PlateCarree())

    kwargs = dict(transform=ccrs.PlateCarree(), zorder=5, s=0.30,
                  edgecolors='none', linewidths=0)
    if depth is not None:
        sb = ax.scatter(lon2, lat2, c=dep2, cmap='magma', **kwargs)
        sc = ax.scatter(lon1, lat1, c=dep1, cmap='viridis', **kwargs)
        cb = plt.colorbar(sb, ax=ax, orientation='horizontal', pad=0.03)
        cb.set_label('Depth (m, +down)')
    else:
        ax.scatter(lon2, lat2, color='crimson', **kwargs)

    ax.axis("off")
   
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    plt.axis("off")
    plt.box(False)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.top_labels = False
    ax.right_labels = False
#   ax.bottom_labels = False
#   ax.left_labels = False
#   gl.top_labels = False
#   gl.right_labels = False
#   gl.bottom_labels = True
#   gl.left_labels = True 


    plt.title('VTK Cartesian Points on lon/lat chart')
    plt.tight_layout()
  
    merc_file = pathlib.Path(pathlib.Path(labels_json_path).stem + "_plate").with_suffix(".png")
    merc_dir = pathlib.Path(labels_json_path).parent
    filename = (merc_dir / merc_file)
    _dbg(filename)
    try:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    except Exception as e:
        _dbg("e: ",  e)
        plt.savefig(merc_file, dpi=150, bbox_inches="tight")

def plot_compare_with_original_plt(labels_json_path: str, 
                               original_points,
                               index: int = 0,
                               n_points: int = 512,
                               max_features: int | None = None,
                               max_points_per_cloud: int | None = 256,
                               show_matplotlib: bool = False,
                               mercator_2d: bool = False,
                               indices: int | None = None):
    
    """Load labeled fingerprints, reconstruct feature(s), and compare to original."""
    fps = load_fingerprints_json(labels_json_path)

    _dbg("fps length", len(fps))
    _dbg("fps[0].keys", fps[0].keys() if fps else None)
    _dbg("fps[0][\"grid\"].keys() ", fps[0]["grid"].keys())
    _dbg("index, label, type_label, circularity, pose, len(cloud) ",
         i, fp.get("index"), fp.get("label"), fp.get("type_label"), fp["metrics"].get("circularity"), 
         fp.get("pose"), len_X_orig)

    summarize_metrics(fps)

    if not fps:
        print(" no fingerprints!")
        return None, None

    clouds, mesh = load_point_clouds_vtm(original_points)
    if not clouds:
        print(" no clouds!")
        return None, None

    n_total = min(len(fps), len(clouds))
    step = max(1, int(indices)) if indices is not None else 1

    if max_features is None or max_features <= 0:
        indices = list(range(0, n_total, step))
    elif max_features == 1:
        indices = [min(max(index, 0), n_total - 1)]
    else:
        n = min(n_total, max_features)
        indices = list(range(0, n, step))

    selection = ["eddy", ]
    if show_matplotlib:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i in indices:
            fp = fps[i]
            #_dbg("index, type_label, circularity", i, fp.get("type_label"), fp["metrics"].get("circularity"))
            X_recon = reconstruct_points_from_fingerprint(fp, n_points=n_points, return_geo=False )
            X_orig = clouds[fp.get("id")]
            X_orig = _downsample_points(X_orig, max_points_per_cloud, seed=42)
            X_recon = _downsample_points(X_recon, max_points_per_cloud, seed=42)
            ax.scatter(X_orig[:, 0], X_orig[:, 1], X_orig[:, 2], s=2, c="dodgerblue")
            ax.scatter(X_recon[:, 0], X_recon[:, 1], X_recon[:, 2], s=2, c="orange")

        #ax.legend()

        scat_file = pathlib.Path(pathlib.Path(labels_json_path).stem+"_comp").with_suffix(".png")
        scat_dir = pathlib.Path(labels_json_path).parent.parent/"Plots/"
        plotfile = scat_dir / scat_file
        plt.savefig(plotfile)
        print("saved compare_scatter.png in ", plotfile)
        del plotfile
        if sys.flags.interactive:
            plt.show()
        else:
            pass
        plt.close()

    if mercator_2d:

        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': ccrs.Mercator()})


        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.set_extent([-179, 179, -85, 85], crs=ccrs.PlateCarree())

        for i in indices:
            fp = fps[i]

            if fp.get("type_label") in selection:
                X_orig = clouds[i]
                len_X_orig = len(X_orig)
                _dbg("index, label, type_label, circularity, pose, len(cloud) ",
                     i, fp.get("index"), fp.get("label"), fp.get("type_label"), fp["metrics"].get("circularity"), 
                     fp.get("pose"), len_X_orig)
#                    fp["pose"].get("geo"), len_X_orig)

                if len_X_orig > n_points:
                    X_orig = _downsample_points(clouds[i], n_points, seed=42)
                    X_recon = _downsample_points(reconstruct_points_from_fingerprint(fp, n_points=n_points),
                                                 max_points_per_cloud, seed=42, return_geo=False )
                sc1 = ax.scatter(
                    X_orig[:, 0], X_orig[:, 1],
                    c=-X_orig[:, 2], s=0.3, cmap="viridis",
                    transform=ccrs.PlateCarree(), alpha=0.6, 
                    edgecolors='none', linewidths=0
                                )
                sc2 = ax.scatter(
                    X_recon[:, 0], X_recon[:, 1],
                    c=-X_recon[:, 2], s=3, cmap="plasma",
                    transform=ccrs.PlateCarree(), alpha=0.6
                                )
        plt.colorbar(sc1, ax=ax, label="Depth (m)")    
        merc_file = pathlib.Path(pathlib.Path(labels_json_path).stem + "_mercator").with_suffix(".png")
        merc_dir = pathlib.Path(labels_json_path).parent
        _dbg(merc_dir / merc_file)
        plt.savefig(merc_dir / merc_file, dpi=150, bbox_inches="tight")
        plt.close()

    return 

def estimate_eddy_geometry(X_local: np.ndarray) -> dict:
    """
    Approximate eddy center (x,y,z), depth, and radius from local-tangent meters.
    Returns center (x,y), depth (m, +down), radius (m).
    """
    X = np.asarray(X_local, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or X.size == 0:
        return {"center_xy": [0.0, 0.0], "depth": 0.0, "radius": 0.0}

    # center (robust)
    cx, cy, cz = np.median(X, axis=0)

    # radius from horizontal distances
    r = np.sqrt((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2)
    radius = float(np.median(r))

#   return {"center_xy": [float(cx), float(cy)], "depth": float(cz), "radius": radius}
    return float(cx), float(cy), float(cz),  radius


def main():
    """Plot embeddings for synthetic features and clustered outputs."""
    try:
        from .synthetic import SynthConfig, mixed_point_cloud, save_point_cloud_vtk
    except Exception:
        from synthetic import SynthConfig, mixed_point_cloud, save_point_cloud_vtk


    
    parser = argparse.ArgumentParser(description="Fingerprint pipeline")
    parser.add_argument(
        "outdir_parent",
        nargs="?",
        default= "/data/scratch/orca12/BBecker_frontal_assessment/output/",
        help="enter a full path to output directory",
    )

    parser.add_argument(
        "outdir_nametag",
        nargs="?",
        default="level1_gl_orca12_asm12_20260220_20260222T12/",
        help="enter a file name tag for output filenames",
    )


    args = parser.parse_args()

    print("List of items: {}".format(args))
    #   orca12data = pathlib.Path(args.outdir_nametag)
    indir = args.outdir_parent
    #   outdir = outdir_parent + orca12data.stem
    filetag = args.outdir_nametag

    # save features to "synthetic_feature_point_cloud.vtk"
    #X = mixed_point_cloud(SynthConfig())
    #save_point_cloud_vtk("synthetic_points.vtk", X)

    canny_features_vtk = pathlib.Path(indir + filetag + "CSV/all_three_3D_cluster.vtk")
#   canny_features_vtk = pathlib.Path(indir + filetag + "VTK/all_three_3D_cluster.vtk")
    canny_features_vtk = pathlib.Path(indir + filetag + "VTK/NW_atlantic_salinity_on_salinity_fronts.vtk")
#   canny_features_vtk = pathlib.Path("synthetic_feature_point_cloud.vtk")

    canny_features_vtm = (canny_features_vtk.parent/canny_features_vtk.stem).with_suffix(".vtm")
    fingerprints_jsonl = (canny_features_vtk.parent/canny_features_vtk.stem).with_suffix(".jsonl")

    print(" process one big point cloud to separate into features") 
    print(" point cloud from : ", canny_features_vtk) 
    print(" features point clouds to : ", canny_features_vtm) 
    print(" features finger prints to : ", fingerprints_jsonl) 

    min_cluster_size=1450
    eps = 0.0001
    random_state = 42
    max_workers = 24
    first_stage= "iterative"

    """
    """
    # "detect":
    detect_features_from_vtk(canny_features_vtk, canny_features_vtm,
                             min_cluster_size=min_cluster_size,
                             eps=eps,
                             random_state=random_state,
                             first_stage=first_stage)
    # make "fingerprint":
    fingerprint_features_from_vtm(canny_features_vtk, fingerprints_jsonl,
                                  random_state=random_state,
                                  max_workers=max_workers)
    # "identify":
    identify_features_from_fingerprints( fingerprints_jsonl,
                                        min_cluster_size=min_cluster_size,
                                        random_state=random_state)

    # plot_features_from_json("core/synthetic_features.json", title="Synthetic Features")
    # plot_features_from_json("run_fp.jsonl", title="Clustered Features")
    # for index in range(0,31):
    print("plot_compare_with_original :", fingerprints_jsonl, " with \n", canny_features_vtm)

    plot_compare_with_original_gv(fingerprints_jsonl, canny_features_vtm, \
                               indices=1)
    plot_from_fp_plt(fingerprints_jsonl, canny_features_vtm, \
                      indices=1, show_matplotlib=False, mercator_2d=True)
    plot_features_from_json(fingerprints_jsonl, show=False)
    return

if __name__ == "__main__":
    main()
