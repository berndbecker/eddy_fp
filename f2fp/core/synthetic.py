from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class SynthConfig:
    """Configuration for synthetic eddy generation."""
    # Center longitude and latitude for eddy
    center_lon: float = -30.0
    center_lat: float =  15.0
    # Radius in degrees
    radius_deg: float =   5.0
    # Random seed for reproducibility
    seed: int = 42
    # Number of points for ring and curtain
    n_ring: int = 2048
    n_curtain: int = 2048
    # Ratio of noise points to add
    noise_ratio: float = 0.2

def ring_points(cfg: SynthConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate points in a ring shape around a center."""
    assert cfg.n_ring > 0, "n_ring must be positive"
    assert cfg.radius_deg > 0, "radius_deg must be positive"
    rng = np.random.default_rng(cfg.seed)
    # Uniformly sample angles for ring
    theta = rng.uniform(0, 2*np.pi, cfg.n_ring)
    # Convert polar coordinates to lon/lat
    lon = cfg.center_lon + cfg.radius_deg*np.cos(theta)
    lat = cfg.center_lat + cfg.radius_deg*np.sin(theta)
    # Wrap longitude to [-180, 180]
    lon = (lon + 180) % 360 - 180
    # Clamp latitude to [-90, 90]
    lat = np.clip(lat, -90, 90)
    return lon, lat

def curtain_points(cfg: SynthConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate points in a curtain shape around a center."""
    assert cfg.n_curtain > 0, "n_curtain must be positive"
    rng = np.random.default_rng(cfg.seed + 1)
    # Longitude: normal distribution around center
    lon = cfg.center_lon + rng.normal(0, 0.25, cfg.n_curtain)
    # Latitude: uniform distribution within radius
    lat = cfg.center_lat + rng.uniform(-cfg.radius_deg, cfg.radius_deg, cfg.n_curtain)
    lon = (lon + 180) % 360 - 180
    lat = np.clip(lat, -90, 90)
    return lon, lat

def mix_with_noise(lon, lat, cfg: SynthConfig) -> tuple[np.ndarray, np.ndarray]:
    """Mix input points with random noise points."""
    assert lon.shape == lat.shape, "lon and lat must have the same shape"
    assert 0.0 <= cfg.noise_ratio, "noise_ratio must be non-negative"
    rng = np.random.default_rng(cfg.seed + 2)
    # Number of noise points to add (same base as input length)
    n_noise = int(cfg.noise_ratio * lon.size)
    # Generate random noise points
    lon_n = rng.uniform(-180, 180, n_noise)
    lat_n = rng.uniform(-90,  90,  n_noise)
    # Concatenate original and noise points
    return np.concatenate([lon, lon_n]), np.concatenate([lat, lat_n])

def to_point_cloud(lon: np.ndarray, lat: np.ndarray, depth: float = 0.0) -> np.ndarray:
    """Convert lon/lat arrays to an (N,3) point cloud."""
    return np.stack([lon, lat, np.full_like(lon, depth, dtype=float)], axis=1)

def mixed_point_cloud(cfg: SynthConfig,
                      include_ring: bool = True,
                      include_curtain: bool = True,
                      add_noise: bool = True) -> np.ndarray:
    """Build a mixed synthetic point cloud (ring + curtain + noise)."""
    lons = []
    lats = []
    if include_ring:
        lon_r, lat_r = ring_points(cfg)
        lons.append(lon_r); lats.append(lat_r)
    if include_curtain:
        lon_c, lat_c = curtain_points(cfg)
        lons.append(lon_c); lats.append(lat_c)
    if not lons:
        return np.zeros((0, 3), dtype=float)
    lon = np.concatenate(lons)
    lat = np.concatenate(lats)
    if add_noise:
        lon, lat = mix_with_noise(lon, lat, cfg)
    return to_point_cloud(lon, lat)

def save_point_cloud_vtk(path: str, points: np.ndarray) -> None:
    """Save raw point cloud to .vtk using pyvista or vtk fallback."""
    try:
        import pyvista as pv
        pv.PolyData(points).save(path)
        return
    except Exception:
        try:
            import vtk
            pts = vtk.vtkPoints()
            for p in points:
                pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(path)
            writer.SetInputData(poly)
            writer.Write()
        except Exception:
            pass

def merged_point_cloud(cfgs: list[SynthConfig],
                       include_ring: bool = True,
                       include_curtain: bool = True,
                       add_noise: bool = True) -> np.ndarray:
    """Build one big point cloud by merging multiple synthetic features."""
    clouds = []
    for cfg in cfgs:
        clouds.append(mixed_point_cloud(cfg,
                                        include_ring=include_ring,
                                        include_curtain=include_curtain,
                                        add_noise=add_noise))
    return np.vstack(clouds) if clouds else np.zeros((0, 3), dtype=float)

def save_merged_point_cloud_vtk(path: str,
                                cfgs: list[SynthConfig] | np.ndarray | list[np.ndarray],
                                include_ring: bool = True,
                                include_curtain: bool = True,
                                add_noise: bool = True) -> np.ndarray:
    """Create one big merged cloud (or save a provided cloud) and save it to .vtk."""
    if isinstance(cfgs, np.ndarray):
        X = np.asarray(cfgs, dtype=float)
    elif isinstance(cfgs, list) and cfgs and isinstance(cfgs[0], np.ndarray):
        X = np.vstack([np.asarray(c, dtype=float) for c in cfgs])
    else:
        X = merged_point_cloud(cfgs,
                               include_ring=include_ring,
                               include_curtain=include_curtain,
                               add_noise=add_noise)
    save_point_cloud_vtk(path, X)
    return X

def make_separable_configs(n: int,
                           base: SynthConfig | None = None,
                           separation_deg: float = 10.0) -> list[SynthConfig]:
    """Create configs with spaced centers for easier first-stage clustering tests."""
    base = base or SynthConfig()
    cfgs = []
    for i in range(n):
        cfgs.append(SynthConfig(
            center_lon=base.center_lon + i * separation_deg,
            center_lat=base.center_lat + (i % 2) * (separation_deg / 2),
            radius_deg=base.radius_deg,
            seed=base.seed + i,
            n_ring=base.n_ring,
            n_curtain=base.n_curtain,
            noise_ratio=base.noise_ratio,
        ))
    return cfgs
