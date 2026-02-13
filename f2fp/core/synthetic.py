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
    rng = np.random.default_rng(cfg.seed + 2)
    # Number of noise points to add
    n_noise = int(cfg.noise_ratio * (lon.size + lat.size) / 2)
    # Generate random noise points
    lon_n = rng.uniform(-180, 180, n_noise)
    lat_n = rng.uniform(-90,  90,  n_noise)
    # Concatenate original and noise points
    return np.concatenate([lon, lon_n]), np.concatenate([lat, lat_n])
