from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class SynthConfig:
    center_lon: float = -30.0
    center_lat: float =  15.0
    radius_deg: float =   5.0
    seed: int = 42
    n_ring: int = 2048
    n_curtain: int = 2048
    noise_ratio: float = 0.2

def ring_points(cfg: SynthConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    theta = rng.uniform(0, 2*np.pi, cfg.n_ring)
    lon = cfg.center_lon + cfg.radius_deg*np.cos(theta)
    lat = cfg.center_lat + cfg.radius_deg*np.sin(theta)
    lon = (lon + 180) % 360 - 180
    lat = np.clip(lat, -90, 90)
    return lon, lat

def curtain_points(cfg: SynthConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 1)
    lon = cfg.center_lon + rng.normal(0, 0.25, cfg.n_curtain)
    lat = cfg.center_lat + rng.uniform(-cfg.radius_deg, cfg.radius_deg, cfg.n_curtain)
    lon = (lon + 180) % 360 - 180
    lat = np.clip(lat, -90, 90)
    return lon, lat

def mix_with_noise(lon, lat, cfg: SynthConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 2)
    n_noise = int(cfg.noise_ratio * (lon.size + lat.size) / 2)
    lon_n = rng.uniform(-180, 180, n_noise)
    lat_n = rng.uniform(-90,  90,  n_noise)
    return np.concatenate([lon, lon_n]), np.concatenate([lat, lat_n])
