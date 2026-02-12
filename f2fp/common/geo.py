from __future__ import annotations
import numpy as np
import pyvista as pv
from typing import Optional, Tuple

def _wrap_lonlat(lon, lat) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = (lon + 180.0) % 360.0 - 180.0
    lat = np.clip(lat, -90.0, 90.0)
    return lon, lat

def _assert_finite(*arrays) -> None:
    for a in arrays:
        if not np.isfinite(a).all():
            idx = np.where(~np.isfinite(a))[0][:10]
            raise ValueError(f"Non‑finite values in array at indices {idx.tolist()}")

def sphere_radius_from_bounds(mesh: pv.DataSet) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    rx = 0.5 * (xmax - xmin); ry = 0.5 * (ymax - ymin); rz = 0.5 * (zmax - zmin)
    return float(np.median([rx, ry, rz]))

def to_cartesian(
    lon_deg,
    lat_deg,
    *,
    globe_mesh: Optional[pv.DataSet] = None,
    radius: Optional[float] = None,
    close: bool = False,
) -> np.ndarray:
    lon, lat = _wrap_lonlat(lon_deg, lat_deg)
    if close and (lon[0] != lon[-1] or lat[0] != lat[-1]):
        lon = np.append(lon, lon[0])
        lat = np.append(lat, lat[0])

    R = float(radius) if radius is not None else 1.0
    if globe_mesh is not None:
        R = sphere_radius_from_bounds(globe_mesh)

    lon_r = np.radians(lon); lat_r = np.radians(lat)
    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    xyz = np.column_stack((x, y, z)).astype(np.float64, copy=False)
    _assert_finite(xyz)
    return xyz

def as_points_polydata(xyz: np.ndarray) -> pv.PolyData:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")
    return pv.PolyData(np.ascontiguousarray(xyz))

def as_polyline(xyz: np.ndarray, *, closed: bool = False) -> pv.PolyData:
    pts = np.ascontiguousarray(xyz)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")
    n = pts.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 vertices for a line")
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]]); n = pts.shape[0]
    poly = pv.PolyData(pts)
    cell = np.empty(n + 1, dtype=np.int64)
    cell[0] = n; cell[1:] = np.arange(n, dtype=np.int64)
    poly.lines = cell
    return poly
