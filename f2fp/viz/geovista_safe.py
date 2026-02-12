from __future__ import annotations
import os
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import geovista as gv
from geovista import GeoPlotter
from ..common.geo import to_cartesian, as_polyline

def screenshot_ring(center_lon, center_lat, radius_deg, out_png: str) -> None:
    t = np.linspace(0, 2*np.pi, 360)
    lons = center_lon + radius_deg*np.cos(t)
    lats = center_lat + radius_deg*np.sin(t)
    plot = GeoPlotter()
    globe = plot.add_base_layer()
    xyz = to_cartesian(lons, lats, globe_mesh=globe, close=True)
    ring = as_polyline(xyz, closed=True)
    plot.add_mesh(ring, color="yellow", render_lines_as_tubes=True, line_width=4)
    plot.camera.zoom(1.5)
    plot.screenshot(out_png)
    plot.close()
