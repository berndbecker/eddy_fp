
import numpy as np

"""
Fixed synthetic eddy generator test for GeoVista.

This version guarantees that synthetic eddies appear
clearly on the 3D globe. The key fixes are:

– Correct lat/lon ranges
– Convert synthetic geometry to real coordinates
– Project via gv.common.to_cartesian()
– Avoid masked arrays
– Properly close polygons
"""

import numpy as np
import geovista as gv
from geovista import GeoPlotter
import geovista.common as gv_common



def synthetic_ring(center_lon, center_lat, radius_deg=3.0, n=120):
    """
    Generate a synthetic circular eddy in LAT/LON space.
    radius_deg is IN DEGREES, not km.

    This alone makes the feature directly compatible with GeoVista.
    """
    theta = np.linspace(0, 2 * np.pi, n)
    lon = center_lon + radius_deg * np.cos(theta)
    lat = center_lat + radius_deg * np.sin(theta)

    # Normalize ranges
    lon = (lon + 180) % 360 - 180
    lat = np.clip(lat, -90, 90)

    # Close polygon
    if lon[0] != lon[-1] or lat[0] != lat[-1]:
        lon = np.append(lon, lon[0])
        lat = np.append(lat, lat[0])

    return lon.astype(float), lat.astype(float)

def make_ring_feature(lon0, lat0, depth0, radius_km=50, n_pts=2000, noise=0.05, depth_span_m=400):
    deg_per_km = 1.0 / 111.0
    r_deg = radius_km * deg_per_km
    theta = np.random.rand(n_pts) * 2*np.pi
    r = r_deg * (1 + 0.1*np.random.randn(n_pts))
    lon = lon0 + r * np.cos(theta) + noise * (np.random.rand(n_pts) - 0.5) * r_deg
    lat = lat0 + r * np.sin(theta) + noise * (np.random.rand(n_pts) - 0.5) * r_deg
    depth = depth0 + (np.random.rand(n_pts) - 0.5) * depth_span_m
    return lon, lat, depth

def make_curtain_feature(lon0, lat0, depth0, length_km=300, width_km=10, n_pts=2000, noise=0.05, depth_span_m=1200):
    deg_per_km = 1.0 / 111.0
    L = length_km * deg_per_km
    W = width_km * deg_per_km
    az = np.random.rand() * 2*np.pi
    u = np.array([np.cos(az), np.sin(az)])
    v = np.array([-np.sin(az), np.cos(az)])
    t = (np.random.rand(n_pts) - 0.5) * L
    s = (np.random.rand(n_pts) - 0.5) * W
    xy = np.outer(t, u) + np.outer(s, v)
    lon = lon0 + xy[:, 0] + noise * (np.random.rand(n_pts) - 0.5) * W
    lat = lat0 + xy[:, 1] + noise * (np.random.rand(n_pts) - 0.5) * W
    depth = depth0 + (np.random.rand(n_pts) - 0.5) * depth_span_m
    return lon, lat, depth

def make_mixed_scene(n_rings=6, n_curtains=6, seed=42):
    np.random.seed(seed)
    features = []
    fid = 100
    for _ in range(n_rings):
        lon0 = -170 + 340*np.random.rand(); lat0 = -60 + 120*np.random.rand(); depth0 = 100 + 700*np.random.rand()
        lon, lat, depth = make_ring_feature(lon0, lat0, depth0,
                                            radius_km=np.random.uniform(30, 80),
                                            n_pts=np.random.randint(1500, 3000),
                                            noise=0.05, depth_span_m=np.random.uniform(200, 600))
        features.append((fid, lon, lat, depth)); fid += 1
    for _ in range(n_curtains):
        lon0 = -170 + 340*np.random.rand(); lat0 = -60 + 120*np.random.rand(); depth0 = 100 + 1200*np.random.rand()
        lon, lat, depth = make_curtain_feature(lon0, lat0, depth0,
                                               length_km=np.random.uniform(200, 500),
                                               width_km=np.random.uniform(8, 20),
                                               n_pts=np.random.randint(1500, 3000),
                                               noise=0.05, depth_span_m=np.random.uniform(800, 1500))
        features.append((fid, lon, lat, depth)); fid += 1
    return features

def test_geovista_render():
    """
    Test plotting synthetic eddies on a GeoVista globe.
    """

    # ------------------------------------------------------
    # 1. Create example synthetic eddies in geographic space
    # ------------------------------------------------------
    lons1, lats1 = synthetic_ring(-30, 15, radius_deg=5)
    lons2, lats2 = synthetic_ring(60, -20, radius_deg=3)

    # ------------------------------------------------------
    # 2. Project to 3D coordinates for GeoVista
    # ------------------------------------------------------
    #xyz1 = gv.transform.points(lon=lons1, lat=lats1)
    #xyz2 = gv.transform.points(lon=lons2, lat=lats2)
    earth_radius = 6371.
    zscale = 1./earth_radius
#   heights = len(lons1) * [-5.]
    xyz1 = gv_common.to_cartesian(lons1, lats1) # , zlevel=heights, zscale=zscale, stacked=True)
    xyz2 = gv_common.to_cartesian(lons2, lats2) # , zlevel=heights, zscale=zscale, stacked=True)
    print(xyz1[:12])

    features = make_mixed_scene()
    # ------------------------------------------------------
    # 3. Render on the GeoVista globe
    # ------------------------------------------------------
    plotter = gv.GeoPlotter()
    plotter.add_base_layer()

    plotter.add_mesh(xyz1, color="yellow", render_lines_as_tubes=True, line_width=4)
    plotter.add_mesh(xyz2, color="red", render_lines_as_tubes=True, line_width=4)
    for feat in features:
        print(feat)
#       points = np.vstack([feat[1], feat[2], feat[3]])
        points = gv.common.to_cartesian(feat[1], feat[2], zlevel=-feat[3], zscale=zscale)
        print(points)
        plotter.add_mesh(points, color="blue", render_lines_as_tubes=True, line_width=4)
    

    plotter.show()


if __name__ == "__main__":
    test_geovista_render()
