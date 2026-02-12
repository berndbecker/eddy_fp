from __future__ import annotations
import numpy as np

def _wrap_lon(lon): return (lon + 180.0) % 360.0 - 180.0

def normalize_coords_names(obj):
    try: import xarray as xr
    except Exception: xr = None
    try: import pandas as pd
    except Exception: pd = None
    try: import geopandas as gpd
    except Exception: gpd = None

    if xr is not None and isinstance(obj, (xr.Dataset, xr.DataArray)):
        ds = obj
        ren = {}
        for cand, tgt in (("longitude","lon"),("nav_lon","lon"),("latitude","lat"),("nav_lat","lat")):
            if cand in ds.coords and tgt not in ds.coords: ren[cand] = tgt
        if ren: ds = ds.rename(ren)
        if "lon" not in ds.coords or "lat" not in ds.coords:
            raise KeyError("xarray object lacks lon/lat after normalization.")
        ds = ds.assign_coords(lon=_wrap_lon(ds.lon), lat=ds.lat.clip(min=-90.0, max=90.0))
        ds = ds.assign_coords(lon=ds.lon.astype("float64"), lat=ds.lat.astype("float64"))
        return ds

    if pd is not None and isinstance(obj, pd.DataFrame):
        df = obj.copy()
        for cand, tgt in (("longitude","lon"),("nav_lon","lon"),("latitude","lat"),("nav_lat","lat")):
            if cand in df.columns and tgt not in df.columns: df.rename(columns={cand:tgt}, inplace=True)
        if gpd is not None and isinstance(df, gpd.GeoDataFrame):
            if df.crs is None or str(df.crs).lower() not in ("epsg:4326","epsg: 4326"):
                try: df = df.set_crs(4326, allow_override=True)
                except Exception: pass
        if "lon" not in df or "lat" not in df:
            raise KeyError("DataFrame lacks lon/lat after normalization.")
        df["lon"] = _wrap_lon(df["lon"].astype("float64"))
        df["lat"] = df["lat"].astype("float64").clip(-90.0, 90.0)
        return df

    return obj

def select_window(obj, center_lon, center_lat, half_width_deg=5.0):
    center_lon = _wrap_lon(float(center_lon)); center_lat = float(center_lat); w = float(half_width_deg)
    try:
        import xarray as xr
        if isinstance(obj, (xr.Dataset, xr.DataArray)):
            ds = normalize_coords_names(obj)
            return ds.where(
                (ds.lon >= center_lon - w) & (ds.lon <= center_lon + w) &
                (ds.lat >= center_lat - w) & (ds.lat <= center_lat + w),
                drop=True
            )
    except Exception: pass
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            df = normalize_coords_names(obj)
            m = ((df["lon"] >= center_lon - w) & (df["lon"] <= center_lon + w) &
                 (df["lat"] >= center_lat - w) & (df["lat"] <= center_lat + w))
            return df.loc[m]
    except Exception: pass
    raise TypeError("Unsupported object type for select_window().")
