"""Ocean and sea-surface map helpers."""

from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as gridliner
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from shapely.vectorized import contains

from .config import LOGO_PATH, OUTPUT_MAPS_DIR

def fill_entire_ocean_grid(data_array):
    lons, lats = np.meshgrid(data_array.longitude.values, data_array.latitude.values)
    valid_mask = ~np.isnan(data_array.values)
    valid_points = np.column_stack((lons[valid_mask], lats[valid_mask]))
    valid_values = data_array.values[valid_mask]
    all_points = np.column_stack((lons.ravel(), lats.ravel()))
    full_ocean = griddata(valid_points, valid_values, all_points, method='linear', fill_value=np.nan)
    remaining_nans = np.isnan(full_ocean)
    if np.any(remaining_nans):
        full_ocean = griddata(valid_points, valid_values, all_points, method='nearest')
    result = data_array.copy()
    result.values = full_ocean.reshape(lons.shape)
    print('Full ocean grid created')
    return result


def remove_data_under_land(data_array):
    land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m')
    land_geoms = list(land_feature.geometries())
    lons, lats = np.meshgrid(data_array.longitude.values, data_array.latitude.values)
    ocean_mask = np.ones_like(lons, dtype=bool)
    for geom in land_geoms:
        is_land = contains(geom, lons, lats)
        ocean_mask &= ~is_land
    result = data_array.copy()
    result.values = np.where(ocean_mask, data_array.values, np.nan)
    print('Datos bajo tierra eliminados')
    return result
