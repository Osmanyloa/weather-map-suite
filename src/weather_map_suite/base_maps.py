"""Reusable base map drawing helpers."""

from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from shapely.geometry import shape

def draw_base_map_without_provinces_us_states_only(lon_min=-105, lon_max=-33, lat_min=0, lat_max=34):
    """
    Create an empty 3D-style base map:
      - International borders for all countries (admin_0)
      - Internal admin_1 divisions only for the United States
      - No internal divisions for the remaining countries
      - Latitude/longitude grid labels
    Returns: fig, ax
    """
    fig = plt.figure(figsize=(18, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='darkgreen', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='skyblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    countries_shp = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    reader0 = shpreader.Reader(countries_shp)
    country_geoms = [rec.geometry for rec in reader0.records()]
    ax.add_geometries(country_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.3, alpha=0.35, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=5)
    ax.add_geometries(country_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.2, path_effects=[PathEffects.Normal()], zorder=6)
    try:
        admin1_shp = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
        reader1 = shpreader.Reader(admin1_shp)
        us_state_geoms = []
        mexico_geom = None
        for rec0 in reader0.records():
            if rec0.attributes.get('NAME') == 'Mexico':
                mexico_geom = rec0.geometry
                break
        for rec in reader1.records():
            attrs = rec.attributes
            admin_name = attrs.get('admin') or attrs.get('ADMIN') or attrs.get('Admin')
            if admin_name is not None and admin_name.strip().lower() in ('united states of america', 'united states'):
                state_geom = rec.geometry
                for coast_geom in coastline_geoms:
                    if state_geom.intersects(coast_geom):
                        try:
                            state_geom = state_geom.difference(coast_geom.buffer(0.05))
                        except Exception:
                            continue
                if mexico_geom and state_geom.intersects(mexico_geom):
                    try:
                        state_geom = state_geom.difference(mexico_geom.buffer(0.05))
                    except Exception:
                        continue
                if state_geom and state_geom.is_valid and (not state_geom.is_empty):
                    us_state_geoms.append(state_geom)
        if us_state_geoms:
            ax.add_geometries(us_state_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.45, zorder=8, path_effects=[PathEffects.Normal()])
    except Exception as e:
        print('⚠️ Could not load Natural Earth admin_1 geometries (states).')
        print('   Error:', e)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    plt.show()
    return (fig, ax)
