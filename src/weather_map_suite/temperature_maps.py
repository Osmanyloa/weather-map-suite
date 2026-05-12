"""Regional temperature map renderers extracted from the source notebook."""

from __future__ import annotations

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

from .config import LOGO_PATH, OUTPUT_MAPS_DIR, SHAPEFILES_DIR

def plot_cuba_temperature(ds_surface, tipo='maxima', variable='temperature', cuba=None, admin_boundaries=None):
    """
    Plot temperatures o apparent temperature maximum o minimum for Cuba

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature/apparent temperature en °C)
    tipo : str
        'maxima' o 'minima' - defines which temperature type to plot
    variable : str
        'temperature' o 'sensacion' - define si graficar temperature o apparent temperature
    cuba : GeoDataFrame
        GeoDataFrame with Cuba geometry
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all countries (optional if Cuba geometry is already provided)
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    if variable.lower() not in ['temperature', 'sensacion']:
        raise ValueError("The parameter 'variable' must be 'temperature' o 'sensacion'")
    tipo = tipo.lower()
    variable = variable.lower()
    var_name = 't2m'
    lon_min_deg = -87
    lon_max_deg = -73
    lat_min = 18
    lat_max = 25
    lon_orig = ds_surface[var_name].longitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    temp_corrected = ds_surface[var_name].assign_coords(longitude=lon_corrected)
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    if temp_area.values.ndim == 3:
        temp_vals = temp_area.values[0, :, :]
    else:
        temp_vals = temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    if cuba is None and admin_boundaries is not None:
        cuba = admin_boundaries[admin_boundaries['admin'] == 'Cuba'].copy()
    elif cuba is None:
        raise ValueError("You must provide 'cuba' o 'admin_boundaries'")
    cuba_union = cuba.unary_union
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_cuba = gdf_points_fine.within(cuba_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_cuba, temp_fine, np.nan)
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    cuba.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=5)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if tipo == 'maxima':
        cmap_full = plt.get_cmap('turbo')
        cmap_truncated = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', cmap_full(np.linspace(0.35, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_truncated = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    level_count = 15
    levels = np.linspace(vmin, vmax, level_count)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_truncated, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°C)', fontsize=11)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'PR': {'lat': 22.42, 'lon': -83.7}, 'HAB': {'lat': 23.17, 'lon': -82.28}, 'MTZ': {'lat': 23.04, 'lon': -81.58}, 'Nueva Gerona': {'lat': 21.88, 'lon': -82.8}, 'Colón': {'lat': 22.72, 'lon': -80.91}, 'SC': {'lat': 22.41, 'lon': -79.97}, 'SSP': {'lat': 21.93, 'lon': -79.43}, 'CAV': {'lat': 21.84, 'lon': -78.76}, 'CMG': {'lat': 21.38, 'lon': -77.91}, 'Moa': {'lat': 20.66, 'lon': -74.945}, 'HOL': {'lat': 20.89, 'lon': -76.26}, 'Bayamo': {'lat': 20.38, 'lon': -76.64}, 'SCU': {'lat': 20.02, 'lon': -75.82}, 'Maisí': {'lat': 20.24, 'lon': -74.15}, 'LTU': {'lat': 20.96, 'lon': -76.95}, 'CFG': {'lat': 22.15, 'lon': -80.44}, 'ART': {'lat': 22.81, 'lon': -82.76}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci = datos['lat']
        lon_ci = datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(lon_ci, lat_ci + 0.08, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
        ax.text(lon_ci, lat_ci - 0.08, nombre, fontsize=8.0, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.198, 0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print('Logo not found')
    ax.text(0.5, 0.39, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold')
    variable_name = 'temperature' if variable == 'temperature' else 'sensacion_termica'
    plt.savefig(f'output/maps/cuba/cuba_map_{variable_name}_{tipo}.png', dpi=800, bbox_inches='tight')
    plt.show()
    return (fig, ax)


def plot_mexico_temperature(ds_surface, tipo='maxima', admin_boundaries=None):
    """
    Plot maximum or minimum temperatures for Mexico

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima' - defines which temperature type to plot
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    lon_min_deg = -119
    lon_max_deg = -86
    lat_min = 14
    lat_max = 33
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    mexico = admin_boundaries[admin_boundaries['admin'] == 'Mexico'].copy()
    lon_orig = ds_surface['t2m'].longitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    temp_corrected = ds_surface['t2m'].assign_coords(longitude=lon_corrected)
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    if temp_area.values.ndim == 3:
        temp_vals = temp_area.values[0, :, :]
    else:
        temp_vals = temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    mexico_union = mexico.unary_union
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_mexico = gdf_points_fine.within(mexico_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_mexico, temp_fine, np.nan)
    fig = plt.figure(figsize=(15, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    buffer_lat = 1
    ax.set_extent([lon_min_deg, lon_max_deg + buffer_lat, lat_min - buffer_lat, lat_max + buffer_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    cmap_truncated = 'turbo'
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    level_count = 30
    levels = np.linspace(vmin, vmax, level_count)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_truncated, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='40%', height='5%', loc='lower left', bbox_to_anchor=(-0.03, 0.02, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°C)', fontsize=11)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    mexico_geom = mexico.unary_union
    ax.add_geometries([mexico_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=5, alpha=0.8, zorder=5, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)])
    ax.add_geometries([mexico_geom], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='white', linewidth=2, zorder=8, path_effects=[PathEffects.Normal()])
    try:
        mexico_boundary = mexico_geom.boundary
        all_state_lines = []
        for geometry in mexico.geometry:
            if hasattr(geometry, 'boundary'):
                boundary = geometry.boundary
                if isinstance(boundary, LineString):
                    all_state_lines.append(boundary)
                elif isinstance(boundary, MultiLineString):
                    all_state_lines.extend(boundary.geoms)
        if all_state_lines:
            all_lines = unary_union(all_state_lines)
            interior_lines = all_lines.difference(mexico_boundary.buffer(0.01))
            if not interior_lines.is_empty:
                ax.add_geometries([interior_lines], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4, alpha=0.8, zorder=9)
    except Exception as e:
        print(f'Error drawing internal boundaries: {e}')
    major_cities = {'Mexicali': {'state': 'Baja California', 'lat': 32.636, 'lon': -115.475}, 'La Paz': {'state': 'Baja California Sur', 'lat': 24.142, 'lon': -110.313}, 'Hermosillo': {'state': 'Sonora', 'lat': 29.075, 'lon': -110.958}, 'Culiacán': {'state': 'Sinaloa', 'lat': 24.79, 'lon': -107.387}, 'Tepic': {'state': 'Nayarit', 'lat': 21.5, 'lon': -104.9}, 'Guadalajara': {'state': 'Jalisco', 'lat': 20.666, 'lon': -103.391}, 'Cd Colima': {'state': 'Colima', 'lat': 19.1, 'lon': -103.9}, 'Acapulco': {'state': 'Guerrero', 'lat': 16.862, 'lon': -99.887}, 'Oaxaca': {'state': 'Oaxaca', 'lat': 17.06, 'lon': -96.723}, 'Salina Cruz': {'state': 'Oaxaca', 'lat': 16.2, 'lon': -95.195}, 'Tuxtla': {'state': 'Chiapas', 'lat': 16.759, 'lon': -93.113}, 'Villahermosa': {'state': 'Tabasco', 'lat': 17.986, 'lon': -92.93}, 'Cd de Campeche': {'state': 'Campeche', 'lat': 19.843, 'lon': -90.525}, 'Mérida': {'state': 'Yucatán', 'lat': 20.975, 'lon': -89.616}, 'Cancún': {'state': 'Quintana Roo', 'lat': 21.174, 'lon': -86.846}, 'Chetumal': {'state': 'Quintana Roo', 'lat': 18.514, 'lon': -88.303}, 'Cd Veracruz': {'state': 'Veracruz', 'lat': 19.1809, 'lon': -96.142}, 'Poza Rica': {'state': 'Veracruz', 'lat': 20.533, 'lon': -97.459}, 'Cd México': {'state': 'Ciudad de México', 'lat': 19.4369, 'lon': -99.1397}, 'León': {'state': 'Guanajuato', 'lat': 21.129, 'lon': -101.673}, 'Cd Zacatecas': {'state': 'Zacatecas', 'lat': 22.768, 'lon': -102.581}, 'Cd Mante': {'state': 'Tamaulipas', 'lat': 22.743, 'lon': -98.973}, 'Reynosa': {'state': 'Tamaulipas', 'lat': 26.08, 'lon': -98.288}, 'Nuevo Laredo': {'state': 'Tamaulipas', 'lat': 27.41, 'lon': -99.59}, 'Monterrey': {'state': 'Nuevo León', 'lat': 25.675, 'lon': -100.318}, 'Piedras Negras': {'state': 'Coahuila', 'lat': 28.7, 'lon': -100.523}, 'Torreón': {'state': 'Durango', 'lat': 25.543, 'lon': -103.418}, 'Cd Durango': {'state': 'Durango', 'lat': 24.934, 'lon': -104.911}, 'Cd Juárez': {'state': 'Chihuahua', 'lat': 31.72, 'lon': -106.46}, 'Cd Chihuahua': {'state': 'Chihuahua', 'lat': 28.635, 'lon': -106.088}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci = datos['lat']
        lon_ci = datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(lon_ci, lat_ci + 0.18, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.195', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
        ax.text(lon_ci, lat_ci - 0.16, nombre, fontsize=8, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black', facecolor='none'), linewidth=0.8, alpha=0.3, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.4, edgecolor='black', alpha=0.3, zorder=5)
    ax.add_feature(cfeature.STATES, linewidth=1, edgecolor='black', alpha=0.2, zorder=5)
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower left', bbox_to_anchor=(0.86, 0.04, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print('Logo not found')
    ax.text(0.97, 0.03, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', zorder=35, path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.Normal()])
    output_file = f'output/maps/mexico/mexico_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'Map saved: {output_file}')
    plt.show()
    return (fig, ax)


def plot_hispaniola_temperature(ds_surface, tipo='maxima', admin_boundaries=None):
    """
    Plot maximum or minimum temperatures for Hispaniola (Haiti and the Dominican Republic)

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima' - defines which temperature type to plot
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    haiti = admin_boundaries[admin_boundaries['admin'] == 'Haiti'].copy()
    republica_dominicana = admin_boundaries[admin_boundaries['admin'] == 'Dominican Republic'].copy()
    hispaniola = pd.concat([haiti, republica_dominicana])
    lon_min_deg = -76
    lon_max_deg = -67
    lat_min = 16
    lat_max = 20.8
    lon_orig = ds_surface['t2m'].longitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    temp_corrected = ds_surface['t2m'].assign_coords(longitude=lon_corrected)
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    if temp_area.values.ndim == 3:
        temp_vals = temp_area.values[0, :, :]
    else:
        temp_vals = temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    hispaniola_union = hispaniola.unary_union
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_hispaniola = gdf_points_fine.within(hispaniola_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_hispaniola, temp_fine, np.nan)
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    international_border = haiti.unary_union.intersection(republica_dominicana.unary_union)
    ax.add_geometries([international_border], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries([international_border], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    haiti_provinces = admin_boundaries[(admin_boundaries['admin'] == 'Haiti') & (admin_boundaries['type'] != 'Country')]
    haiti_provinces.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3, zorder=4, transform=ccrs.PlateCarree())
    dominican_republic_provinces = admin_boundaries[(admin_boundaries['admin'] == 'Dominican Republic') & (admin_boundaries['type'] != 'Country')]
    dominican_republic_provinces.boundary.plot(ax=ax, edgecolor='black', linewidth=0.3, zorder=4, transform=ccrs.PlateCarree())
    if tipo == 'maxima':
        cmap_full = plt.get_cmap('turbo')
        cmap_truncated = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', cmap_full(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_truncated = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    level_count = 15
    levels = np.linspace(vmin, vmax, level_count)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_truncated, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°C)', fontsize=11)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'Puerto Príncipe': {'lat': 18.54, 'lon': -72.34}, 'Santiago': {'lat': 19.45, 'lon': -70.7}, 'Santo Domingo': {'lat': 18.48, 'lon': -69.9}, 'Cotuí': {'lat': 19.08, 'lon': -70.16}, 'San Juan': {'lat': 18.8, 'lon': -71.22}, 'Barahona': {'lat': 18.2, 'lon': -71.1}, 'Punta Cana': {'lat': 18.58, 'lon': -68.4}, 'Jérémie': {'lat': 18.64, 'lon': -74.12}, 'La Romana': {'lat': 18.43, 'lon': -68.97}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci = datos['lat']
        lon_ci = datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(lon_ci, lat_ci + 0.08, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
        ax.text(lon_ci, lat_ci - 0.08, nombre, fontsize=8, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.198, 0.19, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print('Logo not found')
    ax.text(0.8, 0.35, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', zorder=35, path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.Normal()])
    output_file = f'output/maps/hispaniola/hispaniola_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'Map saved: {output_file}')
    plt.show()
    return (fig, ax)


def plot_puerto_rico_temperature(ds_surface, tipo='maxima', admin_boundaries=None, puerto_rico_counties=None):
    """
    Plot maximum or minimum temperatures para Puerto Rico

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima' - defines which temperature type to plot
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    municipios_pr : GeoDataFrame
        GeoDataFrame with Puerto Rico municipalities (optional)
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    lon_min_deg = -68.0
    lon_max_deg = -65
    lat_min = 17.2
    lat_max = 19.0
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    puerto_rico = admin_boundaries[admin_boundaries['admin'] == 'Puerto Rico'].copy()
    lon_orig = ds_surface['t2m'].longitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    temp_corrected = ds_surface['t2m'].assign_coords(longitude=lon_corrected)
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    if temp_area.values.ndim == 3:
        temp_vals = temp_area.values[0, :, :]
    else:
        temp_vals = temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 300)
    lat_fine = np.linspace(lat.min(), lat.max(), 300)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    temp_fine_smooth = gaussian_filter(temp_fine, sigma=1)
    puerto_rico_union = puerto_rico.unary_union
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_pr = gdf_points_fine.within(puerto_rico_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_pr, temp_fine_smooth, np.nan)
    temp_masked_fine_f = temp_masked_fine * 9 / 5 + 32
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    puerto_rico.boundary.plot(ax=ax, edgecolor='black', linewidth=0.7, zorder=5)
    if puerto_rico_counties is not None:
        puerto_rico_counties.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.2, alpha=0.8, zorder=8)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if tipo == 'maxima':
        cmap_full = plt.get_cmap('turbo')
        cmap_truncated = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', cmap_full(np.linspace(0.35, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_truncated = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine_f)
    vmax = np.nanmax(temp_masked_fine_f)
    level_count = 10
    levels = np.linspace(vmin, vmax, level_count)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine_f, levels=levels, cmap=cmap_truncated, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.088, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°F)', fontsize=11)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'San Juan': {'lat': 18.46, 'lon': -66.105}, 'Ponce': {'lat': 18.01, 'lon': -66.61}, 'Caguas': {'lat': 18.24, 'lon': -66.04}, 'Mayagüez': {'lat': 18.2, 'lon': -67.14}, 'Arecibo': {'lat': 18.47, 'lon': -66.72}, 'Fajardo': {'lat': 18.33, 'lon': -65.65}, 'Aguadilla': {'lat': 18.43, 'lon': -67.15}, 'Humacao': {'lat': 18.15, 'lon': -65.83}, 'Utuado': {'lat': 18.27, 'lon': -66.7}, 'Guayama': {'lat': 17.98, 'lon': -66.11}, 'Vieques': {'lat': 18.12, 'lon': -65.44}, 'Culebra': {'lat': 18.32, 'lon': -65.29}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine_f.ravel()
    for nombre, datos in major_cities.items():
        lat_ci = datos['lat']
        lon_ci = datos['lon']
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(lon_ci, lat_ci + 0.02, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
        ax.text(lon_ci, lat_ci - 0.02, nombre, fontsize=8, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.198, 0.2, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print('Logo not found')
    ax.text(0.5, 0.36, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold')
    output_file = f'output/maps/puerto_rico/puerto_rico_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'Map saved: {output_file}')
    plt.show()
    return (fig, ax)


def plot_central_america_temperature(ds_surface, tipo='maxima', admin_boundaries=None):
    """
    Plot maximum or minimum temperatures for Central America

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    countries = ['Guatemala', 'Belize', 'Honduras', 'El Salvador', 'Nicaragua', 'Costa Rica', 'Panama']
    gdf_ca = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -94
    lon_max_deg = -74
    lat_min = 7
    lat_max = 18.7
    lon_orig = ds_surface['t2m'].longitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    temp_corrected = ds_surface['t2m'].assign_coords(longitude=lon_corrected)
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    temp_vals = temp_area.values[0, :, :] if temp_area.values.ndim == 3 else temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    ca_union = gdf_ca.unary_union
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_ca = gdf_points_fine.within(ca_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_ca, temp_fine, np.nan)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = gdf_ca[gdf_ca['admin'] == country]
        borders = country_boundaries.unary_union.boundary
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        provinces = gdf_ca[(gdf_ca['admin'] == country) & (gdf_ca['type'] != 'Country')]
        provinces.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=4, transform=ccrs.PlateCarree())
    if tipo == 'maxima':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', plt.get_cmap('turbo')(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_trunc, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.001, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°C)', fontsize=11)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'Guatemala': {'lat': 14.63, 'lon': -90.53}, 'Quetzaltenango': {'lat': 14.83, 'lon': -91.52}, 'Cobán': {'lat': 15.47, 'lon': -90.37}, 'Flores': {'lat': 16.91, 'lon': -89.89}, 'Tegucigalpa': {'lat': 14.1, 'lon': -87.22}, 'San Pedro Sula': {'lat': 15.5, 'lon': -88.03}, 'La Ceiba': {'lat': 15.75, 'lon': -86.79}, 'Choluteca': {'lat': 13.3, 'lon': -87.2}, 'Tocoa': {'lat': 15.65, 'lon': -85.99}, 'San Salvador': {'lat': 13.69, 'lon': -89.2}, 'San Miguel': {'lat': 13.48, 'lon': -88.18}, 'Managua': {'lat': 12.13, 'lon': -86.25}, 'León': {'lat': 12.44, 'lon': -86.88}, 'Estelí': {'lat': 13.09, 'lon': -86.36}, 'Rivas': {'lat': 11.43, 'lon': -85.82}, 'Bilwi': {'lat': 14.03, 'lon': -83.39}, 'San José': {'lat': 9.93, 'lon': -84.08}, 'Liberia': {'lat': 10.63, 'lon': -85.44}, 'Ciudad de Panamá': {'lat': 9.01, 'lon': -79.52}, 'David': {'lat': 8.43, 'lon': -82.43}, 'Santiago de Veraguas': {'lat': 8.1, 'lon': -80.97}, 'Belmopán': {'lat': 17.25, 'lon': -88.77}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci, lon_ci = (datos['lat'], datos['lon'])
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(lon_ci, lat_ci + 0.08, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
        ax.text(lon_ci, lat_ci - 0.08, nombre, fontsize=8, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.198, 0.25, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except FileNotFoundError:
        print('⚠️ Logo not found at assets/logo.png')
    ax.text(0.8, 0.4, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold')
    output_file = f'output/maps/central_america/central_america_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_file}')
    plt.show()
    return (fig, ax)


def plot_colombia_venezuela_temperature(ds_surface, tipo='maxima', admin_boundaries=None):
    """
    Plot maximum or minimum temperatures para Colombia y Venezuela

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    countries = ['Colombia', 'Venezuela']
    gdf_cv = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -80
    lon_max_deg = -58
    lat_min = 0.5
    lat_max = 13
    lon_orig = ds_surface['t2m'].longitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    temp_corrected = ds_surface['t2m'].assign_coords(longitude=lon_corrected)
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    temp_vals = temp_area.values[0, :, :] if temp_area.values.ndim == 3 else temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    cv_union = gdf_cv.unary_union
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_cv = gdf_points_fine.within(cv_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_cv, temp_fine, np.nan)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = gdf_cv[gdf_cv['admin'] == country]
        borders = country_boundaries.unary_union.boundary
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        provinces = gdf_cv[(gdf_cv['admin'] == country) & (gdf_cv['type'] != 'Country')]
        provinces.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=4, transform=ccrs.PlateCarree())
    if tipo == 'maxima':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', plt.get_cmap('turbo')(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_trunc, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='2.5%', height='45%', loc='lower left', bbox_to_anchor=(0.02, 0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label('(°C)', fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'Bogotá': {'lat': 4.60971, 'lon': -74.08175}, 'Medellín': {'lat': 6.2442, 'lon': -75.5812}, 'Cali': {'lat': 3.4516, 'lon': -76.532}, 'Barranquilla': {'lat': 10.9685, 'lon': -74.7813}, 'Montería': {'lat': 8.7471, 'lon': -75.8894}, 'Pereira': {'lat': 4.8143, 'lon': -75.6946}, 'Cúcuta': {'lat': 7.89, 'lon': -72.4963}, 'Cartagena': {'lat': 10.395, 'lon': -75.4833}, 'Valledupar': {'lat': 10.4719, 'lon': -73.2527}, 'Maicao': {'lat': 11.3775, 'lon': -72.2383}, 'Bucaramanga': {'lat': 7.1238, 'lon': -73.1216}, 'Caracas': {'lat': 10.488, 'lon': -66.8792}, 'Maracaibo': {'lat': 10.6539, 'lon': -71.64597}, 'Valencia': {'lat': 10.162, 'lon': -68.0077}, 'Maturín': {'lat': 9.75, 'lon': -63.18}, 'Barcelona': {'lat': 10.1363, 'lon': -64.6862}, 'Ciudad Bolívar': {'lat': 8.0833, 'lon': -63.6}, 'Mérida': {'lat': 8.57, 'lon': -71.18}, 'Barquisimeto': {'lat': 10.0683, 'lon': -69.3452}, 'Coro': {'lat': 11.398, 'lon': -69.6794}, 'Cd Guayana': {'lat': 8.3525, 'lon': -62.643}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci, lon_ci = (datos['lat'], datos['lon'])
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(lon_ci, lat_ci + 0.08, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
        ax.text(lon_ci, lat_ci - 0.08, nombre, fontsize=7.6, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.12, 0.01, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except FileNotFoundError:
        print('⚠️ Logo not found at assets/logo.png')
    ax.text(0.9, 0.15, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold')
    output_file = f'output/maps/colombia_venezuela/colombia_venezuela_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_file}')
    plt.show()
    return (fig, ax)


def plot_united_states_temperature(ds_surface, tipo='maxima', admin_boundaries=None):
    """
    Plot maximum or minimum temperatures para Estados Unidos

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    countries = ['United States of America']
    gdf_usa = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -125
    lon_max_deg = -66
    lat_min = 24
    lat_max = 50
    lon_orig = ds_surface['t2m'].longitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    temp_corrected = ds_surface['t2m'].assign_coords(longitude=lon_corrected)
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    temp_vals = temp_area.values[0, :, :] if temp_area.values.ndim == 3 else temp_area.values
    temp_vals = temp_vals * 9 / 5 + 32
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    usa_union = gdf_usa.unary_union
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_usa = gdf_points_fine.within(usa_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_usa, temp_fine, np.nan)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = gdf_usa[gdf_usa['admin'] == country]
        borders = country_boundaries.unary_union.boundary
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        states = gdf_usa[(gdf_usa['admin'] == country) & (gdf_usa['type'] != 'Country')]
        states.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=4, transform=ccrs.PlateCarree())
    if tipo == 'maxima':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', plt.get_cmap('turbo')(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_trunc, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.05, 0.001, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°C)', fontsize=11)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'Los Angeles': {'lat': 34.05, 'lon': -118.24}, 'San Francisco': {'lat': 37.77, 'lon': -122.42}, 'Seattle': {'lat': 47.61, 'lon': -122.33}, 'Portland': {'lat': 45.52, 'lon': -122.68}, 'San Diego': {'lat': 32.72, 'lon': -117.16}, 'Las Vegas': {'lat': 36.17, 'lon': -115.14}, 'Reno': {'lat': 39.5305, 'lon': -119.813}, 'Phoenix': {'lat': 33.45, 'lon': -112.07}, 'Denver': {'lat': 39.74, 'lon': -104.99}, 'Albuquerque': {'lat': 35.08, 'lon': -106.65}, 'Salt Lake City': {'lat': 40.76, 'lon': -111.89}, 'Houston': {'lat': 29.76, 'lon': -95.37}, 'Dallas': {'lat': 32.78, 'lon': -96.8}, 'San Antonio': {'lat': 29.42, 'lon': -98.49}, 'McAllen': {'lat': 26.2066, 'lon': -98.2325}, 'Amarillo': {'lat': 35.205, 'lon': -101.8369}, 'Oklahoma City': {'lat': 35.4738, 'lon': -97.5205}, 'Miami': {'lat': 25.76, 'lon': -80.27}, 'Orlando': {'lat': 28.5308, 'lon': -81.3825}, 'Jacksonville': {'lat': 30.3316, 'lon': -81.6577}, 'Atlanta': {'lat': 33.75, 'lon': -84.39}, 'New Orleans': {'lat': 29.95, 'lon': -90.07}, 'Nashville': {'lat': 36.16, 'lon': -86.78}, 'Charlotte': {'lat': 35.23, 'lon': -80.84}, 'New York': {'lat': 40.71, 'lon': -74.01}, 'Boston': {'lat': 42.36, 'lon': -71.06}, 'Washington DC': {'lat': 38.91, 'lon': -77.04}, 'Chicago': {'lat': 41.88, 'lon': -87.63}, 'Detroit': {'lat': 42.33, 'lon': -83.1}, 'Minneapolis': {'lat': 44.98, 'lon': -93.27}, 'Kansas City': {'lat': 39.1, 'lon': -94.58}, 'Wichita': {'lat': 37.6916, 'lon': -97.3286}, 'Omaha': {'lat': 41.2591, 'lon': -95.9347}, 'Memphis': {'lat': 35.1402, 'lon': -90.0355}, 'Cincinati': {'lat': 39.1058, 'lon': -84.5141}, 'St. Louis': {'lat': 38.63, 'lon': -90.2}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci, lon_ci = (datos['lat'], datos['lon'])
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(lon_ci, lat_ci + 0.14, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
        ax.text(lon_ci, lat_ci - 0.12, nombre, fontsize=8, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.0, 0.12, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except FileNotFoundError:
        print('⚠️ Logo not found at assets/logo.png')
    ax.text(0.97, 0.1, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold')
    output_file = f'output/maps/united_states/united_states_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_file}')
    plt.show()
    return (fig, ax)


def plot_florida_temperature(ds_surface, tipo='maxima', admin_boundaries=None, florida_counties=None, show_logo=True):
    """
    Plot maximum or minimum temperatures para Florida en FAHRENHEIT

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    condados_florida : GeoDataFrame
        GeoDataFrame with Florida counties (optional)
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    gdf_florida = admin_boundaries[(admin_boundaries['admin'] == 'United States of America') & (admin_boundaries['name'] == 'Florida')].copy()
    lon_min_deg = -88.5
    lon_max_deg = -77.5
    lat_min = 24.3
    lat_max = 31.5
    temp_da = ds_surface['t2m']
    temp_corrected = temp_da.assign_coords(longitude=(temp_da.longitude + 180) % 360 - 180).sortby('longitude')
    temp_area = temp_corrected.sel(latitude=slice(lat_max + 1, lat_min - 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    print(f'Selected area:')
    print(f'  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f'  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty.')
    temp_vals = temp_area.values[0, :, :] if temp_area.values.ndim == 3 else temp_area.values
    temp_vals_fahrenheit = temp_vals * 9 / 5 + 32
    print(f'  Temperature: Min={temp_vals_fahrenheit.min():.1f}°F, Max={temp_vals_fahrenheit.max():.1f}°F, Mean={temp_vals_fahrenheit.mean():.1f}°F')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals_fahrenheit.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    if len(gdf_florida) > 0:
        florida_union = gdf_florida.union_all()
        points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
        gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
        mask_florida = gdf_points_fine.within(florida_union).values.reshape(lon2d_fine.shape)
        temp_masked_fine = np.where(mask_florida, temp_fine, np.nan)
    else:
        temp_masked_fine = temp_fine
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    if tipo == 'maxima':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', plt.get_cmap('turbo')(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_trunc, extend='both', transform=ccrs.PlateCarree(), alpha=0.8)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if len(gdf_florida) > 0:
        borders = gdf_florida.unary_union.boundary
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if florida_counties is not None:
        print(f'✅ Dibujando {len(florida_counties)} condados de Florida')
        florida_counties.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.3, alpha=0.8, zorder=8, transform=ccrs.PlateCarree())
    else:
        print('⚠️ No Florida counties were provided')
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.02, 0.05, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°F)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Miami': {'lat': 25.76, 'lon': -80.27}, 'Tampa': {'lat': 27.95, 'lon': -82.46}, 'Orlando': {'lat': 28.5308, 'lon': -81.3825}, 'Jacksonville': {'lat': 30.3316, 'lon': -81.6577}, 'Tallahassee': {'lat': 30.44, 'lon': -84.28}, 'West Palm Beach': {'lat': 26.71, 'lon': -80.05}, 'Naples': {'lat': 26.14, 'lon': -81.79}, 'Fort Myers': {'lat': 26.64, 'lon': -81.87}, 'Sarasota': {'lat': 27.34, 'lon': -82.53}, 'Pensacola': {'lat': 30.42, 'lon': -87.22}, 'Gainesville': {'lat': 29.65, 'lon': -82.32}, 'Daytona Beach': {'lat': 29.21, 'lon': -81.02}, 'Panama City': {'lat': 30.18, 'lon': -85.65}, 'Key West': {'lat': 24.55, 'lon': -81.77}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci, lon_ci = (datos['lat'], datos['lon'])
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        if not np.isnan(temp):
            ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
            ax.text(lon_ci, lat_ci + 0.07, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
            ax.text(lon_ci, lat_ci - 0.07, nombre, fontsize=8, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.07, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print('⚠️ Logo not found at assets/logo.png')
    ax.text(0.9, 0.01, 'Created by ElTiempoconLorenzo', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    output_dir = 'output/maps'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/florida/florida_map_{tipo}.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/florida/florida_map_{tipo}.png')
    plt.show()
    return (fig, ax)


def plot_texas_temperature(ds_surface, tipo='maxima', admin_boundaries=None, texas_counties=None, show_logo=True):
    """
    Plot maximum or minimum temperatures para Texas en FAHRENHEIT

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    condados_texas : GeoDataFrame
        GeoDataFrame with Texas counties (optional)
    mostrar_logo : bool
        Si se muestra el logo (default: True)
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    gdf_texas = admin_boundaries[(admin_boundaries['admin'] == 'United States of America') & (admin_boundaries['name'] == 'Texas')].copy()
    lon_min_deg = -108
    lon_max_deg = -92
    lat_min = 25.5
    lat_max = 37
    temp_da = ds_surface['t2m']
    temp_corrected = temp_da.assign_coords(longitude=(temp_da.longitude + 180) % 360 - 180).sortby('longitude')
    temp_area = temp_corrected.sel(latitude=slice(lat_max + 1, lat_min - 1), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    print(f'Selected area:')
    print(f'  LON: [{lon.min():.2f}, {lon.max():.2f}] - {len(lon)} points')
    print(f'  LAT: [{lat.min():.2f}, {lat.max():.2f}] - {len(lat)} points')
    if len(lon) == 0 or len(lat) == 0:
        raise ValueError(f'The selected area is empty.')
    temp_vals = temp_area.values[0, :, :] if temp_area.values.ndim == 3 else temp_area.values
    temp_vals_fahrenheit = temp_vals * 9 / 5 + 32
    print(f'  Temperature: Min={temp_vals_fahrenheit.min():.1f}°F, Max={temp_vals_fahrenheit.max():.1f}°F, Mean={temp_vals_fahrenheit.mean():.1f}°F')
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 500)
    lat_fine = np.linspace(lat.min(), lat.max(), 500)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals_fahrenheit.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    if len(gdf_texas) > 0:
        texas_union = gdf_texas.union_all()
        points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
        gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
        mask_texas = gdf_points_fine.within(texas_union).values.reshape(lon2d_fine.shape)
        temp_masked_fine = np.where(mask_texas, temp_fine, np.nan)
    else:
        temp_masked_fine = temp_fine
    fig = plt.figure(figsize=(18, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue', zorder=1)
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=2)
    if tipo == 'maxima':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', plt.get_cmap('turbo')(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_trunc, extend='both', transform=ccrs.PlateCarree(), alpha=0.8)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if len(gdf_texas) > 0:
        borders = gdf_texas.unary_union.boundary
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
        ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if texas_counties is not None:
        print(f'✅ Dibujando {len(texas_counties)} condados de Texas')
        texas_counties.boundary.plot(ax=ax, edgecolor='grey', linewidth=0.3, alpha=0.8, zorder=8, transform=ccrs.PlateCarree())
    else:
        print('⚠️ No Texas counties were provided')
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.02, 0.05, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°F)', fontsize=11, weight='bold')
    label_obj = cbar.ax.xaxis.get_label()
    label_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8, colors='black', width=1.5)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white'), PathEffects.Normal()])
    major_cities = {'Houston': {'lat': 29.76, 'lon': -95.37}, 'San Antonio': {'lat': 29.42, 'lon': -98.49}, 'Dallas': {'lat': 32.78, 'lon': -96.8}, 'Austin': {'lat': 30.27, 'lon': -97.74}, 'El Paso': {'lat': 31.76, 'lon': -106.49}, 'Corpus Christi': {'lat': 27.8, 'lon': -97.4}, 'Laredo': {'lat': 27.51, 'lon': -99.51}, 'Lubbock': {'lat': 33.58, 'lon': -101.86}, 'Amarillo': {'lat': 35.22, 'lon': -101.83}, 'Brownsville': {'lat': 25.9, 'lon': -97.5}, 'Odessa': {'lat': 31.85, 'lon': -102.37}, 'Beaumont': {'lat': 30.09, 'lon': -94.13}, 'Waco': {'lat': 31.55, 'lon': -97.15}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci, lon_ci = (datos['lat'], datos['lon'])
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        if not np.isnan(temp):
            ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
            ax.text(lon_ci, lat_ci + 0.07, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
            ax.text(lon_ci, lat_ci - 0.07, nombre, fontsize=9, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    if show_logo:
        try:
            logo_img = mpimg.imread(LOGO_PATH)
            axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.07, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
            axins_logo.imshow(logo_img)
            axins_logo.axis('off')
        except:
            print('⚠️ Logo not found at assets/logo.png')
    ax.text(0.9, 0.01, 'Created by ElTiempoconLorenzo', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', path_effects=[PathEffects.withStroke(linewidth=2.5, foreground='white'), PathEffects.Normal()])
    output_dir = 'output/maps'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/texas/texas_map_{tipo}.png', dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_dir}/texas/texas_map_{tipo}.png')
    plt.show()
    return (fig, ax)


def plot_lesser_antilles_temperature(ds_surface, tipo='maxima', admin_boundaries=None):
    """
    Plot maximum or minimum temperatures for the Lesser Antilles

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    countries = ['Antigua and Barbuda', 'Dominica', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Grenada', 'Barbados', 'Trinidad and Tobago', 'Saint Kitts and Nevis', 'Martinique', 'Guadeloupe', 'Montserrat', 'Anguilla', 'British Virgin Islands', 'United States Virgin Islands', 'Aruba', 'Curaçao', 'Sint Maarten', 'Saint Barthélemy', 'Bonaire', 'France']
    gdf_am = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -71
    lon_max_deg = -57
    lat_min = 9.5
    lat_max = 19
    lon_orig = ds_surface['t2m'].longitude.values
    lon_corrected = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    temp_corrected = ds_surface['t2m'].assign_coords(longitude=lon_corrected)
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    temp_vals = temp_area.values[0, :, :] if temp_area.values.ndim == 3 else temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    am_union = gdf_am.unary_union
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_am = gdf_points_fine.within(am_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_am, temp_fine, np.nan)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = gdf_am[gdf_am['admin'] == country]
        if not country_boundaries.empty:
            borders = country_boundaries.unary_union.boundary
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if tipo == 'maxima':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', plt.get_cmap('turbo')(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_trunc, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, 0.35, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('(°C)', fontsize=11)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'Islas Vírgenes Británicas': {'lat': 18.43, 'lon': -64.62}, 'Anguila': {'lat': 18.22, 'lon': -63.05}, 'San Cristóbal': {'lat': 17.3, 'lon': -62.72}, 'Antigua': {'lat': 17.12, 'lon': -61.85}, 'Guadalupe': {'lat': 16.24, 'lon': -61.53}, 'Dominica': {'lat': 15.3, 'lon': -61.39}, 'Martinica': {'lat': 14.61, 'lon': -61.08}, 'Santa Lucía': {'lat': 14.01, 'lon': -60.99}, 'San Vicente': {'lat': 13.1633, 'lon': -61.2233}, 'Barbados': {'lat': 13.1, 'lon': -59.61}, 'Granada': {'lat': 12.05, 'lon': -61.75}, 'Puerto España': {'lat': 10.6702, 'lon': -61.5038}, 'Tobago': {'lat': 11.18, 'lon': -60.74}, 'Aruba': {'lat': 12.52, 'lon': -70.03}, 'Curazao': {'lat': 12.11, 'lon': -68.93}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci, lon_ci = (datos['lat'], datos['lon'])
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
        ax.text(lon_ci, lat_ci + 0.08, f'{temp:.0f}°', fontsize=11, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
        ax.text(lon_ci, lat_ci - 0.08, nombre, fontsize=7.6, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.55, 0.56, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except FileNotFoundError:
        print('⚠️ Logo not found at assets/logo.png')
    ax.text(0.5, 0.7, 'Created by MeteOcean', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold')
    output_file = f'output/maps/lesser_antilles/lesser_antilles_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_file}')
    plt.show()
    return (fig, ax)


def plot_iberia_temperature(ds_surface, tipo='maxima', admin_boundaries=None):
    """
    Plot maximum or minimum temperatures for Spain and Portugal (Iberian Peninsula)

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    countries = ['Spain', 'Portugal']
    gdf_pi = admin_boundaries[admin_boundaries['admin'].isin(countries)].copy()
    lon_min_deg = -12
    lon_max_deg = 5
    lat_min = 35.4
    lat_max = 44.6
    temp_da = ds_surface['t2m']
    temp_corrected = temp_da.assign_coords(longitude=(temp_da.longitude + 180) % 360 - 180).sortby('longitude')
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    temp_vals = temp_area.values[0, :, :] if temp_area.values.ndim == 3 else temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    pi_union = gdf_pi.union_all()
    points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
    gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
    mask_pi = gdf_points_fine.within(pi_union).values.reshape(lon2d_fine.shape)
    temp_masked_fine = np.where(mask_pi, temp_fine, np.nan)
    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor='black', facecolor='steelblue')
    ax.add_feature(lakes_feature, linewidth=0.5, zorder=5)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    for country in countries:
        country_boundaries = gdf_pi[gdf_pi['admin'] == country]
        if not country_boundaries.empty:
            borders = country_boundaries.unary_union.boundary
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
            ax.add_geometries([borders], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    try:
        france_boundaries = admin_boundaries[admin_boundaries['admin'] == 'France']
        if not france_boundaries.empty and (not gdf_pi[gdf_pi['admin'] == 'Spain'].empty):
            spain_union = gdf_pi[gdf_pi['admin'] == 'Spain'].unary_union
            france_union = france_boundaries.unary_union
            spain_france_border = spain_union.boundary.intersection(france_union.boundary)
            ax.add_geometries([spain_france_border], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.8, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=8)
            ax.add_geometries([spain_france_border], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.5, path_effects=[PathEffects.Normal()], zorder=9)
            print('✅ Spain-France border drawn')
    except Exception as e:
        print(f'⚠️ Could not draw the Spain-France border: {e}')
    for country in countries:
        provinces = gdf_pi[(gdf_pi['admin'] == country) & (gdf_pi['type'] != 'Country')]
        if not provinces.empty:
            provinces.boundary.plot(ax=ax, edgecolor='black', linewidth=0.35, zorder=4, transform=ccrs.PlateCarree())
    if tipo == 'maxima':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', plt.get_cmap('turbo')(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_trunc, extend='both', transform=ccrs.PlateCarree())
    cax = inset_axes(ax, width='2.5%', height='45%', loc='lower left', bbox_to_anchor=(0.02, 0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label('(°C)', fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'Ceuta': {'lat': 35.8894, 'lon': -5.3213}, 'Melilla': {'lat': 35.2943, 'lon': -2.9526}, 'Albacete': {'lat': 38.9943, 'lon': -1.8585}, 'Alicante': {'lat': 38.3452, 'lon': -0.481}, 'Almería': {'lat': 36.834, 'lon': -2.4637}, 'Ávila': {'lat': 40.6566, 'lon': -4.681}, 'Badajoz': {'lat': 38.8794, 'lon': -6.9707}, 'Barcelona': {'lat': 41.3879, 'lon': 2.1699}, 'Bilbao': {'lat': 43.263, 'lon': -2.935}, 'Burgos': {'lat': 42.3439, 'lon': -3.6969}, 'Cáceres': {'lat': 39.4767, 'lon': -6.3722}, 'Cádiz': {'lat': 36.5333, 'lon': -6.1852}, 'Castellón': {'lat': 39.9864, 'lon': -0.0513}, 'Ciudad Real': {'lat': 38.986, 'lon': -3.927}, 'Córdoba': {'lat': 37.8882, 'lon': -4.7794}, 'A Coruña': {'lat': 43.3623, 'lon': -8.4115}, 'Cuenca': {'lat': 40.0704, 'lon': -2.1374}, 'Girona': {'lat': 41.9794, 'lon': 2.8214}, 'Granada': {'lat': 37.1773, 'lon': -3.5986}, 'Huelva': {'lat': 37.2614, 'lon': -6.9447}, 'Huesca': {'lat': 42.1401, 'lon': -0.4089}, 'Jaén': {'lat': 37.7796, 'lon': -3.7849}, 'León': {'lat': 42.5987, 'lon': -5.5671}, 'Lleida': {'lat': 41.6167, 'lon': 0.6222}, 'Logroño': {'lat': 42.465, 'lon': -2.445}, 'Lugo': {'lat': 43.0097, 'lon': -7.556}, 'Madrid': {'lat': 40.4168, 'lon': -3.7038}, 'Málaga': {'lat': 36.7283, 'lon': -4.4391}, 'Murcia': {'lat': 37.9922, 'lon': -1.1307}, 'Orense': {'lat': 42.3358, 'lon': -7.8639}, 'Oviedo': {'lat': 43.3619, 'lon': -5.8494}, 'Pontevedra': {'lat': 42.431, 'lon': -8.6444}, 'Salamanca': {'lat': 40.9701, 'lon': -5.6635}, 'Santander': {'lat': 43.4623, 'lon': -3.8099}, 'Segovia': {'lat': 40.9429, 'lon': -4.1088}, 'Sevilla': {'lat': 37.3891, 'lon': -5.9845}, 'Soria': {'lat': 41.7667, 'lon': -2.4797}, 'Tarragona': {'lat': 41.1189, 'lon': 1.2445}, 'Teruel': {'lat': 40.344, 'lon': -1.1064}, 'Toledo': {'lat': 39.8628, 'lon': -4.0273}, 'Valencia': {'lat': 39.4699, 'lon': -0.3763}, 'Valladolid': {'lat': 41.6523, 'lon': -4.7245}, 'Zamora': {'lat': 41.5033, 'lon': -5.744}, 'Zaragoza': {'lat': 41.6488, 'lon': -0.8891}, 'Pamplona': {'lat': 42.8125, 'lon': -1.6458}, 'Palma': {'lat': 39.5786, 'lon': 2.655}, 'Ibiza': {'lat': 38.9105, 'lon': 1.4247}, 'Menorca': {'lat': 40.0002, 'lon': 3.84}, 'Lisboa': {'lat': 38.7223, 'lon': -9.1393}, 'Porto': {'lat': 41.1579, 'lon': -8.6291}, 'Coimbra': {'lat': 40.2033, 'lon': -8.4103}, 'Faro': {'lat': 37.0194, 'lon': -7.9304}, 'Évora': {'lat': 38.5714, 'lon': -7.9093}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci, lon_ci = (datos['lat'], datos['lon'])
        if not (lat_min <= lat_ci <= lat_max and lon_min_deg <= lon_ci <= lon_max_deg):
            continue
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        if not np.isnan(temp):
            ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
            ax.text(lon_ci, lat_ci + 0.07, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
            ax.text(lon_ci, lat_ci - 0.07, nombre, fontsize=9, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread('assets/logo.png')
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.12, 0.01, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except FileNotFoundError:
        print('⚠️ Logo not found at assets/logo.png')
    ax.text(0.9, 0.17, 'Created by MeteoSpain', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold')
    output_file = f'output/maps/iberia/iberia_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_file}')
    plt.show()
    return (fig, ax)


def plot_canary_islands_temperature(ds_surface, tipo='maxima', admin_boundaries=None):
    """
    Plot maximum or minimum temperatures for the Canary Islands

    Parameters:
    -----------
    ds_surface : xarray.Dataset
        Dataset with the variable 't2m' (temperature en °C)
    tipo : str
        'maxima' o 'minima'
    admin_boundaries : GeoDataFrame
        GeoDataFrame with all states/provinces
    """
    if tipo.lower() not in ['maxima', 'minima']:
        raise ValueError("The parameter 'tipo' must be 'maxima' o 'minima'")
    tipo = tipo.lower()
    if admin_boundaries is None:
        raise ValueError("You must provide 'admin_boundaries'")
    canary_islands = admin_boundaries[admin_boundaries['admin'] == 'Spain'].copy()
    lon_min_deg = -19.5
    lon_max_deg = -12
    lat_min = 26.5
    lat_max = 30.5
    temp_da = ds_surface['t2m']
    temp_corrected = temp_da.assign_coords(longitude=(temp_da.longitude + 180) % 360 - 180).sortby('longitude')
    temp_area = temp_corrected.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min_deg, lon_max_deg))
    lat = temp_area.latitude.values
    lon = temp_area.longitude.values
    temp_vals = temp_area.values[0, :, :] if temp_area.values.ndim == 3 else temp_area.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_fine = np.linspace(lon.min(), lon.max(), 400)
    lat_fine = np.linspace(lat.min(), lat.max(), 400)
    lon2d_fine, lat2d_fine = np.meshgrid(lon_fine, lat_fine)
    points_orig = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    values_orig = temp_vals.ravel()
    temp_fine = griddata(points_orig, values_orig, (lon2d_fine, lat2d_fine), method='linear')
    canary_islands_filtered = canary_islands[(canary_islands.geometry.centroid.x >= lon_min_deg) & (canary_islands.geometry.centroid.x <= lon_max_deg) & (canary_islands.geometry.centroid.y >= lat_min) & (canary_islands.geometry.centroid.y <= lat_max)]
    if not canary_islands_filtered.empty:
        canary_islands_union = canary_islands_filtered.union_all()
        points_fine = [Point(xy) for xy in zip(lon2d_fine.ravel(), lat2d_fine.ravel())]
        gdf_points_fine = gpd.GeoSeries(points_fine, crs='EPSG:4326')
        mask_canary_islands = gdf_points_fine.within(canary_islands_union).values.reshape(lon2d_fine.shape)
        temp_masked_fine = np.where(mask_canary_islands, temp_fine, np.nan)
    else:
        temp_masked_fine = temp_fine
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_deg, lon_max_deg, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='steelblue')
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if not canary_islands_filtered.empty:
        for idx, row in canary_islands_filtered.iterrows():
            geom = row.geometry
            ax.add_geometries([geom.boundary], crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=6)
            ax.add_geometries([geom.boundary], crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=7)
    if tipo == 'maxima':
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('turbo_trunc', plt.get_cmap('turbo')(np.linspace(0.2, 1.0, 256)))
    else:
        colors_minima = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        cmap_trunc = mcolors.LinearSegmentedColormap.from_list('temp_minima', colors_minima)
    vmin = np.nanmin(temp_masked_fine)
    vmax = np.nanmax(temp_masked_fine)
    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(lon2d_fine, lat2d_fine, temp_masked_fine, levels=levels, cmap=cmap_trunc, extend='both', transform=ccrs.PlateCarree(), alpha=0.8)
    cax = inset_axes(ax, width='42%', height='5%', loc='lower left', bbox_to_anchor=(-0.015, -0.088, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=7)
    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_label('Temperature (°C)', fontsize=11, weight='bold')
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    cbar.ax.tick_params(labelsize=8)
    major_cities = {'Santa Cruz de Tenerife': {'lat': 28.46, 'lon': -16.25}, 'Las Palmas': {'lat': 28.12, 'lon': -15.43}, 'Arrecife': {'lat': 28.9688, 'lon': -13.5569}, 'Puerto del Rosario': {'lat': 28.5, 'lon': -13.86}, 'Santa Cruz de La Palma': {'lat': 28.6858, 'lon': -17.7663}, 'San Sebastián': {'lat': 28.09, 'lon': -17.11}, 'Valverde': {'lat': 27.81, 'lon': -17.92}}
    flat_points = np.column_stack((lat2d_fine.ravel(), lon2d_fine.ravel()))
    tree = KDTree(flat_points)
    temp_values = temp_masked_fine.ravel()
    for nombre, datos in major_cities.items():
        lat_ci, lon_ci = (datos['lat'], datos['lon'])
        _, idx = tree.query([lat_ci, lon_ci])
        temp = temp_values[idx]
        if not np.isnan(temp):
            ax.plot(lon_ci, lat_ci, 'o', color='red', markersize=4, markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=12)
            ax.text(lon_ci, lat_ci + 0.04, f'{temp:.0f}°', fontsize=11.5, color='white', weight='bold', ha='center', va='bottom', transform=ccrs.PlateCarree(), zorder=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a5490', edgecolor='white', linewidth=1.3, alpha=1), path_effects=[PathEffects.Normal()])
            ax.text(lon_ci, lat_ci - 0.04, nombre, fontsize=9, color='#2C3E50', weight='bold', ha='center', va='top', zorder=15, transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.3), PathEffects.Normal()])
    try:
        logo_img = mpimg.imread('assets/logo.png')
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.0, 0.03, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print('Logo not found')
    ax.text(1, 0.2, 'Created by MeteoSpain', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold')
    output_file = f'output/maps/canary_islands/canary_islands_map_{tipo}.png'
    plt.savefig(output_file, dpi=800, bbox_inches='tight')
    print(f'✅ Map saved: {output_file}')
    plt.show()
    return (fig, ax)
