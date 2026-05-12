"""National Hurricane Center Eastern Pacific wind-field map and GIF tools."""

from __future__ import annotations

import os
import shutil
import traceback

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import imageio
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
from PIL import Image
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from .config import LOGO_PATH, NHC_OUTPUT_DIR

SMOOTHING_METHOD = None


GENERATE_GIF = False


FIXED_POSITION_INDEX = 0


FIGURE_SIZE = (14, 7)


DPI = 300


BBOX_INCHES = 'tight'


PAD_INCHES = 0.05


STORM_LAYERS = {'EP1': {'points': 136, 'rings': 146}, 'EP2': {'points': 162, 'rings': 172}, 'EP3': {'points': 188, 'rings': 198}, 'EP4': {'points': 214, 'rings': 224}, 'EP5': {'points': 240, 'rings': 250}, 'CP1': {'points': 266, 'rings': 276}, 'CP2': {'points': 292, 'rings': 302}, 'CP3': {'points': 318, 'rings': 328}, 'CP4': {'points': 344, 'rings': 354}, 'CP5': {'points': 370, 'rings': 380}}


def check_active_storms():
    """Check which Eastern/Central Pacific storm layers currently have usable data."""
    print('🔍 Checking active storms across all Pacific layers...')
    active_storms = {}
    for storm_name, layers in STORM_LAYERS.items():
        points_layer = layers['points']
        rings_layer = layers['rings']
        print(f'   🌀 Checking {storm_name} (layers {points_layer} and {rings_layer})...')
        try:
            points_data = get_forecast_points_by_layer(points_layer)
            rings_data = get_wind_rings_by_layer(rings_layer)
            if len(points_data) > 0 and len(rings_data) > 0:
                storm_info = {'name': points_data.iloc[0].get('stormname', 'Unknown'), 'id': points_data.iloc[0].get('stormnum', 'N/A'), 'points_count': len(points_data), 'rings_count': len(rings_data), 'layers': layers, 'data': {'points': points_data, 'rings': rings_data}}
                active_storms[storm_name] = storm_info
                print(f"   ✅ {storm_name}: {storm_info['name']} - {storm_info['points_count']} points, {storm_info['rings_count']} wind rings")
            else:
                print(f'   ❌ {storm_name}: no active data')
        except Exception as e:
            print(f'   ⚠️  {storm_name}: check failed - {e}')
    return active_storms


def get_forecast_points_by_layer(layer_id):
    """Fetch forecast points from a specific NHC map layer."""
    url = f'https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/NHC_tropical_weather/MapServer/{layer_id}/query'
    params = {'where': '1=1', 'outFields': '*', 'f': 'geojson', 'returnGeometry': 'true'}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'features' not in data or len(data['features']) == 0:
            return gpd.GeoDataFrame()
        features = []
        for feature in data['features']:
            props = feature['properties']
            geom = feature['geometry']
            if geom and 'coordinates' in geom:
                props['geometry'] = Point(geom['coordinates'])
                features.append(props)
        if features:
            return gpd.GeoDataFrame(features, crs='EPSG:4326')
        else:
            return gpd.GeoDataFrame()
    except Exception as e:
        print(f'Error obteniendo datos de capa {layer_id}: {e}')
        return gpd.GeoDataFrame()


def get_wind_rings_by_layer(layer_id):
    """Fetch wind rings from a specific NHC map layer."""
    url = f'https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/NHC_tropical_weather/MapServer/{layer_id}/query'
    params = {'where': '1=1', 'outFields': '*', 'f': 'geojson', 'returnGeometry': 'true'}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'features' not in data or len(data['features']) == 0:
            return gpd.GeoDataFrame()
        features = []
        for feature in data['features']:
            props = feature['properties']
            geom = feature['geometry']
            if geom and geom['type'] == 'Polygon' and ('coordinates' in geom):
                poly = Polygon(geom['coordinates'][0])
                props['geometry'] = poly
                features.append(props)
        if features:
            return gpd.GeoDataFrame(features, crs='EPSG:4326')
        else:
            return gpd.GeoDataFrame()
    except Exception as e:
        print(f'Error fetching wind rings from layer {layer_id}: {e}')
        return gpd.GeoDataFrame()


def display_storm_menu(active_storms):
    """Muestra un menú para select la storm a procesar"""
    if not active_storms:
        print('❌ No records found active storms.')
        return None
    print('\n' + '=' * 60)
    print('🌀 ACTIVE STORMS AVAILABLE (PACIFIC):')
    print('=' * 60)
    storm_list = list(active_storms.keys())
    for i, storm_key in enumerate(storm_list, 1):
        storm = active_storms[storm_key]
        print(f"{i}. {storm_key}: {storm['name']} (ID: {storm['id']})")
        print(f"   📊 {storm['points_count']} forecast points, {storm['rings_count']} wind rings")
        print(f"   📍 Layers: points={storm['layers']['points']}, rings={storm['layers']['rings']}")
        print()
    print('0. ❌ Exit')
    print('=' * 60)
    while True:
        try:
            choice = input(f'🎯 Select a storm (0-{len(storm_list)}): ').strip()
            choice_num = int(choice)
            if choice_num == 0:
                print('👋 Exiting...')
                return None
            elif 1 <= choice_num <= len(storm_list):
                selected_storm_key = storm_list[choice_num - 1]
                selected_storm = active_storms[selected_storm_key]
                print(f"\n✅ Selected: {selected_storm_key} - {selected_storm['name']}")
                return (selected_storm_key, selected_storm)
            else:
                print(f'❌ Please select a number between 0 and {len(storm_list)}')
        except ValueError:
            print('❌ Please enter a valid number')
        except KeyboardInterrupt:
            print('\n👋 Operation cancelled by the user')
            return None


def ask_generation_mode():
    """Ask whether to generate a static image or an animated GIF."""
    print('\n' + '=' * 60)
    print('📸 GENERATION MODE:')
    print('=' * 60)
    print('1. 📷 Static image (specific position)')
    print('2. 🎞️  Animated GIF (full evolution)')
    print('0. ❌ Exit')
    print('=' * 60)
    while True:
        try:
            choice = input('🎯 Select mode (0-2): ').strip()
            if choice == '0':
                return (None, None)
            elif choice == '1':
                while True:
                    try:
                        pos = input('📍 Specific position (0 = first): ').strip()
                        pos_num = int(pos)
                        if pos_num >= 0:
                            return (False, pos_num)
                        else:
                            print('❌ Position must be 0 or greater')
                    except ValueError:
                        print('❌ Please enter a valid number')
            elif choice == '2':
                return (True, 0)
            else:
                print('❌ Please select 1, 2, or 0')
        except KeyboardInterrupt:
            print('\n👋 Operation cancelled by the user')
            return (None, None)


def get_forecast_points():
    """Compatibility placeholder for the original notebook workflow."""
    pass


def get_max_wind_rings():
    """Compatibility placeholder for the original notebook workflow."""
    pass


def smooth_polygon(polygon, method='simple'):
    coords = np.array(polygon.exterior.coords)
    x, y = (coords[:, 0], coords[:, 1])
    if method == 'simple':
        return polygon.simplify(0.05, preserve_topology=True)
    elif method == 'buffer':
        return polygon.buffer(-0.1).buffer(0.1)
    elif method == 'spline':
        tck, _ = splprep([x, y], s=10, per=True)
        unew = np.linspace(0, 1, len(x) * 100)
        out = splev(unew, tck)
        smooth_coords = list(zip(out[0], out[1]))
        return Polygon(smooth_coords)
    elif method == 'gaussian':
        x_smooth = gaussian_filter1d(x, sigma=15)
        y_smooth = gaussian_filter1d(y, sigma=20)
        smooth_coords = list(zip(x_smooth, y_smooth))
        return Polygon(smooth_coords)
    else:
        return polygon


def calculate_gif_map_extent(gdf_points):
    """Calculate a balanced map extent for GIF output across the full evolution."""
    all_x = [point.x for point in gdf_points.geometry]
    all_y = [point.y for point in gdf_points.geometry]
    min_x, max_x = (min(all_x), max(all_x))
    min_y, max_y = (min(all_y), max(all_y))
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    traj_width = max_x - min_x
    traj_height = max_y - min_y
    if traj_width < 5.0:
        margin_x = max(6.0, traj_width * 0.8)
        margin_y = max(3.5, traj_height * 1)
        print('🔍 GIF - Trayectoria LOCAL detectada')
    elif traj_width < 15.0:
        margin_x = max(7.0, traj_width * 0.6)
        margin_y = max(4, traj_height * 0.8)
        print('🌊 GIF - Trayectoria REGIONAL detectada')
    else:
        margin_x = max(8.0, traj_width * 0.3)
        margin_y = max(4.5, traj_height * 0.4)
        print('🌐 GIF - Trayectoria TRANS-PACÍFICA detectada')
    west = center_x - (traj_width / 2 + margin_x)
    east = center_x + (traj_width / 2 + margin_x)
    south = center_y - (traj_height / 2 + margin_y)
    north = center_y + (traj_height / 2 + margin_y)
    west = max(west, -180.0)
    east = min(east, -80.0)
    south = max(south, 5.0)
    north = min(north, 50.0)
    final_width = east - west
    final_height = north - south
    target_ratio = 2.0
    current_ratio = final_width / final_height
    print(f'📊 Ratio actual: {current_ratio:.2f}, objetivo: {target_ratio:.2f}')
    if current_ratio > target_ratio * 1.2:
        target_width = final_height * target_ratio
        excess_width = final_width - target_width
        west += excess_width / 2
        east -= excess_width / 2
        print(f'⚖️  Reduciendo ancho: {final_width:.2f}° → {target_width:.2f}°')
    elif current_ratio < target_ratio * 0.8:
        target_height = final_width / target_ratio
        excess_height = final_height - target_height
        south += excess_height / 2
        north -= excess_height / 2
        print(f'⚖️  Reduciendo altura: {final_height:.2f}° → {target_height:.2f}°')
    final_width = east - west
    final_height = north - south
    if final_width < 12.0:
        center_x_adj = (west + east) / 2
        west = center_x_adj - 6.0
        east = center_x_adj + 6.0
    if final_height < 6.0:
        center_y_adj = (south + north) / 2
        south = center_y_adj - 3.0
        north = center_y_adj + 3.0
    print(f'📐 Extent GIF final: [{west:.2f}, {east:.2f}, {south:.2f}, {north:.2f}]')
    print(f'📊 Dimensiones GIF: {east - west:.2f}° x {north - south:.2f}° (ratio: {(east - west) / (north - south):.2f})')
    return [west, east, south, north]


def calculate_static_map_extent(gdf_points, posicion_index):
    """Calculate a centered map extent for a specific static-image frame."""
    target_point = gdf_points.iloc[posicion_index]
    center_x = target_point.geometry.x
    center_y = target_point.geometry.y
    margin_x = 13
    margin_y = 7
    west = center_x - margin_x
    east = center_x + margin_x
    south = center_y - margin_y
    north = center_y + margin_y
    west = max(west, -180.0)
    east = min(east, -80.0)
    south = max(south, 5.0)
    north = min(north, 50.0)
    print(f'📐 Extent ESTÁTICA centrada: [{west:.2f}, {east:.2f}, {south:.2f}, {north:.2f}]')
    print(f'📊 Dimensiones ESTÁTICA: {east - west:.2f}° x {north - south:.2f}°')
    return [west, east, south, north]


def create_fixed_axes():
    """Crea ejes con configuración completamente fija y SIN BORDES GRANDES"""
    fig = plt.figure(figsize=FIGURE_SIZE, facecolor='white')
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98], projection=ccrs.PlateCarree())
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    return (fig, ax)


def setup_improved_map(ax):
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='steelblue', edgecolor='none'), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='lightgray', edgecolor='none'), zorder=1)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m', facecolor='#2C3E50', edgecolor='gray'), linewidth=0.8, zorder=2)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[path_effects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=3)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[path_effects.Normal()], zorder=4)
    borders_feature = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m')
    borders_geoms = list(borders_feature.geometries())
    ax.add_geometries(borders_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.0, alpha=0.3, path_effects=[path_effects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=5)
    ax.add_geometries(borders_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.0, path_effects=[path_effects.Normal()], zorder=6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.7, color='black', alpha=0.2, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 10))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}


def draw_wind_ring(ax, poly, color, zorder_base=15):
    """Dibuja un anillo de wind conservando color + estilo 3D"""
    fill_patch = mpatches.Polygon(list(poly.exterior.coords), facecolor=color, edgecolor='none', alpha=0.5, transform=ccrs.PlateCarree(), zorder=zorder_base)
    ax.add_patch(fill_patch)
    shadow_patch = mpatches.Polygon(list(poly.exterior.coords), facecolor='none', edgecolor='black', linewidth=2, alpha=0.35, transform=ccrs.PlateCarree(), zorder=zorder_base + 2, path_effects=[PathEffects.SimpleLineShadow(offset=(1, -1), alpha=0.5)])
    ax.add_patch(shadow_patch)
    border_patch = mpatches.Polygon(list(poly.exterior.coords), facecolor='none', edgecolor=color, linewidth=1.5, alpha=0.7, transform=ccrs.PlateCarree(), zorder=zorder_base + 3)
    ax.add_patch(border_patch)
    return [fill_patch, shadow_patch, border_patch]


def extract_date_time(point):
    """Extract and format the forecast point date and time."""
    try:
        validtime = point.get('validtime', '')
        advdate = point.get('advdate', '')
        if not validtime or not advdate:
            return 'Date unavailable'
        day_str, time_str = validtime.split('/')
        forecast_day = int(day_str)
        forecast_hour = int(time_str[:2])
        forecast_minute = int(time_str[2:]) if len(time_str) > 2 else 0
        advdate_parts = advdate.split()
        month_abbr = advdate_parts[4]
        base_day = int(advdate_parts[5])
        year = int(advdate_parts[6])
        months = {'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April', 'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Aug': 'August', 'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        actual_day = base_day + (forecast_day - 20)
        month_name = months.get(month_abbr, month_abbr)
        day_offset = actual_day - 20
        base_weekday = 2
        weekday_index = (base_weekday + day_offset) % 7
        day_name = days[weekday_index]
        if forecast_minute == 0:
            time_formatted = f'{forecast_hour:02d}:00'
        else:
            time_formatted = f'{forecast_hour:02d}:{forecast_minute:02d}'
        date_formatted = f'{day_name} {month_name} {actual_day}, {year}'
        return f'{date_formatted} - {time_formatted} UTC'
    except Exception as e:
        print(f'Error processing date: {e}')
        return 'Date no disponible'


def add_date_to_map(ax, date_text):
    """Añade la fecha al mapa con position fija"""
    ax.text(0.02, 0.02, date_text, transform=ax.transAxes, fontsize=10, ha='left', va='bottom', color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='black', linewidth=1), zorder=30, path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])


def add_wind_legend(ax):
    """Add a fixed-position wind-field legend."""
    legend_elements = []
    wind_data = [(64, '#cc0000', 'Huracán (+118 km/h / +74 mph)'), (50, '#FF8C00', 'T.T. Fuerte (93-117 km/h / 57-73 mph)'), (34, '#009933', 'T.T. Débil (63-92 km/h / 39-56 mph)')]
    for radii, color, label in wind_data:
        legend_elements.append(patches.Rectangle((0, 0), 1, 1, facecolor=color, alpha=1, edgecolor='black', linewidth=1, label=label))
    legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True, fontsize=8, title='Campo de Wind', title_fontsize=9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    return legend


def categorize_wind_speed(radii_value):
    if radii_value == 64:
        return ('Huracán', '#cc0000')
    elif radii_value == 50:
        return ('Storm Tropical Fuerte', '#FF8C00')
    elif radii_value == 34:
        return ('Storm Tropical Débil', '#009933')
    else:
        return ('Desconecido', 'gray')


def add_logo_and_credit(ax):
    """Añade logo y crédito al mapa con position DINÁMICA que se ajusta al contenido"""
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        axins_logo = inset_axes(ax, width='9.5%', height='9.5%', loc='lower right', bbox_to_anchor=(-0.84, 0.7, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
    except:
        print('Logo not found - continuing without logo')
    ax.text(0.98, 0.02, 'Created by ElTiempoconLorenzo', transform=ax.transAxes, fontsize=6.7, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', zorder=35, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', edgecolor='gray', linewidth=0.5), path_effects=[path_effects.withStroke(linewidth=2, foreground='white'), path_effects.Normal()])


def verify_frame_consistency(frames_dir):
    """Verify that all frames have identical dimensions."""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frame_files:
        print('❌ No records found frames para verificar')
        return False
    print(f'🔍 Checking consistency for {len(frame_files)} frames...')
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_img = Image.open(first_frame_path)
    reference_size = first_img.size
    reference_mode = first_img.mode
    print(f'📏 Tamaño de referencia: {reference_size}')
    print(f'🎨 Modo de color de referencia: {reference_mode}')
    inconsistent_frames = []
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        img = Image.open(frame_path)
        if img.size != reference_size:
            inconsistent_frames.append((frame_file, img.size, reference_size))
        if img.mode != reference_mode:
            print(f'⚠️  Modo de color inconsistente en {frame_file}: {img.mode} vs {reference_mode}')
        img.close()
        if i % 5 == 0:
            print(f'   ✓ Verificados {i + 1}/{len(frame_files)} frames')
    if inconsistent_frames:
        print(f'❌ Founds {len(inconsistent_frames)} frames inconsistentes:')
        for frame_file, actual_size, expected_size in inconsistent_frames:
            print(f'   - {frame_file}: {actual_size} (esperado: {expected_size})')
        return False
    else:
        print('✅ All frames are consistent!')
        return True


def create_gif_with_pillow(frames, output_path, duration=1500):
    """Crea GIF usando Pillow - LA MEJOR OPCIÓN para GIFs estables"""
    print(f'🎞️ Creating GIF OPTIMIZADO con Pillow (duración: {duration}ms por frame)...')
    try:
        pil_frames = []
        reference_size = None
        for i, frame in enumerate(frames):
            if isinstance(frame, np.ndarray):
                pil_img = Image.fromarray(frame)
                if reference_size is None:
                    reference_size = pil_img.size
                    print(f'📏 Tamaño de referencia establecido: {reference_size}')
                elif pil_img.size != reference_size:
                    print(f'⚠️  Redimensionando frame {i + 1}: {pil_img.size} → {reference_size}')
                    pil_img = pil_img.resize(reference_size, Image.Resampling.LANCZOS)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                pil_frames.append(pil_img)
                if i % 5 == 0:
                    print(f'   📄 Frame {i + 1}/{len(frames)}: {pil_img.size}, {pil_img.mode}')
        pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0, optimize=True, disposal=2, transparency=None, dither=Image.Dither.NONE)
        print(f'✅ Optimized GIF created with Pillow: {output_path}')
        print(f'📊 Total frames: {len(pil_frames)}')
        print(f'⏱️  Duración total: {len(pil_frames) * duration / 1000:.1f} segundos')
    except Exception as e:
        print(f'❌ Error creando GIF con Pillow: {e}')


def create_single_frame(gdf_points, gdf_rings, storm_id, frame_index, map_extent, frames_dir, storm_name=''):
    """Crea un frame individual con configuración completamente estable"""
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = None
    plt.rcParams['savefig.pad_inches'] = PAD_INCHES
    fig, ax = create_fixed_axes()
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    setup_improved_map(ax)
    add_logo_and_credit(ax)
    wind_legend = add_wind_legend(ax)
    current_point = gdf_points.iloc[frame_index]
    date_text = extract_date_time(current_point)
    add_date_to_map(ax, date_text)
    trajectory_points = gdf_points.iloc[:frame_index + 1]
    if len(trajectory_points) > 1:
        traj_x, traj_y = zip(*trajectory_points.geometry.apply(lambda p: (p.x, p.y)))
        ax.plot(traj_x, traj_y, color='black', linewidth=2, linestyle='--', label='Trayectoria', zorder=12, transform=ccrs.PlateCarree())
    ax.scatter(current_point.geometry.x, current_point.geometry.y, color='black', s=80, zorder=20, marker='X', edgecolors='black', linewidths=1, transform=ccrs.PlateCarree())
    filtered_rings = gdf_rings[(gdf_rings['stormnum'] == storm_id) & (gdf_rings['advnum'] == current_point['advisnum']) & (gdf_rings['tau'] == current_point['tau'])]
    for _, ring_data in filtered_rings.iterrows():
        category, color = categorize_wind_speed(ring_data['radii'])
        poly = smooth_polygon(ring_data.geometry, method=SMOOTHING_METHOD)
        draw_wind_ring(ax, poly, color)
    safe_storm_name = storm_name.replace(' ', '_').replace('/', '_')
    frame_filename = f'{safe_storm_name}_frame_{frame_index:04d}.png'
    frame_path = os.path.join(frames_dir, frame_filename)
    plt.savefig(frame_path, dpi=DPI, facecolor='white', edgecolor='none', bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES, transparent=False, orientation='landscape')
    plt.close(fig)
    plt.clf()
    print(f'✅ Frame {frame_index + 1} guardado: {frame_filename}')
    return frame_path


def resize_and_standardize_frames(frames_dir, target_size=(1400, 700)):
    """Resize and standardize all frames for GIF consistency."""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    print(f'🔧 Estandarizando {len(frame_files)} frames a {target_size}...')
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        img = Image.open(frame_path)
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_resized.save(frame_path, 'PNG', optimize=True, compress_level=6)
    print('✅ All frames standardized')


def create_stable_gif(selected_storm):
    """Crea un GIF estable con frames completamente consistentes"""
    storm_name = selected_storm['name']
    storm_data = selected_storm['data']
    gdf_points = storm_data['points']
    gdf_rings = storm_data['rings']
    print(f'🌀 GENERADOR DE GIF ESTABLE - {storm_name}')
    print('=' * 70)
    storm_id = gdf_points.iloc[0]['stormnum']
    gdf_points = gdf_points[gdf_points['stormnum'] == storm_id]
    gdf_points = gdf_points.sort_values(by='tau').reset_index(drop=True)
    print(f'🎯 Storm ID: {storm_id}')
    print(f'📊 Total forecast points: {len(gdf_points)}')
    map_extent = calculate_gif_map_extent(gdf_points)
    safe_storm_name = storm_name.replace(' ', '_').replace('/', '_')
    drive_frames_dir = f'output/national_hurricane_center/gifs/hurricane_frames_{safe_storm_name}'
    if os.path.exists(drive_frames_dir):
        shutil.rmtree(drive_frames_dir)
    os.makedirs(drive_frames_dir, exist_ok=True)
    print(f'📁 Frame directory: {drive_frames_dir}')
    print('\n🎬 Generando frames individuales...')
    frame_paths = []
    for frame_index in range(len(gdf_points)):
        frame_path = create_single_frame(gdf_points, gdf_rings, storm_id, frame_index, map_extent, drive_frames_dir, storm_name)
        frame_paths.append(frame_path)
    print('\n🔧 Estandarizando dimensiones de frames...')
    resize_and_standardize_frames(drive_frames_dir, target_size=(1400, 700))
    print('\n🔍 Checking consistencia de frames...')
    is_consistent = verify_frame_consistency(drive_frames_dir)
    if not is_consistent:
        print('⚠️  Detectadas inconsistencias - pero continuamos...')
    print('\n🎞️ Creating GIF con Pillow...')
    frame_files = sorted([f for f in os.listdir(drive_frames_dir) if f.endswith('.png')])
    frames = []
    print(f'📊 Cargando {len(frame_files)} frames...')
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(drive_frames_dir, frame_file)
        frame_img = imageio.imread(frame_path)
        frames.append(frame_img)
        if i % 5 == 0:
            print(f'   📄 {frame_file}: {frame_img.shape}')
    output_path = f'output/national_hurricane_center/wind_fields/{safe_storm_name}_wind_field.gif'
    create_gif_with_pillow(frames, output_path, duration=1500)
    print(f'\n🎉 GIF CREATED SUCCESSFULLY')
    print(f'📁 Individual maps: {drive_frames_dir}')
    print(f'🎞️  GIF created: {output_path}')
    print(f'📊 Total frames: {len(frames)}')
    return (drive_frames_dir, output_path)


def create_static_wind_field_map(selected_storm, posicion_index):
    """Create a static wind-field map for a specific forecast position."""
    storm_name = selected_storm['name']
    storm_data = selected_storm['data']
    gdf_points = storm_data['points']
    gdf_rings = storm_data['rings']
    print(f'🌀 GENERATING STATIC MAP - {storm_name} - Position {posicion_index}')
    storm_id = gdf_points.iloc[0]['stormnum']
    gdf_points = gdf_points[gdf_points['stormnum'] == storm_id]
    gdf_points = gdf_points.sort_values(by='tau').reset_index(drop=True)
    if posicion_index >= len(gdf_points):
        print(f'❌ Error: Position {posicion_index} no disponible. Máximo: {len(gdf_points) - 1}')
        return
    target_point = gdf_points.iloc[posicion_index]
    map_extent = calculate_static_map_extent(gdf_points, posicion_index)
    fig, ax = create_fixed_axes()
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    setup_improved_map(ax)
    add_logo_and_credit(ax)
    wind_legend = add_wind_legend(ax)
    date_text = extract_date_time(target_point)
    add_date_to_map(ax, date_text)
    if posicion_index > 0:
        traj_x, traj_y = zip(*gdf_points.iloc[:posicion_index + 1].geometry.apply(lambda p: (p.x, p.y)))
        traj_line, = ax.plot(traj_x, traj_y, color='black', linewidth=2, linestyle='--', label='Trayectoria', zorder=12, transform=ccrs.PlateCarree())
    ax.scatter(target_point.geometry.x, target_point.geometry.y, color='black', s=80, zorder=20, marker='X', edgecolors='black', linewidths=1, transform=ccrs.PlateCarree())
    filtered_rings = gdf_rings[(gdf_rings['stormnum'] == storm_id) & (gdf_rings['advnum'] == target_point['advisnum']) & (gdf_rings['tau'] == target_point['tau'])]
    for _, ring_data in filtered_rings.iterrows():
        category, color = categorize_wind_speed(ring_data['radii'])
        poly = smooth_polygon(ring_data.geometry, method=SMOOTHING_METHOD)
        draw_wind_ring(ax, poly, color)
    safe_storm_name = storm_name.replace(' ', '_').replace('/', '_')
    output_path = f'output/national_hurricane_center/wind_fields/{safe_storm_name}_static_pos_{posicion_index}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print(f'✅ Static image saved at: {output_path}')


def main():
    """Interactive entry point for Pacific wind-field maps and GIFs."""
    print('🌀 MULTI-STORM WIND FIELD GENERATOR - PACIFIC')
    print('=' * 70)
    try:
        print('🔍 Step 1: Checking active storms...')
        active_storms = check_active_storms()
        if not active_storms:
            print('❌ No active storms were found right now.')
            return
        print('\n🎯 Step 2: Storm selection...')
        selection = display_storm_menu(active_storms)
        if selection is None:
            return
        selected_storm_key, selected_storm = selection
        print(f"✅ Selected storm: {selected_storm_key} - {selected_storm['name']}")
        print('\n📸 Step 3: Output mode selection...')
        mode_selection = ask_generation_mode()
        if mode_selection == (None, None):
            return
        generate_gif, fixed_position = mode_selection
        print(f'\n🚀 Step 4: Generating output...')
        if generate_gif:
            print('🎞️ GIF mode selected')
            create_stable_gif(selected_storm)
        else:
            print(f'📷 Static image mode selected (position {fixed_position})')
            create_static_wind_field_map(selected_storm, fixed_position)
        print('\n🎉 PROCESS COMPLETED SUCCESSFULLY!')
    except KeyboardInterrupt:
        print('\n👋 Operation cancelled by the user')
    except Exception as e:
        print(f'\n❌ Runtime error: {e}')
        import traceback
        traceback.print_exc()
