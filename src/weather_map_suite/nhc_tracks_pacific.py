"""National Hurricane Center Eastern Pacific track and cone map tools."""

from __future__ import annotations

import datetime as dt
import os
import traceback

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import requests
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from .config import LOGO_PATH, NHC_OUTPUT_DIR, SHAPEFILES_DIR

knots_to_kmh = {20: 35, 25: 45, 30: 55, 35: 65, 40: 75, 45: 85, 50: 95, 55: 100, 60: 110, 65: 120, 70: 130, 75: 140, 80: 150, 85: 155, 90: 165, 95: 175, 100: 185, 105: 195, 110: 205, 115: 215, 120: 225, 125: 230, 130: 240, 135: 250, 140: 260, 145: 270, 150: 280, 155: 290, 160: 295, 165: 305}


PACIFIC_LAYERS = {'EP1': {'forecast_info': 135, 'forecast_points': 136, 'forecast_track': 137, 'forecast_cone': 138, 'past_points': 141, 'past_track': 142}, 'EP2': {'forecast_info': 161, 'forecast_points': 162, 'forecast_track': 163, 'forecast_cone': 164, 'past_points': 167, 'past_track': 168}, 'EP3': {'forecast_info': 187, 'forecast_points': 188, 'forecast_track': 189, 'forecast_cone': 190, 'past_points': 193, 'past_track': 194}, 'EP4': {'forecast_info': 213, 'forecast_points': 214, 'forecast_track': 215, 'forecast_cone': 216, 'past_points': 219, 'past_track': 220}, 'EP5': {'forecast_info': 239, 'forecast_points': 240, 'forecast_track': 241, 'forecast_cone': 242, 'past_points': 245, 'past_track': 246}, 'CP1': {'forecast_info': 265, 'forecast_points': 266, 'forecast_track': 267, 'forecast_cone': 268, 'past_points': 271, 'past_track': 272}, 'CP2': {'forecast_info': 291, 'forecast_points': 292, 'forecast_track': 293, 'forecast_cone': 294, 'past_points': 297, 'past_track': 298}, 'CP3': {'forecast_info': 317, 'forecast_points': 318, 'forecast_track': 319, 'forecast_cone': 320, 'past_points': 323, 'past_track': 324}, 'CP4': {'forecast_info': 343, 'forecast_points': 344, 'forecast_track': 345, 'forecast_cone': 346, 'past_points': 349, 'past_track': 350}, 'CP5': {'forecast_info': 369, 'forecast_points': 370, 'forecast_track': 371, 'forecast_cone': 372, 'past_points': 375, 'past_track': 376}}


def parse_coord(coord_str):
    """Parse coordinates from the NHC format."""
    try:
        if isinstance(coord_str, (int, float)):
            return float(coord_str)
        coord_str = str(coord_str).strip().upper()
        import re
        match = re.match('([0-9.]+)([NSEW]?)', coord_str)
        if match:
            value = float(match.group(1))
            direction = match.group(2)
            if direction in ['S', 'W']:
                value = -value
            return value
        else:
            return float(coord_str)
    except:
        return 0.0


def get_storm_classification_status():
    """Get the current classification status for all active storms."""
    try:
        url_storms = 'https://www.nhc.noaa.gov/CurrentStorms.json'
        response = requests.get(url_storms, timeout=15)
        response.raise_for_status()
        data = response.json()
        storm_classifications = {}
        for storm in data.get('activeStorms', []):
            storm_id = storm.get('id', '').lower()
            if storm_id.startswith('al'):
                name = storm.get('name', 'Sin nombre')
                classification = storm.get('classification', '').strip().upper()
                is_post_tropical = 'POST' in classification or 'PTC' in classification or 'POST-TROPICAL' in classification
                storm_classifications[name.upper()] = {'classification': classification, 'is_post_tropical': is_post_tropical, 'storm_id': storm_id}
                print(f'Storm: {name} - Classification: {classification} - Post-tropical: {is_post_tropical}')
        return storm_classifications
    except Exception as e:
        print(f'Error obteniendo clasificaciones de storms: {e}')
        return {}


def detect_active_pacific_storms():
    """Automatically detect active Eastern/Central Pacific storms."""
    print('🔍 Detecting active storms in the Eastern/Central Pacific...')
    print('=' * 60)
    BASE_URL = 'https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/NHC_tropical_weather/MapServer'
    active_storms = {}
    for storm_at, layers in PACIFIC_LAYERS.items():
        print(f'\n📍 Revisando {storm_at}...')
        storm_data = {}
        has_data = False
        storm_name = None
        storm_info = {}
        for layer_type, layer_id in layers.items():
            try:
                query_url = f'{BASE_URL}/{layer_id}/query'
                params = {'where': '1=1', 'outFields': '*', 'f': 'geojson', 'returnGeometry': 'true'}
                response = requests.get(query_url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                if 'features' in data and data['features']:
                    feature_count = len(data['features'])
                    has_data = True
                    temp_gdf = gpd.GeoDataFrame.from_features(data['features'])
                    storm_data[layer_type] = temp_gdf
                    if not storm_name and (not temp_gdf.empty):
                        first_feature = temp_gdf.iloc[0]
                        name_fields = ['STORMNAME', 'NAME', 'STORM', 'STORMNUM']
                        for field in name_fields:
                            if field in first_feature and pd.notna(first_feature[field]):
                                storm_name = str(first_feature[field]).strip()
                                break
                        if not storm_info:
                            storm_info = extract_storm_basic_info(first_feature)
                    print(f'      ✅ {layer_type}: {feature_count} registros')
                else:
                    print(f'      ❌ {layer_type}: Sin datos')
            except Exception as e:
                print(f'      ❌ {layer_type}: Error - {str(e)[:40]}...')
        if has_data:
            display_name = storm_name if storm_name else f'Storm {storm_at}'
            active_storms[display_name] = {'at_id': storm_at, 'name': storm_name, 'data': storm_data, 'info': storm_info}
            print(f'   🌀 {storm_at} ACTIVA: {display_name}')
            if storm_info:
                for key, value in storm_info.items():
                    if value:
                        print(f'      {key}: {value}')
        else:
            print(f'   ❌ {storm_at}: Sin datos')
    return active_storms


def extract_storm_basic_info(storm_record):
    """Extract basic information de un registro de storm"""
    info = {}
    field_mappings = {'Classification': ['CLASS', 'CLASSIFICATION', 'TYPE', 'STORMTYPE'], 'Wind speed': ['MAXWIND', 'WIND', 'WINDSPEED', 'INTENSITY'], 'Pressure': ['MINCP', 'PRESSURE', 'MSLP'], 'Movement': ['SPEED', 'MOVESPEED', 'FWDSPEED'], 'Direction': ['DIR', 'FWDDIR', 'MOVEDIR'], 'Date': ['DATE', 'ADVDATE', 'TIMESTAMP', 'SYNOPTIME']}
    for info_key, possible_fields in field_mappings.items():
        for field in possible_fields:
            if field in storm_record and pd.notna(storm_record[field]):
                value = storm_record[field]
                if str(value).strip():
                    info[info_key] = value
                    break
    return info


def select_storm_interactive(active_storms):
    """Let the user select an active storm interactively."""
    if not active_storms:
        print('❌ No records found active storms')
        return None
    if len(active_storms) == 1:
        storm_name = list(active_storms.keys())[0]
        print(f'✅ Only one storm is active: {storm_name}')
        return (storm_name, active_storms[storm_name])
    print(f'\n🌀 ACTIVE STORMS FOUND ({len(active_storms)}):')
    print('=' * 50)
    storm_list = list(active_storms.keys())
    for i, storm_name in enumerate(storm_list, 1):
        storm_data = active_storms[storm_name]
        print(f"\n{i}. {storm_name} ({storm_data['at_id']})")
        if storm_data['info']:
            for key, value in storm_data['info'].items():
                print(f'   {key}: {value}')
        available_layers = list(storm_data['data'].keys())
        print(f"   Available layers: {', '.join(available_layers)}")
    print('\n' + '=' * 50)
    while True:
        try:
            choice = input(f"\n🎯 Select a storm (1-{len(storm_list)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print('❌ Operation cancelled by the user')
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(storm_list):
                selected_storm = storm_list[choice_num - 1]
                print(f'✅ Selected: {selected_storm}')
                return (selected_storm, active_storms[selected_storm])
            else:
                print(f'❌ Please select a number between 1 and {len(storm_list)}')
        except ValueError:
            print('❌ Please enter a valid number')
        except KeyboardInterrupt:
            print('\n❌ Operation cancelled')
            return None


def convert_storm_data_format(selected_storm_data):
    """Convert storm data into the format expected by the plotting functions."""
    storm_data = {}
    layer_mapping = {'forecast_cone': 'cone', 'forecast_track': 'track', 'forecast_points': 'points', 'past_track': 'past_track', 'past_points': 'past_points'}
    for original_key, target_key in layer_mapping.items():
        if original_key in selected_storm_data['data']:
            storm_data[target_key] = selected_storm_data['data'][original_key]
    return storm_data


def get_storm_center(storm_data):
    """Get el centro ACTUAL de la storm (position presente, no futura)"""
    center_lon, center_lat = (None, None)
    if 'points' in storm_data and storm_data['points'] is not None:
        points_gdf = storm_data['points']
        if not points_gdf.empty:
            print('Searching position actual en forecast points...')
            current_point = None
            if 'validtime' in points_gdf.columns:
                zero_hour_points = points_gdf[points_gdf['validtime'].str.startswith('0/', na=False) | (points_gdf['validtime'] == '0/0000') | (points_gdf['validtime'] == '0/00')]
                if not zero_hour_points.empty:
                    current_point = zero_hour_points.iloc[0]
                    print(f"Found punto actual con validtime: {current_point['validtime']}")
            if current_point is None and any((col in points_gdf.columns for col in ['POINTTYPE', 'TYPE', 'STATUS'])):
                for col in ['POINTTYPE', 'TYPE', 'STATUS']:
                    if col in points_gdf.columns:
                        current_points = points_gdf[points_gdf[col].str.contains('CURRENT|PRESENT|NOW', case=False, na=False)]
                        if not current_points.empty:
                            current_point = current_points.iloc[0]
                            print(f'Found punto actual por {col}')
                            break
            if current_point is None:
                print('Using el primer punto de forecast como position actual')
                if 'validtime' in points_gdf.columns:
                    try:
                        points_gdf_sorted = points_gdf.copy()
                        points_gdf_sorted['hour_num'] = points_gdf_sorted['validtime'].str.extract('(\\d+)/').astype(float)
                        points_gdf_sorted = points_gdf_sorted.sort_values('hour_num')
                        current_point = points_gdf_sorted.iloc[0]
                    except:
                        current_point = points_gdf.iloc[0]
                else:
                    current_point = points_gdf.iloc[0]
            if current_point is not None and current_point.geometry and hasattr(current_point.geometry, 'x'):
                center_lon, center_lat = (current_point.geometry.x, current_point.geometry.y)
                return (center_lon, center_lat)
    if center_lon is None and 'track' in storm_data and (storm_data['track'] is not None):
        track_gdf = storm_data['track']
        if not track_gdf.empty:
            print('Using inicio de la track como position actual')
            first_track = track_gdf.iloc[0]
            if first_track.geometry:
                if first_track.geometry.geom_type == 'LineString':
                    coords = list(first_track.geometry.coords)
                    center_lon, center_lat = coords[0]
                elif first_track.geometry.geom_type == 'Point':
                    center_lon, center_lat = (first_track.geometry.x, first_track.geometry.y)
                else:
                    centroid = first_track.geometry.centroid
                    center_lon, center_lat = (centroid.x, centroid.y)
    if center_lon is None and 'cone' in storm_data and (storm_data['cone'] is not None):
        cone_gdf = storm_data['cone']
        if not cone_gdf.empty:
            print('Using centroide del cone como last option')
            centroid = cone_gdf.geometry.unary_union.centroid
            center_lon, center_lat = (centroid.x, centroid.y)
    return (center_lon, center_lat)


def calculate_map_extent(center_lon, center_lat):
    """Calcular los bounds del mapa basado en la position ACTUAL de la storm"""
    west = center_lon - 30
    east = center_lon + 27
    south = center_lat - 6.9
    north = center_lat + 24
    return [west, east, south, north]


def draw_cone(ax, gdf_cone):
    """Dibujar cone de track"""
    cone_polygon = None
    if gdf_cone is not None and (not gdf_cone.empty):
        try:
            gdf_plot = gdf_cone.to_crs('EPSG:4326') if gdf_cone.crs and gdf_cone.crs != 'EPSG:4326' else gdf_cone
            gdf_plot = gdf_plot[gdf_plot.geometry.notna()]
            if not gdf_plot.empty:
                for geom in gdf_plot.geometry:
                    geoms = [geom] if geom.geom_type == 'Polygon' else geom.geoms
                    for poly in geoms:
                        hatch_patch = mpatches.Polygon(list(poly.exterior.coords), facecolor='white', edgecolor='none', linewidth=0, alpha=0.25, transform=ccrs.PlateCarree(), zorder=15, clip_on=True)
                        ax.add_patch(hatch_patch)
                for geom in gdf_plot.geometry:
                    geoms = [geom] if geom.geom_type == 'Polygon' else geom.geoms
                    for poly in geoms:
                        shadow_patch = mpatches.Polygon(list(poly.exterior.coords), facecolor='none', edgecolor='black', linewidth=4, alpha=0.3, transform=ccrs.PlateCarree(), zorder=19, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], clip_on=True)
                        ax.add_patch(shadow_patch)
                        border_patch = mpatches.Polygon(list(poly.exterior.coords), facecolor='none', edgecolor='#FF0000', linewidth=3, alpha=0.8, transform=ccrs.PlateCarree(), zorder=20, clip_on=True)
                        ax.add_patch(border_patch)
                cone_polygon = gdf_plot.geometry.unary_union
        except Exception as e:
            print(f'Error dibujando cone: {e}')
    return cone_polygon


def draw_track(ax, gdf_track):
    """Dibujar track"""
    if gdf_track is not None and (not gdf_track.empty):
        try:
            gdf_track_plot = gdf_track.to_crs('EPSG:4326') if gdf_track.crs and gdf_track.crs != 'EPSG:4326' else gdf_track
            gdf_track_plot = gdf_track_plot[gdf_track_plot.geometry.notna()]
            if not gdf_track_plot.empty:
                gdf_track_plot.plot(ax=ax, facecolor='none', edgecolor='black', alpha=0.3, linewidth=5, zorder=17, transform=ccrs.PlateCarree(), path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)])
                gdf_track_plot.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.8, linewidth=3, zorder=18, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f'Error dibujando track: {e}')


def is_point_post_tropical(point, storm_classifications, storm_name):
    """Determinar si un punto específico es postropical"""
    post_tropical_indicators = ['PTC', 'POST-TROPICAL', 'POST TROPICAL', 'POSTROPICAL', 'POST-TROP', 'EXTRATROPICAL', 'EXTRA-TROPICAL']
    for col in point.index:
        if isinstance(point[col], str):
            col_value = str(point[col]).upper()
            for indicator in post_tropical_indicators:
                if indicator in col_value:
                    print(f'Punto detectado como postropical por columna {col}: {col_value}')
                    return True
    if storm_name and storm_name.upper() in storm_classifications:
        is_post_tropical = storm_classifications[storm_name.upper()]['is_post_tropical']
        if is_post_tropical:
            print(f'Punto considerado postropical por clasificación general de {storm_name}')
        return is_post_tropical
    return False


def draw_points(ax, gdf_points, cone_polygon, storm_classifications=None, storm_name=None):
    """Dibujar points de la track con detección de ciclones postropicales"""
    if gdf_points is not None and (not gdf_points.empty):
        try:
            gdf_points_plot = gdf_points.to_crs('EPSG:4326') if gdf_points.crs and gdf_points.crs != 'EPSG:4326' else gdf_points
            dias_es = {'Mon': 'LUN', 'Tue': 'MAR', 'Wed': 'MIÉ', 'Thu': 'JUE', 'Fri': 'VIE', 'Sat': 'SÁB', 'Sun': 'DOM'}
            points_list = list(gdf_points_plot.iterrows())
            if 'validtime' in gdf_points_plot.columns:
                try:
                    gdf_points_sorted = gdf_points_plot.copy()
                    gdf_points_sorted['hour_num'] = gdf_points_sorted['validtime'].str.extract('(\\d+)/').astype(float)
                    gdf_points_sorted = gdf_points_sorted.sort_values('hour_num')
                    points_list = list(gdf_points_sorted.iterrows())
                except:
                    pass
            label_positions = []
            points_data = []
            for idx, point in points_list:
                if point.geometry and hasattr(point.geometry, 'x'):
                    points_data.append({'lon': point.geometry.x, 'lat': point.geometry.y, 'point': point, 'idx': idx})

            def labels_overlap(lon1, lat1, width1, height1, lon2, lat2, width2, height2):
                left1, right1 = (lon1 - width1 / 2, lon1 + width1 / 2)
                bottom1, top1 = (lat1 - height1 / 2, lat1 + height1 / 2)
                left2, right2 = (lon2 - width2 / 2, lon2 + width2 / 2)
                bottom2, top2 = (lat2 - height2 / 2, lat2 + height2 / 2)
                return not (right1 < left2 or right2 < left1 or top1 < bottom2 or (top2 < bottom1))
            for i, point_data in enumerate(points_data):
                lon, lat = (point_data['lon'], point_data['lat'])
                point = point_data['point']
                wind_speed_kt = 35
                wind_cols = [col for col in point.index if any((word in col.upper() for word in ['WIND', 'SPEED', 'MPH', 'KT', 'KNOT']))]
                if wind_cols:
                    try:
                        wind_speed_kt = float(str(point[wind_cols[0]]).replace('kt', '').replace('mph', '').strip())
                    except:
                        wind_speed_kt = 35
                wind_speed_kmh = knots_to_kmh.get(int(wind_speed_kt), int(wind_speed_kt * 1.852))
                is_post_tropical = is_point_post_tropical(point, storm_classifications, storm_name)
                if wind_speed_kt < 34:
                    classification, base_color, font_size = ('D', '#0000FF', 10)
                elif wind_speed_kt < 64:
                    classification, base_color, font_size = ('S', '#008000', 11)
                elif wind_speed_kt >= 137:
                    classification, base_color, font_size = ('5', '#800080', 13)
                elif wind_speed_kt >= 113:
                    classification, base_color, font_size = ('4', '#dc2626', 13)
                elif wind_speed_kt >= 96:
                    classification, base_color, font_size = ('3', '#ff922b', 13)
                elif wind_speed_kt >= 83:
                    classification, base_color, font_size = ('2', '#ffd93d', 13)
                else:
                    classification, base_color, font_size = ('1', '#ADFF2F', 13)
                if is_post_tropical:
                    color = 'white'
                    if wind_speed_kt >= 64:
                        classification = 'H'
                else:
                    color = base_color
                size = max(150, min(300, wind_speed_kt * 3))
                ax.scatter(lon, lat, c='black', s=size + 100, alpha=0.3, zorder=25, transform=ccrs.PlateCarree(), clip_on=True)
                ax.scatter(lon, lat, c=color, s=size, marker='o', edgecolors='black' if is_post_tropical else 'white', linewidth=3, zorder=27, transform=ccrs.PlateCarree(), alpha=0.9, clip_on=True)
                text = ax.text(lon, lat, classification, transform=ccrs.PlateCarree(), fontsize=font_size, fontweight='bold', color='black', ha='center', va='center', zorder=28, clip_on=True)
                text.set_path_effects([PathEffects.Normal()])
                try:
                    if i == 0 and len(points_data) > 1:
                        vx = points_data[i + 1]['lon'] - lon
                        vy = points_data[i + 1]['lat'] - lat
                    elif i == len(points_data) - 1 and len(points_data) > 1:
                        vx = lon - points_data[i - 1]['lon']
                        vy = lat - points_data[i - 1]['lat']
                    elif len(points_data) > 2:
                        vx = points_data[i + 1]['lon'] - points_data[i - 1]['lon']
                        vy = points_data[i + 1]['lat'] - points_data[i - 1]['lat']
                    else:
                        vx, vy = (1, 0)
                    perp_x, perp_y = (vy, -vx)
                    norm = np.sqrt(perp_x ** 2 + perp_y ** 2) or 1
                    perp_x /= norm
                    perp_y /= norm
                    base_distance = 3
                    label_width, label_height = (1.2, 0.6)
                    direction = 1 if i % 2 == 0 else -1
                    label_lon = lon + direction * base_distance * perp_x
                    label_lat = lat + direction * base_distance * perp_y
                    overlaps = any((labels_overlap(label_lon, label_lat, label_width, label_height, prev_lon, prev_lat, prev_w, prev_h) for prev_lon, prev_lat, prev_w, prev_h in label_positions))
                    if overlaps:
                        direction = -direction
                        label_lon = lon + direction * base_distance * perp_x
                        label_lat = lat + direction * base_distance * perp_y
                    label_positions.append((label_lon, label_lat, label_width, label_height))
                except:
                    label_lon, label_lat = (lon + 2.5, lat)
                if 'datelbl' in point.index and point['datelbl']:
                    date_info_raw = point['datelbl']
                    try:
                        parts = date_info_raw.split()
                        hour = parts[0].split(':')[0]
                        ampm = parts[1]
                        day_en = parts[2]
                        day_es = dias_es.get(day_en, day_en)
                        date_info = f'{day_es} {hour} {ampm}'
                    except:
                        date_info = date_info_raw
                ax.plot([lon, label_lon], [lat, label_lat], color='black', linewidth=2.5, alpha=0.4, zorder=19, transform=ccrs.PlateCarree(), clip_on=True)
                ax.plot([lon, label_lon], [lat, label_lat], color='white', linewidth=1.5, alpha=0.9, zorder=20, transform=ccrs.PlateCarree(), clip_on=True)
                bbox_props = dict(boxstyle='round,pad=0.25', facecolor='#666666' if is_post_tropical else '#1e40af', alpha=0.95, edgecolor='white', linewidth=1.3)
                suffix = ' (PTC)' if is_post_tropical else ''
                ax.text(label_lon, label_lat, f'{date_info}\n{wind_speed_kmh} km/h{suffix}', transform=ccrs.PlateCarree(), fontsize=6, fontweight='bold', color='white', ha='center', va='center', zorder=30, bbox=bbox_props, clip_on=True)
        except Exception as e:
            print(f'Error drawing points: {e}')


def draw_storm_elements(ax, storm_data, storm_classifications=None, storm_name=None):
    """Dibujar elementos de la storm (cone, track, points)"""
    cone_polygon = None
    if 'cone' in storm_data and storm_data['cone'] is not None:
        gdf_cone = storm_data['cone']
        cone_polygon = draw_cone(ax, gdf_cone)
    if 'track' in storm_data and storm_data['track'] is not None:
        gdf_track = storm_data['track']
        draw_track(ax, gdf_track)
    if 'points' in storm_data and storm_data['points'] is not None:
        gdf_points = storm_data['points']
        draw_points(ax, gdf_points, cone_polygon, storm_classifications, storm_name)
    return cone_polygon


def create_storm_map(storm_name, storm_data, storm_classifications=None):
    """Create a map for a specific storm centered on its current position."""
    print(f'Creating map for {storm_name}...')
    center_lon, center_lat = get_storm_center(storm_data)
    if center_lon is None or center_lat is None:
        print(f'Could not determine the current center for {storm_name}')
        return None
    print(f'Current center for {storm_name}: {center_lon:.2f}, {center_lat:.2f}')
    extent = calculate_map_extent(center_lon, center_lat)
    print(f'Map extent centered on current position: {extent}')
    fig = plt.figure(figsize=(14, 7))
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='steelblue', edgecolor='none'), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='lightgray', edgecolor='none'), zorder=1)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m', facecolor='#2C3E50', edgecolor='gray'), linewidth=0.8, zorder=2)
    coastline_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    coastline_geoms = list(coastline_feature.geometries())
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.5, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=3, clip_on=True)
    ax.add_geometries(coastline_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.3, path_effects=[PathEffects.Normal()], zorder=4, clip_on=True)
    borders_feature = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m')
    borders_geoms = list(borders_feature.geometries())
    ax.add_geometries(borders_geoms, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=2.0, alpha=0.3, path_effects=[PathEffects.SimpleLineShadow(offset=(2, -2), alpha=0.5)], zorder=5, clip_on=True)
    ax.add_geometries(borders_geoms, crs=ccrs.PlateCarree(), edgecolor='white', facecolor='none', linewidth=1.0, path_effects=[PathEffects.Normal()], zorder=6, clip_on=True)
    try:
        shp_path = SHAPEFILES_DIR / 'ne_10m_admin_1_states_provinces.shp'
        gdf_states = gpd.read_file(shp_path)
        estados_eeuu = gdf_states[(gdf_states['admin'] == 'United States of America') & (gdf_states['type'] != 'Country')]
        all_boundaries = []
        for idx1, state1 in estados_eeuu.iterrows():
            for idx2, state2 in estados_eeuu.iterrows():
                if idx1 < idx2:
                    shared_boundary = state1.geometry.boundary.intersection(state2.geometry.boundary)
                    if not shared_boundary.is_empty:
                        all_boundaries.append(shared_boundary)
        for boundary in all_boundaries:
            if boundary.geom_type == 'LineString':
                x_coords, y_coords = boundary.xy
                ax.plot(x_coords, y_coords, color='gray', linewidth=0.3, transform=ccrs.PlateCarree(), zorder=4, clip_on=True)
            elif boundary.geom_type == 'MultiLineString':
                for line in boundary.geoms:
                    x_coords, y_coords = line.xy
                    ax.plot(x_coords, y_coords, color='gray', linewidth=0.3, transform=ccrs.PlateCarree(), zorder=4, clip_on=True)
    except Exception as e:
        print(f'Could not load el shapefile de estados de EE.UU.: {e}')
    lon_range = np.arange(int(extent[0]), int(extent[1]) + 1, 5)
    lat_range = np.arange(int(extent[2]), int(extent[3]) + 1, 5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), xlocs=lon_range, ylocs=lat_range, linewidth=0.5, color='white', alpha=0.6, linestyle='-', draw_labels=True, zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}
    cone_polygon = draw_storm_elements(ax, storm_data, storm_classifications, storm_name)
    try:
        logo_img = mpimg.imread(LOGO_PATH)
        logo_width = 0.095
        logo_height = 0.095
        map_width = extent[1] - extent[0]
        map_height = extent[3] - extent[2]
        anchor_x = -0.04
        anchor_y = 0.085
        axins_logo = inset_axes(ax, width=f'{logo_width * 100}%', height=f'{logo_height * 100}%', loc='lower right', bbox_to_anchor=(anchor_x, anchor_y, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
        axins_logo.imshow(logo_img)
        axins_logo.axis('off')
        print('Logo loaded successfully')
    except Exception as e:
        print(f'Could not load el logo: {e}')
    credit_lon = extent[1] - (extent[1] - extent[0]) * 0.02
    credit_lat = extent[2] + (extent[3] - extent[2]) * 0.02
    ax.text(credit_lon, credit_lat, 'Created by ElTiempoconLorenzo', transform=ccrs.PlateCarree(), fontsize=10, ha='right', va='bottom', color='black', fontstyle='italic', fontweight='bold', zorder=35, path_effects=[PathEffects.withStroke(linewidth=2, foreground='white'), PathEffects.Normal()], clip_on=True)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    filename = f"output/national_hurricane_center/cones/{storm_name.replace(' ', '_').replace('/', '_')}_map.png"
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    print(f'Map saved: {filename}')
    return fig


def main():
    """Interactive entry point for detecting and mapping Pacific storms."""
    print('🌀 NHC STORM TRACKER - EASTERN/CENTRAL PACIFIC')
    print('=' * 60)
    print('\n📡 Fetching storm classifications...')
    storm_classifications = get_storm_classification_status()
    active_storms = detect_active_pacific_storms()
    if not active_storms:
        print('\n❌ No active Pacific storms were found')
        print('   Possible reasons:')
        print('   - There are no active storms right now')
        print('   - NHC data has not been updated yet')
        print('   - Network connectivity failed')
        return
    selection_result = select_storm_interactive(active_storms)
    if selection_result is None:
        print('❌ No storm was selected')
        return
    selected_storm_name, selected_storm_data = selection_result
    print(f'\n🔄 Processing data for {selected_storm_name}...')
    storm_data = convert_storm_data_format(selected_storm_data)
    try:
        print(f'\n🎨 Creating map for {selected_storm_name}...')
        fig = create_storm_map(selected_storm_name, storm_data, storm_classifications)
        if fig is not None:
            plt.show()
            plt.close(fig)
            print(f'✅ Map created successfully for {selected_storm_name}')
        else:
            print(f'❌ Could not create the map for {selected_storm_name}')
    except Exception as e:
        print(f'❌ Error creating map for {selected_storm_name}: {e}')
        import traceback
        traceback.print_exc()


def run_all_storms():
    """Create maps for every active Pacific storm."""
    print('🌀 CREATING MAPS FOR ALL ACTIVE PACIFIC STORMS')
    print('=' * 60)
    storm_classifications = get_storm_classification_status()
    active_storms = detect_active_pacific_storms()
    if not active_storms:
        print('❌ No active storms found')
        return
    figures = []
    for storm_name, storm_data_full in active_storms.items():
        try:
            print(f'\n🎨 Processing {storm_name}...')
            storm_data = convert_storm_data_format(storm_data_full)
            fig = create_storm_map(storm_name, storm_data, storm_classifications)
            if fig is not None:
                figures.append(fig)
                plt.show()
                plt.close(fig)
        except Exception as e:
            print(f'❌ Error creating map for {storm_name}: {e}')
    print(f'\n✅ Successfully created {len(figures)} maps')
