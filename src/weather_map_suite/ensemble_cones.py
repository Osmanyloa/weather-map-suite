"""AI weather ensemble cone plotting helpers."""

from __future__ import annotations

from io import StringIO

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import FancyBboxPatch, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import ConvexHull

def load_csv_data(url):
    response = requests.get(url)
    data = response.text
    lines = data.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('init_time'):
            header_idx = i
            break
    csv_data = '\n'.join(lines[header_idx:])
    df = pd.read_csv(StringIO(csv_data), sep=',', engine='python')
    df.columns = df.columns.str.strip()
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['sample'] = pd.to_numeric(df['sample'], errors='coerce').astype(int)
    df_clean = df.dropna(subset=['lat', 'lon'])
    df_clean = df_clean[(df_clean['lat'] >= 0) & (df_clean['lat'] <= 40)]
    df_clean = df_clean[(df_clean['lon'] >= -100) & (df_clean['lon'] <= -50)]
    return df_clean


def draw_ensemble_cone(df, ax, color='blue', alpha=0.08, label='Modelo'):
    if 'valid_time' in df.columns:
        tiempos = sorted(df['valid_time'].unique())
    else:
        df_temp = df.copy()
        df_temp['idx'] = df_temp.groupby('sample').cumcount()
        tiempos = sorted(df_temp['idx'].unique())
        df = df_temp
    time_col = 'valid_time' if 'valid_time' in df.columns else 'idx'
    for i, tiempo in enumerate(tiempos):
        subset = df[df[time_col] == tiempo]
        if len(subset) < 3:
            continue
        points = subset[['lon', 'lat']].values
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            poly = Polygon(hull_points, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.2, transform=ccrs.PlateCarree())
            ax.add_patch(poly)
        except:
            continue
    mean_track = df.groupby(time_col)[['lon', 'lat']].mean()
    ax.plot(mean_track['lon'], mean_track['lat'], color=color, linewidth=3, marker='o', markersize=6, label=f'{label} (Media)', transform=ccrs.PlateCarree(), zorder=10)
