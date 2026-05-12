"""Geographic layer loading utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def load_admin_and_country_boundaries(
    admin_boundaries_path: str | Path,
    countries_path: str | Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load Natural Earth administrative and country boundary shapefiles."""
    return gpd.read_file(admin_boundaries_path), gpd.read_file(countries_path)
