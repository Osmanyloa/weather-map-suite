"""Project paths and output folder conventions."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SHAPEFILES_DIR = RAW_DATA_DIR / "shapefiles"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_MAPS_DIR = OUTPUT_DIR / "maps"
NHC_OUTPUT_DIR = OUTPUT_DIR / "national_hurricane_center"
LOGO_PATH = ASSETS_DIR / "logo.png"


REGION_SLUGS = [
    "cuba",
    "mexico",
    "hispaniola",
    "puerto_rico",
    "central_america",
    "colombia_venezuela",
    "united_states",
    "florida",
    "texas",
    "lesser_antilles",
    "iberia",
    "canary_islands",
]

NHC_SLUGS = ["atlantic", "pacific", "wind_fields", "cones", "gifs"]

COUNTRY_MAP_DIRS = {slug: OUTPUT_MAPS_DIR / slug for slug in REGION_SLUGS}
NHC_DIRS = {slug: NHC_OUTPUT_DIR / slug for slug in NHC_SLUGS}


def ensure_output_directories() -> None:
    """Create every project output folder expected by the map workflows."""
    for directory in [OUTPUT_DIR, OUTPUT_MAPS_DIR, NHC_OUTPUT_DIR, *COUNTRY_MAP_DIRS.values(), *NHC_DIRS.values()]:
        directory.mkdir(parents=True, exist_ok=True)
