# Weather Map Suite

Production-oriented Python project for meteorological map generation. The code was reorganized from an exploratory notebook into importable modules for temperature maps, wind maps, National Hurricane Center products, ocean maps, ensemble cones, and reusable base maps.

## What This Project Contains

- GFS download and field extraction helpers for temperature, apparent temperature, wind speed, and gusts.
- Regional temperature map renderers.
- Regional wind map renderers.
- National Hurricane Center track/cone tools for Atlantic and Eastern Pacific systems.
- National Hurricane Center wind-field map and GIF tools.
- Ocean and sea-surface map helpers.
- A clean output structure for storing generated images by region and NHC product type.

## Folder Structure

```text
weather-map-suite/
├── src/weather_map_suite/
│   ├── config.py
│   ├── gfs_fields.py
│   ├── geography.py
│   ├── temperature_maps.py
│   ├── wind_maps.py
│   ├── nhc_tracks_atlantic.py
│   ├── nhc_tracks_pacific.py
│   ├── nhc_wind_fields_atlantic.py
│   ├── nhc_wind_fields_pacific.py
│   ├── ocean_maps.py
│   ├── ensemble_cones.py
│   └── base_maps.py
├── output/
│   ├── maps/
│   │   ├── cuba/
│   │   ├── mexico/
│   │   ├── hispaniola/
│   │   ├── puerto_rico/
│   │   ├── central_america/
│   │   ├── colombia_venezuela/
│   │   ├── united_states/
│   │   ├── florida/
│   │   ├── texas/
│   │   ├── lesser_antilles/
│   │   ├── iberia/
│   │   └── canary_islands/
│   └── national_hurricane_center/
│       ├── atlantic/
│       ├── pacific/
│       ├── cones/
│       ├── wind_fields/
│       └── gifs/
├── data/
├── assets/
├── docs/
└── examples/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`cfgrib` requires the system `eccodes` library. On macOS, install it with:

```bash
brew install eccodes
```

## Output Folders

The project already includes the output folders you asked for. You can place generated images manually inside:

- `output/maps/<region>/`
- `output/national_hurricane_center/atlantic/`
- `output/national_hurricane_center/pacific/`
- `output/national_hurricane_center/cones/`
- `output/national_hurricane_center/wind_fields/`
- `output/national_hurricane_center/gifs/`

## Security Note

The source notebook is not included in this repository because it contained local Colab paths and a Copernicus login call. Keep credentials in environment variables or a local `.env` file that is never committed.

## Author

Osmany Lorenzo Amaro
