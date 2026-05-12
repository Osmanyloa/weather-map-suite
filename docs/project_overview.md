# Project Overview

This repository turns a large exploratory notebook into a modular map-production project. The code is organized around operational weather products:

- Temperature maps
- Wind maps
- National Hurricane Center track and cone products
- National Hurricane Center wind-field products
- Ocean and sea-surface maps
- AI ensemble cone helpers
- Base map utilities

The project is designed as a portfolio repository: clear structure, English documentation, no notebook-side effects, no hard-coded credentials, and dedicated output folders for generated products.

## Data Inputs

Typical inputs include:

- GFS GRIB2 files from NOAA NOMADS
- Natural Earth shapefiles
- NHC GIS layers
- Ocean model or SST datasets
- Optional logo assets in `assets/`

## Outputs

Generated products should be stored under `output/`. Regional maps are grouped by region or country. National Hurricane Center products have their own output area.
