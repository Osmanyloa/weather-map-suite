"""GFS download and field extraction helpers."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr


def download_gfs_files(
    date: str,
    cycle: str,
    lead_times: list[int],
    output_dir: str | Path = "/tmp/gfs",
) -> list[Path]:
    """Download selected GFS 0.25-degree GRIB2 forecast hours."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date}/{cycle}/atmos/"

    downloaded: list[Path] = []
    for lead_time in lead_times:
        target = output_path / f"gfs.t{cycle}z.pgrb2.0p25.f{lead_time:03d}"
        if target.exists():
            downloaded.append(target)
            continue

        url = base_url + target.name
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, target.open("wb") as file:
            file.write(response.read())
        downloaded.append(target)
    return downloaded


def compute_temperature_extremes(grib_files: list[str | Path]) -> tuple[xr.Dataset, xr.Dataset]:
    """Return maximum and minimum 2-meter temperature datasets in Celsius."""
    fields = []
    for path in grib_files:
        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2},
                "indexpath": "",
            },
        )
        if "t2m" not in dataset:
            continue
        fields.append(dataset["t2m"] - 273.15)

    if not fields:
        raise ValueError("No t2m fields were found in the provided GRIB files.")

    combined = xr.concat(fields, dim="forecast_step")
    return (
        combined.max(dim="forecast_step").to_dataset(name="temperature_max"),
        combined.min(dim="forecast_step").to_dataset(name="temperature_min"),
    )


def steadman_apparent_temperature(temp_c: xr.DataArray, relative_humidity: xr.DataArray, wind_ms: xr.DataArray) -> xr.DataArray:
    """Compute Steadman's universal apparent temperature in Celsius."""
    vapor_pressure = (relative_humidity / 100.0) * 6.105 * np.exp((17.27 * temp_c) / (237.7 + temp_c))
    return temp_c + 0.348 * vapor_pressure - 0.70 * wind_ms + 0.70 * (wind_ms / (wind_ms + 10.0)) - 4.25


def compute_wind_dataset(grib_files: list[str | Path]) -> xr.Dataset:
    """Compute maximum 10-meter wind speed and gust from GFS GRIB files."""
    wind_fields = []
    gust_fields = []
    for path in grib_files:
        wind_dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10},
                "indexpath": "",
            },
        )
        if {"u10", "v10"}.issubset(wind_dataset.data_vars):
            wind_speed = np.sqrt(wind_dataset["u10"] ** 2 + wind_dataset["v10"] ** 2)
            wind_fields.append(wind_speed)

        gust_dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "surface", "shortName": "gust"},
                "indexpath": "",
            },
        )
        if "gust" in gust_dataset:
            gust_fields.append(gust_dataset["gust"])

    if not wind_fields:
        raise ValueError("No 10-meter wind fields were found in the provided GRIB files.")

    wind_max = xr.concat(wind_fields, dim="forecast_step").max(dim="forecast_step")
    result = wind_max.to_dataset(name="wind_speed")
    if gust_fields:
        result["gust"] = xr.concat(gust_fields, dim="forecast_step").max(dim="forecast_step")
    return result
