"""Create the project output folder structure."""

from weather_map_suite import COUNTRY_MAP_DIRS, NHC_DIRS, ensure_output_directories


def main() -> None:
    ensure_output_directories()
    print("Map output folders:")
    for name, path in COUNTRY_MAP_DIRS.items():
        print(f"  {name}: {path}")

    print("\nNational Hurricane Center output folders:")
    for name, path in NHC_DIRS.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
