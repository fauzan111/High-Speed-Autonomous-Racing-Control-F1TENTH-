"""Download a simulator map and prepare its assets before loading a racing line.

Usage: python -m scripts.prepare_map --map Spielberg
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from racer import config


def prepare_map_assets(track_dir: Path, map_name: str) -> Path:
    """Apply the v1.0.0 YAML naming workaround without replacing existing files."""
    track_dir = Path(track_dir)
    expected_yaml = track_dir / f"{track_dir.stem}_map.yaml"
    legacy_yaml = track_dir / f"{track_dir.stem}.yaml"
    if not expected_yaml.is_file() and legacy_yaml.is_file():
        shutil.copyfile(legacy_yaml, expected_yaml)
    required = [expected_yaml, track_dir / f"{map_name}_raceline.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing downloaded map assets: " + ", ".join(missing))
    return track_dir


def prepare_map(map_name: str) -> Path:
    # Imported lazily so --help and asset-only unit tests need no simulator.
    # Source: v1.0.0 track/utils.py, commit 5a301bd0ae1ceaf7dec653e7549c8d099db58a6b.
    from f1tenth_gym.envs.track.utils import find_track_dir

    track_dir = prepare_map_assets(find_track_dir(map_name), map_name)
    expected_csv = config.raceline_csv(map_name)
    actual_csv = track_dir / f"{map_name}_raceline.csv"
    if actual_csv.resolve() != expected_csv.resolve():
        raise RuntimeError(
            f"Simulator map is at {actual_csv}, but this project expects {expected_csv}. "
            "Install the simulator editable from ./f1tenth_gym as documented."
        )
    return track_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=config.DEFAULT_MAP)
    args = parser.parse_args()
    if not args.map or Path(args.map).name != args.map or args.map in {".", ".."}:
        parser.error("map must be a track name, not a filesystem path")
    print(f"Prepared map: {prepare_map(args.map)}")


if __name__ == "__main__":
    main()
