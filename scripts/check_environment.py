"""Check dependencies and assets without importing the simulator.

Usage: python -m scripts.check_environment
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    checks = {
        "Python 3.9–3.12": (3, 9) <= sys.version_info[:2] <= (3, 12),
        **{
            name: importlib.util.find_spec(name) is not None
            for name in ("numpy", "scipy", "matplotlib", "casadi", "gymnasium", "f1tenth_gym", "pytest")
        },
        "simulator checkout": (root / "f1tenth_gym" / "pyproject.toml").is_file(),
        "Spielberg racing line": (root / "f1tenth_gym/maps/Spielberg/Spielberg_raceline.csv").is_file(),
        "Spielberg map config": (root / "f1tenth_gym/maps/Spielberg/Spielberg_map.yaml").is_file(),
    }
    print(f"Interpreter: {sys.executable}")
    for label, present in checks.items():
        print(f"{'OK     ' if present else 'MISSING'} {label}")
    print("\nImport availability only; run tests and a simulation smoke test after setup.")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
