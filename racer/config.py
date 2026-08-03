"""
Vehicle parameters and paths.

The vehicle parameters mirror the single-track ("st") dynamic model used inside
the F1TENTH gym so that any model-based controller (MPC) reasons about the same
car the simulator integrates. Values are the gym defaults for the F1TENTH car.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

# --- Repository layout -------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GYM_ROOT = REPO_ROOT / "f1tenth_gym"
MAPS_DIR = GYM_ROOT / "maps"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class VehicleParams:
    """F1TENTH single-track model parameters (SI units)."""

    m: float = 3.74        # mass [kg]
    I: float = 0.04712     # yaw moment of inertia [kg m^2]
    lf: float = 0.15875    # CoG -> front axle [m]
    lr: float = 0.17145    # CoG -> rear axle [m]
    h: float = 0.074       # CoG height [m]
    mu: float = 1.0489     # friction coefficient
    C_Sf: float = 4.718    # front cornering stiffness [1/rad]
    C_Sr: float = 5.4562   # rear cornering stiffness [1/rad]

    s_min: float = -0.4189  # min steering angle [rad]
    s_max: float = 0.4189   # max steering angle [rad]
    sv_min: float = -3.2    # min steering velocity [rad/s]
    sv_max: float = 3.2     # max steering velocity [rad/s]

    v_min: float = -5.0     # min speed [m/s]
    v_max: float = 20.0     # max speed [m/s]
    a_max: float = 9.51     # max longitudinal accel [m/s^2]
    v_switch: float = 7.319  # accel derating speed [m/s]

    width: float = 0.31
    length: float = 0.58

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


# --- Simulation defaults -----------------------------------------------------
SIM_TIMESTEP = 0.01          # gym physics step [s] (100 Hz)
DEFAULT_MAP = "Spielberg"


def raceline_csv(map_name: str = DEFAULT_MAP) -> pathlib.Path:
    return MAPS_DIR / map_name / f"{map_name}_raceline.csv"


def centerline_csv(map_name: str = DEFAULT_MAP) -> pathlib.Path:
    return MAPS_DIR / map_name / f"{map_name}_centerline.csv"
