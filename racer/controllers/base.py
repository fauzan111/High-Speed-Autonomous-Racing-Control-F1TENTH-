"""Controller interface shared by every tracking strategy."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarState:
    """Ego state handed to a controller each control step."""

    x: float          # world x [m]
    y: float          # world y [m]
    yaw: float        # heading [rad]
    v: float          # longitudinal speed [m/s]
    yaw_rate: float = 0.0  # [rad/s]
    slip: float = 0.0      # body slip angle [rad]


class Controller:
    """
    Base class. A controller maps a CarState to a raw actuation command
    (steering angle [rad], target speed [m/s]) — the F1TENTH gym action.
    """

    name = "base"

    def reset(self) -> None:
        """Clear any internal state between episodes."""

    def __call__(self, state: CarState) -> tuple[float, float]:
        raise NotImplementedError
