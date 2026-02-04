"""Domain models for aerodynamic analysis."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AirfoilPolar:
    """Represents airfoil polar data.

    Attributes:
        alpha: Angle of attack in degrees
        cl: Lift coefficient
        cd: Drag coefficient
        cm: Moment coefficient
    """

    alpha: float
    cl: float
    cd: float
    cm: Optional[float] = None


@dataclass
class FanConfiguration:
    """Represents a fan configuration.

    Attributes:
        name: Configuration name (e.g., 'VPF' or 'FPF')
        pitch_angle: Pitch angle in degrees
        rpm: Rotational speed in RPM
    """

    name: str
    pitch_angle: float
    rpm: float
