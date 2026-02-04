"""Flight phase definitions for aerodynamic analysis."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FlightPhase:
    """Represents a flight phase with operating conditions.

    Attributes:
        name: Phase name (e.g., 'takeoff', 'cruise')
        altitude_m: Altitude in meters
        mach: Mach number
    """

    name: str
    altitude_m: float
    mach: float

    def __str__(self) -> str:
        """String representation of the flight phase."""
        return f"{self.name} (alt: {self.altitude_m:.0f} m, Mach: {self.mach:.2f})"


def get_default_phases() -> List[FlightPhase]:
    """Get default flight phases for typical commercial aircraft operations.

    Returns:
        List of FlightPhase objects representing standard flight phases:
        - takeoff: Ground level, low Mach number
        - climb: Intermediate altitude, moderate Mach number
        - cruise: High altitude, high Mach number
        - descent: Intermediate altitude, moderate Mach number
        - approach: Low altitude, low Mach number
    """
    return [
        FlightPhase(name="takeoff", altitude_m=0.0, mach=0.25),
        FlightPhase(name="climb", altitude_m=5000.0, mach=0.60),
        FlightPhase(name="cruise", altitude_m=11000.0, mach=0.80),
        FlightPhase(name="descent", altitude_m=7000.0, mach=0.65),
        FlightPhase(name="approach", altitude_m=500.0, mach=0.30),
    ]
