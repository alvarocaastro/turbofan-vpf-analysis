"""International Standard Atmosphere (ISA) model for atmospheric properties."""

import math
from dataclasses import dataclass
from typing import Tuple

# ISA sea level conditions
T0 = 288.15  # K
P0 = 101325.0  # Pa
RHO0 = 1.225  # kg/m³

# Physical constants
R = 287.058  # J/(kg·K) - specific gas constant for air
GAMMA = 1.4  # ratio of specific heats
G = 9.80665  # m/s² - standard gravity
LAPSE_RATE = -0.0065  # K/m - temperature lapse rate in troposphere
TROPOPAUSE_ALT = 11000.0  # m - tropopause altitude


def isa_atmosphere(altitude_m: float) -> Tuple[float, float, float, float]:
    """Calculate ISA atmospheric properties at given altitude.

    Simplified ISA model covering troposphere (0-11 km) and lower stratosphere
    (11-20 km). Above 20 km, returns stratosphere values.

    Args:
        altitude_m: Altitude in meters above sea level

    Returns:
        Tuple of (T, p, rho, a) where:
        - T: Temperature in Kelvin
        - p: Pressure in Pascal
        - rho: Density in kg/m³
        - a: Speed of sound in m/s

    Raises:
        ValueError: If altitude is negative
    """
    if altitude_m < 0:
        raise ValueError(f"Altitude must be non-negative, got {altitude_m}")

    if altitude_m <= TROPOPAUSE_ALT:
        # Troposphere: linear temperature decrease
        T = T0 + LAPSE_RATE * altitude_m
        # Pressure: p = p0 * (T/T0)^(g/(L*R))
        exponent = -G / (LAPSE_RATE * R)
        p = P0 * (T / T0) ** exponent
    else:
        # Stratosphere: constant temperature (isothermal layer)
        T_tropopause = T0 + LAPSE_RATE * TROPOPAUSE_ALT
        T = T_tropopause
        # Pressure at tropopause
        exponent = -G / (LAPSE_RATE * R)
        p_tropopause = P0 * (T_tropopause / T0) ** exponent
        # Isothermal layer: p = p_trop * exp(-g*(h-h_trop)/(R*T))
        p = p_tropopause * math.exp(-G * (altitude_m - TROPOPAUSE_ALT) / (R * T))

    # Density from ideal gas law: rho = p/(R*T)
    rho = p / (R * T)

    # Speed of sound: a = sqrt(gamma * R * T)
    a = math.sqrt(GAMMA * R * T)

    return T, p, rho, a


@dataclass(frozen=True)
class FlowState:
    """Represents flow state at a given altitude and Mach number.

    Attributes:
        altitude_m: Altitude in meters above sea level
        mach: Mach number

    Properties:
        T: Temperature in Kelvin
        p: Static pressure in Pascal
        rho: Density in kg/m³
        a: Speed of sound in m/s
        TAS: True airspeed in m/s
        q: Dynamic pressure in Pascal
    """

    altitude_m: float
    mach: float

    @property
    def T(self) -> float:
        """Temperature in Kelvin."""
        T, _, _, _ = isa_atmosphere(self.altitude_m)
        return T

    @property
    def p(self) -> float:
        """Static pressure in Pascal."""
        _, p, _, _ = isa_atmosphere(self.altitude_m)
        return p

    @property
    def rho(self) -> float:
        """Density in kg/m³."""
        _, _, rho, _ = isa_atmosphere(self.altitude_m)
        return rho

    @property
    def a(self) -> float:
        """Speed of sound in m/s."""
        _, _, _, a = isa_atmosphere(self.altitude_m)
        return a

    @property
    def TAS(self) -> float:
        """True airspeed in m/s."""
        return self.mach * self.a

    @property
    def q(self) -> float:
        """Dynamic pressure in Pascal."""
        return 0.5 * self.rho * self.TAS**2
