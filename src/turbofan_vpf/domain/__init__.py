"""Domain models and data structures."""

from turbofan_vpf.domain.config import CompressibilityConfig, get_config, set_config
from turbofan_vpf.domain.flight_phases import FlightPhase, get_default_phases
from turbofan_vpf.domain.isa import FlowState, isa_atmosphere
from turbofan_vpf.domain.models import AirfoilPolar, FanConfiguration

__all__ = [
    "AirfoilPolar",
    "CompressibilityConfig",
    "FanConfiguration",
    "FlightPhase",
    "FlowState",
    "get_config",
    "get_default_phases",
    "isa_atmosphere",
    "set_config",
]
