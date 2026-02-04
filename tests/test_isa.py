"""Tests for ISA atmosphere model and FlowState."""

import pytest
from turbofan_vpf.domain.isa import FlowState, isa_atmosphere


def test_isa_atmosphere_sea_level() -> None:
    """Test ISA conditions at sea level."""
    T, p, rho, a = isa_atmosphere(0.0)

    assert T == pytest.approx(288.15, rel=1e-3)
    assert p == pytest.approx(101325.0, rel=1e-3)
    assert rho == pytest.approx(1.225, rel=1e-2)
    assert a == pytest.approx(340.29, rel=1e-2)


def test_isa_atmosphere_tropopause() -> None:
    """Test ISA conditions at tropopause (11 km)."""
    T, p, rho, a = isa_atmosphere(11000.0)

    # Temperature should be 288.15 - 0.0065 * 11000 = 216.65 K
    assert T == pytest.approx(216.65, rel=1e-2)
    # Pressure should be significantly lower than sea level
    assert p < 30000.0
    assert p > 20000.0
    # Density should be lower
    assert rho < 0.5
    # Speed of sound should be lower due to lower temperature
    assert a < 300.0


def test_isa_atmosphere_stratosphere() -> None:
    """Test ISA conditions in stratosphere (above 11 km)."""
    T, p, rho, a = isa_atmosphere(15000.0)

    # Temperature should be constant (tropopause temperature)
    assert T == pytest.approx(216.65, rel=1e-2)
    # Pressure should be lower than at tropopause
    assert p < 20000.0
    # Speed of sound should be same as tropopause (same temperature)
    assert a == pytest.approx(295.07, rel=1e-2)


def test_isa_atmosphere_negative_altitude() -> None:
    """Test that negative altitude raises ValueError."""
    with pytest.raises(ValueError, match="Altitude must be non-negative"):
        isa_atmosphere(-100.0)


def test_flow_state_properties() -> None:
    """Test FlowState properties calculation."""
    flow = FlowState(altitude_m=11000.0, mach=0.80)

    # Check atmospheric properties
    assert flow.T == pytest.approx(216.65, rel=1e-2)
    assert flow.p < 30000.0
    assert flow.rho < 0.5
    assert flow.a < 300.0

    # Check derived properties
    assert flow.TAS == pytest.approx(flow.mach * flow.a, rel=1e-3)
    assert flow.q > 0
    assert flow.q == pytest.approx(0.5 * flow.rho * flow.TAS**2, rel=1e-3)


def test_flow_state_sea_level() -> None:
    """Test FlowState at sea level."""
    flow = FlowState(altitude_m=0.0, mach=0.25)

    assert flow.T == pytest.approx(288.15, rel=1e-3)
    assert flow.p == pytest.approx(101325.0, rel=1e-3)
    assert flow.TAS == pytest.approx(0.25 * 340.29, rel=1e-2)
    assert flow.q > 0


def test_flow_state_dynamic_pressure() -> None:
    """Test that dynamic pressure increases with Mach number."""
    flow1 = FlowState(altitude_m=11000.0, mach=0.60)
    flow2 = FlowState(altitude_m=11000.0, mach=0.80)

    assert flow2.q > flow1.q
    assert flow2.TAS > flow1.TAS
