"""Tests for incidence angle control (FPF vs VPF)."""

import numpy as np
import pytest

from turbofan_vpf.aero.incidence_control import alpha_fpf, alpha_vpf, compare_phase
from turbofan_vpf.domain.flight_phases import FlightPhase


def test_alpha_fpf() -> None:
    """Test FPF angle calculation."""
    phase = FlightPhase(name="cruise", altitude_m=11000.0, mach=0.80)
    alpha = alpha_fpf(phase)

    # At mach=0.80, alpha = 4.0 + 2.0 * (0.80 - 0.5) = 4.0 + 0.6 = 4.6
    assert alpha == pytest.approx(4.6)


def test_alpha_fpf_reference_mach() -> None:
    """Test FPF angle at reference Mach number."""
    phase = FlightPhase(name="test", altitude_m=0.0, mach=0.5)
    alpha = alpha_fpf(phase)

    # At reference Mach (0.5), alpha should be alpha_0 = 4.0
    assert alpha == pytest.approx(4.0)


def test_alpha_fpf_low_mach() -> None:
    """Test FPF angle at low Mach number."""
    phase = FlightPhase(name="takeoff", altitude_m=0.0, mach=0.25)
    alpha = alpha_fpf(phase)

    # At mach=0.25, alpha = 4.0 + 2.0 * (0.25 - 0.5) = 4.0 - 0.5 = 3.5
    assert alpha == pytest.approx(3.5)


def test_alpha_vpf_min_cd() -> None:
    """Test VPF angle calculation for minimum CD target."""
    alpha = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    cl = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    cd = np.array([0.02, 0.015, 0.01, 0.012, 0.015])
    polar = (alpha, cl, cd)

    vpf_alpha = alpha_vpf(polar, target="min_cd")
    assert vpf_alpha == pytest.approx(4.0)  # Minimum cd at alpha=4.0


def test_alpha_vpf_max_ld() -> None:
    """Test VPF angle calculation for maximum L/D target."""
    alpha = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    cl = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    cd = np.array([0.02, 0.02, 0.02, 0.03, 0.05])
    polar = (alpha, cl, cd)

    vpf_alpha = alpha_vpf(polar, target="max_ld")
    # At alpha=4.0, L/D = 0.4/0.02 = 20.0 (maximum)
    assert vpf_alpha == pytest.approx(4.0)


def test_alpha_vpf_default_target() -> None:
    """Test VPF angle with default target (max_ld)."""
    alpha = np.array([0.0, 4.0, 8.0])
    cl = np.array([0.0, 0.4, 0.8])
    cd = np.array([0.02, 0.02, 0.05])
    polar = (alpha, cl, cd)

    vpf_alpha = alpha_vpf(polar)
    assert vpf_alpha == pytest.approx(4.0)


def test_alpha_vpf_invalid_target() -> None:
    """Test that invalid target raises ValueError."""
    alpha = np.array([0.0, 4.0])
    cl = np.array([0.0, 0.4])
    cd = np.array([0.02, 0.02])
    polar = (alpha, cl, cd)

    with pytest.raises(ValueError, match="Invalid target"):
        alpha_vpf(polar, target="invalid")  # type: ignore


def test_compare_phase() -> None:
    """Test phase comparison between FPF and VPF."""
    alpha = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    cl = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    cd = np.array([0.02, 0.015, 0.01, 0.012, 0.015])
    polar = (alpha, cl, cd)

    phase = FlightPhase(name="cruise", altitude_m=11000.0, mach=0.80)
    result = compare_phase(polar, phase, vpf_target="min_cd")

    # Check that all expected keys are present
    expected_keys = [
        "fpf_alpha",
        "vpf_alpha",
        "fpf_cd",
        "vpf_cd",
        "fpf_ld",
        "vpf_ld",
        "delta_cd",
        "delta_ld",
        "cd_ratio",
        "ld_ratio",
        "cd_reduction_pct",
        "ld_improvement_pct",
    ]
    for key in expected_keys:
        assert key in result

    # Check that VPF should have lower or equal CD when targeting min_cd
    assert result["vpf_cd"] <= result["fpf_cd"]

    # Check that angles are calculated
    assert result["fpf_alpha"] == pytest.approx(4.6)  # From alpha_fpf
    assert result["vpf_alpha"] == pytest.approx(4.0)  # Minimum cd


def test_compare_phase_max_ld() -> None:
    """Test phase comparison with max_ld target."""
    alpha = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    cl = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    cd = np.array([0.02, 0.02, 0.02, 0.03, 0.05])
    polar = (alpha, cl, cd)

    phase = FlightPhase(name="cruise", altitude_m=11000.0, mach=0.80)
    result = compare_phase(polar, phase, vpf_target="max_ld")

    # VPF should have better or equal L/D when targeting max_ld
    assert result["vpf_ld"] >= result["fpf_ld"]

    # Check percentage calculations
    assert result["ld_improvement_pct"] >= 0.0
    assert result["cd_reduction_pct"] is not None


def test_compare_phase_deltas() -> None:
    """Test that deltas and ratios are calculated correctly."""
    alpha = np.array([0.0, 4.0, 8.0])
    cl = np.array([0.0, 0.4, 0.8])
    cd = np.array([0.02, 0.01, 0.02])
    polar = (alpha, cl, cd)

    phase = FlightPhase(name="test", altitude_m=0.0, mach=0.5)
    result = compare_phase(polar, phase, vpf_target="min_cd")

    # Check delta calculations
    assert result["delta_cd"] == pytest.approx(result["vpf_cd"] - result["fpf_cd"])
    assert result["delta_ld"] == pytest.approx(result["vpf_ld"] - result["fpf_ld"])

    # Check ratio calculations
    assert result["cd_ratio"] == pytest.approx(result["vpf_cd"] / result["fpf_cd"])
    assert result["ld_ratio"] == pytest.approx(result["vpf_ld"] / result["fpf_ld"])


def test_compare_phase_percentages() -> None:
    """Test percentage calculations."""
    alpha = np.array([0.0, 4.0, 8.0])
    cl = np.array([0.0, 0.4, 0.8])
    cd = np.array([0.02, 0.01, 0.02])
    polar = (alpha, cl, cd)

    phase = FlightPhase(name="test", altitude_m=0.0, mach=0.5)
    result = compare_phase(polar, phase, vpf_target="min_cd")

    # Check percentage formulas
    expected_cd_reduction = (
        (result["fpf_cd"] - result["vpf_cd"]) / result["fpf_cd"] * 100
    )
    expected_ld_improvement = (
        (result["vpf_ld"] - result["fpf_ld"]) / result["fpf_ld"] * 100
    )

    assert result["cd_reduction_pct"] == pytest.approx(expected_cd_reduction)
    assert result["ld_improvement_pct"] == pytest.approx(expected_ld_improvement)
