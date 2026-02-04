"""Tests for aerodynamic calculations."""

import numpy as np
import pytest
from turbofan_vpf.aero.calculations import (
    calculate_drag_coefficient,
    calculate_lift_coefficient,
    interpolate_polar,
)


def test_calculate_lift_coefficient() -> None:
    """Test lift coefficient calculation."""
    cl = calculate_lift_coefficient(alpha=5.0, cl_alpha=0.1, alpha_0=0.0)
    assert cl == pytest.approx(0.5)


def test_calculate_drag_coefficient() -> None:
    """Test drag coefficient calculation."""
    cd = calculate_drag_coefficient(cd_0=0.01, k=0.05, cl=1.0)
    assert cd == pytest.approx(0.06)


def test_interpolate_polar() -> None:
    """Test polar interpolation."""
    alpha_array = np.array([0.0, 5.0, 10.0])
    cl_array = np.array([0.0, 0.5, 1.0])
    cl = interpolate_polar(alpha=2.5, alpha_array=alpha_array, cl_array=cl_array)
    assert cl == pytest.approx(0.25)


def test_interpolate_polar_out_of_range() -> None:
    """Test polar interpolation with out-of-range alpha."""
    alpha_array = np.array([0.0, 5.0, 10.0])
    cl_array = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        interpolate_polar(alpha=15.0, alpha_array=alpha_array, cl_array=cl_array)
