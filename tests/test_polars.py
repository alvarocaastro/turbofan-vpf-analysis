"""Tests for polar data loading and interpolation."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from turbofan_vpf.aero.polars import (
    interp_cl_cd,
    load_polar_csv,
    set_polar_data,
)


def test_load_polar_csv(tmp_path: Path) -> None:
    """Test loading polar data from CSV file."""
    # Create test CSV
    csv_file = tmp_path / "test_polar.csv"
    csv_file.write_text("alpha_deg,cl,cd\n0.0,0.0,0.01\n5.0,0.5,0.02\n10.0,1.0,0.03\n")

    alpha, cl, cd = load_polar_csv(csv_file)

    assert len(alpha) == 3
    assert len(cl) == 3
    assert len(cd) == 3
    assert alpha[0] == pytest.approx(0.0)
    assert cl[1] == pytest.approx(0.5)
    assert cd[2] == pytest.approx(0.03)


def test_load_polar_csv_missing_file() -> None:
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_polar_csv(Path("nonexistent.csv"))


def test_load_polar_csv_missing_columns(tmp_path: Path) -> None:
    """Test that missing required columns raises ValueError."""
    csv_file = tmp_path / "bad_polar.csv"
    csv_file.write_text("alpha,cl\n0.0,0.0\n")

    with pytest.raises(ValueError, match="alpha_deg"):
        load_polar_csv(csv_file)


def test_interp_cl_cd() -> None:
    """Test interpolation of cl and cd."""
    # Set test data
    alpha = np.array([0.0, 5.0, 10.0])
    cl = np.array([0.0, 0.5, 1.0])
    cd = np.array([0.01, 0.02, 0.03])
    set_polar_data(alpha, cl, cd)

    # Interpolate at midpoint
    cl_val, cd_val = interp_cl_cd(2.5)
    assert cl_val == pytest.approx(0.25)
    assert cd_val == pytest.approx(0.015)


def test_interp_cl_cd_out_of_range_warning() -> None:
    """Test that out-of-range interpolation emits warnings."""
    alpha = np.array([0.0, 5.0, 10.0])
    cl = np.array([0.0, 0.5, 1.0])
    cd = np.array([0.01, 0.02, 0.03])
    set_polar_data(alpha, cl, cd)

    # Test below minimum
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cl_val, cd_val = interp_cl_cd(-5.0)
        assert len(w) == 1
        assert "below minimum" in str(w[0].message).lower()
        assert cl_val == pytest.approx(0.0)  # Should use boundary value

    # Test above maximum
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cl_val, cd_val = interp_cl_cd(15.0)
        assert len(w) == 1
        assert "above maximum" in str(w[0].message).lower()
        assert cl_val == pytest.approx(1.0)  # Should use boundary value


def test_interp_cl_cd_no_data() -> None:
    """Test that interpolation without loaded data raises RuntimeError."""
    # Reset any loaded data
    set_polar_data(np.array([]), np.array([]), np.array([]))

    with pytest.raises(RuntimeError, match="No polar data loaded"):
        interp_cl_cd(5.0)


def test_load_polar_csv_example() -> None:
    """Test loading the example CSV file."""
    example_path = Path("data/polars/example.csv")
    if not example_path.exists():
        pytest.skip("Example CSV file not found")

    alpha, cl, cd = load_polar_csv(example_path)

    assert len(alpha) > 0
    assert len(cl) == len(alpha)
    assert len(cd) == len(alpha)
    assert alpha.min() <= -5.0
    assert alpha.max() >= 15.0

    # Test interpolation
    cl_val, cd_val = interp_cl_cd(7.5)
    assert cl_val > 0
    assert cd_val > 0
