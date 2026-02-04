"""Tests for aerodynamic metrics calculation."""

import numpy as np
import pytest

from turbofan_vpf.aero.metrics import (
    compute_metrics_at_alpha,
    find_alpha_min_cd,
    find_alpha_max_ld,
)


def test_find_alpha_min_cd() -> None:
    """Test finding angle of attack with minimum drag."""
    alpha = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    cl = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    cd = np.array([0.02, 0.015, 0.01, 0.012, 0.015])
    polar = (alpha, cl, cd)

    alpha_min_cd = find_alpha_min_cd(polar)
    assert alpha_min_cd == pytest.approx(4.0)


def test_find_alpha_min_cd_empty() -> None:
    """Test that empty polar raises ValueError."""
    polar = (np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="empty"):
        find_alpha_min_cd(polar)


def test_find_alpha_max_ld() -> None:
    """Test finding angle of attack with maximum L/D."""
    alpha = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    cl = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    cd = np.array([0.02, 0.02, 0.02, 0.03, 0.05])
    polar = (alpha, cl, cd)

    alpha_max_ld = find_alpha_max_ld(polar)
    # At alpha=4.0, L/D = 0.4/0.02 = 20.0 (maximum)
    # At alpha=6.0, L/D = 0.6/0.03 = 20.0
    # At alpha=8.0, L/D = 0.8/0.05 = 16.0
    assert alpha_max_ld == pytest.approx(4.0)


def test_find_alpha_max_ld_empty() -> None:
    """Test that empty polar raises ValueError."""
    polar = (np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="empty"):
        find_alpha_max_ld(polar)


def test_find_alpha_max_ld_zero_cd() -> None:
    """Test that zero drag coefficient raises ValueError."""
    alpha = np.array([0.0, 2.0])
    cl = np.array([0.0, 0.2])
    cd = np.array([0.0, 0.02])
    polar = (alpha, cl, cd)

    with pytest.raises(ValueError, match="zero or negative"):
        find_alpha_max_ld(polar)


def test_compute_metrics_at_alpha() -> None:
    """Test computing metrics at a given angle of attack."""
    alpha = np.array([0.0, 5.0, 10.0])
    cl = np.array([0.0, 0.5, 1.0])
    cd = np.array([0.01, 0.02, 0.03])
    polar = (alpha, cl, cd)

    metrics = compute_metrics_at_alpha(polar, 5.0)
    assert metrics["cl"] == pytest.approx(0.5)
    assert metrics["cd"] == pytest.approx(0.02)
    assert metrics["ld"] == pytest.approx(25.0)  # 0.5 / 0.02


def test_compute_metrics_at_alpha_interpolated() -> None:
    """Test computing metrics with interpolation."""
    alpha = np.array([0.0, 5.0, 10.0])
    cl = np.array([0.0, 0.5, 1.0])
    cd = np.array([0.01, 0.02, 0.03])
    polar = (alpha, cl, cd)

    metrics = compute_metrics_at_alpha(polar, 2.5)
    # Interpolated: cl = 0.25, cd = 0.015
    assert metrics["cl"] == pytest.approx(0.25)
    assert metrics["cd"] == pytest.approx(0.015)
    assert metrics["ld"] == pytest.approx(0.25 / 0.015)


def test_compute_metrics_at_alpha_out_of_range() -> None:
    """Test that out-of-range alpha raises ValueError."""
    alpha = np.array([0.0, 5.0, 10.0])
    cl = np.array([0.0, 0.5, 1.0])
    cd = np.array([0.01, 0.02, 0.03])
    polar = (alpha, cl, cd)

    with pytest.raises(ValueError, match="outside polar range"):
        compute_metrics_at_alpha(polar, 15.0)


def test_compute_metrics_at_alpha_empty() -> None:
    """Test that empty polar raises ValueError."""
    polar = (np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="empty"):
        compute_metrics_at_alpha(polar, 5.0)


def test_compute_metrics_at_alpha_zero_cd() -> None:
    """Test handling of zero drag coefficient in L/D calculation."""
    alpha = np.array([0.0, 5.0])
    cl = np.array([0.0, 0.5])
    cd = np.array([0.0, 0.02])
    polar = (alpha, cl, cd)

    metrics = compute_metrics_at_alpha(polar, 0.0)
    assert metrics["cl"] == pytest.approx(0.0)
    assert metrics["cd"] == pytest.approx(0.0)
    assert np.isinf(metrics["ld"])


def test_compute_metrics_at_alpha_negative_cl_zero_cd() -> None:
    """Test handling of negative lift with zero drag."""
    alpha = np.array([-5.0, 0.0])
    cl = np.array([-0.5, 0.0])
    cd = np.array([0.0, 0.01])
    polar = (alpha, cl, cd)

    metrics = compute_metrics_at_alpha(polar, -5.0)
    assert metrics["cl"] == pytest.approx(-0.5)
    assert metrics["cd"] == pytest.approx(0.0)
    assert np.isinf(metrics["ld"]) and metrics["ld"] < 0
