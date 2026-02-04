"""Aerodynamic calculation functions."""

import numpy as np
from typing import List


def calculate_lift_coefficient(alpha: float, cl_alpha: float, alpha_0: float = 0.0) -> float:
    """Calculate lift coefficient using linear approximation.

    Args:
        alpha: Angle of attack in degrees
        cl_alpha: Lift curve slope per degree
        alpha_0: Zero-lift angle of attack in degrees

    Returns:
        Lift coefficient
    """
    return cl_alpha * (alpha - alpha_0)


def calculate_drag_coefficient(
    cd_0: float, k: float, cl: float
) -> float:
    """Calculate drag coefficient using drag polar.

    Args:
        cd_0: Zero-lift drag coefficient
        k: Induced drag factor
        cl: Lift coefficient

    Returns:
        Drag coefficient
    """
    return cd_0 + k * cl**2


def interpolate_polar(
    alpha: float, alpha_array: np.ndarray, cl_array: np.ndarray
) -> float:
    """Interpolate lift coefficient from polar data.

    Args:
        alpha: Angle of attack in degrees
        alpha_array: Array of angle of attack values
        cl_array: Array of corresponding lift coefficients

    Returns:
        Interpolated lift coefficient

    Raises:
        ValueError: If alpha is outside the range of alpha_array
    """
    if alpha < alpha_array.min() or alpha > alpha_array.max():
        raise ValueError(
            f"Alpha {alpha} is outside the range [{alpha_array.min()}, {alpha_array.max()}]"
        )
    return np.interp(alpha, alpha_array, cl_array)
