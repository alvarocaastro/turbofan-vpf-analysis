"""Aerodynamic metrics calculation from polar data."""

from typing import Dict, Tuple

import numpy as np


def find_alpha_min_cd(
    polar: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> float:
    """Find angle of attack with minimum drag coefficient.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays

    Returns:
        Angle of attack in degrees where cd is minimum

    Raises:
        ValueError: If polar data is empty
    """
    alpha, cl, cd = polar

    if len(alpha) == 0:
        raise ValueError("Polar data is empty")

    min_cd_idx = np.argmin(cd)
    return float(alpha[min_cd_idx])


def find_alpha_max_ld(
    polar: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> float:
    """Find angle of attack with maximum lift-to-drag ratio.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays

    Returns:
        Angle of attack in degrees where L/D (cl/cd) is maximum

    Raises:
        ValueError: If polar data is empty or contains zero drag coefficients
    """
    alpha, cl, cd = polar

    if len(alpha) == 0:
        raise ValueError("Polar data is empty")

    # Avoid division by zero
    if np.any(cd <= 0):
        raise ValueError("Polar data contains zero or negative drag coefficients")

    ld_ratio = cl / cd
    max_ld_idx = np.argmax(ld_ratio)
    return float(alpha[max_ld_idx])


def compute_metrics_at_alpha(
    polar: Tuple[np.ndarray, np.ndarray, np.ndarray], alpha: float
) -> Dict[str, float]:
    """Compute aerodynamic metrics at a given angle of attack.

    Interpolates cl and cd from polar data and calculates L/D ratio.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays
        alpha: Angle of attack in degrees

    Returns:
        Dictionary with keys:
        - 'cl': Lift coefficient
        - 'cd': Drag coefficient
        - 'ld': Lift-to-drag ratio (cl/cd)

    Raises:
        ValueError: If polar data is empty or alpha is outside range
    """
    alpha_array, cl_array, cd_array = polar

    if len(alpha_array) == 0:
        raise ValueError("Polar data is empty")

    if alpha < alpha_array.min() or alpha > alpha_array.max():
        raise ValueError(
            f"Alpha {alpha:.2f}° is outside polar range "
            f"[{alpha_array.min():.2f}°, {alpha_array.max():.2f}°]"
        )

    # Interpolate cl and cd
    cl = np.interp(alpha, alpha_array, cl_array)
    cd = np.interp(alpha, alpha_array, cd_array)

    # Calculate L/D ratio
    if cd <= 0:
        ld = float("inf") if cl > 0 else float("-inf")
    else:
        ld = cl / cd

    return {
        "cl": float(cl),
        "cd": float(cd),
        "ld": float(ld),
    }
