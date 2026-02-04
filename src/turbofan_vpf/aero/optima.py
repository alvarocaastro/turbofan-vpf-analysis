"""Functions to find optimal angles of attack from polar data."""

from typing import Tuple

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
    alpha, _, cd = polar

    if len(alpha) == 0:
        raise ValueError("Polar data is empty")

    min_cd_idx = np.argmin(cd)
    return float(alpha[min_cd_idx])
