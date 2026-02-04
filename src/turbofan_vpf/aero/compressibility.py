"""Compressibility corrections for subsonic aerodynamics.

This module implements Prandtl-Glauert compressibility corrections for
2D airfoil polars. The corrections are valid for subsonic flow (Mach < 0.7)
and do not account for wave drag or shock effects.
"""

import warnings
from typing import Tuple

import numpy as np

from turbofan_vpf.domain.config import get_config


def beta(mach: float) -> float:
    """Calculate Prandtl-Glauert compressibility factor.

    β = √(1 - M²)

    Args:
        mach: Mach number

    Returns:
        Compressibility factor β

    Raises:
        ValueError: If Mach number >= 1.0 (supersonic)
    """
    if mach >= 1.0:
        raise ValueError(f"Mach number {mach:.2f} is supersonic. This model is for subsonic flow only.")

    return np.sqrt(1.0 - mach**2)


def correct_cl_pg(cl: float | np.ndarray, mach: float) -> float | np.ndarray:
    """Apply Prandtl-Glauert correction to lift coefficient.

    CL_corrected = CL_incompressible / β

    where β = √(1 - M²)

    Args:
        cl: Lift coefficient (incompressible)
        mach: Mach number

    Returns:
        Corrected lift coefficient accounting for compressibility
    """
    beta_val = beta(mach)
    return cl / beta_val


def correct_cd_pg(
    cd: float | np.ndarray,
    mach: float,
    cl: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Apply Prandtl-Glauert correction to drag coefficient.

    Uses a simple approximation:
    CD_corrected = CD_incompressible / β²

    where β = √(1 - M²)

    This approximation assumes:
    - No wave drag (valid only for subsonic flow)
    - No shock effects
    - Drag scales with dynamic pressure correction

    For more accurate results, a drag polar correction could be used:
    CD_corrected = CD0/β² + CDi/β², where CDi is induced drag.

    Args:
        cd: Drag coefficient (incompressible)
        mach: Mach number
        cl: Optional lift coefficient (not used in current approximation)

    Returns:
        Corrected drag coefficient accounting for compressibility

    Warns:
        UserWarning: Explicit warning that wave drag and shocks are not modeled
    """
    if cl is not None:
        # Future: could use cl-dependent correction for induced drag
        pass

    beta_val = beta(mach)

    # Explicit warning about limitations
    if mach > 0.6:
        warnings.warn(
            f"Compressibility correction applied at Mach {mach:.2f}. "
            "This model does NOT account for wave drag or shock effects. "
            "Results may be inaccurate near transonic speeds (M > 0.7).",
            UserWarning,
            stacklevel=2,
        )

    return cd / (beta_val**2)


def apply_compressibility_to_polar(
    polar: Tuple[np.ndarray, np.ndarray, np.ndarray],
    mach: float,
    phase_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply compressibility corrections to an entire polar.

    Corrects both CL and CD arrays for the given Mach number using
    Prandtl-Glauert theory.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays
        mach: Mach number for correction
        phase_name: Optional phase name for validation warnings

    Returns:
        Tuple of (alpha_deg, cl_corrected, cd_corrected) arrays

    Raises:
        ValueError: If Mach number is invalid
    """
    config = get_config()

    if not config.use_compressibility:
        return polar

    # Validate Mach number
    config.validate_mach(mach, phase_name)

    alpha, cl, cd = polar

    # Apply corrections
    cl_corrected = correct_cl_pg(cl, mach)
    cd_corrected = correct_cd_pg(cd, mach, cl=cl_corrected)

    return (alpha, cl_corrected, cd_corrected)
