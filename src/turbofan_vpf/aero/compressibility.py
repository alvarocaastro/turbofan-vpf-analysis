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

    Uses a more realistic approximation that accounts for induced drag:
    CD_corrected = CD0/β² + CDi/β³

    where:
    - β = √(1 - M²)
    - CD0 is zero-lift drag (scales with 1/β²)
    - CDi is induced drag (scales with 1/β³ because it depends on CL²)

    This approximation:
    - Separates zero-lift drag (CD0) from induced drag (CDi)
    - Accounts for the fact that induced drag scales differently with Mach
    - Changes the optimal angle of attack with Mach number

    Args:
        cd: Drag coefficient (incompressible)
        mach: Mach number
        cl: Lift coefficient (required for induced drag correction)

    Returns:
        Corrected drag coefficient accounting for compressibility

    Warns:
        UserWarning: Explicit warning that wave drag and shocks are not modeled
    """
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

    # More realistic correction: separate zero-lift drag from induced drag
    # CD = CD0 + CDi, where CDi depends on CL²
    # CD0 scales with 1/β², CDi scales with 1/β³ (because CL scales with 1/β)
    if cl is not None:
        if isinstance(cd, np.ndarray) and isinstance(cl, np.ndarray):
            # Find CD0: minimum CD (typically occurs near zero lift)
            # Find the point closest to zero lift
            cl_abs = np.abs(cl)
            zero_lift_idx = np.argmin(cl_abs)
            cd0_estimate = cd[zero_lift_idx]
            
            # Estimate induced drag component: CDi = CD - CD0
            # But we need to account for the fact that CDi depends on CL²
            # For compressibility: CDi_corrected = CDi_incomp * (CL_corrected/CL_incomp)² / β²
            # Since CL_corrected = CL_incomp / β, we get:
            # CDi_corrected = CDi_incomp / β³
            
            cdi_estimate = cd - cd0_estimate
            # Zero-lift drag: CD0/β²
            # Induced drag: CDi/β³ (because it scales with CL², and CL scales with 1/β)
            cd_corrected = cd0_estimate / (beta_val**2) + cdi_estimate / (beta_val**3)
            # Ensure non-negative and physically reasonable
            cd_corrected = np.maximum(cd_corrected, cd0_estimate / (beta_val**2))
        else:
            # Scalar case
            cd0_estimate = cd  # Simple fallback
            cd_corrected = cd / (beta_val**2)
    else:
        # Fallback to simple scaling if CL not provided
        cd_corrected = cd / (beta_val**2)

    return cd_corrected


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
