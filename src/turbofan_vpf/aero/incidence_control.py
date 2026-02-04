"""Incidence angle control for Fixed Pitch Fan (FPF) vs Variable Pitch Fan (VPF)."""

from typing import Dict, Literal, Tuple

import numpy as np

from turbofan_vpf.aero.compressibility import apply_compressibility_to_polar
from turbofan_vpf.aero.metrics import (
    compute_metrics_at_alpha,
    find_alpha_max_ld,
    find_alpha_min_cd,
)
from turbofan_vpf.domain.flight_phases import FlightPhase


def alpha_fpf(phase: FlightPhase) -> float:
    """Calculate angle of attack for Fixed Pitch Fan (FPF).

    Simple linear law based on Mach number:
        alpha_fpf = alpha_0 + k_mach * (mach - mach_ref)

    Where:
        - alpha_0 = 4.0°: Base angle at reference Mach
        - mach_ref = 0.5: Reference Mach number
        - k_mach = 2.0: Sensitivity to Mach number (deg/Mach)

    This model assumes that FPF operates at a fixed geometric angle,
    but the effective angle of attack varies with Mach number due to
    flow conditions. Higher Mach numbers require slightly higher angles
    to maintain similar performance.

    Args:
        phase: Flight phase with altitude and Mach number

    Returns:
        Angle of attack in degrees for FPF
    """
    # Model parameters
    alpha_0 = 4.0  # Base angle at reference Mach (degrees)
    mach_ref = 0.5  # Reference Mach number
    k_mach = 2.0  # Mach sensitivity (deg/Mach)

    # Linear law: alpha increases with Mach
    alpha = alpha_0 + k_mach * (phase.mach - mach_ref)

    return alpha


def alpha_vpf(
    polar: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target: Literal["min_cd", "max_ld"] = "max_ld",
) -> float:
    """Calculate optimal angle of attack for Variable Pitch Fan (VPF).

    VPF can adjust its pitch angle to optimize performance. This function
    returns the optimal angle based on the selected target:
    - "min_cd": Minimize drag coefficient (best for efficiency at constant speed)
    - "max_ld": Maximize lift-to-drag ratio (best overall efficiency)

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays
        target: Optimization target, either "min_cd" or "max_ld"

    Returns:
        Optimal angle of attack in degrees for VPF

    Raises:
        ValueError: If target is invalid or polar data is invalid
    """
    if target == "min_cd":
        return find_alpha_min_cd(polar)
    elif target == "max_ld":
        return find_alpha_max_ld(polar)
    else:
        raise ValueError(f"Invalid target: {target}. Must be 'min_cd' or 'max_ld'")


def compare_phase(
    polar: Tuple[np.ndarray, np.ndarray, np.ndarray],
    phase: FlightPhase,
    vpf_target: Literal["min_cd", "max_ld"] = "max_ld",
) -> Dict[str, float]:
    """Compare FPF and VPF performance for a given flight phase.

    Calculates aerodynamic metrics (CD, L/D) for both FPF and VPF configurations,
    and provides comparison metrics (deltas and ratios).

    Applies compressibility corrections to the polar based on the phase Mach number
    before computing metrics. The correction is applied automatically according to
    the global configuration.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays (incompressible/base polar)
        phase: Flight phase with altitude and Mach number
        vpf_target: VPF optimization target, either "min_cd" or "max_ld"

    Returns:
        Dictionary with keys:
        - 'fpf_alpha': FPF angle of attack (degrees)
        - 'vpf_alpha': VPF angle of attack (degrees)
        - 'fpf_cd': FPF drag coefficient (compressibility corrected)
        - 'vpf_cd': VPF drag coefficient (compressibility corrected)
        - 'fpf_ld': FPF lift-to-drag ratio (compressibility corrected)
        - 'vpf_ld': VPF lift-to-drag ratio (compressibility corrected)
        - 'delta_cd': CD difference (vpf_cd - fpf_cd)
        - 'delta_ld': L/D difference (vpf_ld - fpf_ld)
        - 'cd_ratio': CD ratio (vpf_cd / fpf_cd)
        - 'ld_ratio': L/D ratio (vpf_ld / fpf_ld)
        - 'cd_reduction_pct': CD reduction percentage ((fpf_cd - vpf_cd) / fpf_cd * 100)
        - 'ld_improvement_pct': L/D improvement percentage ((vpf_ld - fpf_ld) / fpf_ld * 100)

    Raises:
        ValueError: If polar data is invalid or angles are out of range
    """
    # Apply compressibility correction for this phase's Mach number
    polar_corrected = apply_compressibility_to_polar(polar, phase.mach, phase_name=phase.name)

    # Calculate angles (using corrected polar for VPF optimization)
    fpf_alpha = alpha_fpf(phase)
    vpf_alpha = alpha_vpf(polar_corrected, target=vpf_target)

    # Calculate metrics for both configurations (using corrected polar)
    fpf_metrics = compute_metrics_at_alpha(polar_corrected, fpf_alpha)
    vpf_metrics = compute_metrics_at_alpha(polar_corrected, vpf_alpha)

    fpf_cd = fpf_metrics["cd"]
    fpf_ld = fpf_metrics["ld"]
    vpf_cd = vpf_metrics["cd"]
    vpf_ld = vpf_metrics["ld"]

    # Calculate comparison metrics
    delta_cd = vpf_cd - fpf_cd
    delta_ld = vpf_ld - fpf_ld
    cd_ratio = vpf_cd / fpf_cd if fpf_cd > 0 else float("inf")
    ld_ratio = vpf_ld / fpf_ld if fpf_ld > 0 else float("inf")

    # Calculate percentages
    cd_reduction_pct = ((fpf_cd - vpf_cd) / fpf_cd * 100) if fpf_cd > 0 else 0.0
    ld_improvement_pct = ((vpf_ld - fpf_ld) / fpf_ld * 100) if fpf_ld > 0 else 0.0

    return {
        "fpf_alpha": fpf_alpha,
        "vpf_alpha": vpf_alpha,
        "fpf_cd": fpf_cd,
        "vpf_cd": vpf_cd,
        "fpf_ld": fpf_ld,
        "vpf_ld": vpf_ld,
        "delta_cd": delta_cd,
        "delta_ld": delta_ld,
        "cd_ratio": cd_ratio,
        "ld_ratio": ld_ratio,
        "cd_reduction_pct": cd_reduction_pct,
        "ld_improvement_pct": ld_improvement_pct,
    }
