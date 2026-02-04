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


def alpha_fpf(
    polar: Tuple[np.ndarray, np.ndarray, np.ndarray],
    cruise_phase: FlightPhase,
    target: Literal["min_cd", "max_ld"] = "max_ld",
) -> float:
    """Calculate fixed angle of attack for Fixed Pitch Fan (FPF).

    FPF operates at a CONSTANT angle of attack, optimized for cruise conditions.
    This angle is determined once using the cruise phase polar (with compressibility
    correction) and then maintained across all flight phases.

    The fixed angle is chosen to optimize performance at cruise (typically max L/D
    or min CD), representing the design point where the fan is most efficient.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays (base/incompressible polar)
        cruise_phase: Cruise flight phase used to determine the fixed angle
        target: Optimization target for cruise, either "min_cd" or "max_ld"

    Returns:
        Fixed angle of attack in degrees for FPF (constant across all phases)
    """
    # Apply compressibility correction for cruise Mach number
    polar_cruise = apply_compressibility_to_polar(polar, cruise_phase.mach, phase_name=cruise_phase.name)

    # Find optimal angle for cruise conditions
    if target == "min_cd":
        return find_alpha_min_cd(polar_cruise)
    elif target == "max_ld":
        return find_alpha_max_ld(polar_cruise)
    else:
        raise ValueError(f"Invalid target: {target}. Must be 'min_cd' or 'max_ld'")


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
    fpf_alpha: float,
    vpf_target: Literal["min_cd", "max_ld"] = "max_ld",
) -> Dict[str, float]:
    """Compare FPF and VPF performance for a given flight phase.

    Calculates aerodynamic metrics (CD, L/D) for both FPF and VPF configurations,
    and provides comparison metrics (deltas and ratios).

    FPF uses a CONSTANT angle of attack (optimized for cruise), while VPF finds
    the optimal angle for each phase using the compressibility-corrected polar.

    Applies compressibility corrections to the polar based on the phase Mach number
    before computing metrics. The correction is applied automatically according to
    the global configuration.

    Args:
        polar: Tuple of (alpha_deg, cl, cd) arrays (incompressible/base polar)
        phase: Flight phase with altitude and Mach number
        fpf_alpha: Fixed angle of attack for FPF (constant, optimized for cruise)
        vpf_target: VPF optimization target, either "min_cd" or "max_ld"

    Returns:
        Dictionary with keys:
        - 'fpf_alpha': FPF angle of attack (degrees, constant)
        - 'vpf_alpha': VPF angle of attack (degrees, optimized for this phase)
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

    # FPF uses constant angle (optimized for cruise)
    # VPF finds optimal angle for this specific phase
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
