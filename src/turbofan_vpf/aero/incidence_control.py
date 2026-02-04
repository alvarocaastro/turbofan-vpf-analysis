"""Incidence angle control for Fixed Pitch Fan (FPF) vs Variable Pitch Fan (VPF)."""

import warnings
from typing import Dict, Tuple

import numpy as np

from turbofan_vpf.aero.compressibility import apply_compressibility_to_polar
from turbofan_vpf.aero.metrics import compute_metrics_at_alpha
from turbofan_vpf.aero.optima import find_alpha_min_cd
from turbofan_vpf.domain.flight_phases import FlightPhase


def compute_phase_results(
    polar_base: Tuple[np.ndarray, np.ndarray, np.ndarray],
    phase: FlightPhase,
    alpha_fpf: float,
) -> Dict[str, float]:
    """Compute FPF and VPF results for a given flight phase.

    This function:
    1. Applies compressibility correction to base polar for phase Mach number
    2. Finds alpha_min_cd for this phase (VPF optimal angle)
    3. Evaluates FPF at fixed alpha_fpf (constant across all phases)
    4. Evaluates VPF at alpha_min_cd(phase) (optimal for this phase)
    5. Computes all comparison metrics

    Args:
        polar_base: Tuple of (alpha_deg, cl, cd) arrays (base/incompressible polar)
        phase: Flight phase with altitude and Mach number
        alpha_fpf: Fixed FPF angle of attack (constant, optimized for cruise)

    Returns:
        Dictionary with keys:
        - 'phase': Phase name
        - 'mach': Mach number
        - 'alpha_min_cd': Optimal angle for minimum CD in this phase
        - 'alpha_fpf': Fixed FPF angle (constant)
        - 'delta_alpha_pitch': Pitch compensation = alpha_min_cd - alpha_fpf
        - 'cd_at_alpha_min': CD at alpha_min_cd (VPF optimal CD)
        - 'cd_fpf': CD at alpha_fpf (FPF CD)
        - 'delta_cd': CD difference = cd_fpf - cd_at_alpha_min (benefit of VPF)
        - 'ratio_cd': CD ratio = cd_fpf / cd_at_alpha_min
        - 'cl_at_alpha_min': CL at alpha_min_cd (VPF)
        - 'cl_fpf': CL at alpha_fpf (FPF)
        - 'ld_at_alpha_min': L/D at alpha_min_cd (VPF)
        - 'ld_fpf': L/D at alpha_fpf (FPF)

    Raises:
        ValueError: If polar data is invalid or angles are out of range
    """
    # Step 1: Apply compressibility correction for this phase's Mach number
    polar_corrected = apply_compressibility_to_polar(polar_base, phase.mach, phase_name=phase.name)

    alpha_corr, cl_corr, cd_corr = polar_corrected

    # Step 2: Find alpha_min_cd for this phase (VPF optimal angle)
    alpha_min_cd = find_alpha_min_cd(polar_corrected)

    # Step 3: Validate and clamp alpha_fpf if necessary
    alpha_min = float(alpha_corr.min())
    alpha_max = float(alpha_corr.max())

    if alpha_fpf < alpha_min or alpha_fpf > alpha_max:
        warnings.warn(
            f"FPF angle {alpha_fpf:.2f}° is outside corrected polar range "
            f"[{alpha_min:.2f}°, {alpha_max:.2f}°] for phase '{phase.name}'. "
            f"Clamping to valid range.",
            UserWarning,
            stacklevel=2,
        )
        alpha_fpf_clamped = np.clip(alpha_fpf, alpha_min, alpha_max)
    else:
        alpha_fpf_clamped = alpha_fpf

    # Step 4: Evaluate metrics at both angles
    # VPF: at alpha_min_cd (optimal for this phase)
    vpf_metrics = compute_metrics_at_alpha(polar_corrected, alpha_min_cd)

    # FPF: at fixed alpha_fpf (constant)
    fpf_metrics = compute_metrics_at_alpha(polar_corrected, alpha_fpf_clamped)

    # Step 5: Compute comparison metrics
    cd_at_alpha_min = vpf_metrics["cd"]
    cd_fpf = fpf_metrics["cd"]
    delta_cd = cd_fpf - cd_at_alpha_min  # Positive = VPF benefit
    ratio_cd = cd_fpf / cd_at_alpha_min if cd_at_alpha_min > 0 else float("inf")

    delta_alpha_pitch = alpha_min_cd - alpha_fpf

    return {
        "phase": phase.name,
        "mach": phase.mach,
        "alpha_min_cd": alpha_min_cd,
        "alpha_fpf": alpha_fpf,
        "delta_alpha_pitch": delta_alpha_pitch,
        "cd_at_alpha_min": cd_at_alpha_min,
        "cd_fpf": cd_fpf,
        "delta_cd": delta_cd,
        "ratio_cd": ratio_cd,
        "cl_at_alpha_min": vpf_metrics["cl"],
        "cl_fpf": fpf_metrics["cl"],
        "ld_at_alpha_min": vpf_metrics["ld"],
        "ld_fpf": fpf_metrics["ld"],
    }
