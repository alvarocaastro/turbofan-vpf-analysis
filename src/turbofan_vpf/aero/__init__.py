"""Aerodynamic calculations and analysis."""

from turbofan_vpf.aero.calculations import (
    calculate_drag_coefficient,
    calculate_lift_coefficient,
    interpolate_polar,
)
from turbofan_vpf.aero.compressibility import (
    apply_compressibility_to_polar,
    beta,
    correct_cd_pg,
    correct_cl_pg,
)
from turbofan_vpf.aero.incidence_control import alpha_fpf, alpha_vpf, compare_phase
from turbofan_vpf.aero.metrics import (
    compute_metrics_at_alpha,
    find_alpha_max_ld,
    find_alpha_min_cd,
)
from turbofan_vpf.aero.polars import interp_cl_cd, load_polar_csv, set_polar_data

__all__ = [
    "alpha_fpf",
    "alpha_vpf",
    "apply_compressibility_to_polar",
    "beta",
    "calculate_drag_coefficient",
    "calculate_lift_coefficient",
    "compare_phase",
    "compute_metrics_at_alpha",
    "correct_cd_pg",
    "correct_cl_pg",
    "find_alpha_max_ld",
    "find_alpha_min_cd",
    "interpolate_polar",
    "interp_cl_cd",
    "load_polar_csv",
    "set_polar_data",
]
