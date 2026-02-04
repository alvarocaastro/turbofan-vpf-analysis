"""Advanced plotting functions for VPF vs FPF comparison."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbofan_vpf.aero.compressibility import apply_compressibility_to_polar
from turbofan_vpf.viz.constants import (
    ALPHA_CURVE,
    ALPHA_GRID,
    ALPHA_POINT,
    COLOR_FPF,
    COLOR_POLAR,
    COLOR_VPF,
    EDGE_WIDTH,
    FIG_SIZE_STANDARD,
    FIG_SIZE_TALL,
    FIG_SIZE_WIDE,
    FIGURE_SAVE_KWARGS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_SUBTITLE,
    FONT_SIZE_TITLE,
    FONT_SIZE_VALUE,
    LINE_WIDTH,
    MARKER_FPF,
    MARKER_SIZE,
    MARKER_VPF,
)
from turbofan_vpf.utils.paths import ensure_dir_exists

# Type aliases
PolarData = tuple[np.ndarray, np.ndarray, np.ndarray]
PhaseResult = dict[str, Any]
PhaseResults = list[PhaseResult]


def _save_figure(output_path: Path) -> None:
    """Save figure to file with standard settings.

    Args:
        output_path: Path to save the figure
    """
    ensure_dir_exists(output_path.parent)
    plt.savefig(output_path, **FIGURE_SAVE_KWARGS)
    plt.close()


def _get_phase_colors(num_phases: int) -> np.ndarray:
    """Get color array for phases.

    Args:
        num_phases: Number of phases

    Returns:
        Array of colors
    """
    return plt.cm.tab10(np.linspace(0, 1, num_phases))


def _format_phase_label(phase_name: str, mach: float) -> str:
    """Format phase label with Mach number.

    Args:
        phase_name: Phase name
        mach: Mach number

    Returns:
        Formatted label string
    """
    return f"{phase_name.capitalize()} (M={mach:.2f})"


def plot_cd_vs_alpha(
    polar_base: PolarData,
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot CD vs alpha by phase, marking alpha_min_cd and alpha_fpf.

    Shows CD curves for each phase (with compressibility correction) and marks
    the optimal angles (alpha_min_cd for VPF, alpha_fpf constant for FPF).

    Args:
        polar_base: Tuple of (alpha_deg, cl, cd) arrays (base/incompressible polar)
        phase_results: List of phase result dictionaries
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Drag Coefficient vs Angle of Attack by Flight Phase\nMarking $\\alpha_{min,CD}$ (VPF) and $\\alpha_{FPF}$ (constant)",
        fontsize=FONT_SIZE_TITLE + 2,
        fontweight="bold",
        y=0.995,
    )

    axes_flat = axes.flatten()
    colors = _get_phase_colors(len(phase_results))

    for idx, result in enumerate(phase_results):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        phase_name = result["phase"]
        mach = result["mach"]

        # Apply compressibility correction for this phase
        polar_corr = apply_compressibility_to_polar(polar_base, mach, phase_name=phase_name)
        alpha_corr, _, cd_corr = polar_corr

        # Plot corrected CD curve
        ax.plot(
            alpha_corr,
            cd_corr,
            COLOR_POLAR,
            linewidth=LINE_WIDTH,
            label=f"CD (M={mach:.2f})",
            alpha=ALPHA_CURVE,
            zorder=1,
        )

        # Mark alpha_min_cd (VPF optimal)
        alpha_min_cd = result["alpha_min_cd"]
        cd_at_min = result["cd_at_alpha_min"]
        ax.scatter(
            alpha_min_cd,
            cd_at_min,
            marker=MARKER_VPF,
            s=MARKER_SIZE,
            color=COLOR_VPF,
            edgecolors=COLOR_POLAR,
            linewidth=EDGE_WIDTH,
            label=f"VPF: $\\alpha_{{min,CD}}$ = {alpha_min_cd:.2f}°",
            alpha=ALPHA_POINT,
            zorder=3,
        )

        # Mark alpha_fpf (FPF constant)
        alpha_fpf = result["alpha_fpf"]
        cd_fpf = result["cd_fpf"]
        ax.scatter(
            alpha_fpf,
            cd_fpf,
            marker=MARKER_FPF,
            s=MARKER_SIZE,
            color=COLOR_FPF,
            edgecolors=COLOR_POLAR,
            linewidth=EDGE_WIDTH,
            label=f"FPF: $\\alpha_{{FPF}}$ = {alpha_fpf:.2f}°",
            alpha=ALPHA_POINT,
            zorder=3,
        )

        # Add vertical lines
        ax.axvline(alpha_min_cd, color=COLOR_VPF, linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(alpha_fpf, color=COLOR_FPF, linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Angle of Attack, $\\alpha$ [deg]", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Drag Coefficient, $C_d$ [-]", fontsize=FONT_SIZE_LABEL)
        ax.set_title(_format_phase_label(phase_name, mach), fontsize=FONT_SIZE_SUBTITLE, fontweight="bold")
        ax.grid(True, alpha=ALPHA_GRID, linestyle="--", linewidth=0.5)
        ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND - 1, framealpha=0.9)

    # Hide unused subplots
    for idx in range(len(phase_results), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    _save_figure(output_path)


def plot_delta_cd_bars(
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot bar chart of delta_CD (benefit of VPF) by phase.

    Args:
        phase_results: List of phase result dictionaries
        output_path: Path to save the figure
    """
    df = pd.DataFrame(phase_results)
    phases = df["phase"].values
    mach_numbers = df["mach"].values
    delta_cd = df["delta_cd"].values

    # Format phase labels with Mach numbers
    phase_labels = [_format_phase_label(phase, mach) for phase, mach in zip(phases, mach_numbers)]

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    fig.patch.set_facecolor("white")

    bars = ax.bar(phase_labels, delta_cd, color=COLOR_VPF, alpha=ALPHA_POINT, edgecolor=COLOR_POLAR, linewidth=1.5)
    ax.axhline(y=0, color=COLOR_POLAR, linestyle="--", linewidth=1.5, label="Reference (VPF = FPF)")

    ax.set_ylabel("Drag Coefficient Difference\n$\\Delta C_d = C_{d,FPF} - C_{d,VPF}$ [-]", fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Flight Phase", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "VPF Benefit: Drag Reduction by Flight Phase\nPositive values indicate VPF advantage",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=ALPHA_GRID, axis="y", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.0001 if height >= 0 else -0.0003),
            f"{height:.4f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=FONT_SIZE_VALUE,
            fontweight="bold",
        )

    plt.tight_layout()
    _save_figure(output_path)


def plot_delta_alpha_pitch(
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot bar chart of delta_alpha_pitch (pitch compensation) by phase.

    Shows how much the VPF needs to adjust its pitch angle relative to FPF.

    Args:
        phase_results: List of phase result dictionaries
        output_path: Path to save the figure
    """
    df = pd.DataFrame(phase_results)
    phases = df["phase"].values
    mach_numbers = df["mach"].values
    delta_alpha = df["delta_alpha_pitch"].values

    # Format phase labels with Mach numbers
    phase_labels = [_format_phase_label(phase, mach) for phase, mach in zip(phases, mach_numbers)]

    fig, ax = plt.subplots(figsize=FIG_SIZE_STANDARD)
    fig.patch.set_facecolor("white")

    bars = ax.bar(phase_labels, delta_alpha, color=COLOR_VPF, alpha=ALPHA_POINT, edgecolor=COLOR_POLAR, linewidth=1.5)
    ax.axhline(y=0, color=COLOR_POLAR, linestyle="--", linewidth=1.5, label="Reference (no pitch adjustment)")

    ax.set_ylabel(
        "Pitch Angle Compensation\n$\\Delta\\alpha_{pitch} = \\alpha_{min,CD} - \\alpha_{FPF}$ [deg]",
        fontsize=FONT_SIZE_LABEL,
    )
    ax.set_xlabel("Flight Phase", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        "VPF Pitch Compensation by Flight Phase\nRequired angle adjustment relative to FPF",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=ALPHA_GRID, axis="y", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND, framealpha=0.9)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.1 if height >= 0 else -0.2),
            f"{height:.2f}°",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=FONT_SIZE_VALUE,
            fontweight="bold",
        )

    plt.tight_layout()
    _save_figure(output_path)


def plot_polar_with_phases_v2(
    polar_base: PolarData,
    phase_results: PhaseResults,
    output_path: Path,
) -> None:
    """Plot polar CL-CD with points for each phase (FPF vs VPF).

    Args:
        polar_base: Tuple of (alpha_deg, cl, cd) arrays (base/incompressible polar)
        phase_results: List of phase result dictionaries
        output_path: Path to save the figure
    """
    alpha_base, cl_base, cd_base = polar_base

    fig, ax = plt.subplots(figsize=FIG_SIZE_TALL)
    fig.patch.set_facecolor("white")

    # Plot base polar curve (incompressible)
    ax.plot(
        cd_base,
        cl_base,
        COLOR_POLAR,
        linewidth=LINE_WIDTH,
        label="Airfoil Polar (incompressible, base)",
        alpha=ALPHA_CURVE,
        zorder=1,
    )

    # Plot points for each phase
    num_phases = len(phase_results)
    colors = _get_phase_colors(num_phases)

    for i, result in enumerate(phase_results):
        phase_name = result["phase"]
        mach = result["mach"]
        color = colors[i]

        # Apply compressibility correction for this phase
        polar_corr = apply_compressibility_to_polar(polar_base, mach, phase_name=phase_name)
        alpha_corr, cl_corr, cd_corr = polar_corr

        # Interpolate CL for FPF and VPF points
        alpha_fpf = result["alpha_fpf"]
        alpha_min_cd = result["alpha_min_cd"]

        cl_fpf = np.interp(alpha_fpf, alpha_corr, cl_corr)
        cl_vpf = np.interp(alpha_min_cd, alpha_corr, cl_corr)

        # FPF point
        ax.scatter(
            result["cd_fpf"],
            cl_fpf,
            marker=MARKER_FPF,
            s=MARKER_SIZE,
            color=color,
            edgecolors=COLOR_POLAR,
            linewidth=EDGE_WIDTH,
            label="FPF" if i == 0 else "",
            alpha=ALPHA_POINT,
            zorder=3,
        )

        # VPF point
        ax.scatter(
            result["cd_at_alpha_min"],
            cl_vpf,
            marker=MARKER_VPF,
            s=MARKER_SIZE,
            color=color,
            edgecolors=COLOR_POLAR,
            linewidth=EDGE_WIDTH,
            label="VPF" if i == 0 else "",
            alpha=ALPHA_POINT,
            zorder=3,
        )

        # Add phase annotation
        phase_label = _format_phase_label(phase_name, mach)
        mid_cd = (result["cd_fpf"] + result["cd_at_alpha_min"]) / 2
        mid_cl = (cl_fpf + cl_vpf) / 2
        ax.annotate(
            phase_label,
            xy=(mid_cd, mid_cl),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=FONT_SIZE_VALUE,
            alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor=color, linewidth=1.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=color, alpha=0.6, lw=1),
        )

    ax.set_xlabel("Drag Coefficient, $C_d$ [-]", fontsize=FONT_SIZE_LABEL, fontweight="normal")
    ax.set_ylabel("Lift Coefficient, $C_l$ [-]", fontsize=FONT_SIZE_LABEL, fontweight="normal")
    ax.set_title(
        "Aerodynamic Polar: Operating Points Comparison\nFPF vs VPF by Flight Phase",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.9, fancybox=True, shadow=True)

    # Add note about compressibility
    ax.text(
        0.02,
        0.98,
        "Note: Operating points are compressibility-corrected\nusing Prandtl-Glauert theory for each phase Mach number",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7, edgecolor="gray"),
    )

    plt.tight_layout()
    _save_figure(output_path)
